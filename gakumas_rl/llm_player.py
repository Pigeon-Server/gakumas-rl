"""LLM 对弈器 — 用 LLM 选动作生成示范 trajectory 数据。"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from openai import OpenAI

from .state_snapshot import build_state_snapshot, format_action_list, extract_action_list_context, action_label_for_llm
from .data import TRAJECTORIES_DIR
from .service import build_env_from_config, get_repository, get_scenario, resolve_idol_card_pool
from .prompt_renderer import render, load_system_prompt


def _read_text_file(path: str | Path) -> str:
    """读取外部 prompt / 策略文本，尽量兼容常见编码。"""

    file_path = Path(path)
    for encoding in ('utf-8', 'utf-8-sig', 'gb18030', 'shift_jis'):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding='utf-8', errors='replace')


class EmptyFinalResponseError(RuntimeError):
    """模型结束了请求，但没有给出最终答复文本。"""

    def __init__(self, reason_label: str, finish_reason: str, thinking_len: int = 0):
        self.reason_label = reason_label
        self.finish_reason = finish_reason or 'unknown'
        self.thinking_len = thinking_len
        self.retryable = self.finish_reason == 'length'
        if self.retryable:
            message = f'LLM exhausted output budget before final answer ({self.reason_label}={self.finish_reason})'
        else:
            message = f'LLM returned empty final response ({self.reason_label}={self.finish_reason})'
        super().__init__(message)


class EpisodeStopRequested(RuntimeError):
    """用于在外部请求停止时中断当前 episode 收集。"""


class LLMPlayer:
    """用 LLM 选动作跑完整局战斗，收集 trajectory 数据。"""

    THINK_CHOICES = ('true', 'false', 'low', 'medium', 'high')
    THINK_LEVELS = ('low', 'medium', 'high')

    def __init__(
        self,
        model: str = 'qwen3:4b',
        base_url: str = 'http://localhost:11434',
        timeout: float = 15.0,
        system_prompt: str | None = None,
        verbose: bool = False,
        debug: bool = False,
        api_key: str = 'ollama',
        think: str | bool = 'true',
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        retry_output_tokens: int | None = None,
        stream: bool = True,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.system_prompt = system_prompt or load_system_prompt()
        self.verbose = verbose
        self.debug = debug
        self.api_key = api_key
        self.think = self._resolve_think_setting(think, reasoning_effort=reasoning_effort)
        self.max_output_tokens = max(int(max_output_tokens), 16) if max_output_tokens is not None else None
        self.retry_output_tokens = max(int(retry_output_tokens), 16) if retry_output_tokens is not None else None
        self.stream = stream
        self._log_lock = threading.Lock()
        self._debug_system_prompt_printed = False
        self._openai_client = None

    @classmethod
    def _normalize_think_setting(cls, value: Any) -> str:
        if value is None:
            return 'true'
        if isinstance(value, bool):
            return 'true' if value else 'false'
        text = str(value).strip().lower()
        alias_map = {
            '1': 'true',
            '0': 'false',
            'yes': 'true',
            'no': 'false',
            'on': 'true',
            'off': 'false',
            'none': 'false',
        }
        text = alias_map.get(text, text)
        if text not in cls.THINK_CHOICES:
            allowed = ', '.join(cls.THINK_CHOICES)
            raise ValueError(f'unsupported think setting {value!r}; expected one of: {allowed}')
        return text

    @classmethod
    def _resolve_think_setting(cls, think: Any, *, reasoning_effort: str | None = None) -> str:
        normalized = cls._normalize_think_setting(think)
        if reasoning_effort is not None and normalized == 'true':
            return cls._normalize_think_setting(reasoning_effort)
        return normalized

    @staticmethod
    def _base_url_netloc(base_url: str) -> str:
        parsed = urlparse(base_url if '://' in base_url else f'http://{base_url}')
        return str(parsed.netloc or parsed.path or '').lower()

    def _is_ollama_backend(self) -> bool:
        netloc = self._base_url_netloc(self.base_url)
        return ':11434' in netloc or netloc.endswith('11434') or 'ollama' in netloc

    def _is_gpt_oss_model(self) -> bool:
        return self.model.strip().lower().startswith('gpt-oss')

    def _build_think_payload_entry(self) -> tuple[str, Any]:
        if self._is_ollama_backend():
            if self.think == 'true':
                return ('think', 'low' if self._is_gpt_oss_model() else True)
            if self.think == 'false':
                return ('think', False)
            return ('think', self.think)
        if self.think == 'false':
            return ('reasoning_effort', 'none')
        if self.think == 'true':
            return ('reasoning_effort', 'low')
        return ('reasoning_effort', self.think)

    def _print_locked(self, message: str, *, stream = None) -> None:
        target = stream or sys.stdout
        with self._log_lock:
            print(message, file=target, flush=True)

    def _debug_block(self, title: str, text: str, context: str = '') -> None:
        if not self.debug:
            return
        prefix = f'[{context}] ' if context else ''
        with self._log_lock:
            print(f'{prefix}{title} >>>', file=sys.stderr, flush=True)
            print(text, file=sys.stderr, flush=True)
            print(f'{prefix}<<< END {title}', file=sys.stderr, flush=True)

    def _debug_json(self, title: str, payload: Any, context: str = '') -> None:
        if not self.debug:
            return
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except TypeError:
            text = repr(payload)
        self._debug_block(title, text, context)

    @staticmethod
    def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
        chain: list[BaseException] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return chain

    def _format_exception_details(
        self,
        exc: BaseException,
        *,
        include_traceback: bool = False,
        request_context: str = '',
    ) -> str:
        lines: list[str] = []
        if request_context:
            lines.append(f'request: {request_context}')
        for index, current in enumerate(self._iter_exception_chain(exc)):
            prefix = 'error' if index == 0 else f'cause[{index}]'
            message = str(current) or repr(current)
            lines.append(f'{prefix}: {type(current).__name__}: {message}')
        if include_traceback:
            tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
            if tb:
                lines.append('traceback:')
                lines.append(tb)
        return '\n'.join(lines)

    def _debug_system_prompt_once(self, context: str = '') -> None:
        if not self.debug:
            return
        with self._log_lock:
            if self._debug_system_prompt_printed:
                return
            self._debug_system_prompt_printed = True
            prefix = f'[{context}] ' if context else ''
            print(f'{prefix}SYSTEM PROMPT >>>', file=sys.stderr, flush=True)
            print(self.system_prompt, file=sys.stderr, flush=True)
            print(f'{prefix}<<< END SYSTEM PROMPT', file=sys.stderr, flush=True)

    def _build_action_prompt(self, env: Any, last_action_label: str = '') -> str:
        """构造包含局面描述和合法动作列表的 prompt。"""
        rt = self._get_runtime(env)
        repo = self._get_repository(env)
        if rt is None:
            return '(无法获取运行时状态)'

        snapshot = build_state_snapshot(rt, repo, last_action_label)
        actions = extract_action_list_context(env)
        return render('action_select.jinja2', snapshot=snapshot, actions=actions)

    def _choose_action(self, env: Any, obs: dict, last_action_label: str = '', debug_context: str = '') -> int:
        """调用 LLM 选动作，解析返回的编号。"""
        prompt = self._build_action_prompt(env, last_action_label)
        mask = obs['action_mask']
        valid_actions = [i for i, m in enumerate(mask) if m > 0.5]

        if not valid_actions:
            raise RuntimeError('No valid actions available')

        try:
            final_text = self._call_llm(prompt, debug_context=debug_context)
            chosen = self._parse_action_from_final_output(final_text, valid_actions, debug_context=debug_context)
            if chosen is None:
                repair_prompt = self._build_repair_prompt(prompt, final_text, valid_actions)
                repaired_text = self._call_llm(repair_prompt, debug_context=f'{debug_context} repair')
                chosen = self._parse_action_from_final_output(
                    repaired_text,
                    valid_actions,
                    debug_context=f'{debug_context} repair',
                )
            if chosen is not None:
                return chosen
            if self.verbose:
                self._print_locked(
                    f'  [LLM] Invalid final output "{final_text.strip()}", falling back to first valid action',
                )
        except Exception as exc:
            request_context = f'model={self.model} base_url={self.base_url.rstrip("/")} timeout={self.timeout}s think={self.think}'
            detail = self._format_exception_details(
                exc,
                include_traceback=self.debug,
                request_context=request_context,
            )
            if self.debug:
                self._debug_block('LLM ERROR', detail, debug_context)
            if self.verbose:
                summary = detail.splitlines()[1] if detail.startswith('request: ') and len(detail.splitlines()) > 1 else detail.splitlines()[0]
                self._print_locked(f'  [LLM] Error: {summary}, falling back to first valid action')

        return valid_actions[0]

    def _call_llm(self, prompt: str, debug_context: str = '') -> str:
        """调用 LLM API，返回最终文本。"""
        current_budget = self.max_output_tokens
        try:
            return self._call_llm_openai(prompt, debug_context=debug_context, max_output_tokens=current_budget)
        except EmptyFinalResponseError as exc:
            if not exc.retryable:
                raise
            retry_candidates = [1024]
            if self.retry_output_tokens is not None:
                retry_candidates.append(self.retry_output_tokens)
            if current_budget is not None:
                retry_candidates.append(current_budget * 2)
            else:
                retry_candidates.append(2048)
            retry_tokens = max(retry_candidates)
            if self.verbose:
                self._print_locked(
                    f'  [LLM] {exc}; retrying with larger output budget ({retry_tokens})',
                )
            text = self._call_llm_openai(
                prompt,
                debug_context=f'{debug_context} retry=budget',
                max_output_tokens=retry_tokens,
            )
            if self.max_output_tokens is None or retry_tokens > self.max_output_tokens:
                self.max_output_tokens = retry_tokens
                if self.verbose:
                    self._print_locked(
                        f'  [LLM] Promoted output budget to {self.max_output_tokens} for subsequent requests',
                    )
            return text

    # @staticmethod
    # def _openai_base_url(base_url: str) -> str:
    #     base = base_url.rstrip('/')
    #     if base.endswith('/v1/chat/completions'):
    #         return base[: -len('/chat/completions')] + '/'
    #     if base.endswith('/v1'):
    #         return f'{base}/'
    #     return f'{base}/v1/'

    def _get_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        self._openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key or 'ollama')
        return self._openai_client

    @staticmethod
    def _coerce_reasoning_text(value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    summary = item.get('summary')
                    if summary is not None:
                        parts.append(LLMPlayer._coerce_reasoning_text(summary))
                        continue
                    text = item.get('text')
                    if text is not None:
                        parts.append(LLMPlayer._coerce_reasoning_text(text))
                        continue
                    content = item.get('content')
                    if content is not None:
                        parts.append(LLMPlayer._coerce_reasoning_text(content))
            return ''.join(parts)
        if isinstance(value, dict):
            if 'text' in value:
                return LLMPlayer._coerce_reasoning_text(value.get('text'))
            if 'content' in value:
                return LLMPlayer._coerce_reasoning_text(value.get('content'))
            if 'summary' in value:
                return LLMPlayer._coerce_reasoning_text(value.get('summary'))
        return str(value)

    @staticmethod
    def _coerce_json_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, 'model_dump'):
            dumped = value.model_dump(mode='json')
            return dumped if isinstance(dumped, dict) else {}
        if hasattr(value, 'to_dict'):
            dumped = value.to_dict()
            return dumped if isinstance(dumped, dict) else {}
        try:
            dumped = dict(value)
        except Exception:
            dumped = getattr(value, '__dict__', {})
        return dumped if isinstance(dumped, dict) else {}

    @staticmethod
    def _coerce_final_output_text(value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get('type') or '')
                if item_type and item_type not in {'text', 'output_text'}:
                    continue
                text = item.get('text')
                if text is not None:
                    parts.append(LLMPlayer._coerce_final_output_text(text))
                    continue
                content = item.get('content')
                if content is not None:
                    parts.append(LLMPlayer._coerce_final_output_text(content))
            return ''.join(parts)
        if isinstance(value, dict):
            text = value.get('text')
            if text is not None:
                return LLMPlayer._coerce_final_output_text(text)
            content = value.get('content')
            if content is not None:
                return LLMPlayer._coerce_final_output_text(content)
        return str(value)

    @staticmethod
    def _strip_reasoning_markup(text: str) -> str:
        cleaned = text
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def _parse_action_from_final_output(self, text: str, valid_actions: list[int], debug_context: str = '') -> int | None:
        cleaned = self._strip_reasoning_markup(text)
        if self.debug:
            self._debug_block('PARSED FINAL OUTPUT', cleaned or '(empty)', debug_context)
        if not cleaned:
            return None
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        candidates = lines[:1] if lines else [cleaned]
        for chunk in candidates:
            exact = re.fullmatch(r'\D*(\d+)\D*', chunk)
            if exact:
                chosen = int(exact.group(1))
                if chosen in valid_actions:
                    return chosen
        numbers = re.findall(r'\d+', cleaned)
        for number in reversed(numbers):
            chosen = int(number)
            if chosen in valid_actions:
                return chosen
        return None

    def _build_repair_prompt(self, original_prompt: str, invalid_output: str, valid_actions: list[int]) -> str:
        valid_text = ', '.join(str(action) for action in valid_actions)
        return (
            f'{original_prompt}\n\n'
            '## 修正要求\n'
            '你上一个回答无效，因为它没有严格只输出一个合法动作编号。\n'
            f'合法动作编号只有这些：{valid_text}\n'
            f'你上一个无效回答：{invalid_output.strip() or "(empty)"}\n\n'
            '现在不要解释，不要分析，不要重复题面，只输出一个合法动作编号。'
        )

    def _call_llm_openai(self, prompt: str, debug_context: str = '', max_output_tokens: int | None = None) -> str:
        """调用官方 openai 客户端，经 /v1/chat/completions 访问兼容接口。"""
        if self.stream:
            return self._call_llm_openai_stream(prompt, debug_context=debug_context, max_output_tokens=max_output_tokens)
        return self._call_llm_openai_once(prompt, debug_context=debug_context, max_output_tokens=max_output_tokens)

    def _build_openai_payload(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
        stream: bool,
    ) -> dict[str, Any]:
        output_tokens = max_output_tokens or self.max_output_tokens
        think_key, think_value = self._build_think_payload_entry()
        payload_obj = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt},
            ],
            'stream': stream,
            'temperature': 0.3,
            think_key: think_value,
        }
        if output_tokens is not None:
            payload_obj['max_tokens'] = output_tokens
        return payload_obj

    def _supports_think_retry(self, exc: BaseException) -> bool:
        """判断错误是否说明当前模型/服务端不支持 think / reasoning_effort 参数。"""

        messages = [str(item or '') for item in self._iter_exception_chain(exc)]
        haystack = '\n'.join(messages).lower()
        unsupported_markers = (
            'think value',
            'reasoning_effort',
            'reasoning effort',
            'not supported for this model',
            'unsupported value',
            'unsupported parameter',
        )
        return any(marker in haystack for marker in unsupported_markers)

    @staticmethod
    def _payload_signature(payload_obj: dict[str, Any]) -> str:
        return json.dumps(payload_obj, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _build_openai_request_kwargs(payload_obj: dict[str, Any]) -> dict[str, Any]:
        request_kwargs = dict(payload_obj)
        if 'think' in request_kwargs:
            think_value = request_kwargs.pop('think')
            extra_body = request_kwargs.get('extra_body')
            if not isinstance(extra_body, dict):
                extra_body = {}
            else:
                extra_body = dict(extra_body)
            extra_body['think'] = think_value
            request_kwargs['extra_body'] = extra_body
        return request_kwargs

    def _debug_openai_request_payload(self, payload_obj: dict[str, Any], debug_context: str = '') -> None:
        self._debug_json('OPENAI REQUEST PAYLOAD', self._build_openai_request_kwargs(payload_obj), debug_context)

    def _write_stream_debug(
        self,
        title: str,
        started: bool,
        text: str = '',
        context: str = '',
        *,
        close_after: bool = False,
    ) -> bool:
        if not self.debug:
            return started
        if not text and not close_after:
            return started
        if close_after and not started and not text:
            return started

        prefix = f'[{context}] ' if context else ''
        with self._log_lock:
            if not started:
                print(f'{prefix}{title} >>>', file=sys.stderr, flush=True)
                started = True
            if text:
                sys.stderr.write(text)
                sys.stderr.flush()
            if close_after:
                if not text.endswith('\n'):
                    print(file=sys.stderr, flush=True)
                print(f'{prefix}<<< END {title}', file=sys.stderr, flush=True)
                return False
        return started

    def _build_openai_retry_payloads(self, payload_obj: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
        fallbacks: list[tuple[dict[str, Any], str]] = []
        if 'think' in payload_obj:
            rejected_value = payload_obj.get('think')
            if isinstance(rejected_value, str) and rejected_value in self.THINK_LEVELS:
                fallback_true = dict(payload_obj)
                fallback_true['think'] = True
                fallbacks.append(
                    (
                        fallback_true,
                        f'server/model rejected think={rejected_value!r}; retrying with think=True for compatibility',
                    )
                )
            fallback_without = dict(payload_obj)
            fallback_without.pop('think', None)
            fallbacks.append(
                (
                    fallback_without,
                    f'server/model rejected think={rejected_value!r}; retrying without think for compatibility',
                )
            )
        if 'reasoning_effort' in payload_obj:
            rejected_value = payload_obj.get('reasoning_effort')
            fallback_without = dict(payload_obj)
            fallback_without.pop('reasoning_effort', None)
            fallbacks.append(
                (
                    fallback_without,
                    f'server/model rejected reasoning_effort={rejected_value!r}; '
                    'retrying without reasoning_effort for compatibility',
                )
            )
        return fallbacks

    def _create_openai_response(
        self,
        client: Any,
        payload_obj: dict[str, Any],
        *,
        debug_context: str = '',
    ):
        """向 OpenAI-compatible 接口发请求，并在必要时自动降级 think / reasoning_effort。"""

        try:
            request_kwargs = self._build_openai_request_kwargs(payload_obj)
            return client.chat.completions.create(timeout=self.timeout, **request_kwargs), payload_obj
        except Exception as exc:
            if not self._supports_think_retry(exc):
                raise
            seen_signatures = {self._payload_signature(payload_obj)}
            last_exc: BaseException = exc
            for fallback_payload, note in self._build_openai_retry_payloads(payload_obj):
                signature = self._payload_signature(fallback_payload)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                if self.debug:
                    self._debug_block('OPENAI COMPAT FALLBACK', note, debug_context)
                    self._debug_json(
                        'OPENAI REQUEST PAYLOAD FALLBACK',
                        self._build_openai_request_kwargs(fallback_payload),
                        debug_context,
                    )
                if self.verbose:
                    self._print_locked(f'  [LLM] {note}')
                try:
                    request_kwargs = self._build_openai_request_kwargs(fallback_payload)
                    response = client.chat.completions.create(timeout=self.timeout, **request_kwargs)
                    return response, fallback_payload
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if not self._supports_think_retry(retry_exc):
                        raise
            raise last_exc

    def _format_openai_request_context(self, payload_obj: dict[str, Any]) -> str:
        request_kwargs = self._build_openai_request_kwargs(payload_obj)
        parts = [
            f'model={self.model}',
            f'base_url={self.base_url.rstrip("/")}',
            f'timeout={self.timeout}s',
            f'stream={request_kwargs.get("stream")}',
            f'max_tokens={request_kwargs.get("max_tokens", "server-default")}',
        ]
        extra_body = request_kwargs.get('extra_body')
        if isinstance(extra_body, dict) and 'think' in extra_body:
            parts.append(f'think={extra_body.get("think")}')
        if 'reasoning_effort' in request_kwargs:
            parts.append(f'reasoning_effort={request_kwargs.get("reasoning_effort")}')
        return ' '.join(parts)

    def _call_llm_openai_once(self, prompt: str, debug_context: str = '', max_output_tokens: int | None = None) -> str:
        self._debug_system_prompt_once(debug_context)
        self._debug_block('USER PROMPT', prompt, debug_context)
        payload_obj = self._build_openai_payload(prompt, max_output_tokens=max_output_tokens, stream=False)
        self._debug_openai_request_payload(payload_obj, debug_context)
        client = self._get_openai_client()
        try:
            response, payload_obj = self._create_openai_response(client, payload_obj, debug_context=debug_context)
        except Exception as exc:
            request_context = self._format_openai_request_context(payload_obj)
            if self.debug:
                self._debug_block(
                    'OPENAI REQUEST ERROR',
                    self._format_exception_details(exc, include_traceback=True, request_context=request_context),
                    debug_context,
                )
            raise
        result = self._coerce_json_dict(response)
        choices = result.get('choices') or []
        first_choice = choices[0] if choices else {}
        message = self._coerce_json_dict(first_choice.get('message'))
        reasoning_text = self._coerce_reasoning_text(
            message.get('reasoning_content') or message.get('reasoning')
        )
        # if reasoning_text:
        #     self._debug_block('MODEL THINKING', reasoning_text, debug_context)
        raw_text = self._coerce_final_output_text(message.get('content'))
        if not raw_text.strip():
            finish_reason = str(first_choice.get('finish_reason') or '')
            raise EmptyFinalResponseError('finish_reason', finish_reason, thinking_len=len(reasoning_text.strip()))
        self._debug_block('MODEL OUTPUT', raw_text, debug_context)
        return raw_text.strip()

    def _call_llm_openai_stream(self, prompt: str, debug_context: str = '', max_output_tokens: int | None = None) -> str:
        self._debug_system_prompt_once(debug_context)
        self._debug_block('USER PROMPT', prompt, debug_context)
        payload_obj = self._build_openai_payload(prompt, max_output_tokens=max_output_tokens, stream=True)
        self._debug_openai_request_payload(payload_obj, debug_context)
        client = self._get_openai_client()
        try:
            response, payload_obj = self._create_openai_response(client, payload_obj, debug_context=debug_context)
        except Exception as exc:
            request_context = self._format_openai_request_context(payload_obj)
            if self.debug:
                self._debug_block(
                    'OPENAI REQUEST ERROR',
                    self._format_exception_details(exc, include_traceback=True, request_context=request_context),
                    debug_context,
                )
            raise

        reasoning_parts: list[str] = []
        output_parts: list[str] = []
        reasoning_debug_started = False
        output_debug_started = False
        finish_reason = ''
        close_response = getattr(response, 'close', None)
        try:
            for chunk in response:
                chunk_payload = self._coerce_json_dict(chunk)
                choices = chunk_payload.get('choices') or []
                first_choice = choices[0] if choices else {}
                finish_reason = str(first_choice.get('finish_reason') or finish_reason or '')
                delta = self._coerce_json_dict(first_choice.get('delta') or first_choice.get('message'))
                reasoning_piece = self._coerce_reasoning_text(
                    delta.get('reasoning_content') or delta.get('reasoning')
                )
                if reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
                    reasoning_debug_started = self._write_stream_debug(
                        'MODEL THINKING CHUNK',
                        reasoning_debug_started,
                        reasoning_piece,
                        debug_context,
                    )
                output_piece = self._coerce_final_output_text(delta.get('content'))
                if output_piece:
                    output_parts.append(output_piece)
                    output_debug_started = self._write_stream_debug(
                        'MODEL OUTPUT CHUNK',
                        output_debug_started,
                        output_piece,
                        debug_context,
                    )
        except Exception as exc:
            request_context = self._format_openai_request_context(payload_obj)
            request_context = (
                f'{request_context} partial_reasoning_chars={sum(len(part) for part in reasoning_parts)} '
                f'partial_output_chars={sum(len(part) for part in output_parts)}'
            )
            if self.debug:
                self._debug_block(
                    'OPENAI STREAM ERROR',
                    self._format_exception_details(exc, include_traceback=True, request_context=request_context),
                    debug_context,
                )
            raise
        finally:
            reasoning_debug_started = self._write_stream_debug(
                'MODEL THINKING CHUNK',
                reasoning_debug_started,
                context=debug_context,
                close_after=True,
            )
            output_debug_started = self._write_stream_debug(
                'MODEL OUTPUT CHUNK',
                output_debug_started,
                context=debug_context,
                close_after=True,
            )
            if callable(close_response):
                close_response()

        reasoning_text = ''.join(reasoning_parts)
        raw_text = ''.join(output_parts)
        if reasoning_text:
            self._debug_block('MODEL THINKING', reasoning_text, debug_context)
        if not raw_text.strip():
            raise EmptyFinalResponseError('finish_reason', finish_reason, thinking_len=len(reasoning_text.strip()))
        self._debug_block('MODEL OUTPUT', raw_text, debug_context)
        return raw_text.strip()

    @staticmethod
    def _get_runtime(env: Any):
        """穿透 wrapper 链获取 ExamRuntime。"""
        current = env
        while hasattr(current, 'env'):
            if hasattr(current, 'runtime'):
                return current.runtime
            current = current.env
        return getattr(current, 'runtime', None)

    @staticmethod
    def _get_repository(env: Any):
        """穿透 wrapper 链获取 MasterDataRepository。"""
        current = env
        while hasattr(current, 'env'):
            if hasattr(current, 'repository'):
                return current.repository
            current = current.env
        return getattr(current, 'repository', None)

    def _run_single_episode(
        self,
        env_config: dict,
        episode_id: int,
        seed: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """构建独立环境并执行单局，便于并发收集轨迹。"""

        env_cfg = dict(env_config)
        env_cfg['include_action_labels_in_step_info'] = True
        if seed is not None:
            env_cfg['seed'] = seed
        env = build_env_from_config(env_cfg)
        try:
            t0 = time.time()
            trajectory = self.run_episode(env, episode_id=episode_id, stop_event=stop_event)
            elapsed = time.time() - t0
        finally:
            close = getattr(env, 'close', None)
            if callable(close):
                close()

        ep_reward = sum(record['reward'] for record in trajectory)
        ep_score = trajectory[-1]['info'].get('score', 0) if trajectory else 0
        return {
            'episode_id': episode_id,
            'trajectory': trajectory,
            'elapsed': elapsed,
            'steps': len(trajectory),
            'reward': ep_reward,
            'score': ep_score,
        }

    def run_episode(
        self,
        env: Any,
        episode_id: int = 0,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """跑完一局，收集 trajectory 数据。"""
        obs, info = env.reset()
        trajectory: list[dict] = []
        step = 0
        last_action_label = ''

        while True:
            if stop_event is not None and stop_event.is_set():
                raise EpisodeStopRequested(f'episode={episode_id} stop requested')
            debug_context = f'ep={episode_id} step={step}'
            action = self._choose_action(env, obs, last_action_label, debug_context=debug_context)
            candidates = getattr(env, '_candidates', None)
            if candidates is None:
                # 穿透 wrapper
                inner = env
                while hasattr(inner, 'env'):
                    if hasattr(inner, '_candidates'):
                        candidates = inner._candidates
                        break
                    inner = inner.env

            action_label = ''
            if candidates and action < len(candidates):
                action_label = action_label_for_llm(env, action)

            obs_record = {
                'global': obs['global'].tolist(),
                'action_features': obs['action_features'].tolist(),
                'action_mask': obs['action_mask'].tolist(),
            }

            next_obs, reward, terminated, truncated, info = env.step(action)

            record = {
                'episode_id': episode_id,
                'step': step,
                'obs': obs_record,
                'action': action,
                'reward': float(reward),
                'terminated': bool(terminated),
                'info': {
                    'score': float(info.get('score', info.get('final_score', 0))),
                    'turn': int(info.get('turn', 0)),
                    'action_label': action_label,
                    'battle_kind': str(info.get('battle_kind', '')),
                    'reward_mode': str(info.get('reward_mode', '')),
                    'clear_state': str(info.get('clear_state', '')),
                    'scenario': str(info.get('scenario', '')),
                    'stage_type': str(info.get('stage_type', '')),
                    'lesson_action_type': str(info.get('lesson_action_type', '')),
                    'lesson_level_index': int(info.get('lesson_level_index', 0) or 0),
                    'lesson_target_value': float(info.get('lesson_target_value', 0.0) or 0.0),
                    'lesson_perfect_value': float(info.get('lesson_perfect_value', 0.0) or 0.0),
                    'lesson_target_remaining': float(info.get('lesson_target_remaining', 0.0) or 0.0),
                    'lesson_perfect_remaining': float(info.get('lesson_perfect_remaining', 0.0) or 0.0),
                },
            }
            trajectory.append(record)

            if self.verbose:
                self._print_locked(
                    f'  step={step} action={action} ({action_label}) reward={reward:.4f} terminated={terminated}',
                )

            obs = next_obs
            last_action_label = action_label
            step += 1

            if terminated or truncated:
                break

        return trajectory

    def run_episodes(
        self,
        env_config: dict,
        n: int,
        output_path: str | Path,
        seed: int | None = None,
        workers: int = 1,
    ) -> dict:
        """跑 n 局，逐行写入 jsonl。返回统计摘要。"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        total_steps = 0
        total_reward = 0.0
        scores: list[float] = []
        max_workers = max(int(workers), 1)

        def _episode_seed(episode_id: int) -> int | None:
            return None if seed is None else seed + episode_id

        def _write_episode(handle, result: dict[str, Any]) -> None:
            for record in result['trajectory']:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

        def _build_summary(*, interrupted: bool, completed_episodes: int) -> dict[str, Any]:
            summary = {
                'episodes': completed_episodes,
                'requested_episodes': int(n),
                'total_steps': total_steps,
                'mean_reward': total_reward / max(completed_episodes, 1),
                'mean_score': float(np.mean(scores)) if scores else 0.0,
                'std_score': float(np.std(scores)) if scores else 0.0,
                'workers': max_workers,
                'output': str(output),
                'interrupted': bool(interrupted),
            }
            print(f'[Summary] {json.dumps(summary, ensure_ascii=False)}')
            return summary

        with output.open('w', encoding='utf-8') as f:
            if max_workers == 1:
                completed_episodes = 0
                try:
                    for ep in range(n):
                        result = self._run_single_episode(env_config, ep, seed=_episode_seed(ep))
                        _write_episode(f, result)
                        total_steps += int(result['steps'])
                        total_reward += float(result['reward'])
                        scores.append(float(result['score']))
                        completed_episodes += 1
                        print(
                            f'[Episode {ep + 1}/{n}] steps={result["steps"]} '
                            f'reward={result["reward"]:.4f} score={result["score"]:.0f} '
                            f'time={result["elapsed"]:.1f}s'
                        )
                except KeyboardInterrupt:
                    self._print_locked('[Interrupted] Stop requested; returning partial trajectory summary.')
                    return _build_summary(interrupted=True, completed_episodes=completed_episodes)
                return _build_summary(interrupted=False, completed_episodes=completed_episodes)
            else:
                stop_event = threading.Event()
                task_queue: queue.Queue[tuple[int, int | None] | None] = queue.Queue()
                result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
                for ep in range(n):
                    task_queue.put((ep, _episode_seed(ep)))

                def _worker_loop(worker_id: int) -> None:
                    while not stop_event.is_set():
                        try:
                            task = task_queue.get_nowait()
                        except queue.Empty:
                            return
                        if task is None:
                            task_queue.task_done()
                            return
                        ep, ep_seed = task
                        try:
                            if stop_event.is_set():
                                continue
                            result = self._run_single_episode(
                                env_config,
                                ep,
                                seed=ep_seed,
                                stop_event=stop_event,
                            )
                            result_queue.put(('result', result))
                        except EpisodeStopRequested:
                            result_queue.put(('stopped', {'episode_id': ep}))
                            return
                        except BaseException as exc:
                            result_queue.put(('error', {'episode_id': ep, 'exception': exc}))
                            stop_event.set()
                            return
                        finally:
                            task_queue.task_done()

                threads: list[threading.Thread] = []
                for worker_id in range(max_workers):
                    thread = threading.Thread(
                        target=_worker_loop,
                        args=(worker_id,),
                        name=f'llm-player-{worker_id}',
                        daemon=True,
                    )
                    thread.start()
                    threads.append(thread)

                completed_episodes = 0
                pending_results = int(n)
                interrupted = False
                try:
                    while pending_results > 0:
                        try:
                            kind, payload = result_queue.get(timeout=0.2)
                        except queue.Empty:
                            alive = any(thread.is_alive() for thread in threads)
                            if not alive and result_queue.empty():
                                break
                            continue
                        if kind == 'result':
                            result = payload
                            completed_episodes += 1
                            pending_results -= 1
                            _write_episode(f, result)
                            total_steps += int(result['steps'])
                            total_reward += float(result['reward'])
                            scores.append(float(result['score']))
                            ep = int(result['episode_id'])
                            print(
                                f'[Episode {ep + 1}/{n}] steps={result["steps"]} '
                                f'reward={result["reward"]:.4f} score={result["score"]:.0f} '
                                f'time={result["elapsed"]:.1f}s'
                            )
                        elif kind == 'stopped':
                            pending_results -= 1
                        elif kind == 'error':
                            exc = payload['exception']
                            if isinstance(exc, KeyboardInterrupt):
                                interrupted = True
                                stop_event.set()
                                break
                            raise exc
                except KeyboardInterrupt:
                    interrupted = True
                    stop_event.set()
                    self._print_locked('[Interrupted] Stop requested; cancelling pending episodes.')
                finally:
                    stop_event.set()
                    for _ in threads:
                        task_queue.put(None)
                    for thread in threads:
                        thread.join(timeout=0.1)
                return _build_summary(interrupted=interrupted, completed_episodes=completed_episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LLM player: generate demonstration trajectories.')
    parser.add_argument('--mode', default='exam', choices=('exam', 'lesson'))
    parser.add_argument('--scenario', default='nia_master')
    parser.add_argument('--stage-type', default=None, help='exam 模式下的考试阶段；lesson 模式下忽略')
    parser.add_argument('--exam-reward-mode', default='score', choices=('score', 'clear'))
    parser.add_argument('--lesson-action-type', default='', help='lesson 模式下固定课程类型；留空时按环境默认课程池采样')
    parser.add_argument('--lesson-level-index', type=int, default=0, help='lesson 模式下固定课程等级序号；0 表示按主数据候选采样')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--model', default='qwen3:4b', help='model name')
    parser.add_argument('--base-url', default='http://localhost:11434', help='OpenAI-compatible API base URL')
    parser.add_argument(
        '--api-key',
        default='ollama',
        help='API key for the OpenAI-compatible endpoint',
    )
    parser.add_argument('--timeout', type=float, default=15.0)
    parser.add_argument('--output', default=str(TRAJECTORIES_DIR / 'llm_demo.jsonl'))
    parser.add_argument('--workers', type=int, default=1, help='parallel episodes to generate in parallel')
    parser.add_argument('--idol-card-id', default='')
    parser.add_argument('--producer-level', type=int, default=35)
    parser.add_argument('--idol-rank', type=int, default=0)
    parser.add_argument('--dearness-level', type=int, default=0)
    parser.add_argument('--use-after-item', action='store_true')
    parser.add_argument(
        '--manual-exam-setup',
        action='append',
        default=[],
        help='manual exam setup jsonl; when provided, episodes sample real deck/drink/item records',
    )
    parser.add_argument(
        '--guarantee-card-effect',
        action='append',
        default=[],
        help='ensure at least N cards of one effect tag, e.g. review=3, 打分=4, 元气=2',
    )
    parser.add_argument(
        '--force-card',
        action='append',
        default=[],
        help='force player-axis cards into the random initial deck via JSON or @json file, e.g. {"好印象":["p_card..."]} or {"干劲":["p_card..."]}',
    )
    parser.add_argument('--include-deck-features', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--system-prompt-file', default='', help='replace built-in system prompt with a local text/markdown file')
    parser.add_argument(
        '--append-system-prompt-file',
        action='append',
        default=[],
        help='append extra local rule/strategy notes to the built-in system prompt; can be repeated',
    )
    parser.add_argument(
        '--think',
        nargs='?',
        const='true',
        default='true',
        type=lambda value: str(value).lower(),
        choices=LLMPlayer.THINK_CHOICES,
        help='thinking mode for compatible backends: true / false / low / medium / high (default: true)',
    )
    parser.add_argument(
        '--max-output-tokens',
        type=int,
        default=None,
        help='explicitly set max output tokens; omitted by default so the server decides',
    )
    parser.add_argument(
        '--retry-output-tokens',
        type=int,
        default=None,
        help='output token budget used after a length cutoff; defaults to 2048 when no explicit max is set',
    )
    parser.add_argument('--stream', dest='stream', action='store_true', help='use streaming responses (default)')
    parser.add_argument('--no-stream', dest='stream', action='store_false', help='disable streaming and wait for the full response')
    parser.add_argument('--no-think', dest='think', action='store_const', const='false', help=argparse.SUPPRESS)
    parser.add_argument('--reasoning-effort', dest='think', choices=LLMPlayer.THINK_LEVELS, help=argparse.SUPPRESS)
    parser.set_defaults(stream=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true', help='print prompts sent to the LLM and raw model outputs')
    parser.add_argument('--dry-run', action='store_true', help='Run 1 episode and print trajectory to stdout')
    return parser.parse_args()


def _build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    """把 LLM 对弈 CLI 参数整理成统一环境配置。"""

    return {
        'mode': str(args.mode or 'exam'),
        'scenario': args.scenario,
        'stage_type': args.stage_type,
        'exam_reward_mode': args.exam_reward_mode,
        'seed': args.seed,
        'idol_card_id': args.idol_card_id,
        'producer_level': args.producer_level,
        'idol_rank': args.idol_rank,
        'dearness_level': args.dearness_level,
        'use_after_item': True if args.use_after_item else None,
        'lesson_action_type': args.lesson_action_type,
        'lesson_level_index': args.lesson_level_index,
        'manual_exam_setup_paths': list(args.manual_exam_setup),
        'guarantee_card_effects': list(args.guarantee_card_effect),
        'force_card_groups': list(args.force_card),
        'include_deck_features': args.include_deck_features,
    }


def main() -> int:
    args = parse_args()

    env_config = _build_env_config(args)

    base_prompt = _read_text_file(args.system_prompt_file) if args.system_prompt_file else None
    extra_prompt_sections = [_read_text_file(path) for path in args.append_system_prompt_file]
    system_prompt = load_system_prompt(extra_sections=extra_prompt_sections, base_prompt=base_prompt)

    player = LLMPlayer(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        system_prompt=system_prompt,
        verbose=args.verbose,
        debug=args.debug,
        api_key=args.api_key,
        think=args.think,
        max_output_tokens=args.max_output_tokens,
        retry_output_tokens=args.retry_output_tokens,
        stream=args.stream,
    )

    if args.dry_run:
        env_config['include_action_labels_in_step_info'] = True
        env = build_env_from_config(env_config)
        trajectory = player.run_episode(env, episode_id=0)
        for record in trajectory:
            print(json.dumps(record, ensure_ascii=False))
        return 0

    player.run_episodes(
        env_config=env_config,
        n=args.episodes,
        output_path=args.output,
        seed=args.seed,
        workers=args.workers,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
