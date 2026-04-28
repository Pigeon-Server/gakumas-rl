"""LLM 在线 reward shaping wrapper — 多维局面评估。"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from .state_snapshot import build_state_snapshot
from .prompt_renderer import render, load_system_prompt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMRewardConfig:
    """LLM reward shaping 配置。"""

    enabled: bool = False
    model: str = 'qwen3:4b'
    base_url: str = 'http://localhost:11434'
    reward_weight: float = 0.3
    eval_interval: int = 1
    timeout: float = 5.0
    fallback_reward: float = 0.0
    cache_size: int = 256
    system_prompt: str | None = None
    dimension_weights: tuple[float, ...] = (0.3, 0.2, 0.3, 0.2)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class LLMRewardShaper(gym.Wrapper):
    """在 base env 的 reward 上叠加 LLM 多维局面评估信号。"""

    def __init__(self, env: gym.Env, config: LLMRewardConfig):
        super().__init__(env)
        self.config = config
        self._step_count = 0
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._last_action_label: str = ''

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._last_action_label = info.get('action_label', info.get('action', ''))

        if self.config.enabled and self._step_count % self.config.eval_interval == 0:
            scores = self._evaluate_state()
            weights = self.config.dimension_weights
            llm_reward = sum(w * s for w, s in zip(weights, scores))
            info['llm_reward'] = llm_reward
            info['llm_dimensions'] = {
                'tempo': scores[0],
                'resource': scores[1],
                'burst': scores[2],
                'risk': scores[3],
            }
            reward = reward + self.config.reward_weight * llm_reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._step_count = 0
        self._last_action_label = ''
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    # Runtime 访问
    # ------------------------------------------------------------------

    def _get_runtime(self):
        """穿透 wrapper 链获取 ExamRuntime。"""
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'runtime'):
                return env.runtime
            env = env.env
        return getattr(env, 'runtime', None)

    def _get_repository(self):
        """穿透 wrapper 链获取 MasterDataRepository。"""
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'repository'):
                return env.repository
            env = env.env
        return getattr(env, 'repository', None)

    # ------------------------------------------------------------------
    # 状态快照构建
    # ------------------------------------------------------------------

    def _build_state_snapshot(self) -> str:
        """从 ExamRuntime 提取完整局面描述（委托给共享模块）。"""
        rt = self._get_runtime()
        if rt is None:
            return '(无法获取运行时状态)'
        repo = self._get_repository()
        return build_state_snapshot(rt, repo, self._last_action_label)

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        """构造多维评估 prompt（user message 部分）。"""
        snapshot = self._build_state_snapshot()
        return render('reward_eval.jinja2', snapshot=snapshot)

    # ------------------------------------------------------------------
    # LLM 调用与解析
    # ------------------------------------------------------------------

    def _evaluate_state(self) -> list[float]:
        """调用 LLM 评估当前局面，返回 4 维评分。"""
        cache_key = self._cache_key()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt()
        try:
            scores = self._call_llm(prompt)
            if len(self._cache) >= self.config.cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = scores
            return scores
        except Exception:
            fallback = [self.config.fallback_reward] * 4
            return fallback

    def _call_llm(self, prompt: str) -> list[float]:
        """调用 ollama API，解析多维评分。"""
        system_prompt = self.config.system_prompt or load_system_prompt()
        payload = json.dumps({
            'model': self.config.model,
            'system': system_prompt,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 32},
        }).encode()
        req = urllib.request.Request(
            f'{self.config.base_url}/api/generate',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            result = json.loads(resp.read())
        text = result.get('response', '').strip()
        return self._parse_scores(text)

    @staticmethod
    def _parse_scores(text: str) -> list[float]:
        """从 LLM 输出中解析 4 个浮点数。"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if len(numbers) < 4:
            raise ValueError(f'Expected 4 scores, got {len(numbers)}: {text!r}')
        return [_clamp(float(n)) for n in numbers[:4]]

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """基于完整运行时状态生成缓存 key。"""
        rt = self._get_runtime()
        if rt is None:
            return hashlib.md5(str(self._step_count).encode()).hexdigest()

        res = rt.resources
        key_parts = (
            rt.turn,
            int(rt.score),
            int(rt.stamina),
            rt.stance,
            rt.stance_level,
            len(rt.hand),
            len(rt.deck),
            int(res.get('block', 0)),
            int(res.get('review', 0)),
            int(res.get('aggressive', 0)),
            int(res.get('parameter_buff', 0)),
            int(res.get('lesson_buff', 0)),
            int(res.get('full_power_point', 0)),
            int(res.get('enthusiastic', 0)),
            len(rt.active_effects),
            len(rt.active_enchants),
            tuple(c.card_id for c in rt.hand),
        )
        return hashlib.md5(str(key_parts).encode()).hexdigest()
