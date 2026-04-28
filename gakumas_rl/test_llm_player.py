"""LLM player OpenAI-compatible API 与响应解析测试。"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from gakumas_rl.bc_pretrain import BCTrainer
from gakumas_rl.bc_pretrain import _build_env_config as build_bc_env_config
from gakumas_rl.bc_pretrain import parse_args as parse_bc_args
from gakumas_rl.llm_player import LLMPlayer, _build_env_config as build_llm_env_config
from gakumas_rl.llm_player import parse_args
from gakumas_rl.prompt_renderer import load_system_prompt, render


class _FakeOpenAIResponse:
    """最小 openai client 响应桩。"""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def model_dump(self, mode: str = 'json') -> dict[str, Any]:
        return self._payload


class _FakeOpenAIStream:
    """最小流式响应桩。"""

    def __init__(self, payloads: list[dict[str, Any]]):
        self._payloads = payloads
        self.closed = False

    def __iter__(self):
        for payload in self._payloads:
            yield _FakeOpenAIResponse(payload)

    def close(self) -> None:
        self.closed = True


class _FakeChatCompletions:
    """记录 create 调用参数。"""

    def __init__(self, payloads: list[dict[str, Any]], calls: list[dict[str, Any]]):
        self._payloads = payloads
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        index = len(self._calls) - 1
        payload = self._payloads[index]
        if kwargs.get('stream'):
            chunks = payload if isinstance(payload, list) else [payload]
            return _FakeOpenAIStream(chunks)
        return _FakeOpenAIResponse(payload)


class _FakeOpenAIClient:
    """最小 openai.OpenAI client 桩。"""

    def __init__(self, payloads: list[dict[str, Any]], calls: list[dict[str, Any]]):
        self.chat = type('ChatNamespace', (), {'completions': _FakeChatCompletions(payloads, calls)})()


class _ExplodingChatCompletions:
    """抛出底层超时异常的 completions 桩。"""

    def create(self, **kwargs):
        try:
            raise TimeoutError('socket connect timed out')
        except TimeoutError as inner:
            raise RuntimeError('Request timed out.') from inner


class _ExplodingOpenAIClient:
    """请求阶段直接失败的 openai client 桩。"""

    def __init__(self):
        self.chat = type('ChatNamespace', (), {'completions': _ExplodingChatCompletions()})()


class _UnsupportedThinkChatCompletions:
    """第一次拒绝 think 分级，第二次接受兼容回退的 completions 桩。"""

    def __init__(self, calls: list[dict[str, Any]]):
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        if len(self._calls) == 1:
            raise RuntimeError(
                'Error code: 400 - {"error": {"message": "think value \\"low\\" is not supported for this model"}}'
            )
        return _FakeOpenAIStream(
            [
                {'choices': [{'delta': {'content': '7'}, 'finish_reason': None}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            ]
        )


class _UnsupportedThinkOpenAIClient:
    """用于测试 think/reasoning 兼容回退。"""

    def __init__(self, calls: list[dict[str, Any]]):
        self.chat = type('ChatNamespace', (), {'completions': _UnsupportedThinkChatCompletions(calls)})()


def test_llm_player_ollama_request_supports_bool_think(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama 风格接口下，think=false 应映射为 bool，而不是旧的 reasoning_effort=none。"""

    calls: list[dict[str, Any]] = []
    fake_client = _FakeOpenAIClient(
        payloads=[
            {
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'message': {'content': '51', 'reasoning_content': None},
                    }
                ]
            }
        ],
        calls=calls,
    )

    player = LLMPlayer(
        model='qwen3.5:9b',
        base_url='http://localhost:11434',
        api_key='ollama',
        think='false',
    )
    monkeypatch.setattr(player, '_get_openai_client', lambda: fake_client)
    text = player._call_llm('只输出数字 51')

    assert text == '51'
    assert len(calls) == 1
    assert calls[0]['stream'] is True
    assert calls[0]['timeout'] == 15.0
    assert calls[0]['extra_body']['think'] is False
    assert 'reasoning_effort' not in calls[0]
    assert 'think' not in calls[0]
    assert 'max_tokens' not in calls[0]
    assert calls[0]['messages'][0]['role'] == 'system'
    assert calls[0]['messages'][1]['role'] == 'user'


def test_llm_player_retries_with_bool_think_when_model_rejects_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama 风格接口若拒绝 think=low，应先自动降级成 think=True。"""

    calls: list[dict[str, Any]] = []
    player = LLMPlayer(model='qwen3.5:9b', base_url='http://localhost:11434/v1', verbose=True, think='low')
    monkeypatch.setattr(player, '_get_openai_client', lambda: _UnsupportedThinkOpenAIClient(calls))

    text = player._call_llm('只输出数字 7')

    assert text == '7'
    assert len(calls) == 2
    assert calls[0]['extra_body']['think'] == 'low'
    assert calls[1]['extra_body']['think'] is True


def test_llm_player_non_ollama_request_uses_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 Ollama 的 OpenAI-compatible 接口仍应继续使用 reasoning_effort。"""

    calls: list[dict[str, Any]] = []
    fake_client = _FakeOpenAIClient(
        payloads=[
            {
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'message': {'content': '13', 'reasoning_content': None},
                    }
                ]
            }
        ],
        calls=calls,
    )

    player = LLMPlayer(
        model='qwen-plus',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        think='medium',
    )
    monkeypatch.setattr(player, '_get_openai_client', lambda: fake_client)

    text = player._call_llm('只输出数字 13')

    assert text == '13'
    assert len(calls) == 1
    assert calls[0]['reasoning_effort'] == 'medium'
    assert 'think' not in calls[0]


def test_llm_player_streaming_assembles_delta_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认应走流式接口，并把多个 delta chunk 拼成最终答案。"""

    calls: list[dict[str, Any]] = []
    fake_client = _FakeOpenAIClient(
        payloads=[
            [
                {'choices': [{'delta': {'reasoning_content': '先比较候选动作。'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'content': '5'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'content': '1'}, 'finish_reason': None}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            ]
        ],
        calls=calls,
    )

    player = LLMPlayer(model='qwen-plus', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
    monkeypatch.setattr(player, '_get_openai_client', lambda: fake_client)

    text = player._call_llm('只输出数字 51')

    assert text == '51'
    assert len(calls) == 1
    assert calls[0]['stream'] is True
    assert calls[0]['timeout'] == 15.0


def test_llm_player_debug_logs_stream_chunks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """debug 模式下，流式响应应按缓冲后的片段打印，而不是一 token 一条。"""

    calls: list[dict[str, Any]] = []
    fake_client = _FakeOpenAIClient(
        payloads=[
            [
                {'choices': [{'delta': {'reasoning_content': '先'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'reasoning_content': '看'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'reasoning_content': '可用'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'reasoning_content': '动作。'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'content': '5'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'content': '1'}, 'finish_reason': None}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            ]
        ],
        calls=calls,
    )

    player = LLMPlayer(model='qwen-plus', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', debug=True)
    monkeypatch.setattr(player, '_get_openai_client', lambda: fake_client)

    text = player._call_llm('只输出数字 51', debug_context='stream')

    captured = capsys.readouterr()
    assert text == '51'
    assert captured.err.count('MODEL THINKING CHUNK >>>') == 1
    assert captured.err.count('<<< END MODEL THINKING CHUNK') == 1
    assert '先看可用动作。' in captured.err
    assert captured.err.count('MODEL OUTPUT CHUNK >>>') == 1
    assert captured.err.count('<<< END MODEL OUTPUT CHUNK') == 1
    assert 'MODEL OUTPUT CHUNK >>>\n51\n' in captured.err or 'MODEL OUTPUT CHUNK >>>\r\n51\r\n' in captured.err


def test_llm_player_parses_only_final_output_not_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """动作解析只能看 final output，debug 可展示 reasoning，但不能把它当结果。"""

    calls: list[dict[str, Any]] = []
    fake_client = _FakeOpenAIClient(
        payloads=[
            {
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'message': {
                            'content': '在 0 和 51 中，我最终选择 51',
                            'reasoning': '先比较 0 和 51，51 更优。',
                        },
                    }
                ]
            }
        ],
        calls=calls,
    )

    player = LLMPlayer(model='gpt-oss:20b', base_url='http://localhost:11434', debug=True)
    monkeypatch.setattr(player, '_get_openai_client', lambda: fake_client)

    chosen = player._parse_action_from_final_output(player._call_llm('只输出数字 51', debug_context='test'), [0, 51], debug_context='test')

    assert chosen == 51


def test_llm_player_strips_think_markup_from_final_output() -> None:
    """即便服务端把 think 标签塞进 content，动作解析也应忽略它。"""

    player = LLMPlayer(model='gpt-oss:20b', base_url='http://localhost:11434')

    chosen = player._parse_action_from_final_output(
        '<think>先在 0 和 51 中分析，0 也可行。</think>\n51',
        [0, 51],
    )

    assert chosen == 51


def test_llm_player_retries_with_larger_budget_after_length_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认不设 max_tokens；若服务端仍因 length 截断，应显式扩容并沿用新预算。"""

    calls: list[dict[str, Any]] = []
    fake_client = _FakeOpenAIClient(
        payloads=[
            {
                'choices': [
                    {
                        'finish_reason': 'length',
                        'message': {'content': '', 'reasoning_content': 'reasoning only'},
                    }
                ]
            },
            {
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'message': {'content': '51', 'reasoning_content': 'done'},
                    }
                ]
            },
            {
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'message': {'content': '48', 'reasoning_content': 'done again'},
                    }
                ]
            },
        ],
        calls=calls,
    )

    player = LLMPlayer(model='gpt-oss:20b', base_url='http://localhost:11434')
    monkeypatch.setattr(player, '_get_openai_client', lambda: fake_client)
    text = player._call_llm('只输出数字 51')
    text_again = player._call_llm('只输出数字 48')

    assert text == '51'
    assert text_again == '48'
    assert len(calls) == 3
    assert 'max_tokens' not in calls[0]
    assert calls[1]['max_tokens'] == 2048
    assert calls[2]['max_tokens'] == 2048


def test_parse_args_defaults_think_to_true_and_accepts_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 只保留一个 --think，默认 true，也允许显式传 false/medium 等值。"""

    monkeypatch.setattr(sys, 'argv', ['llm_player'])
    args = parse_args()
    assert args.think == 'true'
    assert args.stream is True

    monkeypatch.setattr(sys, 'argv', ['llm_player', '--think', 'false', '--no-stream'])
    args = parse_args()
    assert args.think == 'false'
    assert args.stream is False

    monkeypatch.setattr(sys, 'argv', ['llm_player', '--think', 'medium'])
    args = parse_args()
    assert args.think == 'medium'


def test_parse_args_keeps_deprecated_think_aliases_hidden_but_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧脚本若还传 --no-think 或 --reasoning-effort，行为应兼容到新 think 语义。"""

    monkeypatch.setattr(sys, 'argv', ['llm_player', '--no-think'])
    args = parse_args()
    assert args.think == 'false'

    monkeypatch.setattr(sys, 'argv', ['llm_player', '--reasoning-effort', 'high'])
    args = parse_args()
    assert args.think == 'high'
    assert args.stream is True


def test_llm_player_env_config_defaults_to_exam(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 CLI 仍应保持 exam 语义，避免回退现有考试链路。"""

    monkeypatch.setattr(sys, 'argv', ['llm_player', '--stage-type', 'ProduceStepType_AuditionMid1'])
    args = parse_args()
    env_config = build_llm_env_config(args)

    assert env_config['mode'] == 'exam'
    assert env_config['stage_type'] == 'ProduceStepType_AuditionMid1'
    assert env_config['exam_reward_mode'] == 'score'
    assert env_config['lesson_action_type'] == ''
    assert env_config['lesson_level_index'] == 0


def test_llm_player_env_config_supports_lesson_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 对弈器应支持独立 lesson 训练数据生成。"""

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'llm_player',
            '--mode',
            'lesson',
            '--scenario',
            'nia_master',
            '--lesson-action-type',
            'lesson_vocal_normal',
            '--lesson-level-index',
            '2',
            '--include-deck-features',
        ],
    )
    args = parse_args()
    env_config = build_llm_env_config(args)

    assert env_config['mode'] == 'lesson'
    assert env_config['scenario'] == 'nia_master'
    assert env_config['stage_type'] is None
    assert env_config['lesson_action_type'] == 'lesson_vocal_normal'
    assert env_config['lesson_level_index'] == 2
    assert env_config['include_deck_features'] is True


def test_bc_pretrain_env_config_supports_lesson_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """BC 预训练应能用 lesson 环境推断观测维度。"""

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'bc_pretrain',
            '--data',
            'trajectories/demo.jsonl',
            '--mode',
            'lesson',
            '--scenario',
            'nia_master',
            '--lesson-action-type',
            'lesson_vocal_sp',
            '--lesson-level-index',
            '3',
            '--idol-card-id',
            'i_card-amao-3-000',
            '--include-deck-features',
        ],
    )
    args = parse_bc_args()
    env_config = build_bc_env_config(args)

    assert env_config['mode'] == 'lesson'
    assert env_config['scenario'] == 'nia_master'
    assert env_config['lesson_action_type'] == 'lesson_vocal_sp'
    assert env_config['lesson_level_index'] == 3
    assert env_config['idol_card_id'] == 'i_card-amao-3-000'
    assert env_config['include_deck_features'] is True


def test_bc_pretrain_parse_args_accepts_multiple_data_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """BC 预训练应支持重复传入 --data 合并多个 jsonl 文件。"""

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'bc_pretrain',
            '--data',
            'trajectories/a.jsonl',
            '--data',
            'trajectories/b.jsonl',
        ],
    )
    args = parse_bc_args()

    assert args.data == ['trajectories/a.jsonl', 'trajectories/b.jsonl']


def test_bc_trainer_load_trajectories_remaps_episode_ids_across_files(tmp_path) -> None:
    """多文件合并时，重复的 episode_id 不应被错误拼成同一局。"""

    file_a = tmp_path / 'a.jsonl'
    file_b = tmp_path / 'b.jsonl'
    file_a.write_text(
        '\n'.join(
            [
                '{"episode_id": 0, "step": 0, "obs": {"global": [], "action_features": [], "action_mask": []}, "action": 0, "reward": 1.0}',
                '{"episode_id": 0, "step": 1, "obs": {"global": [], "action_features": [], "action_mask": []}, "action": 1, "reward": 2.0}',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    file_b.write_text(
        '\n'.join(
            [
                '{"episode_id": 0, "step": 0, "obs": {"global": [], "action_features": [], "action_mask": []}, "action": 2, "reward": 3.0}',
                '{"episode_id": 0, "step": 1, "obs": {"global": [], "action_features": [], "action_mask": []}, "action": 3, "reward": 4.0}',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    trainer = BCTrainer(global_dim=1, action_dim=1, max_actions=1)
    records = trainer.load_trajectories([file_a, file_b])

    assert [record['episode_id'] for record in records] == [0, 0, 1, 1]
    assert [record['action'] for record in records] == [0, 1, 2, 3]


def test_run_episodes_returns_partial_summary_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """单线程生成轨迹时按 Ctrl+C 中断，应返回部分 summary，而不是卡住或丢异常。"""

    player = LLMPlayer(model='qwen3:4b', base_url='http://localhost:11434')
    output_path = tmp_path / 'partial.jsonl'
    calls = {'count': 0}

    def _fake_run_single_episode(env_config: dict, episode_id: int, seed: int | None = None, stop_event=None):
        calls['count'] += 1
        if calls['count'] == 1:
            return {
                'episode_id': episode_id,
                'trajectory': [
                    {
                        'episode_id': episode_id,
                        'step': 0,
                        'obs': {'global': [], 'action_features': [], 'action_mask': []},
                        'action': 0,
                        'reward': 1.25,
                        'terminated': True,
                        'info': {'score': 12.0, 'turn': 1, 'action_label': 'A'},
                    }
                ],
                'elapsed': 0.1,
                'steps': 1,
                'reward': 1.25,
                'score': 12.0,
            }
        raise KeyboardInterrupt()

    monkeypatch.setattr(player, '_run_single_episode', _fake_run_single_episode)

    summary = player.run_episodes(
        env_config={'mode': 'exam', 'scenario': 'nia_master'},
        n=3,
        output_path=output_path,
        workers=1,
    )

    lines = output_path.read_text(encoding='utf-8').strip().splitlines()
    assert summary['interrupted'] is True
    assert summary['episodes'] == 1
    assert summary['requested_episodes'] == 3
    assert summary['mean_reward'] == pytest.approx(1.25)
    assert len(lines) == 1


def test_prompts_explain_drink_timing_and_auto_end_turn() -> None:
    """提示词应明确：饮料不占用次数，但只能在剩余出牌次数内使用。"""

    system_prompt = load_system_prompt()
    action_prompt = render(
        'action_select.jinja2',
        snapshot='本回合剩余技能卡出牌次数: 0/1',
        actions=[{'index': 48, 'kind': 'drink', 'label': '初星汤'}],
    )

    assert 'P饮料通常不消耗技能卡出牌次数' in system_prompt
    assert '如果出牌次数耗尽，本回合会直接结束，不能再喝饮料' in system_prompt
    assert '只能在本回合仍有剩余出牌次数时使用' in action_prompt
    assert '一次回答只选择“当前这一步”' in system_prompt
    assert '本题也只要求选择“当前第一步”' in action_prompt
    assert '系统会把新局面重新发给你' in system_prompt
    assert '合法动作列表是“当前这一步”的单步动作列表' in action_prompt


def test_prompts_include_official_runtime_constraints() -> None:
    """提示词应同步官方文档约束，并明确当前结算优先于脑补隐藏常数。"""

    system_prompt = load_system_prompt()
    action_prompt = render(
        'action_select.jinja2',
        snapshot='模式: 课程 | 课程状态: 已达成目标\n目标剩余: 0 | Perfect剩余: 5',
        actions=[{'index': 7, 'kind': 'card', 'label': '集中训练'}],
    )

    assert '考场机制(gimmick)的条件判定与效果结算先于 P道具、附魔和技能卡' in system_prompt
    assert '状态变动' in system_prompt
    assert '同一个P道具在同一效果触发 timing 下只会触发一次' in system_prompt
    assert '达到 Perfect 会立刻结束，并按剩余回合回复体力' in system_prompt
    assert '优先相信局面中的当前结算' in system_prompt
    assert '好調 / 絶好調 / 集中 / 好印象 / やる気 / 元気' in system_prompt
    assert 'やる気: 每增加1，会提高元気的增加量' in system_prompt
    assert '先看考场机制(gimmick)会不会在当前 timing 先结算' in action_prompt
    assert '如果是 lesson 模式，要同时看“目标剩余”和“Perfect剩余”' in action_prompt
    assert '不要再自己猜隐藏倍率或常数' in action_prompt


def test_llm_player_verbose_fallback_logs_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """回退日志不能只剩一句 timeout，至少要包含异常类型和原因链。"""

    player = LLMPlayer(
        model='qwen-plus',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        timeout=60,
        verbose=True,
        debug=True,
    )
    monkeypatch.setattr(player, '_build_action_prompt', lambda env, last_action_label='': '只输出数字 1')

    def _raise_timeout(prompt: str, debug_context: str = '') -> str:
        try:
            raise TimeoutError('socket connect timed out')
        except TimeoutError as inner:
            raise RuntimeError('Request timed out.') from inner

    monkeypatch.setattr(player, '_call_llm', _raise_timeout)

    action = player._choose_action(
        env=object(),
        obs={'action_mask': np.array([0.0, 1.0, 1.0], dtype=np.float32)},
        debug_context='ep=0 step=0',
    )

    captured = capsys.readouterr()
    assert action == 1
    assert 'LLM ERROR >>>' in captured.err
    assert 'request: model=qwen-plus base_url=https://dashscope.aliyuncs.com/compatible-mode/v1 timeout=60s think=true' in captured.err
    assert 'error: RuntimeError: Request timed out.' in captured.err
    assert 'cause[1]: TimeoutError: socket connect timed out' in captured.err
    assert 'falling back to first valid action' in captured.out


def test_llm_player_repairs_non_numeric_action_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型若先输出长篇分析而非编号，应触发一次短重试而不是直接回退首个动作。"""

    calls: list[dict[str, Any]] = []
    fake_client = _FakeOpenAIClient(
        payloads=[
            [
                {'choices': [{'delta': {'content': '**决策逻辑：**\n1. 先分析资源。'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'content': '\n2. 再比较饮料和出牌。'}, 'finish_reason': None}]},
                {'choices': [{'delta': {'content': '\n**结论：** 先喝饮料。'}, 'finish_reason': None}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            ],
            [
                {'choices': [{'delta': {'content': '49'}, 'finish_reason': None}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            ],
        ],
        calls=calls,
    )
    player = LLMPlayer(model='qwen-plus', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
    monkeypatch.setattr(player, '_get_openai_client', lambda: fake_client)
    monkeypatch.setattr(player, '_build_action_prompt', lambda env, last_action_label='': '只输出动作编号')

    chosen = player._choose_action(
        env=object(),
        obs={'action_mask': np.array([0.0] * 49 + [1.0, 1.0], dtype=np.float32)},
        debug_context='ep=0 step=2',
    )

    assert chosen == 49
    assert len(calls) == 2
    assert '只输出动作编号' in calls[1]['messages'][1]['content']
    assert '## 修正要求' in calls[1]['messages'][1]['content']
    assert '合法动作编号只有这些：49, 50' in calls[1]['messages'][1]['content']


def test_llm_player_debug_logs_openai_request_payload_on_request_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """请求失败时，debug 输出应包含 payload 和底层错误链。"""

    player = LLMPlayer(
        model='qwen-plus',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        timeout=60,
        debug=True,
    )
    monkeypatch.setattr(player, '_get_openai_client', lambda: _ExplodingOpenAIClient())

    with pytest.raises(RuntimeError, match='Request timed out.'):
        player._call_llm_openai('只输出数字 51', debug_context='req')

    captured = capsys.readouterr()
    assert 'OPENAI REQUEST PAYLOAD >>>' in captured.err
    assert '"model": "qwen-plus"' in captured.err
    assert '"stream": true' in captured.err
    assert '"reasoning_effort": "low"' in captured.err
    assert 'OPENAI REQUEST ERROR >>>' in captured.err
    assert 'error: RuntimeError: Request timed out.' in captured.err
    assert 'cause[1]: TimeoutError: socket connect timed out' in captured.err
