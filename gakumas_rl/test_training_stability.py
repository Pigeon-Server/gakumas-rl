"""训练稳定性相关回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from gakumas_rl.backends import _build_sb3_learning_rate_schedule
from gakumas_rl.data import MasterDataRepository
from gakumas_rl.envs import GakumasExamEnv


def test_sb3_learning_rate_schedule_supports_linear_decay() -> None:
    """SB3 学习率调度应支持从起始值线性衰减到末值。"""

    schedule = _build_sb3_learning_rate_schedule(1e-4, 3e-5)

    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(1e-4)
    assert schedule(0.5) == pytest.approx(6.5e-5)
    assert schedule(0.0) == pytest.approx(3e-5)


def test_exam_env_numeric_safeguard_converts_infinite_reward() -> None:
    """环境收到非有限 reward 时应立刻截断并返回有限惩罚。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(
        repository,
        scenario,
        seed=103,
        include_deck_features=False,
    )

    obs, _info = env.reset(seed=103)
    action = int(np.flatnonzero(obs['action_mask'] > 0.5)[0])

    def _explode(_runtime_action):
        return float('inf'), {'score': env.runtime.score}

    env.runtime.step = _explode  # type: ignore[method-assign]

    _next_obs, reward, terminated, truncated, info = env.step(action)

    assert reward == pytest.approx(-25.0)
    assert terminated is True
    assert truncated is False
    assert info['numeric_safeguard_triggered'] is True
    assert info['numeric_state_unstable'] is False


def test_exam_env_action_masks_reuse_cached_candidate_mask(monkeypatch) -> None:
    """观测里的 action_mask 应与缓存掩码一致，且重复读取不应重建候选。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(
        repository,
        scenario,
        seed=211,
        include_deck_features=False,
    )

    obs, _info = env.reset(seed=211)
    expected_mask = obs['action_mask'].astype(bool)

    def _should_not_be_called():
        raise AssertionError('action_masks should reuse cached candidates and mask')

    monkeypatch.setattr(env, '_build_candidates', _should_not_be_called)

    np.testing.assert_array_equal(env.action_masks(), expected_mask)


def test_exam_runtime_reward_profile_config_reuses_cached_dict() -> None:
    """reward profile 应直接复用 runtime 上的 RewardConfig 对象。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(
        repository,
        scenario,
        seed=223,
        include_deck_features=False,
    )

    env.reset(seed=223)

    first = env.runtime._reward_profile_config()
    second = env.runtime._reward_profile_config()

    assert first is second
    assert first is env.runtime.reward_config


def test_exam_runtime_step_reuses_cached_reward_signal(monkeypatch) -> None:
    """step 应复用 reset 后缓存的 reward signal，避免重复计算动作前状态。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(
        repository,
        scenario,
        seed=227,
        include_deck_features=False,
    )

    obs, _info = env.reset(seed=227)
    action = int(np.flatnonzero(obs['action_mask'] > 0.5)[0])
    runtime = env.runtime
    original_reward_signal = runtime._reward_signal
    call_count = 0

    def _tracked_reward_signal():
        nonlocal call_count
        call_count += 1
        return original_reward_signal()

    monkeypatch.setattr(runtime, '_reward_signal', _tracked_reward_signal)

    env.step(action)

    assert call_count == 1
