"""训练稳定性相关回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from src.repository.master_data import MasterDataRepository
from src.simulation.envs import GakumasExamEnv
from src.training.device import resolve_torch_device
from src.training.backends import TrainingSpec, _build_sb3_learning_rate_schedule, run_training
from src.interfaces.service import build_env_from_config


def test_sb3_learning_rate_schedule_supports_linear_decay() -> None:
    """SB3 学习率调度应支持从起始值线性衰减到末值。"""

    schedule = _build_sb3_learning_rate_schedule(1e-4, 3e-5)

    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(1e-4)
    assert schedule(0.5) == pytest.approx(6.5e-5)
    assert schedule(0.0) == pytest.approx(3e-5)


def test_auto_device_resolves_to_supported_torch_device() -> None:
    """默认 auto 设备应解析为 SB3/PyTorch 可直接接收的明确设备名。"""

    assert resolve_torch_device('auto') in {'cuda', 'mps', 'cpu'}


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

    assert reward == pytest.approx(-float(env.reward_config.reward_clip))
    assert terminated is True
    assert truncated is False
    assert info['numeric_safeguard_triggered'] is True
    assert info['numeric_state_unstable'] is False


def test_exam_env_action_masks_reuse_cached_candidate_mask() -> None:
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

    with patch.object(env, '_build_candidates', _should_not_be_called):
        np.testing.assert_array_equal(env.action_masks(), expected_mask)


def test_planning_env_keeps_fixed_action_slots_across_episode() -> None:
    """完整育成环境在特殊阶段也必须保持固定动作槽，供 MaskablePPO 使用。"""

    env = build_env_from_config({'mode': 'planning', 'scenario': 'first_star_regular'})
    obs, _info = env.reset(seed=211)
    expected_actions = int(env.action_space.n)

    for _ in range(64):
        assert obs['action_mask'].shape == (expected_actions,)
        assert obs['action_features'].shape[0] == expected_actions
        mask = obs['action_mask'].astype(bool)
        np.testing.assert_array_equal(env.action_masks(), mask)
        valid_actions = np.flatnonzero(mask)
        assert valid_actions.size > 0
        obs, _reward, terminated, truncated, _info = env.step(int(valid_actions[0]))
        if terminated or truncated:
            break
    else:
        pytest.fail('planning episode did not finish within the expected smoke-test horizon')

    env.close()


def test_planning_env_fixed_loadout_keeps_expanded_action_slots() -> None:
    """固定偶像上下文产生额外候选时，planning env 仍应保留固定动作槽。"""

    env = build_env_from_config(
        {
            'mode': 'planning',
            'scenario': 'nia_pro',
            'idol_card_id': 'i_card-amao-1-000',
            'idol_rank': 4,
            'dearness_level': 20,
            'seed': 1005,
        }
    )
    obs, _info = env.reset(seed=1005)
    expected_actions = int(env.action_space.n)

    for _step in range(40):
        assert obs['action_mask'].shape == (expected_actions,)
        assert obs['action_features'].shape[0] == expected_actions
        assert len(env.runtime.legal_actions()) <= expected_actions
        valid_actions = np.flatnonzero(obs['action_mask'] > 0.5)
        assert valid_actions.size > 0
        obs, _reward, terminated, truncated, _info = env.step(int(valid_actions[-1]))
        if terminated or truncated:
            break

    env.close()


def test_sb3_planning_training_consumes_fixed_action_masks(tmp_path) -> None:
    """SB3 planning 小步训练应能真实进入 rollout，不再出现 mask 长度退化为 1。"""

    result = run_training(
        TrainingSpec(
            backend='sb3',
            env_config={'mode': 'planning', 'scenario': 'first_star_regular', 'seed': 42},
            total_timesteps=512,
            checkpoint_freq=0,
            eval_freq=0,
            eval_episodes=1,
            rollout_steps=128,
            learning_rate=1e-4,
            device='cpu',
            run_dir=tmp_path,
            seed=42,
        )
    )

    assert result.latest_checkpoint is not None
    assert result.latest_checkpoint.exists()


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


def test_exam_runtime_step_reuses_cached_reward_signal() -> None:
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

    with patch.object(runtime, '_reward_signal', _tracked_reward_signal):
        env.step(action)

    assert call_count == 1


def test_exam_env_reports_competitive_outcome_for_exam_terminal_state() -> None:
    """exam-only 环境在终局时应显式给出竞技名次字段。"""

    env = build_env_from_config({'mode': 'exam', 'scenario': 'nia_pro', 'stage_type': 'ProduceStepType_AuditionFinal', 'seed': 307})
    obs, _info = env.reset(seed=307)

    for _ in range(128):
        action = int(np.flatnonzero(obs['action_mask'] > 0.5)[-1])
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert 'competitive_rank' in info
            assert 'competitive_pass' in info
            assert 'competitive_top1' in info
            assert isinstance(info['rival_scores'], list)
            break
    else:
        pytest.fail('exam episode did not terminate within the smoke-test horizon')

    env.close()
