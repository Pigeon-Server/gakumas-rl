"""无 LLM 自举训练流程的轻量回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.training.bc_pretrain import BCTrainer
from src.training.cli import parse_args as parse_train_args
from src.training.competitive_metrics import competitive_outcome_from_info
from src.training.self_bootstrap import (
    EpisodeCandidate,
    _build_env_config,
    _clear_rank_from_info,
    _resolve_final_rl_timesteps,
    discover_sb3_checkpoints,
    parse_args as parse_bootstrap_args,
)


def test_bc_pretrain_rejects_target_action_outside_mask() -> None:
    """BC 数据集构建时应拒绝目标动作不合法的样本。"""

    trainer = BCTrainer(global_dim=2, action_dim=3, max_actions=2)
    trajectories = [
        {
            'episode_id': 0,
            'step': 0,
            'obs': {
                'global': [0.0, 0.0],
                'action_features': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                'action_mask': [1.0, 0.0],
            },
            'action': 1,
            'reward': 0.0,
        }
    ]

    with pytest.raises(ValueError, match='outside action_mask'):
        trainer.build_dataset(trajectories)


def test_bc_pretrain_history_contains_masked_accuracy() -> None:
    """BC 训练历史应暴露 masked accuracy，便于判断自举数据是否学进去。"""

    trainer = BCTrainer(global_dim=2, action_dim=3, max_actions=2)
    trajectories = [
        {
            'episode_id': 0,
            'step': 0,
            'obs': {
                'global': [0.0, 0.0],
                'action_features': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                'action_mask': [1.0, 1.0],
            },
            'action': 0,
            'reward': 1.0,
        }
    ]
    dataset = trainer.build_dataset(trajectories)

    history = trainer.train(dataset, epochs=1, batch_size=1, learning_rate=1e-3)

    assert len(history) == 1
    assert 'masked_accuracy' in history[0]
    assert 0.0 <= history[0]['masked_accuracy'] <= 1.0


def test_bc_pretrain_can_oversample_success_episodes() -> None:
    """成功轨迹应支持重复采样放大，避免被大量失败样本淹没。"""

    trainer = BCTrainer(global_dim=2, action_dim=3, max_actions=2)
    trajectories = [
        {
            'episode_id': 0,
            'step': 0,
            'obs': {
                'global': [0.0, 0.0],
                'action_features': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                'action_mask': [1.0, 1.0],
            },
            'action': 0,
            'reward': 1.0,
            'info': {'final_summary': {'route': 'nia', 'route_clear': True, 'competitive_pass': True, 'competitive_top1': True, 'final_rank': 1, 'all_auditions_first': True}},
        },
        {
            'episode_id': 1,
            'step': 0,
            'obs': {
                'global': [0.0, 0.0],
                'action_features': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                'action_mask': [1.0, 1.0],
            },
            'action': 1,
            'reward': 0.0,
            'info': {'final_summary': {'route': 'nia', 'route_clear': True, 'competitive_pass': False, 'competitive_top1': False, 'final_rank': 2, 'all_auditions_first': False}},
        },
    ]

    base_dataset = trainer.build_dataset(trajectories, success_oversample_factor=1)
    boosted_dataset = trainer.build_dataset(trajectories, success_oversample_factor=4)

    assert len(base_dataset) == 2
    assert len(boosted_dataset) == 5


def test_bc_pretrain_can_filter_to_success_episodes_only() -> None:
    """只保留成功轨迹时，应过滤掉未过线的 episode。"""

    trainer = BCTrainer(global_dim=2, action_dim=3, max_actions=2)
    trajectories = [
        {
            'episode_id': 0,
            'step': 0,
            'obs': {
                'global': [0.0, 0.0],
                'action_features': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                'action_mask': [1.0, 1.0],
            },
            'action': 0,
            'reward': 1.0,
            'info': {'final_summary': {'route': 'first_star', 'route_clear': True, 'competitive_pass': True, 'competitive_top1': False, 'final_rank': 3, 'all_auditions_first': False}},
        },
        {
            'episode_id': 1,
            'step': 0,
            'obs': {
                'global': [0.0, 0.0],
                'action_features': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                'action_mask': [1.0, 1.0],
            },
            'action': 1,
            'reward': 0.0,
            'info': {'final_summary': {'route': 'first_star', 'route_clear': True, 'competitive_pass': False, 'competitive_top1': False, 'final_rank': 4, 'all_auditions_first': False}},
        },
    ]

    success_only_dataset = trainer.build_dataset(trajectories, success_only=True)

    assert len(success_only_dataset) == 1


def test_episode_candidate_prefers_valid_clear_high_score() -> None:
    """轨迹选优应优先选择无非法动作、clear 等级更高、分数更高的候选。"""

    weak = EpisodeCandidate(
        checkpoint_path=Path('weak.zip'),
        seed=1,
        episode_id=0,
        records=[{}],
        total_reward=10.0,
        terminal_score=100.0,
        invalid_actions=0,
        clear_rank=0,
        top1=False,
        route='first_star',
        final_rank=4,
        all_auditions_first=False,
        steps=10,
    )
    invalid_but_high_score = EpisodeCandidate(
        checkpoint_path=Path('invalid.zip'),
        seed=1,
        episode_id=0,
        records=[{}],
        total_reward=20.0,
        terminal_score=200.0,
        invalid_actions=1,
        clear_rank=2,
        top1=True,
        route='nia',
        final_rank=1,
        all_auditions_first=True,
        steps=10,
    )
    clear = EpisodeCandidate(
        checkpoint_path=Path('clear.zip'),
        seed=1,
        episode_id=0,
        records=[{}],
        total_reward=9.0,
        terminal_score=120.0,
        invalid_actions=0,
        clear_rank=1,
        top1=False,
        route='first_star',
        final_rank=3,
        all_auditions_first=False,
        steps=12,
    )

    assert max([weak, invalid_but_high_score, clear], key=lambda item: item.quality_key()) is clear


def test_discover_sb3_checkpoints_supports_run_directory(tmp_path: Path) -> None:
    """全自动 SB3 自举应能从 run_dir/checkpoints 自动发现 zip checkpoint。"""

    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    (tmp_path / 'best_model.zip').write_bytes(b'best')
    (checkpoint_dir / 'step_10.zip').write_bytes(b'x')
    (checkpoint_dir / 'step_20.zip').write_bytes(b'y')

    checkpoints = discover_sb3_checkpoints([tmp_path])

    assert [path.name for path in checkpoints] == ['best_model.zip', 'step_10.zip', 'step_20.zip']


def test_clear_rank_recognizes_lesson_cleared_flag() -> None:
    """lesson 环境的 lesson_cleared 字段也应进入轨迹选优。"""

    assert _clear_rank_from_info({'lesson_cleared': True}) == 1


def test_clear_rank_uses_competitive_rules_for_first_star_and_nia() -> None:
    """竞技模式下的成功层级应按初/NIA 的新口径解释。"""

    assert _clear_rank_from_info(
        {
            'final_summary': {
                'route': 'first_star',
                'route_clear': True,
                'competitive_pass': True,
                'competitive_top1': False,
                'final_rank': 3,
                'all_auditions_first': False,
            }
        }
    ) == 1
    assert _clear_rank_from_info(
        {
            'final_summary': {
                'route': 'first_star',
                'route_clear': True,
                'competitive_pass': False,
                'competitive_top1': False,
                'final_rank': 4,
                'all_auditions_first': False,
            }
        }
    ) == 0
    assert _clear_rank_from_info(
        {
            'final_summary': {
                'route': 'nia',
                'route_clear': True,
                'competitive_pass': True,
                'competitive_top1': True,
                'final_rank': 1,
                'all_auditions_first': True,
            }
        }
    ) == 2
    assert _clear_rank_from_info(
        {
            'final_summary': {
                'route': 'nia',
                'route_clear': True,
                'competitive_pass': False,
                'competitive_top1': False,
                'final_rank': 1,
                'all_auditions_first': False,
            }
        }
    ) == 0


def test_competitive_outcome_marks_nia_non_first_as_failure() -> None:
    """NIA 只要不是每场第一，就算最终名次第一也应视作失败。"""

    outcome = competitive_outcome_from_info(
        {
            'final_summary': {
                'route': 'nia',
                'route_clear': True,
                'competitive_pass': False,
                'competitive_top1': False,
                'final_rank': 1,
                'all_auditions_first': False,
            }
        }
    )

    assert outcome.passed is False
    assert outcome.top1 is False
    assert outcome.tier == 0


def test_train_cli_rejects_removed_llm_reward_flag() -> None:
    """训练 CLI 不应再接受 LLM reward shaping 参数。"""

    with patch.object(sys, 'argv', ['train', '--llm-reward']), pytest.raises(SystemExit):
        parse_train_args()


def test_train_cli_rejects_removed_torch_backend() -> None:
    """训练 CLI 不应再接受已删除的 Torch 后端。"""

    with patch.object(sys, 'argv', ['train', '--backend', 'torch']), pytest.raises(SystemExit):
        parse_train_args()


def test_bootstrap_cli_rejects_removed_torch_backend() -> None:
    """自举 CLI 不应再接受已删除的 Torch 后端。"""

    with patch.object(sys, 'argv', ['bootstrap', '--backend', 'torch']), pytest.raises(SystemExit):
        parse_bootstrap_args()


def test_bootstrap_final_rl_zero_is_respected() -> None:
    """显式传入 final_rl_timesteps=0 时，autopilot 不应再强行追加微调。"""

    with patch.object(sys, 'argv', ['bootstrap', '--autopilot', '--final-rl-timesteps', '0']):
        args = parse_bootstrap_args()

    assert _resolve_final_rl_timesteps(args) == 0


def test_bootstrap_final_rl_default_uses_autopilot_timesteps() -> None:
    """未显式配置时，autopilot 仍保留自动追加最终 RL 的默认行为。"""

    with patch.object(sys, 'argv', ['bootstrap', '--autopilot', '--rl-timesteps', '1234']):
        args = parse_bootstrap_args()

    assert _resolve_final_rl_timesteps(args) == 1234


def test_bootstrap_cli_collects_produce_reward_config() -> None:
    """自举 CLI 应支持透传培育奖励配置文件和覆盖项。"""

    with patch.object(
        sys,
        'argv',
        [
            'bootstrap',
            '--produce-reward-config',
            'configs/test.json',
            '--produce-reward-route-clear-bonus',
            '3.5',
            '--produce-reward-pp-left-waste-penalty',
            '0.8',
        ],
    ):
        args = parse_bootstrap_args()

    assert args.produce_reward_config == 'configs/test.json'
    assert args.produce_reward_route_clear_bonus == pytest.approx(3.5)
    assert args.produce_reward_pp_left_waste_penalty == pytest.approx(0.8)


def test_bootstrap_env_config_collects_force_lowest_audition_route() -> None:
    """自举 CLI 应把 force-lowest-audition-route 透传到环境配置。"""

    with patch.object(sys, 'argv', ['bootstrap', '--force-lowest-audition-route']):
        args = parse_bootstrap_args()
    env_config = _build_env_config(args)

    assert env_config['force_lowest_audition_route'] is True


def test_bootstrap_cli_collects_bc_success_only() -> None:
    """自举 CLI 应支持显式开启 success-only BC 蒸馏。"""

    with patch.object(sys, 'argv', ['bootstrap', '--bc-success-only']):
        args = parse_bootstrap_args()

    assert args.bc_success_only is True
