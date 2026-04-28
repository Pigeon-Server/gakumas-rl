"""demo_exam 中与 checkpoint 恢复相关的轻量回归测试。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from gakumas_rl.demo_exam import _build_env_config, _resolve_rllib_api_stack, _resolve_rllib_custom_model_config


def test_resolve_rllib_custom_model_config_prefers_training_metadata() -> None:
    """若 run 目录已有训练元数据，应优先复用其中记录的模型配置。"""

    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        checkpoint_path = run_dir / 'checkpoints' / 'checkpoint_00001024'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (run_dir / 'training_metadata.json').write_text(
            json.dumps(
                {
                    'config': {
                        'custom_model_config': {
                            'hidden_dim': 384,
                        }
                    }
                }
            ),
            encoding='utf-8',
        )

        assert _resolve_rllib_custom_model_config(checkpoint_path) == {'hidden_dim': 384}


def test_resolve_rllib_custom_model_config_falls_back_to_checkpoint_state() -> None:
    """若缺少训练元数据，应从 RLlib checkpoint state 中推回模型配置。"""

    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        checkpoint_path = run_dir / 'checkpoints' / 'checkpoint_00001024'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / 'algorithm_state.pkl').write_bytes(
            pickle.dumps(
                {
                    'config': {
                        'model': {
                            'custom_model_config': {
                                'hidden_dim': 320,
                            }
                        }
                    }
                }
            )
        )

        assert _resolve_rllib_custom_model_config(checkpoint_path) == {'hidden_dim': 320}


def test_resolve_rllib_custom_model_config_falls_back_to_ctor_args() -> None:
    """新 API checkpoint 若尚未写入 training_metadata，也应从构造参数恢复模型配置。"""

    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        checkpoint_path = run_dir / 'checkpoints' / 'checkpoint_00001024'
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        (checkpoint_path / 'class_and_ctor_args.pkl').write_bytes(
            pickle.dumps(
                {
                    'ctor_args_and_kwargs': (
                        (
                            {
                                '_rl_module_spec': SimpleNamespace(model_config={'hidden_dim': 448}),
                            },
                        ),
                        {},
                    )
                }
            )
        )

        assert _resolve_rllib_custom_model_config(checkpoint_path) == {'hidden_dim': 448}


def test_resolve_rllib_api_stack_prefers_training_metadata() -> None:
    """若 run 目录元数据已记录 API stack，应优先采用该值。"""

    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        checkpoint_path = run_dir / 'checkpoints' / 'checkpoint_00001024'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (run_dir / 'training_metadata.json').write_text(
            json.dumps({'config': {'api_stack': 'new'}}),
            encoding='utf-8',
        )

        assert _resolve_rllib_api_stack(checkpoint_path) == 'new'


def test_resolve_rllib_api_stack_falls_back_to_ctor_args() -> None:
    """新 API checkpoint 缺少 algorithm_state.config 时，也应从构造参数识别新栈。"""

    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        checkpoint_path = run_dir / 'checkpoints' / 'checkpoint_00001024'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / 'class_and_ctor_args.pkl').write_bytes(
            pickle.dumps(
                {
                    'ctor_args_and_kwargs': (
                        (
                            {
                                'enable_rl_module_and_learner': True,
                                'enable_env_runner_and_connector_v2': True,
                            },
                        ),
                        {},
                    )
                }
            )
        )

        assert _resolve_rllib_api_stack(checkpoint_path) == 'new'


def test_resolve_rllib_api_stack_falls_back_to_checkpoint_state() -> None:
    """缺少元数据时，应从 algorithm_state 中推断新旧 API stack。"""

    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        checkpoint_path = run_dir / 'checkpoints' / 'checkpoint_00001024'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / 'algorithm_state.pkl').write_bytes(
            pickle.dumps(
                {
                    'config': {
                        'enable_rl_module_and_learner': True,
                        'enable_env_runner_and_connector_v2': True,
                    }
                }
            )
        )

        assert _resolve_rllib_api_stack(checkpoint_path) == 'new'


def test_build_env_config_includes_manual_support_cards() -> None:
    """demo_exam 的 CLI 环境配置应把手动支援编成参数透传给环境层。"""

    args = SimpleNamespace(
        mode='exam',
        scenario='nia_master',
        stage_type='ProduceStepType_AuditionFinal',
        idol_card_id='i_card-amao-1-000',
        producer_level=35,
        idol_rank=4,
        dearness_level=10,
        use_after_item=True,
        exam_score_bonus_multiplier=1.2,
        support_card_id=['s_card-1-0000', 's_card-1-0001', 's_card-1-0002', 's_card-1-0003', 's_card-1-0004', 's_card-1-0005'],
        support_card_level=40,
        fan_votes=9000.0,
        exam_randomize_context=False,
        exam_randomize_use_after_item=False,
        exam_randomize_stage_type=False,
        exam_reward_mode='score',
        lesson_action_type='',
        lesson_level_index=0,
        manual_exam_setup=[],
        guarantee_card_effect=[],
        force_card=[],
    )

    config = _build_env_config(args)

    assert config['support_card_ids'] == ['s_card-1-0000', 's_card-1-0001', 's_card-1-0002', 's_card-1-0003', 's_card-1-0004', 's_card-1-0005']
    assert config['support_card_level'] == 40
