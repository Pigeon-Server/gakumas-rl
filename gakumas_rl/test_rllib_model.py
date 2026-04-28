"""RLlib 新 API stack 自定义 RLModule 的轻量回归测试。"""

from __future__ import annotations

import gymnasium as gym
import torch

from ray.rllib.core.columns import Columns

from gakumas_rl.rllib_model import build_rllib_module_spec


def test_rllib_module_spec_builds_action_masking_module_and_masks_logits() -> None:
    """自定义 RLModule 应能处理现有观测结构，并对非法动作施加掩码。"""

    obs_space = gym.spaces.Dict(
        {
            'global': gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=float),
            'action_features': gym.spaces.Box(-1.0, 1.0, shape=(4, 5), dtype=float),
            'action_mask': gym.spaces.Box(0.0, 1.0, shape=(4,), dtype=float),
        }
    )
    action_space = gym.spaces.Discrete(4)
    spec = build_rllib_module_spec({'hidden_dim': 32})
    spec.observation_space = obs_space
    spec.action_space = action_space
    module = spec.build()

    batch = {
        Columns.OBS: {
            'global': torch.zeros((2, 6), dtype=torch.float32),
            'action_features': torch.randn((2, 4, 5), dtype=torch.float32),
            'action_mask': torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        }
    }

    inference_out = module.forward_inference(batch)
    train_out = module.forward_train(batch)
    values = module.compute_values(batch, embeddings=train_out[Columns.EMBEDDINGS])
    deterministic_actions = (
        module.get_inference_action_dist_cls()
        .from_logits(inference_out[Columns.ACTION_DIST_INPUTS])
        .to_deterministic()
        .sample()
    )

    assert inference_out[Columns.ACTION_DIST_INPUTS].shape == (2, 4)
    assert train_out[Columns.EMBEDDINGS].shape == (2, 64)
    assert values.shape == (2,)
    assert inference_out[Columns.ACTION_DIST_INPUTS][0, 1].item() < -1.0e8
    assert inference_out[Columns.ACTION_DIST_INPUTS][0, 3].item() < -1.0e8
    assert deterministic_actions.shape == (2,)
