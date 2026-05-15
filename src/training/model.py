"""动作掩码策略网络与观测张量化工具。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MaskedPolicyValueNet(nn.Module):
    """共享全局编码器的策略/价值双头网络。

    相比纯 MLP，这里把动作特征拆成：
    - 前半段稀疏 one-hot / multi-hot 结构
    - 末尾 41 维稠密数值
    再分别编码后融合，减少大规模稀疏拼接直接过线性层带来的表达浪费。
    """

    ACTION_DENSE_TAIL_DIM = 41

    def __init__(self, global_dim: int, action_dim: int, hidden_dim: int = 128):
        """按 `global + action_features + action_mask` 结构初始化网络。"""

        super().__init__()
        self.use_split_action_encoder = int(action_dim) > self.ACTION_DENSE_TAIL_DIM
        sparse_action_dim = max(int(action_dim) - self.ACTION_DENSE_TAIL_DIM, 1) if self.use_split_action_encoder else int(action_dim)
        dense_action_dim = min(self.ACTION_DENSE_TAIL_DIM, int(action_dim)) if self.use_split_action_encoder else 0
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.action_sparse_encoder = nn.Sequential(
            nn.Linear(sparse_action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        if self.use_split_action_encoder:
            self.action_dense_encoder = nn.Sequential(
                nn.Linear(dense_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.action_fusion = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.action_dense_encoder = None
            self.action_fusion = nn.Identity()
        self.logit_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_action_features(self, action_features: torch.Tensor) -> torch.Tensor:
        """把候选动作特征编码成稠密动作表示。"""

        if self.use_split_action_encoder:
            sparse_features = action_features[..., :-self.ACTION_DENSE_TAIL_DIM]
            dense_features = action_features[..., -self.ACTION_DENSE_TAIL_DIM:]
            sparse_hidden = self.action_sparse_encoder(sparse_features)
            dense_hidden = self.action_dense_encoder(dense_features)
            return self.action_fusion(torch.cat([sparse_hidden, dense_hidden], dim=-1))
        return self.action_sparse_encoder(action_features)

    def forward(
        self,
        global_obs: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """输出已应用动作掩码的 logits，以及共享状态价值估计。"""

        global_hidden = self.global_encoder(global_obs)
        action_hidden = self.encode_action_features(action_features)
        expanded_global = global_hidden.unsqueeze(1).expand(-1, action_hidden.size(1), -1)
        logits = self.logit_head(torch.cat([expanded_global, action_hidden], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(action_mask <= 0, -1.0e9)

        # masked mean pooling: 对有效动作的 action_hidden 取平均
        mask_expanded = action_mask.unsqueeze(-1)  # (B, A, 1)
        masked_sum = (action_hidden * mask_expanded).sum(dim=1)  # (B, H)
        mask_count = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        action_context = masked_sum / mask_count  # (B, H)
        value_input = torch.cat([global_hidden, action_context], dim=-1)  # (B, 2H)
        value = self.value_head(value_input).squeeze(-1)
        return logits, value

    def act(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从策略分布采样动作，并返回对应对数概率与价值。"""

        logits, value = self.forward(obs['global'], obs['action_features'], obs['action_mask'])
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在 PPO/A2C 更新时重新评估给定动作的概率、熵和值。"""

        logits, value = self.forward(obs['global'], obs['action_features'], obs['action_mask'])
        distribution = torch.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy, value


def tensorize_observation(obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """把单步环境观测转成带 batch 维度的 Torch 张量。"""

    return {
        'global': torch.as_tensor(obs['global'], dtype=torch.float32, device=device).unsqueeze(0),
        'action_features': torch.as_tensor(obs['action_features'], dtype=torch.float32, device=device).unsqueeze(0),
        'action_mask': torch.as_tensor(obs['action_mask'], dtype=torch.float32, device=device).unsqueeze(0),
    }
