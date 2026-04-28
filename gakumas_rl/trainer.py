"""轻量 actor-critic 调试训练器，保留给快速本地实验使用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import RUNS_DIR
from .model import MaskedPolicyValueNet, tensorize_observation


@dataclass
class TrainResult:
    """轻量训练器输出的检查点与步数摘要。"""

    checkpoint_path: Path
    updates: int
    total_steps: int


class ActorCriticTrainer:
    """基于动作掩码策略网络的最小化 actor-critic 训练循环。"""

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = 'cpu',
        run_dir: str | Path = '',
    ):
        """根据环境观测维度初始化模型、优化器与输出目录。"""

        self.env = env
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        self.run_dir = Path(run_dir) if run_dir else RUNS_DIR
        self.run_dir.mkdir(parents=True, exist_ok=True)
        global_dim = int(env.observation_space['global'].shape[0])
        action_dim = int(env.observation_space['action_features'].shape[-1])
        self.model = MaskedPolicyValueNet(global_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.obs: dict[str, np.ndarray] | None = None
        self.num_timesteps = 0

    def _ensure_observation(self) -> dict[str, np.ndarray]:
        """确保训练器持有当前 rollout 的起始观测。"""

        if self.obs is None:
            self.obs, _ = self.env.reset()
        return self.obs

    def save_checkpoint(self, checkpoint_path: str | Path) -> Path:
        """保存模型、优化器和训练步数，便于续训。"""

        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'num_timesteps': int(self.num_timesteps),
            },
            target,
        )
        return target

    def load_checkpoint(self, checkpoint_path: str | Path) -> Path:
        """从 checkpoint 恢复模型、优化器和步数。"""

        target = Path(checkpoint_path)
        payload = torch.load(target, map_location=self.device)
        state_dict = payload.get('model_state_dict', payload)
        self.model.load_state_dict(state_dict, strict=True)
        optimizer_state = payload.get('optimizer_state_dict')
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        self.num_timesteps = int(payload.get('num_timesteps') or 0)
        self.obs = None
        return target

    def load_model_weights(self, checkpoint_path: str | Path, strict: bool = False) -> None:
        """仅加载模型权重，用于 BC 热启动等场景。"""

        payload = torch.load(Path(checkpoint_path), map_location=self.device)
        state_dict = payload.get('model_state_dict', payload)
        self.model.load_state_dict(state_dict, strict=strict)

    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> int:
        """对单个观测执行推理，返回动作索引。"""

        obs_tensor = tensorize_observation(obs, self.device)
        with torch.no_grad():
            logits, _ = self.model.forward(
                obs_tensor['global'],
                obs_tensor['action_features'],
                obs_tensor['action_mask'],
            )
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
        return int(action.item())

    def evaluate(self, env, episodes: int = 5, deterministic: bool = True, base_seed: int = 1000) -> tuple[float, float]:
        """在独立环境上评估当前策略。"""

        rewards: list[float] = []
        num_episodes = max(int(episodes), 1)
        for episode_idx in range(num_episodes):
            obs, _ = env.reset(seed=base_seed + episode_idx)
            episode_reward = 0.0
            while True:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
        return float(np.mean(rewards)), float(np.std(rewards))

    def train_update(self, rollout_steps: int = 256) -> dict[str, float]:
        """采样一段 rollout 并执行一次 actor-critic 更新。"""

        obs = self._ensure_observation()
        batch_obs: list[dict[str, np.ndarray]] = []
        batch_actions: list[int] = []
        batch_rewards: list[float] = []
        batch_values: list[float] = []
        batch_dones: list[float] = []
        steps_collected = 0

        for _ in range(max(int(rollout_steps), 1)):
            obs_tensor = tensorize_observation(obs, self.device)
            with torch.no_grad():
                action, _, value = self.model.act(obs_tensor)
            next_obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
            batch_obs.append(obs)
            batch_actions.append(int(action.item()))
            batch_rewards.append(float(reward))
            batch_values.append(float(value.item()))
            batch_dones.append(float(terminated or truncated))
            steps_collected += 1
            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset()

        bootstrap_value = 0.0
        if batch_dones and batch_dones[-1] < 0.5:
            obs_tensor = tensorize_observation(obs, self.device)
            with torch.no_grad():
                _, bootstrap = self.model.forward(
                    obs_tensor['global'],
                    obs_tensor['action_features'],
                    obs_tensor['action_mask'],
                )
            bootstrap_value = float(bootstrap.item())

        returns = []
        running_return = bootstrap_value
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)):
            running_return = reward + self.gamma * running_return * (1.0 - done)
            returns.append(running_return)
        returns.reverse()
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values_tensor = torch.as_tensor(batch_values, dtype=torch.float32, device=self.device)
        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        stacked_obs = {
            'global': torch.as_tensor(np.stack([item['global'] for item in batch_obs]), dtype=torch.float32, device=self.device),
            'action_features': torch.as_tensor(np.stack([item['action_features'] for item in batch_obs]), dtype=torch.float32, device=self.device),
            'action_mask': torch.as_tensor(np.stack([item['action_mask'] for item in batch_obs]), dtype=torch.float32, device=self.device),
        }
        actions_tensor = torch.as_tensor(batch_actions, dtype=torch.int64, device=self.device)
        log_probs, entropy, values = self.model.evaluate_actions(stacked_obs, actions_tensor)
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
        entropy_loss = entropy.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.obs = obs
        self.num_timesteps += steps_collected
        return {
            'steps': float(steps_collected),
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy_loss.item()),
            'total_loss': float(loss.item()),
        }

    def train(self, updates: int = 20, rollout_steps: int = 256, checkpoint_name: str = 'policy.pt') -> TrainResult:
        """采样 rollout 并执行多轮 actor-critic 更新。"""

        self.obs, _ = self.env.reset()
        for _ in range(max(int(updates), 1)):
            self.train_update(rollout_steps=rollout_steps)
        checkpoint_path = self.run_dir / checkpoint_name
        self.save_checkpoint(checkpoint_path)
        return TrainResult(checkpoint_path=checkpoint_path, updates=updates, total_steps=int(self.num_timesteps))
