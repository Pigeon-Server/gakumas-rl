"""Behavioral Cloning 预训练 — 从 LLM 示范 trajectory 训练策略网络。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import CHECKPOINTS_DIR
from .model import MaskedPolicyValueNet
from .service import build_env_from_config


class BCTrainer:
    """从 trajectory jsonl 文件训练 MaskedPolicyValueNet。"""

    def __init__(
        self,
        global_dim: int,
        action_dim: int,
        max_actions: int,
        hidden_dim: int = 128,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.max_actions = max_actions
        self.model = MaskedPolicyValueNet(global_dim, action_dim, hidden_dim).to(self.device)

    def load_trajectories(self, paths: str | Path | list[str] | list[Path] | tuple[str | Path, ...]) -> list[dict]:
        """读取一个或多个 jsonl trajectory 文件，并重映射 episode_id 避免冲突。"""

        if isinstance(paths, (str, Path)):
            resolved_paths = [paths]
        else:
            resolved_paths = list(paths)
        records = []
        next_episode_id = 0
        for path in resolved_paths:
            episode_map: dict[int, int] = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    episode_id = int(payload['episode_id'])
                    if episode_id not in episode_map:
                        episode_map[episode_id] = next_episode_id
                        next_episode_id += 1
                    normalized = dict(payload)
                    normalized['episode_id'] = episode_map[episode_id]
                    records.append(normalized)
        return records

    def build_dataset(
        self, trajectories: list[dict], gamma: float = 0.99,
    ) -> TensorDataset:
        """把 trajectory 记录转为训练用张量数据集。

        返回 TensorDataset(global_obs, action_features, action_mask, actions, returns)。
        """
        # 按 episode 分组计算 discounted return
        episodes: dict[int, list[dict]] = {}
        for record in trajectories:
            ep_id = record['episode_id']
            episodes.setdefault(ep_id, []).append(record)

        all_global = []
        all_action_features = []
        all_action_mask = []
        all_actions = []
        all_returns = []

        for ep_id in sorted(episodes.keys()):
            steps = sorted(episodes[ep_id], key=lambda r: r['step'])
            # 计算 discounted return (从后往前)
            returns = [0.0] * len(steps)
            running_return = 0.0
            for i in reversed(range(len(steps))):
                running_return = steps[i]['reward'] + gamma * running_return
                returns[i] = running_return

            for i, record in enumerate(steps):
                obs = record['obs']
                all_global.append(obs['global'])
                all_action_features.append(obs['action_features'])
                all_action_mask.append(obs['action_mask'])
                all_actions.append(record['action'])
                all_returns.append(returns[i])

        return TensorDataset(
            torch.tensor(np.array(all_global), dtype=torch.float32),
            torch.tensor(np.array(all_action_features), dtype=torch.float32),
            torch.tensor(np.array(all_action_mask), dtype=torch.float32),
            torch.tensor(all_actions, dtype=torch.long),
            torch.tensor(all_returns, dtype=torch.float32),
        )

    def train(
        self,
        dataset: TensorDataset,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        value_loss_weight: float = 0.5,
    ) -> list[dict]:
        """标准 supervised learning 循环。"""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()

        history = []
        for epoch in range(epochs):
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_loss = 0.0
            n_batches = 0

            for global_obs, action_features, action_mask, actions, returns in loader:
                global_obs = global_obs.to(self.device)
                action_features = action_features.to(self.device)
                action_mask = action_mask.to(self.device)
                actions = actions.to(self.device)
                returns = returns.to(self.device)

                logits, value = self.model(global_obs, action_features, action_mask)

                # 策略损失：masked cross-entropy
                policy_loss = F.cross_entropy(logits, actions)

                # 价值辅助损失：用 discounted return 做监督
                value_loss = F.mse_loss(value, returns)

                loss = policy_loss + value_loss_weight * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()
                n_batches += 1

            avg_policy = total_policy_loss / max(n_batches, 1)
            avg_value = total_value_loss / max(n_batches, 1)
            avg_total = total_loss / max(n_batches, 1)
            history.append({
                'epoch': epoch + 1,
                'policy_loss': avg_policy,
                'value_loss': avg_value,
                'total_loss': avg_total,
            })
            print(
                f'[BC Epoch {epoch + 1}/{epochs}] '
                f'policy_loss={avg_policy:.4f} value_loss={avg_value:.4f} total={avg_total:.4f}'
            )

        return history

    def save_checkpoint(self, path: str | Path) -> None:
        """保存 checkpoint，格式与 trainer 兼容。"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'global_dim': self.model.global_encoder[0].in_features,
                'action_dim': self.model.action_encoder[0].in_features,
                'hidden_dim': self.model.global_encoder[0].out_features // 2,
            },
        }, str(output))
        print(f'[BC] Checkpoint saved to {output}')


def _infer_dims_from_env(env_config: dict) -> tuple[int, int, int]:
    """从环境推断网络维度。"""
    env = build_env_from_config(env_config)
    obs, _ = env.reset()
    global_dim = obs['global'].shape[0]
    max_actions, action_dim = obs['action_features'].shape
    return global_dim, action_dim, max_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Behavioral Cloning pretraining from LLM trajectories.')
    parser.add_argument(
        '--data',
        action='append',
        required=True,
        help='Path to trajectory .jsonl file; repeat to combine multiple files',
    )
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for return computation')
    parser.add_argument('--value-loss-weight', type=float, default=0.5)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--output', default=str(CHECKPOINTS_DIR / 'bc_pretrained.pt'))
    parser.add_argument('--mode', choices=('exam', 'lesson'), default='exam')
    parser.add_argument('--scenario', default='nia_master')
    parser.add_argument('--stage-type', default=None, help='exam 模式下用于推断环境维度的考试阶段；lesson 模式下忽略')
    parser.add_argument('--exam-reward-mode', choices=('score', 'clear'), default='score')
    parser.add_argument('--lesson-action-type', default='', help='lesson 模式下固定课程类型；留空时按环境默认课程池采样')
    parser.add_argument('--lesson-level-index', type=int, default=0, help='lesson 模式下固定课程等级序号；0 表示按主数据候选采样')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--idol-card-id', default='')
    parser.add_argument('--producer-level', type=int, default=35)
    parser.add_argument('--idol-rank', type=int, default=0)
    parser.add_argument('--dearness-level', type=int, default=0)
    parser.add_argument('--use-after-item', action='store_true')
    parser.add_argument('--include-deck-features', action='store_true')
    parser.add_argument(
        '--manual-exam-setup',
        action='append',
        default=[],
        help='manual exam setup jsonl; when provided, env dims are inferred against that dataset',
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
    return parser.parse_args()


def _build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    """把 BC 预训练 CLI 参数整理成统一环境配置。"""

    return {
        'mode': str(args.mode or 'exam'),
        'scenario': args.scenario,
        'stage_type': args.stage_type,
        'exam_reward_mode': args.exam_reward_mode,
        'seed': 0,
        'idol_card_id': args.idol_card_id,
        'producer_level': args.producer_level,
        'idol_rank': args.idol_rank,
        'dearness_level': args.dearness_level,
        'use_after_item': True if args.use_after_item else None,
        'lesson_action_type': args.lesson_action_type,
        'lesson_level_index': args.lesson_level_index,
        'include_deck_features': args.include_deck_features,
        'manual_exam_setup_paths': list(args.manual_exam_setup),
        'guarantee_card_effects': list(args.guarantee_card_effect),
        'force_card_groups': list(args.force_card),
    }


def main() -> int:
    args = parse_args()

    env_config = _build_env_config(args)

    print(
        f'[BC] Inferring network dimensions from env '
        f'(mode={env_config["mode"]}, scenario={args.scenario})...'
    )
    global_dim, action_dim, max_actions = _infer_dims_from_env(env_config)
    print(f'[BC] global_dim={global_dim}, action_dim={action_dim}, max_actions={max_actions}')

    trainer = BCTrainer(
        global_dim=global_dim,
        action_dim=action_dim,
        max_actions=max_actions,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )

    joined_paths = ', '.join(str(path) for path in args.data)
    print(f'[BC] Loading trajectories from {joined_paths}...')
    trajectories = trainer.load_trajectories(args.data)
    print(f'[BC] Loaded {len(trajectories)} records')

    dataset = trainer.build_dataset(trajectories, gamma=args.gamma)
    print(f'[BC] Dataset size: {len(dataset)} samples')

    trainer.train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_loss_weight=args.value_loss_weight,
    )

    trainer.save_checkpoint(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
