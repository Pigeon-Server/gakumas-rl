"""无 LLM / 无规则老师的冷启动自举训练流程。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..interfaces.service import build_env_from_config
from ..loadout import DEFAULT_DEARNESS_LEVEL
from ..repository.master_data import RUNS_DIR
from .backends import TrainingSpec, run_training
from .bc_pretrain import BCTrainer, _infer_dims_from_env
from .competitive_metrics import CompetitiveEpisodeOutcome, competitive_outcome_from_info
from .device import log_resolved_device, resolve_torch_device


BOOTSTRAP_SCHEMA_VERSION = 1

_PRODUCE_REWARD_CLI_MAP = {
    'produce_reward_shape_scale': 'shape_scale',
    'produce_reward_param_weight': 'param_weight',
    'produce_reward_fan_weight': 'fan_weight',
    'produce_reward_resource_weight': 'resource_weight',
    'produce_reward_terminal_score_scale': 'terminal_score_scale',
    'produce_reward_grade_s': 'terminal_grade_s',
    'produce_reward_grade_a': 'terminal_grade_a',
    'produce_reward_grade_b': 'terminal_grade_b',
    'produce_reward_grade_c': 'terminal_grade_c',
    'produce_reward_scale': 'reward_scale',
    'produce_reward_clip': 'reward_clip',
    'produce_reward_route_clear_bonus': 'terminal_route_clear_bonus',
    'produce_reward_route_fail_penalty': 'terminal_route_fail_penalty',
    'produce_reward_stage_progress_weight': 'terminal_stage_progress_weight',
    'produce_reward_pp_left_waste_penalty': 'terminal_pp_left_waste_penalty',
    'produce_reward_nia_param_fallback_weight': 'terminal_nia_param_fallback_weight',
    'produce_reward_nia_vote_rank_bonus': 'terminal_nia_vote_rank_bonus',
}


@dataclass(frozen=True)
class EpisodeCandidate:
    """一个 checkpoint 在一个 seed 上跑出的完整候选轨迹。"""

    checkpoint_path: Path
    seed: int
    episode_id: int
    records: list[dict[str, Any]]
    total_reward: float
    terminal_score: float
    invalid_actions: int
    clear_rank: int
    top1: bool
    route: str
    final_rank: int | None
    all_auditions_first: bool
    steps: int

    def quality_key(self) -> tuple[int, int, int, float, float, int]:
        """返回用于同 seed 轨迹选优的排序键。"""

        return (
            -int(self.invalid_actions),
            int(self.top1),
            int(self.clear_rank),
            float(self.terminal_score),
            float(self.total_reward),
            -int(self.steps),
        )


def _json_default(value: Any) -> Any:
    """把 numpy / pathlib 类型转换成 JSON 可序列化值。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _collect_produce_reward_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    """从 CLI 参数中收集培育奖励覆盖项。"""

    overrides: dict[str, Any] = {}
    args_values = vars(args)
    for cli_attr, config_key in _PRODUCE_REWARD_CLI_MAP.items():
        value = args_values.get(cli_attr)
        if value is not None:
            overrides[config_key] = value
    return overrides if overrides else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写入一个缩进后的 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding='utf-8',
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """追加写入一行 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + '\n')


def _score_from_info(info: dict[str, Any]) -> float:
    """从环境 info 中提取可比较的终局分数。"""

    final_summary = info.get('final_summary')
    if isinstance(final_summary, dict):
        produce_result = final_summary.get('produce_result')
        if isinstance(produce_result, dict):
            for key in ('score', 'parameter_total', 'fan_votes'):
                value = produce_result.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        for key in ('final_score', 'final_exam_score'):
            value = final_summary.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    for key in ('final_score', 'score', 'exam_score', 'effective_score', 'lesson_score', 'lesson_target_value'):
        value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _clear_rank_from_info(info: dict[str, Any]) -> int:
    """把 clear / perfect 等终局状态压成排序等级。"""

    return competitive_outcome_from_info(info).tier


def _action_label(info: dict[str, Any], action: int) -> str:
    """从当前 info 的 action_labels 中取出动作标签。"""

    labels = info.get('action_labels') or []
    if isinstance(labels, list) and 0 <= int(action) < len(labels):
        return str(labels[int(action)])
    return ''


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """按 step 编号和文件名排序 checkpoint。"""

    digits = ''.join(char for char in path.stem if char.isdigit())
    return (int(digits) if digits else -1, path.name)


def discover_sb3_checkpoints(paths: Iterable[str | Path]) -> list[Path]:
    """从文件或目录中解析可用于 SB3 MaskablePPO 的 checkpoint 列表。"""

    checkpoints: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint path not found: {path}')
        if path.is_file() and path.suffix == '.zip':
            checkpoints[str(path.resolve())] = path
            continue
        if path.is_dir():
            for candidate in path.glob('*.zip'):
                checkpoints[str(candidate.resolve())] = candidate
            checkpoint_dir = path / 'checkpoints'
            if checkpoint_dir.exists():
                for candidate in checkpoint_dir.glob('*.zip'):
                    checkpoints[str(candidate.resolve())] = candidate
    return sorted(checkpoints.values(), key=_checkpoint_sort_key)


def discover_policy_checkpoints(
    paths: Iterable[str | Path],
    *,
    backend: str = 'sb3',
) -> list[Path]:
    """按训练后端自动发现可评估的策略 checkpoint。"""

    if str(backend) == 'sb3':
        return discover_sb3_checkpoints(paths)
    raise ValueError(f'Unsupported bootstrap backend: {backend}')


def preflight_env(env_config: dict[str, Any]) -> dict[str, Any]:
    """执行训练前环境预检，确保 mask 与 step 基本可用。"""

    env = build_env_from_config(dict(env_config))
    try:
        obs, info = env.reset(seed=int(env_config.get('seed') or 0))
        required_keys = ('global', 'action_features', 'action_mask')
        missing = [key for key in required_keys if key not in obs]
        if missing:
            raise ValueError(f'Observation missing required keys: {missing}')
        mask = np.asarray(obs['action_mask'], dtype=np.float32)
        valid_actions = np.flatnonzero(mask > 0.5)
        if valid_actions.size <= 0:
            raise ValueError('Action mask has no valid action on reset.')
        action = int(valid_actions[0])
        _next_obs, reward, terminated, truncated, step_info = env.step(action)
        if not np.isfinite(float(reward)):
            raise ValueError(f'Environment returned non-finite reward: {reward}')
        return {
            'global_shape': list(np.asarray(obs['global']).shape),
            'action_features_shape': list(np.asarray(obs['action_features']).shape),
            'action_mask_shape': list(mask.shape),
            'first_action': action,
            'first_action_label': _action_label(info, action),
            'first_reward': float(reward),
            'terminated': bool(terminated),
            'truncated': bool(truncated),
            'step_info_keys': sorted(str(key) for key in step_info.keys()),
        }
    finally:
        try:
            env.close()
        except AttributeError:
            pass


def run_checkpoint_episode(
    *,
    checkpoint_path: str | Path,
    env_config: dict[str, Any],
    seed: int,
    episode_id: int,
    backend: str = 'sb3',
    device: str = 'cpu',
    deterministic: bool = True,
    max_steps: int = 512,
) -> EpisodeCandidate:
    """让一个 checkpoint 在指定 seed 上跑一局并返回完整轨迹。"""

    if str(backend) != 'sb3':
        raise ValueError(f'Unsupported checkpoint backend: {backend}')
    resolved_device = resolve_torch_device(device)
    try:
        from sb3_contrib import MaskablePPO
    except ModuleNotFoundError as exc:
        raise SystemExit('SB3 trajectory selection requires sb3-contrib in the active environment.') from exc

    resolved_config = dict(env_config)
    resolved_config['seed'] = int(seed)
    resolved_config['include_action_labels_in_step_info'] = True
    env = build_env_from_config(resolved_config)

    records: list[dict[str, Any]] = []
    total_reward = 0.0
    terminal_score = 0.0
    clear_rank = 0
    top1 = False
    route = ''
    final_rank: int | None = None
    all_auditions_first = False
    invalid_actions = 0
    try:
        sb3_model = MaskablePPO.load(str(checkpoint_path), device=resolved_device)
        obs, info = env.reset(seed=int(seed))
        for step in range(max(int(max_steps), 1)):
            mask = np.asarray(obs['action_mask'], dtype=np.float32)
            action_value, _ = sb3_model.predict(obs, deterministic=deterministic, action_masks=mask)
            action = int(action_value)
            action_valid = 0 <= action < len(mask) and bool(mask[action] > 0.5)
            if not action_valid:
                invalid_actions += 1

            action_label = _action_label(info, action)
            obs_record = {
                'global': np.asarray(obs['global'], dtype=np.float32).tolist(),
                'action_features': np.asarray(obs['action_features'], dtype=np.float32).tolist(),
                'action_mask': mask.tolist(),
            }
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            total_reward += float(reward)
            terminal_score = _score_from_info(next_info)
            outcome: CompetitiveEpisodeOutcome = competitive_outcome_from_info(next_info)
            clear_rank = max(clear_rank, outcome.tier)
            top1 = top1 or outcome.top1
            route = outcome.route or route
            final_rank = outcome.final_rank if outcome.final_rank is not None else final_rank
            all_auditions_first = all_auditions_first or outcome.all_auditions_first
            final_summary_info = next_info.get('final_summary') if isinstance(next_info.get('final_summary'), dict) else {}
            produce_result_info = (
                final_summary_info.get('produce_result')
                if isinstance(final_summary_info.get('produce_result'), dict)
                else {}
            )

            records.append(
                {
                    'schema_version': BOOTSTRAP_SCHEMA_VERSION,
                    'episode_id': int(episode_id),
                    'step': int(step),
                    'seed': int(seed),
                    'teacher_checkpoint': str(Path(checkpoint_path)),
                    'obs': obs_record,
                    'action': int(action),
                    'action_valid': bool(action_valid),
                    'reward': float(reward),
                    'terminated': bool(terminated),
                    'truncated': bool(truncated),
                    'info': {
                        'score': float(terminal_score),
                        'turn': int(next_info.get('turn', 0) or 0),
                        'action_label': action_label,
                        'clear_state': str(next_info.get('clear_state', '')),
                        'lesson_cleared': bool(next_info.get('lesson_cleared') or False),
                        'scenario': str(next_info.get('scenario', '')),
                        'stage_type': str(next_info.get('stage_type', '')),
                        'battle_kind': str(next_info.get('battle_kind', '')),
                        'reward_mode': str(next_info.get('reward_mode', '')),
                        'invalid_action': bool(next_info.get('invalid_action', False)),
                        'route_clear': bool(final_summary_info.get('route_clear')),
                        'competitive_pass': bool(final_summary_info.get('competitive_pass')),
                        'competitive_top1': bool(final_summary_info.get('competitive_top1')),
                        'all_auditions_first': bool(final_summary_info.get('all_auditions_first')),
                        'final_rank': final_summary_info.get('final_rank'),
                        'produce_rank': str(produce_result_info.get('rank') or ''),
                    },
                }
            )
            obs = next_obs
            info = next_info
            if terminated or truncated:
                break
    finally:
        try:
            env.close()
        except AttributeError:
            pass

    return EpisodeCandidate(
        checkpoint_path=Path(checkpoint_path),
        seed=int(seed),
        episode_id=int(episode_id),
        records=records,
        total_reward=float(total_reward),
        terminal_score=float(terminal_score),
        invalid_actions=int(invalid_actions),
        clear_rank=int(clear_rank),
        top1=bool(top1),
        route=str(route),
        final_rank=final_rank,
        all_auditions_first=bool(all_auditions_first),
        steps=len(records),
    )


def collect_best_trajectories(
    *,
    checkpoint_paths: Iterable[str | Path],
    env_config: dict[str, Any],
    seeds: Iterable[int],
    output_path: str | Path,
    summary_path: str | Path | None = None,
    backend: str = 'sb3',
    device: str = 'cpu',
    deterministic: bool = True,
    max_steps: int = 512,
) -> dict[str, Any]:
    """多 checkpoint 跑同一批 seed，并把每个 seed 的最优轨迹写成 JSONL。"""

    checkpoints = discover_policy_checkpoints(checkpoint_paths, backend=backend)
    if not checkpoints:
        raise ValueError(f'No {backend} checkpoints were provided for trajectory selection.')

    output = Path(output_path)
    if output.exists():
        output.unlink()

    selected_summaries: list[dict[str, Any]] = []
    attempted = 0
    episode_id = 0
    for seed in seeds:
        candidates: list[EpisodeCandidate] = []
        for checkpoint in checkpoints:
            attempted += 1
            candidates.append(
                run_checkpoint_episode(
                    checkpoint_path=checkpoint,
                    env_config=env_config,
                    seed=int(seed),
                    episode_id=episode_id,
                    backend=backend,
                    device=device,
                    deterministic=deterministic,
                    max_steps=max_steps,
                )
            )
        valid_candidates = [candidate for candidate in candidates if candidate.invalid_actions == 0 and candidate.records]
        if not valid_candidates:
            raise RuntimeError(f'No valid trajectory found for seed={seed}; refusing to train on invalid actions.')
        best = max(valid_candidates, key=lambda candidate: candidate.quality_key())
        for record in best.records:
            payload = dict(record)
            payload['episode_id'] = episode_id
            payload['quality'] = {
                'terminal_score': best.terminal_score,
                'total_reward': best.total_reward,
                'clear_rank': best.clear_rank,
                'invalid_actions': best.invalid_actions,
                'selected_from': len(candidates),
            }
            _append_jsonl(output, payload)
        selected_summaries.append(
            {
                'seed': int(seed),
                'episode_id': int(episode_id),
                'checkpoint_path': str(best.checkpoint_path),
                'terminal_score': best.terminal_score,
                'total_reward': best.total_reward,
                'clear_rank': best.clear_rank,
                'top1': best.top1,
                'route': best.route,
                'final_rank': best.final_rank,
                'all_auditions_first': best.all_auditions_first,
                'steps': best.steps,
                'invalid_actions': best.invalid_actions,
            }
        )
        episode_id += 1

    summary = {
        'schema_version': BOOTSTRAP_SCHEMA_VERSION,
        'checkpoint_count': len(checkpoints),
        'attempted_episodes': attempted,
        'selected_episodes': len(selected_summaries),
        'output_path': str(output),
        'mean_score': float(np.mean([item['terminal_score'] for item in selected_summaries])) if selected_summaries else 0.0,
        'mean_reward': float(np.mean([item['total_reward'] for item in selected_summaries])) if selected_summaries else 0.0,
        'top1_count': int(np.count_nonzero([item['top1'] for item in selected_summaries])) if selected_summaries else 0,
        'competitive_pass_count': int(np.count_nonzero([item['clear_rank'] >= 1 for item in selected_summaries])) if selected_summaries else 0,
        'selected': selected_summaries,
    }
    if summary_path is not None:
        _write_json(Path(summary_path), summary)
    return summary


def _build_sb3_masked_env(env_config: dict[str, Any]):
    """构造 SB3 MaskablePPO 可用的 action-mask 环境。"""

    try:
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise SystemExit('SB3 bootstrap requires stable-baselines3 and sb3-contrib in the active environment.') from exc

    env = build_env_from_config(dict(env_config))
    try:
        env.action_masks
    except AttributeError:
        pass
    else:
        env = ActionMasker(env, lambda wrapped_env: wrapped_env.action_masks())
    return Monitor(env)


def train_sb3_bc_from_trajectories(
    *,
    data_path: str | Path,
    env_config: dict[str, Any],
    output_path: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    value_loss_weight: float,
    success_oversample_factor: int,
    success_only: bool,
    device: str,
    seed: int,
    rollout_steps: int,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """用自举轨迹直接监督训练 SB3 MaskablePPO policy，并保存 SB3 原生 zip。"""

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    try:
        from sb3_contrib import MaskablePPO
    except ModuleNotFoundError as exc:
        raise SystemExit('SB3 BC distillation requires sb3-contrib in the active environment.') from exc

    resolved_device = resolve_torch_device(device)
    global_dim, action_dim, max_actions = _infer_dims_from_env(env_config)
    dataset_builder = BCTrainer(
        global_dim=global_dim,
        action_dim=action_dim,
        max_actions=max_actions,
        device=resolved_device,
    )
    trajectories = dataset_builder.load_trajectories(data_path)
    dataset = dataset_builder.build_dataset(
        trajectories,
        gamma=gamma,
        success_oversample_factor=success_oversample_factor,
        success_only=success_only,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    env = _build_sb3_masked_env(env_config)
    try:
        if initial_checkpoint and Path(initial_checkpoint).suffix == '.zip' and Path(initial_checkpoint).exists():
            model = MaskablePPO.load(str(initial_checkpoint), env=env, device=resolved_device)
        else:
            model = MaskablePPO(
                'MultiInputPolicy',
                env,
                verbose=0,
                seed=int(seed),
                device=resolved_device,
                n_steps=max(int(rollout_steps), 32),
                learning_rate=float(learning_rate),
            )

        policy = model.policy
        policy.set_training_mode(True)
        optimizer = policy.optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = float(learning_rate)

        history: list[dict[str, Any]] = []
        for epoch in range(max(int(epochs), 1)):
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0
            for global_obs, action_features, action_mask, actions, returns in loader:
                global_obs = global_obs.to(policy.device)
                action_features = action_features.to(policy.device)
                action_mask = action_mask.to(policy.device)
                actions = actions.to(policy.device)
                returns = returns.to(policy.device)
                mask_bool = action_mask > 0.5
                obs_tensor = {
                    'global': global_obs,
                    'action_features': action_features,
                    'action_mask': action_mask,
                }

                values, log_prob, _entropy = policy.evaluate_actions(obs_tensor, actions, action_masks=mask_bool)
                policy_loss = -log_prob.mean()
                value_loss = F.mse_loss(values.flatten(), returns)
                loss = policy_loss + float(value_loss_weight) * value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    distribution = policy.get_distribution(obs_tensor, action_masks=mask_bool)
                    predictions = distribution.get_actions(deterministic=True)
                total_correct += int((predictions == actions).sum().item())
                total_samples += int(actions.numel())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_loss += float(loss.item())
                n_batches += 1

            avg_policy = total_policy_loss / max(n_batches, 1)
            avg_value = total_value_loss / max(n_batches, 1)
            avg_total = total_loss / max(n_batches, 1)
            accuracy = total_correct / max(total_samples, 1)
            history.append(
                {
                    'epoch': epoch + 1,
                    'policy_loss': avg_policy,
                    'value_loss': avg_value,
                    'total_loss': avg_total,
                    'masked_accuracy': accuracy,
                }
            )
            print(
                f'[SB3 BC Epoch {epoch + 1}/{epochs}] '
                f'policy_loss={avg_policy:.4f} value_loss={avg_value:.4f} '
                f'total={avg_total:.4f} masked_acc={accuracy:.4f}'
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f'[SB3 BC] Checkpoint saved to {output}')
    finally:
        env.close()

    return {
        'data_path': str(data_path),
        'output_path': str(output_path),
        'samples': len(dataset),
        'success_oversample_factor': int(success_oversample_factor),
        'success_only': bool(success_only),
        'history': history,
    }


def _checkpoint_compatible_with_backend(path: Path, backend: str) -> bool:
    """判断 checkpoint 文件是否能被指定后端直接评估或加载。"""

    if str(backend) == 'sb3':
        return path.suffix == '.zip'
    return False


def evaluate_policy_checkpoints(
    *,
    checkpoint_paths: Iterable[str | Path],
    env_config: dict[str, Any],
    seeds: Iterable[int],
    backend: str,
    output_path: str | Path | None = None,
    device: str = 'cpu',
    deterministic: bool = True,
    max_steps: int = 512,
    score_cap: float = 300.0,
) -> dict[str, Any]:
    """用固定 seed 横评 checkpoint，并给出稳健推荐模型。"""

    checkpoints = discover_policy_checkpoints(checkpoint_paths, backend=backend)
    resolved_device = resolve_torch_device(device)
    seed_values = [int(seed) for seed in seeds]
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        candidates = [
            run_checkpoint_episode(
                checkpoint_path=checkpoint,
                env_config=env_config,
                seed=seed,
                episode_id=idx,
                backend=backend,
                device=resolved_device,
                deterministic=deterministic,
                max_steps=max_steps,
            )
            for idx, seed in enumerate(seed_values)
        ]
        rewards = np.asarray([candidate.total_reward for candidate in candidates], dtype=np.float64)
        scores = np.asarray([candidate.terminal_score for candidate in candidates], dtype=np.float64)
        clear_ranks = np.asarray([candidate.clear_rank for candidate in candidates], dtype=np.int64)
        top1_flags = np.asarray([candidate.top1 for candidate in candidates], dtype=np.bool_)
        all_auditions_first_flags = np.asarray([candidate.all_auditions_first for candidate in candidates], dtype=np.bool_)
        invalid_actions = int(sum(candidate.invalid_actions for candidate in candidates))
        steps = np.asarray([candidate.steps for candidate in candidates], dtype=np.float64)
        capped_scores = np.minimum(scores, float(score_cap))
        row = {
            'checkpoint_path': str(checkpoint),
            'episodes': len(candidates),
            'mean_reward': float(rewards.mean()) if len(rewards) else 0.0,
            'median_reward': float(np.median(rewards)) if len(rewards) else 0.0,
            'mean_score': float(scores.mean()) if len(scores) else 0.0,
            'median_score': float(np.median(scores)) if len(scores) else 0.0,
            'p75_score': float(np.percentile(scores, 75)) if len(scores) else 0.0,
            'p90_score': float(np.percentile(scores, 90)) if len(scores) else 0.0,
            'max_score': float(scores.max()) if len(scores) else 0.0,
            'mean_score_capped': float(capped_scores.mean()) if len(capped_scores) else 0.0,
            'clear_count': int(np.count_nonzero(clear_ranks >= 1)),
            'top1_count': int(np.count_nonzero(top1_flags)),
            'best_tier_count': int(np.count_nonzero(clear_ranks >= 2)),
            'all_auditions_first_count': int(np.count_nonzero(all_auditions_first_flags)),
            'invalid_actions': invalid_actions,
            'mean_steps': float(steps.mean()) if len(steps) else 0.0,
        }
        row['selection_key'] = [
            -int(row['invalid_actions']),
            int(row['top1_count']),
            int(row['all_auditions_first_count']),
            int(row['best_tier_count']),
            int(row['clear_count']),
            float(row['median_score']),
            float(row['mean_reward']),
            float(row['mean_score_capped']),
            -float(row['mean_steps']),
        ]
        rows.append(row)

    recommended = max(rows, key=lambda item: tuple(item['selection_key'])) if rows else None
    summary = {
        'schema_version': BOOTSTRAP_SCHEMA_VERSION,
        'backend': str(backend),
        'seed_values': seed_values,
        'score_cap': float(score_cap),
        'checkpoint_count': len(checkpoints),
        'recommended_checkpoint': recommended['checkpoint_path'] if recommended else None,
        'recommended': recommended,
        'checkpoints': rows,
    }
    if output_path is not None:
        _write_json(Path(output_path), summary)
    return summary


def _resolve_final_rl_timesteps(args: argparse.Namespace) -> int:
    """解析最终 RL 微调步数，确保显式 0 能真正跳过。"""

    final_rl_timesteps = int(args.final_rl_timesteps)
    if final_rl_timesteps < 0:
        return int(args.rl_timesteps) if bool(args.autopilot) else 0
    return final_rl_timesteps


def run_self_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整的 RL 探索、轨迹选优、BC 蒸馏循环。"""

    base_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / 'self_bootstrap'
    base_dir.mkdir(parents=True, exist_ok=True)
    backend = str(args.backend)
    resolved_device = resolve_torch_device(args.device)
    log_resolved_device(args.device, resolved_device, prefix='[Bootstrap Device]')
    final_rl_timesteps = _resolve_final_rl_timesteps(args)
    env_config = _build_env_config(args)
    preflight = preflight_env(env_config)
    _write_json(base_dir / 'preflight.json', preflight)

    seed_values = list(range(int(args.bootstrap_seed_start), int(args.bootstrap_seed_start) + int(args.bootstrap_seed_count)))
    current_pretrained = Path(args.initial_checkpoint) if args.initial_checkpoint else None
    iteration_summaries: list[dict[str, Any]] = []
    extra_final_checkpoint_inputs: list[Path] = []

    for iteration in range(1, int(args.iterations) + 1):
        iteration_dir = base_dir / f'iter_{iteration:03d}'
        rl_dir = iteration_dir / 'rl'
        trajectory_path = iteration_dir / 'selected_trajectories.jsonl'
        trajectory_summary_path = iteration_dir / 'trajectory_summary.json'
        bc_checkpoint = iteration_dir / 'bc_distilled.zip'

        training_result = run_training(
            TrainingSpec(
                backend=backend,
                env_config=env_config,
                total_timesteps=int(args.rl_timesteps),
                checkpoint_freq=int(args.checkpoint_freq),
                eval_freq=int(args.eval_freq),
                eval_episodes=int(args.eval_episodes),
                rollout_steps=int(args.rollout_steps),
                learning_rate=float(args.learning_rate),
                device=resolved_device,
                run_dir=rl_dir,
                seed=int(args.seed),
                pretrained_checkpoint=str(current_pretrained) if current_pretrained is not None else None,
            )
        )

        checkpoint_inputs: list[Path] = [training_result.run_dir]
        if (
            current_pretrained is not None
            and current_pretrained.exists()
            and _checkpoint_compatible_with_backend(current_pretrained, backend)
        ):
            checkpoint_inputs.append(current_pretrained)
        trajectory_summary = collect_best_trajectories(
            checkpoint_paths=checkpoint_inputs,
            env_config=env_config,
            seeds=seed_values,
            output_path=trajectory_path,
            summary_path=trajectory_summary_path,
            backend=backend,
            device=resolved_device,
            deterministic=not bool(args.stochastic_eval),
            max_steps=int(args.max_episode_steps),
        )
        bc_summary = train_sb3_bc_from_trajectories(
            data_path=trajectory_path,
            env_config=env_config,
            output_path=bc_checkpoint,
            epochs=int(args.bc_epochs),
            batch_size=int(args.bc_batch_size),
            learning_rate=float(args.bc_learning_rate),
            gamma=float(args.gamma),
            value_loss_weight=float(args.value_loss_weight),
            success_oversample_factor=int(args.bc_success_oversample_factor),
            success_only=False,
            device=resolved_device,
            seed=int(args.seed),
            rollout_steps=int(args.rollout_steps),
            initial_checkpoint=training_result.latest_checkpoint,
        )
        success_only_bc_checkpoint = iteration_dir / 'bc_success_only.zip'
        success_only_bc_summary = None
        if bool(args.bc_success_only):
            success_only_bc_summary = train_sb3_bc_from_trajectories(
                data_path=trajectory_path,
                env_config=env_config,
                output_path=success_only_bc_checkpoint,
                epochs=int(args.bc_epochs),
                batch_size=int(args.bc_batch_size),
                learning_rate=float(args.bc_learning_rate),
                gamma=float(args.gamma),
                value_loss_weight=float(args.value_loss_weight),
                success_oversample_factor=int(args.bc_success_oversample_factor),
                success_only=True,
                device=resolved_device,
                seed=int(args.seed),
                rollout_steps=int(args.rollout_steps),
                initial_checkpoint=training_result.latest_checkpoint,
            )
        current_pretrained = success_only_bc_checkpoint if success_only_bc_summary is not None else bc_checkpoint
        iteration_summary = {
            'iteration': iteration,
            'backend': backend,
            'rl_run_dir': str(training_result.run_dir),
            'rl_checkpoint': str(training_result.latest_checkpoint) if training_result.latest_checkpoint else None,
            'trajectory_summary': trajectory_summary,
            'bc_summary': {
                'output_path': bc_summary['output_path'],
                'samples': bc_summary['samples'],
                'last_epoch': bc_summary['history'][-1] if bc_summary['history'] else None,
            },
        }
        if success_only_bc_summary is not None:
            iteration_summary['bc_success_only_summary'] = {
                'output_path': success_only_bc_summary['output_path'],
                'samples': success_only_bc_summary['samples'],
                'last_epoch': success_only_bc_summary['history'][-1] if success_only_bc_summary['history'] else None,
            }
            extra_final_checkpoint_inputs.append(success_only_bc_checkpoint)
        iteration_summaries.append(iteration_summary)
        _write_json(iteration_dir / 'iteration_summary.json', iteration_summary)

    final_training_summary: dict[str, Any] | None = None
    final_policy_checkpoint = current_pretrained
    if final_rl_timesteps > 0:
        final_training_result = run_training(
            TrainingSpec(
                backend=backend,
                env_config=env_config,
                total_timesteps=final_rl_timesteps,
                checkpoint_freq=int(args.checkpoint_freq),
                eval_freq=int(args.eval_freq),
                eval_episodes=int(args.eval_episodes),
                rollout_steps=int(args.rollout_steps),
                learning_rate=float(args.learning_rate),
                device=resolved_device,
                run_dir=base_dir / 'final_rl',
                seed=int(args.seed),
                pretrained_checkpoint=str(current_pretrained) if current_pretrained is not None else None,
            )
        )
        final_policy_checkpoint = final_training_result.latest_checkpoint
        final_training_summary = {
            'backend': final_training_result.backend,
            'run_dir': str(final_training_result.run_dir),
            'latest_checkpoint': str(final_training_result.latest_checkpoint) if final_training_result.latest_checkpoint else None,
            'total_timesteps': final_training_result.total_timesteps,
            'evaluation_log': str(final_training_result.evaluation_log) if final_training_result.evaluation_log else None,
            'metadata_log': str(final_training_result.metadata_log) if final_training_result.metadata_log else None,
            'replay_json': str(final_training_result.replay_json) if final_training_result.replay_json else None,
        }

    final_checkpoint_inputs: list[Path] = []
    if current_pretrained is not None and current_pretrained.exists():
        final_checkpoint_inputs.append(current_pretrained)
    final_checkpoint_inputs.extend(path for path in extra_final_checkpoint_inputs if path.exists())
    if final_training_summary is not None and final_training_summary.get('run_dir'):
        final_checkpoint_inputs.append(Path(str(final_training_summary['run_dir'])))
    elif final_policy_checkpoint is not None and final_policy_checkpoint.exists():
        final_checkpoint_inputs.append(final_policy_checkpoint)
    final_evaluation = None
    if final_checkpoint_inputs:
        final_evaluation = evaluate_policy_checkpoints(
            checkpoint_paths=final_checkpoint_inputs,
            env_config=env_config,
            seeds=seed_values,
            backend=backend,
            output_path=base_dir / 'final_checkpoint_evaluation.json',
            device=resolved_device,
            deterministic=not bool(args.stochastic_eval),
            max_steps=int(args.max_episode_steps),
            score_cap=float(args.selection_score_cap),
        )

    summary = {
        'schema_version': BOOTSTRAP_SCHEMA_VERSION,
        'backend': backend,
        'autopilot': bool(args.autopilot),
        'requested_device': str(args.device),
        'resolved_device': resolved_device,
        'env_config': env_config,
        'preflight': preflight,
        'seed_values': seed_values,
        'iterations': iteration_summaries,
        'final_rl': final_training_summary,
        'final_evaluation': final_evaluation,
        'final_pretrained_checkpoint': str(current_pretrained) if current_pretrained is not None else None,
        'recommended_checkpoint': (
            final_evaluation.get('recommended_checkpoint')
            if isinstance(final_evaluation, dict)
            else (str(final_policy_checkpoint) if final_policy_checkpoint is not None else None)
        ),
    }
    _write_json(base_dir / 'bootstrap_summary.json', summary)
    return summary


def parse_args() -> argparse.Namespace:
    """解析自举训练命令行参数。"""

    parser = argparse.ArgumentParser(description='Run no-LLM self-bootstrap training.')
    parser.add_argument('--mode', choices=('planning', 'exam', 'lesson', 'battle'), default='lesson')
    parser.add_argument('--backend', choices=('sb3',), default='sb3')
    parser.add_argument(
        '--autopilot',
        action='store_true',
        help='全自动模式：默认使用 SB3，并在每轮 BC 后追加最终 RL 微调与 checkpoint 横评',
    )
    parser.add_argument('--scenario', default='nia_master')
    parser.add_argument('--stage-type', default=None)
    parser.add_argument('--exam-reward-mode', choices=('score', 'clear'), default='clear')
    parser.add_argument('--lesson-action-type', default='lesson_vocal_normal')
    parser.add_argument('--lesson-level-index', type=int, default=2)
    parser.add_argument('--lesson-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', help='训练设备；auto 会按 cuda -> mps -> cpu 优先级选择')
    parser.add_argument('--run-dir', default='')
    parser.add_argument('--initial-checkpoint', default='')

    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--rl-timesteps', type=int, default=20_000)
    parser.add_argument('--rollout-steps', type=int, default=512)
    parser.add_argument('--checkpoint-freq', type=int, default=5_000)
    parser.add_argument('--eval-freq', type=int, default=5_000)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument(
        '--final-rl-timesteps',
        type=int,
        default=-1,
        help='BC 蒸馏后追加的最终 RL 微调步数；-1 表示按入口默认值，0 表示跳过',
    )
    parser.add_argument('--max-episode-steps', type=int, default=512)
    parser.add_argument('--stochastic-eval', action='store_true', help='轨迹选优时使用采样动作；默认使用确定性 argmax')
    parser.add_argument('--selection-score-cap', type=float, default=300.0, help='最终横评均分的截断上限，仅用于抗 outlier 汇总')

    parser.add_argument('--bootstrap-seed-start', type=int, default=1000)
    parser.add_argument('--bootstrap-seed-count', type=int, default=16)

    parser.add_argument('--bc-epochs', type=int, default=8)
    parser.add_argument('--bc-batch-size', type=int, default=64)
    parser.add_argument('--bc-learning-rate', type=float, default=1e-3)
    parser.add_argument('--bc-success-oversample-factor', type=int, default=4)
    parser.add_argument('--bc-success-only', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--value-loss-weight', type=float, default=0.5)

    parser.add_argument('--idol-card-id', default='')
    parser.add_argument('--producer-level', type=int, default=35)
    parser.add_argument('--idol-rank', type=int, default=0)
    parser.add_argument('--dearness-level', type=int, default=DEFAULT_DEARNESS_LEVEL)
    parser.add_argument('--use-after-item', action='store_true')
    parser.add_argument('--force-lowest-audition-route', action='store_true')
    parser.add_argument('--include-deck-features', action='store_true')
    parser.add_argument('--manual-exam-setup', action='append', default=[])
    parser.add_argument('--guarantee-card-effect', action='append', default=[])
    parser.add_argument('--force-card', action='append', default=[])
    parser.add_argument('--produce-reward-config', default=None, help='培育奖励配置 JSON 文件路径')
    parser.add_argument('--produce-reward-shape-scale', type=float, default=None, help='培育阶段势函数差分缩放')
    parser.add_argument('--produce-reward-param-weight', type=float, default=None, help='培育参数势函数权重')
    parser.add_argument('--produce-reward-fan-weight', type=float, default=None, help='培育粉丝票数势函数权重')
    parser.add_argument('--produce-reward-resource-weight', type=float, default=None, help='培育资源势函数权重')
    parser.add_argument('--produce-reward-terminal-score-scale', type=float, default=None, help='培育终局评分缩放')
    parser.add_argument('--produce-reward-grade-s', type=float, default=None, help='培育 S 评价奖励')
    parser.add_argument('--produce-reward-grade-a', type=float, default=None, help='培育 A 评价奖励')
    parser.add_argument('--produce-reward-grade-b', type=float, default=None, help='培育 B 评价奖励')
    parser.add_argument('--produce-reward-grade-c', type=float, default=None, help='培育 C 评价奖励')
    parser.add_argument('--produce-reward-route-clear-bonus', type=float, default=None, help='完整育成通关奖励')
    parser.add_argument('--produce-reward-route-fail-penalty', type=float, default=None, help='完整育成失败惩罚')
    parser.add_argument('--produce-reward-stage-progress-weight', type=float, default=None, help='阶段进度奖励权重')
    parser.add_argument('--produce-reward-pp-left-waste-penalty', type=float, default=None, help='剩余 P 点浪费惩罚')
    parser.add_argument('--produce-reward-nia-param-fallback-weight', type=float, default=None, help='NIA 参数兜底奖励权重')
    parser.add_argument('--produce-reward-nia-vote-rank-bonus', type=float, default=None, help='NIA vote rank 奖励')
    parser.add_argument('--produce-reward-scale', type=float, default=None, help='培育奖励全局缩放')
    parser.add_argument('--produce-reward-clip', type=float, default=None, help='培育奖励裁剪范围 (0=不裁剪)')
    return parser.parse_args()


def _build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    """把自举 CLI 参数整理成环境配置。"""

    return {
        'mode': str(args.mode),
        'scenario': str(args.scenario),
        'stage_type': args.stage_type,
        'exam_reward_mode': str(args.exam_reward_mode),
        'seed': int(args.seed),
        'idol_card_id': str(args.idol_card_id or ''),
        'producer_level': int(args.producer_level),
        'idol_rank': int(args.idol_rank),
        'dearness_level': int(args.dearness_level),
        'use_after_item': True if args.use_after_item else None,
        'force_lowest_audition_route': bool(args.force_lowest_audition_route),
        'lesson_action_type': str(args.lesson_action_type or ''),
        'lesson_level_index': int(args.lesson_level_index),
        'lesson_ratio': float(args.lesson_ratio),
        'include_deck_features': bool(args.include_deck_features),
        'manual_exam_setup_paths': list(args.manual_exam_setup),
        'guarantee_card_effects': list(args.guarantee_card_effect),
        'force_card_groups': list(args.force_card),
        'produce_reward_config_path': str(args.produce_reward_config or '') or None,
        'produce_reward_overrides': _collect_produce_reward_overrides(args),
        'planning_exam_checkpoints': dict(vars(args).get('planning_exam_checkpoints', {}) or {}),
    }


def main() -> int:
    """自举训练命令行入口。"""

    summary = run_self_bootstrap(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
