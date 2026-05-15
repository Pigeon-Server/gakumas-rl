"""全自动无 LLM 课程自举训练入口。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ..loadout import DEFAULT_DEARNESS_LEVEL
from ..repository.master_data import RUNS_DIR
from .self_bootstrap import _build_env_config, _json_default, parse_args as parse_bootstrap_args, preflight_env, run_self_bootstrap


@dataclass(frozen=True)
class AutopilotStage:
    """autopilot 课程中的一个训练阶段。"""

    name: str
    scenario: str
    mode: str = 'lesson'
    stage_type: str | None = None
    lesson_level_index: int = 2
    producer_level: int = 35
    dearness_level: int = DEFAULT_DEARNESS_LEVEL


@dataclass(frozen=True)
class AutopilotControlArgs:
    """autopilot 自身参数和透传给 self-bootstrap 的参数。"""

    curriculum: bool
    no_curriculum: bool
    curriculum_name: str
    curriculum_start_stage: int
    curriculum_stage_specs: tuple[str, ...]
    bootstrap_args: tuple[str, ...]


DEFAULT_CURRICULUM_STAGES: tuple[AutopilotStage, ...] = (
    AutopilotStage(
        name='stage_001_first_star_mid_exam',
        scenario='first_star_regular',
        mode='exam',
        stage_type='ProduceStepType_AuditionMid1',
        dearness_level=0,
    ),
    AutopilotStage(
        name='stage_002_first_star_final_exam',
        scenario='first_star_regular',
        mode='exam',
        stage_type='ProduceStepType_AuditionFinal',
        dearness_level=0,
    ),
    AutopilotStage(
        name='stage_003_nia_mid_exam',
        scenario='nia_pro',
        mode='exam',
        stage_type='ProduceStepType_AuditionMid1',
        dearness_level=20,
    ),
    AutopilotStage(
        name='stage_004_nia_final_exam',
        scenario='nia_pro',
        mode='exam',
        stage_type='ProduceStepType_AuditionMid2',
        dearness_level=20,
    ),
    AutopilotStage(
        name='stage_005_nia_selection_exam',
        scenario='nia_pro',
        mode='exam',
        stage_type='ProduceStepType_AuditionFinal',
        dearness_level=20,
    ),
    AutopilotStage(
        name='stage_006_first_star_regular_full',
        scenario='first_star_regular',
        mode='planning',
        dearness_level=0,
    ),
    AutopilotStage(
        name='stage_007_first_star_master_full',
        scenario='first_star_master',
        mode='planning',
        dearness_level=10,
    ),
    AutopilotStage(
        name='stage_008_nia_pro_full',
        scenario='nia_pro',
        mode='planning',
        dearness_level=20,
    ),
    AutopilotStage(
        name='stage_009_nia_master_full',
        scenario='nia_master',
        mode='planning',
        dearness_level=20,
    ),
)

_EXAM_STAGE_TYPES_BY_STAGE_NAME: dict[str, str] = {
    'stage_001_first_star_mid_exam': 'ProduceStepType_AuditionMid1',
    'stage_002_first_star_final_exam': 'ProduceStepType_AuditionFinal',
    'stage_003_nia_mid_exam': 'ProduceStepType_AuditionMid1',
    'stage_004_nia_final_exam': 'ProduceStepType_AuditionMid2',
    'stage_005_nia_selection_exam': 'ProduceStepType_AuditionFinal',
}


def _has_option(args: Sequence[str], option: str) -> bool:
    """判断参数中是否显式传入某个长选项。"""

    return any(value == option or value.startswith(f'{option}=') for value in args)


def parse_autopilot_control_args(argv: Sequence[str]) -> AutopilotControlArgs:
    """解析 autopilot 专属参数，并保留 self-bootstrap 参数。"""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--curriculum', action='store_true', default=False)
    parser.add_argument('--no-curriculum', action='store_true', default=False)
    parser.add_argument('--curriculum-name', default='exam_to_full_produce')
    parser.add_argument('--curriculum-start-stage', type=int, default=1)
    parser.add_argument('--curriculum-stage', action='append', default=[])
    parsed, bootstrap_args = parser.parse_known_args(list(argv))
    return AutopilotControlArgs(
        curriculum=bool(parsed.curriculum),
        no_curriculum=bool(parsed.no_curriculum),
        curriculum_name=str(parsed.curriculum_name),
        curriculum_start_stage=max(int(parsed.curriculum_start_stage), 1),
        curriculum_stage_specs=tuple(str(value) for value in parsed.curriculum_stage),
        bootstrap_args=tuple(str(value) for value in bootstrap_args),
    )


def should_run_curriculum(control_args: AutopilotControlArgs) -> bool:
    """判断当前 autopilot 调用是否应该启用课程模式。"""

    if control_args.no_curriculum:
        return False
    if control_args.curriculum:
        return True
    return not _has_option(control_args.bootstrap_args, '--scenario')


def _parse_custom_stage(spec: str, index: int) -> AutopilotStage:
    """解析用户传入的课程阶段描述。"""

    if '=' in spec:
        name, scenario = spec.split('=', 1)
    elif ':' in spec:
        name, scenario = spec.split(':', 1)
    else:
        scenario = spec
        name = f'stage_{index:03d}_{scenario}'
    name = name.strip()
    scenario = scenario.strip()
    if not name or not scenario:
        raise ValueError(f'Invalid --curriculum-stage value: {spec}')
    dearness_level = 20 if scenario.startswith('nia_') else (10 if scenario == 'first_star_master' else 0)
    return AutopilotStage(name=name, scenario=scenario, dearness_level=dearness_level)


def resolve_curriculum_stages(stage_specs: Sequence[str]) -> tuple[AutopilotStage, ...]:
    """返回本次 autopilot 要执行的课程阶段。"""

    if not stage_specs:
        return DEFAULT_CURRICULUM_STAGES
    return tuple(_parse_custom_stage(spec, index + 1) for index, spec in enumerate(stage_specs))


def parse_bootstrap_namespace(args: Sequence[str]) -> argparse.Namespace:
    """用 self-bootstrap 的解析器解析透传参数。"""

    prior_argv = sys.argv
    sys.argv = ['gakumas-rl-bootstrap', *args]
    try:
        parsed = parse_bootstrap_args()
    finally:
        sys.argv = prior_argv
    parsed.autopilot = True
    return parsed


def _env_signature(env_config: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """返回可用于判断 checkpoint 是否能跨阶段加载的观测签名。"""

    preflight = preflight_env(env_config)
    return (
        tuple(int(value) for value in preflight['global_shape']),
        tuple(int(value) for value in preflight['action_features_shape']),
        tuple(int(value) for value in preflight['action_mask_shape']),
    )


def _stage_args(
    *,
    base_args: argparse.Namespace,
    stage: AutopilotStage,
    stage_run_dir: Path,
    input_checkpoint: Path | None,
    explicit_mode: bool,
    explicit_lesson_level: bool,
) -> argparse.Namespace:
    """基于基础参数生成某个课程阶段的 self-bootstrap 参数。"""

    resolved = argparse.Namespace(**vars(base_args))
    resolved.scenario = stage.scenario
    if explicit_mode:
        resolved.mode = str(base_args.mode)
        resolved.stage_type = vars(base_args).get('stage_type') if resolved.mode == 'exam' else None
    else:
        # 默认 curriculum 阶段定义是权威来源；全自动流程不应被全局 --mode/--stage-type 带偏。
        resolved.mode = stage.mode
        resolved.stage_type = stage.stage_type
    if not explicit_lesson_level or stage.mode != 'lesson':
        resolved.lesson_level_index = stage.lesson_level_index
    resolved.producer_level = max(int(resolved.producer_level), int(stage.producer_level))
    resolved.dearness_level = max(int(resolved.dearness_level), int(stage.dearness_level))
    resolved.run_dir = str(stage_run_dir)
    resolved.initial_checkpoint = str(input_checkpoint) if input_checkpoint is not None else ''
    return resolved


def _planning_exam_checkpoint_map(
    *,
    stage: AutopilotStage,
    stage_summaries: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """为完整培育阶段整理可复用的考试 checkpoint 映射。"""

    if stage.mode != 'planning':
        return {}

    relevant_stage_names: tuple[str, ...]
    if stage.scenario.startswith('nia_'):
        relevant_stage_names = (
            'stage_003_nia_mid_exam',
            'stage_004_nia_final_exam',
            'stage_005_nia_selection_exam',
        )
    else:
        relevant_stage_names = (
            'stage_001_first_star_mid_exam',
            'stage_002_first_star_final_exam',
        )

    checkpoint_map: dict[str, str] = {}
    for item in stage_summaries:
        stage_name = str(item.get('name') or '')
        if stage_name not in relevant_stage_names:
            continue
        checkpoint_path = str(item.get('recommended_checkpoint') or '')
        stage_type = _EXAM_STAGE_TYPES_BY_STAGE_NAME.get(stage_name)
        if checkpoint_path and stage_type:
            checkpoint_map[stage_type] = checkpoint_path
    return checkpoint_map


def run_curriculum(control_args: AutopilotControlArgs) -> dict[str, Any]:
    """按从低难到高难的剧本顺序执行全自动自举训练。"""

    base_args = parse_bootstrap_namespace(control_args.bootstrap_args)
    resolved_stages = resolve_curriculum_stages(control_args.curriculum_stage_specs)
    stages = resolved_stages[control_args.curriculum_start_stage - 1 :]
    if not stages:
        raise ValueError(
            f'--curriculum-start-stage={control_args.curriculum_start_stage} exceeds stage count {len(resolved_stages)}.'
        )
    explicit_mode = bool(control_args.curriculum_stage_specs) and _has_option(control_args.bootstrap_args, '--mode')
    explicit_lesson_level = _has_option(control_args.bootstrap_args, '--lesson-level-index')
    explicit_final_rl = _has_option(control_args.bootstrap_args, '--final-rl-timesteps')
    if not explicit_final_rl and int(base_args.final_rl_timesteps) <= 0:
        base_args.final_rl_timesteps = min(20_000, max(1_000, int(base_args.rl_timesteps) // 8))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_run_dir = Path(base_args.run_dir) if str(base_args.run_dir or '') else RUNS_DIR / 'autopilot_curriculum' / timestamp
    base_run_dir.mkdir(parents=True, exist_ok=True)

    current_checkpoint: Path | None = Path(base_args.initial_checkpoint) if str(base_args.initial_checkpoint or '') else None
    current_signature: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None
    stage_summaries: list[dict[str, Any]] = []

    for stage in stages:
        stage_dir = base_run_dir / stage.name
        print(
            f'[Autopilot] stage={stage.name} mode={stage.mode} '
            f'scenario={stage.scenario} stage_type={stage.stage_type or "-"}',
            flush=True,
        )
        preview_args = _stage_args(
            base_args=base_args,
            stage=stage,
            stage_run_dir=stage_dir,
            input_checkpoint=None,
            explicit_mode=explicit_mode,
            explicit_lesson_level=explicit_lesson_level,
        )
        stage_signature = _env_signature(_build_env_config(preview_args))
        transfer_status = 'none'
        input_checkpoint = current_checkpoint
        if current_checkpoint is not None:
            if current_signature is None:
                transfer_status = 'user_initial'
            elif current_signature == stage_signature:
                transfer_status = 'compatible'
            else:
                input_checkpoint = None
                transfer_status = 'skipped_shape_mismatch'

        actual_args = _stage_args(
            base_args=base_args,
            stage=stage,
            stage_run_dir=stage_dir,
            input_checkpoint=input_checkpoint,
            explicit_mode=explicit_mode,
            explicit_lesson_level=explicit_lesson_level,
        )
        planning_exam_checkpoint_map = _planning_exam_checkpoint_map(
            stage=stage,
            stage_summaries=stage_summaries,
        )
        if planning_exam_checkpoint_map:
            vars(actual_args)['planning_exam_checkpoints'] = planning_exam_checkpoint_map
        summary = run_self_bootstrap(actual_args)
        recommended = summary.get('recommended_checkpoint') or summary.get('final_pretrained_checkpoint')
        current_checkpoint = Path(str(recommended)) if recommended else None
        current_signature = stage_signature
        stage_summaries.append(
            {
                'name': stage.name,
                'scenario': stage.scenario,
                'mode': str(actual_args.mode),
                'stage_type': str(actual_args.stage_type or ''),
                'run_dir': str(stage_dir),
                'input_checkpoint': str(input_checkpoint) if input_checkpoint is not None else None,
                'transfer_status': transfer_status,
                'observation_signature': [list(part) for part in stage_signature],
                'recommended_checkpoint': str(current_checkpoint) if current_checkpoint is not None else None,
                'planning_exam_checkpoints': planning_exam_checkpoint_map or None,
                'summary_path': str(stage_dir / 'bootstrap_summary.json'),
            }
        )

    curriculum_summary = {
        'schema_version': 1,
        'curriculum_name': control_args.curriculum_name,
        'autopilot': True,
        'curriculum_start_stage': control_args.curriculum_start_stage,
        'stage_count': len(stage_summaries),
        'total_stage_count': len(resolved_stages),
        'base_run_dir': str(base_run_dir),
        'stages': stage_summaries,
        'recommended_checkpoint': str(current_checkpoint) if current_checkpoint is not None else None,
    }
    summary_path = base_run_dir / 'curriculum_summary.json'
    summary_path.write_text(json.dumps(curriculum_summary, ensure_ascii=False, indent=2, default=_json_default), encoding='utf-8')
    return curriculum_summary


def run_single_stage(bootstrap_args: Sequence[str]) -> int:
    """保留显式指定场景时的单阶段 autopilot 行为。"""

    prior_argv = sys.argv
    args = list(bootstrap_args)
    if '--autopilot' not in args:
        args.insert(0, '--autopilot')
    sys.argv = ['gakumas-rl-bootstrap', *args]
    try:
        summary = run_self_bootstrap(parse_bootstrap_args())
    finally:
        sys.argv = prior_argv
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """根据参数自动选择课程模式或单阶段自举模式。"""

    raw_args = tuple(sys.argv[1:] if argv is None else argv)
    control_args = parse_autopilot_control_args(raw_args)
    if should_run_curriculum(control_args):
        summary = run_curriculum(control_args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    return run_single_stage(control_args.bootstrap_args)


if __name__ == '__main__':
    raise SystemExit(main())
