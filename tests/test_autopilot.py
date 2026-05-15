"""autopilot 课程编排的轻量测试。"""

from __future__ import annotations

from pathlib import Path

from src.loadout import DEFAULT_DEARNESS_LEVEL
from src.training.autopilot import (
    DEFAULT_CURRICULUM_STAGES,
    _planning_exam_checkpoint_map,
    _stage_args,
    parse_autopilot_control_args,
    parse_bootstrap_namespace,
    resolve_curriculum_stages,
    should_run_curriculum,
)


def test_autopilot_defaults_to_curriculum_without_explicit_scenario() -> None:
    """未显式指定 scenario 时，autopilot 应默认启用考试到全流程课程。"""

    control_args = parse_autopilot_control_args(['--mode', 'lesson'])

    assert should_run_curriculum(control_args)
    assert [stage.mode for stage in DEFAULT_CURRICULUM_STAGES] == [
        'exam',
        'exam',
        'exam',
        'exam',
        'exam',
        'planning',
        'planning',
        'planning',
        'planning',
    ]
    assert DEFAULT_CURRICULUM_STAGES[0].scenario == 'first_star_regular'
    assert DEFAULT_CURRICULUM_STAGES[0].stage_type == 'ProduceStepType_AuditionMid1'
    assert DEFAULT_CURRICULUM_STAGES[-1].scenario == 'nia_master'


def test_autopilot_keeps_single_stage_when_scenario_is_explicit() -> None:
    """显式指定 scenario 时，autopilot 应保持单阶段兼容行为。"""

    control_args = parse_autopilot_control_args(['--scenario', 'nia_master'])

    assert not should_run_curriculum(control_args)


def test_autopilot_no_curriculum_overrides_default_curriculum() -> None:
    """用户传入 no-curriculum 时，即使没有 scenario 也不应启用课程。"""

    control_args = parse_autopilot_control_args(['--no-curriculum', '--mode', 'lesson'])

    assert not should_run_curriculum(control_args)


def test_autopilot_parses_curriculum_start_stage() -> None:
    """课程起始阶段用于从失败阶段继续跑，避免重跑已经完成的前置阶段。"""

    control_args = parse_autopilot_control_args(['--curriculum-start-stage', '6'])

    assert control_args.curriculum_start_stage == 6


def test_custom_curriculum_stage_specs_are_supported() -> None:
    """自定义课程阶段应支持 name=scenario 和裸 scenario 两种写法。"""

    stages = resolve_curriculum_stages(['warmup=first_star_regular', 'nia_master'])

    assert [stage.name for stage in stages] == ['warmup', 'stage_002_nia_master']
    assert [stage.scenario for stage in stages] == ['first_star_regular', 'nia_master']
    assert stages[1].dearness_level == 20


def test_stage_args_apply_default_dearness_and_stage_directory() -> None:
    """阶段参数应保留全局默认亲爱度并设置独立 run_dir。"""

    base_args = parse_bootstrap_namespace(['--iterations', '1', '--mode', 'lesson'])
    stage = DEFAULT_CURRICULUM_STAGES[6]
    resolved = _stage_args(
        base_args=base_args,
        stage=stage,
        stage_run_dir=Path('runs/test_curriculum/stage_007'),
        input_checkpoint=None,
        explicit_mode=False,
        explicit_lesson_level=False,
    )

    assert resolved.scenario == 'first_star_master'
    assert resolved.mode == 'planning'
    assert resolved.stage_type is None
    assert resolved.dearness_level == DEFAULT_DEARNESS_LEVEL
    assert resolved.lesson_level_index == 2
    assert resolved.run_dir == 'runs/test_curriculum/stage_007'


def test_stage_args_apply_exam_stage_type() -> None:
    """考试局部阶段应把 stage_type 透传给 self-bootstrap。"""

    base_args = parse_bootstrap_namespace(['--iterations', '1'])
    stage = DEFAULT_CURRICULUM_STAGES[4]
    resolved = _stage_args(
        base_args=base_args,
        stage=stage,
        stage_run_dir=Path('runs/test_curriculum/stage_005'),
        input_checkpoint=None,
        explicit_mode=False,
        explicit_lesson_level=False,
    )

    assert resolved.scenario == 'nia_pro'
    assert resolved.mode == 'exam'
    assert resolved.stage_type == 'ProduceStepType_AuditionFinal'
    assert resolved.dearness_level == 20


def test_planning_stage_reuses_prior_exam_checkpoints() -> None:
    """完整培育阶段应能收到前置考试阶段的推荐 checkpoint 映射。"""

    checkpoint_map = _planning_exam_checkpoint_map(
        stage=DEFAULT_CURRICULUM_STAGES[7],
        stage_summaries=[
            {
                'name': 'stage_003_nia_mid_exam',
                'recommended_checkpoint': 'runs/test/stage_003/best.zip',
            },
            {
                'name': 'stage_004_nia_final_exam',
                'recommended_checkpoint': 'runs/test/stage_004/best.zip',
            },
            {
                'name': 'stage_005_nia_selection_exam',
                'recommended_checkpoint': 'runs/test/stage_005/best.zip',
            },
        ],
    )

    assert checkpoint_map == {
        'ProduceStepType_AuditionMid1': 'runs/test/stage_003/best.zip',
        'ProduceStepType_AuditionMid2': 'runs/test/stage_004/best.zip',
        'ProduceStepType_AuditionFinal': 'runs/test/stage_005/best.zip',
    }
