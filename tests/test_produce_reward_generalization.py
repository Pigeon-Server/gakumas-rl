"""培育奖励泛化能力的回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.idol_config import build_idol_loadout
from src.interfaces.service import build_env_from_config
from src.repository.master_data import MasterDataRepository
from src.simulation.produce.runtime import (
    NIA_PARAM_TARGET_STAGE_BASE_RATIO,
    NIA_PARAM_TARGET_STAGE_PROGRESS_RATIO,
    ProduceRewardSnapshot,
    ProduceRuntime,
)

HSKI_SSR = 'i_card-hski-3-008'


def _set_parameter_values(runtime: ProduceRuntime, values: tuple[float, float, float]) -> None:
    """设置运行时三维参数，便于隔离比较势函数结果。"""

    runtime.state['vocal'] = float(values[0])
    runtime.state['dance'] = float(values[1])
    runtime.state['visual'] = float(values[2])


def test_param_phi_prefers_three_stat_coverage_over_single_stat_stack() -> None:
    """参数势函数应奖励三维覆盖，而不是只堆单个高权重属性。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=503)
    runtime.reset()

    target = runtime._produce_param_target()
    weights = np.array(runtime._next_param_weights(), dtype=np.float32)
    dominant_index = int(np.argmax(weights))
    weighted_goal = target * 0.75

    balanced_values = (weighted_goal, weighted_goal, weighted_goal)
    lopsided_values = [0.0, 0.0, 0.0]
    lopsided_values[dominant_index] = weighted_goal / max(float(weights[dominant_index]), 1e-6)

    _set_parameter_values(runtime, balanced_values)
    balanced_phi = runtime._phi_param()
    _set_parameter_values(runtime, tuple(lopsided_values))
    lopsided_phi = runtime._phi_param()

    assert balanced_phi > lopsided_phi


def test_nia_param_phi_strongly_penalizes_single_stat_stack() -> None:
    """NIA 参数势函数应显著压低单属性堆叠路线。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=504)
    runtime.reset()

    target = runtime._produce_param_target()
    weights = np.array(runtime._next_param_weights(), dtype=np.float32)
    dominant_index = int(np.argmax(weights))
    weighted_goal = target * 0.75

    balanced_values = (weighted_goal, weighted_goal, weighted_goal)
    lopsided_values = [0.0, 0.0, 0.0]
    lopsided_values[dominant_index] = weighted_goal / max(float(weights[dominant_index]), 1e-6)

    _set_parameter_values(runtime, balanced_values)
    balanced_phi = runtime._phi_param()
    _set_parameter_values(runtime, tuple(lopsided_values))
    lopsided_phi = runtime._phi_param()

    assert balanced_phi - lopsided_phi > 0.10


def test_param_target_uses_stage_growth_floor() -> None:
    """参数目标线应包含阶段成长下限，避免初始参数直接让势函数饱和。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=505)
    runtime.reset()

    stage_ratio = 1.0 / max(len(scenario.audition_sequence), 1)
    expected_floor = scenario.parameter_growth_limit * (0.45 + 0.35 * stage_ratio)

    assert runtime._produce_param_target() + 1e-6 >= expected_floor


def test_nia_param_target_uses_route_growth_floor() -> None:
    """NIA 参数目标线应按多阶段试镜提高，避免只满足低基准线。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=506)
    runtime.reset()

    stage_ratio = 1.0 / max(len(scenario.audition_sequence), 1)
    expected_floor = scenario.parameter_growth_limit * (
        NIA_PARAM_TARGET_STAGE_BASE_RATIO + NIA_PARAM_TARGET_STAGE_PROGRESS_RATIO * stage_ratio
    )

    assert runtime._produce_param_target() == pytest.approx(expected_floor)


def test_nia_param_weights_follow_loadout_audition_profile() -> None:
    """NIA 参数势函数应按当前偶像卡的试镜配置计算属性权重。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(
        repository,
        scenario,
        HSKI_SSR,
        producer_level=35,
        idol_rank=4,
        dearness_level=20,
    )
    runtime = ProduceRuntime(repository, scenario, seed=527, idol_loadout=loadout)
    runtime.reset()

    vocal_weight, dance_weight, visual_weight = runtime._next_param_weights()

    assert visual_weight > dance_weight > vocal_weight


def test_nia_fan_value_depends_on_parameter_readiness() -> None:
    """NIA 粉丝票价值应受参数准备度折扣，避免模型只刷票。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=507)
    runtime.reset()
    runtime.state['fan_votes'] = 4000.0

    _set_parameter_values(runtime, (100.0, 100.0, 100.0))
    low_param_fan_phi = runtime._phi_fan()
    _set_parameter_values(runtime, (1800.0, 1800.0, 1800.0))
    high_param_fan_phi = runtime._phi_fan()

    assert high_param_fan_phi > low_param_fan_phi


def test_nia_fan_phi_ignores_surplus_votes_after_unlock() -> None:
    """NIA 粉丝势函数只表达解锁价值，不奖励大量溢出票。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=517)
    runtime.reset()
    _set_parameter_values(runtime, (1800.0, 1800.0, 1800.0))
    threshold = runtime._next_fan_vote_threshold()

    runtime.state['fan_votes'] = threshold * 100.0
    unlocked_phi = runtime._phi_fan()
    runtime.state['fan_votes'] = threshold * 1000.0
    surplus_phi = runtime._phi_fan()

    assert surplus_phi == pytest.approx(unlocked_phi)


def test_nia_state_delta_penalizes_raw_surplus_fan_votes() -> None:
    """参数短板明显时，即使 fan 势函数饱和，原始溢出票增长也应扣分。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=518)

    before = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.50,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
        fan_votes=1000.0,
        next_fan_vote_threshold=1000.0,
    )
    surplus_after = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.50,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
        fan_votes=50000.0,
        next_fan_vote_threshold=0.0,
    )

    assert runtime._state_delta_bonus(before, surplus_after) < -0.05


def test_nia_state_delta_penalizes_fan_only_progress() -> None:
    """NIA 只涨票不涨参数时，应弱于真实参数进展。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=508)

    before = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.10,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
    )
    fan_only_after = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.20,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
    )
    param_after = ProduceRewardSnapshot(
        param_phi=0.34,
        fan_phi=0.10,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.32,
        param_weakest_phi=0.24,
    )

    assert runtime._state_delta_bonus(before, fan_only_after) < runtime._state_delta_bonus(before, param_after)


def test_state_delta_bonus_depends_on_snapshot_delta_only() -> None:
    """状态增量奖励应只由前后状态差决定，不能隐含动作类型偏好。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=509)

    before = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.20,
        resource_phi=0.40,
        stamina_ratio=0.50,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
    )
    after = ProduceRewardSnapshot(
        param_phi=0.45,
        fan_phi=0.25,
        resource_phi=0.46,
        stamina_ratio=0.55,
        param_coverage_phi=0.40,
        param_weakest_phi=0.32,
    )

    first_bonus = runtime._state_delta_bonus(before, after)
    second_bonus = runtime._state_delta_bonus(before, after)

    assert first_bonus == pytest.approx(second_bonus)
    assert first_bonus > 0.0


def test_state_delta_bonus_rewards_weakest_parameter_progress() -> None:
    """弱项参数进展应获得额外即时信号，减少模型只堆单项参数。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=510)

    before = ProduceRewardSnapshot(
        param_phi=0.40,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.50,
        param_coverage_phi=0.38,
        param_weakest_phi=0.20,
    )
    after = ProduceRewardSnapshot(
        param_phi=0.43,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.50,
        param_coverage_phi=0.43,
        param_weakest_phi=0.35,
    )

    assert runtime._state_delta_bonus(before, after) > 0.0


def test_state_delta_bonus_penalizes_lopsided_parameter_progress() -> None:
    """只提升优势项而弱项无进展时，应扣除偏科收益。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=512)

    before = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.50,
        param_coverage_phi=0.30,
        param_weakest_phi=0.20,
    )
    balanced_after = ProduceRewardSnapshot(
        param_phi=0.34,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.50,
        param_coverage_phi=0.34,
        param_weakest_phi=0.25,
    )
    lopsided_after = ProduceRewardSnapshot(
        param_phi=0.34,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.50,
        param_coverage_phi=0.30,
        param_weakest_phi=0.20,
    )

    assert runtime._state_delta_bonus(before, balanced_after) > runtime._state_delta_bonus(before, lopsided_after)
    assert runtime._state_delta_bonus(before, lopsided_after) < 0.0


def test_nia_state_delta_strongly_penalizes_stalled_weakest_progress() -> None:
    """NIA 参数成长不补弱项时，应给出足够强的负向信号。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=514)

    before = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.60,
        param_coverage_phi=0.30,
        param_weakest_phi=0.20,
    )
    lopsided_after = ProduceRewardSnapshot(
        param_phi=0.34,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.50,
        param_coverage_phi=0.30,
        param_weakest_phi=0.20,
    )

    assert runtime._state_delta_bonus(before, lopsided_after) < -0.10


def test_state_delta_bonus_penalizes_unproductive_high_stamina_recovery() -> None:
    """高体力时只回体且无进展应被视作低价值操作。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=511)

    before = ProduceRewardSnapshot(param_phi=0.40, fan_phi=0.20, resource_phi=0.30, stamina_ratio=0.70)
    after = ProduceRewardSnapshot(param_phi=0.40, fan_phi=0.20, resource_phi=0.30, stamina_ratio=0.90)

    assert runtime._state_delta_bonus(before, after) < 0.0


def test_state_delta_bonus_rewards_low_stamina_recovery() -> None:
    """低体力回体仍应保留正向即时信号。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=513)

    before = ProduceRewardSnapshot(param_phi=0.40, fan_phi=0.20, resource_phi=0.30, stamina_ratio=0.15)
    after = ProduceRewardSnapshot(param_phi=0.40, fan_phi=0.20, resource_phi=0.30, stamina_ratio=0.45)

    assert runtime._state_delta_bonus(before, after) > 0.0


def test_state_delta_bonus_penalizes_no_progress_snapshot() -> None:
    """完全无进展的状态转移应被扣分，避免模型刷空动作。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=515)

    snapshot = ProduceRewardSnapshot(param_phi=0.40, fan_phi=0.0, resource_phi=0.30, stamina_ratio=0.50)

    assert runtime._state_delta_bonus(snapshot, snapshot) < 0.0


def test_state_delta_bonus_penalizes_high_stamina_idle_more() -> None:
    """高体力空转应比中体力空转扣分更重，避免刷安全休息。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=516)

    middle = ProduceRewardSnapshot(param_phi=0.95, fan_phi=0.0, resource_phi=0.30, stamina_ratio=0.25)
    high = ProduceRewardSnapshot(param_phi=0.95, fan_phi=0.0, resource_phi=0.30, stamina_ratio=0.95)

    assert runtime._state_delta_bonus(high, high) < runtime._state_delta_bonus(middle, middle)


def test_normal_lesson_rows_fallback_to_parameter_gain() -> None:
    """普通课主数据行没有直接效果时，应回落为课程参数收益。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=820)
    runtime.reset()

    normal_lesson = next(
        action
        for action in runtime.legal_actions()
        if action.action_type == 'lesson_vocal_normal' and action.available
    )

    assert normal_lesson.stat_deltas[0] > 0.0
    assert normal_lesson.stamina_delta < 0.0
    assert normal_lesson.produce_point_delta > 0.0


def test_deck_quality_uses_bounded_card_prior_scale() -> None:
    """卡组质量应保留区分度，不能被客户端大数估值直接打满。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=820)
    runtime.reset()

    assert 0.0 <= runtime.state['deck_quality'] < 20.0
    assert runtime._phi_resource() < 1.0


def test_first_star_hard_lesson_only_available_before_audition() -> None:
    """初路线追込课只应在考试前指定周出现，避免模型刷固定 hard lesson。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=820)
    runtime.reset()

    initial_actions = {action.action_type: action.available for action in runtime.legal_actions()}

    assert not initial_actions['lesson_vocal_hard']
    assert not initial_actions['lesson_dance_hard']
    assert not initial_actions['lesson_visual_hard']

    first_checkpoint_step, _stage_type = runtime.checkpoints[0]
    runtime.state['step'] = first_checkpoint_step - 2
    hard_week_actions = {action.action_type: action.available for action in runtime.legal_actions()}

    assert hard_week_actions['lesson_vocal_hard']
    assert hard_week_actions['lesson_dance_hard']
    assert hard_week_actions['lesson_visual_hard']
    assert not hard_week_actions['lesson_vocal_normal']

    runtime.state['step'] = first_checkpoint_step - 1
    refresh_week_actions = {action.action_type: action.available for action in runtime.legal_actions()}

    assert refresh_week_actions['refresh']
    assert not refresh_week_actions.get('lesson_vocal_hard', False)


def test_non_full_stamina_refresh_is_available_outside_forced_refresh_week() -> None:
    """初路线普通休息在体力未满时可选，考前强制恢复周满体力也保持可用。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=821)
    runtime.reset()

    initial_actions = {action.action_type: action.available for action in runtime.legal_actions()}
    assert not initial_actions['refresh']

    runtime.state['stamina'] = runtime.state['max_stamina'] * 0.50
    middle_stamina_actions = {action.action_type: action.available for action in runtime.legal_actions()}
    assert middle_stamina_actions['refresh']

    runtime.state['stamina'] = runtime.state['max_stamina'] * 0.25
    low_stamina_actions = {action.action_type: action.available for action in runtime.legal_actions()}
    assert low_stamina_actions['refresh']

    first_checkpoint_step, _stage_type = runtime.checkpoints[0]
    runtime.state['step'] = first_checkpoint_step - 1
    runtime.state['stamina'] = runtime.state['max_stamina']
    refresh_week_actions = runtime.legal_actions()

    assert len(refresh_week_actions) == 1
    assert refresh_week_actions[0].action_type == 'refresh'
    assert refresh_week_actions[0].available


def test_negative_produce_point_action_requires_enough_points() -> None:
    """消耗 P 点的 weekly 动作不能在余额不足时执行。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=823)
    runtime.reset()
    runtime.state['produce_points'] = 0.0
    outing = runtime._sample_action('outing')

    assert outing.produce_point_delta < 0.0
    assert not runtime._action_available(outing)

    runtime.state['produce_points'] = abs(outing.produce_point_delta)

    assert runtime._action_available(outing)


def test_first_star_stamina_readiness_tracks_hard_lesson_runway() -> None:
    """追込课前体力准备度应随体力不足下降。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=824)
    runtime.reset()

    assert runtime._stamina_readiness_phi() == pytest.approx(1.0)

    runtime.state['stamina'] = runtime._first_star_hard_lesson_stamina_cost() * 0.5

    assert runtime._stamina_readiness_phi() < 1.0


def test_state_delta_bonus_penalizes_stamina_readiness_drop() -> None:
    """追込课前消耗过多体力应降低即时奖励。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=825)

    before = ProduceRewardSnapshot(
        param_phi=0.40,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.80,
        param_coverage_phi=0.40,
        param_weakest_phi=0.35,
        stamina_readiness_phi=1.0,
    )
    safe_after = ProduceRewardSnapshot(
        param_phi=0.43,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.43,
        param_weakest_phi=0.38,
        stamina_readiness_phi=1.0,
    )
    risky_after = ProduceRewardSnapshot(
        param_phi=0.43,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.43,
        param_weakest_phi=0.38,
        stamina_readiness_phi=0.20,
    )

    assert runtime._state_delta_bonus(before, safe_after) > runtime._state_delta_bonus(before, risky_after)


def test_candidate_parameter_delta_estimates_direct_effects() -> None:
    """候选收益预估应读取直接参数 ProduceEffect，供动作特征使用。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=820)
    runtime.reset()

    sp_lesson = next(
        action
        for action in runtime.legal_actions()
        if action.action_type == 'lesson_dance_sp' and action.available
    )
    estimated_deltas = runtime.estimate_candidate_parameter_deltas(sp_lesson)

    assert estimated_deltas[1] > 0.0


def test_stage_repeat_counts_penalize_unhelpful_lesson_spam() -> None:
    """同一阶段内重复课程且不补弱项时，应暴露风险并给出惩罚。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=826)
    runtime.reset()
    parameter_limit = float(scenario.parameter_growth_limit or 1000.0)
    _set_parameter_values(runtime, (80.0, parameter_limit, 80.0))

    sp_lesson = next(
        action
        for action in runtime.legal_actions()
        if action.action_type == 'lesson_dance_sp' and action.available
    )
    runtime._record_stage_action(sp_lesson)
    runtime._record_stage_action(sp_lesson)

    assert runtime.stage_action_repeat_counts(sp_lesson) == (2, 2, 2)
    assert runtime.estimate_candidate_repeat_risk(sp_lesson) > 0.0

    before = ProduceRewardSnapshot(
        param_phi=0.40,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.30,
        param_weakest_phi=0.20,
        stamina_readiness_phi=1.0,
    )
    unhelpful_after = ProduceRewardSnapshot(
        param_phi=0.41,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.60,
        param_coverage_phi=0.301,
        param_weakest_phi=0.201,
        stamina_readiness_phi=1.0,
    )
    helpful_after = ProduceRewardSnapshot(
        param_phi=0.41,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.60,
        param_coverage_phi=0.305,
        param_weakest_phi=0.205,
        stamina_readiness_phi=1.0,
    )

    assert runtime._stage_repetition_penalty(sp_lesson, before, unhelpful_after) < -0.03
    assert runtime._stage_repetition_penalty(sp_lesson, before, helpful_after) == pytest.approx(0.0)

    runtime._reset_stage_action_counts()
    assert runtime.stage_action_repeat_counts(sp_lesson) == (0, 0, 0)


def test_nia_repeated_lesson_requires_weakest_progress() -> None:
    """NIA 重复课程即使提升覆盖率，只要不补弱项也应扣分。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=520)
    runtime.reset()
    _set_parameter_values(runtime, (450.0, 900.0, 450.0))

    dance_lesson = next(
        action
        for action in runtime.legal_actions()
        if action.action_type == 'self_lesson_dance_normal' and action.available
    )
    runtime._record_stage_action(dance_lesson)
    runtime._record_stage_action(dance_lesson)

    before = ProduceRewardSnapshot(
        param_phi=0.25,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
    )
    coverage_only_after = ProduceRewardSnapshot(
        param_phi=0.27,
        fan_phi=0.0,
        resource_phi=0.30,
        stamina_ratio=0.60,
        param_coverage_phi=0.31,
        param_weakest_phi=0.20,
    )

    assert runtime.estimate_candidate_repeat_risk(dance_lesson) > 0.0
    assert runtime._stage_repetition_penalty(dance_lesson, before, coverage_only_after) < -0.08


def test_nia_idle_event_repeat_penalizes_business_spam_after_votes_enough() -> None:
    """票数已足但参数不足时，重复无参数事件应暴露风险并扣分。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=519)
    runtime.reset()
    _set_parameter_values(runtime, (450.0, 450.0, 450.0))
    runtime.state['fan_votes'] = runtime._next_fan_vote_threshold() * 2.0

    business = next(action for action in runtime.legal_actions() if action.action_type == 'business' and action.available)
    runtime._record_stage_action(business)
    runtime._record_stage_action(business)

    before = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.60,
        resource_phi=0.30,
        stamina_ratio=0.70,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
        fan_votes=2000.0,
        next_fan_vote_threshold=1000.0,
    )
    after = ProduceRewardSnapshot(
        param_phi=0.30,
        fan_phi=0.60,
        resource_phi=0.34,
        stamina_ratio=0.66,
        param_coverage_phi=0.28,
        param_weakest_phi=0.20,
        fan_votes=5000.0,
        next_fan_vote_threshold=0.0,
    )

    assert runtime.estimate_candidate_idle_event_risk(business) > 0.0
    assert runtime._stage_idle_event_penalty(business, before, after) < -0.03


def test_planning_reward_breakdown_uses_state_delta_bonus() -> None:
    """planning 环境应报告状态增量奖励，并保留旧动作名奖励为零。"""

    env = build_env_from_config({'mode': 'planning', 'scenario': 'first_star_regular', 'seed': 521})
    try:
        obs, _info = env.reset(seed=521)
        valid_actions = np.flatnonzero(obs['action_mask'] > 0.5)

        assert valid_actions.size > 0

        _next_obs, _reward, _terminated, _truncated, info = env.step(int(valid_actions[0]))
        breakdown = info['reward_breakdown']

        assert 'state_delta_bonus' in breakdown
        assert 'stage_repetition_penalty' in breakdown
        assert 'idle_event_repeat_penalty' in breakdown
        assert breakdown['action_doc_bonus'] == pytest.approx(0.0)
    finally:
        env.close()
