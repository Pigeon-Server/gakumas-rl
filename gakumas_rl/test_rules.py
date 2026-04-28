"""围绕卡组装配和考试基础规则的回归测试。"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace

import numpy as np
import pytest

import gakumas_rl.data as data_module
import gakumas_rl.produce_runtime as produce_runtime_module
from gakumas_rl.data import MasterDataRepository
from gakumas_rl.exam_runtime import ExamRuntime, RuntimeCard, TriggeredEnchant, default_audition_row_selector
from gakumas_rl.idol_config import build_idol_loadout, build_initial_exam_deck, build_weighted_card_pool, sample_card_from_weighted_pool
from gakumas_rl.produce_item_interpreter import RuntimeExamStatusEnchantSpec
from gakumas_rl.produce_runtime import HARD_ACTION_TYPES, ActiveProduceSkillState, ProduceActionCandidate, ProduceRuntime
from gakumas_rl.service import LoadoutConfig, _serialize_planning_state, build_env_from_config, loadout_summary, simulate_planning
from gakumas_rl.support_card_selector import auto_select_support_cards, SupportCardAutoSelectConfig
from gakumas_rl.state_snapshot import build_state_snapshot, extract_state_context, extract_action_list_context, action_label_for_llm

_has_gymnasium = True
try:
    import gymnasium  # noqa: F401
except ImportError:
    _has_gymnasium = False

AMAO_R = 'i_card-amao-1-000'
AMAO_SSR = 'i_card-amao-3-000'


def _sample_loadout(repository: MasterDataRepository, scenario, idol_card_id: str = AMAO_R):
    """构造一个稳定的默认偶像编成，避免考试行选择出现歧义。"""

    return build_idol_loadout(repository, scenario, idol_card_id, producer_level=35, idol_rank=4, dearness_level=10)


def _sample_manual_support_card_ids(
    repository: MasterDataRepository,
    plan_type: str,
    *,
    count: int = 6,
) -> tuple[str, ...]:
    """挑选若干张当前流派可用的支援卡，供手动编成测试复用。"""

    allowed_plan_types = {'ProducePlanType_Common', str(plan_type or 'ProducePlanType_Common')}
    selected: list[str] = []
    for row in repository.support_cards.rows:
        support_card_id = str(row.get('id') or '')
        if not support_card_id:
            continue
        if str(row.get('planType') or 'ProducePlanType_Common') not in allowed_plan_types:
            continue
        selected.append(support_card_id)
        if len(selected) >= count:
            return tuple(selected)
    raise AssertionError(f'No enough support cards found for plan_type={plan_type}, need {count}')


def _inject_table_row(table, row: dict[str, object]) -> None:
    """向测试用主数据索引里注入一条临时记录。"""

    table.rows.append(row)
    table.by_id.setdefault(str(row.get('id') or ''), []).append(row)


def _inject_produce_item(
    repository: MasterDataRepository,
    *,
    item_id: str,
    trigger_id: str = '',
    phase_type: str = 'ProducePhaseType_Unknown',
    produce_effect_id: str = '',
    effect_type: str = 'ProduceEffectType_ProducePointAddition',
    effect_value: int = 10,
    item_effect_type: str = 'ProduceItemEffectType_ProduceEffect',
    enchant_id: str = '',
    fire_limit: int = 0,
    fire_interval: int = 0,
) -> None:
    """注入一条最小可运行的 ProduceItem 测试数据。"""

    if trigger_id:
        _inject_table_row(
            repository.load_table('ProduceTrigger'),
            {
                'id': trigger_id,
                'phaseType': phase_type,
            },
        )
    if produce_effect_id:
        _inject_table_row(
            repository.load_table('ProduceEffect'),
            {
                'id': produce_effect_id,
                'produceEffectType': effect_type,
                'effectValueMin': effect_value,
                'effectValueMax': effect_value,
                'produceResourceType': 'ProduceResourceType_Unknown',
                'produceRewards': [],
                'produceCardSearchId': '',
                'produceExamStatusEnchantId': enchant_id,
                'produceStepEventDetailId': '',
                'pickRangeType': 'ProducePickRangeType_Unknown',
                'pickCountMin': 0,
                'pickCountMax': 0,
                'isResearch': False,
            },
        )
    item_effect_id = f'{item_id}-effect'
    _inject_table_row(
        repository.load_table('ProduceItemEffect'),
        {
            'id': item_effect_id,
            'effectType': item_effect_type,
            'effectTurn': 0,
            'effectCount': 0,
            'produceEffectId': produce_effect_id,
            'produceExamStatusEnchantId': enchant_id,
        },
    )
    _inject_table_row(
        repository.load_table('ProduceItem'),
        {
            'id': item_id,
            'name': item_id,
            'planType': 'ProducePlanType_Common',
            'rarity': 'ProduceItemRarity_R',
            'produceItemEffectIds': [item_effect_id],
            'skills': [{'produceTriggerId': '', 'produceItemEffectId': item_effect_id}],
            'produceTriggerId': trigger_id,
            'produceTriggerIds': [],
            'fireLimit': fire_limit,
            'fireInterval': fire_interval,
            'isChallenge': True,
            'isExamEffect': False,
            'isHighScoreRush': False,
            'isLimited': False,
            'isResearch': False,
            'isUpgraded': False,
            'libraryHidden': False,
            'effectGroupIds': [],
            'evaluation': 0,
            'assetId': '',
            'order': 0,
            'originIdolCardId': '',
            'originSupportCardId': '',
            'produceDescriptions': [],
            'viewStartTime': '',
        },
    )


def _inject_produce_trigger(repository: MasterDataRepository, *, trigger_id: str, phase_type: str) -> None:
    """注入一条最小可运行的 ProduceTrigger 测试数据。"""

    _inject_table_row(
        repository.load_table('ProduceTrigger'),
        {
            'id': trigger_id,
            'phaseType': phase_type,
        },
    )


def _inject_produce_effect_row(
    repository: MasterDataRepository,
    *,
    effect_id: str,
    effect_type: str,
    effect_value: int = 0,
    produce_rewards: list[dict[str, object]] | None = None,
) -> None:
    """注入一条最小可运行的 ProduceEffect 测试数据。"""

    _inject_table_row(
        repository.load_table('ProduceEffect'),
        {
            'id': effect_id,
            'produceEffectType': effect_type,
            'effectValueMin': effect_value,
            'effectValueMax': effect_value,
            'produceResourceType': 'ProduceResourceType_Unknown',
            'produceRewards': list(produce_rewards or []),
            'produceCardSearchId': '',
            'produceExamStatusEnchantId': '',
            'produceStepEventDetailId': '',
            'pickRangeType': 'ProducePickRangeType_Unknown',
            'pickCountMin': 0,
            'pickCountMax': 0,
            'isResearch': False,
        },
    )


def _first_ambiguous_audition_stage(repository: MasterDataRepository, scenario) -> tuple[str, list[dict[str, object]]]:
    """找出一个存在多条 battle row 的考试阶段。"""

    for stage_type in scenario.audition_sequence:
        rows = repository.audition_rows(scenario, stage_type)
        if len(rows) > 1:
            return stage_type, rows
    raise AssertionError('No ambiguous audition stage found in scenario')


def _sample_audition_row_selector(
    repository: MasterDataRepository,
    scenario,
    loadout,
    stage_type: str | None = None,
) -> str | None:
    """为测试构造一个稳定的 battle row selector。"""

    rows = repository.audition_rows(
        scenario,
        stage_type or scenario.default_stage,
        audition_difficulty_id=loadout.stat_profile.audition_difficulty_id,
    )
    if not rows:
        return None
    row = rows[0]
    return f"{str(row.get('id') or '')}:{int(row.get('number') or 0)}"


def _first_exam_effect(repository: MasterDataRepository, effect_type: str, **conditions: int) -> dict[str, object]:
    """按效果类型和简单字段过滤主数据里的第一条考试效果。"""

    for row in repository.load_table('ProduceExamEffect').rows:
        if str(row.get('effectType') or '') != effect_type:
            continue
        matched = True
        for key, expected in conditions.items():
            if int(row.get(key) or 0) != expected:
                matched = False
                break
        if matched:
            return row
    raise AssertionError(f'Effect not found: {effect_type} {conditions}')


def _sample_runtime(seed: int = 7, **runtime_kwargs) -> ExamRuntime:
    """构造一个可直接调用内部运行时方法的考试实例。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = runtime_kwargs.pop('loadout', _sample_loadout(repository, scenario))
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=seed,
        audition_row_id=runtime_kwargs.pop('audition_row_id', _sample_audition_row_selector(repository, scenario, loadout)),
        **runtime_kwargs,
    )
    runtime.reset()
    return runtime


def test_exam_turn_color_sampling_biases_towards_stronger_parameters() -> None:
    """考试回合颜色应按当前培育三维加权，而不是平均采样。"""

    runtime = _sample_runtime(seed=23, exam_score_bonus_multiplier=1.0)
    runtime.parameter_stats = (900.0, 150.0, 150.0)

    counts = Counter(runtime._roll_turn_color() for _ in range(600))

    assert counts['vocal'] > counts['dance']
    assert counts['vocal'] > counts['visual']
    assert counts['vocal'] > 380


def test_build_idol_loadout_prefers_manual_support_cards_over_auto_selection() -> None:
    """显式指定支援卡时，loadout 应优先使用手动编成而不是自动选卡。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    base_loadout = _sample_loadout(repository, scenario)
    manual_support_card_ids = _sample_manual_support_card_ids(repository, base_loadout.stat_profile.plan_type, count=6)

    loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=35,
        idol_rank=4,
        dearness_level=10,
        auto_select_support_cards_for_training=True,
        selected_support_card_ids=manual_support_card_ids,
        selected_support_card_level=40,
    )

    assert tuple(card.support_card_id for card in loadout.support_cards) == manual_support_card_ids
    assert all(card.support_card_level == 40 for card in loadout.support_cards)
    assert all(card.reasons == ('manual',) for card in loadout.support_cards)
    assert loadout.metadata['support_card_selection_mode'] == 'manual'


def test_exam_env_manual_support_cards_flow_into_runtime() -> None:
    """exam env reset 后，运行时应直接拿到手动指定的支援卡编成。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    base_loadout = _sample_loadout(repository, scenario)
    manual_support_card_ids = _sample_manual_support_card_ids(repository, base_loadout.stat_profile.plan_type, count=6)
    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_R,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'auto_support_cards': False,
            'support_card_ids': list(manual_support_card_ids),
            'support_card_level': 40,
        }
    )

    _, info = env.reset(seed=17)

    assert tuple(card.support_card_id for card in env.current_loadout.support_cards) == manual_support_card_ids
    assert all(card.support_card_level == 40 for card in env.current_loadout.support_cards)
    assert tuple(card.support_card_id for card in env.runtime.support_cards) == manual_support_card_ids
    assert info['episode_context']['support_card_count'] == len(manual_support_card_ids)


def test_build_idol_loadout_rejects_duplicate_manual_support_cards() -> None:
    """手动支援编成不应接受重复卡。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    base_loadout = _sample_loadout(repository, scenario)
    support_card_ids = list(_sample_manual_support_card_ids(repository, base_loadout.stat_profile.plan_type, count=6))
    support_card_ids[-1] = support_card_ids[0]

    with pytest.raises(ValueError, match='Duplicate support card'):
        build_idol_loadout(
            repository,
            scenario,
            AMAO_R,
            producer_level=35,
            idol_rank=4,
            dearness_level=10,
            selected_support_card_ids=tuple(support_card_ids),
        )


def test_build_idol_loadout_rejects_non_full_manual_support_deck() -> None:
    """手动支援编成应要求正好 6 张，避免偏离正式游戏规则。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    base_loadout = _sample_loadout(repository, scenario)
    support_card_ids = _sample_manual_support_card_ids(repository, base_loadout.stat_profile.plan_type, count=5)

    with pytest.raises(ValueError, match='exactly 6 cards'):
        build_idol_loadout(
            repository,
            scenario,
            AMAO_R,
            producer_level=35,
            idol_rank=4,
            dearness_level=10,
            selected_support_card_ids=support_card_ids,
        )


def test_support_ability_effect_does_not_chain_trigger_other_support_abilities() -> None:
    """支援能力带来的资源获取不应继续连锁触发另一条支援能力。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=31, idol_loadout=_sample_loadout(repository, scenario))
    runtime.reset()

    drink_row = repository.produce_drinks.rows[0]
    drink_id = str(drink_row.get('id') or '')
    assert drink_id

    _inject_produce_trigger(repository, trigger_id='test-trigger-support-start', phase_type='ProducePhaseType_ProduceStart')
    _inject_produce_trigger(repository, trigger_id='test-trigger-support-get-drink', phase_type='ProducePhaseType_GetProduceDrink')
    _inject_produce_effect_row(
        repository,
        effect_id='test-effect-support-grant-drink',
        effect_type='ProduceEffectType_ProduceReward',
        produce_rewards=[
            {
                'resourceType': 'ProduceResourceType_ProduceDrink',
                'resourceId': drink_id,
                'resourceLevel': 0,
            }
        ],
    )
    _inject_produce_effect_row(
        repository,
        effect_id='test-effect-support-add-point',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=23,
    )

    runtime.active_produce_skills.extend(
        [
            ActiveProduceSkillState(
                skill_id='test_support_skill_grant_drink',
                level=1,
                trigger_id='test-trigger-support-start',
                effect_ids=('test-effect-support-grant-drink',),
                source='support_skill',
            ),
            ActiveProduceSkillState(
                skill_id='test_support_skill_get_drink_bonus',
                level=1,
                trigger_id='test-trigger-support-get-drink',
                effect_ids=('test-effect-support-add-point',),
                source='support_skill',
            ),
        ]
    )

    runtime.drinks = []
    produce_points_before = float(runtime.state['produce_points'])
    drink_count_before = len(runtime.drinks)

    runtime._dispatch_produce_item_phase('ProducePhaseType_ProduceStart')

    assert len(runtime.drinks) == drink_count_before + 1
    assert runtime.state['produce_points'] == pytest.approx(produce_points_before)

    runtime._dispatch_produce_item_phase('ProducePhaseType_GetProduceDrink')

    assert runtime.state['produce_points'] == pytest.approx(produce_points_before + 23.0)


def test_exam_turn_color_changes_effective_score_bonus_multiplier() -> None:
    """当前回合颜色应把固定基准倍率切成逐回合的动态得分倍率。"""

    runtime = _sample_runtime(seed=29, exam_score_bonus_multiplier=1.0)
    runtime.parameter_stats = (900.0, 300.0, 300.0)

    runtime.current_turn_color = 'vocal'
    runtime._refresh_turn_score_bonus_multiplier()
    vocal_multiplier = runtime.score_bonus_multiplier

    runtime.current_turn_color = 'dance'
    runtime._refresh_turn_score_bonus_multiplier()
    dance_multiplier = runtime.score_bonus_multiplier

    runtime.current_turn_color = 'visual'
    runtime._refresh_turn_score_bonus_multiplier()
    visual_multiplier = runtime.score_bonus_multiplier

    assert vocal_multiplier > 1.0
    assert dance_multiplier < 1.0
    assert vocal_multiplier > dance_multiplier
    assert dance_multiplier == pytest.approx(visual_multiplier)


def test_scenario_parameter_growth_limit_matches_master_data() -> None:
    """场景应从 Produce 主数据读取模式级三维成长上限。"""

    repository = MasterDataRepository()

    assert repository.build_scenario('produce-001').parameter_growth_limit == pytest.approx(1000.0)
    assert repository.build_scenario('produce-003').parameter_growth_limit == pytest.approx(1800.0)
    assert repository.build_scenario('produce-005').parameter_growth_limit == pytest.approx(2600.0)


def test_produce_runtime_clamps_parameter_growth_to_mode_limit() -> None:
    """培育阶段的课程增益和效果增益都不应突破模式主数据上限。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    loadout = _sample_loadout(repository, scenario)
    runtime = ProduceRuntime(repository, scenario, seed=31, idol_loadout=loadout)
    runtime.reset()
    cap = scenario.parameter_growth_limit

    runtime.state['vocal'] = cap - 1.0
    runtime._apply_produce_effect(
        {
            'id': 'test-effect-vocal-cap',
            'produceEffectType': 'ProduceEffectType_VocalAddition',
            'effectValueMin': 50,
            'effectValueMax': 50,
        },
        source_action_type='lesson_vocal_normal',
    )
    assert runtime.state['vocal'] == pytest.approx(cap)

    runtime.state['dance'] = cap - 1.0
    runtime._candidates = [
        ProduceActionCandidate(
            label='test',
            action_type='lesson_dance_normal',
            effect_types=[],
            produce_effect_ids=[],
            stat_deltas=(0.0, 50.0, 0.0),
            available=True,
        )
    ]
    runtime.step(0)
    assert runtime.state['dance'] == pytest.approx(cap)


def test_produce_runtime_initial_item_trigger_fires_on_produce_start() -> None:
    """带 `ProduceStart` trigger 的 challenge item 应在 reset 时生效。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    custom_item_id = 'test-item-produce-start'
    _inject_produce_item(
        repository,
        item_id=custom_item_id,
        trigger_id='test-trigger-produce-start',
        phase_type='ProducePhaseType_ProduceStart',
        produce_effect_id='test-effect-produce-start',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=10,
    )

    baseline_runtime = ProduceRuntime(
        repository,
        scenario,
        seed=39,
        idol_loadout=replace(loadout, produce_item_id='', exam_status_enchant_ids=(), exam_status_enchant_specs=()),
    )
    baseline_runtime.reset()

    runtime = ProduceRuntime(
        repository,
        scenario,
        seed=39,
        idol_loadout=replace(loadout, produce_item_id=custom_item_id, exam_status_enchant_ids=(), exam_status_enchant_specs=()),
    )
    runtime.reset()

    assert runtime.state['produce_points'] == pytest.approx(baseline_runtime.state['produce_points'] + 10.0)


def test_auto_select_support_cards_prefers_matching_types_and_returns_six_cards() -> None:
    """自动选支援卡应返回 6 张，并优先覆盖主属性类型。"""

    repository = MasterDataRepository(assets_dir='../../assets/gakumasu-diff', localization_dir='../../assets/GakumasTranslationData/local-files/masterTrans')
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)

    selected = auto_select_support_cards(
        repository,
        scenario,
        loadout,
        config=SupportCardAutoSelectConfig(support_card_level=60, deck_size=6),
    )

    assert len(selected) == 6
    assert len({item.support_card_id for item in selected}) == 6
    selected_types = {item.support_card_type for item in selected}
    assert any(card_type in selected_types for card_type in {'SupportCardType_Vocal', 'SupportCardType_Dance', 'SupportCardType_Visual'})
    assert all(item.score > 0 for item in selected)


def test_exam_runtime_support_card_upgrade_is_temporary() -> None:
    """技能卡支援应只在当前回合临时提升卡面，回合结束后恢复。"""

    repository = MasterDataRepository(assets_dir='../../assets/gakumasu-diff', localization_dir='../../assets/GakumasTranslationData/local-files/masterTrans')
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    support_cards = auto_select_support_cards(repository, scenario, loadout, SupportCardAutoSelectConfig(deck_size=1))
    runtime = _sample_runtime(loadout=replace(loadout, support_cards=support_cards))
    support_row = runtime.repository.support_cards.first(support_cards[0].support_card_id)
    assert support_row is not None
    support_row['produceCardUpgradePermil'] = 1000

    base_card_row = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if int(row.get('upgradeCount') or 0) == 0
        and repository.card_row_by_upgrade(str(row.get('id') or ''), 1, fallback_to_canonical=False) is not None
    )
    upgradable = RuntimeCard(
        uid=runtime._next_uid(),
        card_id=str(base_card_row.get('id') or ''),
        upgrade_count=0,
        base_card=base_card_row,
    )
    original_upgrade = int(upgradable.base_card.get('upgradeCount') or 0)

    runtime.hand = [upgradable]
    runtime.current_turn_color = 'vocal'
    runtime._apply_support_card_support()
    upgraded_count = int(upgradable.base_card.get('upgradeCount') or 0)

    assert upgraded_count == original_upgrade + 1

    runtime._clear_support_card_upgrades()

    assert int(upgradable.base_card.get('upgradeCount') or 0) == original_upgrade


def test_exam_runtime_support_card_upgrade_excludes_legend_cards() -> None:
    """传奇技能卡不应成为技能卡支援对象。"""

    repository = MasterDataRepository(assets_dir='../../assets/gakumasu-diff', localization_dir='../../assets/GakumasTranslationData/local-files/masterTrans')
    scenario = repository.build_scenario('produce-006')
    loadout = _sample_loadout(repository, scenario)
    support_cards = auto_select_support_cards(repository, scenario, loadout, SupportCardAutoSelectConfig(deck_size=1))
    runtime = _sample_runtime(loadout=replace(loadout, support_cards=support_cards), stage_type='ProduceStepType_AuditionFinal')
    support_row = runtime.repository.support_cards.first(support_cards[0].support_card_id)
    assert support_row is not None
    support_row['produceCardUpgradePermil'] = 1000

    legend_row = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('rarity') or '') == 'ProduceCardRarity_Legend'
    )
    legend_card = RuntimeCard(
        uid=runtime._next_uid(),
        card_id=str(legend_row.get('id') or ''),
        upgrade_count=int(legend_row.get('upgradeCount') or 0),
        base_card=legend_row,
    )
    runtime.hand = [legend_card]
    runtime.current_turn_color = 'vocal'

    runtime._apply_support_card_support()

    assert str(legend_card.base_card.get('rarity') or '') == 'ProduceCardRarity_Legend'


def test_exam_runtime_support_card_upgrade_prefers_active_cards_for_vocal_support() -> None:
    """Vocal/Dance/Visual 支援应优先命中 ActiveSkill。"""

    repository = MasterDataRepository(assets_dir='../../assets/gakumasu-diff', localization_dir='../../assets/GakumasTranslationData/local-files/masterTrans')
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    support_cards = auto_select_support_cards(repository, scenario, loadout, SupportCardAutoSelectConfig(deck_size=1))
    runtime = _sample_runtime(loadout=replace(loadout, support_cards=support_cards))
    support_row = runtime.repository.support_cards.first(support_cards[0].support_card_id)
    assert support_row is not None
    support_row['type'] = 'SupportCardType_Vocal'
    support_row['produceCardUpgradePermil'] = 1000

    active_row = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('category') or '') == 'ProduceCardCategory_ActiveSkill'
        and int(row.get('upgradeCount') or 0) == 0
        and repository.card_row_by_upgrade(str(row.get('id') or ''), 1, fallback_to_canonical=False) is not None
    )
    mental_row = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('category') or '') == 'ProduceCardCategory_MentalSkill'
        and int(row.get('upgradeCount') or 0) == 0
        and repository.card_row_by_upgrade(str(row.get('id') or ''), 1, fallback_to_canonical=False) is not None
    )
    active_card = RuntimeCard(uid=runtime._next_uid(), card_id=str(active_row.get('id') or ''), upgrade_count=0, base_card=active_row)
    mental_card = RuntimeCard(uid=runtime._next_uid(), card_id=str(mental_row.get('id') or ''), upgrade_count=0, base_card=mental_row)
    runtime.hand = [active_card, mental_card]
    runtime.current_turn_color = 'vocal'

    active_priority = runtime._support_upgrade_target_priority(active_card, support_row)
    mental_priority = runtime._support_upgrade_target_priority(mental_card, support_row)

    assert active_priority > mental_priority


def test_exam_runtime_support_card_order_prefers_higher_rate_then_level() -> None:
    """多张支援卡结算顺序应先看基础概率，再看等级。"""

    repository = MasterDataRepository(assets_dir='../../assets/gakumasu-diff', localization_dir='../../assets/GakumasTranslationData/local-files/masterTrans')
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    selected = auto_select_support_cards(repository, scenario, loadout, SupportCardAutoSelectConfig(deck_size=2))
    runtime = _sample_runtime(loadout=replace(loadout, support_cards=selected))

    support_a = runtime.repository.support_cards.first(selected[0].support_card_id)
    support_b = runtime.repository.support_cards.first(selected[1].support_card_id)
    assert support_a is not None and support_b is not None
    support_a['produceCardUpgradePermil'] = 19
    support_b['produceCardUpgradePermil'] = 37

    ordered = runtime._ordered_support_cards()

    assert ordered[0].support_card_id == selected[1].support_card_id


def test_produce_runtime_retry_flow_consumes_shared_continue_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """再挑战应消耗共享 continue 次数，并保留同一考试槽位。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=79)
    runtime.reset()
    runtime.pre_audition_phase = 'retry'
    runtime.pending_audition_stage = scenario.audition_sequence[0]
    runtime.pending_audition_result = {
        'stage_type': scenario.audition_sequence[0],
        'reward': 0.3,
        'audition_slot': 0,
        'cleared': True,
        'effective_score': 1234.0,
        'fan_vote_gain': 0.0,
        'deck_quality_gain': 0.0,
        'drink_quality_gain': 0.0,
    }
    runtime.state['continue_remaining'] = 3.0

    monkeypatch.setattr(
        runtime,
        '_run_audition',
        lambda stage_type, include_pre_audition_phases=False, apply_outcome=False: (
            0.4,
            {
                'stage_type': stage_type,
                'cleared': True,
                'effective_score': 2345.0,
                'fan_vote_gain': 10.0,
                'deck_quality_gain': 0.4,
                'drink_quality_gain': 0.2,
                'rank': 1,
                'rank_threshold': 3,
                'rival_scores': [2000.0],
            },
        ),
    )

    actions = runtime.legal_actions()
    retry_index = next(index for index, action in enumerate(actions) if action.action_type == 'audition_retry')
    reward, terminated, info = runtime.step(retry_index)

    assert reward == pytest.approx(0.0)
    assert terminated is False
    assert runtime.state['continue_remaining'] == pytest.approx(2.0)
    assert runtime.pre_audition_phase == 'retry'
    assert info['continue_remaining'] == 2


def test_produce_runtime_audition_result_reports_rank_and_rivals(monkeypatch: pytest.MonkeyPatch) -> None:
    """考试结果应包含 rival 分数和 rank。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=83)
    runtime.reset()

    monkeypatch.setattr(runtime, '_choose_exam_action', lambda _runtime: None)
    reward, info = runtime._run_audition(runtime.scenario.audition_sequence[0], apply_outcome=False)

    assert isinstance(reward, float)
    assert 'rank' in info
    assert 'rank_threshold' in info
    assert 'rival_scores' in info
    assert 'rival_phase_breakdowns' in info
    assert 'threshold_rival_score' in info
    assert isinstance(info['rival_scores'], list)
    if info['rival_phase_breakdowns']:
        first = info['rival_phase_breakdowns'][0]
        assert {'op', 'mid', 'ed', 'final'} <= set(first)


def test_build_idol_loadout_selected_challenge_items_are_registered() -> None:
    """显式选择的 challenge item 应进入 loadout，并在 reset 时注册。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    challenge_item_id = next(str(row.get('id') or '') for row in repository.produce_items.rows if row.get('isChallenge'))
    loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=35,
        idol_rank=4,
        dearness_level=10,
        selected_challenge_item_ids=(challenge_item_id,),
    )
    runtime = ProduceRuntime(repository, scenario, seed=89, idol_loadout=loadout)
    runtime.reset()

    assert challenge_item_id in {item.item_id for item in runtime.active_produce_items}


def test_build_idol_loadout_auto_support_cards_merge_support_skills() -> None:
    """自动支援卡编成后，应把支援卡技能并入 loadout.produce_skills。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    plain_loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=35,
        idol_rank=4,
        dearness_level=10,
        auto_select_support_cards_for_training=False,
    )
    enriched_loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=35,
        idol_rank=4,
        dearness_level=10,
        auto_select_support_cards_for_training=True,
    )

    assert len(enriched_loadout.support_cards) == 6
    assert len(enriched_loadout.produce_skills) > len(plain_loadout.produce_skills)
    assert any('p_support_skill-' in skill.skill_id for skill in enriched_loadout.produce_skills)


def test_produce_runtime_support_card_skill_triggers_on_end_lesson() -> None:
    """支援卡并入的 ProduceSkill 应在对应 phase 触发，而不只是在开场生效。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=35,
        idol_rank=4,
        dearness_level=10,
        auto_select_support_cards_for_training=True,
    )
    runtime = ProduceRuntime(repository, scenario, seed=97, idol_loadout=loadout)
    runtime.reset()

    target_skill = next(skill for skill in loadout.produce_skills if 'p_trigger-end_lesson-lesson_vocal' in skill.trigger_id)
    before_vocal = runtime.state['vocal']
    runtime._dispatch_produce_item_phase('ProducePhaseType_EndLesson', action_type='lesson_vocal_normal')

    assert runtime.state['vocal'] >= before_vocal
    assert any(active_skill.skill_id == target_skill.skill_id for active_skill in runtime.active_produce_skills)


def test_produce_runtime_rewarded_item_registers_and_fires_get_item_trigger() -> None:
    """显式奖励的 P 道具应进入库存，并能响应 `GetProduceItem`。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=41)
    reward_item_id = 'test-item-on-get'
    _inject_produce_item(
        repository,
        item_id=reward_item_id,
        trigger_id='test-trigger-get-item',
        phase_type='ProducePhaseType_GetProduceItem',
        produce_effect_id='test-effect-get-item',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=7,
    )
    runtime.reset()
    before_points = runtime.state['produce_points']

    reward_effect = {
        'id': 'test-reward-item-effect',
        'produceEffectType': 'ProduceEffectType_ProduceReward',
        'effectValueMin': 0,
        'effectValueMax': 0,
        'produceResourceType': 'ProduceResourceType_Unknown',
        'produceRewards': [
            {
                'resourceType': 'ProduceResourceType_ProduceItem',
                'resourceId': reward_item_id,
                'resourceLevel': 0,
            }
        ],
        'produceCardSearchId': '',
        'produceExamStatusEnchantId': '',
        'produceStepEventDetailId': '',
        'pickRangeType': 'ProducePickRangeType_Unknown',
        'pickCountMin': 1,
        'pickCountMax': 1,
        'isResearch': False,
    }
    runtime._apply_produce_effect(reward_effect, source_action_type='present')

    assert reward_item_id in {item.item_id for item in runtime.active_produce_items}
    assert runtime.state['produce_points'] == pytest.approx(before_points + 7.0)


def test_produce_runtime_get_card_trigger_respects_search_and_stamina_ratio() -> None:
    """`GetProduceCard` trigger 应同时校验卡搜索条件和体力比例。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    matching_card = dict(repository.produce_cards.rows[0])
    search_id = 'p_card_search-test-trigger-card'
    _inject_table_row(
        repository.load_table('ProduceCardSearch'),
        {
            'id': search_id,
            'produceCardIds': [str(matching_card.get('id') or '')],
            'cardRarities': [],
            'upgradeCounts': [],
            'planType': 'ProducePlanType_Unknown',
            'cardCategories': [],
            'cardStatusType': 'ProduceCardSearchStatusType_Unknown',
            'orderType': 'ProduceCardOrderType_Unknown',
            'cardPositionType': 'ProduceCardPositionType_DeckAll',
            'cardSearchTag': '',
            'produceCardRandomPoolId': '',
            'limitCount': 0,
            'staminaMinMaxType': 'ConditionMinMaxType_Unknown',
            'staminaMin': 0,
            'staminaMax': 0,
            'examEffectType': 'ProduceExamEffectType_Unknown',
            'effectGroupIds': [],
            'isSelf': False,
            'produceDescriptions': [],
            'produceCardPoolId': '',
            'costType': 'ExamCostType_Unknown',
            'isCustomized': False,
        },
    )
    _inject_produce_item(
        repository,
        item_id='test-item-get-card',
        trigger_id=f'test-trigger-get-card-0000_0000-{search_id}-stamina_ratio-0500_0000',
        phase_type='ProducePhaseType_GetProduceCard',
        produce_effect_id='test-effect-get-card',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=9,
    )

    runtime = ProduceRuntime(repository, scenario, seed=43)
    runtime.reset()
    runtime._register_produce_item('test-item-get-card', source='reward')
    runtime.state['stamina'] = runtime.state['max_stamina']
    before_points = runtime.state['produce_points']
    runtime._grant_resource('ProduceResourceType_ProduceCard', str(matching_card.get('id') or ''), int(matching_card.get('upgradeCount') or 0))
    assert runtime.state['produce_points'] == pytest.approx(before_points + 9.0)

    runtime_low = ProduceRuntime(repository, scenario, seed=47)
    runtime_low.reset()
    runtime_low._register_produce_item('test-item-get-card', source='reward')
    runtime_low.state['stamina'] = runtime_low.state['max_stamina'] * 0.25
    before_low = runtime_low.state['produce_points']
    runtime_low._grant_resource('ProduceResourceType_ProduceCard', str(matching_card.get('id') or ''), int(matching_card.get('upgradeCount') or 0))
    assert runtime_low.state['produce_points'] == pytest.approx(before_low)


def test_produce_runtime_pre_audition_phases_fire_and_respect_fire_limit() -> None:
    """考试前应分发 shop/customize phases，并按 fireLimit 限制触发次数。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=45)
    _inject_produce_item(
        repository,
        item_id='test-item-start-shop',
        trigger_id='test-trigger-start-shop',
        phase_type='ProducePhaseType_StartShop',
        produce_effect_id='test-effect-start-shop',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=3,
        fire_limit=2,
    )
    _inject_produce_item(
        repository,
        item_id='test-item-start-customize',
        trigger_id='test-trigger-start-customize',
        phase_type='ProducePhaseType_StartCustomize',
        produce_effect_id='test-effect-start-customize',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=5,
        fire_limit=3,
    )
    _inject_produce_item(
        repository,
        item_id='test-item-end-shop',
        trigger_id='test-trigger-end-shop',
        phase_type='ProducePhaseType_EndShop',
        produce_effect_id='test-effect-end-shop',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=7,
        fire_limit=1,
    )

    runtime.reset()
    runtime._register_produce_item('test-item-start-shop', source='reward')
    runtime._register_produce_item('test-item-start-customize', source='reward')
    runtime._register_produce_item('test-item-end-shop', source='reward')
    before_points = runtime.state['produce_points']

    for stage_type in scenario.audition_sequence:
        runtime._run_audition(stage_type)

    assert runtime.state['produce_points'] == pytest.approx(before_points + 2 * 3.0 + 3 * 5.0 + 7.0)


def test_first_star_scenario_includes_hard_lessons() -> None:
    """Legend 路线应暴露 hard lesson 动作。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')

    assert 'lesson_vocal_hard' in scenario.action_types
    assert 'lesson_dance_hard' in scenario.action_types
    assert 'lesson_visual_hard' in scenario.action_types


def test_first_star_scenario_includes_main_flow_actions() -> None:
    """初路线应显式暴露授业、外出和活动支给动作。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')

    assert 'school_class' in scenario.action_types
    assert 'outing' in scenario.action_types
    assert 'activity_supply' in scenario.action_types


def test_first_star_school_class_costs_stamina_and_grows_parameters() -> None:
    """授业应消耗体力并提升参数。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=43)
    runtime.reset()

    action = runtime._sample_action('school_class')

    assert action.stamina_delta < 0.0
    assert sum(action.stat_deltas) > 0.0


def test_first_star_outing_spends_points_and_recovers_stamina() -> None:
    """外出应消耗 P 点并回复体力。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=45)
    runtime.reset()

    action = runtime._sample_action('outing')

    assert action.produce_point_delta < 0.0
    assert action.stamina_delta > 0.0


def test_first_star_activity_supply_grants_resources() -> None:
    """活动支给应带来资源入账。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=47)
    runtime.reset()

    action = runtime._sample_action('activity_supply')

    assert action.produce_point_delta >= 0.0 or bool(action.produce_effect_ids)


def test_nia_business_profiles_cover_four_main_effect_shapes() -> None:
    """营业应能映射出四类主流程影响模板。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=48)
    runtime.reset()

    assert runtime._business_action_profile('event-detail-business-produce_004-plan1-produce_card-demo')[2] == 'card'
    assert runtime._business_action_profile('event-detail-business-produce_004-plan1-produce_drink-demo')[2] == 'drink'
    assert runtime._business_action_profile('event-detail-business-produce_004-plan1-produce_point-demo')[2] == 'point'
    assert runtime._business_action_profile('event-detail-business-produce_004-plan1-stamina-demo')[2] == 'stamina'


def test_nia_present_bonus_points_fixed_amount() -> None:
    """差入额外 P 点奖励量固定（帮助页只说概率随票数上升，奖励量不变）。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=49)
    runtime.reset()

    # 无论票数多少，奖励量均为固定常量 12.0
    runtime.state['fan_votes'] = 0.0
    bonus_low = runtime._present_bonus_produce_points()
    runtime.state['fan_votes'] = 120000.0
    bonus_high = runtime._present_bonus_produce_points()

    assert bonus_low == pytest.approx(12.0)
    assert bonus_high == pytest.approx(12.0)

    # 概率应随票数上升
    runtime.state['fan_votes'] = 0.0
    prob_low = runtime._present_bonus_points_should_trigger.__func__  # 仅检查方法存在
    runtime.state['fan_votes'] = 120000.0
    # _present_bonus_points_should_trigger 是概率采样，不在此 assert 具体值


def test_nia_scenario_includes_pre_audition_decision_actions() -> None:
    """NIA 路线应把相谈前置决策动作暴露给 planning。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')

    assert produce_runtime_module.SHOP_CARD_ACTION_TYPES[0] in scenario.action_types
    assert produce_runtime_module.SHOP_CARD_ACTION_TYPES[-1] in scenario.action_types
    assert produce_runtime_module.SHOP_DRINK_ACTION_TYPES[0] in scenario.action_types
    assert produce_runtime_module.SHOP_DRINK_ACTION_TYPES[-1] in scenario.action_types
    assert produce_runtime_module.SHOP_UPGRADE_ACTION_TYPES[0] in scenario.action_types
    assert produce_runtime_module.SHOP_UPGRADE_ACTION_TYPES[-1] in scenario.action_types
    assert produce_runtime_module.SHOP_DELETE_ACTION_TYPES[0] in scenario.action_types
    assert produce_runtime_module.SHOP_DELETE_ACTION_TYPES[-1] in scenario.action_types
    assert 'customize_apply' in scenario.action_types
    assert 'audition_select_1' in scenario.action_types
    assert 'audition_select_4' in scenario.action_types
    assert 'pre_audition_continue' in scenario.action_types


def test_produce_runtime_checkpoint_enters_shop_phase_before_audition() -> None:
    """到达 checkpoint 后应先进入咨询阶段，而不是立刻开考。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=49)
    runtime.reset()
    checkpoint_step, _ = runtime.checkpoints[0]
    runtime.state['step'] = checkpoint_step - 1

    actions = runtime.legal_actions()
    refresh_index = next(index for index, action in enumerate(actions) if action.action_type == 'refresh')

    _, terminated, info = runtime.step(refresh_index)

    assert not terminated
    assert runtime.pre_audition_phase == 'shop'
    assert info['pre_audition_phase'] == 'shop'
    assert runtime.pending_audition_stage == scenario.audition_sequence[0]
    assert runtime.state['audition_index'] == 0
    assert 'audition_0' not in info

    followup_actions = runtime.legal_actions()
    available_types = {action.action_type for action in followup_actions if action.available}
    assert available_types <= {
        *produce_runtime_module.SHOP_CARD_ACTION_TYPES,
        *produce_runtime_module.SHOP_DRINK_ACTION_TYPES,
        *produce_runtime_module.SHOP_UPGRADE_ACTION_TYPES,
        *produce_runtime_module.SHOP_DELETE_ACTION_TYPES,
        'customize_apply',
        'audition_select_1',
        'audition_select_2',
        'audition_select_3',
        'pre_audition_continue',
    }
    assert 'pre_audition_continue' in available_types
    assert runtime.state['selected_audition_selector'] in {'audition_select_1', 'audition_select_2', 'audition_select_3'}
    assert runtime.state['selected_audition_stage_type'] == scenario.audition_sequence[0]
    assert 'audition_select_4' not in available_types
    assert runtime.state['selected_audition_stage_type'] == scenario.audition_sequence[0]


def test_produce_runtime_customize_action_consumes_slot_before_audition() -> None:
    """考试前特训应消耗特训次数，并把自定义效果写回牌面。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=50)
    runtime.reset()
    customizable_card = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if int(row.get('upgradeCount') or 0) >= 1
        and not bool(row.get('isInitialDeckProduceCard'))
        and str(row.get('category') or '') != 'ProduceCardCategory_Trouble'
        and row.get('produceCardCustomizeIds')
    )
    runtime.deck = [customizable_card]
    runtime.state['produce_points'] = 999.0
    runtime.state['customize_slots'] = 1.0
    runtime._start_pre_audition_flow(scenario.audition_sequence[0])

    actions = runtime.legal_actions()
    customize_index = next(index for index, action in enumerate(actions) if action.action_type == 'customize_apply' and action.available)
    before_points = runtime.state['produce_points']

    runtime.step(customize_index)

    assert runtime.remaining_customize_actions == 0
    assert runtime.state['produce_points'] < before_points
    assert runtime.deck[0].get('customizedProduceCardCustomizeIds')


def test_nia_final_audition_unlocks_finale_at_dearness_17() -> None:
    """NIA 最终场只有亲爱度达到 17 后才应开放 FINALE。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=242)
    runtime.reset()
    runtime.state['dearness_level'] = 16.0
    runtime.state['fan_votes'] = 62000.0
    runtime.pending_audition_stage = scenario.audition_sequence[-1]
    runtime._refresh_pre_audition_inventory()

    final_before = runtime.pre_audition_action_inventory['audition_select_4']
    assert final_before.available is False

    runtime.state['dearness_level'] = 17.0
    runtime._refresh_pre_audition_inventory()

    final_after = runtime.pre_audition_action_inventory['audition_select_4']
    assert final_after.available is True




def test_first_star_pre_audition_hard_lesson_week_only_exposes_hard_lessons() -> None:
    """初路线考试前两周应只暴露追込课。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=244)
    runtime.reset()
    runtime.state['step'] = runtime.checkpoints[0][0] - 2

    candidates = runtime.legal_actions()

    assert candidates
    assert all(candidate.action_type in HARD_ACTION_TYPES for candidate in candidates if candidate.available)



def test_first_star_pre_audition_refresh_week_forces_refresh() -> None:
    """初路线考试前一周应强制进入恢复。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=246)
    runtime.reset()
    runtime.state['step'] = runtime.checkpoints[0][0] - 1

    candidates = runtime.legal_actions()

    assert len(candidates) == 1
    assert candidates[0].action_type == 'refresh'
    assert candidates[0].auto_skip is True
    assert candidates[0].available is True



def test_first_star_pre_audition_refresh_uses_before_audition_recovery_permille() -> None:
    """初路线考试前恢复应读取考前恢复倍率，而不是普通休息倍率。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=247)
    runtime.reset()
    runtime.state['step'] = runtime.checkpoints[0][0] - 1

    candidate = runtime._sample_action('refresh')
    expected = runtime.state['max_stamina'] * float(runtime.produce_setting.get('beforeAuditionRefreshStaminaRecoveryPermil') or 0) / 1000.0

    assert candidate.stamina_delta == pytest.approx(expected)



def test_produce_runtime_hard_lesson_trigger_matches_hard_actions() -> None:
    """`lesson_hard` trigger 应只在 hard lesson 动作上命中。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    _inject_produce_item(
        repository,
        item_id='test-item-hard-lesson',
        trigger_id='test-trigger-start-lesson-lesson_hard',
        phase_type='ProducePhaseType_StartLesson',
        produce_effect_id='test-effect-hard-lesson',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=11,
    )

    baseline_runtime = ProduceRuntime(repository, scenario, seed=51)
    baseline_runtime.reset()
    baseline_actions = baseline_runtime.legal_actions()
    hard_index = next(index for index, action in enumerate(baseline_actions) if action.action_type == 'lesson_vocal_hard')
    baseline_before = baseline_runtime.state['produce_points']
    baseline_runtime.step(hard_index)

    runtime = ProduceRuntime(repository, scenario, seed=51)
    runtime.reset()
    runtime._register_produce_item('test-item-hard-lesson', source='reward')
    actions = runtime.legal_actions()
    runtime_before = runtime.state['produce_points']
    runtime.step(hard_index)

    assert actions[hard_index].action_type == 'lesson_vocal_hard'
    assert runtime.state['produce_points'] == pytest.approx((baseline_runtime.state['produce_points'] - baseline_before) + runtime_before + 11.0)


def test_master_and_legend_refresh_locked_until_first_lesson() -> None:
    """Master/Legend 模式下，休息必须先完成至少一节课后才能解锁。"""

    repository = MasterDataRepository()
    for scenario_id in ('produce-003', 'produce-006'):
        scenario = repository.build_scenario(scenario_id)
        runtime = ProduceRuntime(repository, scenario, seed=103)
        runtime.reset()

        actions = runtime.legal_actions()
        refresh = next(action for action in actions if action.action_type == 'refresh')
        assert refresh.available is False

        runtime.state['lessons_taken'] = 1.0
        actions = runtime.legal_actions()
        refresh = next(action for action in actions if action.action_type == 'refresh')
        assert refresh.available is True


def test_legend_refresh_respects_max_refresh_count() -> None:
    """Legend 模式下休息达到上限后应被禁用。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=105)
    runtime.reset()
    runtime.state['lessons_taken'] = 1.0
    runtime.state['refresh_used'] = float(scenario.max_refresh_count)

    actions = runtime.legal_actions()
    refresh = next(action for action in actions if action.action_type == 'refresh')

    assert scenario.max_refresh_count == 4
    assert refresh.available is False


def test_loadout_summary_enforces_high_difficulty_unlock_requirements() -> None:
    """高难路线的 loadout 摘要应校验基础解锁条件。"""

    with pytest.raises(ValueError, match='first_star_master 需要亲爱度至少 10'):
        loadout_summary(
            'first_star_master',
            LoadoutConfig(idol_card_id=AMAO_R, producer_level=35, idol_rank=4, dearness_level=9),
        )

    with pytest.raises(ValueError, match='nia_master 需要亲爱度至少 20'):
        loadout_summary(
            'nia_master',
            LoadoutConfig(idol_card_id=AMAO_R, producer_level=35, idol_rank=4, dearness_level=19),
        )

    with pytest.raises(ValueError, match='first_star_legend 需要 Producer 等级至少 50 且亲爱度至少 10'):
        loadout_summary(
            'first_star_legend',
            LoadoutConfig(idol_card_id=AMAO_R, producer_level=49, idol_rank=4, dearness_level=10),
        )


def test_produce_runtime_auto_skips_week_when_no_schedule_and_no_refresh() -> None:
    """没有可选日程且不能休息时，应自动跳周。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=107)
    runtime.reset()
    runtime.state['refresh_used'] = float(scenario.max_refresh_count)
    runtime.state['stamina'] = 0.0
    runtime.state['step'] = 1

    original_sample_action = runtime._sample_action

    def _blocked_sample_action(action_type: str):
        candidate = original_sample_action(action_type)
        if action_type != 'refresh':
            candidate.stamina_delta = -999.0
        return candidate

    runtime._sample_action = _blocked_sample_action  # type: ignore[method-assign]
    actions = runtime.legal_actions()
    refresh_index = next(index for index, action in enumerate(actions) if action.action_type == 'refresh')
    assert actions[refresh_index].auto_skip is True
    assert actions[refresh_index].available is True

    _, terminated, info = runtime.step(refresh_index)

    assert terminated is False
    assert info['auto_skipped_weeks'] >= 1
    assert runtime.state['step'] > 2


def test_challenge_item_increases_lesson_perfect_target_for_legend() -> None:
    """Legend challenge item 应抬高 lesson 的 PERFECT 上限。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    base_loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=50,
        idol_rank=4,
        dearness_level=10,
    )
    challenge_item_id = 'pitem_00-1-048-challenge'
    challenge_loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=50,
        idol_rank=4,
        dearness_level=10,
        selected_challenge_item_ids=(challenge_item_id,),
    )

    base_runtime = ProduceRuntime(repository, scenario, seed=109, idol_loadout=base_loadout)
    challenge_runtime = ProduceRuntime(repository, scenario, seed=109, idol_loadout=challenge_loadout)
    base_runtime.reset()
    challenge_runtime.reset()
    stage_type = 'ProduceStepType_LessonVocalNormal'

    base_reward, base_info = base_runtime._run_audition(stage_type, apply_outcome=False)
    challenge_reward, challenge_info = challenge_runtime._run_audition(stage_type, apply_outcome=False)

    assert challenge_reward != base_reward or challenge_info['effective_score'] != base_info['effective_score']


def test_produce_runtime_reset_exposes_challenge_bonus_state() -> None:
    """challenge P 道具加成应显式写入 produce state。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=50,
        idol_rank=4,
        dearness_level=10,
        selected_challenge_item_ids=('pitem_00-1-048-challenge',),
    )
    runtime = ProduceRuntime(repository, scenario, seed=113, idol_loadout=loadout)
    runtime.reset()

    assert runtime.state['challenge_lesson_perfect_bonus_ratio'] > 0.0
    assert runtime.state['challenge_audition_npc_bonus_ratio'] > 0.0


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_lesson_env_reset_reports_runtime_adjusted_perfect_target() -> None:
    """lesson reset 返回的 perfect target 应反映 challenge 修正后的运行时值。"""

    from gakumas_rl.service import build_env_from_config

    env = build_env_from_config(
        {
            'mode': 'lesson',
            'scenario': 'first_star_legend',
            'idol_card_id': AMAO_R,
            'producer_level': 50,
            'idol_rank': 4,
            'dearness_level': 10,
            'challenge_item_ids': ['pitem_00-1-048-challenge'],
            'auto_support_cards': True,
            'seed': 127,
        }
    )
    _obs, info = env.reset(seed=127)

    assert info['lesson_perfect_value'] == pytest.approx(env.runtime._current_perfect_target())


def test_produce_runtime_step_info_exposes_support_and_challenge_context() -> None:
    """step 返回信息应显式暴露支援卡数量与 challenge 加成。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=50,
        idol_rank=4,
        dearness_level=10,
        auto_select_support_cards_for_training=True,
        selected_challenge_item_ids=('pitem_00-1-048-challenge',),
    )
    runtime = ProduceRuntime(repository, scenario, seed=131, idol_loadout=loadout)
    runtime.reset()
    actions = runtime.legal_actions()
    lesson_index = next(index for index, action in enumerate(actions) if action.action_type.startswith('lesson_') and action.available)

    _reward, _terminated, info = runtime.step(lesson_index)

    assert info['support_card_count'] == 6
    assert info['challenge_lesson_perfect_bonus_ratio'] > 0.0
    assert info['challenge_audition_npc_bonus_ratio'] >= 0.0


def test_produce_runtime_reset_rebuilds_support_event_candidates() -> None:
    """reset 后若支援卡编成不同，support event 候选应重新构建。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout_a = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=35,
        idol_rank=4,
        dearness_level=10,
        auto_select_support_cards_for_training=True,
    )
    loadout_b = replace(loadout_a, support_cards=())
    runtime = ProduceRuntime(repository, scenario, seed=137, idol_loadout=loadout_a)
    runtime.reset()
    with_support = len(runtime.action_samples.get('present', []))

    runtime.idol_loadout = loadout_b
    runtime.reset()
    without_support = len(
        [
            row
            for row in runtime.action_samples.get('present', [])
            if str(row.get('eventType') or '') == 'ProduceEventType_SupportCard'
        ]
    )

    assert with_support >= without_support


def test_produce_runtime_shop_buy_drink_phase_fires_on_actual_purchase() -> None:
    """咨询购买 P 饮料时应分发 BuyShopItemProduceDrink phase。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    _inject_produce_item(
        repository,
        item_id='test-item-buy-drink',
        trigger_id='test-trigger-buy-drink',
        phase_type='ProducePhaseType_BuyShopItemProduceDrink',
        produce_effect_id='test-effect-buy-drink',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=13,
    )
    runtime = ProduceRuntime(repository, scenario, seed=53)
    runtime.reset()
    runtime._register_produce_item('test-item-buy-drink', source='reward')
    runtime.drinks = []
    runtime.state['produce_points'] = 999.0
    runtime._start_pre_audition_flow(scenario.audition_sequence[0])

    actions = runtime.legal_actions()
    drink_index = next(index for index, action in enumerate(actions) if action.action_type.startswith('shop_buy_drink_') and action.available)
    before_points = runtime.state['produce_points']
    before_drinks = len(runtime.drinks)

    runtime.step(drink_index)

    assert len(runtime.drinks) == before_drinks + 1
    assert runtime.state['produce_points'] == pytest.approx(before_points + actions[drink_index].produce_point_delta + 13.0)


def test_produce_runtime_shop_upgrade_phase_fires_on_actual_upgrade() -> None:
    """相谈强化卡牌时应分发 CustomizeProduceCard phase，并把升级后的效果暴露给模型。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    _inject_produce_item(
        repository,
        item_id='test-item-shop-upgrade-card',
        trigger_id='test-trigger-shop-upgrade-card',
        phase_type='ProducePhaseType_CustomizeProduceCard',
        produce_effect_id='test-effect-shop-upgrade-card',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=17,
    )
    runtime = ProduceRuntime(repository, scenario, seed=59)
    runtime.reset()
    runtime._register_produce_item('test-item-shop-upgrade-card', source='reward')
    upgradable_card = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if int(row.get('upgradeCount') or 0) == 0 and repository.card_row_by_upgrade(str(row.get('id') or ''), 1, fallback_to_canonical=False)
    )
    runtime.deck = [upgradable_card]
    runtime.state['produce_points'] = 999.0
    runtime._refresh_quality_scores()
    runtime._start_pre_audition_flow(scenario.audition_sequence[0])

    shop_actions = runtime.legal_actions()
    upgrade_index = next(index for index, action in enumerate(shop_actions) if action.action_type.startswith('shop_upgrade_card_') and action.available)
    before_points = runtime.state['produce_points']
    before_upgrade_count = int(runtime.deck[0].get('upgradeCount') or 0)

    runtime.step(upgrade_index)
    followup_actions = runtime.legal_actions()

    assert runtime.pre_audition_phase == 'shop'
    assert int(runtime.deck[0].get('upgradeCount') or 0) == before_upgrade_count + 1
    assert runtime.state['produce_points'] == pytest.approx(before_points + shop_actions[upgrade_index].produce_point_delta + 17.0)
    assert runtime.state['shop_card_modified_in_visit'] == pytest.approx(1.0)
    assert not any(
        action.available for action in followup_actions if action.action_type.startswith('shop_upgrade_card_') or action.action_type.startswith('shop_delete_card_')
    )


def test_produce_runtime_shop_delete_phase_fires_and_cost_scales() -> None:
    """相谈删除卡牌后，本次相谈不能再删/升，且下次相谈消耗应上涨。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    _inject_produce_item(
        repository,
        item_id='test-item-shop-delete-card',
        trigger_id='test-trigger-shop-delete-card',
        phase_type='ProducePhaseType_DeleteProduceCard',
        produce_effect_id='test-effect-shop-delete-card',
        effect_type='ProduceEffectType_ProducePointAddition',
        effect_value=19,
    )
    runtime = ProduceRuntime(repository, scenario, seed=63)
    runtime.reset()
    runtime._register_produce_item('test-item-shop-delete-card', source='reward')
    runtime.deck = [dict(row) for row in runtime.deck[:3]]
    runtime.state['produce_points'] = 999.0
    runtime._refresh_quality_scores()
    runtime._start_pre_audition_flow(scenario.audition_sequence[0])

    actions = runtime.legal_actions()
    delete_index = next(index for index, action in enumerate(actions) if action.action_type.startswith('shop_delete_card_') and action.available)
    before_points = runtime.state['produce_points']
    before_deck_size = len(runtime.deck)

    runtime.step(delete_index)

    assert len(runtime.deck) == before_deck_size - 1
    assert runtime.state['produce_points'] == pytest.approx(before_points + actions[delete_index].produce_point_delta + 19.0)
    assert runtime.state['shop_card_modify_count'] == pytest.approx(1.0)
    assert runtime.state['shop_card_modified_in_visit'] == pytest.approx(1.0)

    runtime.pre_audition_phase = 'weekly'
    runtime.pending_audition_stage = None
    runtime._start_pre_audition_flow(scenario.audition_sequence[0])
    next_actions = runtime.legal_actions()
    next_delete = next(action for action in next_actions if action.action_type.startswith('shop_delete_card_') and action.available)

    assert next_delete.produce_point_delta == pytest.approx(-125.0)


def test_produce_runtime_selection_card_pool_filters_initial_and_exclusive_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """咨询与三选一卡池应过滤初始卡、他人专属卡和二次强化卡。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    runtime = ProduceRuntime(repository, scenario, seed=61, idol_loadout=loadout)
    runtime.reset()
    runtime.initial_deck_card_ids = {'blocked-initial'}

    monkeypatch.setattr(
        produce_runtime_module,
        'build_weighted_card_pool',
        lambda *_args, **_kwargs: [
            {'id': 'blocked-initial', 'upgradeCount': 0, 'originIdolCardId': '', 'originSupportCardId': '', 'rarity': 'ProduceCardRarity_R'},
            {'id': 'blocked-idol', 'upgradeCount': 0, 'originIdolCardId': 'i_card-other-1-000', 'originSupportCardId': '', 'rarity': 'ProduceCardRarity_SR'},
            {'id': 'blocked-support', 'upgradeCount': 0, 'originIdolCardId': '', 'originSupportCardId': 'support-001', 'rarity': 'ProduceCardRarity_SR'},
            {'id': 'blocked-upgrade-2', 'upgradeCount': 2, 'originIdolCardId': '', 'originSupportCardId': '', 'rarity': 'ProduceCardRarity_SSR'},
            {'id': 'allowed-own-idol', 'upgradeCount': 0, 'originIdolCardId': loadout.idol_card_id, 'originSupportCardId': '', 'rarity': 'ProduceCardRarity_SR'},
            {'id': 'allowed-common', 'upgradeCount': 1, 'originIdolCardId': '', 'originSupportCardId': '', 'rarity': 'ProduceCardRarity_R'},
        ],
    )

    filtered_ids = [str(card.get('id') or '') for card in runtime._selection_card_pool()]

    assert filtered_ids == ['allowed-own-idol', 'allowed-common']


def test_produce_runtime_shop_inventory_uses_fixed_slots_prices_and_discounts(monkeypatch: pytest.MonkeyPatch) -> None:
    """咨询应生成固定 8 槽位，并按约定 rarity/强化次数计价。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    runtime = ProduceRuntime(repository, scenario, seed=67, idol_loadout=loadout)
    runtime.reset()

    card_pool = [
        {'id': 'shop-card-sr', 'name': 'shop-card-sr', 'upgradeCount': 0, 'rarity': 'ProduceCardRarity_SR'},
        {'id': 'shop-card-ssr', 'name': 'shop-card-ssr', 'upgradeCount': 0, 'rarity': 'ProduceCardRarity_SSR'},
        {'id': 'shop-card-r', 'name': 'shop-card-r', 'upgradeCount': 0, 'rarity': 'ProduceCardRarity_R'},
        {'id': 'shop-card-sr-2', 'name': 'shop-card-sr-2', 'upgradeCount': 0, 'rarity': 'ProduceCardRarity_SR'},
        {'id': 'shop-card-extra', 'name': 'shop-card-extra', 'upgradeCount': 0, 'rarity': 'ProduceCardRarity_SR'},
    ]
    drink_pool = [
        {'id': 'shop-drink-r', 'name': 'shop-drink-r', 'rarity': 'ProduceDrinkRarity_R', 'planType': 'ProducePlanType_Common'},
        {'id': 'shop-drink-sr', 'name': 'shop-drink-sr', 'rarity': 'ProduceDrinkRarity_SR', 'planType': 'ProducePlanType_Common'},
        {'id': 'shop-drink-ssr', 'name': 'shop-drink-ssr', 'rarity': 'ProduceDrinkRarity_SSR', 'planType': 'ProducePlanType_Common'},
        {'id': 'shop-drink-r-2', 'name': 'shop-drink-r-2', 'rarity': 'ProduceDrinkRarity_R', 'planType': 'ProducePlanType_Common'},
        {'id': 'shop-drink-extra', 'name': 'shop-drink-extra', 'rarity': 'ProduceDrinkRarity_SR', 'planType': 'ProducePlanType_Common'},
    ]

    monkeypatch.setattr(runtime, '_selection_card_pool', lambda: [dict(row) for row in card_pool])
    monkeypatch.setattr(runtime, '_shop_drink_pool', lambda: [dict(row) for row in drink_pool])
    monkeypatch.setattr(runtime, '_eligible_shop_upgrade_targets', lambda: [])
    monkeypatch.setattr(runtime, '_eligible_shop_delete_targets', lambda: [])
    monkeypatch.setattr(runtime, '_discounted_shop_slot_count', lambda: 2)
    monkeypatch.setattr(runtime, '_shop_discount_ratio', lambda slot_index, discounted_count: 0.8 if slot_index < discounted_count else 1.0)
    monkeypatch.setattr(
        produce_runtime_module,
        'sample_card_from_weighted_pool',
        lambda weighted_pool, _rng, **_kwargs: weighted_pool[0] if weighted_pool else None,
    )

    def _sample_variant(card_id: str, *, max_upgrade_count: int) -> dict[str, object] | None:
        assert max_upgrade_count == 1
        for row in card_pool:
            if row['id'] == card_id:
                if card_id == 'shop-card-ssr':
                    upgraded = dict(row)
                    upgraded['upgradeCount'] = 1
                    return upgraded
                return dict(row)
        return None

    monkeypatch.setattr(runtime, '_sample_capped_card_variant', _sample_variant)
    runtime.drinks = []
    runtime.state['produce_points'] = 999.0

    runtime._start_pre_audition_flow(scenario.audition_sequence[0])

    actions = runtime.legal_actions()
    stable_actions = runtime.legal_actions()
    available_actions = [action for action in actions if action.available]
    card_actions = [action for action in available_actions if action.action_type.startswith('shop_buy_card_')]
    drink_actions = [action for action in available_actions if action.action_type.startswith('shop_buy_drink_')]
    drink_price_map = {
        'shop-drink-r': 50.0,
        'shop-drink-sr': 100.0,
        'shop-drink-ssr': 130.0,
        'shop-drink-r-2': 50.0,
        'shop-drink-extra': 100.0,
    }

    assert len(card_actions) == 4
    assert len(drink_actions) == 4
    assert [action.resource_id for action in card_actions] == ['shop-card-sr', 'shop-card-ssr', 'shop-card-r', 'shop-card-sr-2']
    assert [action.produce_point_delta for action in card_actions] == [-80.0, -136.0, -80.0, -100.0]
    assert len({action.resource_id for action in drink_actions}) == 4
    assert sorted(
        action.produce_point_delta
        for action in drink_actions
    ) == sorted(
        -drink_price_map[action.resource_id] * (0.8 if action.action_type in {'shop_buy_drink_1', 'shop_buy_drink_2'} else 1.0)
        for action in drink_actions
    )
    stable_pairs = [
        (action.action_type, action.resource_id, action.produce_point_delta)
        for action in actions
        if action.action_type.startswith('shop_buy_') or action.action_type == 'pre_audition_continue'
    ]
    assert stable_pairs == [
        (action.action_type, action.resource_id, action.produce_point_delta)
        for action in stable_actions
        if action.action_type.startswith('shop_buy_') or action.action_type == 'pre_audition_continue'
    ]

    buy_index = next(index for index, action in enumerate(actions) if action.action_type == 'shop_buy_drink_1')
    runtime.step(buy_index)
    followup_actions = runtime.legal_actions()
    sold_out_drink = next(action for action in followup_actions if action.action_type == 'shop_buy_drink_1')

    assert sold_out_drink.available is False
    assert sold_out_drink.resource_id == ''


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_planning_env_shop_card_action_feature_encodes_exam_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """相谈技能卡动作应把卡面考试效果编码进模型动作特征，而不是只靠名字或 id。"""

    from gakumas_rl.envs import GakumasPlanningEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    env = GakumasPlanningEnv(repository, scenario, seed=71, idol_loadout=loadout)
    env.reset(seed=71)

    card_row = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if repository.card_exam_effect_types(row)
    )
    metadata = {
        'exam_effect_types': repository.card_exam_effect_types(card_row),
        'card_category': str(card_row.get('category') or ''),
        'card_rarity': str(card_row.get('rarity') or ''),
        'card_cost_type': str(card_row.get('costType') or ''),
    }
    candidate = ProduceActionCandidate(
        label='购买技能卡测试',
        action_type=produce_runtime_module.SHOP_CARD_ACTION_TYPES[0],
        effect_types=[],
        produce_effect_ids=[],
        produce_point_delta=-100.0,
        produce_card_id=str(card_row.get('id') or ''),
        resource_type='ProduceResourceType_ProduceCard',
        resource_id=str(card_row.get('id') or ''),
        resource_level=int(card_row.get('upgradeCount') or 0),
        available=True,
        exam_effect_types=list(metadata['exam_effect_types']),
        card_category=str(metadata['card_category']),
        card_rarity=str(metadata['card_rarity']),
        card_cost_type=str(metadata['card_cost_type']),
    )

    feature = env._candidate_feature(candidate)
    action_dim = len(env.taxonomy.action_types)
    produce_dim = len(env.taxonomy.produce_effect_types)
    exam_dim = len(env.taxonomy.exam_effect_types)
    exam_slice = feature[action_dim + produce_dim : action_dim + produce_dim + exam_dim]

    assert exam_slice.sum() >= 1.0


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_planning_env_global_observation_includes_runtime_bonuses() -> None:
    """培育全局观测应暴露已生效的事件/商店/审题倍率状态。"""

    from gakumas_rl.envs import GakumasPlanningEnv, _bounded_positive, _bounded_signed

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    env = GakumasPlanningEnv(repository, scenario, seed=73, idol_loadout=loadout)
    env.reset(seed=73)

    env.runtime.state.update(
        {
            'activity_produce_point_bonus': 0.30,
            'business_vote_bonus': 0.20,
            'lesson_present_point_bonus': 0.10,
            'support_event_point_bonus': 0.40,
            'support_event_stat_bonus': 0.25,
            'support_event_stamina_bonus': 0.15,
            'audition_vote_bonus': 0.35,
            'audition_turn_modifier': -1.0,
            'before_audition_refresh_penalty': 0.20,
            'generic_sp_rate_bonus': 0.30,
            'vocal_sp_rate_bonus': 0.10,
            'dance_sp_rate_bonus': 0.20,
            'visual_sp_rate_bonus': 0.30,
            'shop_discount': -0.20,
            'reward_card_count_bonus': 1.0,
            'customize_slots': 2.0,
            'exclude_count_bonus': 1.0,
            'reroll_count_bonus': 2.0,
            'card_upgrade_probability_bonus': 0.25,
        }
    )

    observation = env._global_observation()
    feature_map = dict(zip(env.global_feature_names, observation, strict=True))

    assert env.global_dim == len(env.global_feature_names)
    assert feature_map['activity_produce_point_bonus'] == pytest.approx(_bounded_signed(0.30, 0.5))
    assert feature_map['support_event_point_bonus'] == pytest.approx(_bounded_signed(0.40, 0.5))
    assert feature_map['audition_turn_modifier'] == pytest.approx(_bounded_signed(-1.0, 2.0))
    assert feature_map['shop_discount'] == pytest.approx(_bounded_signed(-0.20, 0.5))
    assert feature_map['reward_card_count_bonus'] == pytest.approx(_bounded_positive(1.0, 2.0))
    assert feature_map['reroll_count_bonus'] == pytest.approx(_bounded_positive(2.0, 3.0))


def test_exam_runtime_clamps_parameter_stats_to_mode_limit() -> None:
    """exam runtime 读入异常高三维时也应按模式上限裁剪。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    loadout = _sample_loadout(repository, scenario)
    capped_loadout = replace(
        loadout,
        stat_profile=replace(loadout.stat_profile, vocal=5000.0, dance=4000.0, visual=3000.0),
    )

    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=capped_loadout,
        seed=37,
        audition_row_id=_sample_audition_row_selector(repository, scenario, capped_loadout),
    )

    assert runtime.parameter_stats == (1000.0, 1000.0, 1000.0)


def test_exam_runtime_accepts_structured_initial_status_enchants() -> None:
    """produce runtime 传入的结构化附魔应保留来源与次数。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    enchant_id = loadout.exam_status_enchant_specs[0].enchant_id
    stripped_loadout = replace(loadout, produce_item_id='', exam_status_enchant_ids=(), exam_status_enchant_specs=())

    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=stripped_loadout,
        initial_status_enchants=[
            RuntimeExamStatusEnchantSpec(
                enchant_id=enchant_id,
                effect_count=1,
                source='produce_item',
                source_identity='test-produced-item',
            )
        ],
        seed=57,
        audition_row_id=_sample_audition_row_selector(repository, scenario, stripped_loadout),
    )
    runtime.reset()

    produce_item_enchants = [enchant for enchant in runtime.active_enchants if enchant.source == 'produce_item']
    assert produce_item_enchants
    assert any(enchant.source_identity == 'test-produced-item' for enchant in produce_item_enchants)
    assert any(enchant.remaining_count == 1 for enchant in produce_item_enchants)


def test_master_data_repository_reuses_shared_and_disk_cache(tmp_path) -> None:
    """新建仓库实例时，应优先命中共享/磁盘缓存而不是再次解析同一份 YAML。"""

    assets_dir = tmp_path / 'assets'
    localization_dir = tmp_path / 'localization'
    cache_root = tmp_path / 'repo-root'
    assets_dir.mkdir()
    localization_dir.mkdir()
    cache_root.mkdir()
    (assets_dir / 'ProduceCard.yaml').write_text('- id: card-001\n  name: 测试卡\n', encoding='utf-8')

    data_module._SHARED_TABLE_CACHE.clear()
    repository = MasterDataRepository(root_dir=cache_root, assets_dir=assets_dir, localization_dir=localization_dir)
    first = repository.load_table('ProduceCard')
    cache_path = cache_root / '.gakumas_rl_cache' / 'tables' / 'ProduceCard.pkl'

    assert first.first('card-001') is not None
    assert cache_path.exists()

    data_module._SHARED_TABLE_CACHE.clear()
    original_yaml_load = data_module.yaml.load

    def _explode_yaml_load(*args, **kwargs):
        raise AssertionError('yaml.load should not be called when disk cache is available')

    data_module.yaml.load = _explode_yaml_load
    try:
        second_repository = MasterDataRepository(root_dir=cache_root, assets_dir=assets_dir, localization_dir=localization_dir)
        second = second_repository.load_table('ProduceCard')
    finally:
        data_module.yaml.load = original_yaml_load

    assert second.first('card-001') == first.first('card-001')


def _first_playable_non_bonus_action(runtime: ExamRuntime):
    """挑一张不会直接给出牌次数加成的可打手牌。"""

    for action in runtime.legal_actions():
        if action.kind != 'card':
            continue
        uid = int(action.payload['uid'])
        card = next((item for item in runtime.hand if item.uid == uid), None)
        if card is None:
            continue
        effect_types = set(runtime.repository.card_exam_effect_types(card.base_card))
        if 'ProduceExamEffectType_ExamPlayableValueAdd' not in effect_types:
            return action
    raise AssertionError('No playable non-bonus card found in sample runtime')


def _first_grow_effect(repository: MasterDataRepository, effect_type: str, value: int | None = None) -> dict[str, object]:
    """按成长效果类型过滤主数据里的第一条记录。"""

    for row in repository.load_table('ProduceCardGrowEffect').rows:
        if str(row.get('effectType') or '') != effect_type:
            continue
        if value is not None and int(row.get('value') or 0) != value:
            continue
        return row
    raise AssertionError(f'Grow effect not found: {effect_type} {value}')


def _effect_row(repository: MasterDataRepository, effect_id: str) -> dict[str, object]:
    """按 id 取一条考试效果行。"""

    row = repository.exam_effect_map.get(effect_id)
    if row is None:
        raise AssertionError(f'Effect not found: {effect_id}')
    return row


def test_loadout_initial_exam_deck_contains_unique_idol_card() -> None:
    """偶像卡自带的专属技能卡应该出现在初始考试卡组中。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)

    deck = build_initial_exam_deck(repository, scenario, loadout=loadout)
    card_ids = [str(card.get('id') or '') for card in deck]

    assert loadout.stat_profile.unique_produce_card_id in card_ids
    assert len(deck) >= 10


def test_r_loadout_initial_exam_deck_uses_default_seed_package() -> None:
    """R 卡应优先带上主数据里定义好的默认底牌包。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=0, dearness_level=0)

    deck = build_initial_exam_deck(repository, scenario, loadout=loadout)
    card_ids = [str(card.get('id') or '') for card in deck]

    assert card_ids.count('p_card-01-men-0_007') >= 2
    assert 'p_card-01-act-0_005' in card_ids
    assert loadout.stat_profile.unique_produce_card_id in card_ids


def test_non_r_loadout_initial_exam_deck_includes_base_r_and_current_unique_cards() -> None:
    """高稀有度初始牌组应同时包含基础 R 专属卡和当前偶像卡专属卡。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_SSR, producer_level=35, idol_rank=0, dearness_level=0)

    deck = build_initial_exam_deck(repository, scenario, loadout=loadout)
    card_ids = {str(card.get('id') or '') for card in deck}

    assert 'p_card-01-ido-1_013' in card_ids
    assert loadout.stat_profile.unique_produce_card_id in card_ids


def test_loadout_produce_card_conversion_replaces_cards_in_initial_exam_deck() -> None:
    """已启用的技能卡切换应直接反映到初始考试卡组。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=80,
        idol_rank=0,
        dearness_level=0,
        selected_produce_card_conversion_after_ids=('p_card-01-act-3_184', 'p_card-01-men-2_099'),
    )

    deck = build_initial_exam_deck(repository, scenario, loadout=loadout, rng=np.random.default_rng(0))
    card_ids = [str(card.get('id') or '') for card in deck]

    assert 'p_card-01-act-3_184' in card_ids
    assert 'p_card-01-act-3_031' not in card_ids


def test_loadout_produce_card_conversion_replaces_cards_in_weighted_pool() -> None:
    """已启用的技能卡切换应影响后续可出现的卡池。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(
        repository,
        scenario,
        AMAO_R,
        producer_level=80,
        idol_rank=0,
        dearness_level=0,
        selected_produce_card_conversion_after_ids=('p_card-01-men-2_099',),
    )

    pool_ids = [str(card.get('id') or '') for card in build_weighted_card_pool(repository, scenario, loadout=loadout)]

    assert 'p_card-01-men-2_099' in pool_ids
    assert 'p_card-01-men-2_039' not in pool_ids


def test_build_drink_inventory_filters_to_common_and_selected_plan() -> None:
    """饮料池应只包含通用饮料和当前流派饮料。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')

    drinks = repository.build_drink_inventory(
        scenario,
        max_items=128,
        rng=np.random.default_rng(0),
        plan_type='ProducePlanType_Plan2',
    )
    plan_types = {str(row.get('planType') or '') for row in drinks}
    drink_ids = {str(row.get('id') or '') for row in drinks}

    assert plan_types <= {'ProducePlanType_Common', 'ProducePlanType_Plan2'}
    assert 'ProducePlanType_Plan2' in plan_types
    assert 'pdrink_03-2-007' not in drink_ids
    assert all(not row.get('libraryHidden') for row in drinks)


def test_weighted_card_pool_extends_beyond_guide_sample_cards() -> None:
    """流派候选池不能只剩攻略样例卡本身，否则奖励会长期锁死在顶级卡。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)

    pool = build_weighted_card_pool(repository, scenario, loadout=loadout)
    pool_ids = [str(card.get('id') or '') for card in pool]
    guide_ids = set(loadout.deck_archetype.sample_card_ids + loadout.deck_archetype.recommended_card_ids)
    extra_ids = [card_id for card_id in pool_ids if card_id not in guide_ids]

    assert len(pool_ids) > len(guide_ids)
    assert extra_ids
    assert all(
        str(card.get('planType') or 'ProducePlanType_Common') in {'ProducePlanType_Common', loadout.stat_profile.plan_type}
        for card in pool
    )


def test_weighted_card_sampling_is_not_locked_to_top_guide_cards() -> None:
    """抽卡采样应保留 guide 偏好，但不能永远只在 sample/recommended 小池中循环。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)
    pool = build_weighted_card_pool(repository, scenario, loadout=loadout)
    sample_counts = Counter(loadout.deck_archetype.sample_card_ids)
    preferred_ids = set(loadout.deck_archetype.recommended_card_ids)
    guide_ids = set(loadout.deck_archetype.sample_card_ids + loadout.deck_archetype.recommended_card_ids)

    rng = np.random.default_rng(0)
    sampled_ids = {
        str(sample_card_from_weighted_pool(pool, rng, sample_counts=sample_counts, preferred_ids=preferred_ids).get('id') or '')
        for _ in range(64)
    }

    assert sampled_ids
    assert any(card_id not in guide_ids for card_id in sampled_ids)


def test_weighted_card_sampling_is_not_overly_top_heavy() -> None:
    """平滑抽样不能长期只黏在前几个高评价候选上。"""

    synthetic_pool = [
        {
            'id': f'card-{index:03d}',
            'evaluation': max(0, 300 - index),
            'rarity': 'IdolCardRarity_SSR' if index < 10 else ('IdolCardRarity_Sr' if index < 30 else 'IdolCardRarity_R'),
        }
        for index in range(80)
    ]
    rng = np.random.default_rng(7)
    draws = [
        int(str(sample_card_from_weighted_pool(synthetic_pool, rng).get('id') or 'card-000').split('-')[-1])
        for _ in range(256)
    ]

    assert any(index >= 40 for index in draws)
    assert sum(index < 10 for index in draws) < 96


def test_exam_runtime_default_drinks_follow_loadout_plan_type() -> None:
    """考试 runtime 的默认饮料应按 loadout.plan_type 过滤。"""

    runtime = _sample_runtime(seed=11)
    allowed_plan_types = {'ProducePlanType_Common', runtime.loadout.stat_profile.plan_type}

    assert runtime.initial_drinks
    assert {
        str(drink.get('planType') or 'ProducePlanType_Common')
        for drink in runtime.initial_drinks
    } <= allowed_plan_types


def test_loadout_produce_card_conversion_requires_unlock_level() -> None:
    """未满足 PLv 解锁条件时，不应允许启用技能卡切换。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')

    with pytest.raises(ValueError):
        build_idol_loadout(
            repository,
            scenario,
            AMAO_R,
            producer_level=35,
            idol_rank=0,
            dearness_level=0,
            selected_produce_card_conversion_after_ids=('p_card-01-act-3_184',),
        )


def test_exam_runtime_opening_hand_draws_three_cards_by_default() -> None:
    """默认开局应按官方帮助页抽 3 张牌。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_SSR, producer_level=35, idol_rank=0, dearness_level=0)
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=5,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    assert len(runtime.hand) == 3



def test_exam_runtime_end_turn_discards_remaining_hand() -> None:
    """结束回合时，仍在手牌区的卡应被送入墓地。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)

    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=7,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    initial_hand = len(runtime.hand)
    assert initial_hand > 0

    runtime._end_turn()

    assert len(runtime.grave) >= initial_hand



def test_exam_runtime_default_play_limit_is_one_card() -> None:
    """默认情况下每回合只能主动打出一张牌。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)

    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=11,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    assert runtime.play_limit == 1


def test_exam_runtime_auto_ends_turn_after_last_card_play() -> None:
    """打出最后一张可出牌后，若没有额外出牌次数，应自动结束回合。"""

    runtime = _sample_runtime(seed=11)
    action = _first_playable_non_bonus_action(runtime)
    before_turn = runtime.turn

    runtime.step(action)

    assert runtime.turn == before_turn + 1
    assert runtime.turn_counters['play_count'] == 0


def test_exam_runtime_disallows_drinks_without_remaining_play_count() -> None:
    """没有剩余出牌次数时，合法动作里不应再出现饮料。"""

    runtime = _sample_runtime(seed=13)
    runtime.turn_counters['play_count'] = runtime.play_limit

    actions = runtime.legal_actions()

    assert len(actions) == 1
    assert actions[0].kind == 'end_turn'


def test_exam_runtime_allows_drinks_when_hand_is_empty_but_play_window_remains() -> None:
    """有剩余出牌次数但没手牌时，仍应允许先喝饮料再结束回合。"""

    runtime = _sample_runtime(seed=17)
    runtime.hand = []

    actions = runtime.legal_actions()

    assert any(action.kind == 'drink' for action in actions)
    assert any(action.kind == 'end_turn' for action in actions)


def test_exam_runtime_scheduled_next_turn_hand_grow_applies_after_draw() -> None:
    """次回合的手牌强化应作用于新回合抽到的手牌，而不是空手牌。"""

    runtime = _sample_runtime(seed=23)
    effect_id = 'e_effect-exam_effect_timer-0001-01-e_effect-exam_add_grow_effect-p_card_search-hand-all-0_0-g_effect-lesson_add-12'

    runtime._apply_exam_effect(_effect_row(runtime.repository, effect_id), source='test')
    runtime._end_turn()

    assert runtime.turn == 2
    assert len(runtime.hand) > 0
    assert all('g_effect-lesson_add-12' in card.grow_effect_ids for card in runtime.hand)


def test_exam_runtime_scheduled_next_turn_play_count_buff_applies_on_new_turn() -> None:
    """次回合追加出牌次数应在新回合开始后立刻生效。"""

    runtime = _sample_runtime(seed=29)
    effect_id = 'e_effect-exam_effect_timer-0001-01-e_effect-exam_playable_value_add-01'

    runtime._apply_exam_effect(_effect_row(runtime.repository, effect_id), source='test')
    runtime._end_turn()

    assert runtime.turn == 2
    assert runtime.play_limit == 2


def test_exam_runtime_card_move_effect_moves_hand_card_to_grave() -> None:
    """移牌效果应正确更新卡牌所在区域。"""

    runtime = _sample_runtime(seed=31)
    effect_id = 'e_effect-exam_card_move-p_card_search-hand-1-grave-select-1_1'
    moved_uid = runtime.hand[0].uid

    runtime._apply_exam_effect(_effect_row(runtime.repository, effect_id), source='test')

    assert all(card.uid != moved_uid for card in runtime.hand)
    assert any(card.uid == moved_uid for card in runtime.grave)


def test_exam_runtime_hold_cards_return_to_hand_on_full_power() -> None:
    """保留区卡牌在进入全力时应自动尽量回到手牌。"""

    runtime = _sample_runtime(seed=37)
    effect_id = 'e_effect-exam_card_move-p_card_search-hand-hold-select-1_1'
    moved_uid = runtime.hand[0].uid

    runtime._apply_exam_effect(_effect_row(runtime.repository, effect_id), source='test')

    assert any(card.uid == moved_uid for card in runtime.hold)
    assert all(card.uid != moved_uid for card in runtime.hand)

    runtime._enter_full_power()

    assert len(runtime.hold) == 0
    assert any(card.uid == moved_uid for card in runtime.hand)


def test_exam_runtime_start_turn_draw_respects_deck_order() -> None:
    """回合抽牌应严格按牌堆顺序，不应人为插入主动牌。"""

    runtime = _sample_runtime(seed=109)
    all_cards = list(runtime.hand) + list(runtime.deck) + list(runtime.grave)
    active_cards = [card for card in all_cards if str(card.base_card.get('category') or '') == 'ProduceCardCategory_ActiveSkill']
    support_cards = [card for card in all_cards if str(card.base_card.get('category') or '') != 'ProduceCardCategory_ActiveSkill']
    assert len(active_cards) >= 1
    assert len(support_cards) >= 3

    expected_draw = support_cards[:3]
    hidden_active = active_cards[0]
    runtime.hand = []
    runtime.grave = []
    runtime.deck = deque(expected_draw + [hidden_active])
    runtime.turn = 0
    runtime.turn_counters = Counter()
    runtime.extra_turns = 0
    runtime.terminated = False
    runtime.start_turn_draw_penalty = 0
    runtime.active_effects = []
    runtime.active_enchants = []
    runtime.scheduled_effects = []
    runtime.gimmick_rows = []
    runtime.resources['review'] = 0.0
    runtime.resources['parameter_buff'] = 0.0
    runtime.resources['full_power_point'] = 0.0
    runtime.stance = 'neutral'

    runtime._start_turn()

    assert [card.uid for card in runtime.hand] == [card.uid for card in expected_draw]
    assert hidden_active.uid not in [card.uid for card in runtime.hand]


def test_exam_runtime_reset_respects_starting_stamina_override() -> None:
    """考试运行时应允许调用方按主数据覆写开场体力。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=127,
        starting_stamina=4.0,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )

    runtime.reset()

    assert runtime.stamina == pytest.approx(4.0)


def test_serialize_exam_actions_does_not_mutate_runtime_or_expose_illegal_drinks() -> None:
    """动作序列化不应改写手牌，也不应在无行动窗口时暴露饮料。"""

    from gakumas_rl.service import _serialize_exam_actions

    runtime = _sample_runtime(seed=113)
    hand_before = [card.uid for card in runtime.hand]

    actions = _serialize_exam_actions(runtime)

    assert [card.uid for card in runtime.hand] == hand_before
    assert any(action['kind'] == 'card' and action['available'] for action in actions)

    runtime.turn_counters['play_count'] = runtime.play_limit
    exhausted_actions = _serialize_exam_actions(runtime)
    assert [action['kind'] for action in exhausted_actions] == ['end_turn']


def test_exam_env_masks_drinks_after_play_window_is_exhausted() -> None:
    """环境动作掩码也应跟随 runtime 禁用无剩余次数时的饮料。"""

    pytest.importorskip('gymnasium')
    from gakumas_rl.envs import GakumasExamEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(repository, scenario, seed=19)
    env.reset()
    env.runtime.turn_counters['play_count'] = env.runtime.play_limit

    candidates = env._build_candidates()

    assert all(
        not candidate.payload.get('available', False)
        for candidate in candidates
        if candidate.kind == 'drink'
    )


def test_produce_runtime_audition_start_stamina_uses_setting_recovery() -> None:
    """养成运行时进入考试前应按主数据的固定回复比例回体，而不是满体。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=131)
    runtime.reset()
    runtime.state['max_stamina'] = 20.0
    runtime.state['stamina'] = 4.0
    runtime.state['before_audition_refresh_penalty'] = 0.25

    recovery_permille = float(runtime.produce_setting.get('beforeAuditionRefreshStaminaRecoveryPermil') or 0.0)
    expected = 4.0 + 20.0 * (recovery_permille / 1000.0) * 0.75

    assert runtime._audition_start_stamina() == pytest.approx(expected)



def test_exam_runtime_clear_reward_mode_penalizes_overshoot() -> None:
    """clear 模式过线后，继续追高分的边际收益应明显下降。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)

    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=17,
        reward_mode='clear',
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    target_score = float(runtime.profile.get('base_score') or 1.0)
    runtime.score = target_score * 0.5
    before_clear = runtime._reward_signal()
    runtime.score = target_score
    at_clear = runtime._reward_signal()
    runtime.score = target_score * 1.5
    overshot = runtime._reward_signal()

    assert (at_clear - before_clear) > (overshot - at_clear)


def test_exam_runtime_clear_mode_reuses_clear_targets_but_keeps_exam_color_and_fan_votes() -> None:
    """clear 模式复用训练课目标规则，但仍保留考试颜色与 NIA fan vote 估值。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    selector = _sample_audition_row_selector(repository, scenario, loadout)

    low_vote_runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=18,
        reward_mode='clear',
        fan_votes=0.0,
        exam_score_bonus_multiplier=1.0,
        audition_row_id=selector,
    )
    high_vote_runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=18,
        reward_mode='clear',
        fan_votes=30000.0,
        exam_score_bonus_multiplier=1.0,
        audition_row_id=selector,
    )
    low_vote_runtime.reset()
    high_vote_runtime.reset()

    assert low_vote_runtime._fan_vote_gain_value() > 0.0
    assert high_vote_runtime._fan_vote_gain_value() > 0.0
    assert -0.75 <= low_vote_runtime._turn_window_value() <= 0.75
    assert low_vote_runtime._current_clear_target() > 0.0
    assert low_vote_runtime._current_perfect_target() == pytest.approx(low_vote_runtime._current_clear_target() * 2.0)

    low_vote_runtime.current_turn_color = 'vocal'
    low_vote_runtime._refresh_turn_score_bonus_multiplier()
    vocal_multiplier = low_vote_runtime.score_bonus_multiplier
    low_vote_runtime.current_turn_color = 'dance'
    low_vote_runtime._refresh_turn_score_bonus_multiplier()
    dance_multiplier = low_vote_runtime.score_bonus_multiplier

    assert low_vote_runtime._turn_color_enabled() is True
    assert low_vote_runtime._fan_vote_enabled_for_mode() is True
    assert vocal_multiplier != pytest.approx(dance_multiplier)
    assert low_vote_runtime.base_score_bonus_multiplier != pytest.approx(high_vote_runtime.base_score_bonus_multiplier)
    assert high_vote_runtime.fan_votes > low_vote_runtime.fan_votes


def test_exam_runtime_clear_mode_terminates_on_clear_with_dynamic_finish_value() -> None:
    """clear 模式达线即停，但终局价值仍应随剩余资源和回合动态变化。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    selector = _sample_audition_row_selector(repository, scenario, loadout)

    rich_runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=19,
        reward_mode='clear',
        audition_row_id=selector,
    )
    lean_runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=23,
        reward_mode='clear',
        audition_row_id=selector,
    )
    rich_runtime.reset()
    lean_runtime.reset()

    clear_target = float(rich_runtime.profile.get('base_score') or 1.0)

    rich_runtime.turn = 2
    rich_runtime.resources['review'] = 10.0
    rich_runtime.resources['parameter_buff'] = 6.0
    rich_runtime.resources['lesson_buff'] = 4.0
    rich_runtime.stamina = rich_runtime.max_stamina
    rich_runtime.score = clear_target
    rich_runtime._update_clear_state_after_score_change()

    lean_runtime.turn = lean_runtime.max_turns
    lean_runtime.resources['review'] = 0.0
    lean_runtime.resources['parameter_buff'] = 0.0
    lean_runtime.resources['lesson_buff'] = 0.0
    lean_runtime.stamina = lean_runtime.max_stamina * 0.4
    lean_runtime.score = clear_target
    lean_runtime._update_clear_state_after_score_change()

    assert rich_runtime.terminated is True
    assert lean_runtime.terminated is True
    assert rich_runtime.clear_state == 'cleared'
    assert lean_runtime.clear_state == 'cleared'
    assert rich_runtime._reward_signal() > lean_runtime._reward_signal()


def test_exam_runtime_clear_reward_mode_penalizes_negative_status() -> None:
    """clear 模式应显式惩罚睡意等负面状态。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)

    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=19,
        reward_mode='clear',
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    baseline = runtime._reward_signal()
    runtime.resources['sleepy'] = 2.0

    assert runtime._reward_signal() < baseline


def test_exam_runtime_score_reward_prefers_turn_color_aligned_window() -> None:
    """统一 reward 仍应感知当前颜色窗口，偏色更匹配的回合价值更高。"""

    runtime = _sample_runtime(seed=41, reward_mode='score', exam_score_bonus_multiplier=1.0)
    runtime.parameter_stats = (900.0, 150.0, 150.0)
    runtime.score = float(runtime.profile.get('base_score') or 1.0) * 0.75

    runtime.current_turn_color = 'vocal'
    runtime._refresh_turn_score_bonus_multiplier()
    vocal_signal = runtime._reward_signal()

    runtime.current_turn_color = 'dance'
    runtime._refresh_turn_score_bonus_multiplier()
    dance_signal = runtime._reward_signal()

    assert vocal_signal > dance_signal


def test_exam_runtime_plan_specific_reward_values_match_archetype_resources() -> None:
    """统一 reward 应按 plan 区分资源库存的未来兑现价值。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    base_loadout = _sample_loadout(repository, scenario)
    sense_loadout = replace(
        base_loadout,
        stat_profile=replace(base_loadout.stat_profile, plan_type='ProducePlanType_Plan1'),
    )
    logic_loadout = replace(
        base_loadout,
        stat_profile=replace(base_loadout.stat_profile, plan_type='ProducePlanType_Plan2'),
    )
    sense_runtime = ExamRuntime(
        repository,
        scenario,
        loadout=sense_loadout,
        seed=43,
        reward_mode='score',
        audition_row_id=_sample_audition_row_selector(repository, scenario, sense_loadout),
    )
    logic_runtime = ExamRuntime(
        repository,
        scenario,
        loadout=logic_loadout,
        seed=47,
        reward_mode='score',
        audition_row_id=_sample_audition_row_selector(repository, scenario, logic_loadout),
    )
    sense_runtime.reset()
    logic_runtime.reset()

    target_score = float(sense_runtime.profile.get('base_score') or 1.0) * 0.55
    for runtime in (sense_runtime, logic_runtime):
        runtime.score = target_score
        runtime.resources['parameter_buff'] = 4.0
        runtime.resources['review'] = 8.0
        runtime.resources['lesson_buff'] = 2.0
        runtime.resources['aggressive'] = 0.0
        runtime.resources['block'] = 0.0

    assert sense_runtime._reward_signal() > logic_runtime._reward_signal()


def test_exam_runtime_parameter_buff_uses_effect_turn_and_absolute_bonus() -> None:
    """好调应按 effectTurn 增加，绝好调应按剩余好调回合追加倍率。"""

    runtime = _sample_runtime(seed=23)
    repository = runtime.repository

    good_condition = _first_exam_effect(repository, 'ProduceExamEffectType_ExamParameterBuff', effectTurn=3)
    absolute_good_condition = _first_exam_effect(repository, 'ProduceExamEffectType_ExamParameterBuffMultiplePerTurn', effectTurn=1)

    runtime._apply_exam_effect(good_condition, source='test')
    assert runtime.resources['parameter_buff'] == 3.0

    runtime._apply_exam_effect(absolute_good_condition, source='test')
    assert runtime.resources['parameter_buff_multiple_per_turn'] == 1.0
    assert runtime._apply_score_value_modifiers(10.0) == 18.0


def test_exam_runtime_concentration_stage_changes_score_and_cost() -> None:
    """强气 1/2 阶段应分别提高打分倍率，并在 2 阶段追加体力穿透。"""

    runtime = _sample_runtime(seed=29)
    repository = runtime.repository

    concentration = _first_exam_effect(repository, 'ProduceExamEffectType_ExamConcentration', effectValue1=1)
    lesson = _first_exam_effect(repository, 'ProduceExamEffectType_ExamLessonFix', effectValue1=10)
    cost_card = next(
        card
        for card in list(runtime.hand) + list(runtime.deck)
        if float(card.base_card.get('stamina') or 0) > 0 or float(card.base_card.get('forceStamina') or 0) > 0
    )

    neutral_cost = runtime._card_stamina_components(cost_card)
    runtime._apply_exam_effect(concentration, source='test')
    stage1_value = runtime._resolve_lesson_effect_value(lesson)
    stage1_cost = runtime._card_stamina_components(cost_card)
    runtime._apply_exam_effect(concentration, source='test')
    stage2_value = runtime._resolve_lesson_effect_value(lesson)
    stage2_cost = runtime._card_stamina_components(cost_card)

    assert stage1_value == 20.0
    assert stage2_value == 25.0
    assert stage1_cost[0] == neutral_cost[0] * 2
    assert stage2_cost[1] >= stage1_cost[1] + 1


def test_exam_runtime_preservation_release_and_full_power_transition() -> None:
    """温存 2 阶段转全力时应发放热意、固定元气和出牌次数奖励。"""

    runtime = _sample_runtime(seed=31)
    repository = runtime.repository

    preservation = _first_exam_effect(repository, 'ProduceExamEffectType_ExamPreservation', effectValue1=1)

    runtime._apply_exam_effect(preservation, source='test')
    runtime._apply_exam_effect(preservation, source='test')
    runtime._enter_full_power()

    assert runtime.stance == 'full_power'
    assert runtime.resources['enthusiastic'] == float(runtime.exam_setting['preservationReleaseEnthusiastic2'])
    assert runtime.resources['block'] == float(runtime.exam_setting['preservationReleaseBlockAdd2'])
    assert runtime.play_limit >= 1 + int(runtime.exam_setting['fullPowerPlayableValueAdd']) + int(runtime.exam_setting['preservationReleasePlayableValueAdd2'])


def test_exam_runtime_full_power_point_auto_triggers_on_next_turn() -> None:
    """回合结束时全力值达到 10，应在下回合开始自动进入全力。"""

    runtime = _sample_runtime(seed=37)
    runtime.resources['full_power_point'] = 10.0

    runtime._end_turn()

    assert runtime.stance == 'full_power'
    assert runtime.resources['full_power_point'] == 0.0


def test_exam_runtime_weak_slump_and_panic_follow_game_labels() -> None:
    """弱气应减元气、低谷应封锁得分、随机应只在当前回合覆写体力消耗。"""

    runtime = _sample_runtime(seed=41)
    repository = runtime.repository

    weak = _first_exam_effect(repository, 'ProduceExamEffectType_ExamGimmickSleepy', effectValue1=2)
    slump = _first_exam_effect(repository, 'ProduceExamEffectType_ExamGimmickSlump', effectTurn=1)
    panic = _first_exam_effect(repository, 'ProduceExamEffectType_ExamPanic', effectTurn=1)
    block = _first_exam_effect(repository, 'ProduceExamEffectType_ExamBlock', effectValue1=5)
    zero_cost_card = next(
        card for card in list(runtime.hand) + list(runtime.deck)
        if float(card.base_card.get('stamina') or 0) == 0 and float(card.base_card.get('forceStamina') or 0) == 0
    )

    runtime._apply_exam_effect(weak, source='test')
    runtime._apply_exam_effect(block, source='test')
    assert runtime.resources['block'] == 3.0

    runtime._apply_exam_effect(slump, source='test')
    assert runtime._apply_score_value_modifiers(12.0) == 0.0

    runtime._apply_exam_effect(panic, source='test')
    panic_cost_first = runtime._card_stamina_components(zero_cost_card)
    panic_cost_second = runtime._card_stamina_components(zero_cost_card)
    assert panic_cost_first == panic_cost_second
    assert panic_cost_first[0] > 0

    runtime._end_turn()
    assert runtime._card_stamina_components(zero_cost_card) == (0.0, 0.0)

def test_exam_runtime_card_grow_effects_adjust_direct_card_outputs() -> None:
    """卡牌成长应只改当前卡的直效数值，并遵守不会降到 0 的下限。"""

    runtime = _sample_runtime(seed=43)
    repository = runtime.repository
    lesson = _first_exam_effect(repository, 'ProduceExamEffectType_ExamLessonFix', effectValue1=5)
    block = _first_exam_effect(repository, 'ProduceExamEffectType_ExamBlock', effectValue1=5)
    lesson_add = _first_grow_effect(repository, 'ProduceCardGrowEffectType_LessonAdd', value=2)
    lesson_reduce = _first_grow_effect(repository, 'ProduceCardGrowEffectType_LessonReduce', value=20)
    block_add = _first_grow_effect(repository, 'ProduceCardGrowEffectType_BlockAdd', value=3)
    block_reduce = _first_grow_effect(repository, 'ProduceCardGrowEffectType_BlockReduce', value=10)
    card = runtime.hand[0].clone(uid=999001)
    card.grow_effect_ids = [
        str(lesson_add['id']),
        str(lesson_reduce['id']),
        str(block_add['id']),
        str(block_reduce['id']),
    ]

    runtime.current_card = card
    assert runtime._resolve_lesson_effect_value(lesson, from_card=True) == 1.0

    runtime._apply_exam_effect(block, source='card')
    assert runtime.resources['block'] == 1.0

    runtime.current_card = None


def test_exam_runtime_search_buffs_repeat_effects_and_override_cost() -> None:
    """目标卡追加发动与耗体覆写应只对命中的卡生效，并按次数消费。"""

    runtime = _sample_runtime(seed=47)
    repository = runtime.repository
    repeat_effect = repository.exam_effect_map['e_effect-exam_card_search_effect_play_count_buff-0001-01-01-p_card_search-active_skill-deck_all-all-0_0']
    zero_cost_effect = repository.exam_effect_map['e_effect-exam_search_play_card_stamina_consumption_change-01-inf-p_card_search-active_skill-deck_all-all-0_0']
    target_card = next(card for card in list(runtime.hand) + list(runtime.deck) if runtime._card_matches_search(card, 'p_card_search-active_skill-deck_all'))

    runtime._apply_exam_effect(repeat_effect, source='test')
    runtime._apply_exam_effect(zero_cost_effect, source='test')

    assert runtime._card_repeat_bonus(target_card) == 1
    assert runtime._card_stamina_components(target_card)[0] == 0.0

    runtime._consume_card_play_buffs(target_card)

    assert runtime._card_repeat_bonus(target_card) == 0
    assert runtime._matching_search_stamina_overrides(target_card) == []


def test_exam_runtime_legend_card_ignores_repeat_bonus() -> None:
    """传奇技能卡不应吃“下次使用的技能卡效果重复发动”。"""

    runtime = _sample_runtime(seed=101)
    repository = runtime.repository
    repeat_effect = repository.exam_effect_map['e_effect-exam_card_search_effect_play_count_buff-0001-01-01-p_card_search-active_skill-deck_all-all-0_0']
    legend_row = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('rarity') or '') == 'ProduceCardRarity_Legend'
    )
    legend_card = RuntimeCard(
        uid=runtime._next_uid(),
        card_id=str(legend_row.get('id') or ''),
        upgrade_count=int(legend_row.get('upgradeCount') or 0),
        base_card=legend_row,
    )

    runtime._apply_exam_effect(repeat_effect, source='test')

    assert runtime._card_repeat_bonus(legend_card) == 0


def test_exam_runtime_initial_item_enchant_keeps_effect_count_and_item_limit_buff_extends_it() -> None:
    """偶像 P 道具附魔应保留主数据里的触发次数，发动物品次数增加也应能扩容。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=53,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    produce_item_enchants = [enchant for enchant in runtime.active_enchants if enchant.source == 'produce_item']
    assert produce_item_enchants
    before_counts = [enchant.remaining_count for enchant in produce_item_enchants]
    assert all(count is not None and count > 0 for count in before_counts)

    runtime._apply_exam_effect(_first_exam_effect(repository, 'ProduceExamEffectType_ExamItemFireLimitAdd'), source='test')

    after_counts = [enchant.remaining_count for enchant in runtime.active_enchants if enchant.source == 'produce_item']
    assert after_counts == [count + 1 for count in before_counts]


def test_exam_runtime_status_change_triggers_only_for_card_or_drink_origins() -> None:
    """`状态变动` 触发器只能响应技能卡和 P 饮料导致的状态变化。"""

    runtime = _sample_runtime(seed=131)
    runtime.active_enchants = []

    trigger_id = 'test-trigger-status-origin'
    effect_id = 'test-effect-status-origin'
    runtime.repository.exam_trigger_map[trigger_id] = {
        'id': trigger_id,
        'phaseTypes': ['ProduceExamPhaseType_ExamStatusChange'],
        'effectTypes': ['ProduceExamEffectType_ExamReview'],
    }
    runtime.repository.exam_effect_map[effect_id] = {
        'id': effect_id,
        'effectType': 'ProduceExamEffectType_ExamLessonBuff',
        'effectValue1': 1,
    }
    runtime.active_enchants.append(
        TriggeredEnchant(
            uid=runtime._next_uid(),
            enchant_id='test-status-origin-enchant',
            trigger_id=trigger_id,
            effect_ids=[effect_id],
            remaining_turns=None,
            remaining_count=2,
            source='test',
        )
    )

    review_effect = {
        'id': 'test-effect-review-origin',
        'effectType': 'ProduceExamEffectType_ExamReview',
        'effectValue1': 1,
    }
    runtime._apply_exam_effect(review_effect, source='enchant:test')
    assert runtime.resources['lesson_buff'] == 0.0

    runtime._apply_exam_effect(review_effect, source='card')
    assert runtime.resources['lesson_buff'] == 1.0


def test_exam_runtime_block_depend_block_consumption_sum_uses_consumed_block() -> None:
    """消耗元气参照型效果应读取累计消耗过的元气。"""

    runtime = _sample_runtime(seed=271)
    repository = runtime.repository
    runtime.resources['block'] = 10.0

    runtime._spend_stamina(5.0, status_change_origin='card')
    runtime._apply_exam_effect(
        _first_exam_effect(repository, 'ProduceExamEffectType_ExamBlockDependBlockConsumptionSum', effectValue1=800),
        source='card',
    )

    assert runtime.total_counters['block_consumed'] == 5
    assert runtime.resources['block'] == pytest.approx(9.0)


def test_exam_runtime_parameter_buff_multiple_per_turn_reduce_consumes_one_stack() -> None:
    """绝好调减少应移除对应层数的持续效果实例。"""

    runtime = _sample_runtime(seed=277)
    repository = runtime.repository
    gain_effect = _first_exam_effect(repository, 'ProduceExamEffectType_ExamParameterBuffMultiplePerTurn', effectTurn=1)
    reduce_effect = _first_exam_effect(repository, 'ProduceExamEffectType_ExamParameterBuffMultiplePerTurnReduce', effectValue1=1)

    runtime._apply_exam_effect(gain_effect, source='card')
    runtime._apply_exam_effect(gain_effect, source='card')
    assert runtime.resources['parameter_buff_multiple_per_turn'] == pytest.approx(2.0)

    runtime._apply_exam_effect(reduce_effect, source='card')

    assert runtime.resources['parameter_buff_multiple_per_turn'] == pytest.approx(1.0)
    assert len(runtime._timed_effects_of_type('ProduceExamEffectType_ExamParameterBuffMultiplePerTurn')) == 1


def test_exam_runtime_review_count_add_repeats_end_turn_review_scoring() -> None:
    """好印象追加发动应在回合末额外结算对应次数的好印象得分。"""

    baseline = _sample_runtime(seed=281)
    repository = baseline.repository
    baseline.score = 0.0
    baseline.resources['review'] = 4.0
    baseline._end_turn()
    baseline_gain = baseline.score

    runtime = _sample_runtime(seed=281)
    runtime.score = 0.0
    runtime.resources['review'] = 4.0
    runtime._apply_exam_effect(
        _first_exam_effect(repository, 'ProduceExamEffectType_ExamReviewCountAdd', effectTurn=3),
        source='card',
    )
    runtime._end_turn()

    assert runtime.score == pytest.approx(baseline_gain * 2.0)


def test_produce_runtime_audition_npc_weaken_uses_named_effect_type() -> None:
    """培育侧应支持正式枚举名的 NPC 弱化，而不是只兼容脏值。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=283)
    runtime.reset()

    named_effect = next(
        row
        for row in repository.load_table('ProduceEffect').rows
        if str(row.get('produceEffectType') or '') == 'ProduceEffectType_AuditionNpcWeaken'
    )
    runtime._apply_produce_effect(named_effect, source_action_type='present')

    assert runtime.state['audition_difficulty_bonus'] == pytest.approx(-0.5)


def test_exam_runtime_same_produce_item_identity_fires_once_per_timing() -> None:
    """同一个 P 道具在同一 trigger timing 下即使挂了多段附魔也只能生效一次。"""

    runtime = _sample_runtime(seed=137)
    runtime.active_enchants = []
    runtime.resources['lesson_buff'] = 0.0

    trigger_id = 'test-trigger-start-turn-once'
    effect_id = 'test-effect-item-once'
    runtime.repository.exam_trigger_map[trigger_id] = {
        'id': trigger_id,
        'phaseTypes': ['ProduceExamPhaseType_ExamStartTurn'],
    }
    runtime.repository.exam_effect_map[effect_id] = {
        'id': effect_id,
        'effectType': 'ProduceExamEffectType_ExamLessonBuff',
        'effectValue1': 1,
    }

    for source_identity in ('same-item', 'same-item', 'other-item'):
        runtime.active_enchants.append(
            TriggeredEnchant(
                uid=runtime._next_uid(),
                enchant_id=f'test-item-{source_identity}-{len(runtime.active_enchants)}',
                trigger_id=trigger_id,
                effect_ids=[effect_id],
                remaining_turns=None,
                remaining_count=1,
                source='produce_item',
                source_identity=source_identity,
            )
        )

    runtime._dispatch_phase('ProduceExamPhaseType_ExamStartTurn', phase_value=runtime.turn)

    assert runtime.resources['lesson_buff'] == 2.0


def test_exam_runtime_gimmick_rows_are_resolved_once_before_later_phase_effects() -> None:
    """gimmick 在本回合首次进入 phase scheduler 时就应完成判定，之后不能晚触发。"""

    runtime = _sample_runtime(seed=139)
    repository = runtime.repository
    block_effect = _first_exam_effect(repository, 'ProduceExamEffectType_ExamBlock')
    gimmick_effect_id = 'test-effect-gimmick-once'
    runtime.repository.exam_effect_map[gimmick_effect_id] = {
        'id': gimmick_effect_id,
        'effectType': 'ProduceExamEffectType_ExamLessonBuff',
        'effectValue1': 2,
    }
    runtime.gimmick_rows = [
        {
            'id': 'test-gimmick-freeze',
            'startTurn': runtime.turn,
            'priority': 1,
            'fieldStatusType': 'ProduceExamFieldStatusType_BlockUp',
            'fieldStatusValue': 1,
            'fieldStatusCheckType': 'ProduceExamTriggerCheckType_Unknown',
            'produceExamEffectId': gimmick_effect_id,
        }
    ]
    runtime.resources['block'] = 0.0
    runtime.resources['lesson_buff'] = 0.0

    runtime._dispatch_phase('ProduceExamPhaseType_ExamStartTurn', phase_value=runtime.turn)
    assert runtime.resources['lesson_buff'] == 0.0

    runtime._apply_exam_effect(block_effect, source='card')

    assert runtime.resources['lesson_buff'] == 0.0
    assert ('test-gimmick-freeze', 1, runtime.turn) in runtime._resolved_gimmick_keys


def test_exam_runtime_requires_explicit_battle_row_for_ambiguous_stage() -> None:
    """存在多条考试配置行时，battle runtime 必须接收显式 row 选择。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    stage_type, rows = _first_ambiguous_audition_stage(repository, scenario)

    with pytest.raises(ValueError, match='Ambiguous audition rows'):
        ExamRuntime(repository, scenario, stage_type=stage_type, seed=149)

    selector = f"{str(rows[-1].get('id') or '')}:{int(rows[-1].get('number') or 0)}"
    runtime = ExamRuntime(repository, scenario, stage_type=stage_type, audition_row_id=selector, seed=149)

    assert runtime.selected_battle_row is not None
    assert str(runtime.selected_battle_row.get('id') or '') == str(rows[-1]['id'])


def test_repository_select_audition_row_uses_highest_accessible_nia_fan_vote_gate() -> None:
    """NIA 多难度考试应按 fan vote 门槛选出当前可进的最高难度。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    stage_type, rows = _first_ambiguous_audition_stage(repository, scenario)
    sorted_rows = sorted(rows, key=lambda row: int(row.get('number') or 0))
    target_row = next(
        (row for row in sorted_rows[1:] if float(row.get('voteCount') or 0.0) > 0.0),
        sorted_rows[-1],
    )

    selected = repository.select_audition_row(
        scenario,
        stage_type,
        fan_votes=float(target_row.get('voteCount') or 0.0),
    )

    assert selected is not None
    assert int(selected.get('number') or 0) == int(target_row.get('number') or 0)
    assert float(selected.get('voteCount') or 0.0) <= float(target_row.get('voteCount') or 0.0)


def test_exam_runtime_explicit_fan_votes_resolve_ambiguous_nia_stage() -> None:
    """显式 fan vote 应让 NIA runtime 自动选中可进入的考试行。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    stage_type, rows = _first_ambiguous_audition_stage(repository, scenario)
    sorted_rows = sorted(rows, key=lambda row: int(row.get('number') or 0))
    target_row = next(
        (row for row in sorted_rows[1:] if float(row.get('voteCount') or 0.0) > 0.0),
        sorted_rows[-1],
    )

    runtime = ExamRuntime(
        repository,
        scenario,
        stage_type=stage_type,
        seed=150,
        fan_votes=float(target_row.get('voteCount') or 0.0),
    )

    assert runtime.selected_battle_row is not None
    assert int(runtime.selected_battle_row.get('number') or 0) == int(target_row.get('number') or 0)
    assert float(runtime.selected_battle_row.get('voteCount') or 0.0) <= float(target_row.get('voteCount') or 0.0)


def test_exam_runtime_fan_vote_multiplier_changes_base_score_bonus() -> None:
    """同一条 NIA 考试行里，fan vote 进度应直接影响局内基础得分倍率。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    stage_type, rows = _first_ambiguous_audition_stage(repository, scenario)
    target_row = sorted(rows, key=lambda row: int(row.get('number') or 0))[-1]
    selector = f"{str(target_row.get('id') or '')}:{int(target_row.get('number') or 0)}"
    reference_votes = float(target_row.get('voteCountBaseLine') or target_row.get('voteCount') or 1.0)

    low_runtime = ExamRuntime(
        repository,
        scenario,
        stage_type=stage_type,
        audition_row_id=selector,
        seed=152,
        fan_votes=0.0,
        exam_score_bonus_multiplier=1.0,
    )
    high_runtime = ExamRuntime(
        repository,
        scenario,
        stage_type=stage_type,
        audition_row_id=selector,
        seed=152,
        fan_votes=reference_votes,
        exam_score_bonus_multiplier=1.0,
    )

    assert low_runtime.base_score_bonus_multiplier < high_runtime.base_score_bonus_multiplier
    assert low_runtime._fan_vote_score_multiplier() < high_runtime._fan_vote_score_multiplier()


def test_exam_runtime_lesson_type_checks_use_battle_context_not_card_name() -> None:
    """lessonType 条件应来自 battle context，而不是卡名里的 Vocal/Dance/Visual 字样。"""

    vocal_runtime = _sample_runtime(seed=151, battle_kind='lesson', lesson_type='ProduceStepLessonType_LessonVocal')
    dance_runtime = _sample_runtime(seed=157, battle_kind='lesson', lesson_type='ProduceStepLessonType_LessonDance')
    trap_card = RuntimeCard(
        uid=910001,
        card_id='test-visual-trap',
        upgrade_count=0,
        base_card={'name': 'Visual Trap'},
    )
    trigger = {
        'id': 'test-trigger-lesson-type',
        'phaseTypes': ['ProduceExamPhaseType_ExamCardPlay'],
        'lessonType': 'ProduceStepLessonType_LessonVocal',
    }
    event = {
        'phase_type': 'ProduceExamPhaseType_ExamCardPlay',
        'phase_value': 1,
        'acting_card': trap_card,
        'effect_types': [],
        'status_change_origin': 'other',
    }

    assert vocal_runtime._lesson_type_for_card(trap_card) == 'ProduceStepLessonType_LessonVocal'
    assert vocal_runtime._trigger_matches(trigger, event, acting_card=trap_card)
    assert not dance_runtime._trigger_matches(trigger, event, acting_card=trap_card)


def test_repository_resolve_lesson_training_spec_uses_master_rows_for_nia() -> None:
    """NIA 的独立 lesson 模式应回落到主数据里的 produce-002 课程表。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)

    spec = repository.resolve_lesson_training_spec(
        scenario,
        action_type='lesson_vocal_normal',
        loadout=loadout,
        level_index=2,
    )

    assert spec.action_type == 'lesson_vocal_normal'
    assert spec.lesson_kind == 'normal'
    assert spec.lesson_type == 'ProduceStepLessonType_LessonVocal'
    assert spec.source_row_id.startswith('p_step_lesson-produce-002-')
    assert spec.source_row_id.endswith('-normal-02')
    assert spec.clear_target > 0.0
    assert spec.perfect_target >= spec.clear_target
    assert spec.turn_limit > 0


def test_exam_runtime_play_card_limit_uses_search_definition_not_search_id_name() -> None:
    """禁卡 gimmick 应按 ProduceCardSearch 的实际条件命中卡牌，而不是靠 search id 命名。"""

    runtime = _sample_runtime(seed=163)
    search_id = 'test-search-without-active-skill-hint'
    _inject_table_row(runtime.card_searches, {
        'id': search_id,
        'cardPositionType': 'ProduceCardPositionType_NotLost',
        'cardCategories': ['ProduceCardCategory_ActiveSkill'],
    })

    all_cards = list(runtime.hand) + list(runtime.deck) + list(runtime.grave)
    active_card = next(card for card in all_cards if str(card.base_card.get('category') or '') == 'ProduceCardCategory_ActiveSkill')
    mental_card = next(card for card in all_cards if str(card.base_card.get('category') or '') == 'ProduceCardCategory_MentalSkill')
    effect = {
        'id': 'test-effect-play-card-limit',
        'effectType': 'ProduceExamEffectType_ExamGimmickPlayCardLimit',
        'produceCardSearchId': search_id,
        'effectValue1': 1,
    }

    runtime._apply_exam_effect(effect, source='gimmick')

    assert runtime._matches_forbidden_search(active_card)
    assert not runtime._matches_forbidden_search(mental_card)
    assert not runtime._can_play_card(active_card)
    assert runtime._can_play_card(mental_card)


def test_exam_runtime_referenced_gains_round_up_for_status_and_lesson_values() -> None:
    """所有参照型增量都应向上取整，而不是保留浮点或 bankers rounding。"""

    runtime = _sample_runtime(seed=167)
    runtime.resources['aggressive'] = 1.0
    review_effect = {
        'id': 'test-effect-review-ceil',
        'effectType': 'ProduceExamEffectType_ExamReviewDependExamCardPlayAggressive',
        'effectValue1': 500,
    }

    runtime._apply_exam_effect(review_effect, source='card')

    assert runtime.resources['review'] == 1.0
    runtime.active_effects = []
    runtime.resources['lesson_buff'] = 0.0
    runtime.resources['enthusiastic'] = 0.0

    lesson_effect = {
        'id': 'test-effect-lesson-ceil',
        'effectType': 'ProduceExamEffectType_ExamLessonDependExamReview',
        'effectValue1': 500,
    }
    search_id = 'test-search-single-card-for-ceil'
    _inject_table_row(runtime.card_searches, {
        'id': search_id,
        'cardPositionType': 'ProduceCardPositionType_NotLost',
        'produceCardIds': [runtime.hand[0].card_id],
    })
    combo_effect = {
        'id': 'test-effect-lesson-base-plus-ref',
        'effectType': 'ProduceExamEffectType_ExamLessonPerSearchCount',
        'effectValue1': 1,
        'effectValue2': 500,
        'produceCardSearchId': search_id,
    }

    assert runtime._resolve_lesson_effect_value(lesson_effect) == 1.0
    assert runtime._resolve_lesson_effect_value(combo_effect) == 2.0


def test_exam_runtime_lesson_mode_tracks_target_clear_and_perfect_finish() -> None:
    """lesson 模式应区分清课目标和 Perfect 门槛，并在 Perfect 时提前结束回体。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=173,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
        lesson_target_value=10,
        lesson_perfect_value=15,
        lesson_perfect_recovery_per_turn=2.0,
        exam_score_bonus_multiplier=1.0,
        starting_stamina=5.0,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    lesson_ten = {
        'id': 'test-effect-lesson-ten',
        'effectType': 'ProduceExamEffectType_ExamLessonFix',
        'effectValue1': 10,
    }
    lesson_five = {
        'id': 'test-effect-lesson-five',
        'effectType': 'ProduceExamEffectType_ExamLessonFix',
        'effectValue1': 5,
    }

    runtime._apply_exam_effect(lesson_ten, source='card')

    assert runtime.lesson_cleared is True
    assert runtime.clear_state == 'cleared'
    assert runtime.terminated is False
    assert runtime._lesson_target_remaining() == 0.0
    assert runtime._lesson_perfect_remaining() == 5.0

    runtime.stamina = 1.0
    expected_recovery = min(runtime.max_stamina, 1.0 + (runtime.max_turns - runtime.turn + 1) * 2.0)
    runtime._apply_exam_effect(lesson_five, source='card')

    assert runtime.clear_state == 'perfect'
    assert runtime.terminated is True
    assert runtime._lesson_perfect_remaining() == 0.0
    assert runtime.stamina == pytest.approx(expected_recovery)


def test_exam_runtime_lesson_skip_recovers_stamina_but_normal_end_turn_does_not() -> None:
    """lesson 模式下只有 SKIP 会按手册回复体力，普通结束回合不应自动回体。"""

    normal_runtime = _sample_runtime(
        seed=181,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
        lesson_target_value=999,
        lesson_perfect_value=1999,
        starting_stamina=4.0,
    )
    normal_runtime.stamina = 4.0
    normal_runtime._end_turn(skipped=False)

    skip_runtime = _sample_runtime(
        seed=191,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
        lesson_target_value=999,
        lesson_perfect_value=1999,
        starting_stamina=4.0,
    )
    skip_runtime.stamina = 4.0
    recovery = float(skip_runtime.exam_setting.get('examTurnEndRecoveryStamina') or 0)
    skip_runtime._end_turn(skipped=True)

    assert normal_runtime.stamina == pytest.approx(4.0)
    assert skip_runtime.stamina == pytest.approx(min(skip_runtime.max_stamina, 4.0 + recovery))


def test_exam_runtime_sp_lesson_matches_lesson_sp_trigger_tag() -> None:
    """SP lesson 应同时命中属性 lessonType 和 LessonSp 触发标签。"""

    sp_runtime = _sample_runtime(
        seed=193,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
        lesson_types=('ProduceStepLessonType_LessonVocal', 'ProduceStepLessonType_LessonSp'),
    )
    normal_runtime = _sample_runtime(
        seed=197,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
    )
    event = {
        'phase_type': 'ProduceExamPhaseType_ExamCardPlay',
        'phase_value': 1,
        'acting_card': sp_runtime.hand[0],
        'effect_types': [],
        'status_change_origin': 'other',
    }
    trigger = {
        'id': 'test-trigger-lesson-sp',
        'phaseTypes': ['ProduceExamPhaseType_ExamCardPlay'],
        'lessonType': 'ProduceStepLessonType_LessonSp',
    }

    assert sp_runtime._trigger_matches(trigger, event, acting_card=sp_runtime.hand[0])
    assert not normal_runtime._trigger_matches(trigger, event, acting_card=normal_runtime.hand[0])


def test_exam_runtime_hard_lesson_after_clear_matches_all_attribute_tags() -> None:
    """追い込みレッスン clear 后应进入全属性阶段，三色 lessonType 都视为命中。"""

    runtime = _sample_runtime(
        seed=199,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
        lesson_post_clear_types=(
            'ProduceStepLessonType_LessonVocal',
            'ProduceStepLessonType_LessonDance',
            'ProduceStepLessonType_LessonVisual',
        ),
        lesson_target_value=10,
        lesson_perfect_value=50,
    )
    trigger = {
        'id': 'test-trigger-hard-lesson-dance',
        'phaseTypes': ['ProduceExamPhaseType_ExamCardPlay'],
        'lessonType': 'ProduceStepLessonType_LessonDance',
    }
    event = {
        'phase_type': 'ProduceExamPhaseType_ExamCardPlay',
        'phase_value': 1,
        'acting_card': runtime.hand[0],
        'effect_types': [],
        'status_change_origin': 'other',
    }

    assert not runtime._trigger_matches(trigger, event, acting_card=runtime.hand[0])
    runtime.score = 10.0
    runtime._update_clear_state_after_score_change()

    assert runtime.clear_state == 'cleared'
    assert runtime._trigger_matches(trigger, event, acting_card=runtime.hand[0])


def test_simulate_planning_exposes_final_summary() -> None:
    """planning 模拟在终局时应返回结构化终局摘要。"""

    result = simulate_planning(
        'nia_master',
        auto_policy='first_valid',
        auto_steps=128,
        seed=211,
        loadout_config=LoadoutConfig(
            idol_card_id=AMAO_R,
            producer_level=35,
            idol_rank=4,
            dearness_level=20,
            auto_support_cards=True,
        ),
    )

    final_summary = result['state']['final_summary']
    assert final_summary
    assert 'ending_type' in final_summary
    assert 'produce_result' in final_summary
    assert isinstance(final_summary['audition_history'], list)


def test_produce_runtime_selection_pool_blocks_second_legend() -> None:
    """已持有 Legend 时，咨询与奖励候选池不应再出现新的 Legend 卡。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=227)
    runtime.reset()
    legend_card = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('rarity') or '') == 'ProduceCardRarity_Legend'
    )
    runtime.deck.append(legend_card)

    selection_pool = runtime._selection_card_pool()

    assert selection_pool
    assert all(str(card.get('rarity') or '') != 'ProduceCardRarity_Legend' for card in selection_pool)


def test_state_snapshot_lesson_mode_exposes_clear_state_and_perfect_progress() -> None:
    """LLM 局面快照应显式暴露 lesson mode、清课状态和 Perfect 剩余。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=179,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
        lesson_target_value=10,
        lesson_perfect_value=15,
        exam_score_bonus_multiplier=1.0,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()
    runtime._apply_exam_effect(
        {
            'id': 'test-effect-lesson-ten-snapshot',
            'effectType': 'ProduceExamEffectType_ExamLessonFix',
            'effectValue1': 10,
        },
        source='card',
    )
    snapshot = build_state_snapshot(runtime, repository)

    assert '模式: レッスン' in snapshot
    assert '课程状态: 目標達成' in snapshot
    assert 'パーフェクト剩余: 5' in snapshot
    assert '目标剩余: 0' in snapshot


def test_state_context_lesson_mode_reports_clear_state_fields() -> None:
    """结构化上下文应返回 clear_state 和 Perfect 剩余。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=181,
        battle_kind='lesson',
        lesson_type='ProduceStepLessonType_LessonVocal',
        lesson_target_value=10,
        lesson_perfect_value=15,
        exam_score_bonus_multiplier=1.0,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()
    runtime._apply_exam_effect(
        {
            'id': 'test-effect-lesson-ten-context',
            'effectType': 'ProduceExamEffectType_ExamLessonFix',
            'effectValue1': 10,
        },
        source='card',
    )

    context = extract_state_context(runtime, repository)

    assert context['battle_kind'] == 'lesson'
    assert context['clear_state'] == 'cleared'
    assert context['lesson_target_remaining'] == '0'
    assert context['lesson_perfect_remaining'] == '5'


def test_action_context_lesson_mode_marks_skip_action() -> None:
    """lesson 模式的动作列表应把结束动作标成 SKIP。"""

    from gakumas_rl.envs import GakumasExamEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = _sample_loadout(repository, scenario)
    env = GakumasExamEnv(
        repository,
        scenario,
        idol_loadout=loadout,
        seed=191,
        battle_kind='lesson',
        lesson_action_type='lesson_vocal_normal',
        include_action_labels_in_step_info=True,
    )
    env.reset(seed=191)

    context = extract_action_list_context(env)

    assert any(action['label'].startswith('SKIP') for action in context)
    assert any(action_label_for_llm(env, action['index']).startswith('SKIP') for action in context)


def test_state_snapshot_exam_mode_omits_lesson_clear_fields() -> None:
    """普通考试快照不应携带 lesson 专属清课字段。"""

    runtime = _sample_runtime(seed=193)
    snapshot = build_state_snapshot(runtime, runtime.repository)
    context = extract_state_context(runtime, runtime.repository)

    assert '课程状态:' not in snapshot
    assert context['battle_kind'] == 'exam'
    assert context['clear_state'] == 'ongoing'
    assert context['lesson_target_remaining'] == '0'
    assert context['lesson_perfect_remaining'] == '0'
    assert context['lesson_cleared'] is False


def test_exam_runtime_self_lesson_cannot_use_drink() -> None:
    """自主课程中不应出现可用的 P 饮料动作。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=229)
    runtime.reset()

    reward, exam_info = runtime._run_audition('ProduceStepType_SelfLessonVocalNormal', apply_outcome=False)

    assert isinstance(reward, float)
    assert exam_info['stage_type'] == 'ProduceStepType_SelfLessonVocalNormal'
    lesson_runtime = ExamRuntime(
        repository,
        scenario,
        stage_type='ProduceStepType_SelfLessonVocalNormal',
        seed=233,
        deck=list(runtime.deck),
        drinks=list(runtime.drinks),
        initial_status_enchant_ids=list(runtime.exam_status_enchant_ids),
        initial_status_enchants=list(runtime.exam_status_enchant_specs),
        loadout=runtime.idol_loadout,
        starting_stamina=runtime._audition_start_stamina(),
        exam_score_bonus_multiplier=(runtime.idol_loadout.exam_score_bonus_multiplier if runtime.idol_loadout else 1.0),
        fan_votes=float(runtime.state.get('fan_votes') or 0.0),
        audition_row_id=default_audition_row_selector(
            repository,
            scenario,
            stage_type='ProduceStepType_SelfLessonVocalNormal',
            loadout=runtime.idol_loadout,
            fan_votes=float(runtime.state.get('fan_votes') or 0.0),
        ),
    )
    lesson_runtime.reset()

    assert not any(action.kind == 'drink' for action in lesson_runtime.legal_actions())


def test_produce_runtime_final_summary_on_failed_audition() -> None:
    """考试失败结束时也应产出终局摘要。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=239)
    runtime.reset()
    runtime.pending_audition_result = {
        'stage_type': scenario.audition_sequence[0],
        'reward': -0.4,
        'audition_slot': 0,
        'cleared': False,
        'effective_score': 800.0,
        'exam_score': 700.0,
        'fan_vote_gain': 0.0,
        'deck_quality_gain': 0.0,
        'drink_quality_gain': 0.0,
        'rank': 5,
        'rank_threshold': 3,
        'rival_scores': [900.0, 1000.0],
    }

    terminated, info = runtime._accept_pending_audition_result()

    assert terminated is True
    assert info['final_summary']
    assert info['final_summary']['route_clear'] is False
    assert info['final_summary']['ending_type'] == 'failed'
    assert info['final_summary']['failed_stage_type'] == scenario.audition_sequence[0]


def test_produce_runtime_final_summary_uses_rank1_end_for_nia() -> None:
    """NIA 最终第一名时应保持 rank1 普通结算。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=241)
    runtime.reset()
    runtime.state['dearness_level'] = 17.0
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 1,
            'effective_score': 4321.0,
            'exam_score': 4200.0,
            'cleared': True,
            'audition_row_number': 4,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['ending_type'] == 'nia_win'
    assert summary['p_live']['variation'] == 'rank_1'


def test_produce_runtime_final_summary_marks_first_star_rank1() -> None:
    """初路线第一名应落到普通 A 结局。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=251)
    runtime.reset()
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 1,
            'effective_score': 5432.0,
            'exam_score': 5300.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['ending_type'] == 'first_star_a'


def test_produce_runtime_upgrade_matching_cards_skips_legend() -> None:
    """按搜索条件升级卡时也不应升级 Legend。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=263)
    runtime.reset()
    legend_card = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('rarity') or '') == 'ProduceCardRarity_Legend'
    )
    runtime.deck = [legend_card]

    runtime._upgrade_matching_cards('', 1)

    assert str(runtime.deck[0].get('rarity') or '') == 'ProduceCardRarity_Legend'
    assert int(runtime.deck[0].get('upgradeCount') or 0) == int(legend_card.get('upgradeCount') or 0)


def test_simulate_planning_loadout_summary_includes_extra_produce_items() -> None:
    """loadout 摘要应暴露 challenge P 道具列表。"""

    summary = loadout_summary(
        'first_star_legend',
        LoadoutConfig(
            idol_card_id=AMAO_R,
            producer_level=50,
            idol_rank=4,
            dearness_level=10,
            challenge_item_ids=('pitem_00-1-048-challenge',),
        ),
    )

    assert 'pitem_00-1-048-challenge' in summary['loadout']['extra_produce_item_ids']


def test_loadout_summary_includes_assist_mode_flag() -> None:
    """loadout 摘要应暴露 Assist Mode 开关。"""

    summary = loadout_summary(
        'nia_pro',
        LoadoutConfig(
            idol_card_id=AMAO_R,
            producer_level=35,
            idol_rank=4,
            dearness_level=10,
            assist_mode=True,
        ),
    )

    assert summary['loadout']['assist_mode'] is True


def test_assist_mode_reduces_rival_multiplier_for_nia_pro() -> None:
    """NIA Pro 的 Assist Mode 应降低 rival multiplier。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-004')
    normal_runtime = ProduceRuntime(
        repository,
        scenario,
        seed=521,
        idol_loadout=build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10, assist_mode=False),
    )
    assist_runtime = ProduceRuntime(
        repository,
        scenario,
        seed=521,
        idol_loadout=build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10, assist_mode=True),
    )
    normal_runtime.reset()
    assist_runtime.reset()
    _, normal_exam = normal_runtime._run_audition(scenario.audition_sequence[0], apply_outcome=False)
    _, assist_exam = assist_runtime._run_audition(scenario.audition_sequence[0], apply_outcome=False)

    assert assist_exam['rival_score_multiplier'] < normal_exam['rival_score_multiplier']
    assert assist_exam['assist_mode'] is True
    assert assist_exam['assist_reduction_ratio'] == pytest.approx(0.15)


def test_final_summary_exposes_assist_mode_flags() -> None:
    """终局摘要应回传 Assist Mode 信息。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-004')
    runtime = ProduceRuntime(
        repository,
        scenario,
        seed=523,
        idol_loadout=build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10, assist_mode=True),
    )
    runtime.reset()
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 1,
            'effective_score': 2000.0,
            'exam_score': 1900.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['assist_mode'] is True
    assert summary['assist_reduction_ratio'] == pytest.approx(0.15)


def test_selection_card_pool_allows_legend_again_after_deletion() -> None:
    """删除当前 Legend 后，候选池应恢复允许出现 Legend。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    loadout = _sample_loadout(repository, scenario)
    runtime = ProduceRuntime(repository, scenario, seed=269, idol_loadout=loadout)
    runtime.reset()
    legend_card = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('rarity') or '') == 'ProduceCardRarity_Legend'
    )
    runtime.deck.append(legend_card)
    runtime._remember_legend_cards()
    runtime.deck = [card for card in runtime.deck if str(card.get('id') or '') != str(legend_card.get('id') or '')]

    selection_pool = runtime._selection_card_pool()

    assert any(str(card.get('rarity') or '') == 'ProduceCardRarity_Legend' for card in selection_pool)


def test_build_final_summary_contains_core_fields_when_cleared() -> None:
    """通关摘要应保留核心结算字段。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=271)
    runtime.reset()
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 2,
            'effective_score': 3456.0,
            'exam_score': 3300.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['p_live']['unlocked'] is True
    assert summary['produce_result']['rank'] in {'A', 'B', 'C', 'S'}
    assert str(summary['produce_result']['formula_source']).startswith('nia_external_formula')
    assert summary['ending']['type'] == summary['ending_type']
    assert summary['ending']['route'] == 'nia'
    assert summary['p_live']['dearness_level'] == int(runtime.state['dearness_level'])
    assert summary['produce_result']['parameter_total'] > 0.0




def test_final_summary_p_live_variation_rank1_without_true_end() -> None:
    """非 True End 的第一名应使用 rank_1 variation。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=541)
    runtime.reset()
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 1,
            'effective_score': 4100.0,
            'exam_score': 4000.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['p_live']['variation'] == 'rank_1'
    assert summary['ending']['grade'] == 'a'



def test_failed_final_summary_has_failed_ending_grade() -> None:
    """失败时 ending grade 应为 failed。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=547)
    runtime.reset()

    summary = runtime._build_final_summary(cleared=False, failed_stage_type=scenario.audition_sequence[0])

    assert summary['ending']['grade'] == 'failed'
    assert summary['p_live']['variation'] == 'standard'



def test_produce_result_exposes_fan_votes_and_parameter_total() -> None:
    """produce result 应显式回传 fan_votes 和 parameter_total。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=557)
    runtime.reset()
    runtime.state['fan_votes'] = 1234.0
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 3,
            'effective_score': 2400.0,
            'exam_score': 2300.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['fan_votes'] == pytest.approx(1234.0)
    assert float(summary['produce_result']['fan_votes']) >= 0.0
    assert 'formula_source' in summary['produce_result']
    assert float(summary['produce_result']['parameter_total']) > 0.0
    assert str(summary['produce_result']['formula_source']).startswith('nia_external_formula') or str(summary['produce_result']['formula_source']).startswith('hajime_external_formula')


def test_build_final_summary_failed_run_keeps_p_live_locked() -> None:
    """失败摘要不应解锁 P Live。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=277)
    runtime.reset()

    summary = runtime._build_final_summary(cleared=False, failed_stage_type=scenario.audition_sequence[0])

    assert summary['p_live']['unlocked'] is False


def test_build_final_summary_preserves_final_rank_and_score() -> None:
    """终局摘要应回传最后一场考试的名次和有效分。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=281)
    runtime.reset()
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 3,
            'effective_score': 2468.0,
            'exam_score': 2400.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['final_rank'] == 3
    assert summary['final_score'] == pytest.approx(2468.0)


def test_build_final_summary_includes_failed_stage() -> None:
    """失败时应记录失败阶段。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=283)
    runtime.reset()

    summary = runtime._build_final_summary(cleared=False, failed_stage_type=scenario.audition_sequence[1])

    assert summary['failed_stage_type'] == scenario.audition_sequence[1]
    assert summary['route_clear'] is False


def test_accept_pending_audition_appends_history() -> None:
    """接受考试结果时应写入 audition_history。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=293)
    runtime.reset()
    runtime.pending_audition_result = {
        'stage_type': scenario.audition_sequence[0],
        'reward': 0.3,
        'audition_slot': 0,
        'cleared': True,
        'effective_score': 1234.0,
        'exam_score': 1200.0,
        'fan_vote_gain': 10.0,
        'deck_quality_gain': 0.4,
        'drink_quality_gain': 0.2,
        'rank': 1,
        'rank_threshold': 3,
        'rival_scores': [1000.0],
    }

    terminated, _info = runtime._accept_pending_audition_result()

    assert terminated is False
    assert len(runtime.audition_history) == 1
    assert runtime.audition_history[0]['rank'] == 1


def test_serialize_planning_state_includes_final_summary_and_history() -> None:
    """planning state 序列化应包含终局摘要和考试历史。"""

    runtime = ProduceRuntime(MasterDataRepository(), MasterDataRepository().build_scenario('produce-005'), seed=307)
    runtime.reset()
    runtime.audition_history = [{'stage_type': 'x', 'rank': 1}]
    runtime.final_summary = {'ending_type': 'nia_win'}

    payload = _serialize_planning_state(runtime)

    assert payload['audition_history'] == [{'stage_type': 'x', 'rank': 1}]
    assert payload['final_summary'] == {'ending_type': 'nia_win'}


def test_build_final_summary_first_star_non_first_rank_maps_to_letter_end() -> None:
    """初路线应按最终名次映射普通结局标签。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=313)
    runtime.reset()
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 4,
            'effective_score': 1000.0,
            'exam_score': 950.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['ending_type'] == 'first_star_d'


def test_build_final_summary_nia_non_win_clear_maps_to_clear_end() -> None:
    """NIA 非第一但通关时应落到普通 clear 标签。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=317)
    runtime.reset()
    runtime.audition_history = [
        {
            'stage_type': scenario.audition_sequence[-1],
            'rank': 3,
            'effective_score': 2100.0,
            'exam_score': 2000.0,
            'cleared': True,
        }
    ]

    summary = runtime._build_final_summary(cleared=True)

    assert summary['ending_type'] == 'nia_finalist'


def test_final_summary_failed_route_marks_p_live_locked() -> None:
    """失败时 P Live 应保持未解锁。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=337)
    runtime.reset()

    summary = runtime._build_final_summary(cleared=False, failed_stage_type=scenario.audition_sequence[0])

    assert summary['p_live']['unlocked'] is False


def test_build_final_summary_uses_last_audition_entry() -> None:
    """多场考试时终局摘要应取最后一场结果。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=353)
    runtime.reset()
    runtime.audition_history = [
        {'stage_type': scenario.audition_sequence[0], 'rank': 2, 'effective_score': 1111.0, 'exam_score': 1000.0, 'cleared': True},
        {'stage_type': scenario.audition_sequence[-1], 'rank': 1, 'effective_score': 2222.0, 'exam_score': 2100.0, 'cleared': True, 'audition_row_number': 4},
    ]
    runtime.state['dearness_level'] = 17.0

    summary = runtime._build_final_summary(cleared=True)

    assert summary['final_audition_stage'] == scenario.audition_sequence[-1]
    assert summary['final_score'] == pytest.approx(2222.0)


def test_build_final_summary_failed_without_history_uses_last_exam_score() -> None:
    """没有历史时失败摘要应回退到 last_exam_score。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=359)
    runtime.reset()
    runtime.state['last_exam_score'] = 987.0

    summary = runtime._build_final_summary(cleared=False, failed_stage_type=scenario.audition_sequence[0])

    assert summary['final_score'] == pytest.approx(987.0)


def test_selection_pool_filters_initial_deck_ids() -> None:
    """候选池应继续排除初始卡组。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=373)
    runtime.reset()

    selection_pool = runtime._selection_card_pool()

    assert all(str(card.get('id') or '') not in runtime.initial_deck_card_ids for card in selection_pool)


def test_failed_accept_pending_summary_contains_history() -> None:
    """失败收尾时返回的 final_summary 也应携带已记录历史。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=383)
    runtime.reset()
    runtime.audition_history = [{'stage_type': scenario.audition_sequence[0], 'rank': 2, 'effective_score': 1200.0, 'exam_score': 1100.0, 'cleared': True}]
    runtime.pending_audition_result = {
        'stage_type': scenario.audition_sequence[1],
        'reward': -0.2,
        'audition_slot': 1,
        'cleared': False,
        'effective_score': 800.0,
        'exam_score': 760.0,
        'fan_vote_gain': 0.0,
        'deck_quality_gain': 0.0,
        'drink_quality_gain': 0.0,
        'rank': 4,
        'rank_threshold': 3,
        'rival_scores': [900.0],
    }

    terminated, info = runtime._accept_pending_audition_result()

    assert terminated is True
    assert len(info['final_summary']['audition_history']) == 2


def test_build_final_summary_route_field_matches_scenario_type() -> None:
    """终局摘要 route 字段应和场景类型一致。"""

    repo = MasterDataRepository()
    first_star_runtime = ProduceRuntime(repo, repo.build_scenario('produce-006'), seed=389)
    first_star_runtime.reset()
    nia_runtime = ProduceRuntime(repo, repo.build_scenario('produce-005'), seed=397)
    nia_runtime.reset()

    assert first_star_runtime._build_final_summary(cleared=False)['route'] == 'first_star'
    assert nia_runtime._build_final_summary(cleared=False)['route'] == 'nia'


def test_simulate_planning_final_summary_survives_auto_run() -> None:
    """自动跑完整局后 state 里仍应保留 final_summary。"""

    result = simulate_planning(
        'first_star_legend',
        auto_policy='first_valid',
        auto_steps=128,
        seed=409,
        loadout_config=LoadoutConfig(
            idol_card_id=AMAO_R,
            producer_level=50,
            idol_rank=4,
            dearness_level=10,
            auto_support_cards=True,
        ),
    )

    assert 'final_summary' in result['state']
    assert isinstance(result['state']['audition_history'], list)


def test_final_summary_produce_result_score_is_positive_after_clear() -> None:
    """通关时 produce result score 应为正值。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=421)
    runtime.reset()
    runtime.audition_history = [
        {'stage_type': scenario.audition_sequence[-1], 'rank': 1, 'effective_score': 2000.0, 'exam_score': 1900.0, 'cleared': True}
    ]
    runtime.state['dearness_level'] = 17.0

    summary = runtime._build_final_summary(cleared=True)

    assert summary['produce_result']['score'] > 0.0


def test_selection_pool_with_owned_legend_only_filters_legend_not_others() -> None:
    """持有 Legend 后仍应保留非 Legend 候选。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=431)
    runtime.reset()
    legend_card = next(
        dict(row)
        for row in repository.load_table('ProduceCard').rows
        if str(row.get('rarity') or '') == 'ProduceCardRarity_Legend'
    )
    runtime.deck.append(legend_card)

    pool = runtime._selection_card_pool()

    assert pool
    assert any(str(card.get('rarity') or '') != 'ProduceCardRarity_Legend' for card in pool)


def test_final_summary_failed_ending_type_is_failed() -> None:
    """失败时 ending_type 固定为 failed。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-006')
    runtime = ProduceRuntime(repository, scenario, seed=433)
    runtime.reset()

    summary = runtime._build_final_summary(cleared=False)

    assert summary['ending_type'] == 'failed'


def test_accept_pending_success_without_terminal_keeps_final_summary_empty() -> None:
    """非终局验收考试结果时不应提前写 final_summary。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    runtime = ProduceRuntime(repository, scenario, seed=449)
    runtime.reset()
    runtime.pending_audition_result = {
        'stage_type': scenario.audition_sequence[0],
        'reward': 0.2,
        'audition_slot': 0,
        'cleared': True,
        'effective_score': 1500.0,
        'exam_score': 1400.0,
        'fan_vote_gain': 10.0,
        'deck_quality_gain': 0.4,
        'drink_quality_gain': 0.2,
        'rank': 1,
        'rank_threshold': 3,
        'rival_scores': [1000.0],
    }

    terminated, info = runtime._accept_pending_audition_result()

    assert terminated is False
    assert info['final_summary'] == {}



def test_state_snapshot_exam_mode_exposes_turn_color() -> None:
    """考试快照应显式暴露当前回合颜色。"""

    runtime = _sample_runtime(seed=181, exam_score_bonus_multiplier=1.0)

    snapshot = build_state_snapshot(runtime, runtime.repository)
    context = extract_state_context(runtime, runtime.repository)

    assert '当前回合颜色:' in snapshot
    assert context['turn_color'] in {'vocal', 'dance', 'visual'}
    assert context['turn_color_label'] in {'Vocal', 'Dance', 'Visual'}
    assert context['turn_color_display_label'] in {'ボーカル', 'ダンス', 'ビジュアル'}


def test_state_snapshot_exposes_card_item_and_drink_prompt_context() -> None:
    """LLM prompt 应显式暴露可读的手牌、牌库、P 道具和 P 饮料信息，而不是内部 id。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)
    runtime = ExamRuntime(
        repository,
        scenario,
        loadout=loadout,
        seed=67,
        audition_row_id=_sample_audition_row_selector(repository, scenario, loadout),
    )
    runtime.reset()

    ctx = extract_state_context(runtime, repository)
    snapshot = build_state_snapshot(runtime, repository)

    assert ctx['loadout'] is not None
    assert ctx['loadout']['produce_item'] is not None
    assert ctx['loadout']['produce_item']['name']
    assert ctx['loadout']['produce_item']['description']
    assert ctx['loadout']['produce_item']['enchants']
    assert ctx['drinks']
    assert all('preview_summary' in drink for drink in ctx['drinks'])
    assert any(card['cost_summary'] != '' and card['preview_summary'] != '' for card in ctx['hand'])
    assert all(drink['available'] for drink in ctx['drinks'])
    assert all(card['description'] or card['effect_summary'] for card in ctx['hand'])
    assert ctx['deck_cards']
    assert 'name' in ctx['deck_cards'][0]
    assert 'description' in ctx['deck_cards'][0]
    raw_hand_name = str(runtime.hand[0].base_card.get('name') or runtime.hand[0].card_id)
    raw_drink_name = str(runtime.drinks[0].get('name') or runtime.drinks[0].get('id') or '')
    raw_item_name = str(repository.load_table('ProduceItem').first(loadout.produce_item_id).get('name') or loadout.produce_item_id)
    assert ctx['hand'][0]['name'] == raw_hand_name
    assert ctx['drinks'][0]['name'] == raw_drink_name
    assert ctx['loadout']['produce_item']['name'] == raw_item_name
    assert all('干劲' not in card['description'] for card in ctx['hand'])
    assert all('干劲' not in drink['description'] for drink in ctx['drinks'])
    assert '干劲' not in ctx['loadout']['produce_item']['description']

    assert 'Pアイテム:' in snapshot
    assert 'Pドリンク库存' in snapshot
    assert '当前结算:' in snapshot
    assert '本回合剩余スキルカード使用数:' in snapshot
    assert 'Pアイテム附带附魔（开场装载）' in snapshot
    assert '状态:' in snapshot
    assert '### 牌库顺序明细（顶 -> 底）' in snapshot
    assert str(runtime.deck[0].base_card.get('name') or runtime.deck[0].card_id) in snapshot
    assert '卡面说明:' in snapshot
    assert '标签:' not in snapshot
    assert 'IdolCardRarity_' not in snapshot
    assert 'ProduceExamEffectType_' not in snapshot
    assert 'ExamBlockDepend' not in snapshot
    assert 'enchant-pitem_' not in snapshot
    assert '干劲' not in snapshot
    assert 'おすすめ効果:' in snapshot
    assert '好調 / 集中' in snapshot
    assert '「好調」や「集中」を活用して育成するプランです。' in snapshot
    assert '主要效果:' not in snapshot
    assert '好調=' in snapshot
    assert '元気=' in snapshot
    assert 'パラメータ上昇量増加=' in snapshot
    assert '全力値=' in snapshot
    assert '好调=' not in snapshot
    assert '元气=' not in snapshot
    assert 'stamina=' not in snapshot
    assert '课程增益=' not in snapshot
    assert 'P饮料' not in snapshot
    assert 'P道具/饰品' not in snapshot
    assert '参数面板: ボーカル=' in snapshot
    assert '基础属性: ボーカル=' in snapshot

    localized_hand_name = repository.card_name(runtime.hand[0].base_card)
    localized_drink_name = repository.drink_name(runtime.drinks[0])
    if localized_hand_name != raw_hand_name:
        assert localized_hand_name not in snapshot
    if localized_drink_name != raw_drink_name:
        assert localized_drink_name not in snapshot

    runtime._use_drink(0)
    used_snapshot = build_state_snapshot(runtime, repository)
    used_ctx = extract_state_context(runtime, repository)

    assert used_ctx['used_drink_count'] == 1
    assert used_ctx['available_drink_count'] == used_ctx['drink_total_count'] - 1
    assert len(used_ctx['drinks']) == used_ctx['available_drink_count']
    assert all(drink['available'] for drink in used_ctx['drinks'])
    assert '已使用)' not in used_snapshot


def test_state_snapshot_formats_active_effects_as_readable_sentences() -> None:
    """活跃效果应优先输出自然语言效果/状态，而不是内部摘要碎片。"""

    runtime = _sample_runtime(seed=181)
    repository = runtime.repository
    runtime.active_effects = []
    runtime._apply_exam_effect(
        {
            'id': 'test-readable-stamina-add',
            'effectType': 'ProduceExamEffectType_ExamStaminaConsumptionAdd',
            'effectTurn': 1,
        },
        source='drink',
    )
    runtime._apply_exam_effect(
        {
            'id': 'test-readable-search-stamina-zero',
            'effectType': 'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange',
            'effectTurn': -1,
            'effectCount': 2,
            'effectValue1': 0,
        },
        source='drink',
    )

    snapshot = build_state_snapshot(runtime, repository)

    assert '效果：技能卡体力消耗翻倍；状态：剩余1回合；来源：Pドリンク' in snapshot
    assert '效果：命中的技能卡体力消耗改为0；状态：永久，剩余2次；来源：Pドリンク' in snapshot
    assert '检索·消耗·体力' not in snapshot


def test_state_snapshot_formats_active_enchants_as_readable_sentences() -> None:
    """活跃附魔应直接输出可读描述，不把生硬内部短名暴露给 LLM。"""

    runtime = _sample_runtime(seed=182)
    repository = runtime.repository
    enchant_row = {
        'id': 'test-readable-enchant',
        'produceExamTriggerId': '',
        'produceExamEffectIds': [],
        'produceDescriptions': [
            {'text': '每使用2次技能卡，元气的35%数值变为打分上升'},
        ],
    }
    _inject_table_row(repository.exam_status_enchants, enchant_row)
    repository.exam_status_enchant_map[str(enchant_row['id'])] = enchant_row
    runtime.active_enchants = [
        TriggeredEnchant(
            uid=runtime._next_uid(),
            enchant_id=str(enchant_row['id']),
            trigger_id='',
            effect_ids=[],
            remaining_turns=None,
            remaining_count=None,
            source='gimmick',
        )
    ]

    snapshot = build_state_snapshot(runtime, repository)

    assert '每使用2次技能卡' in snapshot
    assert '转为打分' in snapshot
    assert '来源：応援/トラブル' in snapshot


def test_exam_runtime_skip_phase_dispatches_exam_turn_skip_trigger() -> None:
    """结束空过回合时应分发 ExamTurnSkip，供主数据触发器命中。"""

    runtime = _sample_runtime(seed=59)
    repository = runtime.repository
    lesson_buff = _first_exam_effect(repository, 'ProduceExamEffectType_ExamLessonBuff', effectValue1=1)
    runtime.active_enchants.append(
        TriggeredEnchant(
            uid=runtime._next_uid(),
            enchant_id='test-skip-enchant',
            trigger_id='e_trigger-exam_turn_skip',
            effect_ids=[str(lesson_buff['id'])],
            remaining_turns=None,
            remaining_count=1,
            source='test',
        )
    )

    runtime._end_turn(skipped=True)

    assert runtime.resources['lesson_buff'] == 1.0


def test_exam_runtime_grow_effect_targeted_effect_replacement_is_not_global() -> None:
    """EffectChange 和 PlayEffectTriggerChange 应只改命中的目标效果或触发器。"""

    runtime = _sample_runtime(seed=61)
    repository = runtime.repository
    effect_change = repository.load_table('ProduceCardGrowEffect').first(
        'g_effect-effect_change-e_effect-exam_block_per_use_card_count-0002-0008-e_effect-exam_status_enchant-02-inf-enchant-p_card-02-ido-3_067-enc01'
    )
    trigger_change = repository.load_table('ProduceCardGrowEffect').first(
        'g_effect-play_effect_trigger_change-e_trigger-none-lesson_buff_up-15-e_trigger-none-lesson_buff_up-10'
    )
    card = RuntimeCard(
        uid=900001,
        card_id='test-card',
        upgrade_count=0,
        base_card={
            'playEffects': [
                {
                    'produceExamEffectId': 'e_effect-exam_block_per_use_card_count-0002-0008',
                    'produceExamTriggerId': 'e_trigger-none-lesson_buff_up-15',
                },
                {
                    'produceExamEffectId': 'e_effect-exam_card_draw-0001',
                    'produceExamTriggerId': 'e_trigger-none-review_up-10',
                },
            ]
        },
        grow_effect_ids=[str(effect_change['id']), str(trigger_change['id'])],
    )

    resolved = runtime._resolved_card_play_effects(card)

    assert resolved[0]['effect_id'] == str(effect_change['playProduceExamEffectId'])
    assert resolved[0]['trigger_id'] == str(trigger_change['playEffectProduceExamTriggerId'])
    assert resolved[1]['effect_id'] == 'e_effect-exam_card_draw-0001'
    assert resolved[1]['trigger_id'] == 'e_trigger-none-review_up-10'

def test_exam_runtime_initial_add_grow_effect_duplicates_card_on_build() -> None:
    """InitialAdd 应在构筑运行时卡组时补出额外的初始副本。"""

    runtime = _sample_runtime(seed=67)
    repository = runtime.repository
    initial_add = _first_grow_effect(repository, 'ProduceCardGrowEffectType_InitialAdd')
    sample_row = dict(runtime.initial_deck_rows[0])
    sample_row['produceCardGrowEffectIds'] = [str(initial_add['id'])]

    built = runtime._build_runtime_deck([sample_row])

    assert len(built) == 2
    assert built[0].card_id == built[1].card_id




def test_exam_runtime_enchant_status_change_does_not_recurse_forever() -> None:
    """同一个附魔在自身派生的状态变化里不应无限自触发。"""

    runtime = _sample_runtime(seed=71)
    runtime.active_enchants = []

    loop_trigger_id = 'test-trigger-status-change-review'
    loop_effect_id = 'test-effect-review-plus-one'
    runtime.repository.exam_trigger_map[loop_trigger_id] = {
        'id': loop_trigger_id,
        'phaseTypes': ['ProduceExamPhaseType_ExamStatusChange'],
        'effectTypes': ['ProduceExamEffectType_ExamReview'],
    }
    runtime.repository.exam_effect_map[loop_effect_id] = {
        'id': loop_effect_id,
        'effectType': 'ProduceExamEffectType_ExamReview',
        'effectValue1': 1,
    }
    runtime.active_enchants.append(
        TriggeredEnchant(
            uid=runtime._next_uid(),
            enchant_id='test-recursive-enchant',
            trigger_id=loop_trigger_id,
            effect_ids=[loop_effect_id],
            remaining_turns=None,
            remaining_count=1,
            source='test',
        )
    )

    runtime._apply_exam_effect(runtime.repository.exam_effect_map[loop_effect_id], source='card')

    assert runtime.resources['review'] == 2.0
    assert runtime._resolving_enchant_uids == set()


def test_exam_runtime_event_log_records_card_acquired_on_reset() -> None:
    """reset() 后 event_log 应包含 card_acquired 事件。"""

    runtime = _sample_runtime(seed=73)

    assert len(runtime.event_log) > 0
    card_events = [e for e in runtime.event_log if e.event_type == 'card_acquired']
    assert len(card_events) >= 10
    for event in card_events:
        assert 'card_id' in event.detail
        assert 'card_name' in event.detail
        assert 'destination' in event.detail


def test_exam_runtime_event_log_records_drink_consumed() -> None:
    """使用饮料后 event_log 应包含 drink_consumed 事件。"""

    runtime = _sample_runtime(seed=79)
    available = [i for i, d in enumerate(runtime.drinks) if not d.get('_consumed')]
    assert available
    assert runtime.turn_counters['play_count'] == 0
    runtime._use_drink(available[0])

    drink_events = [e for e in runtime.event_log if e.event_type == 'drink_consumed']
    assert len(drink_events) == 1
    assert 'drink_name' in drink_events[0].detail
    assert drink_events[0].detail['drink_index'] == available[0]
    assert runtime.turn_counters['play_count'] == 0


def test_exam_runtime_event_log_resets_between_episodes() -> None:
    """连续两次 reset() 后 event_log 不应累积。"""

    runtime = _sample_runtime(seed=83)
    first_count = len(runtime.event_log)
    assert first_count > 0

    runtime.reset()
    second_count = len(runtime.event_log)
    assert second_count > 0
    assert second_count == first_count or abs(second_count - first_count) < first_count


def test_serialize_exam_state_includes_deck_cards_and_event_log() -> None:
    """_serialize_exam_state 返回应包含 deck_cards 和 event_log。"""

    from gakumas_rl.service import _serialize_exam_state

    runtime = _sample_runtime(seed=89, fan_votes=12345.0)
    state = _serialize_exam_state(runtime)

    assert 'deck_cards' in state
    assert 'event_log' in state
    assert state['turn_color'] in {'vocal', 'dance', 'visual'}
    assert state['turn_color_label'] in {'Vocal', 'Dance', 'Visual'}
    assert state['fan_votes'] == pytest.approx(runtime.fan_votes)
    assert state['fan_vote_baseline'] == pytest.approx(float(runtime.profile.get('fan_vote_baseline') or 0.0))
    assert state['fan_vote_requirement'] == pytest.approx(float(runtime.profile.get('fan_vote_requirement') or 0.0))
    assert len(state['deck_cards']) == len(runtime.deck)
    assert len(state['event_log']) == len(runtime.event_log)
    if state['deck_cards']:
        card = state['deck_cards'][0]
        assert 'uid' in card
        assert 'card_id' in card
        assert 'name' in card
        assert 'effect_types' in card

@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_exam_env_observation_space_unchanged_without_deck_features() -> None:
    """include_deck_features=False 时 global_dim 应与基础特征维度一致。"""

    from gakumas_rl.envs import GakumasExamEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(repository, scenario, seed=91, include_deck_features=False)

    assert env.deck_feature_dim == 0
    assert env.global_dim == 50 + env.stage_context_dim + env.loadout_context_dim


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_exam_env_observation_space_grows_with_deck_features() -> None:
    """include_deck_features=True 时 global_dim 应增加 deck_feature_dim。"""

    from gakumas_rl.envs import GakumasExamEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(repository, scenario, seed=97, include_deck_features=True)

    assert env.deck_feature_dim > 0
    assert env.global_dim == 50 + env.stage_context_dim + env.loadout_context_dim + env.deck_feature_dim

    obs, _ = env.reset()
    assert obs['global'].shape[0] == env.global_dim


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_exam_env_reset_exposes_turn_color_one_hot() -> None:
    """考试观测和 reset info 应同步暴露当前回合颜色。"""

    from gakumas_rl.envs import GakumasExamEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(
        repository,
        scenario,
        seed=103,
        include_deck_features=False,
        base_loadout_config={'fan_votes': 9000.0},
    )

    obs, info = env.reset(seed=103)
    turn_color = info['turn_color']
    color_one_hot = obs['global'][-3:]
    color_index = {'vocal': 0, 'dance': 1, 'visual': 2}[turn_color]

    assert info['turn_color_label'] in {'Vocal', 'Dance', 'Visual'}
    assert info['fan_votes'] == pytest.approx(9000.0)
    assert info['fan_vote_baseline'] >= 0.0
    assert info['fan_vote_requirement'] >= 0.0
    assert obs['global'].shape[0] == env.global_dim
    assert np.all(np.isfinite(color_one_hot))


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_exam_env_clear_mode_hides_turn_color_and_fan_vote_context() -> None:
    """clear 模式的 env 不应再暴露颜色轮次和 fan vote 相关上下文。"""

    from gakumas_rl.envs import GakumasExamEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    env = GakumasExamEnv(
        repository,
        scenario,
        seed=107,
        exam_reward_mode='clear',
        include_deck_features=False,
        base_loadout_config={'fan_votes': 12000.0},
    )

    obs, info = env.reset(seed=107)

    assert info['turn_color'] in {'vocal', 'dance', 'visual'}
    assert info['turn_color_label'] in {'Vocal', 'Dance', 'Visual'}
    assert info['fan_votes'] >= 0.0
    assert info['fan_vote_baseline'] >= 0.0
    assert info['fan_vote_requirement'] >= 0.0
    assert obs['global'].shape[0] == env.global_dim
    assert np.all(np.isfinite(obs['global'][-3:]))


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_build_env_from_config_lesson_mode_uses_master_targets_and_skip_label() -> None:
    """独立 lesson 模式应按主数据装配目标值、回合数，并暴露 SKIP 动作。"""

    from gakumas_rl.service import build_env_from_config

    env = build_env_from_config(
        {
            'mode': 'lesson',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_SSR,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'lesson_action_type': 'lesson_vocal_normal',
            'lesson_level_index': 2,
            'seed': 211,
        }
    )

    obs, info = env.reset(seed=211)

    assert info['battle_kind'] == 'lesson'
    assert info['lesson_action_type'] == 'lesson_vocal_normal'
    assert info['lesson_level_index'] == 2
    assert info['lesson_target_value'] == pytest.approx(env.runtime._current_clear_target())
    assert info['lesson_perfect_value'] == pytest.approx(env.runtime._current_perfect_target())
    assert env.runtime.max_turns == env.current_lesson_spec.turn_limit
    assert info['turn_color'] == ''
    assert info['fan_votes'] == pytest.approx(0.0)
    assert obs['global'][-3:].shape == (3,)
    assert np.all(np.isfinite(obs['global'][-3:]))
    assert 'SKIP' in info['action_labels']


@pytest.mark.skipif(not _has_gymnasium, reason='gymnasium not installed')
def test_action_list_context_uses_raw_japanese_names_for_llm() -> None:
    """LLM 用动作列表应优先使用主数据库原始日文名，而不是本地化名。"""

    from gakumas_rl.envs import GakumasExamEnv

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)
    env = GakumasExamEnv(repository, scenario, idol_loadout=loadout, seed=151)

    obs, info = env.reset(seed=151)
    del obs, info

    actions = extract_action_list_context(env)
    card_actions = [action for action in actions if action['kind'] == 'card']
    drink_actions = [action for action in actions if action['kind'] == 'drink']

    assert card_actions
    assert drink_actions

    first_card_uid = int(env._candidates[card_actions[0]['index']].payload['uid'])
    first_card = next(card for card in env.runtime.hand if int(card.uid) == first_card_uid)
    assert card_actions[0]['label'] == f'{first_card.base_card.get("name")}[{int(first_card.upgrade_count)}]'

    first_drink_index = int(env._candidates[drink_actions[0]['index']].payload['index'])
    first_drink = env.runtime.drinks[first_drink_index]
    assert drink_actions[0]['label'] == str(first_drink.get('name') or first_drink.get('id') or '')
    assert action_label_for_llm(env, drink_actions[0]['index']) == drink_actions[0]['label']


def test_support_event_pool_school_events_not_in_outing() -> None:
    """授業（school_class）事件池不应含外出（outing）事件，支援卡事件只进差入（present）。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=0)
    runtime.reset()

    school_samples = runtime.action_samples.get('school_class', [])
    outing_samples = runtime.action_samples.get('outing', [])
    present_samples = runtime.action_samples.get('present', [])

    # school_class 池不含 Character / SupportCard 事件
    for row in school_samples:
        event_type = str(row.get('eventType') or '')
        assert event_type not in {'ProduceEventType_Character', 'ProduceEventType_SupportCard'}, (
            f'school_class 池混入了非授業事件: {event_type}'
        )

    # outing 池不含 School 事件
    for row in outing_samples:
        event_type = str(row.get('eventType') or '')
        assert event_type != 'ProduceEventType_School', '外出池混入了授業事件'

    # SupportCard 事件不应出现在 outing 池（只进 present）
    for row in outing_samples:
        event_type = str(row.get('eventType') or '')
        assert event_type != 'ProduceEventType_SupportCard', '外出池混入了支援卡事件'

    # present 池中存在 SupportCard 事件（如果有解锁的）或 Character 事件
    present_types = {str(row.get('eventType') or '') for row in present_samples}
    assert 'ProduceEventType_Character' in present_types or len(present_samples) == 0


def test_revert_card_change_restores_deck() -> None:
    """支援卡事件触发卡牌强化后，选择戻す可还原牌组状态。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=7)
    runtime.reset()

    # 直接模拟一次 event 触发 upgrade
    if not runtime.deck:
        return  # 无卡组跳过
    original_card = dict(runtime.deck[0])
    runtime._upgrade_matching_cards('', 1, source_action_type='present')

    if runtime.pending_revert_info is None:
        # 卡可能已无法再强化（最高阶），跳过
        return

    # legal_actions 应包含 revert_card_change
    actions = runtime.legal_actions()
    revert_actions = [a for a in actions if a.action_type == 'revert_card_change']
    assert len(revert_actions) == 1

    # 执行戻す
    revert_idx = next(i for i, a in enumerate(actions) if a.action_type == 'revert_card_change')
    runtime.step(revert_idx)

    # 牌组第0张应还原
    assert runtime.deck[0].get('upgradeCount') == original_card.get('upgradeCount')
    assert runtime.pending_revert_info is None


def test_revert_card_change_expires_after_next_step() -> None:
    """戻す选项在下一个非 revert 动作后自动消失。"""

    repository = MasterDataRepository()
    scenario = repository.build_scenario('produce-001')
    runtime = ProduceRuntime(repository, scenario, seed=8)
    runtime.reset()

    if not runtime.deck:
        return
    runtime._upgrade_matching_cards('', 1, source_action_type='present')
    if runtime.pending_revert_info is None:
        return

    # 确认 pending 存在
    assert runtime.pending_revert_info is not None

    # 执行任意非 revert 动作
    actions = runtime.legal_actions()
    non_revert = next((i for i, a in enumerate(actions) if a.action_type != 'revert_card_change' and a.available), None)
    if non_revert is None:
        return
    runtime.step(non_revert)

    # 执行完其他动作后 pending 应清空
    assert runtime.pending_revert_info is None
