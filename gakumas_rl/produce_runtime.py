"""培育阶段运行时，按主数据驱动课程、事件和阶段考试流程。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .data import MasterDataRepository, ScenarioSpec
from .exam_runtime import ExamRuntime, default_audition_row_selector
from .idol_config import build_initial_exam_deck, build_weighted_card_pool, resolve_produce_card_row, sample_card_from_weighted_pool
from .loadout import IdolLoadout
from .produce_item_interpreter import ActiveProduceItem, ProduceItemInterpreter, RuntimeExamStatusEnchantSpec
from .produce_score import calculate_hajime_produce_rating, calculate_nia_produce_rating, resolve_nia_idol_id_from_audition_difficulty_id
from .reward_config import ProduceRewardConfig, build_produce_reward_config


ACTION_STEP_TYPES = {
    'lesson_vocal_normal': 'ProduceStepType_LessonVocalNormal',
    'lesson_dance_normal': 'ProduceStepType_LessonDanceNormal',
    'lesson_visual_normal': 'ProduceStepType_LessonVisualNormal',
    'self_lesson_vocal_normal': 'ProduceStepType_SelfLessonVocalNormal',
    'self_lesson_vocal_sp': 'ProduceStepType_SelfLessonVocalSp',
    'self_lesson_dance_normal': 'ProduceStepType_SelfLessonDanceNormal',
    'self_lesson_dance_sp': 'ProduceStepType_SelfLessonDanceSp',
    'self_lesson_visual_normal': 'ProduceStepType_SelfLessonVisualNormal',
    'self_lesson_visual_sp': 'ProduceStepType_SelfLessonVisualSp',
}

SHOP_CARD_ACTION_TYPES = tuple(f'shop_buy_card_{index}' for index in range(1, 5))
SHOP_DRINK_ACTION_TYPES = tuple(f'shop_buy_drink_{index}' for index in range(1, 5))
SHOP_UPGRADE_ACTION_TYPES = tuple(f'shop_upgrade_card_{index}' for index in range(1, 5))
SHOP_DELETE_ACTION_TYPES = tuple(f'shop_delete_card_{index}' for index in range(1, 5))


def _is_shop_card_action(action_type: str) -> bool:
    """判断动作是否属于咨询里的技能卡槽位。"""

    return action_type in SHOP_CARD_ACTION_TYPES


def _is_shop_drink_action(action_type: str) -> bool:
    """判断动作是否属于咨询里的饮料槽位。"""

    return action_type in SHOP_DRINK_ACTION_TYPES


def _is_shop_upgrade_action(action_type: str) -> bool:
    """判断动作是否属于咨询里的强化槽位。"""

    return action_type in SHOP_UPGRADE_ACTION_TYPES


def _is_shop_delete_action(action_type: str) -> bool:
    """判断动作是否属于咨询里的删除槽位。"""

    return action_type in SHOP_DELETE_ACTION_TYPES


def _shop_slot_index(action_type: str) -> int:
    """解析咨询槽位动作对应的 0-based 下标。"""

    if _is_shop_card_action(action_type):
        return SHOP_CARD_ACTION_TYPES.index(action_type)
    if _is_shop_drink_action(action_type):
        return SHOP_DRINK_ACTION_TYPES.index(action_type)
    if _is_shop_upgrade_action(action_type):
        return SHOP_UPGRADE_ACTION_TYPES.index(action_type)
    if _is_shop_delete_action(action_type):
        return SHOP_DELETE_ACTION_TYPES.index(action_type)
    return -1

ACTION_EFFECT_TYPES = {
    'lesson_vocal_sp': ['ProduceEffectType_VocalAddition', 'ProduceEffectType_LessonVocalSpChangeRatePermilAddition'],
    'lesson_dance_sp': ['ProduceEffectType_DanceAddition', 'ProduceEffectType_LessonDanceSpChangeRatePermilAddition'],
    'lesson_visual_sp': ['ProduceEffectType_VisualAddition', 'ProduceEffectType_LessonVisualSpChangeRatePermilAddition'],
    'lesson_vocal_hard': ['ProduceEffectType_VocalAddition'],
    'lesson_dance_hard': ['ProduceEffectType_DanceAddition'],
    'lesson_visual_hard': ['ProduceEffectType_VisualAddition'],
    'self_lesson_vocal_normal': ['ProduceEffectType_VocalAddition'],
    'self_lesson_vocal_sp': ['ProduceEffectType_VocalAddition'],
    'self_lesson_dance_normal': ['ProduceEffectType_DanceAddition'],
    'self_lesson_dance_sp': ['ProduceEffectType_DanceAddition'],
    'self_lesson_visual_normal': ['ProduceEffectType_VisualAddition'],
    'self_lesson_visual_sp': ['ProduceEffectType_VisualAddition'],
    'activity': ['ProduceEffectType_EventActivityProducePointUp'],
    'business': ['ProduceEffectType_EventBusinessVoteCountUp'],
    'present': ['ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceRewardSet', 'ProduceEffectType_ProduceCardUpgrade'],
    'school_class': ['ProduceEffectType_ProduceReward'],
    'outing': ['ProduceEffectType_StaminaRecoverMultiple', 'ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceCardUpgrade'],
    'activity_supply': ['ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceRewardSet'],
    'refresh': ['ProduceEffectType_StaminaRecoverMultiple'],
    'pre_audition_continue': [],
    **{action_type: [] for action_type in SHOP_CARD_ACTION_TYPES},
    **{action_type: [] for action_type in SHOP_DRINK_ACTION_TYPES},
    **{action_type: [] for action_type in SHOP_UPGRADE_ACTION_TYPES},
    **{action_type: [] for action_type in SHOP_DELETE_ACTION_TYPES},
}

EVENT_ACTION_TYPES = {'activity', 'business', 'present', 'school_class', 'outing', 'activity_supply'}
LESSON_ACTION_TYPES = {
    'lesson_vocal_normal',
    'lesson_dance_normal',
    'lesson_visual_normal',
    'lesson_vocal_sp',
    'lesson_dance_sp',
    'lesson_visual_sp',
    'lesson_vocal_hard',
    'lesson_dance_hard',
    'lesson_visual_hard',
    'self_lesson_vocal_normal',
    'self_lesson_vocal_sp',
    'self_lesson_dance_normal',
    'self_lesson_dance_sp',
    'self_lesson_visual_normal',
    'self_lesson_visual_sp',
}
SP_ACTION_TYPES = {
    'lesson_vocal_sp',
    'lesson_dance_sp',
    'lesson_visual_sp',
    'self_lesson_vocal_sp',
    'self_lesson_dance_sp',
    'self_lesson_visual_sp',
}
HARD_ACTION_TYPES = {
    'lesson_vocal_hard',
    'lesson_dance_hard',
    'lesson_visual_hard',
}
PRE_AUDITION_ACTION_TYPES = {
    'pre_audition_continue',
    'customize_apply',
    'audition_select_1',
    'audition_select_2',
    'audition_select_3',
    'audition_select_4',
    *SHOP_CARD_ACTION_TYPES,
    *SHOP_DRINK_ACTION_TYPES,
    *SHOP_UPGRADE_ACTION_TYPES,
    *SHOP_DELETE_ACTION_TYPES,
}


def _is_lesson_action(action_type: str) -> bool:
    """判断动作是否属于课程或自主训练。"""

    return action_type.startswith('lesson_') or action_type.startswith('self_lesson_')


def _lesson_stat_type(action_type: str) -> str:
    """从动作类型中解析对应的属性分支。"""

    parts = action_type.split('_')
    return parts[1] if action_type.startswith('lesson_') else parts[2]


@dataclass
class ProduceActionCandidate:
    """当前周可选的一个培育动作。"""

    label: str
    action_type: str
    effect_types: list[str]
    produce_effect_ids: list[str]
    success_effect_ids: list[str] = field(default_factory=list)
    fail_effect_ids: list[str] = field(default_factory=list)
    stamina_delta: float = 0.0
    produce_point_delta: float = 0.0
    produce_card_id: str = ''
    success_probability: float = 1.0
    stat_deltas: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 追込课 boost 触发时的均分参数（成功=三参数均分，失败=用 stat_deltas）
    boost_stat_deltas: tuple[float, float, float] = (0.0, 0.0, 0.0)
    available: bool = True
    source_row_id: str = ''
    resource_type: str = ''
    resource_id: str = ''
    resource_level: int = 0
    target_deck_index: int = -1
    customize_id: str = ''
    slot_index: int = -1
    exam_effect_types: list[str] = field(default_factory=list)
    card_category: str = ''
    card_rarity: str = ''
    card_cost_type: str = ''
    auto_skip: bool = False


@dataclass
class ActiveProduceSkillState:
    """运行时中的偶像/支援卡被动技能状态。"""

    skill_id: str
    level: int
    trigger_id: str
    effect_ids: tuple[str, ...]
    fire_limit: int = 0
    fire_count: int = 0
    activation_rate_permille: int = 0
    source: str = 'skill'


class ProduceRuntime:
    """面向训练规划的数据驱动培育运行时。

    这里仍然比正式客户端轻量，但主要转移逻辑已经依赖 ProduceEffect 和事件主数据，
    不再靠少量硬编码卡名来近似。
    """

    def __init__(
        self,
        repository: MasterDataRepository,
        scenario: ScenarioSpec,
        seed: int | None = None,
        idol_loadout: IdolLoadout | None = None,
        produce_reward_config: ProduceRewardConfig | None = None,
    ):
        """初始化培育运行时，并预读取事件、课程和卡组相关主数据。"""

        self.repository = repository
        self.produce_reward_cfg: ProduceRewardConfig = produce_reward_config or build_produce_reward_config()
        self.scenario = scenario
        self.idol_loadout = idol_loadout
        self.np_random = np.random.default_rng(seed)
        self.produce_row = repository.produces.first(scenario.produce_id) or {}
        self.produce_setting = repository.produce_settings.first(str(self.produce_row.get('produceSettingId') or '')) or {}
        self.runtime_setting = (repository.load_table('Setting').rows or [{}])[0]
        self.produce_effects = repository.load_table('ProduceEffect')
        self.event_suggestions = repository.load_table('ProduceStepEventSuggestion')
        self.event_details = repository.load_table('ProduceStepEventDetail')
        self.card_searches = repository.load_table('ProduceCardSearch')
        self.lesson_levels = repository.load_table('ProduceStepLessonLevel')
        self.produce_item_interpreter = ProduceItemInterpreter(repository)
        self.checkpoints = self._build_checkpoint_positions()

        self.state: dict[str, Any] = {}
        self.deck: list[dict[str, Any]] = []
        self.drinks: list[dict[str, Any]] = []
        self.exam_status_enchant_ids: list[str] = []
        self.exam_status_enchant_specs: list[RuntimeExamStatusEnchantSpec] = []
        self.active_produce_items: list[ActiveProduceItem] = []
        self.active_produce_skills: list[ActiveProduceSkillState] = []
        self.support_skills: list[str] = []
        self.selected_support_cards = tuple(self.idol_loadout.support_cards) if self.idol_loadout is not None else ()
        self._candidates: list[ProduceActionCandidate] = []
        self.pending_audition_stage: str | None = None
        self.pending_audition_result: dict[str, Any] | None = None
        self.pre_audition_phase = 'weekly'
        self.remaining_customize_actions = 0
        self.initial_deck_card_ids: set[str] = set()
        self.shop_inventory: dict[str, ProduceActionCandidate] = {}
        self.pre_audition_action_inventory: dict[str, ProduceActionCandidate] = {}
        self.action_samples = self._build_action_samples()
        self.audition_history: list[dict[str, Any]] = []
        self.final_summary: dict[str, Any] = {}
        self.legend_seen_card_ids: set[str] = set()
        self._ability_chain_guard_depth = 0
        # 支援カードイベント / Pアイテムによるカード変更の戻す情報
        self.pending_revert_info: dict[str, Any] | None = None

    def _build_checkpoint_positions(self) -> list[tuple[int, str]]:
        """按路线考试数量计算阶段性考试触发点。"""

        if len(self.scenario.audition_sequence) == 2:
            ratios = [0.5, 1.0]
        else:
            ratios = [0.33, 0.66, 1.0]
        return [
            (max(1, int(round(self.scenario.steps * ratio))), stage)
            for ratio, stage in zip(ratios, self.scenario.audition_sequence)
        ]

    def _base_state(self) -> dict[str, Any]:
        """构造包含属性、成长率和流程加成字段的初始状态。"""

        focus = np.array(self.scenario.score_weights, dtype=np.float32)
        base_stats = 180.0 + focus * 40.0 if self.scenario.route_type == 'nia' else 120.0 + focus * 25.0
        base_stamina = 32.0
        vocal_growth = 0.20
        dance_growth = 0.18
        visual_growth = 0.18
        if self.idol_loadout is not None:
            profile = self.idol_loadout.stat_profile
            base_stats = np.array([profile.vocal, profile.dance, profile.visual], dtype=np.float32)
            base_stamina = float(profile.stamina or base_stamina)
            vocal_growth = float(profile.vocal_growth_rate)
            dance_growth = float(profile.dance_growth_rate)
            visual_growth = float(profile.visual_growth_rate)
        parameter_limit = self._parameter_growth_limit()
        if parameter_limit > 0:
            base_stats = np.clip(base_stats, 0.0, parameter_limit)
        customize_slots = int(self.produce_setting.get('customizeProduceCardCount') or 0)
        return {
            'step': 0,
            'max_steps': int(self.scenario.steps),
            'stamina': float(base_stamina),
            'max_stamina': float(base_stamina),
            'produce_points': float(self.produce_setting.get('initialProducePoint') or 0),
            'fan_votes': 0.0,
            'gold_bonus': 0.0,
            'vocal': float(base_stats[0]),
            'dance': float(base_stats[1]),
            'visual': float(base_stats[2]),
            'vocal_growth': float(vocal_growth),
            'dance_growth': float(dance_growth),
            'visual_growth': float(visual_growth),
            'refresh_used': 0,
            'audition_index': 0,
            'last_exam_score': 0.0,
            'deck_quality': 0.0,
            'drink_quality': 0.0,
            'activity_produce_point_bonus': 0.0,
            'business_vote_bonus': 0.0,
            'lesson_present_point_bonus': 0.0,
            'support_event_point_bonus': 0.0,
            'support_event_stat_bonus': 0.0,
            'support_event_stamina_bonus': 0.0,
            'audition_vote_bonus': 0.0,
            'audition_parameter_bonus': 0.0,
            'audition_difficulty_bonus': 0.0,
            'audition_turn_modifier': 0.0,
            'before_audition_refresh_penalty': 0.0,
            'generic_sp_rate_bonus': 0.0,
            'vocal_sp_rate_bonus': 0.0,
            'dance_sp_rate_bonus': 0.0,
            'visual_sp_rate_bonus': 0.0,
            'reward_card_count_bonus': 0.0,
            'customize_slots': float(customize_slots),
            'exclude_count_bonus': 0.0,
            'reroll_count_bonus': 0.0,
            'shop_discount': 0.0,
            'card_upgrade_probability_bonus': 0.0,
            'shop_card_modify_count': 0.0,
            'shop_card_modified_in_visit': 0.0,
            'producer_level': float(self.idol_loadout.producer_level if self.idol_loadout else 0),
            'idol_rank': float(self.idol_loadout.idol_rank if self.idol_loadout else 0),
            'dearness_level': float(self.idol_loadout.dearness_level if self.idol_loadout else 0),
            'exam_score_bonus_multiplier': float(self.idol_loadout.exam_score_bonus_multiplier if self.idol_loadout else 1.0),
            'parameter_growth_limit': float(parameter_limit),
            'continue_remaining': float(self.produce_setting.get('continueCount') or 0),
            'lessons_taken': 0.0,
            'challenge_lesson_perfect_bonus_ratio': self._challenge_lesson_perfect_bonus_ratio(),
            'challenge_audition_npc_bonus_ratio': self._challenge_audition_npc_bonus_ratio(),
        }

    def _parameter_growth_limit(self) -> float:
        """返回当前模式主数据里的三维成长上限。"""

        return max(float(getattr(self.scenario, 'parameter_growth_limit', 0.0) or 0.0), 0.0)

    def _clamp_parameter_value(self, value: float) -> float:
        """按当前模式上限裁剪单项三维属性。"""

        limit = self._parameter_growth_limit()
        if limit > 0:
            return float(np.clip(value, 0.0, limit))
        return max(float(value), 0.0)

    def _gain_parameter(self, key: str, delta: float) -> None:
        """统一处理培育阶段的三维属性增长，确保不会超过模式上限。"""

        self.state[key] = self._clamp_parameter_value(float(self.state.get(key) or 0.0) + float(delta))

    def reset(self) -> None:
        """重置培育状态、初始牌组、饮料与开场效果。"""

        self.state = self._base_state()
        self.deck = list(build_initial_exam_deck(self.repository, self.scenario, rng=self.np_random, loadout=self.idol_loadout))
        self.initial_deck_card_ids = {str(card.get('id') or '') for card in self.deck if str(card.get('id') or '')}
        self.drinks = list(
            self.repository.build_drink_inventory(
                self.scenario,
                rng=self.np_random,
                plan_type=self.idol_loadout.stat_profile.plan_type if self.idol_loadout is not None else None,
            )
        )
        self.exam_status_enchant_ids = []
        self.exam_status_enchant_specs = []
        self.active_produce_items = []
        self.active_produce_skills = []
        self.support_skills = []
        self.selected_support_cards = tuple(self.idol_loadout.support_cards) if self.idol_loadout is not None else ()
        self.action_samples = self._build_action_samples()
        self.pending_audition_stage = None
        self.pending_audition_result = None
        self.pre_audition_phase = 'weekly'
        self.remaining_customize_actions = 0
        self.shop_inventory = {}
        self.pre_audition_action_inventory = {}
        self._candidates = []
        self.audition_history = []
        self.final_summary = {}
        self.legend_seen_card_ids = {
            str(card.get('id') or '')
            for card in self.deck
            if str(card.get('rarity') or '') == 'ProduceCardRarity_Legend' and str(card.get('id') or '')
        }
        self._ability_chain_guard_depth = 0
        self.pending_revert_info = None
        self._prev_produce_phi: float = 0.0
        self._apply_loadout_start_effects()
        self._dispatch_produce_item_phase('ProducePhaseType_ProduceStart')
        self._trim_drinks()
        self._refresh_quality_scores()
        # 初始化势函数快照（确保 state 已就绪）
        self._prev_produce_phi = self._potential_value_produce(self.produce_reward_cfg)

    def _is_support_or_memory_ability_source(self, source_action_type: str) -> bool:
        """判断当前效果来源是否属于手册限制连锁触发的能力类来源。"""

        return source_action_type in {'support_skill', 'memory_skill'}

    def _apply_loadout_start_effects(self) -> None:
        """把偶像卡自带 P 道具、附魔和开场技能灌入状态。"""

        if self.idol_loadout is None:
            return
        if self.idol_loadout.produce_item_id:
            self._register_produce_item(self.idol_loadout.produce_item_id, source='loadout')
        for extra_item_id in getattr(self.idol_loadout, 'extra_produce_item_ids', ()):
            self._register_produce_item(extra_item_id, source='challenge')
        for skill in self.idol_loadout.produce_skills:
            self._register_produce_skill(skill)

    def _register_produce_skill(self, skill) -> None:
        """把偶像/支援卡提供的培育技能加入运行时。"""

        if not skill.effect_ids:
            return
        skill_rows = [row for row in self.repository.load_table('ProduceSkill').all(skill.skill_id) if int(row.get('level') or 1) == int(skill.level)]
        skill_row = skill_rows[0] if skill_rows else self.repository.load_table('ProduceSkill').first(skill.skill_id)
        if skill_row is None:
            return
        activation_count = max(int(skill_row.get('activationCount') or 0), 0)
        for index in (1, 2, 3):
            trigger_id = str(skill_row.get(f'produceTriggerId{index}') or '')
            effect_id = str(skill_row.get(f'produceEffectId{index}') or '')
            activation_rate = max(int(skill_row.get(f'activationRatePermil{index}') or 0), 0)
            if not effect_id:
                continue
            if trigger_id:
                self.active_produce_skills.append(
                    ActiveProduceSkillState(
                        skill_id=str(skill.skill_id),
                        level=int(skill.level),
                        trigger_id=trigger_id,
                        effect_ids=(effect_id,),
                        fire_limit=activation_count,
                        activation_rate_permille=activation_rate,
                        source='support_skill' if 'p_support_skill-' in str(skill.skill_id) else 'idol_skill',
                    )
                )
            else:
                self._apply_effect_rows([effect_id], source_action_type='idol_skill')

    def _append_exam_status_enchant(
        self,
        enchant_id: str,
        *,
        effect_turn: int | None = None,
        effect_count: int | None = None,
        source: str = 'produce',
        source_identity: str = '',
    ) -> None:
        """记录一个待带入考试运行时的附魔规格。"""

        if not enchant_id:
            return
        self.exam_status_enchant_ids.append(enchant_id)
        self.exam_status_enchant_specs.append(
            RuntimeExamStatusEnchantSpec(
                enchant_id=enchant_id,
                effect_turn=effect_turn,
                effect_count=effect_count,
                source=source,
                source_identity=source_identity,
            )
        )

    def _register_produce_item(self, item_id: str, *, source: str = 'reward') -> None:
        """把一个 P 道具加入运行时库存，并处理无 trigger 的静态效果。"""

        active_item = self.produce_item_interpreter.activate_item(item_id, source=source)
        if active_item is None:
            return
        self.active_produce_items.append(active_item)
        if active_item.trigger is not None:
            return
        for effect in active_item.spec.effects:
            self._apply_resolved_produce_item_effect(active_item, effect, source_action_type='idol_item')

    def _apply_resolved_produce_item_effect(
        self,
        active_item: ActiveProduceItem,
        effect,
        *,
        source_action_type: str,
    ) -> None:
        """应用一条已解析的 item effect。"""

        if effect.effect_type == 'ProduceItemEffectType_ExamStatusEnchant':
            self._append_exam_status_enchant(
                effect.enchant_id,
                effect_turn=effect.effect_turn,
                effect_count=effect.effect_count,
                source='produce_item',
                source_identity=active_item.item_id,
            )
            return
        if effect.effect_type == 'ProduceItemEffectType_ProduceEffect':
            produce_effect = self.repository.produce_effects.first(effect.produce_effect_id)
            if produce_effect is None:
                return
            self._apply_produce_effect(
                produce_effect,
                source_action_type=source_action_type,
                source='produce_item',
                source_identity=active_item.item_id,
            )

    def _dispatch_produce_item_phase(self, phase_type: str, **context: Any) -> None:
        """按 phase 触发当前持有的 P 道具效果。"""

        if self.active_produce_items:
            snapshot = list(self.active_produce_items)
            for active_item in snapshot:
                if not self.produce_item_interpreter.should_fire(
                    active_item,
                    phase_type=phase_type,
                    scenario=self.scenario,
                    state=self.state,
                    deck=self.deck,
                    context=context,
                ):
                    continue
                self.produce_item_interpreter.mark_fired(active_item)
                for effect in active_item.spec.effects:
                    self._apply_resolved_produce_item_effect(active_item, effect, source_action_type='idol_item')
        if self.active_produce_skills:
            snapshot_skills = list(self.active_produce_skills)
            for active_skill in snapshot_skills:
                if self._ability_chain_guard_depth > 0 and self._is_support_or_memory_ability_source(active_skill.source):
                    continue
                if active_skill.fire_limit > 0 and active_skill.fire_count >= active_skill.fire_limit:
                    continue
                trigger = self.produce_item_interpreter.parse_trigger(active_skill.trigger_id)
                if not self.produce_item_interpreter.trigger_matches(
                    trigger,
                    phase_type=phase_type,
                    scenario=self.scenario,
                    state=self.state,
                    deck=self.deck,
                    context=context,
                ):
                    continue
                if active_skill.activation_rate_permille > 0:
                    if self.np_random.random() > (active_skill.activation_rate_permille / 1000.0):
                        continue
                self._apply_effect_rows(list(active_skill.effect_ids), source_action_type=active_skill.source)
                active_skill.fire_count += 1

    def _stage_trigger_phases(self, stage_type: str) -> tuple[str, ...]:
        """把 checkpoint stage type 映射到 item trigger phase。"""

        phases = ['ProducePhaseType_StartAudition']
        if stage_type == 'ProduceStepType_AuditionMid1':
            phases.append('ProducePhaseType_StartAuditionMid1')
        elif stage_type == 'ProduceStepType_AuditionMid2':
            phases.append('ProducePhaseType_StartAuditionMid2')
        elif stage_type == 'ProduceStepType_AuditionFinal':
            phases.append('ProducePhaseType_StartAuditionFinal')
        return tuple(phases)

    def _business_reward_kind(self, source_row_id: str) -> str:
        """从营业事件 row id 中提取产出类型标签。"""

        if 'produce_card' in source_row_id:
            return 'produce_card'
        if 'produce_drink' in source_row_id:
            return 'produce_drink'
        if 'produce_point' in source_row_id:
            return 'produce_point'
        if 'stamina' in source_row_id or 'rest' in source_row_id:
            return 'stamina'
        return ''

    # ── 培育阶段 RL 势函数（PBRS） ─────────────────────────────────

    def _produce_reward_config(self) -> ProduceRewardConfig:
        """返回当前培育奖励配置。"""
        return self.produce_reward_cfg

    def _next_audition_profile(self) -> dict[str, float]:
        """返回当前阶段下一场要面对的考核 profile。"""
        current_idx = int(self.state.get('audition_index') or 0)
        if current_idx >= len(self.scenario.audition_sequence):
            stage_type = str(self.scenario.audition_sequence[-1] or self.scenario.default_stage)
        else:
            stage_type = str(self.scenario.audition_sequence[current_idx] or self.scenario.default_stage)
        return self.repository.battle_profile(self.scenario, stage_type=stage_type)

    def _produce_param_target(self) -> float:
        """从下一场考核 profile 取 parameterBaseLine 作为参数成长目标。"""
        profile = self._next_audition_profile()
        target = float(profile.get('parameter_baseline') or 0.0)
        return target if target > 0 else 300.0

    def _next_param_weights(self) -> tuple[float, float, float]:
        """返回下一场考核对应的 V/D/Vi 权重。"""
        profile = self._next_audition_profile()
        weights = (
            float(profile.get('vocal_weight') or self.scenario.score_weights[0]),
            float(profile.get('dance_weight') or self.scenario.score_weights[1]),
            float(profile.get('visual_weight') or self.scenario.score_weights[2]),
        )
        total = sum(weights)
        if total <= 0:
            return tuple(float(v) for v in self.scenario.score_weights)
        return tuple(float(v) / total for v in weights)

    def _produce_param_target_legacy_final(self) -> float:
        """保留旧逻辑的最终试镜 baseline，便于对照/调试。"""
        final_stage = str(self.scenario.audition_sequence[-1] or '') if self.scenario.audition_sequence else ''
        for row in self.repository.load_table('ProduceStepAuditionDifficulty').rows:
            if (str(row.get('produceId') or '') == self.scenario.produce_id
                    and str(row.get('stepType') or '') == final_stage):
                v = float(row.get('parameterBaseLine') or 0.0)
                if v > 0:
                    return v
        return 300.0  # fallback

    def _phi_param(self) -> float:
        """φ_param：当前加权参数 vs 下一场考核基准线（带边际递减）。"""
        target = self._produce_param_target()
        if target <= 0:
            return 0.0
        weights = np.array(self._next_param_weights(), dtype=np.float32)
        w_sum = float(weights.sum())
        weights = weights / max(w_sum, 1e-6)
        stats = np.array([self.state['vocal'], self.state['dance'], self.state['visual']], dtype=np.float32)
        weighted = float(np.dot(stats, weights))
        ratio = weighted / target
        progress = min(ratio, 1.0)
        overshoot = max(ratio - 1.0, 0.0)
        return progress + math.log1p(overshoot * 2.0) / math.log(3.0) * 0.30

    def _phi_param_legacy_final(self) -> float:
        """保留旧逻辑：当前加权参数 vs 最终试镜基准线（仅用于调试/对照）。"""
        target = self._produce_param_target_legacy_final()
        if target <= 0:
            return 0.0
        weights = np.array(self.scenario.score_weights, dtype=np.float32)
        w_sum = float(weights.sum())
        weights = weights / max(w_sum, 1e-6)
        stats = np.array([self.state['vocal'], self.state['dance'], self.state['visual']], dtype=np.float32)
        weighted = float(np.dot(stats, weights))
        ratio = weighted / target
        progress = min(ratio, 1.0)
        overshoot = max(ratio - 1.0, 0.0)
        return progress + math.log1p(overshoot * 2.0) / math.log(3.0) * 0.30

    def _phi_fan(self) -> float:
        """φ_fan：NIA 路线粉丝票数的双层价值——门槛进度 + 过门槛后的边际递减加成。"""
        if self.scenario.route_type != 'nia':
            return 0.0
        fan_votes = float(self.state.get('fan_votes') or 0.0)
        next_threshold = self._next_fan_vote_threshold()
        if next_threshold <= 0:
            # 全部试镜已解锁后，票数仍有价值，但边际递减
            return min(math.log1p(fan_votes / 4000.0) / math.log(8.0), 1.2)
        unlock_progress = min(fan_votes / max(next_threshold, 1.0), 1.0)
        overflow = max(fan_votes - next_threshold, 0.0)
        overflow_bonus = min(math.log1p(overflow / 4000.0) / math.log(6.0) * 0.25, 0.25)
        return unlock_progress + overflow_bonus

    def _phi_resource(self) -> float:
        """φ_resource：只奖励资源对下一场更高评分的可兑现价值。"""
        cfg = self._produce_reward_config()

        # 1) 轻量卡组 proxy：仅保留小权重，避免完全丢掉构筑方向性
        deck_q = float(self.state.get('deck_quality') or 0.0)
        deck_value = min(deck_q / max(cfg.deck_quality_soft_cap, 1e-6), 1.0)

        # 2) 饮料价值：默认偏向保留到考试；但如果离考试很远，也允许一小部分视作当前转换价值
        drinks = len(self.drinks)
        remaining_to_audition = 0
        if int(self.state.get('audition_index') or 0) < len(self.checkpoints):
            checkpoint_step, _ = self.checkpoints[int(self.state.get('audition_index') or 0)]
            remaining_to_audition = max(int(checkpoint_step - int(self.state.get('step') or 0)), 0)
        if remaining_to_audition <= int(cfg.pre_audition_window_near):
            drink_exam_window = cfg.drink_window_near_weight
        elif remaining_to_audition <= int(cfg.pre_audition_window_mid):
            drink_exam_window = cfg.drink_window_mid_weight
        else:
            drink_exam_window = cfg.drink_window_far_weight
        drink_future_value = min(drinks / max(self.scenario.drink_limit, 1), 1.0) * drink_exam_window
        drink_current_conversion_value = min(drinks / max(self.scenario.drink_limit, 1), 1.0) * (1.0 - drink_exam_window) * 0.25

        # 3) P点可兑现价值：不是余额越多越好，而是现在能否兑现高价值咨询/强化/删除
        produce_points = float(self.state.get('produce_points') or 0.0)
        if self.pre_audition_phase == 'shop':
            pp_value = min(produce_points / max(cfg.pp_left_cap, 1e-6), 1.0)
        else:
            if remaining_to_audition <= int(cfg.pre_audition_window_near):
                pp_window = cfg.pp_window_near_weight
            elif remaining_to_audition <= int(cfg.pre_audition_window_mid):
                pp_window = cfg.pp_window_mid_weight
            else:
                pp_window = cfg.pp_window_far_weight
            pp_value = min(produce_points / max(cfg.pp_left_cap, 1e-6), 1.0) * pp_window

        # 4) 体力价值：鼓励尽量把体力转成收益，但惩罚容易被迫休息跳周
        stamina = float(self.state.get('stamina') or 0.0)
        max_stamina = max(float(self.state.get('max_stamina') or 1.0), 1.0)
        stamina_ratio = stamina / max_stamina
        stamina_actionability = min(stamina / max(cfg.stamina_actionable_threshold, 1e-6), 1.0)
        forced_rest_risk = max(cfg.stamina_low_threshold - stamina_ratio, 0.0) / max(cfg.stamina_low_threshold, 1e-6)
        stamina_runway = 1.0 - min(forced_rest_risk, 1.0)

        weighted = (
            deck_value * cfg.deck_readiness_weight
            + drink_future_value * cfg.drink_future_weight
            + drink_current_conversion_value * cfg.drink_current_conversion_weight
            + pp_value * cfg.pp_optionality_weight
            + stamina_actionability * cfg.stamina_actionability_weight
            + stamina_runway * cfg.stamina_runway_weight
        )
        total_w = (
            cfg.deck_readiness_weight
            + cfg.drink_future_weight
            + cfg.drink_current_conversion_weight
            + cfg.pp_optionality_weight
            + cfg.stamina_actionability_weight
            + cfg.stamina_runway_weight
        )
        return weighted / max(total_w, 1e-6)

    def _potential_value_produce(self, cfg: ProduceRewardConfig) -> float:
        """3 维培育势函数加权和。"""
        return (
            cfg.param_weight    * self._phi_param()
            + cfg.fan_weight    * self._phi_fan()
            + cfg.resource_weight * self._phi_resource()
        )

    def _hard_lesson_level_row(self, action_type: str) -> dict[str, Any]:
        """按当前 hard lesson 动作精确匹配对应的关卡主数据行。"""

        if action_type not in HARD_ACTION_TYPES:
            return {}
        plan_type = ''
        if self.idol_loadout is not None:
            plan_type = str(self.idol_loadout.stat_profile.plan_type or '')
        plan_token_map = {
            'ProducePlanType_Plan1': 'plan1',
            'ProducePlanType_Plan2': 'plan2',
            'ProducePlanType_Plan3': 'plan3',
        }
        plan_token = plan_token_map.get(plan_type, '')
        stat_token = {
            'lesson_vocal_hard': 'vo',
            'lesson_dance_hard': 'da',
            'lesson_visual_hard': 'vi',
        }.get(action_type, '')
        if not stat_token:
            return {}
        lesson_rows = self.repository.load_table('ProduceStepLesson').rows
        candidate_level_ids: list[str] = []
        for row in lesson_rows:
            lesson_id = str(row.get('id') or '')
            if '-hard-' not in lesson_id or f'-{stat_token}-' not in lesson_id:
                continue
            level_id = str(row.get('produceStepLessonLevelId') or '')
            if not level_id:
                continue
            if plan_token and plan_token in level_id:
                candidate_level_ids.append(level_id)
            elif not plan_token:
                candidate_level_ids.append(level_id)
        if not candidate_level_ids:
            return {}
        unique_level_ids = list(dict.fromkeys(candidate_level_ids))
        stage_index = min(max(int(self.state.get('audition_index') or 0), 0), len(unique_level_ids) - 1)
        matched_level_id = unique_level_ids[stage_index]
        return self.lesson_levels.first(matched_level_id) or {}

    def _remaining_weeks_to_next_audition(self) -> int:
        """返回距离下一场考试前还剩多少个 weekly 回合。"""

        current_idx = int(self.state.get('audition_index') or 0)
        if current_idx >= len(self.checkpoints):
            return 0
        checkpoint_step, _ = self.checkpoints[current_idx]
        return max(int(checkpoint_step - int(self.state.get('step') or 0)), 0)

    def _is_first_star_pre_audition_hard_lesson_week(self) -> bool:
        """判断当前是否处于初路线考试前的追込课周。"""

        return self.scenario.route_type == 'first_star' and self._remaining_weeks_to_next_audition() == 2

    def _is_first_star_pre_audition_refresh_week(self) -> bool:
        """判断当前是否处于初路线考试前的强制恢复周。"""

        return self.scenario.route_type == 'first_star' and self._remaining_weeks_to_next_audition() == 1

    def _next_fan_vote_threshold(self) -> float:
        """返回当前阶段下一场试镜的粉丝票数门槛；已全部解锁时返回 0。"""
        current_idx = int(self.state.get('audition_index') or 0)
        if current_idx >= len(self.scenario.audition_sequence):
            return 0.0
        stage_type = str(self.scenario.audition_sequence[current_idx] or '')
        fan_votes = float(self.state.get('fan_votes') or 0.0)
        for row in self.repository.load_table('ProduceStepAuditionDifficulty').rows:
            if (str(row.get('produceId') or '') == self.scenario.produce_id
                    and str(row.get('stepType') or '') == stage_type):
                threshold = float(row.get('voteCount') or 0.0)
                if threshold > fan_votes:
                    return threshold
        return 0.0

    def _present_bonus_produce_points(self) -> float:
        """差入额外 P 点奖励：触发概率随票数上升，但奖励量本身固定。"""

        # 奖励量固定（帮助页只说概率随票数上升，奖励量本身不变）
        return 12.0

    def _business_action_profile(self, source_row_id: str) -> tuple[float, float, str]:
        """把营业类型映射成基础体力/P点影响和附带资源标签。"""

        reward_kind = self._business_reward_kind(source_row_id)
        # 4 类双重收益（P点/体力为主收益；卡为副收益由 _business_action_bonus_card 给）
        # 企业活动 (card + drink)：体力消耗，PP 少
        if reward_kind == 'produce_drink':
            return -3.0, 2.0, 'drink'
        # 自治体活动 (card + PP)：体力消耗低，额外 PP
        if reward_kind == 'produce_point':
            return -2.0, 8.0, 'point'
        # 度假设施 (card + 体力回复)：回体为主
        if reward_kind == 'stamina':
            return 8.0, 2.0, 'stamina'
        # 商业设施 (强化卡)：给强化卡
        return -2.0, 4.0, 'card'

    def _business_big_success(self, source_row_id: str) -> bool:
        """大成功判定：当前参数越高概率越大（帮助页规定）。"""
        param_baseline = 180.0 if self.scenario.route_type == 'nia' else 140.0
        max_stat = max(float(self.state.get('vocal') or 0.0),
                       float(self.state.get('dance') or 0.0),
                       float(self.state.get('visual') or 0.0))
        # 参数达 baseline 时约 30% 大成功，超过 2× baseline 时约 70%
        stat_ratio = max_stat / max(param_baseline, 1.0)
        big_prob = float(np.clip(0.10 + 0.30 * stat_ratio, 0.05, 0.70))
        return bool(self.np_random.random() < big_prob)

    def _business_action_bonus_card(self, source_row_id: str) -> str:
        """按帮助页为各营业类型抽取附带卡牌奖励。"""

        reward_kind = self._business_reward_kind(source_row_id)
        candidates = self._selection_card_pool()
        if not candidates:
            return ''
        sampled = sample_card_from_weighted_pool(candidates, self.np_random)
        if sampled is None:
            return ''
        card_row = dict(sampled)
        # 商業施設 / 度假設施：给强化卡（强化 1 级）
        if reward_kind in {'card', 'stamina'}:
            upgraded = self._lookup_card_row(str(card_row.get('id') or ''), min(int(card_row.get('upgradeCount') or 0) + 1, 1))
            if upgraded is not None:
                card_row = dict(upgraded)
        return str(card_row.get('id') or '')

    def _present_bonus_points_should_trigger(self) -> bool:
        """按 fan_votes 决定差入额外 P 点奖励是否触发。"""

        fan_votes = max(float(self.state.get('fan_votes') or 0.0), 0.0)
        chance = min(0.15 + fan_votes / 120000.0, 0.8)
        return bool(self.np_random.random() < chance)

    def _pre_audition_item_phases(self) -> tuple[str, ...]:
        """返回考试前自动经历的咨询/特训 phase 顺序。"""

        return (
            'ProducePhaseType_StartShop',
            'ProducePhaseType_StartCustomize',
            'ProducePhaseType_EndShop',
        )

    def _shop_price_by_rarity(self, rarity: str, *, kind: str) -> float:
        """按用户指定的近似规则，把 rarity 映射到咨询价格档。"""

        normalized = str(rarity or '').upper()
        if kind == 'card':
            if 'SSR' in normalized:
                return 150.0
            if 'SR' in normalized:
                return 100.0
            return 80.0
        if 'SSR' in normalized:
            return 130.0
        if 'SR' in normalized:
            return 100.0
        return 50.0

    def _shop_card_price(self, card_row: dict[str, Any]) -> float:
        """计算咨询技能卡价格，最多只按一次强化额外加价。"""

        price = self._shop_price_by_rarity(str(card_row.get('rarity') or ''), kind='card')
        if int(card_row.get('upgradeCount') or 0) >= 1:
            price += 20.0
        return price

    def _shop_drink_price(self, drink_row: dict[str, Any]) -> float:
        """计算咨询 P 饮料价格。"""

        return self._shop_price_by_rarity(str(drink_row.get('rarity') or ''), kind='drink')

    def _shop_modify_cost(self) -> float:
        """计算本次相谈执行一次强化/删除所需的 P 点。"""

        base_cost = 100.0 + 25.0 * float(self.state.get('shop_card_modify_count') or 0.0)
        return self._effective_shop_cost(base_cost, 1.0)

    def _discounted_shop_slot_count(self) -> int:
        """每组前 1~2 个槽位会随机带折扣。"""

        return int(self.np_random.integers(1, 3))

    def _shop_discount_ratio(self, slot_index: int, discounted_count: int) -> float:
        """返回当前槽位的折扣倍率。"""

        if slot_index >= discounted_count:
            return 1.0
        return float(self.np_random.choice(np.array([0.8, 0.9], dtype=np.float64)))

    def _effective_shop_cost(self, base_cost: float, discount_ratio: float) -> float:
        """叠加槽位折扣和运行时商店倍率，统一折算最终消费。"""

        runtime_ratio = max(0.0, 1.0 + float(self.state.get('shop_discount') or 0.0))
        effective_ratio = max(0.0, float(discount_ratio)) * runtime_ratio
        return float(max(1, int(np.floor(max(base_cost, 1.0) * effective_ratio))))

    def _allowed_plan_types(self) -> set[str]:
        """返回当前培育可接受的公共/本流派类型集合。"""

        allowed = {'ProducePlanType_Common'}
        if self.idol_loadout is not None and self.idol_loadout.stat_profile.plan_type:
            allowed.add(str(self.idol_loadout.stat_profile.plan_type))
        return allowed

    def _selection_card_pool(self) -> list[dict[str, Any]]:
        """为咨询和三选一卡池复用同一套过滤规则。"""

        weighted_pool = build_weighted_card_pool(self.repository, self.scenario, loadout=self.idol_loadout)
        legend_owned = self._has_legend_card()
        filtered: list[dict[str, Any]] = []
        for card_row in weighted_pool:
            card_id = str(card_row.get('id') or '')
            if not card_id or card_id in self.initial_deck_card_ids:
                continue
            if int(card_row.get('upgradeCount') or 0) > 1:
                continue
            if str(card_row.get('rarity') or '') == 'ProduceCardRarity_Legend':
                if legend_owned:
                    continue
                if card_id in self.legend_seen_card_ids:
                    continue
            origin_idol_card_id = str(card_row.get('originIdolCardId') or '')
            if origin_idol_card_id and (self.idol_loadout is None or origin_idol_card_id != self.idol_loadout.idol_card_id):
                continue
            if str(card_row.get('originSupportCardId') or ''):
                continue
            filtered.append(card_row)
        return filtered

    def _candidate_card_metadata(self, card_row: dict[str, Any]) -> dict[str, Any]:
        """提取技能卡供动作特征编码使用的元信息。"""

        return {
            'exam_effect_types': self.repository.card_exam_effect_types(card_row),
            'card_category': str(card_row.get('category') or ''),
            'card_rarity': str(card_row.get('rarity') or ''),
            'card_cost_type': str(card_row.get('costType') or ''),
        }

    def _candidate_drink_metadata(self, drink_row: dict[str, Any]) -> dict[str, Any]:
        """提取 P 饮料供动作特征编码使用的元信息。"""

        return {
            'exam_effect_types': self.repository.drink_exam_effect_types(drink_row),
            'card_category': '',
            'card_rarity': '',
            'card_cost_type': '',
        }

    def _shop_drink_pool(self) -> list[dict[str, Any]]:
        """按当前流派、等级和显式来源过滤咨询饮料候选池。"""

        producer_level = int(self.state.get('producer_level') or 0)
        allowed_plan_types = self._allowed_plan_types()
        return [
            dict(row)
            for row in self.repository.produce_drinks.rows
            if not row.get('libraryHidden')
            and str(row.get('planType') or 'ProducePlanType_Common') in allowed_plan_types
            and int(row.get('unlockProducerLevel') or 0) <= producer_level
            and not str(row.get('originSupportCardId') or '')
        ]

    def _sample_capped_card_variant(self, card_id: str, *, max_upgrade_count: int) -> dict[str, Any] | None:
        """按既有随机分布抽卡面，但硬性限制最高强化次数。"""

        for _ in range(8):
            sampled = self.repository.sample_random_card_variant(card_id, self.np_random)
            if sampled is not None and int(sampled.get('upgradeCount') or 0) <= max_upgrade_count:
                return dict(sampled)
        for upgrade_count in range(max_upgrade_count, -1, -1):
            matched = self.repository.card_row_by_upgrade(card_id, upgrade_count, fallback_to_canonical=False)
            if matched is not None:
                return dict(matched)
        canonical = self.repository.canonical_card_row(card_id)
        return dict(canonical) if canonical is not None else None

    def _empty_shop_candidate(self, action_type: str) -> ProduceActionCandidate:
        """构造一个已售空或无货的咨询槽位。"""

        return ProduceActionCandidate(
            label=self._action_label(action_type),
            action_type=action_type,
            effect_types=[],
            produce_effect_ids=[],
            available=False,
            slot_index=_shop_slot_index(action_type),
        )

    def _eligible_shop_upgrade_targets(self) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
        """返回相谈里可强化的未强化技能卡，并按收益优先排序。"""

        targets: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        for index, card in enumerate(self.deck):
            if str(card.get('rarity') or '') == 'ProduceCardRarity_Legend':
                continue
            if int(card.get('upgradeCount') or 0) != 0:
                continue
            upgraded = self._lookup_card_row(str(card.get('id') or ''), 1)
            if upgraded is None or int(upgraded.get('upgradeCount') or 0) != 1:
                continue
            current_prior = float(self.repository.card_play_priors.get(str(card.get('id') or ''), 0.0))
            upgraded_prior = float(self.repository.card_play_priors.get(str(upgraded.get('id') or ''), current_prior))
            current_eval = float(card.get('evaluation') or 0.0)
            upgraded_eval = float(upgraded.get('evaluation') or current_eval)
            score = (upgraded_prior - current_prior) + (upgraded_eval - current_eval) / 10.0
            targets.append((score, index, dict(card), dict(upgraded)))
        targets.sort(key=lambda item: (item[0], float(item[2].get('evaluation') or 0.0)), reverse=True)
        return [(index, current_card, upgraded_card) for _, index, current_card, upgraded_card in targets]

    def _eligible_shop_delete_targets(self) -> list[tuple[int, dict[str, Any]]]:
        """返回相谈里可删除的技能卡，并按低价值优先排序。"""

        targets: list[tuple[float, int, dict[str, Any]]] = []
        for index, card in enumerate(self.deck):
            card_id = str(card.get('id') or '')
            if not card_id:
                continue
            prior = float(self.repository.card_play_priors.get(card_id, 0.0))
            evaluation = float(card.get('evaluation') or 0.0)
            score = prior + evaluation / 10.0
            targets.append((score, index, dict(card)))
        targets.sort(key=lambda item: (item[0], float(item[2].get('evaluation') or 0.0)))
        return [(index, card) for _, index, card in targets]

    def _build_shop_card_inventory(self) -> dict[str, ProduceActionCandidate]:
        """生成固定的 4 个技能卡咨询槽位。"""

        offers: dict[str, ProduceActionCandidate] = {}
        available_pool = list(self._selection_card_pool())
        discounted_count = self._discounted_shop_slot_count()
        for slot_index, action_type in enumerate(SHOP_CARD_ACTION_TYPES):
            if not available_pool:
                offers[action_type] = self._empty_shop_candidate(action_type)
                continue
            sampled = sample_card_from_weighted_pool(available_pool, self.np_random)
            if sampled is None:
                offers[action_type] = self._empty_shop_candidate(action_type)
                continue
            sampled_card_id = str(sampled.get('id') or '')
            card_row = self._sample_capped_card_variant(sampled_card_id, max_upgrade_count=1) or dict(sampled)
            discount_ratio = self._shop_discount_ratio(slot_index, discounted_count)
            cost = self._effective_shop_cost(self._shop_card_price(card_row), discount_ratio)
            metadata = self._candidate_card_metadata(card_row)
            offers[action_type] = ProduceActionCandidate(
                label=f'购买技能卡[{slot_index + 1}]:{self.repository.card_name(card_row)}',
                action_type=action_type,
                effect_types=[],
                produce_effect_ids=[],
                produce_point_delta=-cost,
                produce_card_id=sampled_card_id,
                resource_type='ProduceResourceType_ProduceCard',
                resource_id=sampled_card_id,
                resource_level=int(card_row.get('upgradeCount') or 0),
                source_row_id=sampled_card_id,
                slot_index=slot_index,
                exam_effect_types=list(metadata['exam_effect_types']),
                card_category=str(metadata['card_category']),
                card_rarity=str(metadata['card_rarity']),
                card_cost_type=str(metadata['card_cost_type']),
            )
            available_pool = [row for row in available_pool if str(row.get('id') or '') != sampled_card_id]
        return offers

    def _build_shop_drink_inventory(self) -> dict[str, ProduceActionCandidate]:
        """生成固定的 4 个饮料咨询槽位。"""

        offers: dict[str, ProduceActionCandidate] = {}
        available_pool = list(self._shop_drink_pool())
        discounted_count = self._discounted_shop_slot_count()
        for slot_index, action_type in enumerate(SHOP_DRINK_ACTION_TYPES):
            if not available_pool:
                offers[action_type] = self._empty_shop_candidate(action_type)
                continue
            selected_index = int(self.np_random.integers(0, len(available_pool)))
            drink_row = dict(available_pool.pop(selected_index))
            drink_id = str(drink_row.get('id') or '')
            discount_ratio = self._shop_discount_ratio(slot_index, discounted_count)
            cost = self._effective_shop_cost(self._shop_drink_price(drink_row), discount_ratio)
            metadata = self._candidate_drink_metadata(drink_row)
            offers[action_type] = ProduceActionCandidate(
                label=f'购买P饮料[{slot_index + 1}]:{self.repository.drink_name(drink_row)}',
                action_type=action_type,
                effect_types=[],
                produce_effect_ids=[],
                produce_point_delta=-cost,
                resource_type='ProduceResourceType_ProduceDrink',
                resource_id=drink_id,
                source_row_id=drink_id,
                slot_index=slot_index,
                exam_effect_types=list(metadata['exam_effect_types']),
            )
        return offers

    def _build_shop_upgrade_inventory(self) -> dict[str, ProduceActionCandidate]:
        """生成固定的相谈强化候选槽位。"""

        offers: dict[str, ProduceActionCandidate] = {}
        modify_cost = self._shop_modify_cost()
        for slot_index, action_type in enumerate(SHOP_UPGRADE_ACTION_TYPES):
            targets = self._eligible_shop_upgrade_targets()
            if slot_index >= len(targets):
                offers[action_type] = self._empty_shop_candidate(action_type)
                continue
            deck_index, current_card, upgraded_card = targets[slot_index]
            metadata = self._candidate_card_metadata(upgraded_card)
            offers[action_type] = ProduceActionCandidate(
                label=f'强化技能卡[{slot_index + 1}]:{self.repository.card_name(upgraded_card)}',
                action_type=action_type,
                effect_types=[],
                produce_effect_ids=[],
                produce_point_delta=-modify_cost,
                produce_card_id=str(upgraded_card.get('id') or ''),
                resource_type='ProduceResourceType_ProduceCard',
                resource_id=str(upgraded_card.get('id') or ''),
                resource_level=int(upgraded_card.get('upgradeCount') or 0),
                source_row_id=str(current_card.get('id') or ''),
                target_deck_index=deck_index,
                slot_index=slot_index,
                exam_effect_types=list(metadata['exam_effect_types']),
                card_category=str(metadata['card_category']),
                card_rarity=str(metadata['card_rarity']),
                card_cost_type=str(metadata['card_cost_type']),
            )
        return offers

    def _build_shop_delete_inventory(self) -> dict[str, ProduceActionCandidate]:
        """生成固定的相谈删除候选槽位。"""

        offers: dict[str, ProduceActionCandidate] = {}
        modify_cost = self._shop_modify_cost()
        for slot_index, action_type in enumerate(SHOP_DELETE_ACTION_TYPES):
            targets = self._eligible_shop_delete_targets()
            if slot_index >= len(targets):
                offers[action_type] = self._empty_shop_candidate(action_type)
                continue
            deck_index, card_row = targets[slot_index]
            metadata = self._candidate_card_metadata(card_row)
            offers[action_type] = ProduceActionCandidate(
                label=f'删除技能卡[{slot_index + 1}]:{self.repository.card_name(card_row)}',
                action_type=action_type,
                effect_types=[],
                produce_effect_ids=[],
                produce_point_delta=-modify_cost,
                produce_card_id=str(card_row.get('id') or ''),
                resource_type='ProduceResourceType_ProduceCard',
                resource_id=str(card_row.get('id') or ''),
                resource_level=int(card_row.get('upgradeCount') or 0),
                source_row_id=str(card_row.get('id') or ''),
                target_deck_index=deck_index,
                slot_index=slot_index,
                exam_effect_types=list(metadata['exam_effect_types']),
                card_category=str(metadata['card_category']),
                card_rarity=str(metadata['card_rarity']),
                card_cost_type=str(metadata['card_cost_type']),
            )
        return offers

    def _build_shop_inventory(self) -> dict[str, ProduceActionCandidate]:
        """在进入咨询阶段时一次性生成稳定库存。"""

        inventory = self._build_shop_card_inventory()
        inventory.update(self._build_shop_drink_inventory())
        inventory.update(self._build_shop_upgrade_inventory())
        inventory.update(self._build_shop_delete_inventory())
        return inventory

    def _next_checkpoint_stage(self) -> str | None:
        """返回当前是否已经进入考试前置流程。"""

        if self.pending_audition_stage:
            return self.pending_audition_stage
        if self.state['audition_index'] >= len(self.checkpoints):
            return None
        checkpoint_step, stage_type = self.checkpoints[self.state['audition_index']]
        if self.state['step'] < checkpoint_step:
            return None
        return stage_type

    def _customize_options_for_card(self, card: dict[str, Any]) -> list[dict[str, Any]]:
        """从主数据里解析当前卡还可执行的特训选项。"""

        customize_ids = [str(value) for value in card.get('produceCardCustomizeIds', []) if value]
        if not customize_ids:
            return []
        applied_ids = [str(value) for value in card.get('customizedProduceCardCustomizeIds', []) if value]
        grouped_rows = self.repository.load_table('ProduceCardCustomize')
        options: list[dict[str, Any]] = []
        for customize_id in customize_ids:
            level_rows = [
                row
                for row in grouped_rows.by_id.get(customize_id, [])
                if int(row.get('customizeCount') or 0) > 0
            ]
            if not level_rows:
                continue
            next_count = sum(1 for value in applied_ids if value == customize_id) + 1
            next_row = next(
                (row for row in level_rows if int(row.get('customizeCount') or 0) == next_count),
                None,
            )
            if next_row is not None:
                options.append(dict(next_row))
        return options

    def _can_customize_card(self, card: dict[str, Any]) -> bool:
        """按帮助页限制判断当前卡是否允许进入特训。"""

        if int(card.get('upgradeCount') or 0) <= 0:
            return False
        if bool(card.get('isInitialDeckProduceCard')):
            return False
        if str(card.get('category') or '') == 'ProduceCardCategory_Trouble':
            return False
        return True

    def _sample_customize_candidate(self) -> ProduceActionCandidate:
        """特训阶段随机抽一个仍可继续强化的卡面选项。"""

        candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for index, card in enumerate(self.deck):
            if not self._can_customize_card(card):
                continue
            for option in self._customize_options_for_card(card):
                candidates.append((index, card, option))
        if not candidates:
            return ProduceActionCandidate(label='特训技能卡', action_type='customize_apply', effect_types=[], produce_effect_ids=[], available=False)
        deck_index, card_row, customize_row = candidates[int(self.np_random.integers(0, len(candidates)))]
        cost = float(customize_row.get('producePoint') or 0.0)
        return ProduceActionCandidate(
            label=f'特训技能卡:{self.repository.card_name(card_row)}',
            action_type='customize_apply',
            effect_types=[],
            produce_effect_ids=[],
            produce_point_delta=-cost,
            produce_card_id=str(card_row.get('id') or ''),
            source_row_id=str(card_row.get('id') or ''),
            target_deck_index=deck_index,
            customize_id=str(customize_row.get('id') or ''),
        )

    def _build_customize_inventory(self) -> dict[str, ProduceActionCandidate]:
        """在考试前的特训阶段生成稳定的技能卡候选。"""

        inventory: dict[str, ProduceActionCandidate] = {}
        candidate = self._sample_customize_candidate()
        candidate.action_type = 'customize_apply'
        inventory['customize_apply'] = candidate
        return inventory

    def _build_audition_select_inventory(self, stage_type: str) -> dict[str, ProduceActionCandidate]:
        """在 NIA 考试前暴露可选择的试镜候选。"""

        inventory: dict[str, ProduceActionCandidate] = {}
        if self.scenario.route_type != 'nia':
            return inventory
        rows = self.repository.audition_rows(self.scenario, stage_type)
        if not rows:
            return inventory
        fan_votes = max(float(self.state.get('fan_votes') or 0.0), 0.0)
        finale_available = self._finale_available()
        max_number = max(int(row.get('number') or 0) for row in rows)
        grouped_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped_rows[int(row.get('number') or 0)].append(row)
        for number in range(1, max_number + 1):
            key = f'audition_select_{number}'
            candidates = grouped_rows.get(number, [])
            if not candidates:
                inventory[key] = ProduceActionCandidate(label=f'选择试镜 {number}', action_type=key, effect_types=[], produce_effect_ids=[], available=False)
                continue
            selectable = next((row for row in candidates if float(row.get('voteCount') or 0.0) <= fan_votes), None)
            label = '选择 FINALE' if stage_type == self.scenario.audition_sequence[-1] and number == max_number else f'选择试镜 {number}'
            selected_row = selectable or candidates[0]
            selector = f"{str(selected_row.get('id') or '')}:{int(selected_row.get('number') or 0)}"
            inventory[key] = ProduceActionCandidate(
                label=label,
                action_type=key,
                effect_types=[],
                produce_effect_ids=[],
                available=selectable is not None and (number != max_number or finale_available),
                resource_id=selector,
                slot_index=number,
            )
        return inventory

    def _resolve_selected_audition_row_id(self, stage_type: str) -> str | None:
        """返回考试前阶段当前锁定的试镜行 id。"""

        selector_key = str(self.state.get('selected_audition_selector') or '')
        selected_stage_type = str(self.state.get('selected_audition_stage_type') or '')
        if selector_key and selected_stage_type == stage_type:
            candidate = self.pre_audition_action_inventory.get(selector_key)
            if candidate and candidate.resource_id:
                return str(candidate.resource_id)
        return None

    def _select_audition_candidate(self, candidate: ProduceActionCandidate) -> bool:
        """记录当前考试前阶段选中的试镜难度。"""

        if not candidate.resource_id or not self.pending_audition_stage:
            return False
        self.state['selected_audition_selector'] = candidate.action_type
        self.state['selected_audition_stage_type'] = self.pending_audition_stage
        return True

    def _has_pending_customize_choice(self) -> bool:
        """判断当前考试前阶段是否还有可执行的特训。"""

        for key, candidate in self.pre_audition_action_inventory.items():
            if key != 'customize_apply':
                continue
            if self._action_available(candidate):
                return True
        return False

    def _has_pending_audition_choice(self) -> bool:
        """判断当前考试前阶段是否还没有选定试镜行。"""

        if self.scenario.route_type != 'nia' or not self.pending_audition_stage:
            return False
        if str(self.state.get('selected_audition_stage_type') or '') == self.pending_audition_stage and str(self.state.get('selected_audition_selector') or ''):
            return False
        return any(key.startswith('audition_select_') for key in self.pre_audition_action_inventory)

    def _default_audition_selector(self) -> str:
        """给自动策略和兜底逻辑返回当前默认可选的试镜动作。"""

        if not self.pending_audition_stage:
            return ''
        available_numbers = sorted(
            candidate.slot_index
            for key, candidate in self.pre_audition_action_inventory.items()
            if key.startswith('audition_select_') and self._action_available(candidate)
        )
        if not available_numbers:
            return ''
        target_number = available_numbers[-1]
        return f'audition_select_{target_number}'

    def _ensure_default_audition_selected(self) -> None:
        """在未显式选择时自动锁定当前可进入的最高试镜。"""

        selector = self._default_audition_selector()
        if not selector:
            return
        candidate = self.pre_audition_action_inventory.get(selector)
        if candidate is not None:
            self._select_audition_candidate(candidate)
            self._refresh_pre_audition_inventory()

    def _refresh_pre_audition_inventory(self) -> None:
        """按当前考试前阶段重建相谈/特训/试镜候选。"""

        if not self.pending_audition_stage:
            self.pre_audition_action_inventory = {}
            self._candidates = []
            return
        previous_inventory = dict(self.pre_audition_action_inventory)
        inventory: dict[str, ProduceActionCandidate] = {}
        inventory.update(self._build_shop_inventory())
        for key, candidate in previous_inventory.items():
            if key.startswith('shop_') and not candidate.resource_id:
                inventory[key] = candidate
        inventory.update(self._build_customize_inventory())
        inventory.update(self._build_audition_select_inventory(self.pending_audition_stage))
        self.pre_audition_action_inventory = inventory
        self._candidates = []
        self.shop_inventory = {
            key: value
            for key, value in inventory.items()
            if key.startswith('shop_')
        }
        if not self._has_pending_audition_choice():
            return
        selector = self._default_audition_selector()
        if selector:
            self.state['selected_audition_selector'] = selector
            self.state['selected_audition_stage_type'] = self.pending_audition_stage

    def _pre_audition_customize_keys(self) -> list[str]:
        """返回考试前特训动作键。"""

        return [key for key in self.pre_audition_action_inventory if key == 'customize_apply']

    def _pre_audition_audition_keys(self) -> list[str]:
        """返回考试前试镜选择动作键，按编号顺序排列。"""

        return sorted(
            (key for key in self.pre_audition_action_inventory if key.startswith('audition_select_')),
            key=lambda value: int(value.rsplit('_', 1)[-1]),
        )

    def _pre_audition_action_candidate(self, action_type: str) -> ProduceActionCandidate | None:
        """读取考试前阶段的稳定动作候选。"""

        return self.pre_audition_action_inventory.get(action_type)

    def _apply_customize_candidate(self, candidate: ProduceActionCandidate) -> bool:
        """把一条特训主数据应用到当前牌组卡面。"""

        if candidate.target_deck_index < 0 or candidate.target_deck_index >= len(self.deck) or not candidate.customize_id:
            return False
        card = dict(self.deck[candidate.target_deck_index])
        options = self._customize_options_for_card(card)
        customize_row = next((row for row in options if str(row.get('id') or '') == candidate.customize_id), None)
        if customize_row is None:
            return False
        grow_effect_ids = list(card.get('growEffectIds') or [])
        for grow_effect_id in customize_row.get('produceCardGrowEffectIds', []) or []:
            grow_effect_id = str(grow_effect_id or '')
            if grow_effect_id and grow_effect_id not in grow_effect_ids:
                grow_effect_ids.append(grow_effect_id)
        applied_ids = list(card.get('customizedProduceCardCustomizeIds') or [])
        applied_ids.append(candidate.customize_id)
        card['growEffectIds'] = grow_effect_ids
        card['customizedProduceCardCustomizeIds'] = applied_ids
        self.deck[candidate.target_deck_index] = card
        self.remaining_customize_actions = max(self.remaining_customize_actions - 1, 0)
        self._dispatch_produce_item_phase(
            'ProducePhaseType_CustomizeProduceCard',
            stage_type=self.pending_audition_stage or '',
            card=card,
            customize_id=candidate.customize_id,
        )
        return True

    def _mark_shop_modify_used(self) -> None:
        """记录本次相谈已经执行过一次强化/删除。"""

        self.state['shop_card_modified_in_visit'] = 1.0
        self.state['shop_card_modify_count'] = float(self.state.get('shop_card_modify_count') or 0.0) + 1.0

    def _apply_shop_upgrade_candidate(self, candidate: ProduceActionCandidate) -> bool:
        """执行相谈内的技能卡强化。"""

        if candidate.target_deck_index < 0 or candidate.target_deck_index >= len(self.deck):
            return False
        current_card = self.deck[candidate.target_deck_index]
        if int(current_card.get('upgradeCount') or 0) != 0:
            return False
        upgraded = self._lookup_card_row(str(current_card.get('id') or ''), 1)
        if upgraded is None or int(upgraded.get('upgradeCount') or 0) != 1:
            return False
        upgraded_row = dict(upgraded)
        self.deck[candidate.target_deck_index] = upgraded_row
        self._mark_shop_modify_used()
        self._dispatch_produce_item_phase(
            'ProducePhaseType_CustomizeProduceCard',
            stage_type=self.pending_audition_stage or '',
            card=upgraded_row,
        )
        self._dispatch_produce_item_phase('ProducePhaseType_UpgradeProduceCard', card=upgraded_row)
        return True

    def _apply_shop_delete_candidate(self, candidate: ProduceActionCandidate) -> bool:
        """执行相谈内的技能卡删除。"""

        if candidate.target_deck_index < 0 or candidate.target_deck_index >= len(self.deck):
            return False
        deleted_card = dict(self.deck[candidate.target_deck_index])
        self.deck.pop(candidate.target_deck_index)
        self._mark_shop_modify_used()
        self._dispatch_produce_item_phase('ProducePhaseType_DeleteProduceCard', card=deleted_card)
        return True

    def _start_pre_audition_flow(self, stage_type: str) -> None:
        """在 checkpoint 处进入咨询/特训决策流程。"""

        if self.pending_audition_stage == stage_type and self.pre_audition_phase != 'weekly':
            return
        self.pending_audition_stage = stage_type
        self.pre_audition_phase = 'shop'
        self._dispatch_produce_item_phase('ProducePhaseType_StartShop', stage_type=stage_type)
        self._dispatch_produce_item_phase('ProducePhaseType_StartCustomize', stage_type=stage_type)
        self.state['shop_card_modified_in_visit'] = 0.0
        self.state['selected_audition_selector'] = ''
        self.state['selected_audition_stage_type'] = ''
        self.remaining_customize_actions = max(int(self.state.get('customize_slots') or 0.0), 0)
        self._refresh_pre_audition_inventory()

    def _supports_pre_audition_actions(self) -> bool:
        """判断当前场景是否真的把相谈前置动作暴露给训练环境。"""

        return any(action_type in PRE_AUDITION_ACTION_TYPES for action_type in self.scenario.action_types)

    def _advance_pre_audition_flow(self) -> tuple[float, bool, dict[str, Any]]:
        """结束相谈并推进到考试。"""

        stage_type = self.pending_audition_stage
        if not stage_type:
            return 0.0, self.state['step'] >= self.state['max_steps'], {'pre_audition_phase': self.pre_audition_phase}
        self._dispatch_produce_item_phase('ProducePhaseType_EndShop', stage_type=stage_type)
        self._ensure_default_audition_selected()
        audition_slot = self.state['audition_index']
        reward, exam_info = self._run_audition(stage_type, include_pre_audition_phases=False, apply_outcome=False)
        self.pending_audition_result = dict(exam_info)
        self.pending_audition_result['stage_type'] = stage_type
        self.pending_audition_result['reward'] = reward
        self.pending_audition_result['audition_slot'] = audition_slot
        self.pre_audition_phase = 'retry' if int(self.state.get('continue_remaining') or 0) > 0 else 'weekly'
        self.state['shop_card_modified_in_visit'] = 0.0
        self.shop_inventory = {}
        if self.pre_audition_phase == 'weekly':
            terminated, accepted_info = self._accept_pending_audition_result()
            return reward, terminated, accepted_info
        return 0.0, False, {
            'pre_audition_phase': self.pre_audition_phase,
            f'audition_{audition_slot}': exam_info,
            'continue_remaining': int(self.state.get('continue_remaining') or 0),
        }

    def _legend_card_ids(self) -> set[str]:
        """返回主数据里所有传奇技能卡 id。"""

        return {
            str(row.get('id') or '')
            for row in self.repository.load_table('ProduceCard').rows
            if str(row.get('rarity') or '') == 'ProduceCardRarity_Legend' and str(row.get('id') or '')
        }

    def _has_legend_card(self) -> bool:
        """判断当前培育牌组里是否已持有传奇技能卡。"""

        return any(str(card.get('rarity') or '') == 'ProduceCardRarity_Legend' for card in self.deck)

    def _remember_legend_cards(self) -> None:
        """记录当前牌组里已经见过的 Legend 卡。"""

        for card in self.deck:
            if str(card.get('rarity') or '') != 'ProduceCardRarity_Legend':
                continue
            card_id = str(card.get('id') or '')
            if card_id:
                self.legend_seen_card_ids.add(card_id)

    def _finale_available(self) -> bool:
        """判断 NIA 最终场是否已解锁 FINALE。"""

        if self.scenario.route_type != 'nia':
            return False
        dearness_level = int(self.state.get('dearness_level') or 0)
        return dearness_level >= 17

    def _current_finale_route_selected(self) -> bool:
        """判断当前考试前阶段是否已锁定 FINALE。"""

        stage_type = self.pending_audition_stage or ''
        if self.scenario.route_type != 'nia' or not stage_type or stage_type != str(self.scenario.audition_sequence[-1] or ''):
            return False
        selector = str(self.state.get('selected_audition_selector') or '')
        return selector == 'audition_select_4'

    def _selected_audition_number(self, stage_type: str) -> int | None:
        """返回当前考试前阶段锁定的试镜编号。"""

        selector = str(self.state.get('selected_audition_selector') or '')
        selected_stage_type = str(self.state.get('selected_audition_stage_type') or '')
        if not selector or selected_stage_type != stage_type or not selector.startswith('audition_select_'):
            return None
        try:
            return int(selector.rsplit('_', 1)[-1])
        except ValueError:
            return None

    def _selected_audition_label(self, stage_type: str) -> str:
        """返回当前考试前阶段锁定的试镜标签。"""

        number = self._selected_audition_number(stage_type)
        if number is None:
            return ''
        if stage_type == str(self.scenario.audition_sequence[-1] or '') and number == 4:
            return 'FINALE'
        return f'试镜 {number}'

    def _ending_type(self, *, cleared: bool, final_rank: int | None) -> str:
        """根据路线和最终名次生成简化 ending 类型。"""

        if not cleared:
            return 'failed'
        if final_rank is None:
            return 'clear'
        if self.scenario.route_type == 'nia':
            if final_rank == 1:
                return 'nia_win'
            if final_rank <= 3:
                return 'nia_finalist'
            return 'nia_clear'
        if final_rank == 1:
            return 'first_star_a'
        if final_rank == 2:
            return 'first_star_b'
        if final_rank == 3:
            return 'first_star_c'
        return 'first_star_d'

    def _ending_grade(self, *, cleared: bool, final_rank: int | None) -> str:
        """根据是否通关和最终名次生成结局等级。"""

        if not cleared:
            return 'failed'
        if final_rank == 1:
            return 'a'
        if final_rank == 2:
            return 'b'
        if final_rank == 3:
            return 'c'
        return 'd'

    def _p_live_variation(self, *, cleared: bool, final_rank: int | None) -> str:
        """根据通关情况和名次生成 P Live 演出变体。"""

        if not cleared:
            return 'standard'
        if final_rank == 1:
            return 'rank_1'
        return 'standard'

    def _build_final_summary(self, *, cleared: bool, failed_stage_type: str = '') -> dict[str, Any]:
        """构造培育终局摘要，供 service/api 和测试复用。"""

        final_audition = dict(self.audition_history[-1]) if self.audition_history else {}
        final_rank = int(final_audition.get('rank') or 0) or None
        final_score = float(final_audition.get('effective_score') or self.state.get('last_exam_score') or 0.0)
        ending_type = self._ending_type(cleared=cleared, final_rank=final_rank)
        route_label = 'nia' if self.scenario.route_type == 'nia' else 'first_star'
        dearness_level = int(self.state.get('dearness_level') or 0)
        ending_grade = self._ending_grade(cleared=cleared, final_rank=final_rank)
        p_live_variation = self._p_live_variation(cleared=cleared, final_rank=final_rank)
        produce_result: dict[str, Any]
        if self.scenario.route_type == 'nia':
            score_weights = np.array(self.scenario.score_weights, dtype=np.float32)
            score_weights = score_weights / max(float(score_weights.sum()), 1e-6)
            approx_scores = tuple(float(final_score) * float(w) for w in score_weights)
            difficulty_name = 'master' if self.scenario.produce_id == 'produce-005' else 'pro'
            if difficulty_name == 'master':
                stage_name_map = {
                    'ProduceStepType_AuditionMid1': 'quartet',
                    'ProduceStepType_AuditionFinal': 'finale',
                }
            else:
                stage_name_map = {
                    'ProduceStepType_AuditionMid1': 'melobang',
                    'ProduceStepType_AuditionMid2': 'galaxy',
                    'ProduceStepType_AuditionFinal': 'finale',
                }
            stage_name = stage_name_map.get(str(final_audition.get('stage_type') or ''), 'finale')
            idol_id = 1
            if self.idol_loadout is not None:
                idol_card_row = self.repository.load_table('IdolCard').first(self.idol_loadout.idol_card_id) or {}
                idol_id = resolve_nia_idol_id_from_audition_difficulty_id(str(idol_card_row.get('produceStepAuditionDifficultyId') or '')) or 1
            nia_result = calculate_nia_produce_rating(
                difficulty=difficulty_name,
                idol_id=idol_id,
                stage=stage_name,
                pre_params=(float(self.state.get('vocal') or 0.0), float(self.state.get('dance') or 0.0), float(self.state.get('visual') or 0.0)),
                param_bonuses=(0.0, 0.0, 0.0),
                challenge_param_bonus=0.0,
                pre_votes=max(float(self.state.get('fan_votes') or 0.0) - float(final_audition.get('fan_vote_gain') or 0.0), 0.0),
                affection=max(dearness_level, 10),
                scores=approx_scores,
            )
            nia_rating = nia_result.get('rating')
            if nia_rating is None:
                nia_rating = float(nia_result.get('param_rating') or 0.0)
            produce_result = {
                'score': float(nia_rating or 0.0),
                'rank': str(nia_result.get('rank') or 'C'),
                'parameter_total': float(sum(nia_result.get('post_params') or [])),
                'fan_votes': float(nia_result.get('total_votes') or self.state.get('fan_votes') or 0.0),
                'formula_source': 'nia_external_formula_approx_scores',
                'formula_detail': nia_result,
            }
        else:
            difficulty_map = {
                'produce-001': 'regular',
                'produce-002': 'pro',
                'produce-003': 'master',
                'produce-006': 'legend',
            }
            hajime_result = calculate_hajime_produce_rating(
                difficulty=difficulty_map.get(self.scenario.produce_id, 'regular'),
                place=min(max(int(final_rank or 4), 1), 4),
                params=(float(self.state.get('vocal') or 0.0), float(self.state.get('dance') or 0.0), float(self.state.get('visual') or 0.0)),
                final_score=float(final_score),
                midterm_score=0.0,
            )
            produce_result = {
                'score': float(hajime_result.get('rating') or 0.0),
                'rank': str(hajime_result.get('rank') or 'C'),
                'parameter_total': float(self.state.get('vocal') or 0.0) + float(self.state.get('dance') or 0.0) + float(self.state.get('visual') or 0.0),
                'fan_votes': float(self.state.get('fan_votes') or 0.0),
                'formula_source': 'hajime_external_formula',
                'formula_detail': hajime_result,
            }
        return {
            'route': route_label,
            'route_clear': bool(cleared),
            'ending_type': ending_type,
            'assist_mode': self._assist_mode_enabled(),
            'assist_reduction_ratio': 0.15 if self._assist_mode_enabled() else 0.0,
            'failed_stage_type': failed_stage_type,
            'final_rank': final_rank,
            'final_score': final_score,
            'final_exam_score': float(final_audition.get('exam_score') or 0.0),
            'final_audition_stage': str(final_audition.get('stage_type') or ''),
            'fan_votes': float(self.state.get('fan_votes') or 0.0),
            'dearness_level': dearness_level,
            'ending': {
                'type': ending_type,
                'grade': ending_grade,
                'route': route_label,
                'dearness_level': dearness_level,
                'final_rank': final_rank,
            },
            'produce_result': produce_result,
            'p_live': {
                'unlocked': bool(cleared),
                'dearness_level': dearness_level,
                'final_rank': final_rank,
                'variation': p_live_variation,
            },
            'audition_history': [dict(item) for item in self.audition_history],
        }

    def _set_final_summary(self, *, cleared: bool, failed_stage_type: str = '') -> None:
        """在培育终止时写入统一终局摘要。"""

        self.final_summary = self._build_final_summary(cleared=cleared, failed_stage_type=failed_stage_type)

    def _accept_pending_audition_result(self) -> tuple[bool, dict[str, Any]]:
        """接受当前考试结果，并推进培育流程。"""

        result = dict(self.pending_audition_result or {})
        audition_slot = int(result.get('audition_slot') or self.state['audition_index'])
        reward = float(result.get('reward') or 0.0)
        self.pending_audition_result = None
        self.pending_audition_stage = None
        self.pre_audition_phase = 'weekly'
        self.state['audition_index'] += 1
        self.state['last_exam_score'] = float(result.get('effective_score') or 0.0)
        accepted_result = {
            key: value
            for key, value in result.items()
            if key not in {'audition_slot', 'reward', 'deck_quality_gain', 'drink_quality_gain'}
        }
        self.audition_history.append(dict(accepted_result))
        if bool(result.get('cleared')):
            self.state['fan_votes'] += float(result.get('fan_vote_gain') or 0.0)
            self.state['deck_quality'] += float(result.get('deck_quality_gain') or 0.0)
            self.state['drink_quality'] += float(result.get('drink_quality_gain') or 0.0)
        terminated = (
            (not bool(result.get('cleared')))
            or (
                self.state['step'] >= self.state['max_steps']
                and self.state['audition_index'] >= len(self.checkpoints)
                and self.pending_audition_stage is None
            )
        )
        if terminated:
            self._set_final_summary(cleared=bool(result.get('cleared')), failed_stage_type='' if bool(result.get('cleared')) else str(result.get('stage_type') or ''))
        return terminated, {
            'pre_audition_phase': self.pre_audition_phase,
            f'audition_{audition_slot}': accepted_result,
            'accepted_reward': reward,
            'continue_remaining': int(self.state.get('continue_remaining') or 0),
            'final_summary': dict(self.final_summary) if self.final_summary else {},
        }

    def legal_actions(self) -> list[ProduceActionCandidate]:
        """采样当前周的所有动作候选，并标记可用性。"""

        if self.pre_audition_phase == 'retry':
            self._candidates = [
                ProduceActionCandidate(label='接受当前结果', action_type='audition_accept', effect_types=[], produce_effect_ids=[], available=True),
                ProduceActionCandidate(
                    label=f'再挑战({int(self.state.get("continue_remaining") or 0)})',
                    action_type='audition_retry',
                    effect_types=[],
                    produce_effect_ids=[],
                    available=int(self.state.get('continue_remaining') or 0) > 0,
                ),
            ]
            return self._candidates
        candidates: list[ProduceActionCandidate] = []
        if self.pre_audition_phase == 'shop':
            for action_type in self.scenario.action_types:
                if action_type == 'pre_audition_continue':
                    candidate = self._sample_action(action_type)
                elif action_type in self.pre_audition_action_inventory:
                    candidate = replace(self.pre_audition_action_inventory[action_type])
                else:
                    candidate = ProduceActionCandidate(label=action_type, action_type=action_type, effect_types=[], produce_effect_ids=[], available=False)
                candidate.available = self._action_available(candidate)
                candidates.append(candidate)
            self._candidates = candidates
            return candidates
        for action_type in self.scenario.action_types:
            candidate = self._sample_action(action_type)
            candidate.available = self._action_available(candidate)
            candidates.append(candidate)
        if self._is_first_star_pre_audition_refresh_week():
            forced_refresh = next((candidate for candidate in candidates if candidate.action_type == 'refresh'), None)
            if forced_refresh is not None:
                forced_refresh.available = True
                forced_refresh.auto_skip = True
                self._candidates = [forced_refresh]
                return [forced_refresh]
        if self.pre_audition_phase == 'weekly' and not any(candidate.available for candidate in candidates):
            for candidate in candidates:
                if candidate.action_type == 'refresh':
                    candidate.label = '自动跳周'
                    candidate.available = True
                    candidate.auto_skip = True
                    candidate.stamina_delta = 0.0
                    candidate.produce_point_delta = 0.0
                    candidate.produce_effect_ids = []
                    candidate.success_effect_ids = []
                    candidate.fail_effect_ids = []
                    candidate.success_probability = 1.0
                    break
        # 支援カードイベントのカード変更後は戻す選択肢を追加
        if self.pending_revert_info is not None and self.pre_audition_phase == 'weekly':
            revert_candidate = ProduceActionCandidate(
                label='撤回卡牌变更（戻す）',
                action_type='revert_card_change',
                effect_types=[],
                produce_effect_ids=[],
                available=True,
            )
            candidates.append(revert_candidate)
        self._candidates = candidates
        return candidates

    def step(self, action_index: int) -> tuple[float, bool, dict[str, Any]]:
        """执行一个培育动作，并在到达检查点时触发考试。"""

        candidate = self._candidates[action_index]
        if not candidate.available:
            return -0.25, False, {'invalid_action': True}

        # 戻す：撤回上一步支援事件对卡牌的改动
        if candidate.action_type == 'revert_card_change':
            if self.pending_revert_info:
                for change in self.pending_revert_info.get('changes', []):
                    idx = int(change.get('index', -1))
                    orig = change.get('original_card')
                    if orig is not None and 0 <= idx < len(self.deck):
                        self.deck[idx] = dict(orig)
            self.pending_revert_info = None
            self._refresh_quality_scores()
            return 0.0, False, {'action': '撤回卡牌变更', 'action_type': 'revert_card_change'}

        # 每步开始清空戻す状态（只在当前周有效）
        self.pending_revert_info = None

        if self.pre_audition_phase != 'weekly':
            if self.pre_audition_phase == 'retry':
                if candidate.action_type == 'audition_retry':
                    if int(self.state.get('continue_remaining') or 0) <= 0 or self.pending_audition_result is None:
                        return -0.25, False, {'invalid_action': True}
                    self.state['continue_remaining'] = max(float(self.state.get('continue_remaining') or 0.0) - 1.0, 0.0)
                    stage_type = str(self.pending_audition_result.get('stage_type') or self.pending_audition_stage or '')
                    reward, exam_info = self._run_audition(stage_type, include_pre_audition_phases=False, apply_outcome=False)
                    self.pending_audition_result = {
                        **dict(exam_info),
                        'stage_type': stage_type,
                        'reward': reward,
                        'audition_slot': int(self.pending_audition_result.get('audition_slot') or self.state['audition_index']),
                    }
                    return 0.0, False, {
                        'pre_audition_phase': self.pre_audition_phase,
                        f'audition_{int(self.pending_audition_result["audition_slot"])}': exam_info,
                        'continue_remaining': int(self.state.get('continue_remaining') or 0),
                    }
                if candidate.action_type == 'audition_accept':
                    terminated, info = self._accept_pending_audition_result()
                    return float(info.pop('accepted_reward', 0.0)), terminated, info
                return -0.25, False, {'invalid_action': True}
            reward = -0.01
            before_deck_quality = float(self.state.get('deck_quality') or 0.0)
            before_drink_quality = float(self.state.get('drink_quality') or 0.0)
            succeeded = True
            if candidate.action_type == 'pre_audition_continue':
                flow_reward, terminated, info = self._advance_pre_audition_flow()
                info.update(
                    {
                        'action': candidate.label,
                        'action_type': candidate.action_type,
                        'success': True,
                        'vocal': self.state['vocal'],
                        'dance': self.state['dance'],
                        'visual': self.state['visual'],
                        'stamina': self.state['stamina'],
                        'produce_points': self.state['produce_points'],
                        'fan_votes': self.state['fan_votes'],
                    }
                )
                return reward + flow_reward, terminated, info
            if _is_shop_card_action(candidate.action_type):
                self.state['produce_points'] = max(self.state['produce_points'] + candidate.produce_point_delta, 0.0)
                self._grant_resource(candidate.resource_type, candidate.resource_id, candidate.resource_level)
                self.pre_audition_action_inventory[candidate.action_type] = self._empty_shop_candidate(candidate.action_type)
            elif _is_shop_drink_action(candidate.action_type):
                self.state['produce_points'] = max(self.state['produce_points'] + candidate.produce_point_delta, 0.0)
                self._grant_resource(candidate.resource_type, candidate.resource_id, candidate.resource_level)
                self.pre_audition_action_inventory[candidate.action_type] = self._empty_shop_candidate(candidate.action_type)
                self._dispatch_produce_item_phase(
                    'ProducePhaseType_BuyShopItemProduceDrink',
                    stage_type=self.pending_audition_stage or '',
                    drink_id=candidate.resource_id,
                )
            elif _is_shop_upgrade_action(candidate.action_type):
                self.state['produce_points'] = max(self.state['produce_points'] + candidate.produce_point_delta, 0.0)
                succeeded = self._apply_shop_upgrade_candidate(candidate)
                if not succeeded:
                    return -0.25, False, {'invalid_action': True}
            elif _is_shop_delete_action(candidate.action_type):
                self.state['produce_points'] = max(self.state['produce_points'] + candidate.produce_point_delta, 0.0)
                succeeded = self._apply_shop_delete_candidate(candidate)
                if not succeeded:
                    return -0.25, False, {'invalid_action': True}
            elif candidate.action_type == 'customize_apply':
                self.state['produce_points'] = max(self.state['produce_points'] + candidate.produce_point_delta, 0.0)
                succeeded = self._apply_customize_candidate(candidate)
                if not succeeded:
                    return -0.25, False, {'invalid_action': True}
            elif candidate.action_type.startswith('audition_select_'):
                succeeded = self._select_audition_candidate(candidate)
                if not succeeded:
                    return -0.25, False, {'invalid_action': True}
            self._trim_drinks()
            self._refresh_pre_audition_inventory()
            self._refresh_quality_scores()
            return reward, False, {
                'action': candidate.label,
                'action_type': candidate.action_type,
                'success': succeeded,
                'pre_audition_phase': self.pre_audition_phase,
                'reward_breakdown': {
                    'base_step_penalty': reward,
                    'deck_quality_delta_reward': 0.0,
                    'drink_quality_delta_reward': 0.0,
                },
                'vocal': self.state['vocal'],
                'dance': self.state['dance'],
                'visual': self.state['visual'],
                'stamina': self.state['stamina'],
                'produce_points': self.state['produce_points'],
                'fan_votes': self.state['fan_votes'],
                'support_card_count': len(self.selected_support_cards),
                'challenge_lesson_perfect_bonus_ratio': float(self.state.get('challenge_lesson_perfect_bonus_ratio') or 0.0),
                'challenge_audition_npc_bonus_ratio': float(self.state.get('challenge_audition_npc_bonus_ratio') or 0.0),
            }

        reward = -0.01
        _prod_cfg = self._produce_reward_config()
        _phi_before = self._prev_produce_phi  # 用上一步末缓存的势函数值
        phase_context = {
            'action_type': candidate.action_type,
            'source_row_id': candidate.source_row_id,
            'business_reward_kind': self._business_reward_kind(candidate.source_row_id),
        }
        if candidate.auto_skip:
            phase_context['action_type'] = 'auto_skip'
        elif _is_lesson_action(candidate.action_type):
            self._dispatch_produce_item_phase('ProducePhaseType_StartLesson', **phase_context)
        elif candidate.action_type == 'present':
            self._dispatch_produce_item_phase('ProducePhaseType_StartPresent', **phase_context)
        elif candidate.action_type == 'school_class':
            self._dispatch_produce_item_phase('ProducePhaseType_StartPresent', **phase_context)
        elif candidate.action_type == 'outing':
            self._dispatch_produce_item_phase('ProducePhaseType_StartRefresh', **phase_context)
        elif candidate.action_type == 'activity_supply':
            self._dispatch_produce_item_phase('ProducePhaseType_StartPresent', **phase_context)
        elif candidate.action_type == 'refresh':
            self._dispatch_produce_item_phase('ProducePhaseType_StartRefresh', **phase_context)

        succeeded = True
        if not candidate.auto_skip:
            self.state['stamina'] = float(np.clip(self.state['stamina'] + candidate.stamina_delta, 0.0, self.state['max_stamina']))
            self.state['produce_points'] += candidate.produce_point_delta * self._produce_point_rate(candidate.action_type)
            # 追込课 stat 延迟到 boost 判定后再分路应用（其他动作立即应用）
            is_hard_lesson = candidate.action_type in HARD_ACTION_TYPES
            if not is_hard_lesson:
                self._gain_parameter('vocal', candidate.stat_deltas[0])
                self._gain_parameter('dance', candidate.stat_deltas[1])
                self._gain_parameter('visual', candidate.stat_deltas[2])
            if candidate.produce_card_id:
                card_row = resolve_produce_card_row(self.repository, candidate.produce_card_id, loadout=self.idol_loadout)
                if card_row is not None:
                    appended_card = dict(card_row)
                    self.deck.append(appended_card)
                    if str(appended_card.get('rarity') or '') == 'ProduceCardRarity_Legend':
                        card_id = str(appended_card.get('id') or '')
                        if card_id:
                            self.legend_seen_card_ids.add(card_id)

            self._apply_effect_rows(candidate.produce_effect_ids, source_action_type=candidate.action_type)
            succeeded = self.np_random.random() <= candidate.success_probability
            self._apply_effect_rows(candidate.success_effect_ids if succeeded else candidate.fail_effect_ids, source_action_type=candidate.action_type)
            # 追込课：boost 触发（成功）→ 三参数均分；未触发（失败）→ 单参数减半
            if is_hard_lesson:
                apply_deltas = candidate.boost_stat_deltas if succeeded else candidate.stat_deltas
                self._gain_parameter('vocal', apply_deltas[0])
                self._gain_parameter('dance', apply_deltas[1])
                self._gain_parameter('visual', apply_deltas[2])
            if candidate.action_type == 'refresh':
                self.state['refresh_used'] += 1
            if _is_lesson_action(candidate.action_type):
                self.state['lessons_taken'] = float(self.state.get('lessons_taken') or 0.0) + 1.0
                self._dispatch_produce_item_phase('ProducePhaseType_EndLesson', **phase_context)
                self._dispatch_produce_item_phase('ProducePhaseType_EndLessonBeforePresent', **phase_context)
                # 追込ボーナス成立時：ボーカル/ダンス/ビジュアルを条件とするPアイテムをすべて発動
                if is_hard_lesson and succeeded:
                    hard_stat = _lesson_stat_type(candidate.action_type)
                    for _stat in ('vocal', 'dance', 'visual'):
                        if _stat != hard_stat:
                            _extra_ctx = {**phase_context, 'action_type': f'lesson_{_stat}_hard'}
                            self._dispatch_produce_item_phase('ProducePhaseType_EndLesson', **_extra_ctx)
                            self._dispatch_produce_item_phase('ProducePhaseType_EndLessonBeforePresent', **_extra_ctx)
            elif candidate.action_type == 'activity':
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventActivity', **phase_context)
            elif candidate.action_type == 'business':
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventBusiness', **phase_context)
                # 企業活動（drink 类）：额外给 P 饮料（帮助页：スキルカード + Pドリンク）
                if self._business_reward_kind(candidate.source_row_id) == 'produce_drink':
                    if len(self.drinks) < max(self.scenario.drink_limit, 1):
                        drink_candidates = self.repository.build_drink_inventory(
                            self.scenario,
                            max_items=self.scenario.drink_limit,
                            rng=self.np_random,
                            plan_type=self.idol_loadout.stat_profile.plan_type if self.idol_loadout is not None else None,
                        )
                        if drink_candidates:
                            self.drinks.append(dict(drink_candidates[int(self.np_random.integers(0, len(drink_candidates)))]))
                            self._dispatch_produce_item_phase('ProducePhaseType_GetProduceDrink')
                        self._trim_drinks()
                # 大成功：按当前参数高低决定是否触发，触发后额外+50% fan_votes
                if self._business_big_success(candidate.source_row_id):
                    self.state['fan_votes'] += float(self.state.get('fan_votes') or 0.0) * 0.50
            elif candidate.action_type == 'present':
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventSchool', **phase_context)
                self._dispatch_produce_item_phase('ProducePhaseType_EndPresent', **phase_context)
            elif candidate.action_type == 'school_class':
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventSchool', **phase_context)
            elif candidate.action_type == 'outing':
                # 帮助页：外出概率附带 P 饮料
                if self.np_random.random() < 0.35 and len(self.drinks) < max(self.scenario.drink_limit, 1):
                    drink_candidates = self.repository.build_drink_inventory(
                        self.scenario,
                        max_items=self.scenario.drink_limit,
                        rng=self.np_random,
                        plan_type=self.idol_loadout.stat_profile.plan_type if self.idol_loadout is not None else None,
                    )
                    if drink_candidates:
                        self.drinks.append(dict(drink_candidates[int(self.np_random.integers(0, len(drink_candidates)))]))
                        self._dispatch_produce_item_phase('ProducePhaseType_GetProduceDrink')
                    self._trim_drinks()
                self._dispatch_produce_item_phase('ProducePhaseType_EndPresent', **phase_context)
            elif candidate.action_type == 'activity_supply':
                # 活動支給 饮料分支：PP 归零时代表抽到了饮料
                if candidate.produce_point_delta == 0.0 and not candidate.produce_card_id:
                    if len(self.drinks) < max(self.scenario.drink_limit, 1):
                        drink_candidates = self.repository.build_drink_inventory(
                            self.scenario,
                            max_items=self.scenario.drink_limit,
                            rng=self.np_random,
                            plan_type=self.idol_loadout.stat_profile.plan_type if self.idol_loadout is not None else None,
                        )
                        if drink_candidates:
                            self.drinks.append(dict(drink_candidates[int(self.np_random.integers(0, len(drink_candidates)))]))
                            self._dispatch_produce_item_phase('ProducePhaseType_GetProduceDrink')
                        self._trim_drinks()
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventActivity', **phase_context)
                self._dispatch_produce_item_phase('ProducePhaseType_EndPresent', **phase_context)

        self.state['step'] += 1
        self._trim_drinks()
        self._refresh_quality_scores()

        _reward_breakdown = {
            'base_step_penalty': reward,
            'pbrs_delta': 0.0,
            'terminal_reward': 0.0,
            'pp_left_penalty': 0.0,
        }
        info = {
            'action': candidate.label,
            'action_type': 'auto_skip' if candidate.auto_skip else candidate.action_type,
            'success': succeeded,
            'pre_audition_phase': self.pre_audition_phase,
            'vocal': self.state['vocal'],
            'dance': self.state['dance'],
            'visual': self.state['visual'],
            'stamina': self.state['stamina'],
            'produce_points': self.state['produce_points'],
            'fan_votes': self.state['fan_votes'],
            'support_card_count': len(self.selected_support_cards),
            'challenge_lesson_perfect_bonus_ratio': float(self.state.get('challenge_lesson_perfect_bonus_ratio') or 0.0),
            'challenge_audition_npc_bonus_ratio': float(self.state.get('challenge_audition_npc_bonus_ratio') or 0.0),
        }

        # 考试会复用当前已经组好的牌组、饮料和继承下来的附魔。
        while self.state['audition_index'] < len(self.checkpoints):
            checkpoint_step, stage_type = self.checkpoints[self.state['audition_index']]
            if self.state['step'] < checkpoint_step:
                break
            if self._supports_pre_audition_actions():
                self._start_pre_audition_flow(stage_type)
                info['pre_audition_phase'] = self.pre_audition_phase
                break
            exam_reward, exam_info = self._run_audition(stage_type, apply_outcome=False)
            if int(self.state.get('continue_remaining') or 0) > 0:
                self.pending_audition_stage = stage_type
                self.pending_audition_result = {
                    **dict(exam_info),
                    'stage_type': stage_type,
                    'reward': exam_reward,
                    'audition_slot': int(self.state['audition_index']),
                }
                self.pre_audition_phase = 'retry'
                info['pre_audition_phase'] = self.pre_audition_phase
                info[f'audition_{self.state["audition_index"]}'] = exam_info
                break
            reward += exam_reward
            self.pending_audition_result = {
                **dict(exam_info),
                'stage_type': stage_type,
                'reward': exam_reward,
                'audition_slot': int(self.state['audition_index']),
            }
            terminated, accepted_info = self._accept_pending_audition_result()
            info.update(accepted_info)
            if terminated:
                info['final_summary'] = dict(self.final_summary) if self.final_summary else {}
                return reward, True, info

        auto_skipped_weeks = 0
        while (
            self.pre_audition_phase == 'weekly'
            and self.pending_audition_stage is None
            and self.state['step'] < self.state['max_steps']
            and not self._has_available_weekly_action()
        ):
            self.state['step'] += 1
            auto_skipped_weeks += 1

            while self.state['audition_index'] < len(self.checkpoints):
                checkpoint_step, stage_type = self.checkpoints[self.state['audition_index']]
                if self.state['step'] < checkpoint_step:
                    break
                if self._supports_pre_audition_actions():
                    self._start_pre_audition_flow(stage_type)
                    info['pre_audition_phase'] = self.pre_audition_phase
                    break
                exam_reward, exam_info = self._run_audition(stage_type, apply_outcome=False)
                if int(self.state.get('continue_remaining') or 0) > 0:
                    self.pending_audition_stage = stage_type
                    self.pending_audition_result = {
                        **dict(exam_info),
                        'stage_type': stage_type,
                        'reward': exam_reward,
                        'audition_slot': int(self.state['audition_index']),
                    }
                    self.pre_audition_phase = 'retry'
                    info['pre_audition_phase'] = self.pre_audition_phase
                    info[f'audition_{self.state["audition_index"]}'] = exam_info
                    break
                reward += exam_reward
                self.pending_audition_result = {
                    **dict(exam_info),
                    'stage_type': stage_type,
                    'reward': exam_reward,
                    'audition_slot': int(self.state['audition_index']),
                }
                terminated, accepted_info = self._accept_pending_audition_result()
                info.update(accepted_info)
                if terminated:
                    info['auto_skipped_weeks'] = auto_skipped_weeks
                    info['final_summary'] = dict(self.final_summary) if self.final_summary else {}
                    return reward, True, info
            if self.pre_audition_phase != 'weekly' or self.pending_audition_stage is not None:
                break
        if auto_skipped_weeks > 0:
            info['auto_skipped_weeks'] = auto_skipped_weeks

        terminated = (
            self.state['step'] >= self.state['max_steps']
            and self.state['audition_index'] >= len(self.checkpoints)
            and self.pending_audition_stage is None
        )
        if terminated and not self.final_summary:
            self._set_final_summary(cleared=True)
            info['final_summary'] = dict(self.final_summary)

        # ── 密集势函数塑形（PBRS）──────────────────────────────────
        _phi_after = self._potential_value_produce(_prod_cfg)
        self._prev_produce_phi = _phi_after
        if _prod_cfg.shape_scale > 0:
            _reward_breakdown['pbrs_delta'] = _prod_cfg.shape_scale * (_phi_after - _phi_before)
            reward += _reward_breakdown['pbrs_delta']

        # ── 终局奖励：最终评分/评级主导，流程完成作约束，剩余P点轻惩罚 ────────
        if terminated and self.final_summary:
            produce_result = self.final_summary.get('produce_result') or {}
            raw_score = float(produce_result.get('score') or 0.0)
            norm_score = math.log1p(max(raw_score, 0.0)) / math.log1p(max(_prod_cfg.score_norm_log_base, 1.0))
            grade = str(produce_result.get('rank') or '')
            grade_bonus_map = {
                'S4': _prod_cfg.terminal_grade_s4,
                'SSS+': _prod_cfg.terminal_grade_sss_plus,
                'SSS': _prod_cfg.terminal_grade_sss,
                'SS+': _prod_cfg.terminal_grade_ss_plus,
                'SS': _prod_cfg.terminal_grade_ss,
                'S+': _prod_cfg.terminal_grade_s_plus,
                'S': _prod_cfg.terminal_grade_s,
                'A+': _prod_cfg.terminal_grade_a,
                'A': _prod_cfg.terminal_grade_a,
                'B+': _prod_cfg.terminal_grade_b_plus,
                'B': _prod_cfg.terminal_grade_b,
                'C+': _prod_cfg.terminal_grade_c_plus,
                'C': _prod_cfg.terminal_grade_c,
                'D': _prod_cfg.terminal_grade_d,
                'failed': _prod_cfg.terminal_grade_failed,
            }
            grade_bonus = grade_bonus_map.get(grade, _prod_cfg.terminal_grade_c)
            route_bonus = _prod_cfg.terminal_route_clear_bonus if bool(self.final_summary.get('route_clear')) else _prod_cfg.terminal_route_fail_penalty
            stage_progress = min(float(self.state.get('audition_index') or 0.0) / max(len(self.checkpoints), 1), 1.0)
            # P点不继承，剩余太多说明资源没有转成最终评分
            pp_left_penalty = min(float(self.state.get('produce_points') or 0.0) / max(_prod_cfg.pp_left_cap, 1.0), 1.0) * _prod_cfg.terminal_pp_left_waste_penalty
            _reward_breakdown['pp_left_penalty'] = -pp_left_penalty

            fan_aux = 0.0
            if self.scenario.route_type == 'nia':
                current_fan = float(self.state.get('fan_votes') or 0.0)
                requirement = self._next_fan_vote_threshold()
                if requirement > 0:
                    overflow = max(current_fan - requirement, 0.0)
                    fan_aux = min(math.log1p(overflow / max(_prod_cfg.fan_overflow_scale, 1.0)) / math.log(max(_prod_cfg.fan_unlock_log_base, 1.01)), _prod_cfg.fan_overflow_cap) * _prod_cfg.terminal_fan_aux_scale
                else:
                    fan_aux = min(math.log1p(current_fan / max(_prod_cfg.fan_full_unlock_scale, 1.0)) / math.log(max(_prod_cfg.fan_full_unlock_log_base, 1.01)), _prod_cfg.fan_progress_cap) * _prod_cfg.terminal_fan_aux_scale

            # NIA 外部评分在 voteRank < A 时没有总 rating，此时给一点 param_rating fallback，避免终局分断崖掉零
            nia_param_fallback = 0.0
            if self.scenario.route_type == 'nia' and str(produce_result.get('formula_source') or '').startswith('nia_external_formula'):
                detail = produce_result.get('formula_detail') or {}
                if detail.get('rating') is None:
                    nia_param_fallback = min(float(detail.get('param_rating') or 0.0) / 3000.0, 1.0) * _prod_cfg.terminal_nia_param_fallback_weight
                if detail.get('vote_rank') is not None:
                    nia_param_fallback += _prod_cfg.terminal_nia_vote_rank_bonus

            terminal_r = (
                norm_score * _prod_cfg.terminal_score_scale
                + grade_bonus
                + route_bonus
                + stage_progress * _prod_cfg.terminal_stage_progress_weight
                + fan_aux
                + nia_param_fallback
                - pp_left_penalty
            )
            terminal_r *= _prod_cfg.reward_scale
            _reward_breakdown['terminal_reward'] = float(np.clip(terminal_r, -_prod_cfg.reward_clip, _prod_cfg.reward_clip))
            reward += _reward_breakdown['terminal_reward']

        reward = float(np.clip(reward * _prod_cfg.reward_scale, -_prod_cfg.reward_clip, _prod_cfg.reward_clip))
        info['reward_breakdown'] = _reward_breakdown
        return reward, terminated, info

    def _has_available_weekly_action(self) -> bool:
        """判断当前周在 weekly 阶段是否存在至少一个可选动作。"""

        if self.pre_audition_phase != 'weekly':
            return True
        for action_type in self.scenario.action_types:
            if action_type in PRE_AUDITION_ACTION_TYPES:
                continue
            candidate = self._sample_action(action_type)
            if self._action_available(candidate):
                return True
        return False

    def _action_available(self, candidate: ProduceActionCandidate) -> bool:
        """根据体力和休息次数判断动作当前是否可用。"""

        if self.pre_audition_phase != 'weekly':
            if self.pre_audition_phase == 'retry':
                if candidate.action_type == 'audition_accept':
                    return self.pending_audition_result is not None
                if candidate.action_type == 'audition_retry':
                    return self.pending_audition_result is not None and int(self.state.get('continue_remaining') or 0) > 0
                return False
            if self.pre_audition_phase == 'shop':
                if _is_shop_card_action(candidate.action_type):
                    return bool(candidate.resource_id) and self.state['produce_points'] + candidate.produce_point_delta >= 0.0
                if _is_shop_drink_action(candidate.action_type):
                    return (
                        bool(candidate.resource_id)
                        and len(self.drinks) < max(self.scenario.drink_limit, 1)
                        and self.state['produce_points'] + candidate.produce_point_delta >= 0.0
                    )
                if _is_shop_upgrade_action(candidate.action_type) or _is_shop_delete_action(candidate.action_type):
                    return (
                        candidate.target_deck_index >= 0
                        and float(self.state.get('shop_card_modified_in_visit') or 0.0) < 1.0
                        and self.state['produce_points'] + candidate.produce_point_delta >= 0.0
                    )
                if candidate.action_type == 'customize_apply':
                    return self.remaining_customize_actions > 0 and candidate.target_deck_index >= 0 and self.state['produce_points'] + candidate.produce_point_delta >= 0.0
                if candidate.action_type.startswith('audition_select_'):
                    if not candidate.resource_id:
                        return False
                    selected_stage_type = str(self.state.get('selected_audition_stage_type') or '')
                    selected_selector = str(self.state.get('selected_audition_selector') or '')
                    if selected_stage_type == str(self.pending_audition_stage or '') and selected_selector:
                        return False
                    is_final_stage = bool(self.pending_audition_stage and self.pending_audition_stage == str(self.scenario.audition_sequence[-1] or ''))
                    if is_final_stage and candidate.slot_index == 4:
                        return self._finale_available()
                    return True
                return candidate.action_type == 'pre_audition_continue'
            return False
        if candidate.action_type in PRE_AUDITION_ACTION_TYPES:
            return False
        if self._is_first_star_pre_audition_hard_lesson_week() and candidate.action_type not in HARD_ACTION_TYPES:
            return False
        if self._is_first_star_pre_audition_refresh_week() and candidate.action_type != 'refresh':
            return False
        if candidate.action_type == 'refresh':
            if self.scenario.produce_id in {'produce-003', 'produce-006'} and float(self.state.get('lessons_taken') or 0.0) < 1.0:
                return False
            # maxRefreshCount == 0 = 无次数限制（初路线）；>0 = 硬上限（NIA/レジェンド 为 4 次）
            max_refresh = self.scenario.max_refresh_count
            if max_refresh > 0:
                return self.state['refresh_used'] < max_refresh
            return True
        if candidate.action_type != 'refresh' and self.state['stamina'] <= 0.0:
            return False
        if candidate.action_type != 'refresh' and self.state['stamina'] + candidate.stamina_delta < 0.0:
            return False
        return True

    def _build_action_samples(self) -> dict[str, list[dict[str, Any]]]:
        """预先按动作类型整理事件候选样本。"""

        samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        support_card_level_by_id = {
            str(item.support_card_id): int(item.support_card_level)
            for item in self.selected_support_cards
            if str(getattr(item, 'support_card_id', '') or '')
        }
        unlocked_support_event_ids: set[str] = set()
        if support_card_level_by_id:
            for row in self.repository.produce_event_support_cards.rows:
                support_card_id = str(row.get('supportCardId') or '')
                support_level = support_card_level_by_id.get(support_card_id)
                if support_level is None:
                    continue
                if int(row.get('supportCardLevel') or 0) > support_level:
                    continue
                detail_id = str(row.get('produceStepEventDetailId') or '')
                if detail_id:
                    unlocked_support_event_ids.add(detail_id)
        for row in self.event_suggestions.rows:
            step_type = str(row.get('stepType') or 'ProduceStepType_Unknown')
            for action_type, mapped_step_type in ACTION_STEP_TYPES.items():
                if step_type == mapped_step_type:
                    samples[action_type].append(row)
        for row in self.event_details.rows:
            event_type = str(row.get('eventType') or 'ProduceEventType_Unknown')
            if event_type == 'ProduceEventType_Activity':
                samples['activity'].append(row)
                samples['activity_supply'].append(row)
            elif event_type == 'ProduceEventType_Business':
                samples['business'].append(row)
            elif event_type == 'ProduceEventType_School':
                # 授業は学校イベントのみ；外出・差入には混入させない
                samples['school_class'].append(row)
            elif event_type == 'ProduceEventType_Character':
                samples['outing'].append(row)
                samples['present'].append(row)
            elif event_type == 'ProduceEventType_SupportCard':
                if not unlocked_support_event_ids:
                    continue
                if str(row.get('id') or '') not in unlocked_support_event_ids:
                    continue
                # 支援卡事件只进差入（present），不混入授業 / 外出 / 活动支给
                samples['present'].append(row)
        return samples

    def _sample_action(self, action_type: str) -> ProduceActionCandidate:
        """为指定动作类型采样一条本周可执行动作。"""

        if action_type == 'pre_audition_continue':
            return ProduceActionCandidate(
                label='继续前进',
                action_type=action_type,
                effect_types=[],
                produce_effect_ids=[],
            )
        if _is_shop_card_action(action_type) or _is_shop_drink_action(action_type):
            return replace(self.shop_inventory.get(action_type, self._empty_shop_candidate(action_type)))
        if _is_shop_upgrade_action(action_type) or _is_shop_delete_action(action_type):
            return replace(self.shop_inventory.get(action_type, self._empty_shop_candidate(action_type)))
        if action_type == 'refresh':
            is_pre_audition_refresh = self._is_first_star_pre_audition_refresh_week()
            recovery_permille = float(
                self.produce_setting.get('beforeAuditionRefreshStaminaRecoveryPermil' if is_pre_audition_refresh else 'refreshStaminaRecoveryPermil')
                or 700
            )
            label = '考前恢复' if is_pre_audition_refresh else '休息'
            return ProduceActionCandidate(
                label=label,
                action_type=action_type,
                effect_types=ACTION_EFFECT_TYPES[action_type],
                produce_effect_ids=self._effect_ids_for_types(ACTION_EFFECT_TYPES[action_type]),
                stamina_delta=self.state['max_stamina'] * (recovery_permille / 1000.0),
                auto_skip=is_pre_audition_refresh,
            )
        rows = self.action_samples.get(action_type, [])
        if rows:
            row = rows[int(self.np_random.integers(0, len(rows)))]
            produce_effect_ids = [str(value) for value in row.get('produceEffectIds', []) if value]
            success_effect_ids = [str(value) for value in row.get('successProduceEffectIds', []) if value]
            fail_effect_ids = [str(value) for value in row.get('failProduceEffectIds', []) if value]
            stamina_delta = -float(row.get('stamina') or 0)
            produce_point_delta = float(row.get('producePoint') or 0)
            produce_card_id = str(row.get('produceCardId') or '')
            effect_types = self._effect_types_for_ids(produce_effect_ids + success_effect_ids + fail_effect_ids)
            if not effect_types:
                effect_types = list(ACTION_EFFECT_TYPES.get(action_type, []))
            success_probability = float(row.get('successProbabilityPermyriad') or 10000) / 10000.0
            if action_type in SP_ACTION_TYPES:
                success_probability += self._sp_rate_bonus(action_type)
            success_probability = float(np.clip(success_probability, 0.05, 1.0))
            stat_deltas = (0.0, 0.0, 0.0)
            if action_type == 'school_class':
                stamina_delta = min(stamina_delta, -8.0)
                stat_deltas = (
                    36.0 * (1.0 + float(self.state.get('vocal_growth') or 0.0)),
                    24.0 * (1.0 + float(self.state.get('dance_growth') or 0.0)),
                    24.0 * (1.0 + float(self.state.get('visual_growth') or 0.0)),
                )
            elif action_type == 'outing':
                stamina_delta = max(stamina_delta, self.state['max_stamina'] * 0.35)
                produce_point_delta = -max(abs(produce_point_delta), 12.0)
                # 帮助页：外出可能获得 P饮料 / 技能卡强化删除变化（通过 produce_effect_ids 走）
            elif action_type == 'activity_supply':
                stamina_delta = 0.0
                # 帮助页：活動支給 给 スキルカード / Pポイント / Pドリンク 三选一
                _supply_roll = float(self.np_random.random())
                if _supply_roll < 0.40:
                    # 给卡（由 produce_card_id 驱动，PP 保持主数据值）
                    if not produce_card_id:
                        produce_card_id = self._business_action_bonus_card('')
                    produce_point_delta = 0.0
                elif _supply_roll < 0.75:
                    # 给 PP
                    produce_point_delta = max(produce_point_delta, 18.0)
                else:
                    # 给饮料（在 step 里处理，PP 归零）
                    produce_point_delta = 0.0
            elif action_type == 'business':
                stamina_delta, produce_point_delta, reward_kind = self._business_action_profile(str(row.get('id') or ''))
                # 帮助页：4 类营业均包含 スキルカードの獲得 作为主收益之一
                produce_card_id = self._business_action_bonus_card(str(row.get('id') or ''))
            elif action_type == 'present' and self._present_bonus_points_should_trigger():
                produce_point_delta += self._present_bonus_produce_points()
            return ProduceActionCandidate(
                label=self._action_label(action_type),
                action_type=action_type,
                effect_types=effect_types,
                produce_effect_ids=produce_effect_ids,
                success_effect_ids=success_effect_ids,
                fail_effect_ids=fail_effect_ids,
                stamina_delta=stamina_delta,
                produce_point_delta=produce_point_delta,
                produce_card_id=produce_card_id,
                success_probability=success_probability,
                stat_deltas=stat_deltas,
                source_row_id=str(row.get('id') or ''),
            )
        if action_type in HARD_ACTION_TYPES:
            lesson_profiles = self.repository.lesson_profile_stats
            normal_profile = max(float(lesson_profiles.get('normal') or 170.0), 1.0)
            hard_profile = max(float(lesson_profiles.get('hard') or normal_profile), normal_profile)
            hard_scale = hard_profile / normal_profile
            stage_scale = 1.0 + 0.08 * float(self.state['audition_index'])
            # boost 时的全量增益（三参数均分）
            full_gain = 60.0 * hard_scale * stage_scale
            stamina_cost = 5.0 + 1.5 * hard_scale
            produce_point_delta = 2.0 + 1.2 * hard_scale
            # 从主数据 ProduceStepLessonLevel 读取 successThreshold/resultTargetValueLimit
            level_row = self._hard_lesson_level_row(action_type)
            success_threshold = float(level_row.get('successThreshold') or 75.0)
            result_limit = float(level_row.get('resultTargetValueLimit') or 300.0)
            # threshold_ratio ≈ 0.25 表示只需达到 25% 的满分即可触发 boost
            threshold_ratio = float(np.clip(success_threshold / max(result_limit, 1.0), 0.05, 0.95))
            boost_probability = float(np.clip(1.0 - threshold_ratio - 0.02 * hard_scale, 0.55, 0.92))
            stat_type = _lesson_stat_type(action_type)
            # 失败时：单参数，增益减半
            no_boost_gain = full_gain * 0.5
            no_boost_deltas: tuple[float, float, float] = {
                'vocal': (no_boost_gain, 0.0, 0.0),
                'dance': (0.0, no_boost_gain, 0.0),
                'visual': (0.0, 0.0, no_boost_gain),
            }[stat_type]
            # 成功时：三参数均分（追込ボーナス）
            boost_gain = full_gain / 3.0
            boost_deltas: tuple[float, float, float] = (boost_gain, boost_gain, boost_gain)
            return ProduceActionCandidate(
                label=self._action_label(action_type),
                action_type=action_type,
                effect_types=list(ACTION_EFFECT_TYPES.get(action_type, [])),
                produce_effect_ids=[],
                stamina_delta=-stamina_cost,
                produce_point_delta=produce_point_delta,
                success_probability=boost_probability,
                stat_deltas=no_boost_deltas,
                boost_stat_deltas=boost_deltas,
                source_row_id=f'synthetic-hard-{action_type}',
            )
        synthetic_types = list(ACTION_EFFECT_TYPES.get(action_type, []))
        if action_type.startswith('self_lesson_'):
            stage_index = min(max(int(self.state['audition_index']) + 1, 1), 3)
            scenario_code = self.scenario.produce_id.replace('-', '_')
            lesson_tier = 'sp' if action_type.endswith('_sp') else 'normal'
            lesson_row = self.repository.load_table('ProduceStepSelfLesson').first(f'self_lesson-{scenario_code}-{stage_index:02d}-{lesson_tier}') or {}
            parameter_gain = float(lesson_row.get('parameter') or (120 if lesson_tier == 'sp' else 100))
            stamina_cost = float(lesson_row.get('stamina') or (8 if lesson_tier == 'sp' else 6))
            stat_type = _lesson_stat_type(action_type)
            stat_deltas = {
                'vocal': (parameter_gain, 0.0, 0.0),
                'dance': (0.0, parameter_gain, 0.0),
                'visual': (0.0, 0.0, parameter_gain),
            }[stat_type]
            return ProduceActionCandidate(
                label=self._action_label(action_type),
                action_type=action_type,
                effect_types=synthetic_types,
                produce_effect_ids=[],
                stamina_delta=-stamina_cost,
                produce_point_delta=0.0,
                success_probability=1.0,
                stat_deltas=stat_deltas,
            )
        if not synthetic_types and _is_lesson_action(action_type):
            stat_type = _lesson_stat_type(action_type)
            mapping = {
                'vocal': 'ProduceEffectType_VocalAddition',
                'dance': 'ProduceEffectType_DanceAddition',
                'visual': 'ProduceEffectType_VisualAddition',
            }
            synthetic_types = [mapping[stat_type]]
        success_probability = 1.0
        stamina_delta = 0.0
        produce_point_delta = 0.0
        if action_type in SP_ACTION_TYPES:
            success_probability = float(np.clip(0.82 + self._sp_rate_bonus(action_type), 0.05, 1.0))
            stamina_delta = -8.0
            produce_point_delta = 4.0
        elif action_type in LESSON_ACTION_TYPES:
            success_probability = 0.92
            stamina_delta = -5.0
            produce_point_delta = 2.0
        elif action_type == 'activity':
            success_probability = 0.95
            stamina_delta = 1.0
            produce_point_delta = 6.0
        elif action_type == 'business':
            success_probability = 0.96
            stamina_delta, produce_point_delta, _ = self._business_action_profile('')
        elif action_type == 'present':
            success_probability = 0.98
            produce_point_delta = 2.0
        elif action_type == 'school_class':
            success_probability = 0.95
            stamina_delta = -8.0
            produce_point_delta = 0.0
        elif action_type == 'outing':
            success_probability = 0.97
            stamina_delta = self.state['max_stamina'] * 0.35
            produce_point_delta = -12.0
        elif action_type == 'activity_supply':
            success_probability = 1.0
            stamina_delta = 0.0
            produce_point_delta = 18.0
        stat_deltas = (0.0, 0.0, 0.0)
        if action_type == 'school_class':
            stat_deltas = tuple(self._apply_growth_rates((36.0, 24.0, 24.0)))
        return ProduceActionCandidate(
            label=self._action_label(action_type),
            action_type=action_type,
            effect_types=synthetic_types,
            produce_effect_ids=self._effect_ids_for_types(synthetic_types),
            stamina_delta=stamina_delta,
            produce_point_delta=produce_point_delta,
            success_probability=success_probability,
            stat_deltas=stat_deltas,
        )

    def _effect_ids_for_types(self, effect_types: list[str]) -> list[str]:
        """按效果类型随机抽取对应的 ProduceEffect 行。"""

        effect_ids: list[str] = []
        for effect_type in effect_types:
            candidates = [row for row in self.produce_effects.rows if str(row.get('produceEffectType')) == effect_type]
            if not candidates:
                continue
            effect_ids.append(str(candidates[int(self.np_random.integers(0, len(candidates)))].get('id')))
        return effect_ids

    def _effect_types_for_ids(self, effect_ids: list[str]) -> list[str]:
        """把效果 id 列表反解为效果类型集合。"""

        effect_types: set[str] = set()
        for effect_id in effect_ids:
            effect_row = self.produce_effects.first(str(effect_id))
            if effect_row and effect_row.get('produceEffectType'):
                effect_types.add(str(effect_row['produceEffectType']))
        return sorted(effect_types)

    def _apply_effect_rows(self, effect_ids: list[str], source_action_type: str) -> None:
        """按 id 顺序应用一组 ProduceEffect。"""

        should_block_ability_chains = self._is_support_or_memory_ability_source(source_action_type)
        if should_block_ability_chains:
            self._ability_chain_guard_depth += 1
        try:
            for effect_id in effect_ids:
                effect_row = self.produce_effects.first(str(effect_id))
                if effect_row is not None:
                    self._apply_produce_effect(effect_row, source_action_type=source_action_type)
        finally:
            if should_block_ability_chains:
                self._ability_chain_guard_depth = max(self._ability_chain_guard_depth - 1, 0)

    def _apply_produce_effect(
        self,
        effect: dict[str, Any],
        source_action_type: str,
        *,
        source: str = 'produce',
        source_identity: str = '',
    ) -> None:
        """把单条 ProduceEffect 映射到当前培育状态。"""

        effect_type = str(effect.get('produceEffectType') or '')
        value = self._sample_effect_value(effect)
        event_action = source_action_type in EVENT_ACTION_TYPES

        # 直接增益会立刻写回当前培育状态；下面这类倍率增益则修改后续课程/事件，
        # 这样策略才能在新卡进入卡池时继续泛化。
        if effect_type == 'ProduceEffectType_VocalAddition':
            gain = value * (1.0 + self.state['vocal_growth'])
            if event_action:
                gain *= 1.0 + self.state['support_event_stat_bonus']
            self._gain_parameter('vocal', gain)
            return
        if effect_type == 'ProduceEffectType_DanceAddition':
            gain = value * (1.0 + self.state['dance_growth'])
            if event_action:
                gain *= 1.0 + self.state['support_event_stat_bonus']
            self._gain_parameter('dance', gain)
            return
        if effect_type == 'ProduceEffectType_VisualAddition':
            gain = value * (1.0 + self.state['visual_growth'])
            if event_action:
                gain *= 1.0 + self.state['support_event_stat_bonus']
            self._gain_parameter('visual', gain)
            return
        if effect_type == 'ProduceEffectType_VocalGrowthRateAddition':
            self.state['vocal_growth'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_DanceGrowthRateAddition':
            self.state['dance_growth'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_VisualGrowthRateAddition':
            self.state['visual_growth'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_MaxStaminaAddition':
            self.state['max_stamina'] += value
            self.state['stamina'] = min(self.state['stamina'] + value, self.state['max_stamina'])
            return
        if effect_type == 'ProduceEffectType_MaxStaminaReduceFix':
            self.state['max_stamina'] = max(self.state['max_stamina'] - value, 1.0)
            self.state['stamina'] = min(self.state['stamina'], self.state['max_stamina'])
            return
        if effect_type in {'ProduceEffectType_StaminaRecoverFix', 'ProduceEffectType_EventSchoolStaminaUp'}:
            self.state['stamina'] = min(
                self.state['max_stamina'],
                self.state['stamina'] + value * self._stamina_recovery_rate(source_action_type),
            )
            return
        if effect_type == 'ProduceEffectType_StaminaRecoverMultiple':
            self.state['stamina'] = min(
                self.state['max_stamina'],
                self.state['stamina'] + self.state['max_stamina'] * (value / 1000.0) * self._stamina_recovery_rate(source_action_type),
            )
            return
        if effect_type in {'ProduceEffectType_StaminaReduceFix', 'ProduceEffectType_EventSchoolStaminaDown'}:
            self.state['stamina'] = max(self.state['stamina'] - value, 0.0)
            return
        if effect_type in {'ProduceEffectType_ProducePointAddition', 'ProduceEffectType_ProducePointAdditionDisableTrigger'}:
            self.state['produce_points'] += value * self._produce_point_rate(source_action_type)
            return
        if effect_type == 'ProduceEffectType_ProducePointReduceFix':
            self.state['produce_points'] = max(self.state['produce_points'] - value, 0.0)
            return
        if effect_type == 'ProduceEffectType_VoteCountAddition':
            self.state['fan_votes'] += value * self._vote_rate(source_action_type)
            return
        if effect_type == 'ProduceEffectType_EventActivityProducePointUp':
            self.state['activity_produce_point_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_EventBusinessVoteCountUp':
            self.state['business_vote_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_LessonPresentProducePointUp':
            self.state['lesson_present_point_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_SupportCardEventProducePointAdditionValueUp':
            self.state['support_event_point_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_SupportCardEventParameterAdditionValueUp':
            self.state['support_event_stat_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_SupportCardEventStaminaRecoverUp':
            self.state['support_event_stamina_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_AuditionVoteCountUp':
            self.state['audition_vote_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_AuditionParameterBonusMultiple':
            self.state['audition_parameter_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_AuditionNpcEnhance':
            self.state['audition_difficulty_bonus'] += value / 1000.0
            return
        if effect_type in {'ProduceEffectType_AuditionNpcWeaken', '128'}:
            # 线上主数据既有正式枚举，也残留过直接落原始值 `128` 的脏数据，两者都表示削弱对手分数。
            self.state['audition_difficulty_bonus'] -= value / 1000.0
            return
        if effect_type == 'ProduceEffectType_ExamTurnDown':
            self.state['audition_turn_modifier'] -= value
            return
        if effect_type == 'ProduceEffectType_BeforeAuditionRefreshStaminaDown':
            self.state['before_audition_refresh_penalty'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_BeforeAuditionRefreshStaminaUp':
            self.state['before_audition_refresh_penalty'] -= value / 1000.0
            return
        if effect_type == 'ProduceEffectType_LessonSpChangeRatePermilAddition':
            self.state['generic_sp_rate_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_LessonVocalSpChangeRatePermilAddition':
            self.state['vocal_sp_rate_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_LessonDanceSpChangeRatePermilAddition':
            self.state['dance_sp_rate_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_LessonVisualSpChangeRatePermilAddition':
            self.state['visual_sp_rate_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_LessonPresentProduceCardRewardCountUp':
            self.state['reward_card_count_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_IdolCardProduceCardCustomizeEnable':
            self.state['customize_slots'] += max(value / 1000.0, 1.0)
            return
        if effect_type == 'ProduceEffectType_ProduceCardExcludeCountUp':
            self.state['exclude_count_bonus'] += max(value / 1000.0, 1.0)
            return
        if effect_type in {'ProduceEffectType_ProduceCardSelectRerollCountUp', 'ProduceEffectType_ShopRerollCountUp'}:
            self.state['reroll_count_bonus'] += max(value / 1000.0, 1.0)
            return
        if effect_type in {
            'ProduceEffectType_ShopPriceDiscountMultiple',
            'ProduceEffectType_ShopPriceUpMultiple',
            'ProduceEffectType_ShopProduceCardDeletePriceDiscountMultiple',
            'ProduceEffectType_ShopProduceCardPriceDiscountMultiple',
            'ProduceEffectType_ShopProduceCardUpgradePriceDiscountMultiple',
            'ProduceEffectType_ShopProduceDrinkPriceDiscountMultiple',
        }:
            direction = -1.0 if 'Discount' in effect_type else 1.0
            self.state['shop_discount'] += direction * (value / 1000.0)
            return
        if effect_type == 'ProduceEffectType_SupportCardProduceCardUpgradeProbabilityUp':
            self.state['card_upgrade_probability_bonus'] += value / 1000.0
            return
        if effect_type == 'ProduceEffectType_HighScoreGoldAddition':
            self.state['gold_bonus'] += value
            return
        if effect_type == 'ProduceEffectType_ProduceCardUpgrade':
            self._upgrade_matching_cards(
                str(effect.get('produceCardSearchId') or ''),
                int(max(effect.get('pickCountMin') or 1, 1)),
                source_action_type=source_action_type,
            )
            return
        if effect_type == 'ProduceEffectType_ProduceCardDelete':
            self._delete_matching_cards(
                str(effect.get('produceCardSearchId') or ''),
                int(max(effect.get('pickCountMin') or 1, 1)),
            )
            return
        if effect_type == 'ProduceEffectType_ProduceCardDuplicate':
            self._duplicate_matching_cards(
                str(effect.get('produceCardSearchId') or ''),
                int(max(effect.get('pickCountMin') or 1, 1)),
            )
            return
        if effect_type in {'ProduceEffectType_ProduceCardChange', 'ProduceEffectType_ProduceCardChangeUpgrade'}:
            self._replace_matching_cards(
                str(effect.get('produceCardSearchId') or ''),
                upgraded=effect_type.endswith('Upgrade'),
                source_action_type=source_action_type,
            )
            return
        if effect_type in {'ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceRewardSet'}:
            self._grant_rewards(effect, source_action_type=source_action_type)
            return
        if effect_type in {'ProduceEffectType_ExamStatusEnchant', 'ProduceEffectType_ExamPermanentLessonStatusEnchant', 'ProduceEffectType_ExamPermanentAuditionStatusEnchant'}:
            enchant_id = str(effect.get('produceExamStatusEnchantId') or '')
            if enchant_id:
                self._append_exam_status_enchant(
                    enchant_id,
                    source='produce_item' if source == 'produce_item' else 'produce',
                    source_identity=source_identity,
                )
            return
    def _produce_point_rate(self, source_action_type: str) -> float:
        """计算当前动作来源对应的制作点倍率。"""

        rate = 1.0
        if source_action_type == 'activity':
            rate += self.state['activity_produce_point_bonus']
        if source_action_type in EVENT_ACTION_TYPES:
            rate += self.state['support_event_point_bonus']
        if source_action_type == 'present' or _is_lesson_action(source_action_type):
            rate += self.state['lesson_present_point_bonus']
        return max(rate, 0.0)

    def _vote_rate(self, source_action_type: str) -> float:
        """计算营业类动作的粉丝票数倍率。"""

        rate = 1.0
        if source_action_type == 'business':
            rate += self.state['business_vote_bonus']
        return max(rate, 0.0)

    def _stamina_recovery_rate(self, source_action_type: str) -> float:
        """计算体力回复类效果的倍率。"""

        rate = 1.0
        if source_action_type in EVENT_ACTION_TYPES:
            rate += self.state['support_event_stamina_bonus']
        return max(rate, 0.0)

    def _sp_rate_bonus(self, action_type: str) -> float:
        """返回对应 SP 课程的额外成功率加成。"""

        bonus = self.state['generic_sp_rate_bonus']
        if action_type in {'lesson_vocal_sp', 'self_lesson_vocal_sp'}:
            bonus += self.state['vocal_sp_rate_bonus']
        elif action_type in {'lesson_dance_sp', 'self_lesson_dance_sp'}:
            bonus += self.state['dance_sp_rate_bonus']
        elif action_type in {'lesson_visual_sp', 'self_lesson_visual_sp'}:
            bonus += self.state['visual_sp_rate_bonus']
        return bonus

    def _grant_rewards(self, effect: dict[str, Any], source_action_type: str) -> None:
        """处理课程或事件奖励掉落的卡牌和饮料。"""

        rewards = effect.get('produceRewards', []) or []
        if rewards:
            for reward in rewards:
                self._grant_resource(str(reward.get('resourceType') or ''), str(reward.get('resourceId') or ''), int(reward.get('resourceLevel') or 0))
            self._trim_drinks()
            return
        resource_type = str(effect.get('produceResourceType') or '')
        count = int(max(effect.get('pickCountMax') or effect.get('pickCountMin') or 1, 1))
        if resource_type == 'ProduceResourceType_ProduceCard' and (source_action_type == 'present' or _is_lesson_action(source_action_type)):
            count += int(round(self.state['reward_card_count_bonus']))
        for _ in range(max(count, 0)):
            if resource_type == 'ProduceResourceType_ProduceDrink':
                candidates = self.repository.build_drink_inventory(
                    self.scenario,
                    max_items=self.scenario.drink_limit,
                    rng=self.np_random,
                    plan_type=self.idol_loadout.stat_profile.plan_type if self.idol_loadout is not None else None,
                )
                if candidates:
                    drink_row = dict(candidates[int(self.np_random.integers(0, len(candidates)))])
                    self.drinks.append(drink_row)
                    self._dispatch_produce_item_phase('ProducePhaseType_GetProduceDrink')
            elif resource_type == 'ProduceResourceType_ProduceCard':
                candidates = self._selection_card_pool()
                if candidates:
                    sampled = sample_card_from_weighted_pool(candidates, self.np_random)
                    if sampled is None:
                        continue
                    card_row = self._sample_capped_card_variant(str(sampled.get('id') or ''), max_upgrade_count=1) or dict(sampled)
                    if self.np_random.random() < self.state['card_upgrade_probability_bonus']:
                        upgraded_row = self._lookup_card_row(str(card_row.get('id')), int(card_row.get('upgradeCount') or 0) + 1)
                        if upgraded_row is not None and int(upgraded_row.get('upgradeCount') or 0) <= 1:
                            card_row = dict(upgraded_row)
                    self.deck.append(card_row)
                    if str(card_row.get('rarity') or '') == 'ProduceCardRarity_Legend':
                        card_id = str(card_row.get('id') or '')
                        if card_id:
                            self.legend_seen_card_ids.add(card_id)
                    self._dispatch_produce_item_phase('ProducePhaseType_GetProduceCard', card=card_row)
        self._trim_drinks()

    def _grant_resource(self, resource_type: str, resource_id: str, resource_level: int) -> None:
        """把单个资源奖励写回卡组、饮料或支援技能列表。"""

        if resource_type == 'ProduceResourceType_ProduceCard':
            card_row = resolve_produce_card_row(
                self.repository,
                resource_id,
                loadout=self.idol_loadout,
                upgrade_count=resource_level,
            )
            if card_row is not None:
                resolved_card = dict(card_row)
                self.deck.append(resolved_card)
                if str(resolved_card.get('rarity') or '') == 'ProduceCardRarity_Legend':
                    card_id = str(resolved_card.get('id') or '')
                    if card_id:
                        self.legend_seen_card_ids.add(card_id)
                self._dispatch_produce_item_phase('ProducePhaseType_GetProduceCard', card=resolved_card)
        elif resource_type == 'ProduceResourceType_ProduceDrink':
            drink_row = self.repository.produce_drinks.first(resource_id)
            if drink_row is not None:
                self.drinks.append(dict(drink_row))
                self._dispatch_produce_item_phase('ProducePhaseType_GetProduceDrink')
        elif resource_type == 'ProduceResourceType_ProduceItem':
            self._register_produce_item(resource_id, source='reward')
            self._dispatch_produce_item_phase('ProducePhaseType_GetProduceItem')
        elif resource_type == 'ProduceResourceType_ProduceSkill':
            self.support_skills.append(resource_id)

    def _matching_deck_indices(self, search_id: str) -> list[int]:
        """查找当前牌组里符合搜索条件的卡牌下标。"""

        search = self.card_searches.first(search_id)
        if not search:
            return list(range(len(self.deck)))
        indices: list[int] = []
        for index, card in enumerate(self.deck):
            if self._deck_card_matches(card, search):
                indices.append(index)
        return indices

    def _deck_card_matches(self, card: dict[str, Any], search: dict[str, Any]) -> bool:
        """判断牌组中的卡是否命中 ProduceCardSearch 条件。"""

        return self.produce_item_interpreter.card_matches_search(card, str(search.get('id') or ''))

    def _upgrade_matching_cards(self, search_id: str, count: int, *, source_action_type: str = '') -> None:
        """升级若干张符合条件的卡。"""

        indices = self._matching_deck_indices(search_id)
        self.np_random.shuffle(indices)
        revert_changes: list[dict[str, Any]] = []
        for index in indices[:count]:
            card = self.deck[index]
            if str(card.get('rarity') or '') == 'ProduceCardRarity_Legend':
                continue
            upgraded = self._lookup_card_row(str(card.get('id')), int(card.get('upgradeCount') or 0) + 1)
            if upgraded is not None:
                revert_changes.append({'index': index, 'original_card': dict(card)})
                upgraded_row = dict(upgraded)
                self.deck[index] = upgraded_row
                self._dispatch_produce_item_phase('ProducePhaseType_UpgradeProduceCard', card=upgraded_row)
        # 支援カードイベント起因のカード強化は戻す対象
        if revert_changes and source_action_type in EVENT_ACTION_TYPES:
            self.pending_revert_info = {'type': 'upgrade', 'changes': revert_changes}

    def _delete_matching_cards(self, search_id: str, count: int) -> None:
        """删除若干张符合条件的卡。"""

        indices = self._matching_deck_indices(search_id)
        self.np_random.shuffle(indices)
        for index in sorted(indices[:count], reverse=True):
            deleted_card = dict(self.deck[index])
            self.deck.pop(index)
            self._dispatch_produce_item_phase('ProducePhaseType_DeleteProduceCard', card=deleted_card)

    def _duplicate_matching_cards(self, search_id: str, count: int) -> None:
        """复制若干张符合条件的卡。"""

        indices = self._matching_deck_indices(search_id)
        self.np_random.shuffle(indices)
        for index in indices[:count]:
            duplicated = dict(self.deck[index])
            self.deck.append(duplicated)
            if str(duplicated.get('rarity') or '') == 'ProduceCardRarity_Legend':
                card_id = str(duplicated.get('id') or '')
                if card_id:
                    self.legend_seen_card_ids.add(card_id)
            self._dispatch_produce_item_phase('ProducePhaseType_GetProduceCard', card=duplicated)

    def _replace_matching_cards(self, search_id: str, upgraded: bool, *, source_action_type: str = '') -> None:
        """把命中的卡替换为当前流派候选池中的新卡。"""

        indices = self._matching_deck_indices(search_id)
        if not indices:
            return
        index = int(self.np_random.choice(indices))
        candidates = self._selection_card_pool()
        if not candidates:
            return
        sampled = sample_card_from_weighted_pool(candidates, self.np_random)
        if sampled is None:
            return
        replacement = self._sample_capped_card_variant(str(sampled.get('id') or ''), max_upgrade_count=1) or dict(sampled)
        if self._has_legend_card() and str(replacement.get('rarity') or '') == 'ProduceCardRarity_Legend':
            non_legend_candidates = [row for row in candidates if str(row.get('rarity') or '') != 'ProduceCardRarity_Legend']
            if not non_legend_candidates:
                return
            sampled = sample_card_from_weighted_pool(non_legend_candidates, self.np_random)
            if sampled is None:
                return
            replacement = self._sample_capped_card_variant(str(sampled.get('id') or ''), max_upgrade_count=1) or dict(sampled)
        if upgraded:
            upgraded_row = self._lookup_card_row(str(replacement.get('id')), int(replacement.get('upgradeCount') or 0) + 1)
            if upgraded_row is not None and int(upgraded_row.get('upgradeCount') or 0) <= 1:
                replacement = dict(upgraded_row)
        original_card = dict(self.deck[index])
        self.deck[index] = replacement
        self._dispatch_produce_item_phase('ProducePhaseType_ChangeProduceCard', card=replacement)
        # 支援カードイベント起因のカード置換は戻す対象
        if source_action_type in EVENT_ACTION_TYPES:
            self.pending_revert_info = {'type': 'replace', 'changes': [{'index': index, 'original_card': original_card}]}

    def _lookup_card_row(self, card_id: str, upgrade_count: int) -> dict[str, Any] | None:
        """按卡 id 和强化次数查找主数据行。"""

        return self.repository.card_row_by_upgrade(card_id, upgrade_count, fallback_to_canonical=True)

    def _sample_effect_value(self, effect: dict[str, Any]) -> float:
        """从主数据字段中采样一条效果数值。"""

        minimum = float(effect.get('effectValueMin') or 0)
        maximum = float(effect.get('effectValueMax') or minimum)
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        if minimum == maximum:
            return minimum
        return float(self.np_random.uniform(minimum, maximum))

    def _action_label(self, action_type: str) -> str:
        """把内部动作类型转换成展示文案。"""

        labels = {
            'lesson_vocal_normal': '声乐课',
            'lesson_dance_normal': '舞蹈课',
            'lesson_visual_normal': '形象课',
            'lesson_vocal_sp': 'SP声乐课',
            'lesson_dance_sp': 'SP舞蹈课',
            'lesson_visual_sp': 'SP形象课',
            'lesson_vocal_hard': '追击声乐课',
            'lesson_dance_hard': '追击舞蹈课',
            'lesson_visual_hard': '追击形象课',
            'self_lesson_vocal_normal': '自主声乐课',
            'self_lesson_vocal_sp': '自主SP声乐课',
            'self_lesson_dance_normal': '自主舞蹈课',
            'self_lesson_dance_sp': '自主SP舞蹈课',
            'self_lesson_visual_normal': '自主形象课',
            'self_lesson_visual_sp': '自主SP形象课',
            'activity': '活动',
            'business': '营业',
            'present': '差入/事件',
            'school_class': '授业',
            'outing': '外出',
            'activity_supply': '活动支给',
            'refresh': '休息',
            'pre_audition_continue': '继续前进',
        }
        if _is_shop_card_action(action_type):
            return f'购买技能卡槽位{_shop_slot_index(action_type) + 1}'
        if _is_shop_drink_action(action_type):
            return f'购买P饮料槽位{_shop_slot_index(action_type) + 1}'
        if _is_shop_upgrade_action(action_type):
            return f'强化技能卡槽位{_shop_slot_index(action_type) + 1}'
        if _is_shop_delete_action(action_type):
            return f'删除技能卡槽位{_shop_slot_index(action_type) + 1}'
        return labels.get(action_type, action_type)

    def _trim_drinks(self) -> None:
        """按场景上限裁剪饮料栏。"""

        if len(self.drinks) <= self.scenario.drink_limit:
            return
        self.drinks.sort(
            key=lambda row: (len(self.repository.drink_exam_effect_types(row)), str(row.get('rarity') or '')),
            reverse=True,
        )
        self.drinks = self.drinks[: self.scenario.drink_limit]

    def _refresh_quality_scores(self) -> None:
        """重新估算当前卡组和饮料质量，用于奖励与观测。"""

        card_scores = [self.repository.card_play_priors.get(str(card.get('id')), 0.0) for card in self.deck]
        drink_scores = [len(self.repository.drink_exam_effect_types(drink)) for drink in self.drinks]
        enchant_bonus = 0.2 * len(self.exam_status_enchant_ids)
        self.state['deck_quality'] = (float(np.mean(card_scores)) / 100.0 if card_scores else 0.0) + enchant_bonus
        self.state['drink_quality'] = float(np.mean(drink_scores)) if drink_scores else 0.0

    def _audition_start_stamina(self) -> float:
        """按主数据的试验前回复量规则，计算考试开场体力。"""

        max_stamina = max(float(self.state.get('max_stamina') or 0.0), 1.0)
        current_stamina = float(np.clip(self.state.get('stamina') or 0.0, 0.0, max_stamina))
        recovery_permille = float(self.produce_setting.get('beforeAuditionRefreshStaminaRecoveryPermil') or 0.0)
        recovery_multiple = max(0.0, 1.0 - float(self.state.get('before_audition_refresh_penalty') or 0.0))
        recovered = current_stamina + max_stamina * (recovery_permille / 1000.0) * recovery_multiple
        return float(min(recovered, max_stamina))

    def _challenge_lesson_perfect_bonus_ratio(self) -> float:
        """估算 challenge P 道具对 lesson PERFECT 上限的提升比例。"""

        if self.idol_loadout is None or self.scenario.produce_id not in {'produce-003', 'produce-006'}:
            return 0.0
        bonus_ratio = 0.0
        for item_id in getattr(self.idol_loadout, 'extra_produce_item_ids', ()):
            item_row = self.repository.produce_items.first(str(item_id)) or {}
            for item_effect_id in item_row.get('produceItemEffectIds', []) or []:
                item_effect_row = self.repository.load_table('ProduceItemEffect').first(str(item_effect_id))
                if not item_effect_row:
                    continue
                produce_effect_id = str(item_effect_row.get('produceEffectId') or '')
                effect_row = self.repository.produce_effects.first(produce_effect_id) if produce_effect_id else None
                if effect_row is None or str(effect_row.get('produceEffectType') or '') != 'ProduceEffectType_ExamPermanentLessonStatusEnchant':
                    continue
                enchant_id = str(effect_row.get('produceExamStatusEnchantId') or '')
                enchant_row = self.repository.exam_status_enchants.first(enchant_id) or {}
                for exam_effect_id in enchant_row.get('produceExamEffectIds', []) or []:
                    exam_effect = self.repository.exam_effects.first(str(exam_effect_id))
                    if exam_effect is None:
                        continue
                    effect_type = str(exam_effect.get('effectType') or '')
                    if effect_type == 'ProduceExamEffectType_ExamLessonValueMultiple':
                        bonus_ratio += float(exam_effect.get('effectValue1') or 0.0) / 1000.0
                    elif effect_type == 'ProduceExamEffectType_ExamAddGrowEffect':
                        for grow_effect_id in exam_effect.get('produceCardGrowEffectIds', []) or []:
                            grow_effect = self.repository.load_table('ProduceCardGrowEffect').first(str(grow_effect_id)) or {}
                            if str(grow_effect.get('effectType') or '') == 'ProduceCardGrowEffectType_LessonAdd':
                                bonus_ratio += float(grow_effect.get('value') or 0.0) / 100.0
        return max(bonus_ratio, 0.0)

    def _challenge_audition_npc_bonus_ratio(self) -> float:
        """估算 challenge P 道具带来的 audition 对手强度修正。"""

        if self.idol_loadout is None:
            return 0.0
        bonus_ratio = 0.0
        for item_id in getattr(self.idol_loadout, 'extra_produce_item_ids', ()):
            item_row = self.repository.produce_items.first(str(item_id)) or {}
            for item_effect_id in item_row.get('produceItemEffectIds', []) or []:
                item_effect_row = self.repository.load_table('ProduceItemEffect').first(str(item_effect_id))
                if not item_effect_row:
                    continue
                produce_effect_id = str(item_effect_row.get('produceEffectId') or '')
                effect_row = self.repository.produce_effects.first(produce_effect_id) if produce_effect_id else None
                if effect_row is None:
                    continue
                if str(effect_row.get('produceEffectType') or '') == 'ProduceEffectType_AuditionNpcEnhance':
                    bonus_ratio += float(effect_row.get('effectValueMin') or 0.0) / 1000.0
        return max(bonus_ratio, 0.0)

    def _choose_exam_action(self, runtime: ExamRuntime):
        """用启发式从考试运行时里挑选一个动作。"""

        actions = runtime.legal_actions()
        if not actions:
            return None
        remaining_turns = max(runtime.max_turns - runtime.turn + 1, 1)
        best_action = actions[-1]
        best_score = float('-inf')
        playable_card_count = sum(1 for action in actions if action.kind == 'card')

        for action in actions:
            # 这个兜底控制器只使用效果类型和资源成本等结构先验，不依赖卡名。
            score = 0.0
            if action.kind == 'card':
                card = next((item for item in runtime.hand if item.uid == int(action.payload['uid'])), None)
                if card is None:
                    continue
                effect_types = self.repository.card_exam_effect_types(card.base_card)
                for effect_id in card.transient_effect_ids:
                    effect_row = self.repository.exam_effect_map.get(str(effect_id))
                    if effect_row and effect_row.get('effectType'):
                        effect_types.append(str(effect_row['effectType']))
                prior = self.repository.card_play_priors.get(str(card.card_id), 0.0) / 100.0
                effect_prior = sum(self.repository.exam_effect_priors.get((effect_type, remaining_turns), 0.0) for effect_type in effect_types) / max(len(effect_types), 1)
                score += prior + effect_prior / 100.0
                score -= float(card.base_card.get('stamina') or 0) * 0.03
                score -= float(card.base_card.get('forceStamina') or 0) * 0.05
                score += card.play_count_bonus * 0.1
            elif action.kind == 'drink':
                drink = runtime.drinks[int(action.payload['index'])]
                effect_types = self.repository.drink_exam_effect_types(drink)
                effect_prior = sum(self.repository.exam_effect_priors.get((effect_type, remaining_turns), 0.0) for effect_type in effect_types) / max(len(effect_types), 1)
                score += effect_prior / 100.0
                if runtime.stamina < runtime.max_stamina * 0.45:
                    score += 0.15
            elif action.kind == 'end_turn':
                score -= 0.1
                if playable_card_count == 0:
                    score += 0.25
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _assist_mode_enabled(self) -> bool:
        """仅在 NIA Pro 中启用 Assist Mode。"""

        return bool(
            self.idol_loadout is not None
            and self.idol_loadout.assist_mode
            and self.scenario.scenario_id == 'produce-004'
        )

    def _simulate_rival_scores(self, runtime: ExamRuntime, effective_score: float) -> tuple[list[float], list[dict[str, float]], int, float]:
        """根据主数据 NPC 组估算本场考试对手分数、阶段曲线和最终排名。"""

        selected_row = runtime.selected_battle_row or {}
        npc_group_id = str(selected_row.get('produceExamBattleNpcGroupId') or '')
        npc_rows = self.repository.load_table('ProduceExamBattleNpcGroup').all(npc_group_id) if npc_group_id else []
        rival_multiplier = max(0.0, 1.0 + float(self.state.get('audition_difficulty_bonus') or 0.0))
        if self._assist_mode_enabled():
            rival_multiplier *= 0.85
        rival_multiplier = max(rival_multiplier, 0.0)
        assist_reduction = 0.15 if self._assist_mode_enabled() else 0.0
        rival_scores: list[float] = []
        rival_phase_breakdowns: list[dict[str, float]] = []
        for row in npc_rows:
            score_min = max(float(row.get('scoreMin') or 0.0), 0.0)
            score_max = max(float(row.get('scoreMax') or score_min), score_min)
            if np.isclose(score_min, score_max):
                sampled = score_min
            else:
                midpoint = (score_min + score_max) * 0.5
                sampled = float(self.np_random.triangular(score_min, midpoint, score_max))
            final_score = sampled * rival_multiplier
            phase_weights = np.array(
                [
                    max(float(row.get('opScorePermil') or 0.0), 0.0),
                    max(float(row.get('midScorePermil') or 0.0), 0.0),
                    max(float(row.get('edScorePermil') or 0.0), 0.0),
                ],
                dtype=np.float64,
            )
            if phase_weights.sum() <= 0:
                phase_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            phase_weights = phase_weights / phase_weights.sum()
            phase_scores = final_score * phase_weights
            rival_scores.append(final_score)
            rival_phase_breakdowns.append(
                {
                    'op': float(phase_scores[0]),
                    'mid': float(phase_scores[1]),
                    'ed': float(phase_scores[2]),
                    'final': float(final_score),
                }
            )
        rank = 1 + sum(score > effective_score for score in rival_scores)
        return rival_scores, rival_phase_breakdowns, rank, rival_multiplier

    def _run_audition(self, stage_type: str, *, include_pre_audition_phases: bool = True, apply_outcome: bool = True) -> tuple[float, dict[str, Any]]:
        """把当前培育构筑带入考试运行时，返回考试奖励与摘要。"""

        if include_pre_audition_phases:
            for phase_type in self._pre_audition_item_phases():
                self._dispatch_produce_item_phase(phase_type, stage_type=stage_type)
        self._dispatch_produce_item_phase('ProducePhaseType_EndBeforeAuditionRefresh')
        for phase_type in self._stage_trigger_phases(stage_type):
            self._dispatch_produce_item_phase(phase_type)
        exam_loadout = self.idol_loadout
        if exam_loadout is not None:
            exam_loadout = replace(
                exam_loadout,
                produce_item_id='',
                exam_status_enchant_ids=(),
                exam_status_enchant_specs=(),
            )
        runtime = ExamRuntime(
            self.repository,
            self.scenario,
            stage_type=stage_type,
            seed=int(self.np_random.integers(0, 2**31 - 1)),
            deck=list(self.deck),
            drinks=list(self.drinks),
            initial_status_enchant_ids=list(self.exam_status_enchant_ids),
            initial_status_enchants=list(self.exam_status_enchant_specs),
            loadout=exam_loadout,
            starting_stamina=self._audition_start_stamina(),
            exam_score_bonus_multiplier=(self.idol_loadout.exam_score_bonus_multiplier if self.idol_loadout else 1.0) * (1.0 + self.state['audition_parameter_bonus']),
            fan_votes=float(self.state.get('fan_votes') or 0.0),
            audition_row_id=(
                self._resolve_selected_audition_row_id(stage_type)
                or default_audition_row_selector(
                    self.repository,
                    self.scenario,
                    stage_type=stage_type,
                    loadout=exam_loadout,
                    fan_votes=float(self.state.get('fan_votes') or 0.0),
                )
            ),
        )
        if runtime.battle_kind == 'lesson' and runtime.lesson_perfect_value is not None:
            runtime.lesson_perfect_value *= 1.0 + self._challenge_lesson_perfect_bonus_ratio()
        runtime.reset()
        runtime.max_turns = max(1, runtime.max_turns + int(round(self.state['audition_turn_modifier'])))
        for _ in range(256):
            action = self._choose_exam_action(runtime)
            if action is None:
                break
            runtime.step(action)
            if runtime.terminated:
                break
        self._dispatch_produce_item_phase('ProducePhaseType_EndAudition')
        effective_score = runtime.score
        profile = dict(runtime.profile)
        rank_threshold = max(int(profile.get('rank_threshold') or 3), 1)
        rival_scores, rival_phase_breakdowns, rank, rival_multiplier = self._simulate_rival_scores(runtime, effective_score)
        force_end_score = float(profile.get('force_end_score') or 0.0)
        cleared = (force_end_score > 0 and runtime.score >= force_end_score) or rank <= rank_threshold
        sorted_rivals = sorted(rival_scores, reverse=True)
        threshold_index = min(max(rank_threshold - 1, 0), max(len(sorted_rivals) - 1, 0))
        threshold_rival_score = sorted_rivals[threshold_index] if sorted_rivals else float(profile.get('base_score') or 0.0)
        target_score = max(threshold_rival_score, 1.0)
        margin = (effective_score - target_score) / max(target_score, 1.0)
        reward = (1.0 + min((rank_threshold - rank) * 0.15 + margin, 0.8)) if cleared else (-1.0 + max(margin, -0.8))
        vote_gain = runtime.estimate_fan_vote_gain(effective_score) * (1.0 + self.state['audition_vote_bonus']) if cleared else 0.0
        deck_quality_gain = 0.0
        drink_quality_gain = 0.0
        # 仍保留当前 runtime.score 作为真实考试表现主分，后续的资源价值在 produce shaping 里体现，而不是在此处重复加分。
        # NIA 试镜合格后按 V/D/V 回合得分返还对应参数
        nia_param_gains: dict[str, float] = {}
        if cleared and self.scenario.route_type == 'nia':
            parameter_baseline = float(profile.get('parameter_baseline') or 180.0)
            base_score = float(profile.get('base_score') or 1000.0)
            score_per_color = dict(runtime.score_per_color)
            total_phase_score = sum(score_per_color.values())
            if total_phase_score > 0:
                for color, phase_score in score_per_color.items():
                    # 参数增益 = 该颜色得分比例 × parameterBaseLine × (实际得分 / baseScore)
                    score_ratio = min(runtime.score / max(base_score, 1.0), 2.0)
                    gain = (phase_score / total_phase_score) * parameter_baseline * score_ratio
                    nia_param_gains[color] = gain
        if apply_outcome:
            self.state['fan_votes'] += max(vote_gain, 0.0)
            self.state['deck_quality'] += deck_quality_gain
            self.state['drink_quality'] += drink_quality_gain
            self.state['last_exam_score'] = effective_score
            for color, gain in nia_param_gains.items():
                self._gain_parameter(color, gain)
        return reward, {
            'stage_type': stage_type,
            'audition_row_id': str((runtime.selected_battle_row or {}).get('id') or ''),
            'audition_row_number': int((runtime.selected_battle_row or {}).get('number') or 0),
            'audition_selected_label': self._selected_audition_label(stage_type),
            'finale_route_selected': self._current_finale_route_selected() if apply_outcome is False else (int((runtime.selected_battle_row or {}).get('number') or 0) == 4 and stage_type == str(self.scenario.audition_sequence[-1] or '')),
            'exam_score': runtime.score,
            'parameter_bonus': sum(nia_param_gains.values()),
            'parameter_bonus_multiplier': runtime.score_bonus_multiplier,
            'effective_score': effective_score,
            'target_score': target_score,
            'cleared': cleared,
            'rank': rank,
            'rank_threshold': rank_threshold,
            'rival_scores': [float(score) for score in rival_scores],
            'rival_phase_breakdowns': rival_phase_breakdowns,
            'threshold_rival_score': float(threshold_rival_score),
            'rival_score_multiplier': rival_multiplier,
            'assist_mode': self._assist_mode_enabled(),
            'assist_reduction_ratio': 0.15 if self._assist_mode_enabled() else 0.0,
            'fan_votes': self.state['fan_votes'] + (max(vote_gain, 0.0) if apply_outcome else 0.0),
            'fan_vote_gain': max(vote_gain, 0.0),
            'fan_vote_requirement': float(profile.get('fan_vote_requirement') or 0.0),
            'fan_vote_baseline': float(profile.get('fan_vote_baseline') or 0.0),
            'turns': runtime.turn,
            'deck_quality_gain': deck_quality_gain,
            'drink_quality_gain': drink_quality_gain,
            'challenge_lesson_perfect_bonus_ratio': float(self.state.get('challenge_lesson_perfect_bonus_ratio') or 0.0),
            'challenge_audition_npc_bonus_ratio': float(self.state.get('challenge_audition_npc_bonus_ratio') or 0.0),
        }
