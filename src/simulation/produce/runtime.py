"""培育阶段运行时，按主数据驱动课程、事件和阶段考试流程。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
import logging
from typing import Any, Protocol

import numpy as np

from ...constants.game.action_types import (
    ACTION_ACTIVITY,
    ACTION_ACTIVITY_SUPPLY,
    ACTION_BUSINESS,
    ACTION_OUTING,
    ACTION_PRE_AUDITION_CONTINUE,
    ACTION_PRESENT,
    ACTION_REFRESH,
    ACTION_SCHOOL_CLASS,
)
from ...idol_config import (
    build_initial_exam_deck,
    build_weighted_card_pool,
    list_trainable_idol_card_ids,
    resolve_produce_card_row,
    sample_card_from_weighted_pool,
)
from ...loadout import IdolLoadout
from ...produce_score import calculate_hajime_produce_rating, calculate_nia_produce_rating, resolve_nia_idol_id_from_audition_difficulty_id
from ...repository.master_data import MasterDataRepository, ScenarioSpec
from ...training.reward_config import ProduceRewardConfig, build_produce_reward_config
from ..exam.runtime import ExamActionCandidate, ExamRuntime, default_audition_row_selector
from .items import ActiveProduceItem, ProduceItemInterpreter, RuntimeExamStatusEnchantSpec


logger = logging.getLogger(__name__)


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
FAILED_ROUTE_SCORE_SCALE = 0.25
PARAM_TARGET_STAGE_BASE_RATIO = 0.45
PARAM_TARGET_STAGE_PROGRESS_RATIO = 0.35
NIA_PARAM_TARGET_STAGE_BASE_RATIO = 0.55
NIA_PARAM_TARGET_STAGE_PROGRESS_RATIO = 0.45
PARAM_PROGRESS_WEIGHTED_WEIGHT = 0.58
PARAM_PROGRESS_COVERAGE_WEIGHT = 0.30
PARAM_PROGRESS_WEAKEST_WEIGHT = 0.12
PARAM_PROGRESS_OVERSHOOT_WEIGHT = 0.30
NIA_PARAM_PROGRESS_WEIGHTED_WEIGHT = 0.35
NIA_PARAM_PROGRESS_COVERAGE_WEIGHT = 0.35
NIA_PARAM_PROGRESS_WEAKEST_WEIGHT = 0.30
NIA_PARAM_PROGRESS_OVERSHOOT_WEIGHT = 0.20
NIA_PARAM_PROGRESS_IMBALANCE_PENALTY_WEIGHT = 0.45
NIA_FAN_VALUE_PARAM_FLOOR = 0.05
NIA_FAN_VALUE_PARAM_SCALE = 0.95
STATE_DELTA_PARAM_WEIGHT = 0.140
STATE_DELTA_FAN_WEIGHT = 0.080
STATE_DELTA_RESOURCE_WEIGHT = 0.040
STATE_DELTA_STAMINA_RECOVERY_WEIGHT = 0.020
STATE_DELTA_LOW_STAMINA_PENALTY_WEIGHT = 0.050
STATE_DELTA_WASTED_RECOVERY_PENALTY_WEIGHT = 0.080
STATE_DELTA_NO_PROGRESS_PENALTY = 0.120
STATE_DELTA_HIGH_STAMINA_IDLE_PENALTY = 0.080
STATE_DELTA_PARAM_PRESSURE_PENALTY_WEIGHT = 0.090
STATE_DELTA_PARAM_PRESSURE_RESOURCE_DISCOUNT = 0.70
STATE_DELTA_FAN_PRESSURE_DISCOUNT = 0.85
STATE_DELTA_NIA_PARAM_PRESSURE_MULTIPLIER = 1.80
STATE_DELTA_NIA_FAN_ONLY_PENALTY_WEIGHT = 0.360
STATE_DELTA_NIA_FAN_ONLY_PROGRESS_THRESHOLD = 0.004
STATE_DELTA_NIA_FAN_SURPLUS_PENALTY_WEIGHT = 0.260
STATE_DELTA_NIA_LOPSIDED_PENALTY_MULTIPLIER = 2.20
STATE_DELTA_NIA_STALLED_WEAKEST_PENALTY = 0.140
STATE_DELTA_NIA_IDLE_EVENT_REPEAT_PENALTY = 0.220
STATE_DELTA_COVERAGE_GAIN_WEIGHT = 0.120
STATE_DELTA_WEAKEST_GAIN_WEIGHT = 0.180
STATE_DELTA_LOPSIDED_PARAM_PENALTY = 0.080
STATE_DELTA_STAMINA_READINESS_GAIN_WEIGHT = 0.040
STATE_DELTA_STAMINA_READINESS_DROP_WEIGHT = 0.120
STATE_DELTA_REWARD_CLIP = 0.180
FIRST_STAR_REFRESH_HIGH_STAMINA_LOCK_RATIO = 0.35
STAGE_REPEAT_RATIO_SCALE = 4.0
STAGE_REPEAT_UNHELPFUL_PENALTY = 0.180
NIA_STAGE_REPEAT_UNHELPFUL_PENALTY = 0.380
STAGE_REPEAT_WEAKEST_GAIN_THRESHOLD = 0.0015
STAGE_REPEAT_COVERAGE_GAIN_THRESHOLD = 0.0025
CARD_PLAY_PRIOR_QUALITY_SCALE = 100000.0
CARD_PLAY_PRIOR_QUALITY_MIN = -1.0
CARD_PLAY_PRIOR_QUALITY_MAX = 10.0


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
    ACTION_ACTIVITY: ['ProduceEffectType_EventActivityProducePointUp'],
    ACTION_BUSINESS: ['ProduceEffectType_EventBusinessVoteCountUp'],
    ACTION_PRESENT: ['ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceRewardSet', 'ProduceEffectType_ProduceCardUpgrade'],
    ACTION_SCHOOL_CLASS: ['ProduceEffectType_ProduceReward'],
    ACTION_OUTING: ['ProduceEffectType_StaminaRecoverMultiple', 'ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceCardUpgrade'],
    ACTION_ACTIVITY_SUPPLY: ['ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceRewardSet'],
    ACTION_REFRESH: ['ProduceEffectType_StaminaRecoverMultiple'],
    ACTION_PRE_AUDITION_CONTINUE: [],
    **{action_type: [] for action_type in SHOP_CARD_ACTION_TYPES},
    **{action_type: [] for action_type in SHOP_DRINK_ACTION_TYPES},
    **{action_type: [] for action_type in SHOP_UPGRADE_ACTION_TYPES},
    **{action_type: [] for action_type in SHOP_DELETE_ACTION_TYPES},
}

EVENT_ACTION_TYPES = {ACTION_ACTIVITY, ACTION_BUSINESS, ACTION_PRESENT, ACTION_SCHOOL_CLASS, ACTION_OUTING, ACTION_ACTIVITY_SUPPLY}
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
PARAMETER_EFFECT_INDEX = {
    'ProduceEffectType_VocalAddition': 0,
    'ProduceEffectType_DanceAddition': 1,
    'ProduceEffectType_VisualAddition': 2,
}
PARAMETER_GROWTH_KEYS = ('vocal_growth', 'dance_growth', 'visual_growth')
PRE_AUDITION_ACTION_TYPES = {
    ACTION_PRE_AUDITION_CONTINUE,
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


def _stage_action_family(action_type: str) -> str:
    """把动作归入阶段内重复统计使用的粗粒度家族。"""

    if action_type in HARD_ACTION_TYPES:
        return 'lesson:hard'
    if action_type in SP_ACTION_TYPES:
        return 'lesson:sp'
    if _is_lesson_action(action_type):
        return 'lesson:normal'
    if action_type in EVENT_ACTION_TYPES:
        return f'event:{action_type}'
    return action_type


def _stage_lesson_stat_key(action_type: str) -> str:
    """返回动作对应的课程属性分支；非课程动作返回空字符串。"""

    if not _is_lesson_action(action_type):
        return ''
    return _lesson_stat_type(action_type)


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
    route_feasibility: float = 0.0
    route_vote_margin: float = 0.0
    route_param_margin: float = 0.0


@dataclass(frozen=True)
class ProduceRewardSnapshot:
    """奖励塑形使用的状态快照。"""

    param_phi: float
    fan_phi: float
    resource_phi: float
    stamina_ratio: float
    param_coverage_phi: float = 0.0
    param_weakest_phi: float = 0.0
    stamina_readiness_phi: float = 1.0
    fan_votes: float = 0.0
    next_fan_vote_threshold: float = 0.0


@dataclass(frozen=True)
class ProduceParamProgress:
    """参数势函数的分项进度。"""

    total: float
    weighted: float
    coverage: float
    weakest: float


@dataclass(frozen=True)
class BusinessCardReward:
    """营业附带技能卡奖励。"""

    card_id: str
    upgrade_count: int = 0


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


class ExamActionSelector(Protocol):
    """考试阶段动作选择器协议。"""

    def select_action(self, runtime: ExamRuntime) -> Any | None:
        """根据当前考试运行时返回一个可执行动作。"""


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
        exam_action_selectors: dict[str, ExamActionSelector] | None = None,
        force_lowest_audition_route: bool = False,
    ):
        """初始化培育运行时，并预读取事件、课程和卡组相关主数据。"""

        self.repository = repository
        self.produce_reward_cfg: ProduceRewardConfig = produce_reward_config or build_produce_reward_config()
        self.exam_action_selectors: dict[str, ExamActionSelector] = dict(exam_action_selectors or {})
        self.force_lowest_audition_route = bool(force_lowest_audition_route)
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
        self._produce_effect_ids_by_type: dict[str, list[str]] = defaultdict(list)
        for row in self.produce_effects.rows:
            effect_type = str(row.get('produceEffectType') or '')
            effect_id = str(row.get('id') or '')
            if effect_type and effect_id:
                self._produce_effect_ids_by_type[effect_type].append(effect_id)
        self._selection_card_pool_cache_key: tuple[Any, ...] | None = None
        self._selection_card_pool_cache_value: list[dict[str, Any]] = []

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

    def _default_idol_card_row(self) -> dict[str, Any]:
        """无显式 loadout 时，从主数据库选择稳定的默认偶像卡基础数据。"""

        for idol_card_id in list_trainable_idol_card_ids(self.repository, self.scenario):
            row = self.repository.load_table('IdolCard').first(idol_card_id)
            if row is not None:
                return row
        raise KeyError(f'No trainable idol card found in master database: produce_id={self.scenario.produce_id}')

    def _base_state(self) -> dict[str, Any]:
        """构造包含属性、成长率和流程加成字段的初始状态。"""

        if self.idol_loadout is not None:
            profile = self.idol_loadout.stat_profile
            base_stats = np.array([profile.vocal, profile.dance, profile.visual], dtype=np.float32)
            base_stamina = float(profile.stamina or 0.0)
            vocal_growth = float(profile.vocal_growth_rate)
            dance_growth = float(profile.dance_growth_rate)
            visual_growth = float(profile.visual_growth_rate)
        else:
            idol_card_row = self._default_idol_card_row()
            base_stats = np.array(
                [
                    float(idol_card_row.get('produceVocal') or 0.0),
                    float(idol_card_row.get('produceDance') or 0.0),
                    float(idol_card_row.get('produceVisual') or 0.0),
                ],
                dtype=np.float32,
            )
            base_stamina = float(idol_card_row.get('produceStamina') or 0.0)
            vocal_growth = float(idol_card_row.get('produceVocalGrowthRatePermil') or 0.0) / 1000.0
            dance_growth = float(idol_card_row.get('produceDanceGrowthRatePermil') or 0.0) / 1000.0
            visual_growth = float(idol_card_row.get('produceVisualGrowthRatePermil') or 0.0) / 1000.0
        if base_stamina <= 0.0:
            raise ValueError(f'Idol stamina missing from master database: produce_id={self.scenario.produce_id}')
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
            'before_audition_refresh_applied': False,
            'stage_action_counts': {},
            'stage_action_family_counts': {},
            'stage_lesson_stat_counts': {},
            'challenge_lesson_perfect_bonus_ratio': self._challenge_lesson_perfect_bonus_ratio(),
            'challenge_audition_npc_bonus_ratio': self._challenge_audition_npc_bonus_ratio(),
        }

    def _parameter_growth_limit(self) -> float:
        """返回当前模式主数据里的三维成长上限。"""

        return max(float(self.scenario.parameter_growth_limit or 0.0), 0.0)

    def _clamp_parameter_value(self, value: float) -> float:
        """按当前模式上限裁剪单项三维属性。"""

        limit = self._parameter_growth_limit()
        if limit > 0:
            return float(np.clip(value, 0.0, limit))
        return max(float(value), 0.0)

    def _gain_parameter(self, key: str, delta: float) -> None:
        """统一处理培育阶段的三维属性增长，确保不会超过模式上限。"""

        self.state[key] = self._clamp_parameter_value(float(self.state.get(key) or 0.0) + float(delta))

    def _stage_counter(self, state_key: str) -> dict[str, int]:
        """读取阶段内动作计数字典，缺失或格式异常时重建。"""

        counter = self.state.get(state_key)
        if isinstance(counter, dict):
            return counter
        rebuilt_counter: dict[str, int] = {}
        self.state[state_key] = rebuilt_counter
        return rebuilt_counter

    def _stage_count(self, state_key: str, counter_key: str) -> int:
        """读取阶段内指定键的历史次数。"""

        if not counter_key:
            return 0
        return int(self._stage_counter(state_key).get(counter_key, 0) or 0)

    def _reset_stage_action_counts(self) -> None:
        """在进入下一场考试阶段后清空本阶段动作重复统计。"""

        self.state['stage_action_counts'] = {}
        self.state['stage_action_family_counts'] = {}
        self.state['stage_lesson_stat_counts'] = {}

    def _record_stage_action(self, candidate: ProduceActionCandidate) -> None:
        """记录 weekly 阶段已执行动作，供后续候选动作显式感知重复度。"""

        action_type = 'auto_skip' if candidate.auto_skip else str(candidate.action_type)
        if not action_type:
            return
        action_counter = self._stage_counter('stage_action_counts')
        action_counter[action_type] = int(action_counter.get(action_type, 0) or 0) + 1

        family_key = _stage_action_family(action_type)
        family_counter = self._stage_counter('stage_action_family_counts')
        family_counter[family_key] = int(family_counter.get(family_key, 0) or 0) + 1

        lesson_stat_key = _stage_lesson_stat_key(action_type)
        if lesson_stat_key:
            stat_counter = self._stage_counter('stage_lesson_stat_counts')
            stat_counter[lesson_stat_key] = int(stat_counter.get(lesson_stat_key, 0) or 0) + 1

    def stage_action_repeat_counts(self, candidate: ProduceActionCandidate) -> tuple[int, int, int]:
        """返回候选动作在当前考试阶段内的精确、家族和属性分支重复次数。"""

        action_type = 'auto_skip' if candidate.auto_skip else str(candidate.action_type)
        action_count = self._stage_count('stage_action_counts', action_type)
        family_count = self._stage_count('stage_action_family_counts', _stage_action_family(action_type))
        stat_count = self._stage_count('stage_lesson_stat_counts', _stage_lesson_stat_key(action_type))
        return action_count, family_count, stat_count

    def estimate_candidate_repeat_risk(self, candidate: ProduceActionCandidate) -> float:
        """估算候选动作是否会把当前阶段推向无弱项收益的重复套路。"""

        if not _is_lesson_action(candidate.action_type):
            return 0.0
        action_count, family_count, stat_count = self.stage_action_repeat_counts(candidate)
        if action_count <= 0 and stat_count <= 1 and family_count <= 2:
            return 0.0

        current_progress = self._param_progress()
        projected_progress = self.estimate_candidate_param_progress(candidate)
        weakest_gain = projected_progress.weakest - current_progress.weakest
        coverage_gain = projected_progress.coverage - current_progress.coverage
        if current_progress.weakest >= 0.98:
            return 0.0
        if weakest_gain >= STAGE_REPEAT_WEAKEST_GAIN_THRESHOLD:
            return 0.0
        if self.scenario.route_type != 'nia' and coverage_gain >= STAGE_REPEAT_COVERAGE_GAIN_THRESHOLD:
            return 0.0

        exact_pressure = min(float(action_count) / STAGE_REPEAT_RATIO_SCALE, 1.0)
        stat_pressure = min(max(float(stat_count - 1), 0.0) / STAGE_REPEAT_RATIO_SCALE, 1.0)
        family_pressure = min(max(float(family_count - 2), 0.0) / STAGE_REPEAT_RATIO_SCALE, 1.0)
        return float(np.clip(exact_pressure * 0.40 + stat_pressure * 0.45 + family_pressure * 0.15, 0.0, 1.0))

    def _stage_repetition_penalty(
        self,
        candidate: ProduceActionCandidate,
        before: ProduceRewardSnapshot,
        after: ProduceRewardSnapshot,
    ) -> float:
        """根据真实状态变化惩罚无弱项收益的阶段内重复课程动作。"""

        if not _is_lesson_action(candidate.action_type):
            return 0.0
        action_count, family_count, stat_count = self.stage_action_repeat_counts(candidate)
        if action_count <= 0 and stat_count <= 1 and family_count <= 2:
            return 0.0

        weakest_gain = after.param_weakest_phi - before.param_weakest_phi
        coverage_gain = after.param_coverage_phi - before.param_coverage_phi
        if before.param_weakest_phi >= 0.98:
            return 0.0
        if weakest_gain >= STAGE_REPEAT_WEAKEST_GAIN_THRESHOLD:
            return 0.0
        if self.scenario.route_type != 'nia' and coverage_gain >= STAGE_REPEAT_COVERAGE_GAIN_THRESHOLD:
            return 0.0

        exact_pressure = min(float(action_count) / STAGE_REPEAT_RATIO_SCALE, 1.0)
        stat_pressure = min(max(float(stat_count - 1), 0.0) / STAGE_REPEAT_RATIO_SCALE, 1.0)
        family_pressure = min(max(float(family_count - 2), 0.0) / STAGE_REPEAT_RATIO_SCALE, 1.0)
        repeat_pressure = exact_pressure * 0.40 + stat_pressure * 0.45 + family_pressure * 0.15
        penalty_cap = NIA_STAGE_REPEAT_UNHELPFUL_PENALTY if self.scenario.route_type == 'nia' else STAGE_REPEAT_UNHELPFUL_PENALTY
        return -float(np.clip(repeat_pressure * penalty_cap, 0.0, penalty_cap))

    def estimate_candidate_idle_event_risk(self, candidate: ProduceActionCandidate) -> float:
        """估算候选事件是否会在 NIA 参数不足时继续堆无效票数或资源。"""

        if self.scenario.route_type != 'nia' or candidate.action_type not in EVENT_ACTION_TYPES:
            return 0.0
        projected_progress = self.estimate_candidate_param_progress(candidate)
        current_progress = self._param_progress()
        if projected_progress.total - current_progress.total >= STATE_DELTA_NIA_FAN_ONLY_PROGRESS_THRESHOLD:
            return 0.0

        threshold = max(self._next_fan_vote_threshold(), 1.0)
        fan_votes = max(float(self.state.get('fan_votes') or 0.0), 0.0)
        if self._next_fan_vote_threshold() > 0 and fan_votes < threshold * 0.85:
            return 0.0

        action_count, family_count, _stat_count = self.stage_action_repeat_counts(candidate)
        repeat_pressure = min(
            float(action_count) * 0.70 + max(float(family_count - 1), 0.0) * 0.30,
            STAGE_REPEAT_RATIO_SCALE,
        ) / STAGE_REPEAT_RATIO_SCALE
        if repeat_pressure <= 0.0:
            return 0.0
        param_shortfall = max(0.88 - min(current_progress.total, 0.88), 0.0) / 0.88
        weakest_shortfall = max(0.85 - min(current_progress.weakest, 0.85), 0.0) / 0.85
        return float(np.clip(repeat_pressure * param_shortfall * weakest_shortfall, 0.0, 1.0))

    def _stage_idle_event_penalty(
        self,
        candidate: ProduceActionCandidate,
        before: ProduceRewardSnapshot,
        after: ProduceRewardSnapshot,
    ) -> float:
        """惩罚 NIA 中参数短板未解决时反复选择无参数进展事件。"""

        if self.scenario.route_type != 'nia' or candidate.action_type not in EVENT_ACTION_TYPES:
            return 0.0
        param_gain = max(after.param_phi - before.param_phi, 0.0)
        weakest_gain = max(after.param_weakest_phi - before.param_weakest_phi, 0.0)
        stamina_recovery = max(after.stamina_ratio - before.stamina_ratio, 0.0)
        if param_gain >= STATE_DELTA_NIA_FAN_ONLY_PROGRESS_THRESHOLD or weakest_gain >= 0.001:
            return 0.0
        if stamina_recovery > 0.0 and before.stamina_ratio < 0.28:
            return 0.0
        threshold = max(before.next_fan_vote_threshold, 1.0)
        if before.next_fan_vote_threshold > 0 and after.fan_votes < threshold * 0.85:
            return 0.0

        action_count, family_count, _stat_count = self.stage_action_repeat_counts(candidate)
        repeat_pressure = min(
            float(action_count) * 0.70 + max(float(family_count - 1), 0.0) * 0.30,
            STAGE_REPEAT_RATIO_SCALE,
        ) / STAGE_REPEAT_RATIO_SCALE
        param_shortfall = max(0.88 - min(before.param_phi, 0.88), 0.0) / 0.88
        weakest_shortfall = max(0.85 - min(before.param_weakest_phi, 0.85), 0.0) / 0.85
        penalty = repeat_pressure * param_shortfall * weakest_shortfall * STATE_DELTA_NIA_IDLE_EVENT_REPEAT_PENALTY
        return -float(np.clip(penalty, 0.0, STATE_DELTA_NIA_IDLE_EVENT_REPEAT_PENALTY))

    def reset(self) -> None:
        """重置培育状态、初始牌组、饮料与开场效果。"""

        self.state = self._base_state()
        self.deck = list(build_initial_exam_deck(self.repository, self.scenario, rng=self.np_random, loadout=self.idol_loadout))
        self.initial_deck_card_ids = {str(card.get('id') or '') for card in self.deck if str(card.get('id') or '')}
        self.drinks = []
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
        for extra_item_id in self.idol_loadout.extra_produce_item_ids:
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

        fired_item_ids = context.get('_fired_item_ids')
        if not isinstance(fired_item_ids, set):
            fired_item_ids = set()
        if self.active_produce_items:
            snapshot = list(self.active_produce_items)
            for active_item in snapshot:
                if active_item.item_id in fired_item_ids:
                    continue
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
                fired_item_ids.add(active_item.item_id)
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
        difficulty_id = self._current_audition_difficulty_id()
        selected_row = self._next_audition_row(stage_type, difficulty_id)
        if selected_row is not None:
            return self._battle_profile_from_audition_row(selected_row)
        return self.repository.battle_profile(
            self.scenario,
            stage_type=stage_type,
            audition_difficulty_id=difficulty_id or None,
        )

    def _current_audition_difficulty_id(self) -> str:
        """返回当前偶像卡绑定的试镜难度 id。"""

        if self.idol_loadout is None:
            return ''
        return str(self.idol_loadout.stat_profile.audition_difficulty_id or '')

    def _next_audition_row(self, stage_type: str, difficulty_id: str) -> dict[str, Any] | None:
        """按当前阶段、偶像卡和票数解析下一场试镜行。"""

        if self.scenario.route_type != 'nia':
            return None
        selected_row_id = self._resolve_selected_audition_row_id(stage_type)
        if selected_row_id:
            selected_id, _, selected_number_text = selected_row_id.partition(':')
            try:
                selected_number = int(selected_number_text or 0)
            except ValueError:
                selected_number = 0
            for row in self.repository.audition_rows(self.scenario, stage_type, audition_difficulty_id=difficulty_id or None):
                if str(row.get('id') or '') == selected_id and int(row.get('number') or 0) == selected_number:
                    return row
        return self.repository.select_audition_row(
            self.scenario,
            stage_type=stage_type,
            audition_difficulty_id=difficulty_id or None,
            fan_votes=float(self.state.get('fan_votes') or 0.0),
        )

    def _battle_profile_from_audition_row(self, row: dict[str, Any]) -> dict[str, float]:
        """从试镜难度行构造培育奖励使用的考核 profile。"""

        config = self.repository.battle_config_map.get(str(row.get('produceExamBattleConfigId') or '')) or {}
        weight_vector = np.array(
            [
                float(config.get('vocal') or self.scenario.score_weights[0]),
                float(config.get('dance') or self.scenario.score_weights[1]),
                float(config.get('visual') or self.scenario.score_weights[2]),
            ],
            dtype=np.float32,
        )
        weight_sum = float(weight_vector.sum())
        if weight_sum > 0:
            weight_vector = weight_vector / weight_sum
        return {
            'base_score': float(row.get('baseScore') or 0.0),
            'force_end_score': float(row.get('forceEndScore') or 0.0),
            'rank_threshold': float(row.get('rankThreshold') or 0.0),
            'parameter_baseline': float(row.get('parameterBaseLine') or 0.0),
            'fan_vote_baseline': float(row.get('voteCountBaseLine') or 0.0),
            'fan_vote_requirement': float(row.get('voteCount') or 0.0),
            'turns': float(config.get('turn') or self.scenario.exam_turns),
            'vocal_weight': float(weight_vector[0]),
            'dance_weight': float(weight_vector[1]),
            'visual_weight': float(weight_vector[2]),
        }

    def _produce_param_target(self) -> float:
        """返回当前阶段的参数成长目标。"""

        profile = self._next_audition_profile()
        profile_target = float(profile.get('parameter_baseline') or 0.0)
        current_idx = int(self.state.get('audition_index') or 0)
        stage_count = max(len(self.scenario.audition_sequence), 1)
        stage_ratio = (min(max(current_idx, 0), stage_count - 1) + 1) / stage_count
        if self.scenario.route_type == 'nia':
            base_ratio = NIA_PARAM_TARGET_STAGE_BASE_RATIO
            progress_ratio = NIA_PARAM_TARGET_STAGE_PROGRESS_RATIO
        else:
            base_ratio = PARAM_TARGET_STAGE_BASE_RATIO
            progress_ratio = PARAM_TARGET_STAGE_PROGRESS_RATIO
        growth_target = float(self.scenario.parameter_growth_limit) * (base_ratio + progress_ratio * stage_ratio)
        target = max(profile_target, growth_target)
        if target <= 0.0:
            raise ValueError(f'Produce parameter target missing from master database: produce_id={self.scenario.produce_id}')
        return target

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
        raise KeyError(f'Final audition parameter baseline not found in master database: produce_id={self.scenario.produce_id}')

    def _param_progress_for_stats(self, stats_values: tuple[float, float, float]) -> ProduceParamProgress:
        """返回指定三维参数对当前阶段目标的分项进度。"""

        target = self._produce_param_target()
        if target <= 0:
            return ProduceParamProgress(total=0.0, weighted=0.0, coverage=0.0, weakest=0.0)
        weights = np.array(self._next_param_weights(), dtype=np.float32)
        w_sum = float(weights.sum())
        weights = weights / max(w_sum, 1e-6)
        stats = np.array(stats_values, dtype=np.float32)
        weighted = float(np.dot(stats, weights))
        ratio = weighted / target
        weighted_progress = min(ratio, 1.0)

        # 审查基准不是单一总分门槛，低权重属性也会影响最终分数和大成功概率。
        per_stat_targets = target * (0.45 + weights * 1.65)
        stat_ratios = stats / np.maximum(per_stat_targets, 1.0)
        clipped_ratios = np.minimum(stat_ratios, 1.0)
        coverage_progress = float(np.mean(clipped_ratios))
        weakest_progress = float(np.min(clipped_ratios))

        overshoot = max(ratio - 1.0, 0.0)
        if self.scenario.route_type == 'nia':
            weighted_weight = NIA_PARAM_PROGRESS_WEIGHTED_WEIGHT
            coverage_weight = NIA_PARAM_PROGRESS_COVERAGE_WEIGHT
            weakest_weight = NIA_PARAM_PROGRESS_WEAKEST_WEIGHT
            overshoot_weight = NIA_PARAM_PROGRESS_OVERSHOOT_WEIGHT
        else:
            weighted_weight = PARAM_PROGRESS_WEIGHTED_WEIGHT
            coverage_weight = PARAM_PROGRESS_COVERAGE_WEIGHT
            weakest_weight = PARAM_PROGRESS_WEAKEST_WEIGHT
            overshoot_weight = PARAM_PROGRESS_OVERSHOOT_WEIGHT
        total = (
            weighted_progress * weighted_weight
            + coverage_progress * coverage_weight
            + weakest_progress * weakest_weight
            + math.log1p(overshoot * 2.0) / math.log(3.0) * overshoot_weight
        )
        if self.scenario.route_type == 'nia':
            imbalance = float(np.max(clipped_ratios) - np.min(clipped_ratios))
            weakest_shortfall = max(0.90 - min(weakest_progress, 0.90), 0.0) / 0.90
            total -= imbalance * weakest_shortfall * NIA_PARAM_PROGRESS_IMBALANCE_PENALTY_WEIGHT
            total = max(total, 0.0)
        return ProduceParamProgress(
            total=float(total),
            weighted=float(weighted_progress),
            coverage=coverage_progress,
            weakest=weakest_progress,
        )

    def _param_progress(self) -> ProduceParamProgress:
        """返回当前三维参数对当前阶段目标的分项进度。"""

        return self._param_progress_for_stats(
            (
                float(self.state['vocal']),
                float(self.state['dance']),
                float(self.state['visual']),
            )
        )

    def _phi_param(self) -> float:
        """φ_param：当前三维参数对下一场考核基准线的覆盖进度。"""

        return self._param_progress().total

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
            # 全部试镜已解锁后，不再鼓励继续刷溢出票数。
            return self._nia_fan_value_multiplier()
        unlock_progress = min(fan_votes / max(next_threshold, 1.0), 1.0)
        return unlock_progress * self._nia_fan_value_multiplier()

    def _nia_fan_value_multiplier(self) -> float:
        """返回 NIA 粉丝票价值相对参数准备度的折扣系数。"""

        if self.scenario.route_type != 'nia':
            return 1.0
        param_ready = float(np.clip(self._phi_param(), 0.0, 1.0))
        return NIA_FAN_VALUE_PARAM_FLOOR + NIA_FAN_VALUE_PARAM_SCALE * param_ready

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
        pp_route_scale = 1.0 if self._supports_pre_audition_actions() else (0.20 if ACTION_OUTING in self.scenario.action_types else 0.0)
        if self.pre_audition_phase == 'shop':
            pp_value = min(produce_points / max(cfg.pp_left_cap, 1e-6), 1.0)
        else:
            if remaining_to_audition <= int(cfg.pre_audition_window_near):
                pp_window = cfg.pp_window_near_weight
            elif remaining_to_audition <= int(cfg.pre_audition_window_mid):
                pp_window = cfg.pp_window_mid_weight
            else:
                pp_window = cfg.pp_window_far_weight
            pp_value = min(produce_points / max(cfg.pp_left_cap, 1e-6), 1.0) * pp_window * pp_route_scale

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

    def _produce_reward_snapshot(self) -> ProduceRewardSnapshot:
        """返回不依赖动作名的培育奖励状态快照。"""

        max_stamina = max(float(self.state.get('max_stamina') or 1.0), 1.0)
        stamina_ratio = float(self.state.get('stamina') or 0.0) / max_stamina
        param_progress = self._param_progress()
        return ProduceRewardSnapshot(
            param_phi=param_progress.total,
            fan_phi=self._phi_fan(),
            resource_phi=self._phi_resource(),
            stamina_ratio=float(np.clip(stamina_ratio, 0.0, 1.0)),
            param_coverage_phi=param_progress.coverage,
            param_weakest_phi=param_progress.weakest,
            stamina_readiness_phi=self._stamina_readiness_phi(),
            fan_votes=float(self.state.get('fan_votes') or 0.0),
            next_fan_vote_threshold=self._next_fan_vote_threshold(),
        )

    def _raw_fan_surplus_pressure(self, before: ProduceRewardSnapshot, after: ProduceRewardSnapshot) -> float:
        """返回 NIA 溢出票数对奖励的惩罚压力。"""

        if self.scenario.route_type != 'nia':
            return 0.0
        raw_fan_gain = max(after.fan_votes - before.fan_votes, 0.0)
        if raw_fan_gain <= 0:
            return 0.0
        threshold = max(before.next_fan_vote_threshold, 1.0)
        if before.next_fan_vote_threshold <= 0:
            surplus_ratio = raw_fan_gain / threshold
        else:
            surplus_ratio = max(after.fan_votes - before.next_fan_vote_threshold, 0.0) / threshold
        param_shortfall = max(1.0 - min(before.param_phi, 1.0), 0.0)
        weakest_shortfall = max(0.85 - min(before.param_weakest_phi, 0.85), 0.0) / 0.85
        fan_gain_pressure = raw_fan_gain / (raw_fan_gain + threshold)
        return float(np.clip(max(surplus_ratio, fan_gain_pressure) * param_shortfall * weakest_shortfall, 0.0, 1.0))

    def _state_delta_bonus(self, before: ProduceRewardSnapshot, after: ProduceRewardSnapshot) -> float:
        """按状态改善提供小幅即时奖励，避免奖励函数记忆固定动作名。"""

        param_gain = max(after.param_phi - before.param_phi, 0.0)
        coverage_gain = max(after.param_coverage_phi - before.param_coverage_phi, 0.0)
        weakest_gain = max(after.param_weakest_phi - before.param_weakest_phi, 0.0)
        fan_gain = max(after.fan_phi - before.fan_phi, 0.0)
        raw_fan_gain = max(after.fan_votes - before.fan_votes, 0.0)
        resource_gain = max(after.resource_phi - before.resource_phi, 0.0)
        stamina_recovery = max(after.stamina_ratio - before.stamina_ratio, 0.0)
        stamina_readiness_gain = max(after.stamina_readiness_phi - before.stamina_readiness_phi, 0.0)
        stamina_readiness_drop = max(before.stamina_readiness_phi - after.stamina_readiness_phi, 0.0)
        stamina_recovery_need = max(0.40 - before.stamina_ratio, 0.0) / 0.40
        useful_stamina_recovery = stamina_recovery * stamina_recovery_need
        low_stamina_drop = max(before.stamina_ratio - after.stamina_ratio, 0.0) if after.stamina_ratio < 0.22 else 0.0
        productive_gain = param_gain + fan_gain + resource_gain
        wasted_recovery = stamina_recovery if before.stamina_ratio >= 0.45 and productive_gain < 0.002 else 0.0
        no_progress_penalty = STATE_DELTA_NO_PROGRESS_PENALTY if productive_gain < 0.001 and useful_stamina_recovery < 0.001 else 0.0
        high_stamina_idle_penalty = (
            STATE_DELTA_HIGH_STAMINA_IDLE_PENALTY
            if productive_gain < 0.001
            and stamina_recovery < 0.001
            and before.stamina_ratio >= FIRST_STAR_REFRESH_HIGH_STAMINA_LOCK_RATIO
            else 0.0
        )
        remaining_weeks = self._remaining_weeks_to_next_audition()
        param_shortfall = max(0.88 - before.param_phi, 0.0) / 0.88
        pressure_window = max(float(self._produce_reward_config().pre_audition_window_mid), 1.0)
        param_pressure = param_shortfall * max((pressure_window + 1.0 - float(remaining_weeks)) / pressure_window, 0.0)
        if self.scenario.route_type == 'nia':
            param_pressure *= STATE_DELTA_NIA_PARAM_PRESSURE_MULTIPLIER
        resource_discount = max(1.0 - param_pressure * STATE_DELTA_PARAM_PRESSURE_RESOURCE_DISCOUNT, 0.20)
        fan_discount = max(1.0 - param_pressure * STATE_DELTA_FAN_PRESSURE_DISCOUNT, 0.10)
        param_pressure_penalty = (
            param_pressure * STATE_DELTA_PARAM_PRESSURE_PENALTY_WEIGHT
            if param_gain < 0.004 and useful_stamina_recovery < 0.001
            else 0.0
        )
        fan_only_penalty = (
            min(fan_gain, 0.25)
            * STATE_DELTA_NIA_FAN_ONLY_PENALTY_WEIGHT
            * max(1.0 - min(before.param_phi, 1.0), 0.0)
            if self.scenario.route_type == 'nia'
            and fan_gain > 0.001
            and param_gain < STATE_DELTA_NIA_FAN_ONLY_PROGRESS_THRESHOLD
            else 0.0
        )
        fan_surplus_penalty = (
            self._raw_fan_surplus_pressure(before, after)
            * STATE_DELTA_NIA_FAN_SURPLUS_PENALTY_WEIGHT
            if raw_fan_gain > 0.0 and param_gain < STATE_DELTA_NIA_FAN_ONLY_PROGRESS_THRESHOLD
            else 0.0
        )
        stalled_weakest_penalty = (
            min(param_gain / 0.030, 1.0)
            * STATE_DELTA_NIA_STALLED_WEAKEST_PENALTY
            * max(0.80 - min(before.param_weakest_phi, 0.80), 0.0)
            / 0.80
            if self.scenario.route_type == 'nia'
            and param_gain > 0.002
            and weakest_gain < 0.001
            and before.param_weakest_phi < 0.80
            else 0.0
        )
        lopsided_penalty_weight = STATE_DELTA_LOPSIDED_PARAM_PENALTY
        if self.scenario.route_type == 'nia':
            lopsided_penalty_weight *= STATE_DELTA_NIA_LOPSIDED_PENALTY_MULTIPLIER
        lopsided_param_penalty = (
            lopsided_penalty_weight * min(param_gain / 0.030, 1.0)
            if param_gain > 0.004 and weakest_gain < 0.001 and before.param_weakest_phi < 0.95
            else 0.0
        )

        bonus = (
            param_gain * STATE_DELTA_PARAM_WEIGHT
            + coverage_gain * STATE_DELTA_COVERAGE_GAIN_WEIGHT
            + weakest_gain * STATE_DELTA_WEAKEST_GAIN_WEIGHT
            + fan_gain * STATE_DELTA_FAN_WEIGHT * fan_discount
            + resource_gain * STATE_DELTA_RESOURCE_WEIGHT * resource_discount
            + useful_stamina_recovery * STATE_DELTA_STAMINA_RECOVERY_WEIGHT
            + stamina_readiness_gain * STATE_DELTA_STAMINA_READINESS_GAIN_WEIGHT
            - min(low_stamina_drop, 0.4) * STATE_DELTA_LOW_STAMINA_PENALTY_WEIGHT
            - min(wasted_recovery, 0.4) * STATE_DELTA_WASTED_RECOVERY_PENALTY_WEIGHT
            - stamina_readiness_drop * STATE_DELTA_STAMINA_READINESS_DROP_WEIGHT
            - no_progress_penalty
            - high_stamina_idle_penalty
            - param_pressure_penalty
            - fan_only_penalty
            - fan_surplus_penalty
            - stalled_weakest_penalty
            - lopsided_param_penalty
        )
        return float(np.clip(bonus, -STATE_DELTA_REWARD_CLIP, STATE_DELTA_REWARD_CLIP))

    def _hard_lesson_level_row(self, action_type: str) -> dict[str, Any]:
        """按当前 hard lesson 动作精确匹配对应的关卡主数据行。"""

        if action_type not in HARD_ACTION_TYPES:
            return {}
        level_index = max(int(self.state.get('audition_index') or 0) + 1, 1)
        try:
            spec = self.repository.resolve_lesson_training_spec(
                self.scenario,
                action_type=action_type,
                loadout=self.idol_loadout,
                level_index=level_index,
            )
        except KeyError:
            spec = self.repository.resolve_lesson_training_spec(
                self.scenario,
                action_type=action_type,
                loadout=self.idol_loadout,
                rng=self.np_random,
            )
        level_row = self.lesson_levels.first(spec.source_level_id)
        if level_row is None:
            raise KeyError(f'Lesson level row not found: {spec.source_level_id}')
        return level_row

    def _remaining_weeks_to_next_audition(self) -> int:
        """返回距离下一场考试前还剩多少个 weekly 回合。"""

        current_idx = int(self.state.get('audition_index') or 0)
        if current_idx >= len(self.checkpoints):
            return 0
        checkpoint_step, _ = self.checkpoints[current_idx]
        return max(int(checkpoint_step - int(self.state.get('step') or 0)), 0)

    def _first_star_hard_lesson_stamina_cost(self) -> float:
        """返回初路线追込课的最低体力成本估计。"""

        lesson_profiles = self.repository.lesson_profile_stats
        normal_profile = float(lesson_profiles.get('normal') or 0.0)
        if normal_profile <= 0.0:
            raise ValueError('Normal lesson profile missing from master database')
        hard_profile = max(float(lesson_profiles.get('hard') or normal_profile), normal_profile)
        hard_scale = hard_profile / normal_profile
        return 5.0 + 1.5 * hard_scale

    def _stamina_readiness_phi_for(self, stamina: float) -> float:
        """估算当前体力是否足够支撑到下一次追込课。"""

        if self.scenario.route_type != 'first_star':
            return 1.0
        if int(self.state.get('audition_index') or 0) >= len(self.checkpoints):
            return 1.0
        remaining_weeks = self._remaining_weeks_to_next_audition()
        if remaining_weeks <= 1:
            return 1.0
        hard_cost = self._first_star_hard_lesson_stamina_cost()
        weeks_until_hard = max(remaining_weeks - 2, 0)
        runway_cost = max(weeks_until_hard - 1, 0) * 4.5
        required_stamina = max(hard_cost + runway_cost, hard_cost)
        return float(np.clip(max(float(stamina), 0.0) / max(required_stamina, 1.0), 0.0, 1.0))

    def _stamina_readiness_phi(self) -> float:
        """返回当前体力对后续强制追込课的准备度。"""

        return self._stamina_readiness_phi_for(float(self.state.get('stamina') or 0.0))

    def _is_first_star_pre_audition_hard_lesson_week(self) -> bool:
        """判断当前是否处于初路线考试前的追込课周。"""

        return self.scenario.route_type == 'first_star' and self._remaining_weeks_to_next_audition() == 2

    def _is_first_star_pre_audition_refresh_week(self) -> bool:
        """判断当前是否处于初路线考试前的强制恢复周。"""

        return self.scenario.route_type == 'first_star' and self._remaining_weeks_to_next_audition() == 1

    def _normal_refresh_has_stamina_value(self) -> bool:
        """判断普通休息是否符合体力未满的选择条件。"""

        if self._is_first_star_pre_audition_refresh_week():
            return True
        max_stamina = max(float(self.state.get('max_stamina') or 0.0), 1.0)
        stamina = float(self.state.get('stamina') or 0.0)
        return stamina < max_stamina - 1e-6

    def _is_refresh_locked_by_early_schedule(self) -> bool:
        """判断当前周是否处于文档规定的休息锁定期。"""

        if self._is_first_star_pre_audition_refresh_week():
            return False
        current_step = int(self.state.get('step') or 0)
        if current_step >= 4:
            return False
        return self.scenario.route_type == 'nia' or self.scenario.produce_id in {'produce-003', 'produce-006'}

    def _candidate_has_runtime_effect(self, candidate: ProduceActionCandidate) -> bool:
        """判断候选动作是否会对培育状态产生真实执行效果。"""

        if abs(float(candidate.stamina_delta or 0.0)) > 1e-6:
            return True
        if abs(float(candidate.produce_point_delta or 0.0)) > 1e-6:
            return True
        if any(abs(float(value)) > 1e-6 for value in candidate.stat_deltas + candidate.boost_stat_deltas):
            return True
        if candidate.produce_card_id:
            return True
        return bool(candidate.produce_effect_ids or candidate.success_effect_ids or candidate.fail_effect_ids)

    def _should_apply_candidate_effects(self, candidate: ProduceActionCandidate) -> bool:
        """判断自动动作是否仍应结算其效果。"""

        if not candidate.auto_skip:
            return True
        return self._candidate_has_runtime_effect(candidate)

    def _next_fan_vote_threshold(self) -> float:
        """返回当前阶段下一场试镜的粉丝票数门槛；已全部解锁时返回 0。"""

        current_idx = int(self.state.get('audition_index') or 0)
        if current_idx >= len(self.scenario.audition_sequence):
            return 0.0
        stage_type = str(self.scenario.audition_sequence[current_idx] or '')
        fan_votes = float(self.state.get('fan_votes') or 0.0)
        rows = self.repository.audition_rows(
            self.scenario,
            stage_type=stage_type,
            audition_difficulty_id=self._current_audition_difficulty_id() or None,
        )
        for row in sorted(rows, key=lambda item: float(item.get('voteCount') or 0.0)):
            threshold = float(row.get('voteCount') or 0.0)
            if threshold > fan_votes:
                return threshold
        return 0.0

    def _present_bonus_produce_points(self) -> float:
        """差入额外 P 点奖励：触发概率随票数上升，奖励量从主数据回退值读取。"""

        return self._minimum_produce_point_reward()

    def _minimum_produce_point_reward(self) -> float:
        """从主数据 ProduceEffect 中读取最小正数 P 点奖励。"""

        values: list[float] = []
        for effect in self.produce_effects.rows:
            if str(effect.get('produceEffectType') or '') not in {
                'ProduceEffectType_ProducePointAddition',
                'ProduceEffectType_ProducePointAdditionDisableTrigger',
            }:
                continue
            minimum = float(effect.get('effectValueMin') or 0.0)
            maximum = float(effect.get('effectValueMax') or minimum)
            value = minimum if minimum > 0.0 else maximum
            if value > 0.0:
                values.append(value)
        if not values:
            return 0.0
        return min(values)

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

        profile = self._next_audition_profile()
        param_baseline = float(profile.get('parameter_baseline') or 0.0)
        if param_baseline <= 0.0:
            raise ValueError(
                'Business big success baseline missing from master database: '
                f'produce_id={self.scenario.produce_id}, source_row_id={source_row_id}'
            )
        max_stat = max(float(self.state.get('vocal') or 0.0),
                       float(self.state.get('dance') or 0.0),
                       float(self.state.get('visual') or 0.0))
        # 参数达 baseline 时约 30% 大成功，超过 2× baseline 时约 70%
        stat_ratio = max_stat / max(param_baseline, 1.0)
        big_prob = float(np.clip(0.10 + 0.30 * stat_ratio, 0.05, 0.70))
        return bool(self.np_random.random() < big_prob)

    def _business_action_bonus_card_reward(self, source_row_id: str) -> BusinessCardReward:
        """按帮助页为营业抽取附带技能卡奖励，并保留强化段阶。"""

        reward_kind = self._business_reward_kind(source_row_id)
        candidates = self._selection_card_pool()
        if not candidates:
            return BusinessCardReward(card_id='')
        sampled = sample_card_from_weighted_pool(candidates, self.np_random)
        if sampled is None:
            return BusinessCardReward(card_id='')
        card_row = dict(sampled)
        # 商業施設：官方说明为获得强化済み技能卡；リゾート施設只是普通技能卡加体力回复。
        if reward_kind in {'produce_card', 'card'}:
            upgraded = self._lookup_card_upgrade_row(str(card_row.get('id') or ''), min(int(card_row.get('upgradeCount') or 0) + 1, 1))
            if upgraded is not None:
                card_row = dict(upgraded)
        return BusinessCardReward(
            card_id=str(card_row.get('id') or ''),
            upgrade_count=int(card_row.get('upgradeCount') or 0),
        )

    def _business_action_bonus_card(self, source_row_id: str) -> str:
        """兼容旧调用，返回营业附带技能卡 id。"""

        return self._business_action_bonus_card_reward(source_row_id).card_id

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

        cache_key = (
            str(self.idol_loadout.idol_card_id if self.idol_loadout is not None else ''),
            tuple(sorted(self.initial_deck_card_ids)),
            tuple(sorted(self.legend_seen_card_ids)),
        )
        if self._selection_card_pool_cache_key == cache_key:
            return [dict(row) for row in self._selection_card_pool_cache_value]

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
        self._selection_card_pool_cache_key = cache_key
        self._selection_card_pool_cache_value = [dict(row) for row in filtered]
        return [dict(row) for row in filtered]

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
            upgraded = self._lookup_card_upgrade_row(str(card.get('id') or ''), 1)
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
        current_param = max(
            float(self.state.get('vocal') or 0.0),
            float(self.state.get('dance') or 0.0),
            float(self.state.get('visual') or 0.0),
        )
        finale_available = self._finale_available()
        max_number = max(int(row.get('number') or 0) for row in rows)
        grouped_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped_rows[int(row.get('number') or 0)].append(row)
        prepared_candidates: list[tuple[int, ProduceActionCandidate]] = []
        audition_difficulty_id = ''
        if self.idol_loadout is not None:
            audition_difficulty_id = str(self.idol_loadout.stat_profile.audition_difficulty_id or '')
        for number in range(1, max_number + 1):
            key = f'audition_select_{number}'
            candidates = grouped_rows.get(number, [])
            if not candidates:
                prepared_candidates.append(
                    (
                        number,
                        ProduceActionCandidate(
                            label=f'选择试镜 {number}',
                            action_type=key,
                            effect_types=[],
                            produce_effect_ids=[],
                            available=False,
                        ),
                    )
                )
                continue
            preferred_candidates = [
                row
                for row in candidates
                if audition_difficulty_id and str(row.get('id') or '') == audition_difficulty_id
            ]
            candidate_rows = preferred_candidates or candidates
            selectable = next((row for row in candidate_rows if float(row.get('voteCount') or 0.0) <= fan_votes), None)
            label = '选择 FINALE' if stage_type == self.scenario.audition_sequence[-1] and number == max_number else f'选择试镜 {number}'
            selected_row = selectable or candidate_rows[0]
            selector = f"{str(selected_row.get('id') or '')}:{int(selected_row.get('number') or 0)}"
            feasible = True
            required_votes = max(float(selected_row.get('voteCount') or 0.0), 0.0)
            vote_baseline = max(float(selected_row.get('voteCountBaseLine') or required_votes or 1.0), 1.0)
            parameter_baseline = max(float(selected_row.get('parameterBaseLine') or 0.0), 1.0)
            vote_ratio = fan_votes / vote_baseline
            param_ratio = current_param / parameter_baseline
            feasibility = min(vote_ratio, 1.15) * 0.45 + min(param_ratio, 1.25) * 0.55
            # 票数和参数不足都属于路线风险，应交给 reward/动作特征学习；
            # 试镜选项保持可见，避免把低档路线写成动作掩码死逻辑。
            prepared_candidates.append(
                (
                    number,
                    ProduceActionCandidate(
                        label=label,
                        action_type=key,
                        effect_types=[],
                        produce_effect_ids=[],
                        available=feasible and (number != max_number or finale_available),
                        resource_id=selector,
                        slot_index=number,
                        success_probability=float(min(feasibility, 1.0)),
                        produce_point_delta=(fan_votes - required_votes) / vote_baseline,
                        stamina_delta=(current_param - parameter_baseline) / parameter_baseline,
                        resource_level=number,
                        route_feasibility=float(feasibility),
                        route_vote_margin=(fan_votes - required_votes) / vote_baseline,
                        route_param_margin=(current_param - parameter_baseline) / parameter_baseline,
                    ),
                )
            )

        if self.force_lowest_audition_route:
            feasible_numbers = [number for number, candidate in prepared_candidates if candidate.available]
            locked_number = min(feasible_numbers) if feasible_numbers else None
            for number, candidate in prepared_candidates:
                candidate.available = candidate.available and number == locked_number

        for number, candidate in prepared_candidates:
            inventory[candidate.action_type] = candidate
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

    def _selected_audition_row(self, candidate: ProduceActionCandidate) -> dict[str, Any] | None:
        """从考试前选择动作解析出当前锁定的试镜主数据行。"""

        selector = str(candidate.resource_id or '')
        if not selector or ':' not in selector:
            return None
        row_id, _, number_text = selector.partition(':')
        number = int(number_text or 0)
        for row in self.repository.audition_rows(self.scenario, self.pending_audition_stage):
            if str(row.get('id') or '') == row_id and int(row.get('number') or 0) == number:
                return row
        return None

    def _audition_selection_reward(self, candidate: ProduceActionCandidate) -> float:
        """为试镜路线选择提供即时 shaping，鼓励更稳妥的可达路线。"""

        selected_row = self._selected_audition_row(candidate)
        if selected_row is None:
            return 0.0

        fan_votes = max(float(self.state.get('fan_votes') or 0.0), 0.0)
        required_votes = max(float(selected_row.get('voteCount') or 0.0), 0.0)
        vote_baseline = max(float(selected_row.get('voteCountBaseLine') or required_votes or 1.0), 1.0)
        parameter_baseline = max(float(selected_row.get('parameterBaseLine') or 0.0), 1.0)
        current_param = max(
            float(self.state.get('vocal') or 0.0),
            float(self.state.get('dance') or 0.0),
            float(self.state.get('visual') or 0.0),
        )

        vote_ratio = fan_votes / vote_baseline
        param_ratio = current_param / parameter_baseline
        feasibility = min(vote_ratio, 1.15) * 0.45 + min(param_ratio, 1.25) * 0.55
        number = int(selected_row.get('number') or candidate.slot_index or 0)

        if feasibility >= 1.0:
            return min((feasibility - 1.0) * 0.25, 0.08)

        # 当前资源明显不够时，越高档的路线惩罚越重，避免模型无脑冒进。
        shortfall = 1.0 - feasibility
        return -min(shortfall * (0.10 + 0.04 * max(number - 1, 0)), 0.18)

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

    def _available_audition_choice_count(self) -> int:
        """返回当前考试前阶段可显式选择的试镜路线数量。"""

        count = 0
        for key, candidate in self.pre_audition_action_inventory.items():
            if not key.startswith('audition_select_'):
                continue
            if self._action_available(candidate):
                count += 1
        return count

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

    def _append_pre_audition_pbrs(
        self,
        *,
        reward: float,
        reward_breakdown: dict[str, float],
    ) -> float:
        """给考试前准备动作补上 PBRS 差分奖励。"""

        cfg = self._produce_reward_config()
        phi_before = self._prev_produce_phi
        phi_after = self._potential_value_produce(cfg)
        self._prev_produce_phi = phi_after
        if cfg.shape_scale > 0:
            reward_breakdown['pbrs_delta'] = cfg.shape_scale * (phi_after - phi_before)
            reward += reward_breakdown['pbrs_delta']
        return float(np.clip(reward * cfg.reward_scale, -cfg.reward_clip, cfg.reward_clip))

    def _first_star_retry_penalty(self, action_type: str) -> float:
        """初路线不应把连续再挑战视作常规主策略。"""

        if self.scenario.route_type != 'first_star':
            return 0.0
        if action_type != 'audition_retry':
            return 0.0
        return -0.45 if int(self.state.get('audition_index') or 0) == 0 else -0.20

    def _apply_pending_audition_outcome(self, result: dict[str, Any]) -> None:
        """把待接受考试结果中的路线进度和资源收益写回状态。"""

        if not bool(result.get('cleared')):
            return
        self.state['fan_votes'] += float(result.get('fan_vote_gain') or 0.0)
        self.state['deck_quality'] += float(result.get('deck_quality_gain') or 0.0)
        self.state['drink_quality'] += float(result.get('drink_quality_gain') or 0.0)
        parameter_bonus_by_type = result.get('parameter_bonus_by_type') or {}
        if isinstance(parameter_bonus_by_type, dict):
            for key, gain in parameter_bonus_by_type.items():
                if key in {'vocal', 'dance', 'visual'}:
                    self._gain_parameter(str(key), float(gain or 0.0))

    def _finalize_produce_reward(
        self,
        *,
        reward: float,
        terminated: bool,
    ) -> tuple[float, dict[str, float]]:
        """在接受考试结果时补算完整育成奖励。"""

        cfg = self._produce_reward_config()
        reward_breakdown = {
            'base_step_penalty': 0.0,
            'action_doc_bonus': 0.0,
            'state_delta_bonus': 0.0,
            'stage_repetition_penalty': 0.0,
            'idle_event_repeat_penalty': 0.0,
            'retry_penalty': 0.0,
            'pbrs_delta': 0.0,
            'terminal_reward': 0.0,
            'pp_left_penalty': 0.0,
        }
        phi_before = self._prev_produce_phi
        phi_after = self._potential_value_produce(cfg)
        self._prev_produce_phi = phi_after
        if cfg.shape_scale > 0:
            reward_breakdown['pbrs_delta'] = cfg.shape_scale * (phi_after - phi_before)
            reward += reward_breakdown['pbrs_delta']

        if terminated and self.final_summary:
            produce_result = self.final_summary.get('produce_result') or {}
            raw_score = float(produce_result.get('score') or 0.0)
            norm_score = math.log1p(max(raw_score, 0.0)) / math.log1p(max(cfg.score_norm_log_base, 1.0))
            grade = str(produce_result.get('rank') or '')
            grade_bonus_map = {
                'S4': cfg.terminal_grade_s4,
                'SSS+': cfg.terminal_grade_sss_plus,
                'SSS': cfg.terminal_grade_sss,
                'SS+': cfg.terminal_grade_ss_plus,
                'SS': cfg.terminal_grade_ss,
                'S+': cfg.terminal_grade_s_plus,
                'S': cfg.terminal_grade_s,
                'A+': cfg.terminal_grade_a,
                'A': cfg.terminal_grade_a,
                'B+': cfg.terminal_grade_b_plus,
                'B': cfg.terminal_grade_b,
                'C+': cfg.terminal_grade_c_plus,
                'C': cfg.terminal_grade_c,
                'D': cfg.terminal_grade_d,
                'failed': cfg.terminal_grade_failed,
            }
            grade_bonus = grade_bonus_map.get(grade, cfg.terminal_grade_c)
            route_clear = bool(self.final_summary.get('route_clear'))
            competitive_pass = bool(self.final_summary.get('competitive_pass'))
            competitive_top1_raw = self.final_summary.get('competitive_top1')
            competitive_top1 = bool(competitive_top1_raw)
            route_bonus = cfg.terminal_route_clear_bonus if route_clear else cfg.terminal_route_fail_penalty
            stage_progress = min(float(self.state.get('audition_index') or 0.0) / max(len(self.checkpoints), 1), 1.0)
            stage_progress_bonus = stage_progress * cfg.terminal_stage_progress_weight
            pp_left_penalty = min(float(self.state.get('produce_points') or 0.0) / max(cfg.pp_left_cap, 1.0), 1.0) * cfg.terminal_pp_left_waste_penalty
            reward_breakdown['pp_left_penalty'] = -pp_left_penalty

            fan_aux = 0.0
            if self.scenario.route_type == 'nia':
                current_fan = float(self.state.get('fan_votes') or 0.0)
                requirement = self._next_fan_vote_threshold()
                if requirement > 0:
                    overflow = max(current_fan - requirement, 0.0)
                    fan_aux = min(math.log1p(overflow / max(cfg.fan_overflow_scale, 1.0)) / math.log(max(cfg.fan_unlock_log_base, 1.01)), cfg.fan_overflow_cap) * cfg.terminal_fan_aux_scale
                else:
                    fan_aux = min(math.log1p(current_fan / max(cfg.fan_overflow_scale, 1.0)) / math.log(max(cfg.fan_full_unlock_log_base, 1.01)), cfg.fan_progress_cap) * cfg.terminal_fan_aux_scale

            nia_param_fallback = 0.0
            if self.scenario.route_type == 'nia' and str(produce_result.get('formula_source') or '').startswith('nia_external_formula'):
                detail = produce_result.get('formula_detail') or {}
                if detail.get('rating') is None:
                    nia_param_fallback = min(float(detail.get('param_rating') or 0.0) / 3000.0, 1.0) * cfg.terminal_nia_param_fallback_weight
                if detail.get('vote_rank') is not None:
                    nia_param_fallback += cfg.terminal_nia_vote_rank_bonus

            score_term = norm_score * cfg.terminal_score_scale
            if not route_clear:
                # 路线失败时，终局奖励只能保留很弱的排序信号，
                # 避免“高 produce rank 但路线失败”的策略反而被鼓励。
                grade_bonus = min(grade_bonus, cfg.terminal_grade_c)
                fan_aux = 0.0
                nia_param_fallback = 0.0
                failed_bonus_cap = max(abs(route_bonus), 1.0)
                score_term = min(score_term * FAILED_ROUTE_SCORE_SCALE, failed_bonus_cap * 0.35)
                stage_progress_bonus = min(stage_progress_bonus * 0.25, failed_bonus_cap * 0.15)
            elif self.scenario.route_type == 'nia' and competitive_top1_raw is not None and not competitive_top1:
                # NIA 现在以“每场考试都拿第一”为硬约束。
                # 即使路线 nominal clear，也要把非第一结果当作失败处理。
                route_bonus = min(route_bonus, 0.0) + cfg.terminal_route_fail_penalty
                grade_bonus = min(grade_bonus, cfg.terminal_grade_c)
                fan_aux = 0.0
                nia_param_fallback = 0.0
                score_term = min(score_term * FAILED_ROUTE_SCORE_SCALE, 0.75)
                stage_progress_bonus = min(stage_progress_bonus * 0.20, 0.20)
            elif self.scenario.route_type == 'first_star' and not competitive_pass:
                # “初” 路线前三外都视作失败；高 produce score 不能覆盖名次失败。
                route_bonus = min(route_bonus, 0.0) + cfg.terminal_route_fail_penalty
                grade_bonus = min(grade_bonus, cfg.terminal_grade_c)
                score_term = min(score_term * FAILED_ROUTE_SCORE_SCALE, 1.0)
                stage_progress_bonus = min(stage_progress_bonus * 0.35, 0.30)
            elif self.scenario.route_type == 'first_star' and not competitive_top1:
                # 前三但不是第一仍算通过，不过终局价值应明显低于冠军。
                route_bonus *= 0.55
                grade_bonus = min(grade_bonus, cfg.terminal_grade_b_plus)
                score_term *= 0.72

            terminal_reward = score_term + grade_bonus + route_bonus + stage_progress_bonus - pp_left_penalty
            if route_clear:
                terminal_reward += fan_aux + nia_param_fallback
            reward_breakdown['terminal_reward'] = float(np.clip(terminal_reward * cfg.reward_scale, -cfg.reward_clip, cfg.reward_clip))
            reward += reward_breakdown['terminal_reward']

        return float(np.clip(reward * cfg.reward_scale, -cfg.reward_clip, cfg.reward_clip)), reward_breakdown

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
        upgraded = self._lookup_card_upgrade_row(str(current_card.get('id') or ''), 1)
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
            accepted_reward = float(accepted_info.pop('accepted_reward', reward))
            return accepted_reward, terminated, accepted_info
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
        all_auditions_first = bool(self.audition_history) and all(
            int(item.get('rank') or 0) == 1
            for item in self.audition_history
        )
        competitive_top1 = False
        competitive_pass = False
        if self.scenario.route_type == 'nia':
            competitive_top1 = bool(cleared) and bool(all_auditions_first) and final_rank == 1
            competitive_pass = competitive_top1
        else:
            competitive_top1 = bool(cleared) and final_rank == 1
            competitive_pass = bool(cleared) and final_rank is not None and final_rank <= 3
        return {
            'route': route_label,
            'route_clear': bool(cleared),
            'competitive_pass': competitive_pass,
            'competitive_top1': competitive_top1,
            'all_auditions_first': all_auditions_first,
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
        self._reset_stage_action_counts()
        self.state['before_audition_refresh_applied'] = False
        self.state['last_exam_score'] = float(result.get('effective_score') or 0.0)
        accepted_result = {
            key: value
            for key, value in result.items()
            if key not in {'audition_slot', 'reward', 'deck_quality_gain', 'drink_quality_gain'}
        }
        self.audition_history.append(dict(accepted_result))
        self._apply_pending_audition_outcome(result)
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
        reward, reward_breakdown = self._finalize_produce_reward(
            reward=reward,
            terminated=terminated,
        )
        return terminated, {
            'pre_audition_phase': self.pre_audition_phase,
            f'audition_{audition_slot}': accepted_result,
            'accepted_reward': reward,
            'continue_remaining': int(self.state.get('continue_remaining') or 0),
            'final_summary': dict(self.final_summary) if self.final_summary else {},
            'reward_breakdown': reward_breakdown,
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
                if action_type == ACTION_PRE_AUDITION_CONTINUE:
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
            forced_refresh = next((candidate for candidate in candidates if candidate.action_type == ACTION_REFRESH), None)
            if forced_refresh is not None:
                forced_refresh.available = True
                forced_refresh.auto_skip = True
                self._candidates = [forced_refresh]
                return [forced_refresh]
        if self.pre_audition_phase == 'weekly' and not any(candidate.available for candidate in candidates):
            for candidate in candidates:
                if candidate.action_type == ACTION_REFRESH:
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
                    retry_penalty = self._first_star_retry_penalty(candidate.action_type)
                    return retry_penalty, False, {
                        'pre_audition_phase': self.pre_audition_phase,
                        f'audition_{int(self.pending_audition_result["audition_slot"])}': exam_info,
                        'continue_remaining': int(self.state.get('continue_remaining') or 0),
                        'reward_breakdown': {
                            'base_step_penalty': 0.0,
                            'action_doc_bonus': 0.0,
                            'state_delta_bonus': 0.0,
                            'stage_repetition_penalty': 0.0,
                            'idle_event_repeat_penalty': 0.0,
                            'retry_penalty': retry_penalty,
                            'pbrs_delta': 0.0,
                            'terminal_reward': 0.0,
                            'pp_left_penalty': 0.0,
                        },
                    }
                if candidate.action_type == 'audition_accept':
                    terminated, info = self._accept_pending_audition_result()
                    return float(info.pop('accepted_reward', 0.0)), terminated, info
                return -0.25, False, {'invalid_action': True}
            reward = -0.01
            reward_snapshot_before = self._produce_reward_snapshot()
            succeeded = True
            if candidate.action_type == ACTION_PRE_AUDITION_CONTINUE:
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
                reward += self._audition_selection_reward(candidate)
            self._trim_drinks()
            self._refresh_pre_audition_inventory()
            self._refresh_quality_scores()
            reward_breakdown = {
                'base_step_penalty': reward,
                'action_doc_bonus': 0.0,
                'state_delta_bonus': 0.0,
                'stage_repetition_penalty': 0.0,
                'idle_event_repeat_penalty': 0.0,
                'retry_penalty': 0.0,
                'pbrs_delta': 0.0,
                'terminal_reward': 0.0,
                'pp_left_penalty': 0.0,
            }
            state_delta_bonus = self._state_delta_bonus(reward_snapshot_before, self._produce_reward_snapshot())
            reward += state_delta_bonus
            reward_breakdown['state_delta_bonus'] = state_delta_bonus
            reward = self._append_pre_audition_pbrs(
                reward=reward,
                reward_breakdown=reward_breakdown,
            )
            return reward, False, {
                'action': candidate.label,
                'action_type': candidate.action_type,
                'success': succeeded,
                'pre_audition_phase': self.pre_audition_phase,
                'reward_breakdown': reward_breakdown,
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
        reward_snapshot_before = self._produce_reward_snapshot()
        fan_votes_before_action = float(self.state.get('fan_votes') or 0.0)
        should_apply_candidate_effects = self._should_apply_candidate_effects(candidate)
        phase_context = {
            'action_type': candidate.action_type,
            'source_row_id': candidate.source_row_id,
            'business_reward_kind': self._business_reward_kind(candidate.source_row_id),
        }
        if candidate.auto_skip and not should_apply_candidate_effects:
            phase_context['action_type'] = 'auto_skip'
        elif _is_lesson_action(candidate.action_type):
            self._dispatch_produce_item_phase('ProducePhaseType_StartLesson', **phase_context)
        elif candidate.action_type == ACTION_PRESENT:
            self._dispatch_produce_item_phase('ProducePhaseType_StartPresent', **phase_context)
        elif candidate.action_type == ACTION_SCHOOL_CLASS:
            self._dispatch_produce_item_phase('ProducePhaseType_StartPresent', **phase_context)
        elif candidate.action_type == ACTION_OUTING:
            self._dispatch_produce_item_phase('ProducePhaseType_StartRefresh', **phase_context)
        elif candidate.action_type == ACTION_ACTIVITY_SUPPLY:
            self._dispatch_produce_item_phase('ProducePhaseType_StartPresent', **phase_context)
        elif candidate.action_type == ACTION_REFRESH:
            self._dispatch_produce_item_phase('ProducePhaseType_StartRefresh', **phase_context)

        succeeded = True
        if should_apply_candidate_effects:
            self.state['stamina'] = float(np.clip(self.state['stamina'] + candidate.stamina_delta, 0.0, self.state['max_stamina']))
            self.state['produce_points'] += candidate.produce_point_delta * self._produce_point_rate(candidate.action_type)
            # 追込课 stat 延迟到 boost 判定后再分路应用（其他动作立即应用）
            is_hard_lesson = candidate.action_type in HARD_ACTION_TYPES
            if not is_hard_lesson:
                self._gain_parameter('vocal', candidate.stat_deltas[0])
                self._gain_parameter('dance', candidate.stat_deltas[1])
                self._gain_parameter('visual', candidate.stat_deltas[2])
            if candidate.produce_card_id:
                card_row = resolve_produce_card_row(
                    self.repository,
                    candidate.produce_card_id,
                    loadout=self.idol_loadout,
                    upgrade_count=int(candidate.resource_level or 0),
                )
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
            if candidate.action_type == ACTION_REFRESH and not candidate.auto_skip:
                self.state['refresh_used'] += 1
            if _is_lesson_action(candidate.action_type):
                self.state['lessons_taken'] = float(self.state.get('lessons_taken') or 0.0) + 1.0
                end_lesson_fired_item_ids: set[str] = set()
                before_present_fired_item_ids: set[str] = set()
                self._dispatch_produce_item_phase(
                    'ProducePhaseType_EndLesson',
                    **phase_context,
                    _fired_item_ids=end_lesson_fired_item_ids,
                )
                self._dispatch_produce_item_phase(
                    'ProducePhaseType_EndLessonBeforePresent',
                    **phase_context,
                    _fired_item_ids=before_present_fired_item_ids,
                )
                # 追込ボーナス成立時：ボーカル/ダンス/ビジュアルを条件とするPアイテムをすべて発動
                if is_hard_lesson and succeeded:
                    hard_stat = _lesson_stat_type(candidate.action_type)
                    for _stat in ('vocal', 'dance', 'visual'):
                        if _stat != hard_stat:
                            _extra_ctx = {**phase_context, 'action_type': f'lesson_{_stat}_hard'}
                            self._dispatch_produce_item_phase(
                                'ProducePhaseType_EndLesson',
                                **_extra_ctx,
                                _fired_item_ids=end_lesson_fired_item_ids,
                            )
                            self._dispatch_produce_item_phase(
                                'ProducePhaseType_EndLessonBeforePresent',
                                **_extra_ctx,
                                _fired_item_ids=before_present_fired_item_ids,
                            )
            elif candidate.action_type == ACTION_ACTIVITY:
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventActivity', **phase_context)
            elif candidate.action_type == ACTION_BUSINESS:
                business_vote_gain = max(float(self.state.get('fan_votes') or 0.0) - fan_votes_before_action, 0.0)
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventBusiness', **phase_context)
                # 企業活動（drink 类）：额外给 P 饮料（帮助页：スキルカード + Pドリンク）
                if self._business_reward_kind(candidate.source_row_id) == 'produce_drink':
                    self._grant_random_drink()
                # 大成功：按当前参数高低决定是否触发，只放大本次营业本身获得的 fan_votes。
                if self._business_big_success(candidate.source_row_id):
                    self.state['fan_votes'] += business_vote_gain * 0.50
            elif candidate.action_type == ACTION_PRESENT:
                if candidate.resource_type == 'ProduceResourceType_ProduceDrink':
                    self._grant_random_drink()
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventSchool', **phase_context)
                self._dispatch_produce_item_phase('ProducePhaseType_EndPresent', **phase_context)
            elif candidate.action_type == ACTION_SCHOOL_CLASS:
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventSchool', **phase_context)
            elif candidate.action_type == ACTION_OUTING:
                # 帮助页：外出概率附带 P 饮料
                if self.np_random.random() < 0.35:
                    self._grant_random_drink()
                self._dispatch_produce_item_phase('ProducePhaseType_EndPresent', **phase_context)
            elif candidate.action_type == ACTION_ACTIVITY_SUPPLY:
                # 活動支給：帮助页规定同时包含 P 饮料奖励。
                if candidate.resource_type == 'ProduceResourceType_ProduceDrink':
                    self._grant_random_drink()
                self._dispatch_produce_item_phase('ProducePhaseType_EndStepEventActivity', **phase_context)
                self._dispatch_produce_item_phase('ProducePhaseType_EndPresent', **phase_context)

            if candidate.action_type == ACTION_REFRESH and candidate.auto_skip:
                self.state['before_audition_refresh_applied'] = True

        self.state['step'] += 1
        self._trim_drinks()
        self._refresh_quality_scores()

        _reward_breakdown = {
            'base_step_penalty': reward,
            'action_doc_bonus': 0.0,
            'state_delta_bonus': 0.0,
            'stage_repetition_penalty': 0.0,
            'idle_event_repeat_penalty': 0.0,
            'retry_penalty': 0.0,
            'pbrs_delta': 0.0,
            'terminal_reward': 0.0,
            'pp_left_penalty': 0.0,
        }
        reward_snapshot_after = self._produce_reward_snapshot()
        state_delta_bonus = self._state_delta_bonus(reward_snapshot_before, reward_snapshot_after)
        reward += state_delta_bonus
        _reward_breakdown['state_delta_bonus'] = state_delta_bonus
        stage_repetition_penalty = self._stage_repetition_penalty(candidate, reward_snapshot_before, reward_snapshot_after)
        reward += stage_repetition_penalty
        _reward_breakdown['stage_repetition_penalty'] = stage_repetition_penalty
        idle_event_repeat_penalty = self._stage_idle_event_penalty(candidate, reward_snapshot_before, reward_snapshot_after)
        reward += idle_event_repeat_penalty
        _reward_breakdown['idle_event_repeat_penalty'] = idle_event_repeat_penalty
        retry_penalty = self._first_star_retry_penalty(candidate.action_type)
        reward += retry_penalty
        _reward_breakdown['retry_penalty'] = retry_penalty
        self._record_stage_action(candidate)
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
            self.pending_audition_result = {
                **dict(exam_info),
                'stage_type': stage_type,
                'reward': exam_reward,
                'audition_slot': int(self.state['audition_index']),
            }
            terminated, accepted_info = self._accept_pending_audition_result()
            reward += float(accepted_info.pop('accepted_reward', exam_reward))
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
                self.pending_audition_result = {
                    **dict(exam_info),
                    'stage_type': stage_type,
                    'reward': exam_reward,
                    'audition_slot': int(self.state['audition_index']),
                }
                terminated, accepted_info = self._accept_pending_audition_result()
                reward += float(accepted_info.pop('accepted_reward', exam_reward))
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

        reward, final_reward_breakdown = self._finalize_produce_reward(reward=reward, terminated=terminated)
        final_reward_breakdown['base_step_penalty'] = _reward_breakdown['base_step_penalty']
        final_reward_breakdown['action_doc_bonus'] = _reward_breakdown['action_doc_bonus']
        final_reward_breakdown['state_delta_bonus'] = _reward_breakdown['state_delta_bonus']
        final_reward_breakdown['stage_repetition_penalty'] = _reward_breakdown['stage_repetition_penalty']
        final_reward_breakdown['idle_event_repeat_penalty'] = _reward_breakdown['idle_event_repeat_penalty']
        final_reward_breakdown['retry_penalty'] = _reward_breakdown['retry_penalty']
        info['reward_breakdown'] = final_reward_breakdown
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
                    if not candidate.resource_id or not candidate.available:
                        return False
                    selected_stage_type = str(self.state.get('selected_audition_stage_type') or '')
                    selected_selector = str(self.state.get('selected_audition_selector') or '')
                    if selected_stage_type == str(self.pending_audition_stage or '') and selected_selector:
                        return False
                    is_final_stage = bool(self.pending_audition_stage and self.pending_audition_stage == str(self.scenario.audition_sequence[-1] or ''))
                    if is_final_stage and candidate.slot_index == 4:
                        return self._finale_available()
                    return True
                if candidate.action_type == ACTION_PRE_AUDITION_CONTINUE:
                    is_final_stage = bool(
                        self.pending_audition_stage
                        and self.pending_audition_stage == str(self.scenario.audition_sequence[-1] or '')
                    )
                    if not is_final_stage:
                        return True
                    return not self._has_pending_audition_choice() or self._available_audition_choice_count() <= 1
                return False
            return False
        if candidate.action_type in PRE_AUDITION_ACTION_TYPES:
            return False
        if self._is_first_star_pre_audition_hard_lesson_week() and candidate.action_type not in HARD_ACTION_TYPES:
            return False
        if self._is_first_star_pre_audition_refresh_week() and candidate.action_type != ACTION_REFRESH:
            return False
        if (
            self.scenario.produce_id == 'produce-001'
            and candidate.action_type in HARD_ACTION_TYPES
            and not self._is_first_star_pre_audition_hard_lesson_week()
        ):
            return False
        if candidate.action_type == ACTION_REFRESH:
            if self._is_refresh_locked_by_early_schedule():
                return False
            if not self._normal_refresh_has_stamina_value():
                return False
            # maxRefreshCount == 0 = 无次数限制（初路线）；>0 = 硬上限（NIA/レジェンド 为 4 次）
            max_refresh = self.scenario.max_refresh_count
            if max_refresh > 0:
                return self.state['refresh_used'] < max_refresh
            return True
        if candidate.action_type != ACTION_REFRESH and self.state['stamina'] <= 0.0:
            return False
        if candidate.action_type != ACTION_REFRESH and self.state['stamina'] + candidate.stamina_delta < 0.0:
            return False
        if candidate.produce_point_delta < 0.0 and self.state['produce_points'] + candidate.produce_point_delta < 0.0:
            return False
        return True

    def _build_action_samples(self) -> dict[str, list[dict[str, Any]]]:
        """预先按动作类型整理事件候选样本。"""

        samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        support_card_level_by_id = {
            str(item.support_card_id): int(item.support_card_level)
            for item in self.selected_support_cards
            if str(item.support_card_id or '')
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
                samples[ACTION_ACTIVITY].append(row)
                samples[ACTION_ACTIVITY_SUPPLY].append(row)
            elif event_type == 'ProduceEventType_Business':
                samples[ACTION_BUSINESS].append(row)
            elif event_type == 'ProduceEventType_School':
                # 授業は学校イベントのみ；外出・差入には混入させない
                samples[ACTION_SCHOOL_CLASS].append(row)
            elif event_type == 'ProduceEventType_Character':
                samples[ACTION_OUTING].append(row)
                samples[ACTION_PRESENT].append(row)
            elif event_type == 'ProduceEventType_SupportCard':
                if not unlocked_support_event_ids:
                    continue
                if str(row.get('id') or '') not in unlocked_support_event_ids:
                    continue
                # 支援卡事件只进差入（present），不混入授業 / 外出 / 活动支给
                samples[ACTION_PRESENT].append(row)
        return samples

    def _fallback_lesson_stat_deltas(self, action_type: str) -> tuple[float, float, float]:
        """返回课程主数据缺少直接收益时的参数增益 fallback。"""

        if not _is_lesson_action(action_type):
            return (0.0, 0.0, 0.0)
        profile_key = 'sp' if action_type in SP_ACTION_TYPES else 'normal'
        profile_value = max(float(self.repository.lesson_profile_stats.get(profile_key) or 0.0), 1.0)
        stage_scale = 1.0 + 0.06 * float(self.state.get('audition_index') or 0.0)
        gain = max(profile_value / 3.0, 1.0) * stage_scale
        stat_type = _lesson_stat_type(action_type)
        return {
            'vocal': (gain, 0.0, 0.0),
            'dance': (0.0, gain, 0.0),
            'visual': (0.0, 0.0, gain),
        }[stat_type]

    def _sample_action(self, action_type: str) -> ProduceActionCandidate:
        """为指定动作类型采样一条本周可执行动作。"""

        if action_type == ACTION_PRE_AUDITION_CONTINUE:
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
        if action_type == ACTION_REFRESH:
            is_pre_audition_refresh = self._is_first_star_pre_audition_refresh_week()
            recovery_setting_key = 'beforeAuditionRefreshStaminaRecoveryPermil' if is_pre_audition_refresh else 'refreshStaminaRecoveryPermil'
            recovery_permille = float(
                self.produce_setting.get(recovery_setting_key) or 0.0
            )
            if recovery_permille <= 0.0:
                raise ValueError(
                    'Refresh recovery setting missing from master database: '
                    f'produce_id={self.scenario.produce_id}, setting_key={recovery_setting_key}'
                )
            label = '考前恢复' if is_pre_audition_refresh else '休息'
            return ProduceActionCandidate(
                label=label,
                action_type=action_type,
                effect_types=ACTION_EFFECT_TYPES[action_type],
                produce_effect_ids=[],
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
            produce_card_level = 0
            resource_type = ''
            resource_id = ''
            effect_types = self._effect_types_for_ids(produce_effect_ids + success_effect_ids + fail_effect_ids)
            if not effect_types:
                effect_types = list(ACTION_EFFECT_TYPES.get(action_type, []))
            success_probability = float(row.get('successProbabilityPermyriad') or 10000) / 10000.0
            if action_type in SP_ACTION_TYPES:
                success_probability += self._sp_rate_bonus(action_type)
            success_probability = float(np.clip(success_probability, 0.05, 1.0))
            stat_deltas = (0.0, 0.0, 0.0)
            if action_type == ACTION_SCHOOL_CLASS:
                stamina_delta = min(stamina_delta, -8.0)
                stat_deltas = (
                    36.0 * (1.0 + float(self.state.get('vocal_growth') or 0.0)),
                    24.0 * (1.0 + float(self.state.get('dance_growth') or 0.0)),
                    24.0 * (1.0 + float(self.state.get('visual_growth') or 0.0)),
                )
            elif action_type == ACTION_OUTING:
                stamina_delta = max(stamina_delta, self.state['max_stamina'] * 0.35)
                produce_point_delta = -max(abs(produce_point_delta), 12.0)
                # 帮助页：外出可能获得 P饮料 / 技能卡强化删除变化（通过 produce_effect_ids 走）
            elif action_type == ACTION_ACTIVITY_SUPPLY:
                stamina_delta = 0.0
                # 帮助页：活動支給 同时获得技能卡、P点和P饮料，不是三选一。
                if not produce_card_id:
                    card_reward = self._business_action_bonus_card_reward('')
                    produce_card_id = card_reward.card_id
                    produce_card_level = card_reward.upgrade_count
                produce_point_delta = max(produce_point_delta, self._minimum_produce_point_reward())
                resource_type = 'ProduceResourceType_ProduceDrink'
            elif action_type == ACTION_BUSINESS:
                stamina_delta, produce_point_delta, reward_kind = self._business_action_profile(str(row.get('id') or ''))
                # 帮助页：4 类营业均包含 スキルカードの獲得 作为主收益之一
                card_reward = self._business_action_bonus_card_reward(str(row.get('id') or ''))
                produce_card_id = card_reward.card_id
                produce_card_level = card_reward.upgrade_count
            elif action_type == ACTION_PRESENT:
                # 帮助页：差入获得技能卡、P点和P饮料，并可能额外触发粉丝投票数奖励P点。
                if not produce_card_id:
                    card_reward = self._business_action_bonus_card_reward('')
                    produce_card_id = card_reward.card_id
                    produce_card_level = card_reward.upgrade_count
                produce_point_delta = max(produce_point_delta, self._minimum_produce_point_reward())
                resource_type = 'ProduceResourceType_ProduceDrink'
                if self._present_bonus_points_should_trigger():
                    produce_point_delta += self._present_bonus_produce_points()
            if _is_lesson_action(action_type) and not any(abs(value) > 1e-6 for value in stat_deltas) and not any(
                effect_type in PARAMETER_EFFECT_INDEX for effect_type in effect_types
            ):
                stat_deltas = self._fallback_lesson_stat_deltas(action_type)
                stamina_delta = min(stamina_delta, -5.0)
                produce_point_delta = max(produce_point_delta, self._minimum_produce_point_reward())
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
                resource_type=resource_type,
                resource_id=resource_id,
                resource_level=produce_card_level,
                success_probability=success_probability,
                stat_deltas=stat_deltas,
                source_row_id=str(row.get('id') or ''),
            )
        if action_type in HARD_ACTION_TYPES:
            lesson_profiles = self.repository.lesson_profile_stats
            normal_profile = float(lesson_profiles.get('normal') or 0.0)
            if normal_profile <= 0.0:
                raise ValueError('Normal lesson profile missing from master database')
            hard_profile = max(float(lesson_profiles.get('hard') or normal_profile), normal_profile)
            hard_scale = hard_profile / normal_profile
            stage_scale = 1.0 + 0.08 * float(self.state['audition_index'])
            # boost 时的全量增益（三参数均分）
            full_gain = 60.0 * hard_scale * stage_scale
            stamina_cost = 5.0 + 1.5 * hard_scale
            produce_point_delta = 2.0 + 1.2 * hard_scale
            # 从主数据 ProduceStepLessonLevel 读取 successThreshold/resultTargetValueLimit
            level_row = self._hard_lesson_level_row(action_type)
            success_threshold = float(level_row.get('successThreshold') or 0.0)
            result_limit = float(level_row.get('resultTargetValueLimit') or 0.0)
            if success_threshold <= 0.0 or result_limit <= 0.0:
                raise ValueError(
                    'Hard lesson threshold missing from master database: '
                    f'produce_id={self.scenario.produce_id}, action_type={action_type}'
                )
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
        produce_card_id = ''
        produce_card_level = 0
        resource_type = ''
        if action_type in SP_ACTION_TYPES:
            success_probability = float(np.clip(0.82 + self._sp_rate_bonus(action_type), 0.05, 1.0))
            stamina_delta = -8.0
            produce_point_delta = 4.0
        elif action_type in LESSON_ACTION_TYPES:
            success_probability = 0.92
            stamina_delta = -5.0
            produce_point_delta = 2.0
        elif action_type == ACTION_ACTIVITY:
            success_probability = 0.95
            stamina_delta = 1.0
            produce_point_delta = 6.0
        elif action_type == ACTION_BUSINESS:
            success_probability = 0.96
            stamina_delta, produce_point_delta, _ = self._business_action_profile('')
        elif action_type == ACTION_PRESENT:
            success_probability = 0.98
            produce_point_delta = self._minimum_produce_point_reward()
            card_reward = self._business_action_bonus_card_reward('')
            produce_card_id = card_reward.card_id
            produce_card_level = card_reward.upgrade_count
            resource_type = 'ProduceResourceType_ProduceDrink'
        elif action_type == ACTION_SCHOOL_CLASS:
            success_probability = 0.95
            stamina_delta = -8.0
            produce_point_delta = 0.0
        elif action_type == ACTION_OUTING:
            success_probability = 0.97
            stamina_delta = self.state['max_stamina'] * 0.35
            produce_point_delta = -12.0
        elif action_type == ACTION_ACTIVITY_SUPPLY:
            success_probability = 1.0
            stamina_delta = 0.0
            produce_point_delta = self._minimum_produce_point_reward()
            card_reward = self._business_action_bonus_card_reward('')
            produce_card_id = card_reward.card_id
            produce_card_level = card_reward.upgrade_count
            resource_type = 'ProduceResourceType_ProduceDrink'
        stat_deltas = (0.0, 0.0, 0.0)
        if action_type == ACTION_SCHOOL_CLASS:
            stat_deltas = tuple(self._apply_growth_rates((36.0, 24.0, 24.0)))
        return ProduceActionCandidate(
            label=self._action_label(action_type),
            action_type=action_type,
            effect_types=synthetic_types,
            produce_effect_ids=self._effect_ids_for_types(synthetic_types),
            stamina_delta=stamina_delta,
            produce_point_delta=produce_point_delta,
            produce_card_id=produce_card_id,
            resource_type=resource_type,
            resource_level=produce_card_level,
            success_probability=success_probability,
            stat_deltas=stat_deltas,
        )

    def _effect_ids_for_types(self, effect_types: list[str]) -> list[str]:
        """按效果类型随机抽取对应的 ProduceEffect 行。"""

        effect_ids: list[str] = []
        for effect_type in effect_types:
            candidates = self._produce_effect_ids_by_type.get(effect_type, [])
            if not candidates:
                continue
            effect_ids.append(str(candidates[int(self.np_random.integers(0, len(candidates)))]))
        return effect_ids

    def _effect_types_for_ids(self, effect_ids: list[str]) -> list[str]:
        """把效果 id 列表反解为效果类型集合。"""

        effect_types: set[str] = set()
        for effect_id in effect_ids:
            effect_row = self.produce_effects.first(str(effect_id))
            if effect_row and effect_row.get('produceEffectType'):
                effect_types.add(str(effect_row['produceEffectType']))
        return sorted(effect_types)

    def _effect_value_midpoint(self, effect: dict[str, Any]) -> float:
        """返回 ProduceEffect 数值区间中点，供候选收益预估使用。"""

        minimum = float(effect.get('effectValueMin') or 0.0)
        maximum = float(effect.get('effectValueMax') or minimum)
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        return (minimum + maximum) * 0.5

    def _estimate_effect_parameter_deltas(self, effect_ids: list[str], source_action_type: str) -> tuple[float, float, float]:
        """预估一组 ProduceEffect 对三维参数的直接增益。"""

        deltas = np.zeros(3, dtype=np.float32)
        event_action = source_action_type in EVENT_ACTION_TYPES
        for effect_id in effect_ids:
            effect_row = self.produce_effects.first(str(effect_id))
            if effect_row is None:
                continue
            effect_type = str(effect_row.get('produceEffectType') or '')
            stat_index = PARAMETER_EFFECT_INDEX.get(effect_type)
            if stat_index is None:
                continue
            gain = self._effect_value_midpoint(effect_row)
            gain *= 1.0 + float(self.state.get(PARAMETER_GROWTH_KEYS[stat_index]) or 0.0)
            if event_action:
                gain *= 1.0 + float(self.state.get('support_event_stat_bonus') or 0.0)
            deltas[stat_index] += gain
        return tuple(float(value) for value in deltas)

    def estimate_candidate_parameter_deltas(self, candidate: ProduceActionCandidate) -> tuple[float, float, float]:
        """预估候选动作执行后的期望三维参数增益。"""

        base_deltas = np.array(candidate.stat_deltas, dtype=np.float32)
        direct_deltas = np.array(
            self._estimate_effect_parameter_deltas(candidate.produce_effect_ids, candidate.action_type),
            dtype=np.float32,
        )
        success_deltas = np.array(
            self._estimate_effect_parameter_deltas(candidate.success_effect_ids, candidate.action_type),
            dtype=np.float32,
        )
        fail_deltas = np.array(
            self._estimate_effect_parameter_deltas(candidate.fail_effect_ids, candidate.action_type),
            dtype=np.float32,
        )
        success_probability = float(np.clip(candidate.success_probability, 0.0, 1.0))
        expected_deltas = base_deltas + direct_deltas + success_deltas * success_probability + fail_deltas * (1.0 - success_probability)
        return tuple(float(value) for value in expected_deltas)

    def estimate_candidate_param_progress(self, candidate: ProduceActionCandidate) -> ProduceParamProgress:
        """估算候选动作执行后三维参数势函数的分项进度。"""

        base_stats = np.array(
            [
                float(self.state.get('vocal') or 0.0),
                float(self.state.get('dance') or 0.0),
                float(self.state.get('visual') or 0.0),
            ],
            dtype=np.float32,
        )
        estimated_deltas = np.array(self.estimate_candidate_parameter_deltas(candidate), dtype=np.float32)
        boost_deltas = np.array(candidate.boost_stat_deltas, dtype=np.float32)
        projected_stats = base_stats + np.maximum(estimated_deltas, boost_deltas)
        projected_values = (
            self._clamp_parameter_value(float(projected_stats[0])),
            self._clamp_parameter_value(float(projected_stats[1])),
            self._clamp_parameter_value(float(projected_stats[2])),
        )
        return self._param_progress_for_stats(projected_values)

    def estimate_candidate_stamina_readiness(self, candidate: ProduceActionCandidate) -> float:
        """估算候选动作执行后对强制追込课的体力准备度。"""

        max_stamina = max(float(self.state.get('max_stamina') or 1.0), 1.0)
        projected_stamina = float(np.clip(float(self.state.get('stamina') or 0.0) + candidate.stamina_delta, 0.0, max_stamina))
        return self._stamina_readiness_phi_for(projected_stamina)

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
        if source_action_type == ACTION_ACTIVITY:
            rate += self.state['activity_produce_point_bonus']
        if source_action_type in EVENT_ACTION_TYPES:
            rate += self.state['support_event_point_bonus']
        if source_action_type == ACTION_PRESENT or _is_lesson_action(source_action_type):
            rate += self.state['lesson_present_point_bonus']
        return max(rate, 0.0)

    def _vote_rate(self, source_action_type: str) -> float:
        """计算营业类动作的粉丝票数倍率。"""

        rate = 1.0
        if source_action_type == ACTION_BUSINESS:
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
        if resource_type == 'ProduceResourceType_ProduceCard' and (source_action_type == ACTION_PRESENT or _is_lesson_action(source_action_type)):
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
                        upgraded_row = self._lookup_card_upgrade_row(str(card_row.get('id')), int(card_row.get('upgradeCount') or 0) + 1)
                        if upgraded_row is not None and int(upgraded_row.get('upgradeCount') or 0) <= 1:
                            card_row = dict(upgraded_row)
                    self.deck.append(card_row)
                    if str(card_row.get('rarity') or '') == 'ProduceCardRarity_Legend':
                        card_id = str(card_row.get('id') or '')
                        if card_id:
                            self.legend_seen_card_ids.add(card_id)
                    self._dispatch_produce_item_phase('ProducePhaseType_GetProduceCard', card=card_row)
        self._trim_drinks()

    def _grant_random_drink(self) -> bool:
        """从主数据可用 P 饮料池中随机授予一瓶 P 饮料。"""

        if len(self.drinks) >= max(self.scenario.drink_limit, 1):
            return False
        drink_candidates = self.repository.build_drink_inventory(
            self.scenario,
            max_items=self.scenario.drink_limit,
            rng=self.np_random,
            plan_type=self.idol_loadout.stat_profile.plan_type if self.idol_loadout is not None else None,
        )
        if not drink_candidates:
            return False
        selected_index = int(self.np_random.integers(0, len(drink_candidates)))
        self.drinks.append(dict(drink_candidates[selected_index]))
        self._dispatch_produce_item_phase('ProducePhaseType_GetProduceDrink')
        self._trim_drinks()
        return True

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
            upgraded = self._lookup_card_upgrade_row(str(card.get('id')), int(card.get('upgradeCount') or 0) + 1)
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
            upgraded_row = self._lookup_card_upgrade_row(str(replacement.get('id')), int(replacement.get('upgradeCount') or 0) + 1)
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

    def _lookup_card_upgrade_row(self, card_id: str, upgrade_count: int) -> dict[str, Any] | None:
        """按卡 id 和强化次数精确查找可强化后的主数据行。"""

        return self.repository.card_row_by_upgrade(card_id, upgrade_count, fallback_to_canonical=False)

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
            ACTION_ACTIVITY: '活动',
            ACTION_BUSINESS: '营业',
            ACTION_PRESENT: '差入/事件',
            ACTION_SCHOOL_CLASS: '授业',
            ACTION_OUTING: '外出',
            ACTION_ACTIVITY_SUPPLY: '活动支给',
            ACTION_REFRESH: '休息',
            ACTION_PRE_AUDITION_CONTINUE: '继续前进',
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

        card_scores = [
            self._card_prior_quality_value(str(card.get('id') or ''))
            for card in self.deck
        ]
        drink_scores = [len(self.repository.drink_exam_effect_types(drink)) for drink in self.drinks]
        enchant_bonus = 0.2 * len(self.exam_status_enchant_ids)
        proxy_quality = (float(np.mean(card_scores)) if card_scores else 0.0) + enchant_bonus
        next_profile = self._next_audition_profile()
        param_requirement = max(float(next_profile.get('parameter_baseline') or 0.0), 1.0)
        vote_requirement = max(float(next_profile.get('fan_vote_baseline') or next_profile.get('fan_vote_requirement') or 0.0), 1.0)
        current_param = max(
            float(self.state.get('vocal') or 0.0),
            float(self.state.get('dance') or 0.0),
            float(self.state.get('visual') or 0.0),
        )
        current_votes = max(float(self.state.get('fan_votes') or 0.0), 0.0)
        param_score = min(current_param / param_requirement, 1.25)
        vote_score = min(current_votes / vote_requirement, 1.25) if vote_requirement > 0 else 1.0
        static_feasibility = param_score * 0.6 + vote_score * 0.4
        self.state['deck_quality'] = proxy_quality + static_feasibility
        self.state['drink_quality'] = float(np.mean(drink_scores)) if drink_scores else 0.0

    def _card_prior_quality_value(self, card_id: str) -> float:
        """把客户端出牌估值换算到卡组质量使用的稳定尺度。"""

        raw_prior = float(self.repository.card_play_priors.get(card_id, 0.0))
        scaled_prior = raw_prior / CARD_PLAY_PRIOR_QUALITY_SCALE
        return float(np.clip(scaled_prior, CARD_PLAY_PRIOR_QUALITY_MIN, CARD_PLAY_PRIOR_QUALITY_MAX))

    def _audition_start_stamina(self) -> float:
        """按主数据的试验前回复量规则，计算考试开场体力。"""

        max_stamina = max(float(self.state.get('max_stamina') or 0.0), 1.0)
        current_stamina = float(np.clip(self.state.get('stamina') or 0.0, 0.0, max_stamina))
        if bool(self.state.get('before_audition_refresh_applied')):
            return current_stamina
        recovery_permille = float(self.produce_setting.get('beforeAuditionRefreshStaminaRecoveryPermil') or 0.0)
        recovery_multiple = max(0.0, 1.0 - float(self.state.get('before_audition_refresh_penalty') or 0.0))
        recovered = current_stamina + max_stamina * (recovery_permille / 1000.0) * recovery_multiple
        return float(min(recovered, max_stamina))

    def _challenge_lesson_perfect_bonus_ratio(self) -> float:
        """估算 challenge P 道具对 lesson PERFECT 上限的提升比例。"""

        if self.idol_loadout is None or self.scenario.produce_id not in {'produce-003', 'produce-006'}:
            return 0.0
        bonus_ratio = 0.0
        for item_id in self.idol_loadout.extra_produce_item_ids:
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
        for item_id in self.idol_loadout.extra_produce_item_ids:
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

        scored_actions = self._score_exam_actions(runtime)
        if not scored_actions:
            return None
        return max(scored_actions, key=lambda item: item[1])[0]

    def _score_exam_actions(self, runtime: ExamRuntime) -> list[tuple[ExamActionCandidate, float]]:
        """给当前考试可行动作打分。"""

        actions = runtime.legal_actions()
        if not actions:
            return []
        remaining_turns = max(runtime.max_turns - runtime.turn + 1, 1)
        playable_card_count = sum(1 for action in actions if action.kind == 'card')
        scored_actions: list[tuple[ExamActionCandidate, float]] = []

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

            try:
                preview_state = runtime.capture_preview_state()
                preview_reward, preview_info = runtime.step(
                    ExamActionCandidate(
                        label=action.label,
                        kind=action.kind,
                        payload=dict(action.payload),
                    )
                )
                score += float(preview_reward) * 2.0
                if runtime.terminated:
                    if runtime.battle_kind == 'lesson':
                        clear_state = str(preview_info.get('clear_state') or '')
                        if clear_state == 'perfect':
                            score += 4.0
                        elif clear_state == 'cleared':
                            score += 2.5
                    else:
                        score += float(runtime.score) / max(float(runtime.profile.get('base_score') or 1.0), 1.0)
                else:
                    score += (float(runtime.score) - float(preview_state.score)) / max(float(runtime.profile.get('base_score') or 1.0), 1.0)
                    score += (float(runtime.stamina) - float(preview_state.stamina)) * 0.02
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                # 前瞻失败时退回静态先验，避免影响训练稳定性。
                logger.debug('考试动作前瞻失败，使用静态先验: action=%s, reason=%s', action.label, exc)
            finally:
                if 'preview_state' in locals():
                    runtime.restore_preview_state(preview_state)
            scored_actions.append((action, score))
        return scored_actions

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
            parameter_stats=(
                float(self.state.get('vocal') or 0.0),
                float(self.state.get('dance') or 0.0),
                float(self.state.get('visual') or 0.0),
            ),
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
        runtime.base_score_bonus_multiplier *= max(1.0 + float(self.state.get('audition_parameter_bonus') or 0.0), 0.0)
        runtime.score_bonus_multiplier = runtime.base_score_bonus_multiplier
        if runtime.battle_kind == 'lesson' and runtime.lesson_perfect_value is not None:
            runtime.lesson_perfect_value *= 1.0 + self._challenge_lesson_perfect_bonus_ratio()
        runtime.reset()
        runtime.max_turns = max(1, runtime.max_turns + int(round(self.state['audition_turn_modifier'])))
        selector = self.exam_action_selectors.get(stage_type)
        for _ in range(256):
            action = selector.select_action(runtime) if selector is not None else None
            if action is None:
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
            parameter_baseline = float(profile.get('parameter_baseline') or 0.0)
            base_score = float(profile.get('base_score') or 0.0)
            if parameter_baseline <= 0.0 or base_score <= 0.0:
                raise ValueError(
                    'NIA audition parameter reward profile missing from master database: '
                    f'produce_id={self.scenario.produce_id}, stage_type={stage_type}'
                )
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
            'parameter_bonus_by_type': {str(key): float(value) for key, value in nia_param_gains.items()},
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
