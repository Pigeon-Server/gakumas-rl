"""训练与模拟共享的偶像配装数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IdolStatProfile:
    """偶像卡在培育与考试中的基础属性快照。"""

    idol_card_id: str
    character_id: str
    plan_type: str
    exam_effect_type: str
    initial_exam_deck_id: str
    audition_difficulty_id: str
    unique_produce_card_id: str
    vocal: float
    dance: float
    visual: float
    vocal_growth_rate: float
    dance_growth_rate: float
    visual_growth_rate: float
    stamina: float


@dataclass(frozen=True)
class ProduceSkillEffect:
    """偶像卡随 rank 解锁的培育技能效果。"""

    skill_id: str
    level: int
    trigger_id: str
    effect_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProduceCardConversionSpec:
    """技能卡切换的单条生效配置。"""

    before_card_id: str
    after_card_id: str
    condition_set_id: str = ''
    is_not_reward: bool = False


@dataclass(frozen=True)
class ExamStatusEnchantSpec:
    """考试开场挂载的附魔规格，保留持续回合与触发次数。"""

    enchant_id: str
    effect_turn: int | None = None
    effect_count: int | None = None
    source_identity: str = ''


@dataclass(frozen=True)
class DeckArchetype:
    """主数据里为偶像卡推荐的卡组流派信息。"""

    group_id: str
    sample_group_id: str
    description: str
    recommended_card_ids: tuple[str, ...] = ()
    sample_card_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExamEpisodeRandomizationConfig:
    """考试环境的 episode 级上下文随机化配置。"""

    enabled: bool = False
    stat_jitter_ratio: float = 0.10
    score_bonus_jitter_ratio: float = 0.05
    randomize_use_after_item: bool = False
    randomize_stage_type: bool = False


@dataclass(frozen=True)
class SupportCardSelection:
    """自动编成器输出的一张支援卡及其评分说明。"""

    support_card_id: str
    support_card_level: int
    support_card_type: str
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class IdolLoadout:
    """训练、模拟与 API 共享的统一偶像配装对象。"""

    idol_card_id: str
    producer_level: int
    idol_rank: int
    dearness_level: int
    use_after_item: bool
    stat_profile: IdolStatProfile
    deck_archetype: DeckArchetype | None = None
    produce_skills: tuple[ProduceSkillEffect, ...] = ()
    produce_card_conversions: tuple[ProduceCardConversionSpec, ...] = ()
    produce_item_id: str = ''
    extra_produce_item_ids: tuple[str, ...] = ()
    support_cards: tuple[SupportCardSelection, ...] = ()
    exam_status_enchant_ids: tuple[str, ...] = ()
    exam_status_enchant_specs: tuple[ExamStatusEnchantSpec, ...] = ()
    exam_score_bonus_multiplier: float = 1.0
    assist_mode: bool = False
    metadata: dict[str, str | int | float] = field(default_factory=dict)
