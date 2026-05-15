"""打分效果的数值解析。"""

from __future__ import annotations

from ..ids import ExamEffect, GrowEffect
from .context import ExamEffectContext


def resolve_lesson_effect_value(context: ExamEffectContext, effect: dict[str, Any], from_card: bool = False) -> float:
    """按资源、检索数量和 stance 结算课程分数效果。"""

    effect_type = str(effect.get('effectType') or '')
    base_value = context.direct_value(effect)
    ratio_value = context.ratio_value(effect)
    if from_card and effect_type == ExamEffect.LESSON_DEPEND_REVIEW:
        ratio_value += context.current_card_ratio_bonus(GrowEffect.LESSON_DEPEND_REVIEW_ADD)
    elif from_card and effect_type == ExamEffect.LESSON_DEPEND_AGGRESSIVE:
        ratio_value += context.current_card_ratio_bonus(GrowEffect.LESSON_DEPEND_AGGRESSIVE_ADD)
    elif from_card and effect_type == ExamEffect.LESSON_DEPEND_BLOCK:
        ratio_value += context.current_card_ratio_bonus(GrowEffect.LESSON_DEPEND_BLOCK_ADD)
    search_count = context.search_count(str(effect.get('produceCardSearchId') or ''))
    if effect_type == ExamEffect.LESSON_FIX:
        value = base_value
    elif effect_type == ExamEffect.LESSON_DEPEND_REVIEW:
        value = context.ceil_positive(context.resources['review'] * ratio_value)
    elif effect_type == ExamEffect.LESSON_DEPEND_AGGRESSIVE:
        value = context.ceil_positive(context.resources['aggressive'] * ratio_value)
    elif effect_type == ExamEffect.LESSON_DEPEND_BLOCK:
        value = context.ceil_positive(context.resources['block'] * ratio_value)
    elif effect_type == ExamEffect.LESSON_DEPEND_PARAMETER_BUFF:
        value = context.ceil_positive(context.resources['parameter_buff'] * ratio_value)
    elif effect_type == ExamEffect.LESSON_DEPEND_PLAY_CARD_COUNT_SUM:
        value = context.total_counters['play_count'] * max(base_value, 1.0)
    elif effect_type == ExamEffect.LESSON_DEPEND_STAMINA_CONSUMPTION_SUM:
        value = context.ceil_positive(context.total_counters['stamina_spent'] * ratio_value)
    elif effect_type == ExamEffect.LESSON_DEPEND_BLOCK_CONSUMPTION_SUM:
        value = context.ceil_positive(context.total_counters['block_consumed'] * ratio_value)
    elif effect_type == ExamEffect.LESSON_DEPEND_BLOCK_AND_SEARCH_COUNT:
        extra_ratio = float(effect.get('effectValue2') or 0) / 1000.0
        value = context.compose_referenced_gain(
            base=search_count * max(base_value, 1.0),
            referenced=context.resources['block'] * extra_ratio,
        )
    elif effect_type == ExamEffect.LESSON_PER_SEARCH_COUNT:
        value = context.compose_referenced_gain(
            base=max(base_value, 1.0),
            referenced=search_count * (float(effect.get('effectValue2') or 0) / 1000.0),
        )
    elif effect_type == ExamEffect.LESSON_FULL_POWER_POINT:
        value = context.ceil_positive(context.resources['full_power_point'] * max(base_value, 1.0))
    elif effect_type in {ExamEffect.LESSON_ADD_MULTIPLE_LESSON_BUFF, ExamEffect.MULTIPLE_LESSON_BUFF_LESSON}:
        extra_ratio = float(effect.get('effectValue2') or effect.get('effectValue1') or 0) / 1000.0
        value = context.compose_referenced_gain(
            base=max(base_value, 1.0),
            referenced=context.resources['lesson_buff'] * extra_ratio,
        )
    elif effect_type == ExamEffect.LESSON_ADD_MULTIPLE_PARAMETER_BUFF:
        extra_ratio = float(effect.get('effectValue2') or 0) / 1000.0
        value = context.compose_referenced_gain(
            base=max(base_value, 1.0),
            referenced=context.resources['parameter_buff'] * extra_ratio,
        )
    else:
        value = base_value
    if from_card:
        value = context.adjust_direct_gain(
            value,
            add_grow_type=GrowEffect.LESSON_ADD,
            reduce_grow_type=GrowEffect.LESSON_REDUCE,
        )
    return max(context.apply_score_value_modifiers(value), 0.0)
