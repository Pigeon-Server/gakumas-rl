"""元气/护盾效果器。"""

from __future__ import annotations

from ..ids import ExamEffect, GrowEffect
from .context import ExamEffectContext


BLOCK_EFFECT_TYPES = {
    ExamEffect.BLOCK,
    ExamEffect.BLOCK_FIX,
    ExamEffect.BLOCK_DEPEND_BLOCK_CONSUMPTION_SUM,
    ExamEffect.BLOCK_ADD_MULTIPLE_AGGRESSIVE,
    ExamEffect.BLOCK_DOWN,
    ExamEffect.BLOCK_PER_USE_CARD_COUNT,
    ExamEffect.BLOCK_DEPEND_REVIEW,
}


def apply_block_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理元气获得、倍率变化和依赖资源的元气收益。"""

    effect_type = str(effect.get('effectType') or '')
    if effect_type in {ExamEffect.BLOCK, ExamEffect.BLOCK_FIX}:
        delta = context.raw_value(effect)
    elif effect_type == ExamEffect.BLOCK_DEPEND_BLOCK_CONSUMPTION_SUM:
        delta = context.ceil_positive(context.total_counters['block_consumed'] * context.ratio_value(effect))
    elif effect_type == ExamEffect.BLOCK_ADD_MULTIPLE_AGGRESSIVE:
        delta = context.compose_referenced_gain(
            base=max(context.raw_value(effect), 0.0),
            referenced=context.resources['aggressive'] * (float(effect.get('effectValue2') or 0) / 1000.0),
        )
    elif effect_type == ExamEffect.BLOCK_DOWN:
        context.resources['block'] *= max(1.0 - context.ratio_value(effect), 0.0)
        return
    elif effect_type == ExamEffect.BLOCK_PER_USE_CARD_COUNT:
        delta = context.turn_counters['play_count'] * context.raw_value(effect)
    elif effect_type == ExamEffect.BLOCK_DEPEND_REVIEW:
        delta = context.ceil_positive(context.resources['review'] * context.ratio_value(effect))
    else:
        return

    if source == 'card':
        delta = context.adjust_direct_gain(
            delta,
            add_grow_type=GrowEffect.BLOCK_ADD,
            reduce_grow_type=GrowEffect.BLOCK_REDUCE,
        )
    context.gain_block(delta, effect_type=effect_type, status_change_origin=source)
