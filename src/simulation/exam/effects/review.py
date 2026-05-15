"""好印象效果器。"""

from __future__ import annotations

from ..ids import ExamEffect, GrowEffect
from .context import ExamEffectContext


REVIEW_EFFECT_TYPES = {
    ExamEffect.REVIEW_ADDITIVE,
    ExamEffect.REVIEW_DEPEND_AGGRESSIVE,
    ExamEffect.REVIEW_DEPEND_BLOCK,
    ExamEffect.REVIEW_PER_SEARCH_COUNT,
    ExamEffect.REVIEW_REDUCE,
    ExamEffect.REVIEW_VALUE_MULTIPLE,
}


def apply_review_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理好印象的增减、倍率和依赖收益。"""

    effect_type = str(effect.get('effectType') or '')
    if effect_type == ExamEffect.REVIEW_ADDITIVE:
        delta = context.direct_value(effect)
    elif effect_type == ExamEffect.REVIEW_DEPEND_AGGRESSIVE:
        delta = context.ceil_positive(context.resources['aggressive'] * context.ratio_value(effect))
    elif effect_type == ExamEffect.REVIEW_DEPEND_BLOCK:
        delta = context.ceil_positive(context.resources['block'] * context.ratio_value(effect))
    elif effect_type == ExamEffect.REVIEW_PER_SEARCH_COUNT:
        search_count = context.search_count(str(effect.get('produceCardSearchId') or ''))
        delta = context.ceil_positive(search_count * (float(effect.get('effectValue2') or effect.get('effectValue1') or 0) / 1000.0))
    elif effect_type == ExamEffect.REVIEW_REDUCE:
        context.resources['review'] = max(context.resources['review'] - context.direct_value(effect), 0.0)
        return
    elif effect_type == ExamEffect.REVIEW_VALUE_MULTIPLE:
        context.resources['review'] *= 1.0 + context.ratio_value(effect)
        return
    else:
        return

    if source == 'card':
        delta = context.adjust_direct_gain(delta, add_grow_type=GrowEffect.REVIEW_ADD)
    context.resources['review'] += delta
    if effect_type == ExamEffect.REVIEW_DEPEND_BLOCK and float(effect.get('effectValue2') or 0) >= 1000:
        context.resources['block'] = 0.0
    context.dispatch_status_change(delta, [effect_type], origin=source)
