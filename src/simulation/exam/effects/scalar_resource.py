"""标量资源效果器。"""

from __future__ import annotations

from ..constants import SCALAR_RESOURCE_TYPES
from ..ids import ExamEffect, GrowEffect
from .context import ExamEffectContext


def apply_scalar_resource(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理好印象、集中以外的即时标量资源增长。"""

    effect_type = str(effect.get('effectType') or '')
    resource_key = SCALAR_RESOURCE_TYPES[effect_type]
    delta = context.direct_value(effect)
    if source == 'card' and effect_type == ExamEffect.CARD_PLAY_AGGRESSIVE:
        delta = context.adjust_direct_gain(delta, add_grow_type=GrowEffect.AGGRESSIVE_ADD)
    elif source == 'card' and effect_type == ExamEffect.REVIEW:
        delta = context.adjust_direct_gain(delta, add_grow_type=GrowEffect.REVIEW_ADD)
    elif source == 'card' and effect_type == ExamEffect.LESSON_BUFF:
        delta = context.adjust_direct_gain(delta, add_grow_type=GrowEffect.LESSON_BUFF_ADD)
    elif source == 'card' and effect_type == ExamEffect.FULL_POWER_POINT:
        delta = context.adjust_direct_gain(
            delta,
            add_grow_type=GrowEffect.FULL_POWER_POINT_ADD,
            reduce_grow_type=GrowEffect.FULL_POWER_POINT_REDUCE,
        )
    context.resources[resource_key] += delta
    if resource_key == 'full_power_point':
        context.total_counters['full_power_point_gained'] += context.positive_count(delta)
    context.dispatch_status_change(delta, [effect_type], origin=source)
