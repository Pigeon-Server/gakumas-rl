"""参数强化效果器。"""

from __future__ import annotations

from ..ids import ExamEffect, GrowEffect
from .context import ExamEffectContext


PARAMETER_BUFF_EFFECT_TYPES = {
    ExamEffect.PARAMETER_BUFF_REDUCE,
    ExamEffect.PARAMETER_BUFF_MULTIPLE_PER_TURN_REDUCE,
    ExamEffect.PARAMETER_BUFF_DEPEND_LESSON_BUFF,
}


def apply_parameter_buff_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理参数强化消耗和依赖收益。"""

    effect_type = str(effect.get('effectType') or '')
    if effect_type == ExamEffect.PARAMETER_BUFF_REDUCE:
        context.resources['parameter_buff'] = max(context.resources['parameter_buff'] - context.direct_value(effect), 0.0)
        return
    if effect_type == ExamEffect.PARAMETER_BUFF_MULTIPLE_PER_TURN_REDUCE:
        context.consume_parameter_buff_multiple(context.direct_value(effect))
        return
    if effect_type == ExamEffect.PARAMETER_BUFF_DEPEND_LESSON_BUFF:
        delta = context.ceil_positive(context.resources['lesson_buff'] * context.ratio_value(effect))
        if source == 'card':
            delta = context.adjust_direct_gain(delta, add_grow_type=GrowEffect.PARAMETER_BUFF_TURN_ADD)
        context.resources['parameter_buff'] += delta
        context.dispatch_status_change(delta, [effect_type], origin=source)
