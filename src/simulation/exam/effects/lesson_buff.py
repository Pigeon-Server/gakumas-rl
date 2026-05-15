"""打分强化效果器。"""

from __future__ import annotations

from ..ids import ExamEffect, GrowEffect
from .context import ExamEffectContext


LESSON_BUFF_EFFECT_TYPES = {
    ExamEffect.LESSON_BUFF_REDUCE,
    ExamEffect.LESSON_BUFF_DEPEND_PARAMETER_BUFF,
    ExamEffect.LESSON_BUFF_PER_SEARCH_COUNT,
}


def apply_lesson_buff_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理打分强化的增减和依赖收益。"""

    effect_type = str(effect.get('effectType') or '')
    if effect_type == ExamEffect.LESSON_BUFF_REDUCE:
        context.resources['lesson_buff'] = max(context.resources['lesson_buff'] - context.direct_value(effect), 0.0)
        return
    if effect_type == ExamEffect.LESSON_BUFF_DEPEND_PARAMETER_BUFF:
        delta = context.ceil_positive(context.resources['parameter_buff'] * context.ratio_value(effect))
    elif effect_type == ExamEffect.LESSON_BUFF_PER_SEARCH_COUNT:
        search_count = context.search_count(str(effect.get('produceCardSearchId') or ''))
        delta = context.ceil_positive(search_count * (float(effect.get('effectValue2') or effect.get('effectValue1') or 0) / 1000.0))
    else:
        return
    if source == 'card':
        delta = context.adjust_direct_gain(delta, add_grow_type=GrowEffect.LESSON_BUFF_ADD)
    context.resources['lesson_buff'] += delta
    context.dispatch_status_change(delta, [effect_type], origin=source)
