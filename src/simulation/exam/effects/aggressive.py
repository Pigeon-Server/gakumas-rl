"""やる気效果器。"""

from __future__ import annotations

from ..ids import ExamEffect
from .context import ExamEffectContext


AGGRESSIVE_EFFECT_TYPES = {
    ExamEffect.AGGRESSIVE_REDUCE,
}


def apply_aggressive_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理やる気资源的直接扣减。"""

    context.resources['aggressive'] = max(context.resources['aggressive'] - context.direct_value(effect), 0.0)
