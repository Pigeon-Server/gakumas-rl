"""简单状态切换效果器。"""

from __future__ import annotations

from typing import Any

from ..ids import ExamEffect
from .context import ExamEffectContext


SIMPLE_EFFECT_TYPES = {
    ExamEffect.CONCENTRATION,
    ExamEffect.PRESERVATION,
    ExamEffect.OVER_PRESERVATION,
    ExamEffect.CARD_DRAW,
    ExamEffect.FULL_POWER,
    ExamEffect.STANCE_RESET,
    ExamEffect.EXTRA_TURN,
    ExamEffect.ANTI_DEBUFF,
    ExamEffect.DEBUFF_RECOVER,
}

EXTRA_SIMPLE_EFFECT_TYPES = {
    ExamEffect.FULL_POWER_POINT_REDUCE,
    ExamEffect.HAND_GRAVE_COUNT_CARD_DRAW,
}


def apply_simple_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理不需要复杂派生数值的状态类效果。"""

    effect_type = str(effect.get('effectType') or '')
    if effect_type == ExamEffect.CONCENTRATION:
        context.enter_concentration(int(max(context.raw_value(effect), 1.0)))
    elif effect_type == ExamEffect.PRESERVATION:
        context.enter_preservation(int(max(context.raw_value(effect), 1.0)))
    elif effect_type == ExamEffect.OVER_PRESERVATION:
        context.enter_preservation(3)
    elif effect_type == ExamEffect.CARD_DRAW:
        context.draw(int(max(1.0, round(context.direct_value(effect)))))
    elif effect_type == ExamEffect.FULL_POWER:
        context.enter_full_power()
    elif effect_type == ExamEffect.STANCE_RESET:
        context.reset_stance()
    elif effect_type == ExamEffect.EXTRA_TURN:
        context.extra_turns += 1
    elif effect_type == ExamEffect.ANTI_DEBUFF:
        context.resources['anti_debuff'] += context.count_value(effect)
    elif effect_type == ExamEffect.DEBUFF_RECOVER:
        context.clear_negative_effects()


def apply_extra_simple_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理少量后置简单效果。"""

    effect_type = str(effect.get('effectType') or '')
    if effect_type == ExamEffect.FULL_POWER_POINT_REDUCE:
        context.resources['full_power_point'] = max(context.resources['full_power_point'] - context.direct_value(effect), 0.0)
    elif effect_type == ExamEffect.HAND_GRAVE_COUNT_CARD_DRAW:
        context.draw(len(context.hand) + len(context.grave))
