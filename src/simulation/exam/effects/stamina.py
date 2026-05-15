"""体力效果器。"""

from __future__ import annotations

from ..ids import ExamEffect, ExamPhase
from .context import ExamEffectContext


STAMINA_EFFECT_TYPES = {
    ExamEffect.STAMINA_DAMAGE,
    ExamEffect.STAMINA_REDUCE,
    ExamEffect.STAMINA_REDUCE_FIX,
    ExamEffect.STAMINA_RECOVER,
    ExamEffect.STAMINA_RECOVER_FIX,
    ExamEffect.STAMINA_RECOVER_MULTIPLE,
}


def apply_stamina_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理体力扣减与回复。"""

    effect_type = str(effect.get('effectType') or '')
    if effect_type in {
        ExamEffect.STAMINA_DAMAGE,
        ExamEffect.STAMINA_REDUCE,
        ExamEffect.STAMINA_REDUCE_FIX,
    }:
        context.spend_stamina(
            context.direct_value(effect),
            phase_type=ExamPhase.STAMINA_REDUCE,
            status_change_origin=source,
        )
        return
    if context.has_timed_effect(ExamEffect.STAMINA_RECOVER_RESTRICTION):
        return
    if effect_type in {ExamEffect.STAMINA_RECOVER, ExamEffect.STAMINA_RECOVER_FIX}:
        context.stamina = min(context.max_stamina, context.stamina + context.direct_value(effect))
    elif effect_type == ExamEffect.STAMINA_RECOVER_MULTIPLE:
        context.stamina = min(context.max_stamina, context.stamina + context.max_stamina * context.ratio_value(effect))
