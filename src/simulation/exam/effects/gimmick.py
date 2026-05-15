"""场地 gimmick 效果器。"""

from __future__ import annotations

from ..ids import ExamEffect
from .context import ExamEffectContext


GIMMICK_EFFECT_TYPES = {
    ExamEffect.GIMMICK_PLAY_CARD_LIMIT,
    ExamEffect.GIMMICK_START_TURN_CARD_DRAW_DOWN,
    ExamEffect.GIMMICK_SLEEPY,
    ExamEffect.GIMMICK_SLUMP,
    ExamEffect.GIMMICK_PARAMETER_DEBUFF,
    ExamEffect.GIMMICK_LESSON_DEBUFF,
    ExamEffect.PANIC,
}


def apply_gimmick_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理场地负面效果，并优先消费防弱化层数。"""

    effect_type = str(effect.get('effectType') or '')
    if context.consume_anti_debuff(effect_type):
        return
    if effect_type == ExamEffect.GIMMICK_PLAY_CARD_LIMIT:
        search_id = str(effect.get('produceCardSearchId') or '')
        if not search_id:
            return
        context.forbidden_card_search_ids[search_id] += max(context.positive_count(context.count_value(effect)), 1)
        context.sync_forbidden_search_resources()
    elif effect_type == ExamEffect.GIMMICK_START_TURN_CARD_DRAW_DOWN:
        context.start_turn_draw_penalty += int(round(context.raw_value(effect)))
    elif effect_type == ExamEffect.GIMMICK_SLEEPY:
        context.resources['sleepy'] += context.raw_value(effect)
    else:
        context.register_timed_effect(effect, source)
