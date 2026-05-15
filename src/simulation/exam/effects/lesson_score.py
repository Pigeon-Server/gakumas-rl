"""打分效果器。"""

from __future__ import annotations

from .context import ExamEffectContext


def apply_lesson_score_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """把打分类效果转成得分并同步回合颜色统计。"""

    lesson_value = context.resolve_lesson_effect_value(effect, from_card=source == 'card')
    lesson_delta = context.score_gain(lesson_value)
    context.score += lesson_delta
    if context.current_turn_color in context.score_per_color:
        context.score_per_color[context.current_turn_color] += lesson_delta
    context.turn_counters['lesson_plays'] += 1
    context.total_counters['lesson_plays'] += 1
    context.update_clear_state_after_score_change()
