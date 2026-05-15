"""持续类考试效果器。"""

from __future__ import annotations

from typing import Any

from ..ids import ExamEffect, GrowEffect
from .context import ExamEffectContext


TIMED_EFFECT_TYPES = {
    ExamEffect.AGGRESSIVE_ADDITIVE,
    ExamEffect.AGGRESSIVE_VALUE_MULTIPLE,
    ExamEffect.BLOCK_ADD_DOWN,
    ExamEffect.BLOCK_RESTRICTION,
    ExamEffect.BLOCK_VALUE_MULTIPLE,
    ExamEffect.CARD_SEARCH_EFFECT_PLAY_COUNT_BUFF,
    ExamEffect.ENTHUSIASTIC_ADDITIVE,
    ExamEffect.ENTHUSIASTIC_MULTIPLE,
    ExamEffect.FULL_POWER_POINT_ADDITIVE,
    ExamEffect.LESSON_BUFF_ADDITIVE,
    ExamEffect.LESSON_BUFF_MULTIPLE,
    ExamEffect.LESSON_VALUE_MULTIPLE,
    ExamEffect.LESSON_VALUE_MULTIPLE_DEPEND_REVIEW_OR_AGGRESSIVE,
    ExamEffect.LESSON_VALUE_MULTIPLE_DOWN,
    ExamEffect.PARAMETER_BUFF_ADDITIVE,
    ExamEffect.PARAMETER_BUFF_MULTIPLE_PER_TURN,
    ExamEffect.PLAYABLE_VALUE_ADD,
    ExamEffect.REVIEW_COUNT_ADD,
    ExamEffect.REVIEW_MULTIPLE,
    ExamEffect.SEARCH_PLAY_CARD_STAMINA_CONSUMPTION_CHANGE,
    ExamEffect.STANCE_LOCK,
    ExamEffect.STAMINA_CONSUMPTION_ADD,
    ExamEffect.STAMINA_CONSUMPTION_ADD_FIX,
    ExamEffect.STAMINA_CONSUMPTION_DOWN,
    ExamEffect.STAMINA_CONSUMPTION_DOWN_FIX,
    ExamEffect.STAMINA_RECOVER_RESTRICTION,
}


def apply_timed_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """注册一个持续效果，并处理卡牌成长带来的额外层数或回合数。"""

    effect_type = str(effect.get('effectType') or '')
    if context.consume_anti_debuff(effect_type):
        return
    register_times = 1
    effect_row = dict(effect)
    if source == 'card' and effect_type == ExamEffect.PARAMETER_BUFF_MULTIPLE_PER_TURN:
        # 计数型：注册次数，整数语义
        register_times += int(round(context.current_card_grow_total(GrowEffect.PARAMETER_BUFF_MULTIPLE_PER_TURN_ADD)))
    if source == 'card' and effect_type == ExamEffect.STAMINA_CONSUMPTION_DOWN:
        # 计数型：持续回合数，整数语义
        effect_row['effectTurn'] = int(effect.get('effectTurn') or 0) + int(
            round(context.current_card_grow_total(GrowEffect.STAMINA_CONSUMPTION_DOWN_TURN_ADD))
        )
    for _ in range(max(register_times, 1)):
        context.register_timed_effect(dict(effect_row), source)
