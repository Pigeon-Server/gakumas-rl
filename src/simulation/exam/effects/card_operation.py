"""卡牌操作效果器。"""

from __future__ import annotations

from ..ids import ExamEffect
from .context import ExamEffectContext


CARD_OPERATION_EFFECT_TYPES = {
    ExamEffect.CARD_CREATE_ID,
    ExamEffect.CARD_CREATE_SEARCH,
    ExamEffect.CARD_DUPLICATE,
    ExamEffect.CARD_MOVE,
    ExamEffect.CARD_UPGRADE,
    ExamEffect.FORCE_PLAY_CARD_SEARCH,
}


def apply_card_operation(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """执行造卡、复制、移动、升级、强制打出等卡牌操作。"""

    context.apply_card_operation(effect)
