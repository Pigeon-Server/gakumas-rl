"""追加卡牌成长效果器。"""

from __future__ import annotations

from .context import ExamEffectContext


def apply_grow_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """给检索到的运行时卡牌追加成长效果。"""

    context.add_grow_effect(effect)
