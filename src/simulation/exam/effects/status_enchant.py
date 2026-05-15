"""状态附魔效果器。"""

from __future__ import annotations

from .context import ExamEffectContext


def apply_status_enchant(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """把考试效果转换成可触发的状态附魔实例。"""

    context.apply_status_enchant(effect, source)
