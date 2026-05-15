"""延迟链式效果器。"""

from __future__ import annotations

from .context import ExamEffectContext


def apply_delayed_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """登记延迟触发的链式考试效果。"""

    context.schedule_effect(effect)
