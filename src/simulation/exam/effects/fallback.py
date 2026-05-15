"""未精确分类效果的兜底效果器。"""

from __future__ import annotations

from .context import ExamEffectContext


def apply_fallback_timed_effect(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """保留旧逻辑：未知效果先按持续效果挂载，后续可独立拆文件。"""

    context.register_timed_effect(effect, source)
