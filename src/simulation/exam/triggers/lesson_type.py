"""课程类型触发器。"""

from __future__ import annotations

from typing import Any

from ..ids import LessonType
from .context import ExamTriggerContext


def lesson_type_matches(context: ExamTriggerContext, trigger: dict[str, Any]) -> bool:
    """判断触发器限定的 lesson 类型是否覆盖当前战斗上下文。"""

    lesson_type = str(trigger.get('lessonType') or LessonType.UNKNOWN)
    return lesson_type == LessonType.UNKNOWN or lesson_type in set(context.current_lesson_types())
