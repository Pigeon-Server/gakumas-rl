"""考试效果器入口。"""

from .lesson_value import resolve_lesson_effect_value
from .registry import EXAM_EFFECT_REGISTRY, apply_exam_effect

__all__ = [
    'EXAM_EFFECT_REGISTRY',
    'apply_exam_effect',
    'resolve_lesson_effect_value',
]

