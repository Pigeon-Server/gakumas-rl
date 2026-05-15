"""考试触发器统一编排器。"""

from __future__ import annotations

from typing import Any

from .card_search import trigger_card_search_matches
from .context import ExamTriggerContext
from .effect_types import effect_types_match
from .field_status import trigger_field_status_matches
from .lesson_type import lesson_type_matches
from .phase import phase_matches
from .phase_value import phase_value_matches
from .status_change_origin import status_change_origin_matches


def trigger_matches(
    context: ExamTriggerContext,
    trigger: dict[str, Any],
    event: dict[str, Any],
    acting_card: Any | None = None,
    target_card: Any | None = None,
) -> bool:
    """按主数据字段逐段判断触发器是否命中当前 phase 事件。"""

    return (
        phase_matches(trigger, event)
        and status_change_origin_matches(event)
        and phase_value_matches(trigger, event)
        and lesson_type_matches(context, trigger)
        and effect_types_match(trigger, event)
        and trigger_field_status_matches(context, trigger)
        and trigger_card_search_matches(context, trigger, acting_card=acting_card, target_card=target_card)
    )
