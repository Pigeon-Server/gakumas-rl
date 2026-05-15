"""场地状态触发器。"""

from __future__ import annotations

from typing import Any

from ..ids import FieldStatus, TriggerCheck
from .context import ExamTriggerContext


def trigger_field_status_matches(context: ExamTriggerContext, trigger: dict[str, Any]) -> bool:
    """检查触发器里的场地状态条件是否成立。"""

    status_types = [str(value) for value in trigger.get('fieldStatusTypes', []) if value]
    status_values = [float(value) for value in trigger.get('fieldStatusValues', []) if value not in (None, '')]
    check_types = [str(value) for value in trigger.get('fieldStatusCheckTypes', []) if value]
    search_ids = [str(value) for value in trigger.get('fieldStatusProduceCardSearchIds', []) if value]

    for index, status_type in enumerate(status_types):
        current_value = field_status_value(context, status_type, search_ids[index] if index < len(search_ids) else '')
        expected_value = status_values[index] if index < len(status_values) else 1.0
        check_type = check_types[index] if index < len(check_types) else TriggerCheck.UNKNOWN
        reverse_threshold = status_type.endswith('MultipleDown') or 'LessMultiple' in status_type
        if check_type == TriggerCheck.NOT:
            if reverse_threshold:
                if current_value <= expected_value:
                    return False
            elif current_value >= expected_value:
                return False
        else:
            if reverse_threshold:
                if current_value > expected_value:
                    return False
            elif current_value < expected_value:
                return False
    return True


def field_status_value(context: ExamTriggerContext, field_status_type: str, search_id: str) -> float:
    """读取指定场地状态类型在当前战斗中的数值。"""

    stamina_ratio = float(context.stamina / max(context.max_stamina, 1.0)) * 1000.0
    lesson_progress = float(context.score / max(context.profile.get('base_score') or 1.0, 1.0)) * 1000.0
    mapping = {
        FieldStatus.BLOCK_UP: context.resources['block'],
        FieldStatus.CARD_PLAY_AGGRESSIVE_UP: context.resources['aggressive'],
        FieldStatus.CARD_SEARCH_COUNT_UP: float(context.search_count(search_id)),
        FieldStatus.CONDITION_THRESHOLD_MULTIPLE_DOWN: lesson_progress,
        FieldStatus.CONCENTRATION_CHANGE_COUNT_UP: float(context.total_counters['stance_concentration']),
        FieldStatus.CONCENTRATION_UP: context.resources['concentration'],
        FieldStatus.FULL_POWER_CHANGE_COUNT_UP: float(context.total_counters['stance_full_power']),
        FieldStatus.FULL_POWER_POINT_GET_SUM_UP: float(context.total_counters['full_power_point_gained']),
        FieldStatus.FULL_POWER_POINT_UP: context.resources['full_power_point'],
        FieldStatus.FULL_POWER_UP: 1.0 if context.stance == 'full_power' else 0.0,
        FieldStatus.LESSON_BUFF_UP: context.resources['lesson_buff'],
        FieldStatus.NO_BLOCK: 1.0 if context.resources['block'] <= 0 else 0.0,
        FieldStatus.NO_STANCE: 1.0 if context.stance == 'neutral' else 0.0,
        FieldStatus.PARAMETER_BUFF: context.resources['parameter_buff'],
        FieldStatus.PARAMETER_BUFF_MULTIPLE_PER_TURN_UP: context.resources['parameter_buff_multiple_per_turn'],
        FieldStatus.PARAMETER_BUFF_UP: context.resources['parameter_buff'],
        FieldStatus.PLAY_CARD_LESSON: float(context.turn_counters['lesson_plays']),
        FieldStatus.PLAY_CARD_SKILL: float(context.turn_counters['play_count']),
        FieldStatus.PRESERVATION_CHANGE_COUNT_UP: float(context.total_counters['stance_preservation']),
        FieldStatus.PRESERVATION_UP: context.resources['preservation'],
        FieldStatus.REMAINING_TURN: float(max(context.max_turns - context.turn + 1, 0)),
        FieldStatus.REVIEW_UP: context.resources['review'],
        FieldStatus.STAMINA_CONSUMPTION_DOWN: context.resources['stamina_consumption_down'],
        FieldStatus.STAMINA_LESS_MULTIPLE: stamina_ratio,
        FieldStatus.STAMINA_UP_MULTIPLE: stamina_ratio,
        FieldStatus.STANCE_CHANGE_COUNT_UP: float(context.total_counters['stance_changes']),
        FieldStatus.TURN_PROGRESS_UP: float(context.turn),
    }
    return float(mapping.get(field_status_type, 0.0))
