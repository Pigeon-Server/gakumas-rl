"""phase 数值触发器。"""

from __future__ import annotations

from typing import Any

from ..ids import ExamPhase


def phase_value_matches(trigger: dict[str, Any], event: dict[str, Any]) -> bool:
    """判断回合数、间隔数等 phase value 是否符合主数据条件。"""

    phase_values = [int(value) for value in trigger.get('phaseValues', []) if value not in (None, 0)]
    if not phase_values:
        return True
    current_phase_value = int(event.get('phase_value') or 0)
    if event['phase_type'] == ExamPhase.STATUS_CHANGE:
        return any(current_phase_value >= value for value in phase_values)
    return current_phase_value in phase_values
