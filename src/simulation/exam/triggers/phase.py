"""phase 类型触发器。"""

from __future__ import annotations

from typing import Any


def phase_matches(trigger: dict[str, Any], event: dict[str, Any]) -> bool:
    """判断事件 phase 是否命中主数据触发器声明。"""

    phase_types = [str(value) for value in trigger.get('phaseTypes', []) if value]
    return not phase_types or event['phase_type'] in phase_types

