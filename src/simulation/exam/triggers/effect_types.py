"""效果类型触发器。"""

from __future__ import annotations

from typing import Any


def effect_types_match(trigger: dict[str, Any], event: dict[str, Any]) -> bool:
    """判断状态变化事件里携带的效果类型是否满足触发器要求。"""

    if not trigger.get('effectTypes'):
        return True
    required = {str(value) for value in trigger.get('effectTypes', []) if value}
    return not required or bool(required.intersection(set(event.get('effect_types') or [])))

