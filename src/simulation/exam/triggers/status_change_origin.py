"""状态变化来源触发器。"""

from __future__ import annotations

from typing import Any

from ..constants import STATUS_CHANGE_TRIGGER_ORIGINS
from ..ids import ExamPhase


def status_change_origin_matches(event: dict[str, Any]) -> bool:
    """状态变化 phase 只允许卡牌和饮料触发，避免附魔递归自激。"""

    if event['phase_type'] != ExamPhase.STATUS_CHANGE:
        return True
    return str(event.get('status_change_origin') or '') in STATUS_CHANGE_TRIGGER_ORIGINS
