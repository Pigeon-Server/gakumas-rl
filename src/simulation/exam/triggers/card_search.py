"""卡牌检索触发器。"""

from __future__ import annotations

from typing import Any

from .context import ExamTriggerContext


def trigger_card_search_matches(
    context: ExamTriggerContext,
    trigger: dict[str, Any],
    acting_card: Any | None,
    target_card: Any | None,
) -> bool:
    """检查触发器里的卡牌搜索条件是否成立。"""

    search_id = str(trigger.get('produceCardSearchId') or '')
    if not search_id:
        return True
    selection = context.search_cards(search_id, acting_card=acting_card, target_card=target_card)
    lower = int(trigger.get('lowerSearchCount') or 0)
    upper = int(trigger.get('upperSearchCount') or 0)
    if lower and selection.pool_size < lower:
        return False
    if upper and selection.pool_size > upper:
        return False
    return selection.pool_size > 0 or (lower == 0 and upper == 0)
