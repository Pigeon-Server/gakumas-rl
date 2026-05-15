"""P 道具触发次数效果器。"""

from __future__ import annotations

from .context import ExamEffectContext


def apply_item_fire_limit_add(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """增加来源于 P 道具的附魔触发次数。"""

    delta = context.direct_value(effect)
    context.resources['item_fire_limit_add'] += delta
    for enchant in context.active_enchants:
        if enchant.source != 'produce_item' or enchant.remaining_count is None:
            continue
        # 计数型：附魔触发次数，整数语义
        enchant.remaining_count += int(round(delta))
