"""持续资源效果器。"""

from __future__ import annotations

from ..constants import DURATION_RESOURCE_TYPES
from ..ids import GrowEffect
from .context import ExamEffectContext


def apply_duration_resource(context: ExamEffectContext, effect: dict[str, Any], source: str) -> None:
    """处理参数强化这类按持续回合记录的资源。"""

    effect_type = str(effect.get('effectType') or '')
    resource_key = DURATION_RESOURCE_TYPES[effect_type]
    delta = context.parameter_buff_gain_value(effect)
    if source == 'card':
        delta = context.adjust_direct_gain(delta, add_grow_type=GrowEffect.PARAMETER_BUFF_TURN_ADD)
    context.resources[resource_key] += delta
    context.dispatch_status_change(delta, [effect_type], origin=source)
