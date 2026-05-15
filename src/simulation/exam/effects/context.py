"""效果器访问运行时的显式接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExamEffectContext:
    """效果器专用上下文，集中封装运行时内部操作。"""

    runtime: Any

    @property
    def resources(self):
        return self.runtime.resources

    @property
    def total_counters(self):
        return self.runtime.total_counters

    @property
    def turn_counters(self):
        return self.runtime.turn_counters

    @property
    def active_enchants(self):
        return self.runtime.active_enchants

    @property
    def forbidden_card_search_ids(self):
        return self.runtime.forbidden_card_search_ids

    @property
    def current_turn_color(self) -> str:
        return self.runtime.current_turn_color

    @property
    def score_per_color(self):
        return self.runtime.score_per_color

    @property
    def hand(self):
        return self.runtime.hand

    @property
    def grave(self):
        return self.runtime.grave

    @property
    def max_stamina(self) -> float:
        return self.runtime.max_stamina

    @property
    def stamina(self) -> float:
        return self.runtime.stamina

    @stamina.setter
    def stamina(self, value: float) -> None:
        self.runtime.stamina = value

    @property
    def score(self) -> float:
        return self.runtime.score

    @score.setter
    def score(self, value: float) -> None:
        self.runtime.score = value

    @property
    def extra_turns(self) -> int:
        return self.runtime.extra_turns

    @extra_turns.setter
    def extra_turns(self, value: int) -> None:
        self.runtime.extra_turns = value

    @property
    def stance_locked(self) -> bool:
        return self.runtime.stance_locked

    @stance_locked.setter
    def stance_locked(self, value: bool) -> None:
        self.runtime.stance_locked = value

    @property
    def start_turn_draw_penalty(self) -> int:
        return self.runtime.start_turn_draw_penalty

    @start_turn_draw_penalty.setter
    def start_turn_draw_penalty(self, value: int) -> None:
        self.runtime.start_turn_draw_penalty = value

    def raw_value(self, effect: dict[str, Any]) -> float:
        return self.runtime.raw_effect_value(effect)

    def direct_value(self, effect: dict[str, Any]) -> float:
        return self.runtime.direct_effect_value(effect)

    def ratio_value(self, effect: dict[str, Any]) -> float:
        return self.runtime.ratio_effect_value(effect)

    def count_value(self, effect: dict[str, Any]) -> float:
        return self.runtime.count_effect_value(effect)

    def positive_count(self, value: float) -> int:
        return self.runtime.positive_count(value)

    def ceil_positive(self, value: float) -> float:
        return self.runtime.ceil_positive(value)

    def compose_referenced_gain(self, *, base: float, referenced: float) -> float:
        return self.runtime.compose_referenced_gain(base=base, referenced=referenced)

    def search_count(self, search_id: str) -> int:
        return self.runtime.search_card_count(search_id)

    def adjust_direct_gain(self, value: float, *, add_grow_type: str = '', reduce_grow_type: str = '') -> float:
        return self.runtime.adjust_direct_gain(value, add_grow_type=add_grow_type, reduce_grow_type=reduce_grow_type)

    def current_card_grow_total(self, grow_effect_type: str) -> float:
        return self.runtime.current_card_grow_total(grow_effect_type)

    def current_card_ratio_bonus(self, grow_effect_type: str) -> float:
        return self.runtime.current_card_ratio_bonus(grow_effect_type)

    def parameter_buff_gain_value(self, effect: dict[str, Any]) -> float:
        return self.runtime.parameter_buff_gain_value(effect)

    def dispatch_status_change(self, delta: float, effect_types: list[str], *, origin: str) -> None:
        self.runtime.dispatch_status_change(delta, effect_types, origin=origin)

    def consume_anti_debuff(self, effect_type: str) -> bool:
        return self.runtime.consume_anti_debuff(effect_type)

    def register_timed_effect(self, effect: dict[str, Any], source: str) -> None:
        self.runtime.register_timed_effect(effect, source)

    def apply_status_enchant(self, effect: dict[str, Any], source: str) -> None:
        self.runtime.apply_status_enchant(effect, source)

    def schedule_effect(self, effect: dict[str, Any]) -> None:
        self.runtime.schedule_effect(effect)

    def add_grow_effect(self, effect: dict[str, Any]) -> None:
        self.runtime.add_grow_effect(effect)

    def apply_card_operation(self, effect: dict[str, Any]) -> None:
        self.runtime.apply_card_operation(effect)

    def spend_stamina(self, value: float, *, phase_type: str, status_change_origin: str) -> None:
        self.runtime.spend_stamina(value, phase_type=phase_type, status_change_origin=status_change_origin)

    def has_timed_effect(self, effect_type: str) -> bool:
        return self.runtime.has_timed_effect(effect_type)

    def gain_block(self, delta: float, *, effect_type: str, status_change_origin: str) -> None:
        self.runtime.gain_block(delta, effect_type=effect_type, status_change_origin=status_change_origin)

    def consume_parameter_buff_multiple(self, value: float) -> None:
        self.runtime.consume_parameter_buff_multiple(value)

    def draw(self, count: int) -> None:
        self.runtime.draw(count)

    def enter_concentration(self, level: int) -> None:
        self.runtime.enter_concentration(level)

    def enter_preservation(self, level: int) -> None:
        self.runtime.enter_preservation(level)

    def enter_full_power(self) -> None:
        self.runtime.enter_full_power()

    def reset_stance(self) -> None:
        self.runtime.reset_stance()

    def clear_negative_effects(self) -> None:
        self.runtime.clear_negative_effects()

    def sync_forbidden_search_resources(self) -> None:
        self.runtime.sync_forbidden_search_resources()

    def score_gain(self, value: float) -> float:
        return self.runtime.score_gain(value)

    def resolve_lesson_effect_value(self, effect: dict[str, Any], *, from_card: bool = False) -> float:
        return self.runtime.resolve_lesson_effect_value(effect, from_card=from_card)

    def update_clear_state_after_score_change(self) -> None:
        self.runtime.update_clear_state_after_score_change()

    def apply_score_value_modifiers(self, value: float) -> float:
        return self.runtime.apply_score_value_modifiers(value)
