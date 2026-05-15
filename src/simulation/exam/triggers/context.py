"""触发器访问运行时的显式接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExamTriggerContext:
    """触发器专用上下文，集中封装运行时查询能力。"""

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
    def stamina(self) -> float:
        return self.runtime.stamina

    @property
    def max_stamina(self) -> float:
        return self.runtime.max_stamina

    @property
    def score(self) -> float:
        return self.runtime.score

    @property
    def profile(self) -> dict[str, Any]:
        return self.runtime.profile

    @property
    def stance(self) -> str:
        return self.runtime.stance

    @property
    def max_turns(self) -> int:
        return self.runtime.max_turns

    @property
    def turn(self) -> int:
        return self.runtime.turn

    def current_lesson_types(self) -> tuple[str, ...]:
        return self.runtime.current_lesson_types()

    def search_cards(self, search_id: str, acting_card: Any | None = None, target_card: Any | None = None):
        return self.runtime.search_cards(search_id, acting_card=acting_card, target_card=target_card)

    def search_count(self, search_id: str) -> int:
        return self.runtime.search_card_count(search_id)
