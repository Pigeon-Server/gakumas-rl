"""竞技结果口径工具。

统一解析考试/培育终局信息，给自举选优、BC 成功样本过滤和评估摘要复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CompetitiveEpisodeOutcome:
    """统一描述一局 episode 的竞技结果。"""

    tier: int
    passed: bool
    top1: bool
    route: str
    final_rank: int | None = None
    all_auditions_first: bool = False


def _coerce_int(value: Any) -> int | None:
    """尽量把任意值解析成整数。"""

    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _final_summary_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """从环境 info 中提取终局摘要。"""

    final_summary = info.get('final_summary')
    return dict(final_summary) if isinstance(final_summary, Mapping) else {}


def competitive_outcome_from_info(info: Mapping[str, Any]) -> CompetitiveEpisodeOutcome:
    """按统一竞技口径解析一局结果。

    规则：
    - `初` 路线：最终第 1 名记为 `tier=2`；第 2/3 名记为 `tier=1`；其余失败。
    - `NIA` 路线：只有“每场考试都第 1 且最终第 1”才记为 `tier=2`；否则失败。
    - `lesson`：Perfect 记为 `tier=2`，Clear 记为 `tier=1`。
    - 其他未知场景：保守回退到布尔 clear。
    """

    final_summary = _final_summary_from_info(info)
    if final_summary:
        route = str(final_summary.get('route') or '')
        final_rank = _coerce_int(final_summary.get('final_rank'))
        route_clear = bool(final_summary.get('route_clear'))
        all_auditions_first = bool(final_summary.get('all_auditions_first'))
        if route == 'nia':
            top1 = route_clear and all_auditions_first and final_rank == 1
            return CompetitiveEpisodeOutcome(
                tier=2 if top1 else 0,
                passed=top1,
                top1=top1,
                route=route,
                final_rank=final_rank,
                all_auditions_first=all_auditions_first,
            )
        if route == 'first_star':
            passed = route_clear and final_rank is not None and final_rank <= 3
            top1 = passed and final_rank == 1
            return CompetitiveEpisodeOutcome(
                tier=2 if top1 else (1 if passed else 0),
                passed=passed,
                top1=top1,
                route=route,
                final_rank=final_rank,
                all_auditions_first=all_auditions_first,
            )
        generic_pass = bool(route_clear)
        return CompetitiveEpisodeOutcome(
            tier=1 if generic_pass else 0,
            passed=generic_pass,
            top1=False,
            route=route,
            final_rank=final_rank,
            all_auditions_first=all_auditions_first,
        )

    clear_state = str(info.get('clear_state') or '').lower()
    if 'perfect' in clear_state:
        return CompetitiveEpisodeOutcome(tier=2, passed=True, top1=False, route='lesson')
    if 'clear' in clear_state or 'pass' in clear_state or bool(info.get('lesson_cleared') or info.get('is_clear')):
        return CompetitiveEpisodeOutcome(tier=1, passed=True, top1=False, route='lesson')

    competitive_pass = bool(info.get('competitive_pass') or info.get('cleared'))
    competitive_top1 = bool(info.get('competitive_top1'))
    route = str(info.get('route_type') or '')
    return CompetitiveEpisodeOutcome(
        tier=2 if competitive_top1 else (1 if competitive_pass else 0),
        passed=competitive_pass,
        top1=competitive_top1,
        route=route,
        final_rank=_coerce_int(info.get('competitive_rank')),
    )
