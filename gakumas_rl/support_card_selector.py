"""支援卡自动编成器。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np

from .data import MasterDataRepository, ScenarioSpec
from .loadout import IdolLoadout, SupportCardSelection


_SUPPORT_CARD_TYPE_TO_STAT = {
    'SupportCardType_Vocal': 'vocal',
    'SupportCardType_Dance': 'dance',
    'SupportCardType_Visual': 'visual',
    'SupportCardType_Assist': 'assist',
    'SupportCardType_Stamina': 'stamina',
}

_STAT_TO_SUPPORT_CARD_TYPE = {
    'vocal': 'SupportCardType_Vocal',
    'dance': 'SupportCardType_Dance',
    'visual': 'SupportCardType_Visual',
}

_SUPPORT_SKILL_TABLE_BY_TYPE = {
    'SupportCardType_Vocal': 'vocal',
    'SupportCardType_Dance': 'dance',
    'SupportCardType_Visual': 'visual',
    'SupportCardType_Assist': 'assist',
}

_RARITY_TO_DEFAULT_LEVEL = {
    'SupportCardRarity_R': 40,
    'SupportCardRarity_SR': 50,
    'SupportCardRarity_SSR': 60,
}

_SUPPORT_EVENT_TYPE_BONUS = {
    'ProduceEventType_SupportCard': 0.45,
}


@dataclass(frozen=True)
class SupportCardAutoSelectConfig:
    """自动选支援卡时使用的近似配置。"""

    support_card_level: int = 60
    deck_size: int = 6
    include_assist_slots: bool = True
    support_card_ids_pool: tuple[str, ...] = ()


class SupportCardAutoSelector:
    """基于主数据与系统推荐参数的支援卡自动编成器。"""

    def __init__(self, repository: MasterDataRepository):
        self.repository = repository

    @cached_property
    def _runtime_setting(self) -> dict:
        rows = self.repository.setting.rows
        return dict(rows[0]) if rows else {}

    @cached_property
    def _support_skill_rows_by_card(self) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        table_map = {
            'vocal': self.repository.support_card_skill_level_vocal,
            'dance': self.repository.support_card_skill_level_dance,
            'visual': self.repository.support_card_skill_level_visual,
            'assist': self.repository.support_card_skill_level_assist,
        }
        for rows in table_map.values():
            for row in rows.rows:
                support_card_id = str(row.get('supportCardId') or '')
                if support_card_id:
                    grouped[support_card_id].append(dict(row))
        return grouped

    @cached_property
    def _support_event_rows_by_card(self) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        detail_table = self.repository.produce_step_event_details
        for row in self.repository.produce_event_support_cards.rows:
            support_card_id = str(row.get('supportCardId') or '')
            detail_id = str(row.get('produceStepEventDetailId') or '')
            if not support_card_id or not detail_id:
                continue
            detail_row = detail_table.first(detail_id)
            if detail_row is None:
                continue
            grouped[support_card_id].append(dict(detail_row))
        return grouped

    def auto_select(
        self,
        scenario: ScenarioSpec,
        loadout: IdolLoadout,
        config: SupportCardAutoSelectConfig | None = None,
    ) -> tuple[SupportCardSelection, ...]:
        """按当前偶像和模式自动选择一套支援卡。"""

        resolved = config or SupportCardAutoSelectConfig()
        candidates = self._candidate_rows(loadout, resolved)
        if not candidates:
            return ()

        preferred_types = self._preferred_support_card_types(scenario, loadout, resolved)
        scored = [
            self._score_support_card(
                card_row,
                loadout=loadout,
                preferred_types=preferred_types,
                support_card_level=self._resolve_support_card_level(card_row, resolved.support_card_level),
            )
            for card_row in candidates
        ]
        scored.sort(key=lambda item: item.score, reverse=True)

        chosen: list[SupportCardSelection] = []
        chosen_ids: set[str] = set()
        per_type_counts: dict[str, int] = defaultdict(int)
        preferred_slot_counts = list(self._runtime_setting.get('produceSupportCardRecommendTypePreferentialSlotCounts') or [3, 2, 1])
        support_card_limit = max(int(self._runtime_setting.get('produceMaxSupportCardDeckCount') or resolved.deck_size), resolved.deck_size)
        deck_size = min(int(resolved.deck_size), support_card_limit)

        for preferred_type, slot_limit in zip(preferred_types, preferred_slot_counts, strict=False):
            for item in scored:
                if len(chosen) >= deck_size:
                    break
                if item.support_card_id in chosen_ids or item.support_card_type != preferred_type:
                    continue
                if per_type_counts[preferred_type] >= int(slot_limit):
                    continue
                chosen.append(item)
                chosen_ids.add(item.support_card_id)
                per_type_counts[preferred_type] += 1
            if len(chosen) >= deck_size:
                break

        for item in scored:
            if len(chosen) >= deck_size:
                break
            if item.support_card_id in chosen_ids:
                continue
            chosen.append(item)
            chosen_ids.add(item.support_card_id)

        return tuple(chosen[:deck_size])

    def _candidate_rows(
        self,
        loadout: IdolLoadout,
        config: SupportCardAutoSelectConfig,
    ) -> list[dict]:
        """按流派与候选池过滤支援卡。"""

        allowed_ids = {str(value) for value in config.support_card_ids_pool if str(value or '')}
        allowed_plan_types = {'ProducePlanType_Common', loadout.stat_profile.plan_type}
        rows: list[dict] = []
        for row in self.repository.support_cards.rows:
            support_card_id = str(row.get('id') or '')
            if not support_card_id:
                continue
            if allowed_ids and support_card_id not in allowed_ids:
                continue
            if str(row.get('planType') or 'ProducePlanType_Common') not in allowed_plan_types:
                continue
            rows.append(dict(row))
        return rows

    def _preferred_support_card_types(
        self,
        scenario: ScenarioSpec,
        loadout: IdolLoadout,
        config: SupportCardAutoSelectConfig,
    ) -> tuple[str, ...]:
        """根据剧本审查权重与偶像三维排序支援卡类型优先级。"""

        stats = np.array(
            [
                float(loadout.stat_profile.vocal),
                float(loadout.stat_profile.dance),
                float(loadout.stat_profile.visual),
            ],
            dtype=np.float32,
        )
        weights = np.array(scenario.score_weights, dtype=np.float32)
        priorities = stats * (0.4 + weights)
        ordered_stats = [
            stat_name
            for _, stat_name in sorted(
                zip(priorities, ('vocal', 'dance', 'visual'), strict=True),
                reverse=True,
            )
        ]
        preferred = [_STAT_TO_SUPPORT_CARD_TYPE[stat_name] for stat_name in ordered_stats]
        if config.include_assist_slots:
            preferred.append('SupportCardType_Assist')
        preferred.append('SupportCardType_Stamina')
        return tuple(dict.fromkeys(preferred))

    def _resolve_support_card_level(self, card_row: dict, requested_level: int) -> int:
        """把目标等级裁剪到该稀有度能达到的区间。"""

        rarity = str(card_row.get('rarity') or '')
        default_level = _RARITY_TO_DEFAULT_LEVEL.get(rarity, 40)
        return max(1, min(int(requested_level), default_level))

    def _score_support_card(
        self,
        card_row: dict,
        *,
        loadout: IdolLoadout,
        preferred_types: Iterable[str],
        support_card_level: int,
    ) -> SupportCardSelection:
        """对单张支援卡打分，并记录可解释原因。"""

        support_card_id = str(card_row.get('id') or '')
        support_card_type = str(card_row.get('type') or '')
        reasons: list[str] = []
        score = 0.0

        preferred_types = tuple(preferred_types)
        if support_card_type in preferred_types:
            rank = preferred_types.index(support_card_type)
            additions = list(self._runtime_setting.get('produceSupportCardRecommendUsageRateRankAdditionValues') or [])
            rank_bonus = additions[min(rank, len(additions) - 1)] / 100.0 if additions else max(0.0, 0.6 - rank * 0.1)
            score += rank_bonus
            reasons.append(f'type_rank={rank_bonus:.2f}')
        else:
            penalty = float(self._runtime_setting.get('produceSupportCardRecommendTypeNegativeCoefficientPermil') or 50) / 1000.0
            score -= penalty
            reasons.append(f'type_penalty={penalty:.2f}')

        if support_card_type == 'SupportCardType_Assist':
            score += 0.25
            reasons.append('assist')
        if str(card_row.get('planType') or 'ProducePlanType_Common') == loadout.stat_profile.plan_type:
            score += 0.2
            reasons.append('plan_match')

        level_bonus = self._level_score(card_row, support_card_level)
        score += level_bonus
        reasons.append(f'level={level_bonus:.2f}')

        skill_bonus = self._support_skill_score(card_row, support_card_level)
        score += skill_bonus
        if skill_bonus:
            reasons.append(f'skill={skill_bonus:.2f}')

        event_bonus = self._support_event_score(card_row, support_card_level)
        score += event_bonus
        if event_bonus:
            reasons.append(f'event={event_bonus:.2f}')

        sp_bonus = self._sp_support_score(card_row, support_card_level)
        score += sp_bonus
        if sp_bonus:
            reasons.append(f'sp={sp_bonus:.2f}')

        return SupportCardSelection(
            support_card_id=support_card_id,
            support_card_level=support_card_level,
            support_card_type=support_card_type,
            score=float(score),
            reasons=tuple(reasons),
        )

    def _level_score(self, card_row: dict, support_card_level: int) -> float:
        """按系统配置的等级系数近似计算等级贡献。"""

        coefficients = list(self._runtime_setting.get('produceSupportCardRecommendLevelCoefficientPermils') or [1, 3, 4])
        if support_card_level >= 50:
            coefficient = coefficients[min(2, len(coefficients) - 1)]
        elif support_card_level >= 30:
            coefficient = coefficients[min(1, len(coefficients) - 1)]
        else:
            coefficient = coefficients[0]
        rarity = str(card_row.get('rarity') or '')
        rarity_bonus = {
            'SupportCardRarity_R': 0.10,
            'SupportCardRarity_SR': 0.22,
            'SupportCardRarity_SSR': 0.36,
        }.get(rarity, 0.08)
        return float(coefficient) / 100.0 + rarity_bonus

    def _support_skill_rows(self, card_row: dict, support_card_level: int) -> list[dict]:
        """返回该支援卡在当前等级已解锁的培育技能行。"""

        support_card_id = str(card_row.get('id') or '')
        resolved: list[dict] = []
        for row in self._support_skill_rows_by_card.get(support_card_id, []):
            if int(row.get('supportCardLevel') or 0) <= int(support_card_level):
                resolved.append(row)
        return resolved

    def _support_skill_score(self, card_row: dict, support_card_level: int) -> float:
        """根据支援卡技能的触发器和效果类型估算价值。"""

        score = 0.0
        for row in self._support_skill_rows(card_row, support_card_level):
            produce_skill_id = str(row.get('produceSkillId') or '')
            produce_skill_level = int(row.get('produceSkillLevel') or 1)
            skill_row = self.repository.load_table('ProduceSkill').first(f'{produce_skill_id}.{produce_skill_level}')
            if skill_row is None:
                candidates = [
                    item
                    for item in self.repository.load_table('ProduceSkill').all(produce_skill_id)
                    if int(item.get('level') or 1) == produce_skill_level
                ]
                skill_row = candidates[0] if candidates else self.repository.load_table('ProduceSkill').first(produce_skill_id)
            if skill_row is None:
                continue
            trigger_ids = [str(skill_row.get('produceTriggerId1') or '')]
            for trigger_id in trigger_ids:
                if 'produce_start' in trigger_id:
                    score += 0.25
                elif 'end_lesson_before_present' in trigger_id:
                    score += 0.18
                elif 'end_lesson' in trigger_id:
                    score += 0.15
                elif 'consult' in trigger_id or 'shop' in trigger_id:
                    score += 0.08
            for effect_key in ('produceEffectId1', 'produceEffectId2', 'produceEffectId3'):
                effect_id = str(skill_row.get(effect_key) or '')
                if not effect_id:
                    continue
                effect_row = self.repository.produce_effects.first(effect_id)
                if effect_row is None:
                    continue
                effect_type = str(effect_row.get('produceEffectType') or '')
                if effect_type.endswith('SpChangeRatePermilAddition'):
                    score += 0.22
                elif effect_type in {
                    'ProduceEffectType_VocalAddition',
                    'ProduceEffectType_DanceAddition',
                    'ProduceEffectType_VisualAddition',
                }:
                    score += 0.10
                elif effect_type in {
                    'ProduceEffectType_ProducePointAddition',
                    'ProduceEffectType_LessonPresentProducePointUp',
                    'ProduceEffectType_EventBusinessVoteCountUp',
                }:
                    score += 0.08
                elif effect_type in {
                    'ProduceEffectType_MaxStaminaAddition',
                    'ProduceEffectType_StaminaRecoverFix',
                }:
                    score += 0.06
        return score

    def _support_event_score(self, card_row: dict, support_card_level: int) -> float:
        """根据已解锁支援事件的奖励粗估价值。"""

        support_card_id = str(card_row.get('id') or '')
        score = 0.0
        for detail_row in self._support_event_rows_by_card.get(support_card_id, []):
            required_level = int(detail_row.get('supportCardLevel') or 0)
            if required_level > int(support_card_level):
                continue
            event_type = str(detail_row.get('eventType') or '')
            score += _SUPPORT_EVENT_TYPE_BONUS.get(event_type, 0.18)
            descriptions = detail_row.get('produceDescriptions', []) or []
            for desc in descriptions:
                text = str(desc.get('text') or '') if isinstance(desc, dict) else ''
                if 'Pドリンク' in text:
                    score += 0.08
                if 'スキルカード' in text:
                    score += 0.10
                if '体力' in text:
                    score += 0.05
        return score

    def _sp_support_score(self, card_row: dict, support_card_level: int) -> float:
        """对 SP 课程倾向明显的支援卡额外加分。"""

        score = 0.0
        support_card_type = str(card_row.get('type') or '')
        preferred_slots = list(self._runtime_setting.get('produceSupportCardRecommendSpPreferentialSlotCounts') or [2, 1, 0])
        preferential = float(self._runtime_setting.get('produceSupportCardRecommendSpPreferentialCoefficientPermil') or 200) / 1000.0
        if support_card_type in {'SupportCardType_Vocal', 'SupportCardType_Dance', 'SupportCardType_Visual'}:
            score += preferential * (preferred_slots[0] if preferred_slots else 1)
        for row in self._support_skill_rows(card_row, support_card_level):
            produce_skill_id = str(row.get('produceSkillId') or '')
            if 'sp' in produce_skill_id.lower():
                score += preferential
        return score


def auto_select_support_cards(
    repository: MasterDataRepository,
    scenario: ScenarioSpec,
    loadout: IdolLoadout,
    config: SupportCardAutoSelectConfig | None = None,
) -> tuple[SupportCardSelection, ...]:
    """对外暴露的自动选支援卡入口。"""

    selector = SupportCardAutoSelector(repository)
    return selector.auto_select(scenario, loadout, config=config)
