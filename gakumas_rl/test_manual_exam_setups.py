"""Manual exam setup overrides and deck constraint regressions."""

from __future__ import annotations

from functools import lru_cache
import json
from collections import Counter

import numpy as np
import pytest

from gakumas_rl.data import MasterDataRepository
from gakumas_rl.deck_constraints import normalize_forced_card_groups, normalize_guaranteed_effect_counts
from gakumas_rl.idol_config import build_idol_loadout, build_initial_exam_deck
from gakumas_rl.service import build_env_from_config

pytest.importorskip('gymnasium')

AMAO_R = 'i_card-amao-1-000'


@lru_cache(maxsize=1)
def _repository() -> MasterDataRepository:
    return MasterDataRepository()


def _first_visible_card_with_effect(repository: MasterDataRepository, effect_type: str, *, plan_type: str = '') -> str:
    for rows in repository.produce_cards.by_id.values():
        card = min(rows, key=lambda row: int(row.get('upgradeCount') or 0))
        if card.get('libraryHidden') or card.get('isLimited'):
            continue
        card_plan_type = str(card.get('planType') or 'ProducePlanType_Common')
        if plan_type and card_plan_type not in {'ProducePlanType_Common', plan_type}:
            continue
        if effect_type in repository.card_axis_effect_types(card):
            return str(card.get('id') or '')
    raise AssertionError(f'Unable to find visible card with effect type: {effect_type}')


def _first_card_with_high_upgrade_variants(repository: MasterDataRepository) -> str:
    for card_id, rows in repository.produce_cards.by_id.items():
        if not card_id:
            continue
        visible_rows = [row for row in rows if not row.get('libraryHidden')]
        if visible_rows and max(int(row.get('upgradeCount') or 0) for row in visible_rows) >= 3:
            return str(card_id)
    raise AssertionError('Unable to find card with >=4 upgrade variants')


def _first_visible_drink_id(repository: MasterDataRepository) -> str:
    for drink in repository.produce_drinks.rows:
        if not drink.get('libraryHidden'):
            return str(drink.get('id') or '')
    raise AssertionError('Unable to find visible drink')


def _first_item_with_exam_enchant(repository: MasterDataRepository) -> str:
    item_effects = repository.load_table('ProduceItemEffect')
    for item in repository.produce_items.rows:
        effect_ids = [str(value) for value in item.get('produceItemEffectIds', []) if value]
        for skill in item.get('skills', []) or []:
            effect_id = str(skill.get('produceItemEffectId') or '')
            if effect_id:
                effect_ids.append(effect_id)
        if any(
            str((item_effects.first(effect_id) or {}).get('effectType') or '') == 'ProduceItemEffectType_ExamStatusEnchant'
            for effect_id in effect_ids
        ):
            return str(item.get('id') or '')
    raise AssertionError('Unable to find produce item with exam enchant')


def test_normalize_guaranteed_effect_counts_accepts_chinese_aliases() -> None:
    counts = normalize_guaranteed_effect_counts(['打分=2', '好印象=3'])

    assert counts == {
        'ProduceExamEffectType_Score': 2,
        'ProduceExamEffectType_Review': 3,
    }


def test_normalize_forced_card_groups_accepts_json_specs() -> None:
    groups = normalize_forced_card_groups(
        [
            '{"好印象":["p_card-01","p_card-02"]}',
            '{"干劲":["p_card-03"]}',
        ]
    )

    assert groups == {
        'ProduceExamEffectType_Review': ('p_card-01', 'p_card-02'),
        'ProduceExamEffectType_Aggressive': ('p_card-03',),
    }


def test_normalize_forced_card_groups_rejects_non_axis_labels() -> None:
    with pytest.raises(ValueError, match='Unknown force-card axis alias'):
        normalize_forced_card_groups(['{"打分":["p_card-03"]}'])


def test_build_initial_exam_deck_supports_guarantees_and_force_cards() -> None:
    repository = _repository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)
    review_card_id = _first_visible_card_with_effect(
        repository,
        'ProduceExamEffectType_Review',
        plan_type=loadout.stat_profile.plan_type,
    )
    aggressive_card_id = _first_visible_card_with_effect(
        repository,
        'ProduceExamEffectType_Aggressive',
        plan_type=loadout.stat_profile.plan_type,
    )

    deck = build_initial_exam_deck(
        repository,
        scenario,
        loadout=loadout,
        rng=np.random.default_rng(7),
        guaranteed_effect_counts={'ProduceExamEffectType_Review': 2},
        forced_card_groups={'ProduceExamEffectType_Aggressive': (aggressive_card_id,)},
    )

    review_count = sum(
        1
        for card in deck
        if 'ProduceExamEffectType_Review' in repository.card_axis_effect_types(card)
    )
    assert review_count >= 2
    assert any(str(card.get('id') or '') == aggressive_card_id for card in deck)
    assert any(str(card.get('id') or '') == review_card_id for card in deck)


def test_random_card_variant_distribution_prefers_mid_upgrades() -> None:
    repository = _repository()
    card_id = _first_card_with_high_upgrade_variants(repository)
    rng = np.random.default_rng(123)
    counts: Counter[int] = Counter()

    for _ in range(4000):
        row = repository.sample_random_card_variant(card_id, rng)
        counts[int((row or {}).get('upgradeCount') or 0)] += 1

    assert counts[1] > counts[0]
    assert counts[1] > counts[2]
    assert counts[2] > counts[3]
    assert counts[3] / max(sum(counts.values()), 1) < 0.10


def test_exam_env_can_use_manual_exam_setup_dataset(tmp_path) -> None:
    repository = _repository()
    review_card_id = _first_visible_card_with_effect(repository, 'ProduceExamEffectType_Review')
    score_card_id = _first_visible_card_with_effect(repository, 'ProduceExamEffectType_Score')
    drink_id = _first_visible_drink_id(repository)
    item_id = _first_item_with_exam_enchant(repository)
    dataset_path = tmp_path / 'manual_exam.jsonl'
    dataset_path.write_text(
        json.dumps(
            {
                'label': 'real-mid1-sample',
                'scenario': 'nia_master',
                'idol_card_id': AMAO_R,
                'stage_type': 'ProduceStepType_AuditionMid1',
                'deck': [
                    {'card_id': review_card_id, 'count': 2},
                    {'card_id': score_card_id},
                ],
                'drinks': [drink_id],
                'produce_items': [item_id],
            },
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )

    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'manual_exam_setup_paths': [str(dataset_path)],
        }
    )
    _, info = env.reset(seed=11)

    assert info['stage_type'] == 'ProduceStepType_AuditionMid1'
    assert info['manual_setup']['label'] == 'real-mid1-sample'
    assert info['manual_setup']['deck_size'] == 3
    assert info['episode_context']['idol_card_id'] == AMAO_R
    assert [str(card.get('id') or '') for card in env.runtime.initial_deck_rows] == [
        review_card_id,
        review_card_id,
        score_card_id,
    ]
    assert [str(drink.get('id') or '') for drink in env.runtime.initial_drinks] == [drink_id]
    assert any(
        str(enchant.get('source_identity') or '') == item_id
        for enchant in env.runtime.initial_status_enchants
    )
