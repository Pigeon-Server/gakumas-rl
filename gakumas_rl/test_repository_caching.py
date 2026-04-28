"""主数据派生缓存的轻量回归测试。"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np

from gakumas_rl.data import MasterDataRepository


@lru_cache(maxsize=1)
def _repository() -> MasterDataRepository:
    """缓存主数据仓库，避免测试重复加载资源。"""

    return MasterDataRepository()


def test_card_exam_effect_types_cache_keeps_results_stable(monkeypatch) -> None:
    """同一张卡的考试效果类型应命中缓存，且不改变返回结果。"""

    repository = _repository()
    scenario = repository.build_scenario('produce-005')
    card = repository.weighted_card_pool(scenario)[0]

    expected = repository.card_exam_effect_types(card)

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError('card_exam_effect_types second call should hit cache')

    monkeypatch.setattr(repository, '_resolve_effect_group_types', _should_not_be_called)

    assert repository.card_exam_effect_types(card) == expected


def test_weighted_card_pool_cache_reuses_sorted_pool(monkeypatch) -> None:
    """同一场景和筛选条件下，默认卡池应直接复用缓存结果。"""

    repository = _repository()
    scenario = repository.build_scenario('produce-005')

    expected = [str(card.get('id') or '') for card in repository.weighted_card_pool(scenario, focus_effect_type='ProduceExamEffectType_ExamReview')]

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError('weighted_card_pool second call should hit cache')

    monkeypatch.setattr(repository, 'card_exam_effect_types', _should_not_be_called)

    actual = [str(card.get('id') or '') for card in repository.weighted_card_pool(scenario, focus_effect_type='ProduceExamEffectType_ExamReview')]
    assert actual == expected


def test_interval_phase_values_matches_trigger_rows() -> None:
    """预编译的 interval phase 索引应与原始 trigger rows 一致。"""

    repository = _repository()
    expected: dict[str, set[int]] = defaultdict(set)
    for trigger in repository.exam_triggers.rows:
        phase_types = [str(item) for item in trigger.get('phaseTypes', []) if item]
        values = [int(value) for value in trigger.get('phaseValues', []) if int(value or 0) > 0]
        if not phase_types or not values:
            continue
        for phase_type in phase_types:
            expected[phase_type].update(values)

    actual = repository.interval_phase_values

    assert set(actual) == set(expected)
    for phase_type, values in expected.items():
        assert actual[phase_type] == tuple(sorted(values))


def test_card_row_lookup_cache_reuses_loaded_rows() -> None:
    """基础卡面与强化卡面查找应复用缓存，不再重复访问表索引。"""

    repository = _repository()
    scenario = repository.build_scenario('produce-005')
    card = repository.weighted_card_pool(scenario)[0]
    card_id = str(card.get('id') or '')
    upgrade_count = int(card.get('upgradeCount') or 0)

    expected_canonical = repository.canonical_card_row(card_id)
    expected_upgrade = repository.card_row_by_upgrade(card_id, upgrade_count)

    assert repository.canonical_card_row(card_id) == expected_canonical
    assert repository.card_row_by_upgrade(card_id, upgrade_count) == expected_upgrade
    assert repository._canonical_card_row_cache[card_id] == expected_canonical
    assert repository._card_row_by_upgrade_cache[card_id][upgrade_count] == expected_upgrade


def test_sample_random_card_variant_cache_keeps_distribution_inputs_stable() -> None:
    """随机卡面采样应复用已排序卡面与权重，不再重复读取主数据。"""

    repository = _repository()
    scenario = repository.build_scenario('produce-005')
    card = repository.weighted_card_pool(scenario)[0]
    card_id = str(card.get('id') or '')

    expected = repository.sample_random_card_variant(card_id, np.random.default_rng(17))
    actual = repository.sample_random_card_variant(card_id, np.random.default_rng(17))

    assert actual == expected
    cached_rows, cached_weights = repository._card_variant_sampling_cache[card_id]
    assert cached_rows
    assert cached_weights.shape[0] == len(cached_rows)
