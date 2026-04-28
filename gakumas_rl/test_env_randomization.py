"""考试环境随机化与动作掩码的轻量回归测试。"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from gakumas_rl.data import MasterDataRepository
from gakumas_rl.idol_config import build_idol_loadout, build_initial_exam_deck
from gakumas_rl.service import build_env_from_config

pytest.importorskip('gymnasium')

AMAO_R = 'i_card-amao-1-000'


@lru_cache(maxsize=1)
def _repository() -> MasterDataRepository:
    """缓存主数据仓库，避免重复加载测试资源。"""

    return MasterDataRepository()


def test_loadout_initial_exam_deck_sampling_varies_with_seed() -> None:
    """loadout 路径补牌应按 RNG 抽样，而不是固定拿排序前几张。"""

    repository = _repository()
    scenario = repository.build_scenario('produce-005')
    loadout = build_idol_loadout(repository, scenario, AMAO_R, producer_level=35, idol_rank=4, dearness_level=10)

    decks = []
    for seed in range(6):
        deck = build_initial_exam_deck(repository, scenario, loadout=loadout, rng=np.random.default_rng(seed))
        decks.append(tuple(str(card.get('id') or '') for card in deck))

    assert len(set(decks)) > 1


def test_exam_context_randomization_changes_between_resets() -> None:
    """随机上下文应在不同考试 reset 之间变化。"""

    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_R,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'exam_randomize_context': True,
            'exam_randomize_stage_type': True,
            'exam_randomize_use_after_item': True,
        }
    )

    _, info1 = env.reset(seed=123)
    _, info2 = env.reset(seed=124)

    ctx1 = info1['episode_context']
    ctx2 = info2['episode_context']
    key1 = (info1['stage_type'], ctx1['vocal'], ctx1['dance'], ctx1['visual'], ctx1['exam_score_bonus_multiplier'])
    key2 = (info2['stage_type'], ctx2['vocal'], ctx2['dance'], ctx2['visual'], ctx2['exam_score_bonus_multiplier'])

    assert key1 != key2


def test_exam_context_stays_fixed_within_one_episode() -> None:
    """上下文随机化只应发生在 reset 时，局内不应漂移。"""

    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_R,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'exam_randomize_context': True,
            'exam_randomize_stage_type': True,
        }
    )

    obs, info = env.reset(seed=7)
    before_stage = info['stage_type']
    before_stats = tuple(env.runtime.parameter_stats)
    action = next(index for index, flag in enumerate(obs['action_mask']) if flag > 0.5)

    _, _, _, _, step_info = env.step(action)

    assert step_info['stage_type'] == before_stage
    assert tuple(env.runtime.parameter_stats) == before_stats


def test_exam_use_after_item_randomization_does_not_require_context_jitter() -> None:
    """只打开 use_after_item 随机化时，不应偷偷带上属性抖动。"""

    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_R,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'exam_randomize_use_after_item': True,
        }
    )

    sampled_use_after_item = set()
    sampled_stats = set()
    for seed in range(200, 210):
        _, info = env.reset(seed=seed)
        ctx = info['episode_context']
        sampled_use_after_item.add(bool(ctx['use_after_item']))
        sampled_stats.add((ctx['vocal'], ctx['dance'], ctx['visual'], ctx['exam_score_bonus_multiplier']))

    assert sampled_use_after_item == {False, True}
    assert len(sampled_stats) == 1


def test_exam_env_action_masks_expose_valid_slots() -> None:
    """考试环境应导出可直接给 MaskablePPO 使用的 bool 掩码。"""

    env = build_env_from_config({'mode': 'exam', 'scenario': 'nia_master', 'idol_card_id': AMAO_R, 'producer_level': 35, 'idol_rank': 4, 'dearness_level': 20})
    obs, _ = env.reset(seed=5)

    mask = env.action_masks()

    assert mask.dtype == bool
    assert mask.shape == obs['action_mask'].shape
    assert mask.any()


def test_exam_env_step_info_omits_action_labels_by_default() -> None:
    """训练路径默认不应在 step info 中携带完整动作标签列表。"""

    env = build_env_from_config({'mode': 'exam', 'scenario': 'nia_master', 'idol_card_id': AMAO_R, 'producer_level': 35, 'idol_rank': 4, 'dearness_level': 20})
    obs, _ = env.reset(seed=11)

    action = next(index for index, flag in enumerate(obs['action_mask']) if flag > 0.5)
    _, _, _, _, step_info = env.step(action)

    assert 'action_labels' not in step_info


def test_exam_env_step_info_can_include_action_labels_when_requested() -> None:
    """调试模式下仍可显式保留 step info 中的动作标签。"""

    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_R,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'include_action_labels_in_step_info': True,
        }
    )
    obs, _ = env.reset(seed=12)

    action = next(index for index, flag in enumerate(obs['action_mask']) if flag > 0.5)
    _, _, _, _, step_info = env.step(action)

    assert 'action_labels' in step_info
    assert len(step_info['action_labels']) == env.max_actions


def test_exam_env_defaults_to_all_idol_sampling() -> None:
    """未显式指定 idol_card_id 时，应在全偶像池中按考试粒度采样。"""

    env = build_env_from_config({'mode': 'exam', 'scenario': 'nia_master'})

    sampled_ids = set()
    for seed in range(6):
        _, info = env.reset(seed=seed)
        sampled_ids.add(info['episode_context']['idol_card_id'])

    assert len(sampled_ids) > 1


def test_exam_env_defaults_to_full_starting_stamina() -> None:
    """exam-only 环境默认应满体开考。"""

    env = build_env_from_config({'mode': 'exam', 'scenario': 'nia_master', 'idol_card_id': AMAO_R, 'producer_level': 35, 'idol_rank': 4, 'dearness_level': 20})
    _, info = env.reset(seed=21)

    assert env.runtime.stamina == env.runtime.max_stamina
    assert info['episode_context']['starting_stamina'] == env.runtime.max_stamina
    assert info['episode_context']['starting_stamina_mode'] == 'full'


def test_exam_env_can_randomize_starting_stamina_within_ratio_range() -> None:
    """exam-only 环境可按配置在给定比例区间内随机开场体力。"""

    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_R,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'exam_starting_stamina_mode': 'random',
            'exam_starting_stamina_min_ratio': 0.55,
            'exam_starting_stamina_max_ratio': 0.95,
        }
    )

    sampled = []
    for seed in range(30, 35):
        _, info = env.reset(seed=seed)
        sampled.append(env.runtime.stamina)
        assert env.runtime.max_stamina * 0.55 <= env.runtime.stamina <= env.runtime.max_stamina * 0.95
        assert info['episode_context']['starting_stamina_mode'] == 'random'

    assert len({round(value, 6) for value in sampled}) > 1


def test_exam_env_observation_contains_stage_type_feature() -> None:
    """多 stage 混训时，观测里应显式编码当前 stage_type。"""

    env = build_env_from_config(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'idol_card_id': AMAO_R,
            'producer_level': 35,
            'idol_rank': 4,
            'dearness_level': 20,
            'exam_randomize_stage_type': True,
        }
    )

    obs, info = env.reset(seed=321)
    feature = env._stage_type_feature()

    assert feature.shape == (env.stage_context_dim,)
    assert np.isclose(float(feature.sum()), 1.0)
    feature_index = int(np.argmax(feature))
    if feature_index < len(env.stage_type_ids):
        assert env.stage_type_ids[feature_index] == info['stage_type']
    else:
        assert info['stage_type'] not in env.stage_type_ids
    assert env.observation_space['global'].contains(obs['global'])



def test_exam_env_observation_uses_bounded_encoding_for_extreme_resources() -> None:
    """极端资源值应通过有界编码落在 observation space 内，而不是依赖裁剪。"""

    env = build_env_from_config({'mode': 'exam', 'scenario': 'nia_master'})
    _, _ = env.reset(seed=9)
    env.runtime.resources['review'] = 10_000.0
    env.runtime.resources['block'] = 10_000.0
    env.runtime.resources['lesson_buff'] = 757.0
    env.runtime.score = 10_000_000.0
    env.runtime.parameter_stats = (50_000.0, 50_000.0, 50_000.0)

    bounded = env._build_observation()

    assert env.observation_space['global'].contains(bounded['global'])
    assert env.observation_space['action_features'].contains(bounded['action_features'])
    assert env.observation_space['action_mask'].contains(bounded['action_mask'])
    assert bounded['global'][18] < 1.0
