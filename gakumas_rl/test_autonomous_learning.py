"""自主学习奖励塑形的回归测试。"""

from __future__ import annotations

from gakumas_rl.autonomous_learning import (
    AutonomousLearningPipeline,
    CurriculumStage,
    RewardShapingConfig,
)


def _pipeline(config: RewardShapingConfig | None = None) -> AutonomousLearningPipeline:
    return AutonomousLearningPipeline(
        curriculum_stages=[CurriculumStage(name='test', scenario='produce-005', target_score=2000, timesteps=1000)],
        reward_shaping_config=config,
    )


def test_compute_shaped_reward_prefers_goal_progress_over_stalling() -> None:
    """potential shaping 应显式偏好更接近目标、风险更低的转移。"""

    pipeline = _pipeline(RewardShapingConfig())
    state_before = {
        'score': 800.0,
        'target_score': 2000.0,
        'turn': 4.0,
        'max_turns': 9.0,
        'stamina': 8.0,
        'max_stamina': 12.0,
        'plan_type': 'ProducePlanType_Plan1',
        'parameter_buff': 1.0,
        'review': 2.0,
        'score_bonus_multiplier': 1.0,
        'exam_score_bonus_multiplier': 1.0,
    }
    good_after = {
        **state_before,
        'score': 1300.0,
        'turn': 5.0,
        'parameter_buff': 4.0,
        'review': 6.0,
        'score_bonus_multiplier': 1.2,
    }
    bad_after = {
        **state_before,
        'score': 900.0,
        'turn': 5.0,
        'stamina': 2.0,
        'sleepy': 2.0,
        'panic': 1.0,
        'score_bonus_multiplier': 0.85,
    }

    good_reward = pipeline.compute_shaped_reward(0.0, state_before, good_after, {})
    bad_reward = pipeline.compute_shaped_reward(0.0, state_before, bad_after, {})

    assert good_reward > 0.0
    assert good_reward > bad_reward


def test_compute_shaped_reward_penalizes_invalid_actions() -> None:
    """invalid action 应只通过稀疏事件惩罚体现。"""

    pipeline = _pipeline(RewardShapingConfig())
    state = {
        'score': 1000.0,
        'target_score': 2000.0,
        'turn': 3.0,
        'max_turns': 9.0,
        'stamina': 8.0,
        'max_stamina': 12.0,
    }

    reward = pipeline.compute_shaped_reward(0.5, state, state, {'invalid_action': True})

    assert reward == 0.25


def test_pipeline_without_reward_shaping_returns_base_reward() -> None:
    """显式关闭 shaping 时，不应再偷偷套默认奖励塑形。"""

    pipeline = _pipeline(None)
    state_before = {'score': 200.0, 'target_score': 1000.0, 'turn': 1.0, 'max_turns': 9.0}
    state_after = {'score': 800.0, 'target_score': 1000.0, 'turn': 2.0, 'max_turns': 9.0}

    reward = pipeline.compute_shaped_reward(1.25, state_before, state_after, {'invalid_action': True})

    assert reward == 1.25
