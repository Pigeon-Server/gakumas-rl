"""自主学习训练方案实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math


@dataclass
class CurriculumStage:
    """课程学习阶段配置。"""

    name: str
    scenario: str
    target_score: int
    timesteps: int
    reward_mode: str = 'score'
    pretrained_checkpoint: Path | None = None


@dataclass
class SelfPlayConfig:
    """自我对弈配置。"""

    enabled: bool = False
    opponent_pool_size: int = 5  # 保留最近N个checkpoint作为对手
    update_frequency: int = 50_000  # 每N步更新一次对手池
    win_rate_threshold: float = 0.6  # 胜率超过此值时更新对手


@dataclass
class RewardShapingConfig:
    """奖励塑形配置。"""

    shape_weight: float = 1.0
    goal_weight: float = 1.25
    eval_weight: float = 1.10
    archetype_weight: float = 0.95
    risk_weight: float = 0.80
    efficiency_weight: float = 0.35
    invalid_action_penalty: float = -0.25


class AutonomousLearningPipeline:
    """自主学习训练流程。"""

    def __init__(
        self,
        curriculum_stages: list[CurriculumStage],
        self_play_config: SelfPlayConfig | None = None,
        reward_shaping_config: RewardShapingConfig | None = None,
    ):
        """初始化自主学习流程。

        Args:
            curriculum_stages: 课程学习阶段列表
            self_play_config: 自我对弈配置
            reward_shaping_config: 奖励塑形配置
        """
        self.curriculum_stages = curriculum_stages
        self.self_play_config = self_play_config or SelfPlayConfig()
        self.reward_shaping_config = reward_shaping_config
        self.opponent_pool: list[Path] = []
        self.current_stage_index = 0

    def get_current_stage(self) -> CurriculumStage:
        """获取当前训练阶段。"""
        return self.curriculum_stages[self.current_stage_index]

    def advance_stage(self) -> bool:
        """推进到下一阶段。

        Returns:
            是否成功推进（False表示已是最后阶段）
        """
        if self.current_stage_index < len(self.curriculum_stages) - 1:
            self.current_stage_index += 1
            return True
        return False

    def should_update_opponent_pool(self, current_step: int) -> bool:
        """判断是否应该更新对手池。"""
        if not self.self_play_config.enabled:
            return False
        return current_step % self.self_play_config.update_frequency == 0

    def add_opponent(self, checkpoint_path: Path) -> None:
        """添加对手到对手池。"""
        self.opponent_pool.append(checkpoint_path)
        # 保持对手池大小
        if len(self.opponent_pool) > self.self_play_config.opponent_pool_size:
            self.opponent_pool.pop(0)

    def get_random_opponent(self) -> Path | None:
        """随机选择一个对手。"""
        if not self.opponent_pool:
            return None
        import random
        return random.choice(self.opponent_pool)

    def _state_number(self, state: dict[str, Any], *keys: str, default: float = 0.0) -> float:
        """按候选 key 顺序读取状态中的数值字段。"""

        for key in keys:
            value = state.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    def _state_plan_family(self, state: dict[str, Any]) -> str:
        """把通用状态中的 plan type 归并成资源流派。"""

        mapping = {
            'ProducePlanType_Plan1': 'sense',
            'ProducePlanType_Plan2': 'logic',
            'ProducePlanType_Plan3': 'anomaly',
        }
        return mapping.get(str(state.get('plan_type') or ''), 'common')

    def _score_value_curve(self, ratio: float) -> float:
        """把得分比压成边际递减的终局价值近似。"""

        normalized = max(float(ratio), 0.0)
        progress = min(normalized, 1.0)
        overshoot = max(normalized - 1.0, 0.0)
        return progress + (math.log1p(overshoot * 3.0) / math.log(4.0)) * 0.35

    def _resource_curve(self, value: float, soft_cap: float) -> float:
        """对资源库存做边际递减压缩。"""

        clipped = max(float(value), 0.0)
        if clipped <= 0.0:
            return 0.0
        return min(math.log1p(clipped) / math.log1p(max(soft_cap, 1.0)), 1.5)

    def _state_score_ratio(self, state: dict[str, Any]) -> float:
        """读取局面对目标分数的进度比。"""

        score = self._state_number(state, 'score')
        target = max(self._state_number(state, 'target_score', 'target', default=1.0), 1.0)
        return score / target

    def _state_turn_progress_ratio(self, state: dict[str, Any]) -> float:
        """读取局面的已消耗回合进度。"""

        turn = self._state_number(state, 'turn')
        max_turns = max(self._state_number(state, 'max_turns', 'turns', default=1.0), 1.0)
        return min(max(turn / max_turns, 0.0), 1.0)

    def _state_remaining_turn_ratio(self, state: dict[str, Any]) -> float:
        """读取局面的剩余回合占比。"""

        turn = self._state_number(state, 'turn')
        max_turns = max(self._state_number(state, 'max_turns', 'turns', default=1.0), 1.0)
        return max(max_turns - turn + 1.0, 0.0) / max_turns

    def _state_stamina_ratio(self, state: dict[str, Any]) -> float:
        """读取局面的体力占比。"""

        stamina = self._state_number(state, 'stamina')
        max_stamina = max(self._state_number(state, 'max_stamina', default=1.0), 1.0)
        return stamina / max_stamina

    def _state_turn_window_value(self, state: dict[str, Any]) -> float:
        """把当前颜色窗口换算成 reward 可用价值。"""

        if str(state.get('battle_kind') or state.get('mode') or '') == 'lesson':
            return 0.0
        score_bonus = self._state_number(state, 'score_bonus_multiplier', default=1.0)
        base_bonus = self._state_number(
            state,
            'exam_score_bonus_multiplier',
            'base_score_bonus_multiplier',
            default=score_bonus if score_bonus > 0 else 1.0,
        )
        ratio = score_bonus / max(base_bonus, 0.25)
        return max(min(math.tanh((ratio - 1.0) * 1.4), 0.75), -0.75)

    def _state_judging_alignment(self, state: dict[str, Any]) -> float:
        """从通用状态里粗略估算三维与审查权重的匹配程度。"""

        if str(state.get('battle_kind') or state.get('mode') or '') == 'lesson':
            return 1.0 / 3.0
        stats = [
            self._state_number(state, 'vocal'),
            self._state_number(state, 'dance'),
            self._state_number(state, 'visual'),
        ]
        weights = [
            self._state_number(state, 'vocal_weight'),
            self._state_number(state, 'dance_weight'),
            self._state_number(state, 'visual_weight'),
        ]
        stat_sum = sum(max(value, 0.0) for value in stats)
        weight_sum = sum(max(value, 0.0) for value in weights)
        if stat_sum <= 1e-6 or weight_sum <= 1e-6:
            return 1.0 / 3.0
        normalized_stats = [max(value, 0.0) / stat_sum for value in stats]
        normalized_weights = [max(value, 0.0) / weight_sum for value in weights]
        return sum(stat * weight for stat, weight in zip(normalized_stats, normalized_weights, strict=False))

    def _phi_goal(self, state: dict[str, Any]) -> float:
        """潜势函数：离主要目标还有多远。"""

        score_ratio = self._state_score_ratio(state)
        progress = min(score_ratio, 1.0)
        pace_gap = progress - self._state_turn_progress_ratio(state)
        clear_state = str(state.get('clear_state') or 'ongoing')
        finish_bonus = 0.0
        if clear_state == 'cleared':
            finish_bonus = 0.35
        elif clear_state == 'perfect':
            finish_bonus = 0.80
        return progress * 1.15 + pace_gap * 0.55 + finish_bonus

    def _phi_eval(self, state: dict[str, Any]) -> float:
        """潜势函数：当前局面对应的终局评价边际。"""

        score_ratio = self._state_score_ratio(state)
        alignment = self._state_judging_alignment(state)
        turn_window = self._state_turn_window_value(state)
        return self._score_value_curve(score_ratio) * (0.70 + alignment * 0.60) + turn_window * 0.45

    def _phi_archetype(self, state: dict[str, Any]) -> float:
        """潜势函数：按流派评估资源的可兑现价值。"""

        remaining_turn_ratio = self._state_remaining_turn_ratio(state)
        delayed_scale = 0.35 + 0.65 * remaining_turn_ratio
        window_scale = 0.65 + 0.35 * max(self._state_turn_window_value(state), 0.0)
        family = self._state_plan_family(state)
        if family == 'sense':
            return (
                self._resource_curve(self._state_number(state, 'parameter_buff'), 8.0) * (1.10 * window_scale)
                + self._resource_curve(self._state_number(state, 'review'), 14.0) * (0.95 * delayed_scale)
                + self._resource_curve(self._state_number(state, 'parameter_buff_multiple_per_turn'), 3.0) * delayed_scale
                + self._resource_curve(self._state_number(state, 'lesson_buff'), 10.0) * 0.45
            )
        if family == 'logic':
            return (
                self._resource_curve(self._state_number(state, 'aggressive'), 18.0) * (1.05 * delayed_scale)
                + self._resource_curve(self._state_number(state, 'block'), 18.0) * (0.90 * delayed_scale)
                + self._resource_curve(self._state_number(state, 'lesson_buff'), 10.0) * 0.35
                + self._resource_curve(self._state_number(state, 'stamina_consumption_down'), 6.0) * 0.25
            )
        if family == 'anomaly':
            full_power_point = self._state_number(state, 'full_power_point')
            full_power_progress = self._resource_curve(full_power_point, 10.0)
            if full_power_point >= 10.0:
                full_power_progress += 0.25
            return (
                self._resource_curve(self._state_number(state, 'concentration'), 2.0) * (0.95 * window_scale)
                + self._resource_curve(self._state_number(state, 'preservation'), 3.0) * (0.85 * delayed_scale)
                + self._resource_curve(self._state_number(state, 'over_preservation'), 2.0) * delayed_scale
                + full_power_progress * 1.15
                + self._resource_curve(self._state_number(state, 'enthusiastic'), 10.0) * (0.65 + 0.35 * max(window_scale, delayed_scale))
            )
        return (
            self._resource_curve(self._state_number(state, 'parameter_buff'), 6.0) * 0.65
            + self._resource_curve(self._state_number(state, 'review'), 10.0) * 0.55
            + self._resource_curve(self._state_number(state, 'aggressive'), 10.0) * 0.55
            + self._resource_curve(self._state_number(state, 'lesson_buff'), 10.0) * 0.40
        )

    def _phi_risk(self, state: dict[str, Any]) -> float:
        """潜势函数：负面状态、断线风险和节奏压力。"""

        negative_penalty = (
            self._state_number(state, 'sleepy') * 0.30
            + self._state_number(state, 'panic') * 0.28
            + self._state_number(state, 'slump') * 0.24
            + self._state_number(state, 'active_skill_forbidden') * 0.35
            + self._state_number(state, 'mental_skill_forbidden') * 0.35
        )
        tempo_pressure = max(self._state_turn_progress_ratio(state) - min(self._state_score_ratio(state), 1.20), 0.0)
        low_stamina = max(0.35 - self._state_stamina_ratio(state), 0.0) / 0.35
        return self._state_stamina_ratio(state) * 0.75 - negative_penalty - tempo_pressure * 0.90 - low_stamina * 0.80

    def _phi_efficiency(self, state: dict[str, Any]) -> float:
        """潜势函数：主要目标可达后，鼓励保留更多余量。"""

        score_ratio = self._state_score_ratio(state)
        gate = max(min((score_ratio - 0.85) / 0.15, 1.0), 0.0)
        if gate <= 0.0:
            return 0.0
        spare_value = self._state_stamina_ratio(state) * 0.70 + self._state_remaining_turn_ratio(state) * 0.45
        return gate * spare_value - max(score_ratio - 1.0, 0.0) * 0.20

    def _potential_value(self, state: dict[str, Any], config: RewardShapingConfig) -> float:
        """统一的 potential-based shaping 状态势函数。"""

        return (
            config.goal_weight * self._phi_goal(state)
            + config.eval_weight * self._phi_eval(state)
            + config.archetype_weight * self._phi_archetype(state)
            + config.risk_weight * self._phi_risk(state)
            + config.efficiency_weight * self._phi_efficiency(state)
        )

    def compute_shaped_reward(
        self,
        base_reward: float,
        state_before: dict[str, Any],
        state_after: dict[str, Any],
        action_info: dict[str, Any],
    ) -> float:
        """计算塑形后的奖励。

        Args:
            base_reward: 基础奖励
            state_before: 动作前状态
            state_after: 动作后状态
            action_info: 动作信息

        Returns:
            塑形后的奖励
        """
        config = self.reward_shaping_config
        if config is None:
            return base_reward
        shaped_reward = base_reward + config.shape_weight * (
            self._potential_value(state_after, config) - self._potential_value(state_before, config)
        )
        if action_info.get('invalid_action'):
            shaped_reward += config.invalid_action_penalty
        return shaped_reward

    def save_progress(self, save_path: Path) -> None:
        """保存训练进度。"""
        progress = {
            'current_stage_index': self.current_stage_index,
            'opponent_pool': [str(p) for p in self.opponent_pool],
            'curriculum_stages': [
                {
                    'name': stage.name,
                    'scenario': stage.scenario,
                    'target_score': stage.target_score,
                    'timesteps': stage.timesteps,
                }
                for stage in self.curriculum_stages
            ],
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open('w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    def load_progress(self, load_path: Path) -> None:
        """加载训练进度。"""
        if not load_path.exists():
            return

        with load_path.open('r', encoding='utf-8') as f:
            progress = json.load(f)

        self.current_stage_index = progress.get('current_stage_index', 0)
        self.opponent_pool = [Path(p) for p in progress.get('opponent_pool', [])]


def create_default_curriculum() -> list[CurriculumStage]:
    """创建默认的课程学习阶段。"""
    return [
        CurriculumStage(
            name='入门阶段',
            scenario='first_star_regular',
            target_score=1500,
            timesteps=300_000,
            reward_mode='clear',
        ),
        CurriculumStage(
            name='进阶阶段',
            scenario='first_star_pro',
            target_score=1800,
            timesteps=300_000,
            reward_mode='score',
        ),
        CurriculumStage(
            name='高级阶段',
            scenario='first_star_master',
            target_score=2000,
            timesteps=400_000,
            reward_mode='score',
        ),
        CurriculumStage(
            name='大师阶段',
            scenario='nia_master',
            target_score=2200,
            timesteps=500_000,
            reward_mode='score',
        ),
    ]


def create_autonomous_pipeline(
    enable_self_play: bool = False,
    enable_reward_shaping: bool = True,
) -> AutonomousLearningPipeline:
    """创建自主学习流程。

    Args:
        enable_self_play: 是否启用自我对弈
        enable_reward_shaping: 是否启用奖励塑形

    Returns:
        配置好的自主学习流程
    """
    curriculum = create_default_curriculum()

    self_play_config = SelfPlayConfig(
        enabled=enable_self_play,
        opponent_pool_size=5,
        update_frequency=50_000,
    ) if enable_self_play else None

    reward_shaping_config = RewardShapingConfig() if enable_reward_shaping else None

    return AutonomousLearningPipeline(
        curriculum_stages=curriculum,
        self_play_config=self_play_config,
        reward_shaping_config=reward_shaping_config,
    )
