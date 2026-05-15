"""参数化奖励配置，把所有奖励权重集中管理，支持 CLI / 文件 / API 覆盖。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class RewardConfig:
    """统一奖励参数集。

    所有权重均可通过 CLI ``--reward-<snake_case>=<value>`` 或
    JSON 文件 ``--reward-config=<path>`` 覆盖。
    """

    # ── 奖励模式 ──────────────────────────────────────────────────
    #  'score' : 追求考试高评价
    #  'clear' : 追求课程过线 / 效率优先
    reward_mode: str = 'score'

    # ── Dense shaping（潜势差分）权重 ──────────────────────────────
    #  shape_scale 控制潜势差分在总奖励中的比例
    shape_scale: float = 1.85
    #  5 个潜势函数各自的权重
    goal_weight: float = 0.95        # Φ_goal：离通关目标的距离
    eval_weight: float = 1.55        # Φ_eval：终局评价边际价值
    archetype_weight: float = 1.10   # Φ_archetype：流派资源估值
    risk_weight: float = 0.70        # Φ_risk：负面状态 / 体力风险
    efficiency_weight: float = 0.28  # Φ_efficiency：收官效率

    # ── Dense 子项参数 ─────────────────────────────────────────────
    turn_window_weight: float = 0.80       # 回合颜色窗口价值权重
    judging_alignment_weight: float = 0.70  # 审查基准与当前三维匹配度权重
    efficiency_gate: float = 0.92          # Φ_efficiency 启动阈值
    efficiency_overshoot_penalty: float = 0.32  # 超线惩罚系数

    # ── 资源估值缩放（Φ_archetype）──────────────────────────────
    #  sense / logic / anomaly 流派各自资源的 soft_cap 与权重
    #  这些只是默认值；实际的 per-resource 估值在 exam_runtime 的
    #  _phi_archetype() 中已用 _resource_curve() 做了边际递减压缩，
    #  这里提供流派级别的总系数缩放。
    sense_resource_scale: float = 1.0
    logic_resource_scale: float = 1.0
    anomaly_resource_scale: float = 1.0

    # ── Sparse 终局奖励 ───────────────────────────────────────────
    terminal_pass_reward: float = 5.5      # 考试达标一次性奖励
    terminal_eval_weight: float = 5.0      # 终局分数曲线加权
    terminal_stamina_weight: float = 0.45  # 剩余体力加权
    terminal_speed_weight: float = 0.40    # 剩余回合加权
    terminal_failure_weight: float = 5.0   # 失败惩罚（越大惩罚越重）
    terminal_force_end_bonus: float = 1.8  # 达到最高分自动结束奖励
    terminal_nia_bonus: float = 1.25       # NIA fan vote 终局奖励
    lesson_clear_reward: float = 4.0       # 课程 Clear 一次性奖励
    lesson_perfect_reward: float = 6.5     # 课程 Perfect 一次性奖励
    overshoot_penalty: float = 0.95        # 超线惩罚（终局）

    # ── 截断 / 惩罚机制 ──────────────────────────────────────────
    invalid_action_penalty: float = -0.45  # 选择了不可用动作的即时惩罚
    skip_turn_penalty: float = -0.02       # 空过回合（未出牌直接结束回合）的惩罚
    stamina_death_penalty: float = -1.0    # 体力归零导致强制结束时的额外惩罚
    consecutive_end_turn_penalty: float = -0.04  # 连续结束回合的递增惩罚

    # ── 里程碑奖励（Milestone）──────────────────────────────────
    #  在分数首次达到某些关键比例时发放一次性奖励
    milestone_25_reward: float = 0.0       # 首次达到 25% 目标
    milestone_50_reward: float = 0.5       # 首次达到 50% 目标
    milestone_75_reward: float = 1.0       # 首次达到 75% 目标
    milestone_100_reward: float = 2.0      # 首次达到 100% 目标（通关）

    # ── 额外密集信号 ──────────────────────────────────────────────
    # PBRS 已经覆盖分数和资源变化，默认关闭这两项重复的线性密集奖励。
    score_delta_scale: float = 0.0
    resource_gain_scale: float = 0.0
    card_play_reward: float = 0.0          # 每次成功出牌的微小正向奖励
    drink_use_reward: float = 0.0          # 成功使用饮料的微小正向奖励

    # ── 全局缩放 ──────────────────────────────────────────────────
    reward_scale: float = 1.0              # 最终奖励值全局缩放
    reward_clip: float = 25.0             # >0 时对最终奖励做 clip(-c, c)

    def to_dict(self) -> dict[str, Any]:
        """转成普通 dict，方便序列化与日志记录。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RewardConfig':
        """从 dict 构建，忽略无关的 key。"""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> 'RewardConfig':
        """从 JSON 文件加载。"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        """保存到 JSON 文件。"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def merge(self, overrides: dict[str, Any]) -> 'RewardConfig':
        """用 dict 中的值覆盖当前配置，返回新实例。"""
        current = self.to_dict()
        valid_keys = {f.name for f in fields(self.__class__)}
        for k, v in overrides.items():
            if k in valid_keys:
                current[k] = v
        return self.__class__.from_dict(current)


# ── 预置 profile ──────────────────────────────────────────────────

SCORE_PROFILE = RewardConfig(
    reward_mode='score',
    shape_scale=2.05,  # 提高shaping权重，让模型更关注中间状态
    goal_weight=1.05,  # 提高目标权重，更关注通关
    eval_weight=1.75,  # 提高评价权重，追求高分
    archetype_weight=1.25,  # 提高流派资源权重
    risk_weight=0.85,  # 提高风险权重，避免失败
    efficiency_weight=0.35,  # 提高效率权重，鼓励快速通关
    turn_window_weight=0.95,  # 提高回合窗口权重，更好利用颜色加成
    judging_alignment_weight=0.95,  # 提高审查匹配权重
    efficiency_gate=0.88,  # 降低效率启动阈值，更早开始优化效率
    efficiency_overshoot_penalty=0.28,  # 降低超线惩罚，允许适度追分
    terminal_pass_reward=6.0,  # 提高通关奖励
    terminal_eval_weight=5.5,  # 提高终局评价权重
    terminal_stamina_weight=0.50,  # 提高体力保留奖励
    terminal_speed_weight=0.45,  # 提高速度奖励
    terminal_failure_weight=5.5,  # 提高失败惩罚
    terminal_force_end_bonus=2.0,  # 提高强制结束奖励
    terminal_nia_bonus=1.35,  # 提高NIA奖励
    lesson_clear_reward=4.5,  # 提高课程通关奖励
    lesson_perfect_reward=7.0,  # 提高完美通关奖励
    overshoot_penalty=0.85,  # 降低超线惩罚
    invalid_action_penalty=-0.75,  # 提高非法动作惩罚
    skip_turn_penalty=-0.03,  # 提高空过惩罚
    consecutive_end_turn_penalty=-0.06,  # 提高连续结束回合惩罚
    reward_clip=28.0,  # 提高奖励上限
    milestone_50_reward=0.6,  # 提高里程碑奖励
    milestone_75_reward=1.2,
    milestone_100_reward=2.5,
    score_delta_scale=0.0,
    resource_gain_scale=0.0,
)

CLEAR_PROFILE = RewardConfig(
    reward_mode='clear',
    shape_scale=1.75,  # 提高shaping权重
    goal_weight=1.60,  # 提高目标权重，更关注通关
    eval_weight=1.15,  # 提高评价权重
    archetype_weight=0.78,  # 提高资源权重
    risk_weight=1.35,  # 提高风险权重，避免失败
    efficiency_weight=0.70,  # 提高效率权重
    turn_window_weight=0.55,  # 提高窗口权重
    judging_alignment_weight=0.55,  # 提高匹配权重
    efficiency_gate=0.70,  # 降低效率启动阈值
    efficiency_overshoot_penalty=1.25,  # 降低超线惩罚
    terminal_pass_reward=8.0,  # 提高通关奖励
    terminal_eval_weight=4.2,  # 提高评价权重
    terminal_stamina_weight=1.00,  # 提高体力奖励
    terminal_speed_weight=1.05,  # 提高速度奖励
    terminal_failure_weight=7.5,  # 提高失败惩罚
    terminal_force_end_bonus=2.3,  # 提高强制结束奖励
    terminal_nia_bonus=1.20,
    lesson_clear_reward=6.0,  # 提高课程通关奖励
    lesson_perfect_reward=9.0,  # 提高完美通关奖励
    overshoot_penalty=2.20,  # 降低超线惩罚
    invalid_action_penalty=-1.10,  # 提高非法动作惩罚
    skip_turn_penalty=-0.10,  # 提高空过惩罚
    stamina_death_penalty=-2.5,  # 提高体力归零惩罚
    consecutive_end_turn_penalty=-0.18,  # 提高连续结束回合惩罚
    reward_clip=28.0,  # 提高奖励上限
    milestone_50_reward=1.0,  # 提高里程碑奖励
    milestone_75_reward=1.8,
    milestone_100_reward=3.5,
    score_delta_scale=0.0,
    resource_gain_scale=0.0,
)

REWARD_PROFILES: dict[str, RewardConfig] = {
    'score': SCORE_PROFILE,
    'clear': CLEAR_PROFILE,
}


@dataclass
class ProduceRewardConfig:
    """培育阶段（produce_runtime）的奖励参数。

    与考试阶段的 RewardConfig 独立，通过 CLI ``--produce-reward-<field>=<value>`` 覆盖。
    """

    # ── 密集塑形总开关 ───────────────────────────────────────────────
    shape_scale: float = 0.60

    # ── 势函数各维权重 ──────────────────────────────────────────────
    param_weight: float = 1.20
    fan_weight: float = 0.80
    resource_weight: float = 0.50

    # ── 资源子项权重 ───────────────────────────────────────────────
    deck_readiness_weight: float = 0.20
    drink_future_weight: float = 0.18
    drink_current_conversion_weight: float = 0.07
    pp_optionality_weight: float = 0.20
    stamina_actionability_weight: float = 0.20
    stamina_runway_weight: float = 0.15

    # ── 终局奖励 ────────────────────────────────────────────────────
    terminal_score_scale: float = 4.0
    terminal_grade_s: float = 1.5
    terminal_grade_a: float = 0.8
    terminal_grade_b: float = 0.0
    terminal_grade_c: float = -1.0
    terminal_grade_b_plus: float = 0.4
    terminal_grade_c_plus: float = -0.5
    terminal_grade_s_plus: float = 1.1
    terminal_grade_ss: float = 2.4
    terminal_grade_ss_plus: float = 2.8
    terminal_grade_sss: float = 3.3
    terminal_grade_sss_plus: float = 3.8
    terminal_grade_s4: float = 4.4
    terminal_grade_d: float = -1.4
    terminal_grade_failed: float = -2.0

    terminal_route_clear_bonus: float = 0.6
    terminal_route_fail_penalty: float = -0.6
    terminal_stage_progress_weight: float = 0.4
    terminal_pp_left_waste_penalty: float = 0.35
    terminal_fan_aux_scale: float = 0.08
    terminal_nia_param_fallback_weight: float = 0.30
    terminal_nia_vote_rank_bonus: float = 0.20

    # ── 函数内部常量 ───────────────────────────────────────────────
    score_norm_log_base: float = 10000.0
    pp_left_cap: float = 150.0
    fan_overflow_scale: float = 4000.0
    fan_progress_cap: float = 1.25
    fan_overflow_cap: float = 0.25
    fan_full_unlock_log_base: float = 8.0
    fan_unlock_log_base: float = 6.0
    param_overshoot_scale: float = 2.0
    param_overshoot_log_base: float = 3.0
    stamina_actionable_threshold: float = 8.0
    stamina_low_threshold: float = 0.35
    pre_audition_window_near: int = 2
    pre_audition_window_mid: int = 4
    pp_window_near_weight: float = 0.70
    pp_window_mid_weight: float = 0.35
    pp_window_far_weight: float = 0.10
    drink_window_near_weight: float = 1.00
    drink_window_mid_weight: float = 0.60
    drink_window_far_weight: float = 0.25
    deck_quality_soft_cap: float = 8.0

    # ── 全局 ────────────────────────────────────────────────────────
    reward_scale: float = 1.0
    reward_clip: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典，便于写入 JSON 和日志。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProduceRewardConfig':
        """从字典构建培育奖励配置，并忽略未知字段。"""

        valid_keys = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> 'ProduceRewardConfig':
        """从 JSON 文件加载培育奖励配置。"""

        with open(path, 'r', encoding='utf-8') as handle:
            return cls.from_dict(json.load(handle))

    def save_json(self, path: str | Path) -> None:
        """把培育奖励配置保存到 JSON 文件。"""

        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    def merge(self, overrides: dict[str, Any]) -> 'ProduceRewardConfig':
        """用字典覆盖部分字段，返回新实例。"""
        current = asdict(self)
        for k, v in (overrides or {}).items():
            if k in current:
                current[k] = type(current[k])(v)
        return ProduceRewardConfig(**current)


# 默认培育奖励配置
DEFAULT_PRODUCE_REWARD_CONFIG = ProduceRewardConfig(
    # 保留密集塑形，但避免让中间状态盖过终局目标。
    shape_scale=0.60,
    
    # 提高各维权重，更平衡地关注三维发展
    param_weight=1.35,
    fan_weight=0.90,
    resource_weight=0.60,
    
    # 提高资源子项权重
    deck_readiness_weight=0.25,
    drink_future_weight=0.22,
    drink_current_conversion_weight=0.09,
    pp_optionality_weight=0.25,
    stamina_actionability_weight=0.25,
    stamina_runway_weight=0.18,
    
    # 提高终局奖励，鼓励达成更高评级
    terminal_score_scale=4.5,
    terminal_grade_s=1.8,
    terminal_grade_a=1.0,
    terminal_grade_b=0.2,
    terminal_grade_c=-0.8,
    terminal_grade_b_plus=0.6,
    terminal_grade_c_plus=-0.3,
    terminal_grade_s_plus=1.4,
    terminal_grade_ss=2.8,
    terminal_grade_ss_plus=3.2,
    terminal_grade_sss=3.8,
    terminal_grade_sss_plus=4.3,
    terminal_grade_s4=5.0,
    terminal_grade_d=-1.2,
    terminal_grade_failed=-1.8,
    
    terminal_route_clear_bonus=0.8,
    terminal_route_fail_penalty=-0.8,
    terminal_stage_progress_weight=0.5,
    terminal_pp_left_waste_penalty=0.40,
    terminal_fan_aux_scale=0.10,
    terminal_nia_param_fallback_weight=0.35,
    terminal_nia_vote_rank_bonus=0.25,
)


def build_produce_reward_config(
    overrides: dict[str, Any] | None = None,
    reward_config_path: str | None = None,
) -> ProduceRewardConfig:
    """构建培育阶段奖励配置，支持文件与字典覆盖。"""

    cfg = DEFAULT_PRODUCE_REWARD_CONFIG
    if reward_config_path:
        cfg = ProduceRewardConfig.from_json(reward_config_path)
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


def build_reward_config(
    reward_mode: str = 'score',
    reward_config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> RewardConfig:
    """根据 mode / 文件 / CLI 覆盖构建奖励配置。

    优先级：overrides > config_file > profile_defaults
    """
    config = REWARD_PROFILES.get(reward_mode, SCORE_PROFILE)
    if reward_config_path:
        file_config = RewardConfig.from_json(reward_config_path)
        config = file_config
    if overrides:
        config = config.merge(overrides)
    # 确保 reward_mode 一致
    if config.reward_mode != reward_mode and not reward_config_path:
        config = config.merge({'reward_mode': reward_mode})
    return config
