# 无 LLM 冷启动训练说明

本项目训练主线不依赖 LLM、OpenAI API、Ollama，也不要求人工写规则老师。推荐流程是：

```text
Masked RL 从零探索 -> 固定 seed 轨迹选优 -> masked BC 自我蒸馏 -> RL 微调 -> 循环自举
```

## 1. 安装依赖

最小冷启动训练需要环境和 SB3：

```bash
pip install -e .
pip install -e .[env,sb3]
```

如果要使用 RLlib，可额外安装：

```bash
pip install -e .[rllib]
```

## 2. 先做环境预检

```bash
python -m src.train --mode lesson --scenario first_star_regular --print-observation --dry-run
```

需要确认：

- `action_mask` 不是全 0
- `reward` 是有限值
- 环境能 `reset/step`
- 输出里没有数值保护或非法动作告警

## 3. 推荐全自动冷启动入口

默认不传 `--scenario`、也不需要传 `--mode`，让 autopilot 自己按“局部考试 -> 完整育成”的课程自动训练：

```bash
python -m src.training.autopilot \
  --iterations 3 \
  --rl-timesteps 131072 \
  --final-rl-timesteps 131072 \
  --bootstrap-seed-start 1000 \
  --bootstrap-seed-count 64
```

如果要针对完整育成阶段单独试培育奖励，可以传 `--produce-reward-config`：

```bash
python -m src.training.autopilot \
  --curriculum-start-stage 8 \
  --produce-reward-config configs/produce_reward_nia_planning_v1.json
```

它会自动完成：

- 按 `初中间考试 -> 初最终考试 -> NIA中间考试 -> NIA最终考试 -> NIA选拔 -> 初全流程 -> NIA全流程` 编排课程
- 全流程阶段会先走低难，再热启动到高难：`first_star_regular -> first_star_master`、`nia_pro -> nia_master`
- 用 SB3 `MaskablePPO` 从零探索，并定期保存 `.zip` checkpoint
- 让多个 checkpoint 在同一批 seed 上各打一遍，保留每个 seed 的最佳轨迹
- 用最佳轨迹做 SB3 原生 masked BC，产出下一轮热启动 `.zip`
- 形状兼容时，把上一阶段推荐 checkpoint 作为下一阶段热启动
- 形状不兼容时自动跳过跨阶段热启动，例如 First Star 到 NIA 的全局观测维度不同
- 最后一轮 BC 后自动再做一段短 RL 微调
- 对 BC checkpoint 和最终 RL checkpoint 一起做固定 seed 横评，并在 `bootstrap_summary.json` 写出 `recommended_checkpoint`

如果只想跑单个剧本，显式传 `--scenario` 或 `--no-curriculum`：

```bash
python -m src.training.autopilot --no-curriculum --mode lesson --scenario nia_master --iterations 2
```

如果已经把本项目安装进当前 venv，等价短命令是 `gakumas-rl-autopilot`。遇到 `command not found` 时，说明 venv 里的 console script 没装好或指向旧项目，直接使用 `python -m src.training.autopilot` 最稳。

## 4. 常用控制参数

- `--iterations`：自举轮数
- `--rl-timesteps`：每轮 RL 探索步数
- `--final-rl-timesteps`：所有 BC 蒸馏完成后追加的最终 RL 微调步数；不传时 autopilot 自动给短微调，传 `0` 时跳过
- `--checkpoint-freq`：每隔多少步保存候选 checkpoint
- `--eval-freq` / `--eval-episodes`：训练中评估频率
- `--bootstrap-seed-start` / `--bootstrap-seed-count`：轨迹选优用的固定 seed 集
- `--bc-epochs`：每轮自我蒸馏训练轮数
- `--stochastic-eval`：轨迹选优时使用采样动作；默认使用确定性动作
- `--selection-score-cap`：最终横评时用于抗极端 outlier 的均分截断上限，推荐模型主要看非法动作、perfect/clear 数、中位分和 reward
- `--curriculum-start-stage`：从指定课程阶段继续跑，例如前 5 个考试阶段已完成后可传 `6`
- `--device`：默认 `auto`，按 `cuda -> mps -> cpu` 自动选择；想强制不用 Apple MPS 时传 `--device cpu`
- `--produce-reward-config`：完整育成使用的培育奖励 JSON；适合在 NIA/初全流程阶段做可复现的 reward 微调

## 5. 产物目录

默认写到：

```text
runs/autopilot_curriculum/<timestamp>/
```

常见文件：

- `curriculum_summary.json`：课程总摘要和最终推荐 checkpoint
- `stage_*/bootstrap_summary.json`：单阶段完整自举摘要
- `stage_*/preflight.json`：训练前环境预检结果
- `stage_*/iter_*/rl/checkpoints/step_*.zip`：SB3 每轮 RL 候选模型
- `stage_*/iter_*/selected_trajectories.jsonl`：每个 seed 选出来的最佳轨迹
- `stage_*/iter_*/trajectory_summary.json`：轨迹选优摘要
- `stage_*/iter_*/bc_distilled.zip`：本轮 SB3 BC 自我蒸馏产物
- `stage_*/final_rl/`：最终 BC 后自动 RL 微调产物
- `stage_*/final_checkpoint_evaluation.json`：最终候选 checkpoint 的固定 seed 横评

## 6. 判断是否有效

不要只看一次 reward。至少同时看：

- 固定 seed 的 `mean_score` 是否提升
- `selected_trajectories.jsonl` 中 `action_valid` 是否始终为 true
- BC 输出的 `masked_acc` 是否明显高于随机
- 回放里是否减少无意义 `end_turn`、乱用饮料或体力崩盘

如果一轮没有提升，不要盲目拉长训练；先看 reward 是否太稀、轨迹池是否全是低质量局、lesson 是否过难。

## 7. 课程顺序

默认 autopilot 不再要求人工切换 `lesson/exam/planning`，课程顺序已经内置：

```text
first_star_regular exam(mid)
-> first_star_regular exam(final)
-> nia_pro exam(mid)
-> nia_pro exam(final)
-> nia_pro exam(selection)
-> first_star_regular planning
-> first_star_master planning
-> nia_pro planning
-> nia_master planning
```

其中 `exam` 阶段只练考试局部，`planning` 阶段才包含日常课程、外出、相谈、考试/试镜等完整育成流程。完整育成信用分配更难，所以放在局部考试之后自动执行。

## 8. LLM 删除说明

训练主线已移除 LLM：

- 不再提供 LLM 轨迹生成入口
- 不再提供 LLM reward shaping
- 基础依赖不包含 `openai`
- 文档推荐流程改为自举训练
