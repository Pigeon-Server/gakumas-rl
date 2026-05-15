# Gakumas RL Standalone

Gakumas 强化学习工具组

主数据默认从当前包内的 `assets/` 读取；不要从外层项目隐式复制资源。

## 安装

```bash
pip install -e .
pip install -e .[env,sb3]
```

如果还需要 API 或训练后端，再补装：

```bash
pip install -e .[api]
pip install -e .[rllib]
```

## 运行示例

```bash
python -m src.train --mode exam --scenario nia_master --dry-run
python -m src.train --mode planning --scenario first_star_master --dry-run
python -m src.training.cli --mode exam --scenario nia_master --print-observation --dry-run
python -m src.training.autopilot --iterations 2
python -m src.training.self_bootstrap --mode lesson --scenario nia_master --iterations 2
```

默认训练输出目录改为当前包下的 `runs/`。

如果已用 `pip install -e .[env,sb3]` 安装到当前 venv，也可以使用 `gakumas-rl-autopilot` 等短命令；入口不存在时优先用上面的 `python -m ...` 写法。

## 冷启动

推荐让 autopilot 从低难到高难自动冷启动：

```bash
python -m src.training.autopilot \
  --iterations 3 \
  --rl-timesteps 131072 \
  --final-rl-timesteps 131072
```

如果完整育成阶段需要单独微调培育奖励，可以额外传入培育奖励配置文件：

```bash
python -m src.training.autopilot \
  --curriculum-start-stage 8 \
  --produce-reward-config configs/produce_reward_nia_planning_v1.json
```

默认课程是 `初中间考试 -> 初最终考试 -> NIA中间考试 -> NIA最终考试 -> NIA选拔 -> 初Regular全流程 -> 初Master全流程 -> NIA Pro全流程 -> NIA Master全流程`。流程是：每个阶段用 SB3 MaskablePPO 从零探索或热启动探索，多个 checkpoint 在固定 seed 上选最高质量轨迹，再做 SB3 原生 masked BC 自我蒸馏，最后自动接一段短 RL 微调，并把 BC checkpoint 与微调 checkpoint 一起横评后推荐更稳的模型。

训练设备默认是 `auto`，会按 `cuda -> mps -> cpu` 自动选择；Apple Silicon 上可用时会优先使用 MPS。如果完整育成阶段瓶颈在环境模拟而不是神经网络，也可以传 `--device cpu` 做对照。

如果只想跑单个剧本，显式传 `--scenario` 或 `--no-curriculum`：

```bash
python -m src.training.autopilot --no-curriculum --mode lesson --scenario nia_master --iterations 2
```

## Exam Reward Mode

`exam` 模式现在支持两种训练目标：

- `--exam-reward-mode score`：偏最终分数，鼓励尽量打高分
- `--exam-reward-mode clear`：偏过线率、体力效率和资源节奏，过线后会抑制无意义的过量输出，并显式惩罚睡意/恐慌/低迷等负面状态

`clear` 模式不会跳过真实考试规则；除了继续读取场地 gimmick、负面状态和资源库存外，也会保留考试回合颜色、审查基准和 NIA fan vote 对得分效率的影响，因此训练时仍会学到“为了强势回合留牌/留资源、为了避免负面效果调整出牌顺序”这类行为。

## 目录说明

- `src/`：独立后的 Python 包
- `assets/README.md`：资源目录说明
- `README.source.md`：从原模块复制来的说明原文，保留作对照

## 可视化回放

```bash
python -m src.demo_exam \
  --checkpoint "$(python -c 'import json; print(json.load(open("runs/self_bootstrap/bootstrap_summary.json"))["recommended_checkpoint"])')" \
  --scenario nia_master \
  --exam-reward-mode score
```

或使用安装后的命令入口：

```bash
gakumas-rl-demo --checkpoint runs/sb3_exam_nia_master_xxx/checkpoints/step_500000.zip --scenario nia_master
```

它会输出两个文件：

- `demo_*.html`：接近游戏战斗 UI 的静态回放页面
- `demo_*.json`：同一局的原始 trace，便于排查具体动作与状态

未传 `--checkpoint` 时，会退回到一个简易启发式策略，适合先检查环境和页面渲染。
