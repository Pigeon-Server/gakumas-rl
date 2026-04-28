# Gakumas RL Standalone

Gakumas 强化学习工具组

```bash
python scripts/copy_project_assets.py
```

## 安装

```bash
pip install -e .
pip install -e .[env,torch]
```

如果还需要 API 或训练后端，再补装：

```bash
pip install -e .[api]
pip install -e .[sb3]
pip install -e .[rllib]
```

## 运行示例

```bash
python -m gakumas_rl.train --mode exam --scenario nia_master --dry-run
python -m gakumas_rl.train --mode planning --scenario first_star_master --dry-run
gakumas-rl --mode exam --scenario nia_master --print-observation --dry-run
python -m gakumas_rl.demo_exam --checkpoint runs/sb3_exam_nia_master_xxx/checkpoints/step_500000.zip --scenario nia_master
```

默认训练输出目录改为当前包下的 `runs/`。

## Exam Reward Mode

`exam` 模式现在支持两种训练目标：

- `--exam-reward-mode score`：偏最终分数，鼓励尽量打高分
- `--exam-reward-mode clear`：偏过线率、体力效率和资源节奏，过线后会抑制无意义的过量输出，并显式惩罚睡意/恐慌/低迷等负面状态

`clear` 模式不会跳过真实考试规则；除了继续读取场地 gimmick、负面状态和资源库存外，也会保留考试回合颜色、审查基准和 NIA fan vote 对得分效率的影响，因此训练时仍会学到“为了强势回合留牌/留资源、为了避免负面效果调整出牌顺序”这类行为。

## 目录说明

- `gakumas_rl/`：独立后的 Python 包
- `scripts/copy_project_assets.py`：从当前仓库复制主数据资源到独立包
- `assets/README.md`：资源目录说明
- `README.source.md`：从原模块复制来的说明原文，保留作对照

## 可视化回放

```bash
python -m gakumas_rl.demo_exam \
  --checkpoint runs/sb3_exam_nia_master_xxx/checkpoints/step_500000.zip \
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
