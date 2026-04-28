# 训练启动说明

## 1. 安装依赖

先安装基础依赖，再按训练后端补装：

```bash
pip install -e .
pip install -e .[env,torch]
pip install -e .[sb3]
pip install -e .[rllib]
```

如果你只打算跑 SB3，至少要有：

```bash
pip install -e .[sb3]
pip install sb3-contrib
```

## 2. 训练后端选择

当前可用后端：

- `sb3`：单机训练，默认已切到 `MaskablePPO`
- `rllib`：多 worker / 可扩展训练，内部已做动作掩码
- `torch`：仓库内的轻量调试后端

## 3. 先分清你要训练的是什么

这个项目里常见的训练目标分三类：

- `--mode exam`：只训练考试/出牌/战斗决策
- `--mode lesson` / `--mode battle`：训练局部课程或混合局部战斗能力
- `--mode planning`：训练**完整培育流程**，包括周行动、体力、P点、构筑、考前准备和阶段考试

如果你要的是**全套培育训练**，请使用：

```bash
--mode planning
```

不要误用 `exam`。`exam` 只适合训练局部战斗能力，不是完整养成。

## 4. 直接开始训练

### SB3：局部考试训练

```bash
python -m gakumas_rl.train \
  --backend sb3 \
  --mode exam \
  --scenario nia_master \
  --total-timesteps 1000000 \
  --rollout-steps 512 \
  --learning-rate 1e-4 \
  --checkpoint-freq 10000 \
  --eval-freq 10000
```

### RLlib：局部考试训练

```bash
python -m gakumas_rl.train \
  --backend rllib \
  --mode exam \
  --scenario nia_master \
  --total-timesteps 1000000 \
  --rllib-num-workers 4 \
  --rllib-num-envs-per-worker 1 \
  --rllib-train-batch-size 4000 \
  --rllib-minibatch-size 256 \
  --rllib-num-epochs 4
```

### SB3：完整培育流程训练（初）

```bash
python -m gakumas_rl.train \
  --backend sb3 \
  --mode planning \
  --scenario first_star_master \
  --total-timesteps 10000000 \
  --rollout-steps 512 \
  --learning-rate 1e-4 \
  --checkpoint-freq 50000 \
  --eval-freq 50000 \
  --auto-resume
```

### SB3：完整培育流程训练（NIA）

```bash
python -m gakumas_rl.train \
  --backend sb3 \
  --mode planning \
  --scenario nia_master \
  --total-timesteps 15000000 \
  --rollout-steps 512 \
  --learning-rate 1e-4 \
  --checkpoint-freq 50000 \
  --eval-freq 50000 \
  --auto-resume
```

### 自动训练模式

如果你想让程序自动估算训练步数，并动态调评估/保存频率，可以加：

```bash
--auto-train
```

例如：

```bash
python -m gakumas_rl.train \
  --backend sb3 \
  --mode exam \
  --scenario nia_master \
  --auto-train \
  --auto-min-timesteps 100000 \
  --auto-max-timesteps 5000000
```

## 5. 常用场景

### NIA 完整培育

```bash
python -m gakumas_rl.train --backend sb3 --mode planning --scenario nia_master
```

### 初剧本完整培育

```bash
python -m gakumas_rl.train --backend sb3 --mode planning --scenario first_star_master
```

### 只训练考试能力

```bash
python -m gakumas_rl.train --backend sb3 --mode exam --scenario nia_master
```

### 只做 dry-run

```bash
python -m gakumas_rl.train --backend sb3 --mode exam --scenario nia_master --dry-run
```

### 查看观测/编成

```bash
python -m gakumas_rl.train --mode exam --scenario nia_master --print-observation --dry-run
python -m gakumas_rl.train --mode exam --scenario nia_master --print-loadout
```

## 6. 100 万步够不够

如果你训练的是 `exam`，100 万步可以作为一个早期检查规模，用来确认：

- 环境能跑通
- reward 没炸
- action mask 正常
- 策略开始学会基本合法动作和部分节奏

但如果你训练的是 **完整培育流程（planning）**，100 万步通常**不够收敛**。

### 对完整培育的大致预期

- `first_star_master`：通常建议从 **300万 ~ 1000万+** 步开始看
- `nia_master`：通常建议从 **800万 ~ 3000万+** 步开始看

这不是硬上限，而是经验起步量级。原因是完整培育包含：

- 周行动选择
- 体力管理
- P点管理
- 卡组构筑
- 饮料 / P道具使用节奏
- 考前准备
- 多阶段考试结果反向影响前面决策

这是一个**长流程、长信用分配链**任务，比单独 `exam` 难得多。

## 7. 推荐训练顺序

如果你的目标是最终得到完整培育策略，不建议一开始只硬训 `planning`。

更推荐：

1. 先训 `exam`
2. 再训 `lesson` / `battle`
3. 最后训 `planning`

推荐原因：

- `exam` 先学会局部战斗能力
- `lesson` / `battle` 让局部策略更稳
- `planning` 再学习长期资源调度和全流程整合

如果你有 BC 预训练权重，可以在 PPO 启动时加：

```bash
--pretrained-checkpoint <BC checkpoint>
```

## 8. 推荐起步参数

### 只想先把训练跑起来

建议：

- `--backend sb3`
- `--scenario nia_master`
- `--mode exam`
- `--total-timesteps 500000`
- `--rollout-steps 512`
- `--learning-rate 1e-4`
- `--checkpoint-freq 10000`
- `--eval-freq 10000`

### 要训完整培育流程

建议：

- `--backend sb3`
- `--mode planning`
- `--scenario first_star_master` 或 `nia_master`
- `--total-timesteps` 至少从几百万级开始
- `--rollout-steps 512`
- `--learning-rate 1e-4`
- `--checkpoint-freq 50000`
- `--eval-freq 50000`
- `--auto-resume`

如果想更稳一点，可以再加：

- `--auto-train`
- `--pretrained-checkpoint <BC checkpoint>`

## 9. 怎么判断是不是在收敛

不要只看一次 reward 抬头。建议同时看：

- evaluation reward 是否持续抬升
- 是否更稳定地跑完整局
- planning 模式下最终 `final_summary` 里的结局/评分是否提升
- 非法动作、空转、资源浪费是否下降
- checkpoint 回放里，是否能观察到更合理的体力 / P点 / 构筑节奏

对于完整培育，真正有效的信号通常包括：

- 更少中途崩盘
- 更少无意义休息或空转
- 考前牌组质量更高
- 最终 produce result 更稳定

## 10. 输出文件

训练结果默认会写到 `runs/` 下，常见内容有：

- `checkpoints/step_*.zip`
- `evaluations.jsonl`
- `artifacts.jsonl`
- `tensorboard/`

## 11. 训练时怎么判断是否在正常跑

正常启动时，你应该能看到：

- `SB3 backend requires ...` 之类的依赖报错没有出现
- `Using MaskablePPO with action masking.`
- 训练日志里会周期性打印步数、reward、评估结果
- `runs/<name>/checkpoints/` 里持续生成 checkpoint

## 12. 如果你想先验证环境

先跑 dry-run：

```bash
python -m gakumas_rl.train --backend sb3 --mode exam --scenario nia_master --dry-run
```

如果这里能正常输出动作、奖励和终止信息，再开始正式训练。

如果你打算训完整培育，也建议先做一次：

```bash
python -m gakumas_rl.train --backend sb3 --mode planning --scenario first_star_master --dry-run
```

确认 planning 环境本身没有问题后，再开长训。 
