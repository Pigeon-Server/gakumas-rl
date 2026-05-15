## 基本要求
* 所有回复必须使用简体中文。
* 代码注释必须使用中文。
* Python 新增或修改的方法必须补充类型标注和 Python 官方风格文档注释。
* 不理解需求、训练目标不明确、样本不足、规则冲突或会造成破坏性变更时，必须停止并询问用户。
* 不得伪造测试通过、伪造训练结果、隐藏失败原因、删除失败样本或只汇报最好的单次结果。

## 项目定位
* 本目录是独立的 Gakumas RL 训练与模拟包，默认工作目录为 `train/gakumas_rl`。
* 训练主线为无 LLM、无规则老师的自举流程：
  `Masked RL 从零探索 -> 固定 seed 轨迹选优 -> masked BC 自我蒸馏 -> RL 微调 -> 循环自举`。
* autopilot 默认应从局部到完整流程编排课程：`初中间考试 -> 初最终考试 -> NIA中间考试 -> NIA最终考试 -> NIA选拔 -> 初全流程 -> NIA全流程`。
* 默认全流程阶段应先走低难再热启动到高难：`first_star_regular -> first_star_master`、`nia_pro -> nia_master`。
* 训练数据、轨迹、评估结果必须可复现，关键流程需要显式 seed、显式配置和可落盘摘要。
* 不得重新引入 LLM 作为训练数据来源、reward shaping 来源或在线决策老师。
* 不得重新引入已删除的本地 Torch 训练后端；PyTorch 仍可作为 SB3、RLlib、BC 的底层依赖使用。

## 参考资料
修改训练流程、环境机制、动作空间、奖励或主数据解释前，应优先查看：
* `README.md`
* `TRAINING_GUIDE.md`
* `docs/DEVELOPER_HANDBOOK.md`
* `docs/AUTO_TRAINING_GUIDE.md`
* `docs/help_content_pages/categories`
* `assets/README.md`
* `src/repository/master_data.py`
* `src/simulation/`

如果训练入口、默认流程或产物格式发生变化，必须同步更新 README、训练指南或开发者手册中对应内容。

## 训练后端约定
* `SB3` 是默认后端，用于单机调试、短训、全自动自举和可复现实验。
* `RLlib` 必须保留，用于多环境并行采样、大规模 CPU 吞吐和长训。
* 不应为了“统一”删除 RLlib；本项目环境模拟偏 CPU-bound，多 worker 并行是重要能力。
* 不允许新增第三个训练后端，除非先说明维护成本、产物格式、测试方案和迁移路径。
* 自举流程优先保持 SB3 路线稳定；如接入 RLlib 探索阶段，必须保持轨迹 JSONL、评估摘要和推荐 checkpoint 格式一致。
* `--backend torch`、本地 `.pt` 策略 runner 和 `simple_trainer` 不得恢复为正式训练路径。

## 环境与动作约束
* 训练必须依赖 `action_mask` 约束合法动作；模型、BC、轨迹筛选都不得学习 mask 外动作。
* 环境 reset 后必须保证 `action_mask` 至少有一个合法动作。
* 新增动作、效果、卡牌特征或主数据字段时，应优先从游戏主数据库派生语义特征，不要在训练逻辑里写死游戏文本。
* 游戏对象在程序内部应优先使用主数据库 id，不要依赖本地化文本或 OCR 文案。
* 动作特征、全局观测、reward 字段的变更必须考虑旧 checkpoint、BC 数据集和 replay 工具的兼容性。
* 如果必须破坏兼容性，必须记录版本变化、影响范围和迁移方式。

## 训练流程与可观测性
* 所有正式训练都应输出 run 目录，并保存 checkpoint、评估日志和训练元数据。
* 冷启动训练前应先做 dry-run 或 preflight，确认 observation、reward、action mask 和 reset/step 正常。
* 训练结果不得只看单次 reward，应同时检查固定 seed 指标、非法动作数、clear/perfect、BC masked accuracy 和回放行为。
* 自举轨迹筛选必须优先拒绝非法动作轨迹，不能把 mask 外动作样本用于 BC。
* 评估必须使用固定 seed 集；更换 seed 集时需在摘要中明确记录。
* 长训参数调整需要说明目标：提高采样吞吐、稳定学习、降低 reward 噪声，还是扩大泛化场景。

## 代码组织
* 训练相关代码放在 `src/training/`。
* 环境和模拟规则放在 `src/simulation/`。
* 主数据读取、taxonomy、资源路径放在 `src/repository/`。
* 无状态环境装配和对外接口放在 `src/interfaces/`。
* 不要把训练、环境、数据读取、可视化回放混在同一个文件。
* 复杂数据结构优先使用 `dataclass`，避免用含义不明的 tuple 或层层嵌套 dict 传递。
* 路径处理必须使用 `pathlib.Path`，不要写死本机绝对路径、用户名或私有目录。
* 导入顺序保持：标准库、第三方库、项目内部模块。
* 禁止使用 `from xxx import *`。
* 禁止裸 `except:`；捕获异常后必须记录上下文或向上抛出。

## 文档与说明
* README 只写真实可用的安装、运行、训练和调试入口，不写营销式描述。
* 训练文档应明确区分 SB3 和 RLlib 的用途，不要暗示 GPU 能解决 CPU-bound 环境吞吐问题。
* 文档示例命令必须与当前 CLI 参数一致。
* 删除或改名参数时，必须同步更新 README、训练指南、开发者手册和相关测试。
* 未完成能力必须标注状态，不得写成已经稳定可用。

## 测试要求
* 修改代码后必须运行相关测试，并在交付时说明实际命令和结果。
* 推荐基础回归命令：
  `../../.venv/bin/python -m pytest -q --ignore=tests/test_rllib_model.py`
* 修改 RLlib 相关代码时，应单独运行或说明未运行：
  `../../.venv/bin/python -m pytest -q tests/test_rllib_model.py`
* 修改 CLI、训练后端或自举流程时，至少运行对应的轻量测试和参数解析检查。
* 文档-only 修改可以不跑完整测试，但交付时必须明确说明未运行测试及原因。
* 测试代码不得依赖执行顺序、隐藏缓存、本机绝对路径或残留训练产物。

## 产物与版本控制
* `runs/`、`.pytest_cache/`、`.gakumas_rl_cache/`、模型 checkpoint、临时轨迹和本地调试文件不应作为代码改动提交。
* 不得提交密钥、Token、Cookie、账号信息或个人本机路径。
* 不得随意删除已有测试、文档或样本；如果必须删除，需说明原因和替代方案。
* 在脏工作区中工作时，不得回滚用户已有改动；只处理与当前任务相关的文件。

## 交付回复要求
完成任务后必须说明：
* 修改了哪些文件。
* 实现了什么能力或规范。
* 参考了哪些资料。
* 新增或更新了哪些常量。
* 新增或更新了哪些测试。
* 实际运行的测试命令和结果。
* 是否还有未解决问题或后续风险。

不得只回复“已完成”。
