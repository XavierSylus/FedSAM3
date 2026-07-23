# FedSAM3 服务器训练就绪自动化工作流规范

## Goal

在不改变已完成的阶段 1A–1D 核心契约的前提下，完成阶段 1E–1M 的根因重构与静态审查，使 FedSAM3 多模态联邦医学图像分割项目成为可安全提交目标服务器训练的候选版本。

目标达成必须同时满足：

1. 客户端上传、逐参数 U/R 聚合、`private_cases` 样本权重、parameter/buffer 边界、same-loss FedProx、五类参数组有效性、conflict/drift 诊断与可复现性均符合本规范的明确契约。
2. 四个主实验（U-FedAvg、U-FedProx、R-FedAvg、R-FedProx）和 2/3 客户端参与比例实验仅保留已声明的可比变量，并保存完整 manifest、环境、随机状态及审计结果。
3. 每个改动批次在改动前通过约束门禁、创建并推送不可覆盖的 GitHub 备份标签，且最多涉及三个文件；改动后完成静态验证、提交、推送和远端确认。
4. 本机绝不执行训练、smoke 或模型前向/反向；仅由目标服务器依次完成 S1–S6 验证。仅当所有验证通过后，服务器才可进入 S7 正式实验。

任何一项未满足时，候选版本必须保持 `BLOCKED`，不得提交服务器正式训练。

## 1. 目的与范围

本规范约束从当前基线到可提交服务器正式训练之间的每一个 FedSAM3 多模态联邦医学图像分割版本。

- 基线提交：`545d0c14f4f0b19d556384be09f94c58b41f7447`。
- 唯一目标远端：`origin`（`XavierSylus/FedSAM3`）。
- 本机职责：只读审查、静态检查、配置/模式校验、测试定义审查、版本控制及推送确认。
- 服务器职责：全部可执行测试、模型前向/反向、smoke、过拟合检查及正式实验。
- 禁止兼容性分支、静默降级、隐式状态补齐和基于模态关键词推断上传参数。

在第 5–8 节的所有实现门禁通过，且第 11 节的发布门禁满足前，不得启动服务器训练。

## 2. 不可变更的全局契约

### 2.1 模型与任务契约

- 分割输出严格为 `[WT, TC, ET]` 顺序的三个 logits。
- BraTS 标签只能通过 `0/1/2/4 <-> [WT, TC, ET]` 互转；预测重建必须保持区域嵌套约束。
- `L_seg = Dice + BCEWithLogits`，target 的 shape 与 dtype 必须符合已声明契约。
- 每个通道使用配置中显式声明的 sigmoid 阈值（当前为 `0.5`）。
- 空区域 Dice、IoU、HD95 使用固定的已声明规则，并且必须报告，不能隐藏。
- 禁止将 logits 硬截断为 `-20`，也禁止任何等价的常数输出行为。

### 2.2 客户端与局部目标契约

除非另有已批准的实验规范，否则实验严格包含以下三个客户端：

| 客户端 | 模态 | 局部任务目标 |
| --- | --- | --- |
| `client_1` | `text_only` | `L_text` |
| `client_2` | `image_only` | `L_seg` |
| `client_3` | `multimodal` | `L_seg + lambda_cream * L_cream` |

`L_text` 使用配置化、可微的目标：由 `text_proj` 归一化输出与固定公共多模态 proxy 数据生成且 detach 的全局目标构成。TextOnly 不得使用随机噪声作为监督。

### 2.3 参数、优化器、上传与 buffer 契约

- 仅模型已注册且由参数组注册表分类的可训练命名参数可参与聚合。
- 对客户端 `k`，上传键集合必须严格等于其显式命名的优化器参数集合 `O_k`。
- 每个上传参数必须能相对于本轮全局快照计算 delta：`Delta_k,p = theta_k,p - theta_t,p`。
- 禁止上传任何非优化器参数，包括仅由模态名称启发式选中的参数。
- FedProx 仅对 `O_k` 生效；参考参数必须是全客户端共享的本轮开始全局参数快照。
- 不得上传或聚合任何 named buffer。
- 每个客户端局部训练开始前必须接收相同的服务器 buffer 快照。局部训练改变的 buffer 在训练后丢弃；聚合后服务器恢复并保留自己的 buffer 快照。
- RoPE 等非持久且确定性的 buffer，必须通过模型确定性重建机制生成。

### 2.4 路由与权重契约

所有客户端权重均为正整数，单位为 `private_cases`：客户端私有训练病例/3D volume 数。不得使用 batch 数、slice 数、patch 数、optimizer step 数、epoch 数或公共 proxy 数据量。

在 unrestricted 路由（U）中，所有活跃客户端均进入每个参数的分母：

`Delta_U,p = sum(k in K, n_k * Delta_k,p) / sum(k in K, n_k)`。

若 `p` 不属于 `O_k`，则 `Delta_k,p = 0`，但 `n_k` 仍必须进入分母。

在 restricted 路由（R）中，仅同时真实优化 `p` 且被参数组模态白名单允许的客户端参与：

`K_R,p = { k | p in O_k and modality(k) in allowlist(group(p)) }`。

`Delta_R,p = sum(k in K_R,p, n_k * Delta_k,p) / sum(k in K_R,p, n_k)`。

即使合格客户端的数值 delta 为零，它也必须参与 R。若 `K_R,p` 为空，服务器保持 `theta_t,p`，并写入显式 aggregation audit；不得回退到任何其他聚合规则。

参数组仅限 `TEXT_ADAPTER`、`TEXT_PARAMS`、`VISION_ADAPTER`、`IMAGE_PARAMS`、`FUSION_PARAMS`。每一个可训练参数必须且只能分类一次；未分类参数必须 fail-fast。R 模式下 `FUSION_PARAMS` 仅允许 Multimodal；TextOnly 不得参与视觉组，ImageOnly 不得参与文本组。

### 2.5 实验可比性契约

四个主实验仅允许在路由方式和是否加入 FedProx 项之间不同：

| 单元 | 路由 | 局部目标 |
| --- | --- | --- |
| U-FedAvg | U | 局部任务目标 |
| U-FedProx | U | 局部任务目标 + proximal 项 |
| R-FedAvg | R | 局部任务目标 |
| R-FedProx | R | 局部任务目标 + proximal 项 |

FedProx 不得改变数据、前向计算、任务损失、优化器参数集合或服务端 FedAvg 数学规则。`mu = 0` 时，其任务损失和梯度必须与 FedAvg 一致。不得跨轮复用 optimizer 或 scheduler state。

每次运行必须记录：模型/Python/NumPy/PyTorch/CUDA seed；病例 ID；sampler、slice 选择、数据增强、客户端顺序及 proxy batch 顺序状态；optimizer 和 scheduler 初始状态；AMP 与确定性算法设置；Git commit；Python/PyTorch/CUDA/cuDNN/GPU 环境；配置与数据 manifest 的 SHA256；以及每个客户端的 `private_case_count`。

## 3. 必须遵循的实现顺序

仅当前一阶段已提交、推送并完成静态审查后，才能进入下一阶段。

1. **1E — 逐参数 U/R 聚合：** 从命名优化器参数推导上传对象；实现 U 零更新稀释、R 合格性与重新归一化；拒绝未分类可训练参数。
2. **1F — 样本权重单位：** 强制 `aggregation.sample_weight_unit: private_cases`；推导、对齐、校验和审计客户端病例数。
3. **1G — buffer 隔离：** 分离参数下发/聚合与 buffer 所有权；移除完整 `state_dict` 补齐作为聚合路径。
4. **1H — 参数组真实有效性：** 审计完整的 `forward -> optimizer -> gradient -> delta -> upload -> eligibility -> aggregation` 链路，并对预期活动组断裂 fail-fast。
5. **1I — 诊断：** 仅使用真实重叠的优化器/上传命名参数对齐 drift 和 conflict；记录组级服务端参与者与归一化权重。
6. **1J — same-loss FedProx：** 仅增加优化器参数上的 proximal 项，不得因 U/R 改变任务损失。
7. **1K — 可复现性：** 所有随机状态必须显式、可推导、可序列化，并能在四格实验中比较。
8. **1L — 配置与 manifest：** 以显式 `routing_mode`、聚合策略、样本权重单位、buffer 策略、公式及实验矩阵替代含义模糊的字段。
9. **1M — 测试迁移：** 为新契约新增测试。未经用户批准的逐项清单，不得删除或废弃冲突的旧测试。

## 4. 自动化工作流状态

每个版本迭代只能处于下列一个状态，状态转换单向进行，并记录到迭代记录中：

`PLANNED -> PREFLIGHT_PASSED -> BACKUP_PUSHED -> IMPLEMENTED -> STATIC_VALIDATED -> COMMITTED -> REMOTE_VERIFIED -> SERVER_QUEUED -> SERVER_VALIDATED -> TRAINING_APPROVED`

任一门禁失败，状态变为 `BLOCKED`；必须记录失败原因、证据、责任方和下一步所需动作。处于 `BLOCKED` 的迭代不得合并、不得作为训练候选推送，也不得提交服务器训练。

## 5. 每次迭代前的强制约束门禁

任何代码或配置改动前，必须创建并填写包含下列字段的迭代记录：

```yaml
iteration_id: phase-<stage>-<letter>-<yyyymmdd>-<nnn>
base_commit: <不可变的 40 位 SHA>
target_remote: origin
target_branch: main
scope:
  requirement_ids: []
  files_to_modify: []       # 修改与新增合计最多三个文件
  files_to_add: []
  files_to_delete: []       # 未获单独批准时必须为空
contracts_affected: []
data_flow:
  input: []
  transformation: []
  output: []
side_effects: []
acceptance_tests: []
local_prohibited_actions:
  - training
  - smoke_run
  - model_forward
  - model_backward
status: PLANNED
```

仅当以下条件全部满足，约束门禁才可通过：

1. 工作区干净。
2. 已执行 `git fetch origin --prune`，并确认 `HEAD == origin/main`。
3. 改动是某一阶段需求最短、直接的实现，不含兼容或降级路径。
4. 已在迭代记录中写明完整数据流变化及全局副作用。
5. 计划修改与新增文件合计不超过三个；超过时必须在实现前拆分成独立迭代。
6. 未计划删除文件；若需要删除，必须先给出精确清单和理由并获得用户明确批准。
7. 已枚举受影响的数学契约、配置字段、审计字段及静态验收检查。
8. 已在 `base_commit` 创建新的不可变备份标签 `backup/pre-<iteration_id>`，并在任何改动前推送至 `origin`。

在用户批准该迭代书面方案之前，禁止开始实现。Git 备份是改动前置条件，而不是改动后的补救手段。

## 6. 实现期间的约束

- 仅修改迭代记录中已批准的文件；需要扩展范围时，必须重新进入迭代前门禁。
- 全部超参数与实验策略必须配置化，禁止硬编码。
- 配置字段、运行时行为与 metadata/audit 输出之间必须有直接映射。
- 禁止新增编码旧契约的测试，特别是完整状态自动补齐、含义模糊的 decoupled 路由、或 ImageOnly 上传未优化文本参数。
- 发现需要改变行为的 bug 时，先编写或定义可复现该 bug 的测试，再做根因修复。
- 每个新增审计字段均须记录含义与单位。参数组有效性必须记录 `present_in_model`、`forward_grad_seen`、`in_optimizer`、`nonzero_gradient`、`nonzero_delta`、`uploaded`、`aggregation_eligible`、`aggregated`。
- conflict 只能用双方共同、真实优化且已上传的同名参数计算。无共同键必须记为 `N/A`，不得记为零冲突。

## 7. 本地静态验证门禁

实现后，本地验证严格限于静态检查。允许检查语法、import、AST、YAML/JSON 模式、配置完整性、测试定义、diff 与 Git metadata；禁止实例化模型、执行模型前向/反向、训练或 smoke。

静态验证记录必须确认：

- 所有改动的 Python 文件均可解析。
- YAML 与 JSON 可解析，必填字段完整，四格/比例实验矩阵使用预期的显式字段。
- 上传集合可从代码中确认由命名优化器参数构建，且排除 buffer。
- U/R 公式、空合格集合行为、病例权重单位与 buffer 所有权已在代码/配置/audit 中明确表达，不依赖模糊旧 flag。
- FedProx 仅引用优化器参数，并且不因路由方式改变任务损失。
- 每个预期可训练参数组均被分类，否则 fail-fast。
- 迭代 diff 仅包含获批文件，且没有无关变动。
- 计划在服务器执行的测试覆盖该需求的构造性与反例情形。

若某旧测试已经过时或与新契约冲突，只能将其写入待删除清单，不得在本地删除。

## 8. 提交与远端确认门禁

仅在静态验证通过后，按以下顺序执行：

1. 使用迭代 ID 与需求 ID 提交已批准的改动。
2. 推送该提交至 `origin/main`，绝不 force push。
3. fetch `origin`，确认 `origin/main` 等于新的不可变 commit SHA。
4. 确认迭代前备份标签在本地及 `origin` 均解析到 `base_commit`。
5. 将 commit SHA、备份标签、远端确认结果、修改文件及静态验证证据追加到迭代记录。

交接时必须报告备份引用、新 commit 与远端确认结果。除非用户明确授权，不得使用其他 remote。

## 9. 必需审计产物

训练候选版本最少必须输出：

- Run metadata：可复现性契约、配置/数据 manifest SHA256、Git SHA、运行环境及各客户端 `private_case_count`。
- 每轮 aggregation audit：routing mode、parameter key 数、buffer key 数、每个参数的合格客户端 ID、未归一化/归一化样本权重、U 零更新纳入情况及空合格集合事件。
- 参数组有效性 audit：按客户端/模态/组记录完整链路与计数。
- `parameter_group_diagnostics.jsonl` 与 `parameter_group_diagnostics.csv`：客户端 drift、相对 drift、RMS、参数数量、numel、非零比例、样本权重、两两 conflict、共同键数/numel、模态组合、服务端 drift 及真实参与客户端。
- 训练历史、最佳/最终 checkpoint、WT/TC/ET Dice/IoU/HD95 与空区域计数。

## 10. 仅在服务器执行的验证流水线

服务器接收已完成远端确认的候选版本，并按以下顺序执行门禁。任一步失败都必须停止流水线，附带证据回到新的本地迭代。

1. **S1 — 契约测试：** 分割、标签、文本损失/梯度、U/R 算术、优化器上传、buffer 下发、FedProx same-loss、参数组有效性、诊断、实验矩阵与可复现性。
2. **S2 — 参数组预检：** 仅最小 batch；验证预期的有限非零梯度/delta、上传键等于优化器键、所有上传键均可分类、U/R 参与者正确。任何预期活动组在整轮均为 `grad is None` 时必须停止。
3. **S3 — 单客户端小样本过拟合：** 固定 slice，使用一至三个病例，在 ImageOnly 或 Multimodal 上验证。要求 loss 明显下降、logits 非常数、WT 有非零前景、分割头/相关 adapter 有非零更新。
4. **S4 — 一轮三客户端 smoke：** 验证全部客户端、病例数、U/R 集合、客户端顺序变化下的 buffer 隔离、checkpoint/metadata/history，以及透明的空区域指标。
5. **S5 — U/R 构造性对照：** 相同初始化、seed、batch 和局部更新；验证仅服务端聚合不同，且 U 稀释/R 重新归一化完全符合契约。
6. **S6 — FedAvg/FedProx 构造性对照：** 验证初始任务损失相同、初始 proximal 项为零、参数移动后 proximal 项非零、局部任务损失独立于路由、服务端 FedAvg 公式不变。
7. **S7 — 正式实验矩阵：** 依次执行 U-FedAvg、U-FedProx、R-FedAvg、R-FedProx，以及配置化的 2/3 客户端参与比例实验。

## 11. 正式训练放行门禁

仅当下列问题全部回答“是”时，候选版本才可开始服务器正式训练：

- 精确候选 commit 是否已存在于 `origin/main`，且 `origin` 上是否存在不可变的改动前备份标签？
- 1E–1M 的每项需求是否都已实现、静态审查并覆盖服务器测试？
- 运行时产物是否能审计参数上传、U/R 路由、private-case 权重与 buffer 隔离？
- 在 S2 中，每个预期客户端参数组是否都展示了有效的 forward 到 aggregation 链路？
- S3–S6 是否通过，且没有隐藏 fallback、客户端顺序泄漏、全背景坍塌或可比性约束失败？
- 五个实验配置与 manifest 是否显式、模式合法、相互可比，并绑定候选 SHA？
- 是否已配置保存全部必需运行产物与诊断？

任一项回答“否”，状态必须维持 `BLOCKED`，不得启动服务器正式训练。

## 12. 每版本交接模板

```markdown
迭代：<iteration_id>
需求：<例如 1E-1、1E-2、1E-3>
基线提交：<SHA>
origin 上的备份标签：<tag> -> <base SHA>
已批准文件范围：<最多三个文件>
数据流与副作用：<摘要>
静态检查：<命令/结果；无训练或模型执行>
提交：<new SHA>
origin/main 确认：<confirmed SHA>
下一服务器门禁：<S1–S7>
阻塞项/待批准删除项：<无或逐项清单>
```

每一个迭代以及最终服务器训练候选版本交接，均必须使用该模板。
