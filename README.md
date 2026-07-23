# FedSAM3-Cream

> 最后更新：2026-04-24
> 当前口径：以仓库真实代码为准的主线版本 26.1

FedSAM3-Cream 是一个面向 BraTS MRI 异构联邦医学分割的研究仓库。当前代码主线不是通用 MRI+CT 联合分割，也不是完整多任务平台；它围绕 `image_only` 与 `multimodal` 客户端，在 SAM3 医学分割集成模型上验证服务端解耦聚合、全局图文表征记忆和文本知识注入。

当前唯一权威训练入口是：

- `main.py`

当前权威配置系统是：

- `src/config_manager.py` 中的 `FederatedConfig`

当前主实验配置是：

- `configs/exp_group_a.yaml`
- `configs/exp_group_b.yaml`
- `configs/exp_group_c.yaml`

仓库中仍保留历史入口、旧脚本、旧配置和部分漂移测试。对外说明、论文叙事和复现实验应以 `main.py + configs/exp_group_[abc].yaml` 为准。

## 当前主线

主链路如下：

1. `main.py` 解析 CLI，加载 YAML，并应用覆盖项。
2. `src/config_manager.py` 展平配置、校验参数和实验协议。
3. `data/heterogeneous_dataset_loader.py` 按客户端模态装载 BraTS 切片、mask 与文本特征。
4. `src/federated_trainer.py` 创建客户端训练器、执行联邦轮次、验证、日志和 checkpoint。
5. `src/client.py` 通过 `ImageOnlyTrainer` / `MultimodalTrainer` / `TextOnlyTrainer` 执行本地更新。
6. `src/server.py` 中的 `CreamAggregator` 聚合客户端权重并维护全局图文表征。
7. `src/integrated_model.py` 中的 `SAM3MedicalIntegrated` 承载 SAM3/mock SAM3、Adapter、融合头和文本 prompt 注入。

需要明确：

- 当前 A/B/C 主实验默认只使用 `image_only` 和 `multimodal`。
- `text_only` 是框架能力和测试覆盖对象，不是当前 A/B/C 默认激活客户端。
- 当前三通道 mask 语义是 `BG / WT / ET`，不是完整独立的 `WT / TC / ET` 闭环。

## 实验协议

ICASSP 投稿实验使用严格的 `restricted routing × FedProx` 2×2 协议，并增加客户端级缺模态比例设置。完整的模型结构、上传—聚合—下发流程、变量控制和诊断定义见：

- `docs/ICASSP_EXPERIMENT_PROTOCOL.md`
- `configs/icassp_experiment_manifest.json`

### 主实验 A/B/C

| Group | 配置 | 客户端 | 聚合与开关 | 目的 |
|---|---|---|---|---|
| A | `configs/exp_group_a.yaml` | `image_only` | `fedavg`, `use_decoupled_agg=false`, `lambda_cream=0.0` | 纯视觉联邦基线 |
| B | `configs/exp_group_b.yaml` | `image_only + multimodal` | `fedavg`, `use_decoupled_agg=false`, `lambda_cream=0.1` | 文本参与但不启用路由约束，用于观察跨模态污染 |
| C | `configs/exp_group_c.yaml` | `image_only + multimodal` | `fedavg`, `use_decoupled_agg=true`, `lambda_cream=0.1` | 主方法：服务端白名单路由与文本知识注入 |

共同主超参数：

- `img_size = 256`
- `num_classes = 3`
- `batch_size = 1`
- `accumulation_steps = 4`
- `effective_batch_size = 4`
- `learning_rate = 5e-5`
- `rounds = 60`

Group B 与 Group C 的区别不在于是否存在多模态客户端，而在于是否启用服务端参数白名单路由。Group C 仍以 `fedavg` 为基础聚合方式，但在服务端按参数类型筛选合法上传方。

### 扩展 baseline 与消融

当前仓库还存在两条已经进入配置层和训练层的扩展实验：

| 名称 | 配置 | 真实代码开关 | 用途 |
|---|---|---|---|
| Baseline D / FedProx | `configs/exp_baseline_d_fedprox.yaml` | `baseline.method=fedprox`, `baseline.mu=0.01` | 异构客户端 FedProx 外部 baseline |
| C without global rep update | `configs/ablation_c_wo_global_rep_update.yaml` | `server.disable_global_rep_update=true` | 关闭全局表征刷新，验证 server-side memory 作用 |

这两条是扩展 baseline/消融，不替代 A/B/C 主实验协议。

## 核心方法实现

### 服务端解耦聚合

`src/server.py` 的 `CreamAggregator._get_participating_clients_dynamic()` 按参数名执行白名单路由，当前主要路由类别包括：

- `TEXT_ADAPTER`
- `VISION_ADAPTER`
- `TEXT_PARAMS`
- `IMAGE_PARAMS`
- `COMPAT_FALLBACK`

目标是：

- 文本侧上传不污染视觉参数池。
- 图像侧上传不污染文本参数池。
- 没有合格上传方时保留全局值，不用错误平均覆盖模型。

在 `use_decoupled_agg=false` 时，训练编排器关闭路由白名单，让 Group B 保留“文本参数参与全量聚合”的污染对照语义。

### 全局图文表征

服务端维护：

- `global_image_rep`
- `global_text_rep`

它们通过 EMA 更新，用于为联邦轮次提供全局共识记忆，并为图文对齐和文本 prompt 注入提供稳定锚点。`server.disable_global_rep_update=true` 会冻结这一刷新路径，用于消融。

### FedProx baseline

`src/config_manager.py` 会从 YAML 的 `baseline:` 段读取：

- `baseline.method`
- `baseline.mu`

当前允许值为 `none` 和 `fedprox`。当 `baseline.method=fedprox` 时，`src/federated_trainer.py` 会为每轮保存全局参考参数，`src/client.py` 在可上传可训练参数上加入 FedProx 近端约束。

## 数据与指标口径

`data/heterogeneous_dataset_loader.py` 当前将 BraTS mask 转为三通道：

- `channel 0 = BG`
- `channel 1 = WT`
- `channel 2 = ET`

`src/metrics.py` 当前稳定评估口径主要是：

- `WT`
- `ET`

因此当前代码不能被描述成完整独立的 `WT / TC / ET` 三输出系统。若后续要恢复独立 `TC` 通道，需要同步修改数据标签构造、模型输出语义、指标还原逻辑和可视化链路。

## 核心文件

| 文件 | 当前职责 |
|---|---|
| `main.py` | 当前唯一权威训练入口 |
| `src/config_manager.py` | YAML 展平、字段映射、基础校验、A/B/C 协议校验、FedProx 配置校验 |
| `src/federated_trainer.py` | 联邦训练主循环、客户端调度、协议校验、日志、验证、checkpoint、运行元数据 |
| `src/integrated_model.py` | `SAM3MedicalIntegrated`、SAM3/mock 封装、Adapter 注入、融合头、Text Prompt Encoder |
| `src/client.py` | `BaseClientTrainer`、`ImageOnlyTrainer`、`MultimodalTrainer`、`TextOnlyTrainer`、FedProx penalty |
| `src/server.py` | `CreamAggregator`、参数白名单路由、全局图文表征 EMA、聚合安全守卫 |
| `src/cream_losses.py` | 分割损失与对比损失 |
| `src/metrics.py` | WT/ET 分割指标与 HD95 |
| `data/heterogeneous_dataset_loader.py` | 异构客户端数据加载与切片策略 |
| `scripts/extract_main_table.py` | 从 `server_data/group_[abc]/training_history.json` 提取 A/B/C 主表和训练曲线 |
| `data_processing/plot_table4_endpoint_comparison.py` | 基于硬编码 A/B/C/D 结果生成 Table 4 endpoint 对比图 |

## 快速开始

最小 smoke run：

```bash
python main.py --config configs/exp_group_a.yaml --use_mock --rounds 1 --batch_size 1 --device cpu
```

当前主实验：

```bash
python main.py --config configs/exp_group_a.yaml
python main.py --config configs/exp_group_b.yaml
python main.py --config configs/exp_group_c.yaml
```

扩展 baseline 与消融：

```bash
python main.py --config configs/exp_baseline_d_fedprox.yaml
python main.py --config configs/ablation_c_wo_global_rep_update.yaml
```

常用 CLI 覆盖项：

- `--data_root`
- `--rounds`
- `--batch_size`
- `--lr`
- `--lambda_cream`
- `--seed`
- `--use_mock`
- `--device`

说明：

- `main.py` 会先完成参数与配置预检查，再延迟导入训练栈。
- `run_training.py` 仍在仓库中，但属于历史入口，不应作为当前主线使用说明。

## 结果与复现

每次训练会在配置中的 `log_dir` 下产出：

- `checkpoints/latest_checkpoint.pth`
- `checkpoints/final_model.pth`
- `checkpoints/training_history.json`
- `checkpoints/run_metadata.json`
- `tensorboard/`

当前实现会显式记录：

- 随机种子
- 训练轮次与损失曲线
- 验证指标
- Python / PyTorch / CUDA 环境信息
- 设备信息
- Git commit
- 配置摘要与协议摘要

这部分是复现链路的一部分，不是附属信息。

## 当前论文结果口径

README 中记录的 A/B/C 主结果基于 3 个随机种子：

- `3407`
- `3408`
- `3409`

| Group | Client Setting | Aggregation | Best Dice | Final Dice | Final HD95 | Avg. Grad Conflict |
|---|---|---|---:|---:|---:|---:|
| A | image-only | FedAvg | 0.8464 ± 0.0090 | 0.8328 ± 0.0147 | 12.013 ± 0.663 | - |
| B | image-only + multimodal | FedAvg | 0.8687 ± 0.0133 | 0.8649 ± 0.0153 | 10.416 ± 0.956 | 83.855 ± 2.157 |
| C | image-only + multimodal | FedAvg + routed aggregation | 0.8689 ± 0.0139 | 0.8649 ± 0.0158 | 10.288 ± 1.138 | 83.778 ± 2.714 |

`data_processing/plot_table4_endpoint_comparison.py` 中还记录了 A/B/C/D endpoint 图的硬编码统计：

- A Final Dice / HD95: `0.8328 ± 0.0147` / `12.013 ± 0.663`
- B Final Dice / HD95: `0.8649 ± 0.0153` / `10.416 ± 0.956`
- C Final Dice / HD95: `0.8649 ± 0.0158` / `10.288 ± 1.138`
- D Final Dice / HD95: `0.8843 ± 0.0071` / `9.128 ± 0.891`

该脚本输出：

- `results/paper_figures/table4_endpoint_comparison.png`
- `results/paper_figures/table4_endpoint_comparison.pdf`

## 论文主表提取

若需从主实验结果提取 A/B/C 主表与曲线图，先确保以下文件存在：

- `server_data/group_a/training_history.json`
- `server_data/group_b/training_history.json`
- `server_data/group_c/training_history.json`

然后执行：

```bash
python scripts/extract_main_table.py --data_dir server_data
```

当前真实脚本输出为：

- `server_data/main_table.csv`
- `server_data/training_curves.png`

注意：当前仓库没有 `results/paper_tables/` 目录，不应把该目录写成现有权威输出。

## 测试建议

不建议直接把 `pytest -q` 当作仓库健康检查入口，当前测试收集会被以下文件或依赖影响：

- `tests/smoke_test.py`
- `overfit_test.py`
- `core_projects/SAM-Adapter-PyTorch-main/.../test_time_aug.py`
- `tests/test_text_fusion.py`

建议分层执行：

```bash
pytest -q tests/test_aggregation_routing.py tests/test_server_aggregation_guard.py tests/test_experiment_protocol.py
pytest -q tests/test_integrated_model.py tests/test_phase_b_preflight.py tests/test_ema_decoupled_update.py
pytest -q tests/test_fedprox_config.py
```

环境敏感测试：

```bash
pytest -q tests/test_amp_and_dataset_fix.py
pytest -q tests/test_logging.py
```

手动 smoke：

```bash
python main.py --config configs/exp_group_a.yaml --use_mock --rounds 1 --batch_size 1 --device cpu
```

## 当前已知限制

1. 测试收集未收口
   `tests/smoke_test.py` 在模块顶层执行并可能 `sys.exit(1)`，`overfit_test.py` 在收集阶段可能直接占用 GPU。

2. 旧测试与当前实现漂移
   `tests/test_text_fusion.py` 依赖已不存在的旧接口，不能作为当前主线健康信号。

3. 入口双轨并存
   `main.py + src/config_manager.py` 是当前主线；`run_training.py + src/config.py` 属于历史路径，仓库中仍有残留引用。

4. 状态字典键名契约仍待统一
   当前实现保留了 `adapter_manager.` / `fusion_head.` 等前缀，而部分测试要求去除这些前缀。

5. 配置治理仍需收口
   YAML 中存在的字段只有被 `FederatedConfig.from_yaml()` 吸收并被训练链路消费后，才算真实实验变量。

## 项目结构

```text
.
├── main.py
├── configs/
├── data/
├── data_processing/
├── docs/
├── logs/
├── results/
├── scripts/
├── server_data/
├── src/
├── tests/
└── core_projects/
```

## 使用约束

- 对外叙事必须以 `main.py` + `configs/exp_group_[abc].yaml` 为准。
- FedProx 与全局表征冻结是扩展 baseline/消融，不替代 A/B/C 主协议。
- 不要再把 CT 或 `text_only` 默认客户端写进当前 A/B/C 主实验叙事。
- 不要把当前实现写成完整独立的 `WT / TC / ET` 三输出系统。
- 新实验必须保留随机种子、运行环境和 Git commit。
- 文档、方法图和论文描述应以当前主线代码为准，而不是历史残留脚本。
