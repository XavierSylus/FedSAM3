# FedSAM3-Cream 技术路线

> 更新日期：2026-04-24
> 口径：根据当前仓库真实代码、配置文件和脚本整理
> 范围：主入口、配置层、训练编排器、模型集成层、客户端训练器、服务端聚合器、数据加载层、指标层、结果脚本和测试边界

## 1. 当前定位

FedSAM3-Cream 当前最准确的定位是：

一个围绕 BraTS MRI 异构联邦医学分割的研究仓库，主线已经收敛到 `main.py + FederatedConfig + A/B/C 实验协议`，核心验证目标是多模态文本知识进入联邦分割训练后，服务端如何通过参数白名单路由与全局图文表征记忆降低跨模态污染。

它当前不是：

- MRI+CT 联合分割系统
- 通用多模态医学大平台
- 完整独立 `WT / TC / ET` 三通道闭环系统
- 以历史脚本为入口的旧版联邦训练仓库

当前主线可概括为：

1. BraTS MRI 切片分割作为任务。
2. `image_only` 与 `multimodal` 作为 A/B/C 主实验客户端。
3. SAM3 医学集成模型作为本地训练模型。
4. FedAvg 作为基础聚合框架。
5. Group C 在 FedAvg 外启用服务端参数白名单路由。
6. 全局图文表征通过 EMA 形成 server-side knowledge memory。
7. 训练产物显式保存随机种子、运行环境、Git commit 和配置摘要。

## 2. 权威链路

当前权威训练入口：

- `main.py`

当前权威配置对象：

- `src/config_manager.py` 中的 `FederatedConfig`

当前权威主实验配置：

- `configs/exp_group_a.yaml`
- `configs/exp_group_b.yaml`
- `configs/exp_group_c.yaml`

当前核心训练链路：

```text
main.py
  -> FederatedConfig.from_yaml()
  -> CLI overrides
  -> FederatedTrainer
  -> create_heterogeneous_data_loaders()
  -> ImageOnlyTrainer / MultimodalTrainer / TextOnlyTrainer
  -> SAM3MedicalIntegrated
  -> CreamAggregator
  -> checkpoint / training_history / run_metadata
```

历史路径仍存在，但不应作为当前主线叙事：

- `run_training.py`
- `src/config.py`
- `scripts/train_brats_federated.py`
- 旧版 `docs/` 中部分教程

这些文件的存在本身不是问题；问题是不能再把它们写成当前权威入口。

## 3. 模块职责

| 模块 | 当前真实职责 |
|---|---|
| `main.py` | CLI 解析、YAML 加载、覆盖项应用、训练栈延迟导入、启动 `FederatedTrainer` |
| `src/config_manager.py` | 配置展平、字段默认值、基础合法性检查、A/B/C 协议检查、FedProx 配置检查 |
| `src/federated_trainer.py` | 联邦轮次编排、客户端过滤、协议校验、随机种子设置、训练/验证/日志/checkpoint、运行元数据保存 |
| `src/integrated_model.py` | `SAM3MedicalIntegrated`、真实 SAM3/mock SAM3 封装、Adapter、融合头、Text Prompt Encoder |
| `src/client.py` | `BaseClientTrainer` 与三类客户端训练器；AMP、梯度累积、梯度裁剪、FedProx penalty、模态特定上传 |
| `src/server.py` | `CreamAggregator`、参数白名单路由、全局图文表征 EMA、聚合安全守卫、梯度冲突诊断 |
| `data/heterogeneous_dataset_loader.py` | 异构客户端数据加载、BraTS 切片选择、三通道 mask 构造、文本特征加载 |
| `src/metrics.py` | WT/ET Dice 与 HD95 评估 |
| `scripts/extract_main_table.py` | 从 A/B/C `training_history.json` 提取 `main_table.csv` 和 `training_curves.png` |
| `data_processing/plot_table4_endpoint_comparison.py` | 生成 A/B/C/D endpoint comparison 图 |

## 4. 实验协议

### 4.1 A/B/C 主协议

| Group | 配置 | 客户端 | 关键开关 | 技术目的 |
|---|---|---|---|---|
| A | `configs/exp_group_a.yaml` | `image_only` | `lambda_cream=0.0`, `use_decoupled_agg=false` | 纯视觉联邦基线 |
| B | `configs/exp_group_b.yaml` | `image_only + multimodal` | `lambda_cream=0.1`, `use_decoupled_agg=false` | 文本参与但不路由约束，用于保留跨模态污染对照 |
| C | `configs/exp_group_c.yaml` | `image_only + multimodal` | `lambda_cream=0.1`, `use_decoupled_agg=true` | 主方法：服务端白名单路由与文本知识注入 |

A/B/C 的逻辑闭环是：

1. Group A 给出纯视觉下限。
2. Group B 引入文本相关训练但关闭路由约束，暴露污染风险。
3. Group C 在相同客户端结构下开启参数白名单路由，验证服务端解耦聚合效果。

### 4.2 扩展 baseline 与消融

当前仓库新增两条真实存在的扩展路线：

| 名称 | 配置 | 真实代码路径 | 技术目的 |
|---|---|---|---|
| Baseline D / FedProx | `configs/exp_baseline_d_fedprox.yaml` | `baseline.method=fedprox`, `baseline.mu=0.01` | 外部联邦优化 baseline |
| C without global rep update | `configs/ablation_c_wo_global_rep_update.yaml` | `server.disable_global_rep_update=true` | 验证全局图文表征刷新是否贡献性能 |

FedProx 已进入：

- `FederatedConfig.from_yaml()`
- `FederatedConfig.__post_init__()`
- `FederatedTrainer._train_single_round()`
- `BaseClientTrainer._compute_fedprox_penalty()`
- `tests/test_fedprox_config.py`

全局表征冻结已进入：

- `FederatedConfig.from_yaml()`
- `FederatedTrainer._train_single_round()`
- `configs/ablation_c_wo_global_rep_update.yaml`

这两条路线应写成扩展 baseline/消融，不应改写 A/B/C 主协议。

## 5. 数据流与副作用

### 5.1 配置流

```text
YAML
  -> FederatedConfig.from_yaml()
  -> flattened dataclass fields
  -> __post_init__ validation
  -> FederatedTrainer
```

重要约束：

- YAML 中写了某个字段，不等于训练主线一定消费了该字段。
- 只有被 `FederatedConfig.from_yaml()` 展平，并被后续训练链路读取的字段，才算真实实验变量。
- 当前配置层真实吸收 `training`、`cream`、`model`、`server`、`baseline`、`federated`、`options`、`logging`、`checkpoint`、`validation`、顶层 `data_root/device/seed`。

### 5.2 客户端训练流

每轮训练中，`FederatedTrainer` 会：

1. 按配置过滤启用客户端。
2. 校验 Group A/B/C 协议。
3. 为客户端加载初始权重。
4. 按客户端模态构造 optimizer 参数组。
5. 下发全局图文表征。
6. 调用对应 trainer 的 `run()`。
7. 收集权重、图像表征、文本表征和训练统计。
8. 调用服务端聚合。

模态副作用：

- `image_only` 不应上传文本参数，除非 Group B 需要保留污染对照。
- `multimodal` 可同时产生图像与文本相关训练信号。
- `text_only` 是框架能力；当前 A/B/C 默认不启用它。

### 5.3 服务端聚合流

```text
client weights + client reps + client modalities
  -> safe filter
  -> dynamic param name set
  -> optional modality-aware route
  -> weighted/fedavg aggregation
  -> safe fill missing params
  -> global model state
  -> global reps update
```

`use_decoupled_agg=true` 时，服务端根据参数名执行白名单路由：

- `TEXT_ADAPTER`: `text_only + multimodal`
- `VISION_ADAPTER`: `image_only + multimodal`
- `TEXT_PARAMS`: `text_only + multimodal`
- `IMAGE_PARAMS`: `image_only + multimodal`
- `COMPAT_FALLBACK`: 实际上传者

`use_decoupled_agg=false` 时，路由约束关闭，用于 Group B 的污染对照。

## 6. 模型路线

当前主模型是：

- `src/integrated_model.py` 中的 `SAM3MedicalIntegrated`

主要组件：

- SAM3 或 mock SAM3 后端
- `AdapterInjector`
- `MultimodalFusionHead`
- `TextPromptEncoder`
- 医学分割输出头
- 图像/文本投影路径

文本知识进入分割的主要方式是：

1. 客户端侧通过 multimodal 数据产生文本相关训练信号。
2. 服务端维护 `global_text_rep`。
3. 模型侧通过 `TextPromptEncoder` 将 `global_text_rep` 投入分割 prompt 路径。

需要注意：

- 当前实现仍保留 `adapter_manager.`、`fusion_head.` 等 state dict 前缀。
- 部分测试对键名契约的预期与当前实现不一致。
- 在未统一 `state_dict`、上传参数过滤和路由键名之前，不应在文档中声称键名契约已经完全收口。

## 7. 数据与指标路线

### 7.1 数据口径

`data/heterogeneous_dataset_loader.py` 当前 mask 构造为：

```text
channel 0 = mask == 0     -> BG
channel 1 = mask > 0      -> WT
channel 2 = mask == 4     -> ET
```

这不是完整独立 `WT / TC / ET` 三通道。

### 7.2 指标口径

`src/metrics.py` 当前将三通道输出按 `BG / WT / ET` 理解，并稳定计算：

- `WT_Dice`
- `ET_Dice`
- `WT_HD95`
- `ET_HD95`
- `Mean_Dice`
- `Mean_HD95`

如果后续要恢复完整 `WT / TC / ET`，必须同时改：

1. 数据标签构造。
2. 模型输出语义。
3. 指标还原逻辑。
4. 可视化脚本。
5. 论文表述。

不能只改文档或局部 if-else。

## 8. 结果路线

### 8.1 训练输出

训练会在 `log_dir` 下保存：

- `checkpoints/latest_checkpoint.pth`
- `checkpoints/final_model.pth`
- `checkpoints/training_history.json`
- `checkpoints/run_metadata.json`
- `tensorboard/`

`run_metadata.json` 与训练历史是复现链路的一部分，应长期保留以下信息：

- seed
- Python / PyTorch / CUDA 环境
- device
- Git commit
- config summary
- protocol summary

### 8.2 A/B/C 主表提取

`scripts/extract_main_table.py` 当前读取：

- `server_data/group_a/training_history.json`
- `server_data/group_b/training_history.json`
- `server_data/group_c/training_history.json`

实际输出：

- `server_data/main_table.csv`
- `server_data/training_curves.png`

README 里不应再写不存在的 `main_table_by_seed.csv`、`main_table_group_summary.csv` 或 `results/paper_tables/...`。

### 8.3 Table 4 endpoint 图

`data_processing/plot_table4_endpoint_comparison.py` 当前硬编码 A/B/C/D endpoint 统计并输出：

- `results/paper_figures/table4_endpoint_comparison.png`
- `results/paper_figures/table4_endpoint_comparison.pdf`

该脚本中的 D 对应 hetero FedProx：

- Final Dice: `0.8843 ± 0.0071`
- Final HD95: `9.128 ± 0.891`

这属于结果展示脚本，不是训练结果自动提取脚本。

## 9. 测试路线

当前不能把裸 `pytest -q` 当成可靠健康检查入口。原因包括：

1. `tests/smoke_test.py` 在模块顶层执行脚本逻辑并可能 `sys.exit(1)`。
2. `overfit_test.py` 可能在收集阶段直接占用 GPU。
3. `core_projects/SAM-Adapter-PyTorch-main/.../test_time_aug.py` 依赖未安装的 `mmcv`。
4. `tests/test_text_fusion.py` 依赖旧接口，已经与当前 `src/models/text_fusion.py` 漂移。

当前更接近主线的测试入口是：

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

## 10. 当前风险

### 10.1 高优先级：测试边界未收口

测试目录混合了单元测试、脚本型测试和历史测试。直接收集会产生假失败或环境失败。

应后续处理：

- 建立明确 pytest 收集边界。
- 将脚本型文件移出自动收集范围，或加入口保护。
- 下线或重写旧 API 测试。

### 10.2 高优先级：入口双轨仍存在

当前主线是 `main.py + src/config_manager.py`，但旧路径仍保留。

应后续处理：

- 文档继续只写主线入口。
- 代码治理阶段再决定旧入口归档、兼容或删除。
- 删除任何文件前必须先列出待删除清单并获得批准。

### 10.3 中优先级：state dict 键名契约未统一

当前实现与部分测试对 `adapter_manager.` / `fusion_head.` 前缀的预期不一致。

应后续处理：

- 先确定真实聚合键名契约。
- 再同步 `state_dict`、客户端上传过滤、服务端路由和测试。
- 不应通过局部补丁掩盖键名漂移。

### 10.4 中优先级：配置治理仍需加强

配置文件中存在一些字段，未必已经完整进入主线对象或训练逻辑。

应后续处理：

- 列出 YAML 字段到 `FederatedConfig` 字段的映射。
- 区分“保留字段”和“已消费字段”。
- 对未消费字段做 fail-fast、文档标注或清理。

## 11. 后续治理顺序

若后续进入代码治理，建议顺序为：

1. 收口测试边界。
2. 统一入口与配置系统。
3. 明确 `state_dict` 与聚合路由键名契约。
4. 清理或重写漂移测试。
5. 对齐数据与指标口径。
6. 再考虑删除历史文件。

其中删除或移动文件必须先给出待删除清单和理由，获得明确批准后再执行。

## 12. 结论

当前 FedSAM3-Cream 的主线已经形成：

- 入口明确。
- A/B/C 实验协议清楚。
- 服务端解耦聚合有真实代码承载。
- 全局图文表征与文本 prompt 注入链路存在。
- FedProx baseline 和全局表征冻结消融已经进入配置与训练层。
- 复现元数据记录相对完整。

但工程边界仍未完全收口：

- 测试收集不可靠。
- 历史路径仍在。
- 少数测试与实现接口漂移。
- 数据/指标口径不能过度宣传为完整 `WT / TC / ET`。

因此，当前技术路线应写成“主线清楚、实验协议清楚、仍需工程治理”的状态，而不是“生产就绪”或“完全闭环”。
