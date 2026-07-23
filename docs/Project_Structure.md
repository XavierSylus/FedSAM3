# FedSAM3-Cream 项目文件与脚本详细指南

这份指南详细列出了 `FedSAM3-Cream` 项目中**每一个文件和脚本的作用**，并明确为你分类了**哪些是必须保留的核心文件**，**哪些开发/调试脚本是可以安全删除的**。

---

## 📁 1. 核心源代码目录 (`src/`) —— **【全部必须保留】**

`src/` 包含了联邦学习、医学图像分割和模型加载的核心逻辑。

### 核心模型与架构

* `model.py` / `integrated_model.py`：定义了全局模型的前向传播逻辑。`integrated_model` 通常是集成了 SAM3 与微调/适配器机制后的完整网络。
* `client.py` / `integrated_client.py`：联邦学习客户端逻辑，处理本地数据加载、本地训练、损失计算（如焦距损失、Dice损失）。
* `server.py` / `server_manager.py`：联邦学习服务端逻辑，负责聚合客户端上传的模型权重（如联邦平均 FedAvg）。
* `federated_trainer.py`：联邦学习的主循环控制器，统筹全局通信轮次（Communication Rounds）、下发模型和评估过程。
* `sam3_components_loader.py`：专用脚本，用于加载预训练的 SAM3 (Segment Anything Model 3) 组件。

### 算法与工具模块

* `cream_losses.py`：实现了各类损失函数，如 Focal Loss、Dice Loss 等，用于处理 BraTS 医疗图像的类别不平衡。
* `contrastive_aggregation.py`：服务端对比聚合模块，可能使用了 CREAM 框架相关的对比学习聚合策略。
* `knowledge_distillation.py`：基于知识蒸馏的代码，用于在联邦场景中指导本地模型或统一表征。
* `config.py` / `config_manager.py`：配置管理器，用于解析和检查 `configs/` 中的参数设置（如训练轮数、学习率、数据集路径）。
* `metrics.py`：评估指标库，包含准确率、召回率、Dice系数、IoU 等医学体素分割评估函数。
* `logger.py` / `saver.py`：日志记录与模型保存（Checkpoints）模块。
* `visualization.py`：训练过程和分割结果的可视化辅助工具。

### 子目录

* `src/data/`：**【必须保留】**
  * `brats_loader.py`：专门用于加载和处理 BraTS 数据集的 Dataloader。
  * `dataset_wrappers.py` / `text_aware_dataset.py`：由于你的模型引入了文本提示（Text Prompt），这些是处理文本和图像多模态输入的数据集包装器。
  * `client_split.py` / `partition_data.py`：用于将全局 BraTS 数据划分为多个非独立同分布（Non-IID）的客户端本地数据集。
  * `normalization.py`, `slice_extractor.py`, `prompt_generator.py`：数据预处理、3D切片提取和提示词生成工具。
* `src/models/`：**【必须保留】**包含 `adapter.py`（Lora/Adapter注入逻辑）、`freeze_utils.py`（冻结 SAM3 主干网络的工具）、`text_fusion.py`（文本与视觉特征融合的模块）。
* `src/agent/`：Agent 相关逻辑，若项目接入了某些大语言模型分析节点，这部分作为支撑。

---

## 📁 2. 运行与操作脚本 (`scripts/`) —— **【部分保留，部分可清理】**

这个目录包含了你的主要训练启动脚本和大量历史调试脚本。

### ✅ 必须保留的脚本（日常使用）

* `main_federated.py` / `train_brats_federated.py`：**核心入口！** 用于启动完整的联邦学习训练流程。
* `prepare_brats_federated_data.py`：数据预处理总脚本，在第一次训练前必须运行，用于对原始 BraTS 进行划定。
* `split_train_val_test.py`：划分训练集、验证集和测试集。
* `train_with_wandb.py`：如果使用 Weights & Biases 记录实验曲线，这是带 Wandb 支持的训练脚本。
* `evaluate_model.py`：用于在测试集上评估已训练完毕的最终模型。
* `view_training_results.py` / `visualize_results.py` / `plot_metrics.py`：绘制损失曲线和性能指标图表的工具。
* `setup_serial_clients.py` / `serial_training_utils.py` / `use_dataset_wrappers.py`：配置串行模拟联邦环境和数据集的相关辅助程序。
* `QUICK_START_训练脚本.md` / `快速评估指南.md`：文档文件，保留以供参考。

### ❌ 可以删除或归档的脚本（历史调试/验证用途）

如果你目前的系统已经稳定运行，**以下脚本可以安全删除**（或者建议你把它们移动到一个独立的 `deprecated_tests/` 文件夹中以防万一想看）：

* `sanity_check_overfit.py` / `test_focal_overfit.py` / `test_class_detection.py`：之前我们用来测试模型在此前小批量数据上能否跑通和过拟合的单步测试脚本。
* `diagnose_rope.py` / `debug_adapter_dim.py` / `test_rope_reset.py`：先前由于 RoPE (旋转位置编码) 维度错误和 Adapter 不匹配时创建的专门侦查脚本。错误修好后已经不需要了。
* `verify_data_allocation.py` / `verify_fixes.py` / `verify_fusion_integration.py` / `check_data_leakage.py`：验证数据泄露和模块融合是否正确的过渡性检验脚本。
* `debug_visualization.py` / `load_sam3_components_example.py` / `agent_inference_example.py`：早期用于观察组件加载或可视化样例的零散代码。

---

## 📁 3. 根目录文件与脚本 (`/`) —— **【部分保留，部分可清理】**

### ✅ 必须保留的文件和目录

* `main.py` / `run_training.py`：看你的使用习惯，如果作为项目的主入口，需要保留。实际上它们多数调用的都是 `src` 的逻辑。
* `configs/`目录（含 `exp_baseline.yaml` 等）：**核心训练参数配置**，必须保留。
* `data/` 或 `results/` 本地文件夹：包含实际的数据和模型保存处，切勿删除。
* `mcp_servers/` / `tools_factory/` / `tools_server.py`：这部分是你构建 MCP（Model Context Protocol）工具集以及大模型服务器交互的库，涉及到架构基础，必须保留。
* `setup_env.sh` / `requirements.txt`：环境配置文件，必须保留用于服务器重新部署。
* `docs/`：文档目录，必须保留。
* `README.md` 等项目说明文档：包含核心介绍和运行说明，必须保留。

### ❌ 可以删除的废弃/验证脚本（根目录）

根目录下通常不应堆积执行脚本，以下均为你可以**直接删除**的历史残留物：

* `check_pytorch.py`：测试 PyTorch CUDA 环境的简单脚本。
* `debug_adapter_injection.py` / `debug_visualization.png`：早期的测试注入和可视化输出文件。
* `test_model.py` / `test_trunk_fix.py` / `verify_arch.py` / `verify_fix.py` / `verify_fusion_integration.py` / `verify_sam3_access.py`：全都是前期为了排查基础架构（如修复 SAM3 forward 中的 assert 报错）所写的打游击脚本。**由于目前架构已经通过这些测试并稳定，建议一并删除！**
* 各类本地输出的日志乱码和结果：`overfit_log.txt`, `overfit_log_utf8.txt`, `overfit_result.txt`, `overfit_synthetic.txt`, `focal_overfit_result.txt`, `training_log*.txt`, `smoke_verify.log` 等日志文件可以直接清空或删除。

---

## 📁 4. 其他结构 (`tests/` 等)

* **`tests/`** 目录：包含了 `smoke_test.py`, `test_io_optimization.py`, `test_text_fusion.py` 等标准的单元测试代码。
  * **建议：保留。** 因为它们是标准的模块化测试代码，未来如果你修改了某些核心底层逻辑，可以通过运行 `pytest tests/` 来确定你有没有搞坏原有框架，这有助于项目长期维护。

## 💡 总结与建议的整理方式

为了保持工程清晰并便于长期维护，建议采取以下步骤：

1. 取消跟训练无关的 `txt` 和 `log` 的追踪。
2. 在根目录建立一个名字叫 `.trash` 或 `archived_debugs/` (已归档调试) 的文件夹。
3. 把上述列出的**所有标为 ❌ 可以删除的脚本**丢进去，这不会影响你的完整训练流程，但会让你打开项目时目录一目了然！
