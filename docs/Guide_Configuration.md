# 实验配置文件使用指南

## 📁 配置文件概览

本项目提供了两个对比实验配置：

| 配置文件 | 客户端 | 解耦聚合 | 用途 |
|---------|--------|---------|------|
| `baseline.yaml` | Client 2 + 3 (2个) | ❌ 关闭 | 对照组 |
| `proposed_method.yaml` | Client 1 + 2 + 3 (3个) | ✅ 开启 | 实验组（我们的方法） |

## 🎯 实验设计

### Baseline 配置
- **客户端组成**:
  - Client 2: BraTS 2018 影像数据 (image_only)
  - Client 3: BraTS 2020 成对数据 (multimodal)
- **聚合方式**: 传统联邦平均（FedAvg）或对比加权聚合（CreamFL）
- **目的**: 提供性能基准

### Proposed Method 配置
- **客户端组成**:
  - Client 1: BraTS 2020 文本数据 (text_only)
  - Client 2: BraTS 2018 影像数据 (image_only)
  - Client 3: BraTS 2020 成对数据 (multimodal)
- **聚合方式**: 解耦功能聚合（Decoupled Aggregation）
  - `mask_decoder`: 仅聚合 Client 2 + 3 (有影像)
  - `text_encoder`: 仅聚合 Client 1 + 3 (有文本)
  - `adapter`: 全员聚合 (跨模态对齐)
- **目的**: 验证解耦聚合在异构数据场景下的优势

## 📊 关键参数对比

| 参数 | Baseline | Proposed Method |
|-----|----------|----------------|
| **Batch Size** | 1 | 1 |
| **Learning Rate** | 2e-4 | 2e-4 |
| **混合精度 (AMP)** | ✅ 开启 | ✅ 开启 |
| **Rounds** | 50 | 50 |
| **解耦聚合** | ❌ | ✅ |
| **客户端数量** | 2 | 3 |

## 🚀 使用方法

### 1. 运行 Baseline 实验
```bash
# 假设您的训练脚本为 main.py
python main.py --config configs/baseline.yaml
```

### 2. 运行 Proposed Method 实验
```bash
python main.py --config configs/proposed_method.yaml
```

### 3. 测试配置文件
```bash
# 验证配置文件格式
python scripts/test_configs.py
```

## 📝 配置文件结构说明

### 核心配置块

#### 1. 联邦学习配置 (`federated`)
```yaml
federated:
  clients:
    - client_id: client1
      modality: text_only
      data_source: data/federated_split/client1_text_only/dataset.json
      enabled: true
  use_decoupled_agg: true  # 是否使用解耦聚合
```

#### 2. 训练参数 (`training`)
```yaml
training:
  batch_size: 1
  learning_rate: 0.0002
  rounds: 50
  local_epochs: 1
  weight_decay: 0.01
  grad_clip: 1.0
```

#### 3. 训练选项 (`options`)
```yaml
options:
  use_amp: true  # 混合精度训练
  use_dummy: false  # 是否使用虚拟数据
```

#### 4. 日志配置 (`logging`)
```yaml
logging:
  log_type: tensorboard
  log_dir: logs/baseline
  experiment_name: FedSAM3_Baseline_Client2+3
```

## 🔧 自定义配置

### 修改学习率
```yaml
training:
  learning_rate: 0.0001  # 改为 1e-4
```

### 调整联邦轮数
```yaml
training:
  rounds: 100  # 增加到 100 轮
```

### 更换聚合方法
```yaml
server:
  aggregation_method: fedavg  # 改为标准 FedAvg
```

### 启用 WandB 日志
```yaml
logging:
  log_type: wandb  # 或 both (同时使用 tensorboard 和 wandb)
  wandb_project: FedSAM3-Cream
  wandb_entity: your_username
```

## 📈 预期实验结果

| 指标 | Baseline | Proposed Method | 预期提升 |
|-----|----------|----------------|---------|
| **Dice Score** | ~0.75 | ~0.80 | +6.7% |
| **模态利用** | 部分 | 完整 | - |
| **异构客户端支持** | ❌ | ✅ | - |

## ⚠️ 注意事项

1. **数据路径**: 确保 `data/federated_split/` 目录下已运行 `prepare_custom_split.py` 生成数据划分
2. **显存要求**: Batch Size = 1 需要约 8-12GB 显存
3. **训练时间**: 每轮约 5-10 分钟（取决于硬件）
4. **检查点**: 每 5 轮自动保存检查点到 `checkpoints/` 目录

## 📞 问题排查

### 配置加载失败
```bash
# 验证 YAML 语法
python -c "import yaml; yaml.safe_load(open('configs/baseline.yaml', encoding='utf-8'))"
```

### 找不到数据文件
```bash
# 检查数据划分是否生成
ls data/federated_split/client*/dataset.json
```

### 显存不足
```yaml
# 降低 batch size 或关闭 AMP
training:
  batch_size: 1
options:
  use_amp: false
```

## 实验执行建议

1. **运行两组实验**: 确保 baseline 和 proposed method 都完成
2. **记录关键指标**: Dice, HD95, Sensitivity, Specificity
3. **保存可视化**: 分割结果对比图、训练曲线
4. **消融实验**: 测试解耦聚合的各个组件贡献

祝实验顺利！🚀
