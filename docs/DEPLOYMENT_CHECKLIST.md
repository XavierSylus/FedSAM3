# FedSAM3-Cream 服务器部署检查清单

## ✅ 已完成的修改

### 1. 服务器聚合功能（src/server.py）
- ✅ 添加 `_get_participating_clients` 方法实现解耦聚合逻辑
- ✅ `aggregate_weights` 方法支持 `client_modalities` 参数
- ✅ 权重归一化逻辑已实现
- ✅ 调试日志已添加

### 2. 数据划分脚本（scripts/prepare_custom_split.py）
- ✅ 创建完成并测试通过
- ✅ 生成了 3 个客户端的数据划分
- ✅ 输出目录: `data/federated_split/`

### 3. 实验配置文件
- ✅ `configs/baseline.yaml` - Baseline 实验（2 个客户端）
- ✅ `configs/proposed_method.yaml` - 提出方法（3 个客户端 + 解耦聚合）
- ✅ YAML 格式验证通过

## ⚠️ 需要修改的部分

### 关键问题 1: 配置文件字段不兼容

**问题**: 新创建的 YAML 配置文件包含 `federated` 字段，但 `FederatedConfig` 类不支持该字段。

**影响文件**:
- `configs/baseline.yaml`
- `configs/proposed_method.yaml`
- `src/config_manager.py`

**需要做的修改**:

#### A. 修改 `src/config_manager.py`

在 `FederatedConfig` 类中添加以下字段：

```python
# ==================== 联邦学习客户端配置 ====================
use_decoupled_agg: bool = False
"""是否使用解耦功能聚合"""

clients: Optional[List[Dict[str, Any]]] = None
"""客户端配置列表"""
```

在 `from_yaml` 方法中添加解析逻辑（约第 276 行）：

```python
# 处理联邦学习配置
if 'federated' in config_dict:
    federated = config_dict['federated']
    flattened['use_decoupled_agg'] = federated.get('use_decoupled_agg', False)
    flattened['clients'] = federated.get('clients', None)
```

#### B. 修改 `src/federated_trainer.py`

在 `train_round` 方法中（约第 379 行），修改聚合调用：

```python
# 当前代码（第 379-382 行）:
aggregated_state = self.server.aggregate_weights(
    [round_client_updates[cid] for cid in client_ids_sorted],
    [round_client_reps[cid] for cid in client_ids_sorted]
)

# 修改为:
# 获取客户端模态列表
client_modalities = None
if self.config.use_decoupled_agg and self.config.clients:
    # 从配置中提取模态信息，按照 client_ids_sorted 的顺序
    client_modality_map = {c['client_id']: c['modality'] for c in self.config.clients}
    client_modalities = [client_modality_map.get(cid, 'multimodal') for cid in client_ids_sorted]

aggregated_state = self.server.aggregate_weights(
    [round_client_updates[cid] for cid in client_ids_sorted],
    [round_client_reps[cid] for cid in client_ids_sorted],
    client_modalities=client_modalities  # 新增参数
)
```

### 关键问题 2: DataLoader 不支持 "empty" 标记

**问题**: `prepare_custom_split.py` 生成的 JSON 文件包含 `"empty"` 标记（用于 text_only 和 image_only 客户端），但现有的 DataLoader 可能无法正确处理。

**需要检查**:
- `data/dataset_loader.py`
- `data/multimodal_dataset.py`

**建议解决方案**:

修改数据集加载器，添加对 `"empty"` 的处理：

```python
# 在 multimodal_dataset.py 的 __getitem__ 方法中
def __getitem__(self, idx):
    item = self.data_list[idx]

    # 处理影像
    if item.get('image') == 'empty':
        # 返回全零张量
        image = torch.zeros(self.num_modalities, self.img_size, self.img_size)
    else:
        image = self._load_image(item['image'])

    # 处理文本
    if item.get('text_feature') == 'empty':
        # 返回全零向量
        text_features = torch.zeros(self.embed_dim)
    else:
        text_features = self._load_text_features(item['text_feature'])

    # ... 其他代码
```

### 关键问题 3: 客户端配置加载逻辑

**问题**: 当前的客户端设置逻辑可能不支持从 YAML 配置中读取客户端信息。

**涉及文件**:
- `src/federated_trainer.py` (setup_clients 方法)
- `scripts/setup_serial_clients.py`

**需要验证**:
- 客户端数据源路径是否正确映射到 `data/federated_split/client*/dataset.json`
- 客户端模态类型是否正确识别

## 📋 建议的修改步骤

### 第 1 步: 修改配置管理器 ✅（必须）
执行上述 **关键问题 1 - A** 的修改。

### 第 2 步: 修改联邦训练器 ✅（必须）
执行上述 **关键问题 1 - B** 的修改。

### 第 3 步: 修改数据加载器（可选，但强烈建议）
执行上述 **关键问题 2** 的修改，确保能正确处理 `"empty"` 标记。

### 第 4 步: 测试运行
```bash
# 使用虚拟数据快速测试
python main.py --config configs/baseline.yaml

# 检查是否有错误
```

### 第 5 步: 完整训练测试
```bash
# Baseline 实验
python main.py --config configs/baseline.yaml

# Proposed Method 实验
python main.py --config configs/proposed_method.yaml
```

## 🚨 其他潜在问题

### 1. SAM3 预训练权重
**位置**: `data/checkpoints/sam3.pt`
**检查**: 确保该文件存在或在配置中设置 `use_dummy: true`

### 2. 依赖包
确保服务器安装了所有依赖：
```bash
pip install torch torchvision nibabel pyyaml tensorboard
```

### 3. 显存要求
- Batch Size = 1 约需要 8-12GB 显存
- 如果显存不足，可以关闭 `use_amp: false`

### 4. 数据路径
确保所有数据文件已上传到服务器：
```
data/
├── source_images/
│   ├── BraTS2020/
│   ├── BraTS2018/
│   ├── TextBraTS2020.json
│   └── TextBraTS2018.json
└── federated_split/
    ├── client1_text_only/dataset.json
    ├── client2_image_only/dataset.json
    └── client3_multimodal/dataset.json
```

## ✅ 修改完成后的最终检查

运行以下测试脚本：

```bash
# 1. 测试配置加载
python scripts/test_configs.py

# 2. 测试数据加载
python -c "from data.dataset_loader import create_data_loaders; print('DataLoader OK')"

# 3. 测试模型初始化
python -c "from src.integrated_model import SAM3MedicalIntegrated; m = SAM3MedicalIntegrated(use_sam3=False); print('Model OK')"

# 4. 快速训练测试（1轮）
python main.py --config configs/baseline.yaml --rounds 1
```

## 📊 预期训练时间

| 实验 | 轮数 | 预计时间 |
|-----|------|---------|
| Baseline | 50 | 4-8 小时 |
| Proposed Method | 50 | 6-10 小时 |

## 🎯 总结

**当前状态**: ⚠️ 需要修改后才能运行

**必须完成的修改**:
1. ✅ **必须**: 修改 `src/config_manager.py` 添加字段支持
2. ✅ **必须**: 修改 `src/federated_trainer.py` 传递 `client_modalities`
3. ⚠️ **强烈建议**: 修改数据加载器支持 `"empty"` 标记

**预计修改时间**: 30-60 分钟

**修改完成后**: 可以直接提交服务器训练
