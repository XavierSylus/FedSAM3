# MCP工具封装清单

## 📋 概述

本项目共封装了 **2个MCP工具**，用于医学图像分割任务。

---

## 🛠️ 已封装的MCP工具

### 1. `sam3-medical-segmentation` ⭐ **基础版**

**位置**: `tools_factory/incoming_raw/sam3_medical_model/`

**功能**:
- ✅ 基于CreamFL的ResNet架构加载
- ✅ 预训练权重加载
- ✅ 参数冻结配置
- ✅ 医学图像分割推理

**核心类**: `SAM3ModelLoader`

**支持的架构**:
- `resnet11`: ResNet-11 (轻量级)
- `resnet56`: ResNet-56 (推荐)
- `resnet18`: ResNet-18
- `resnet50`: ResNet-50
- `resnet18_client`: 客户端ResNet-18

**MCP工具接口**:
- `run_tool`: 加载模型并运行推理
- `get_model_info`: 获取模型信息（如果server_template支持）

**输入参数**:
```python
{
    "model_arch": str,           # 模型架构名称
    "pretrained_path": str,      # 预训练权重路径（可选）
    "freeze_encoder": bool,      # 是否冻结编码器
    "freeze_layers": list[str],  # 要冻结的层名称列表
    "img_size": int,            # 输入图像尺寸
    "num_classes": int,         # 分割类别数
    "image_path": str           # 输入图像路径（用于推理）
}
```

**输出结果**:
```python
{
    "mask": list[list[float]],   # 分割掩码
    "shape": list[int],          # 输出形状
    "device": str,               # 使用的设备
    "model_info": dict           # 模型配置信息
}
```

**特点**:
- 使用CreamFL的ResNet作为编码器
- 支持多种ResNet变体
- 灵活的冻结策略
- 适合单机推理场景

---

### 2. `sam3-federated-medical-segmentation` ⭐⭐ **完整整合版**

**位置**: `tools_factory/incoming_raw/sam3_federated_model/`

**功能**:
- ✅ SAM3真实模型（sam3-main）
- ✅ SAM-Adapter适配器机制
- ✅ CreamFL多模态对比学习
- ✅ FedFMS联邦架构
- ✅ 参数冻结策略
- ✅ 医学图像分割推理

**核心类**: `SAM3FederatedModelLoader`

**整合的模块**:
1. **SAM3真实模型** (`sam3-main`)
   - 使用 `build_sam3_image()` 加载真实SAM3
   - 支持SAM3预训练权重

2. **SAM-Adapter适配器** (`SAM-Adapter-PyTorch-main`)
   - 轻量级适配器模块（1-5%参数）
   - 注入到Transformer blocks

3. **CreamFL多模态学习** (`CreamFL-main`)
   - MCSoftContrastiveLoss对比学习
   - 概率跨模态嵌入

4. **FedFMS联邦架构** (`FedFMS-main`)
   - 联邦训练策略
   - 加权聚合

**MCP工具接口**:
- `run_tool`: 加载整合模型并运行推理
- `get_model_info`: 获取模型信息和参数统计

**输入参数**:
```python
{
    "use_sam3": bool,            # 是否使用真实SAM3模型
    "freeze_encoder": bool,      # 是否冻结编码器
    "use_adapter": bool,         # 是否使用适配器
    "sam3_checkpoint": str,      # SAM3预训练权重路径（可选）
    "img_size": int,            # 输入图像尺寸
    "num_classes": int,         # 分割类别数
    "adapter_dim": int,         # 适配器维度
    "image_path": str,          # 输入图像路径（用于推理）
    "device": str               # 设备（'cuda' 或 'cpu'）
}
```

**输出结果**:
```python
{
    "mask": list[list[float]],          # 分割掩码
    "shape": list[int],                 # 输出形状
    "device": str,                      # 使用的设备
    "model_info": dict,                 # 模型配置信息
    "trainable_params_info": dict       # 可训练参数信息
}
```

**特点**:
- 整合了四个核心模块
- 支持真实SAM3模型
- 参数高效（只训练1-5%参数）
- 支持多模态联邦学习
- 适合联邦学习场景

---

## 📊 两个工具对比

| 特性 | sam3-medical-segmentation | sam3-federated-medical-segmentation |
|------|---------------------------|-------------------------------------|
| **SAM3真实模型** | ❌ Mock实现 | ✅ 真实SAM3 |
| **SAM-Adapter** | ❌ 无 | ✅ 支持 |
| **CreamFL对比学习** | ❌ 无 | ✅ 支持 |
| **FedFMS联邦架构** | ❌ 无 | ✅ 支持 |
| **ResNet架构** | ✅ 支持多种 | ✅ 通过SAM3 |
| **预训练权重** | ✅ ResNet权重 | ✅ SAM3权重 |
| **参数冻结** | ✅ 支持 | ✅ 支持 |
| **适用场景** | 单机推理 | 联邦学习 |

---

## 🚀 使用方式

### 编译MCP Server

```bash
# 在项目根目录执行
python -m tools_factory.builder
```

**生成结果**:
```
mcp_servers/
├── sam3-medical-segmentation/
│   ├── server.py
│   ├── README.md
│   └── raw_tool/
│       ├── main.py
│       └── tool.yaml
│
└── sam3-federated-medical-segmentation/
    ├── server.py
    ├── README.md
    └── raw_tool/
        ├── main.py
        └── tool.yaml
```

### 调用工具1: sam3-medical-segmentation

```python
# 通过MCP调用
inputs = {
    "model_arch": "resnet56",
    "pretrained_path": "resnet56.pth",
    "freeze_encoder": True,
    "image_path": "medical_image.png"
}

result = mcp_client.call_tool(
    "sam3-medical-segmentation",
    "run_tool",
    inputs
)
```

### 调用工具2: sam3-federated-medical-segmentation

```python
# 通过MCP调用
inputs = {
    "use_sam3": True,
    "freeze_encoder": True,
    "use_adapter": True,
    "sam3_checkpoint": "sam3.pth",
    "image_path": "medical_image.png"
}

result = mcp_client.call_tool(
    "sam3-federated-medical-segmentation",
    "run_tool",
    inputs
)
```

---

## 📁 文件结构

```
tools_factory/
├── incoming_raw/
│   ├── sam3_medical_model/          # 工具1：基础版
│   │   ├── main.py                  # SAM3ModelLoader
│   │   └── tool.yaml
│   │
│   └── sam3_federated_model/        # 工具2：完整整合版
│       ├── main.py                  # SAM3FederatedModelLoader
│       └── tool.yaml
│
├── templates/
│   └── server_template.py           # MCP Server模板（自动调用raw_tool/main.py）
│
└── builder.py                        # MCP Server编译器
```

---

## 🔍 工具功能对比表

### 工具1: sam3-medical-segmentation

| 功能 | 状态 | 说明 |
|------|------|------|
| ResNet架构加载 | ✅ | 支持resnet11/56/18/50 |
| 预训练权重加载 | ✅ | 支持.pth文件 |
| 参数冻结 | ✅ | 完全/部分冻结 |
| 图像分割推理 | ✅ | 医学图像分割 |
| SAM3真实模型 | ❌ | 使用Mock实现 |
| SAM-Adapter | ❌ | 不支持 |
| CreamFL对比学习 | ❌ | 不支持 |
| 联邦学习 | ❌ | 不支持 |

### 工具2: sam3-federated-medical-segmentation

| 功能 | 状态 | 说明 |
|------|------|------|
| SAM3真实模型 | ✅ | 使用sam3-main |
| SAM-Adapter | ✅ | 轻量级适配器 |
| CreamFL对比学习 | ✅ | MCSoftContrastiveLoss |
| FedFMS联邦架构 | ✅ | 联邦训练策略 |
| 参数冻结 | ✅ | ImageEncoder冻结 |
| 图像分割推理 | ✅ | 医学图像分割 |
| 参数统计 | ✅ | 可训练参数信息 |

---

## 🎯 选择建议

### 使用工具1 (`sam3-medical-segmentation`) 如果：
- ✅ 只需要基础的医学图像分割功能
- ✅ 使用CreamFL的ResNet作为编码器
- ✅ 单机推理场景
- ✅ 不需要SAM3真实模型

### 使用工具2 (`sam3-federated-medical-segmentation`) 如果：
- ✅ 需要完整的SAM3功能
- ✅ 需要多模态联邦学习
- ✅ 需要参数高效的训练（Adapter）
- ✅ 需要整合所有四个模块的功能

---

## 📝 编译和运行

### 步骤1: 安装依赖

```bash
# 如果缺少yaml模块
pip install pyyaml
```

### 步骤2: 编译MCP Server

```bash
python -m tools_factory.builder
```

### 步骤3: 运行MCP Server

```bash
# 工具1
cd mcp_servers/sam3-medical-segmentation
python server.py

# 工具2
cd mcp_servers/sam3-federated-medical-segmentation
python server.py
```

---

## ✅ 总结

**已封装的MCP工具**:
1. ✅ `sam3-medical-segmentation` - 基础版（CreamFL ResNet）
2. ✅ `sam3-federated-medical-segmentation` - 完整整合版（SAM3 + Adapter + CreamFL + FedFMS）

**两个工具都已**:
- ✅ 创建了`main.py`入口文件
- ✅ 创建了`tool.yaml`配置文件
- ✅ 支持MCP Server自动编译
- ✅ 支持通过MCP客户端调用

**推荐使用**: `sam3-federated-medical-segmentation`（工具2），因为它整合了所有四个模块的功能，功能更完整。

