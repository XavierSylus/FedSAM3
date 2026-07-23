# 底层特征融合（Surgical Deep Fusion）实现说明

## 📌 概述

本文档详细说明了在 `src/integrated_model.py` 中实现的**底层特征融合（Surgical Deep Fusion）**方案，用于将预提取的文本特征张量与 SAM3 的图像特征进行深度融合。

---

## 🎯 核心问题

### 问题背景
- **真实 SAM3 模型**（use_real_sam3=True）是一个黑盒系统
- **文本特征已预提取**：从 `.npy` 文件加载的 `torch.Tensor`，无法使用 SAM3 官方的 `find_text_batch` 接口（该接口只接受原始字符串）
- **需求**：在 SAM3 的 Image Encoder 和 Mask Decoder 之间插入自定义的图文特征融合

### 解决方案
**"外科手术式"拆解 SAM3 的前向传播流程**，在中间环节插入融合模块。

---

## 🔧 实现的核心修改

### 1️⃣ 增强的 `_apply_fusion` 方法

**位置**：`src/integrated_model.py` 第 654-820 行

#### 核心功能
- ✅ **维度自适应**：动态创建线性投影层，将文本特征维度（如 CLIP 的 768 维）对齐到图像通道数（如 SAM3 的 1024 维）
- ✅ **门控融合**：通过学习的门控网络，自适应地混合图像和文本特征
- ✅ **空间广播**：将全局文本特征广播到空间维度（H × W）
- ✅ **灵活格式支持**：自动处理 3D (B, N, C) 和 4D (B, C, H, W) 输入

#### 关键特性

```python
def _apply_fusion(
    self,
    image_embeddings: torch.Tensor,  # (B, C, H, W) 或 (B, N, C)
    text_features: Optional[torch.Tensor]  # (B, D) 或 (B, L, D)
) -> torch.Tensor:
    """
    【维度自适应】
    - 如果文本维度 D ≠ 图像通道数 C，自动创建投影层：D → C
    - 懒初始化：第一次调用时创建，后续复用
    - 防御性检查：如果维度变化，自动重新创建

    【融合策略】
    - 方案 A（默认）：门控融合（Gated Fusion）
      fused = gate × text + (1 - gate) × image
    - 方案 B（备选）：Cross-Attention（已注释，可启用）
    """
```

#### 维度防御机制

```python
# 示例：文本特征 (B, 768)，图像特征 (B, 1024, 64, 64)
# 自动创建投影层：768 → 1024
self._text_projection = nn.Linear(768, 1024, bias=False)
text_aligned = self._text_projection(text_features)  # (B, 1024)

# 广播到空间维度
text_broadcasted = text_aligned.unsqueeze(-1).unsqueeze(-1).expand(B, 1024, 64, 64)

# 门控融合
gate = self._fusion_gate(concat([image, text]))  # 学习混合权重
fused = gate * text_broadcasted + (1 - gate) * image_embeddings
```

---

### 2️⃣ 重写的 `forward` 方法

**位置**：`src/integrated_model.py` 第 953-1318 行

#### 两种运行模式

##### 🟢 **模式 1：标准模式**（text_features=None）
- 使用 SAM3 原生的黑盒前向传播
- 保持原有功能不变

##### 🔥 **模式 2：融合模式**（text_features ≠ None 且为 torch.Tensor）
拆解 SAM3 的前向传播，分 4 步执行：

```python
# Step 1: 调用 Image Encoder 提取图像特征
image_encoder = self.sam3_model.backbone.vision_backbone.trunk
image_embeddings = image_encoder(images)
# 输出: (B, C, H', W') 或 (B, N, C)

# Step 2: 应用底层特征融合
fused_embeddings = self._apply_fusion(image_embeddings, text_features)
# 融合后特征: 相同形状，但包含了文本信息

# Step 3: 生成空 Prompt Embeddings
sparse_embeddings = torch.zeros(B, 0, embed_dim)  # 空 prompt
dense_embeddings = None  # 或者从 prompt_encoder.get_dense_pe() 获取

# Step 4: 送入 Mask Decoder
mask_decoder = self.sam3_model.mask_decoder
logits = mask_decoder(
    image_embeddings=fused_embeddings,
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    ...
)
```

#### 智能组件查找

代码包含智能的组件查找逻辑，兼容不同的 SAM3 版本：

```python
# 查找 Image Encoder
if hasattr(self.sam3_model, 'backbone'):
    if hasattr(backbone, 'vision_backbone'):
        if hasattr(vision_backbone, 'trunk'):
            image_encoder = vision_backbone.trunk  # ✓ 标准路径
        else:
            image_encoder = vision_backbone  # ✓ 备选路径
    else:
        image_encoder = backbone  # ✓ 简化版本
elif hasattr(self.sam3_model, 'image_encoder'):
    image_encoder = self.sam3_model.image_encoder  # ✓ 直接访问

# 类似的逻辑也适用于 Prompt Encoder 和 Mask Decoder
```

---

## 📊 使用示例

### 基础用法

```python
import torch
from src.integrated_model import SAM3MedicalIntegrated

# 1. 初始化模型
model = SAM3MedicalIntegrated(
    img_size=1024,
    num_classes=1,
    use_sam3=True,  # 使用真实 SAM3
    freeze_encoder=True,
    use_adapter=True,
    sam3_checkpoint="path/to/sam3_checkpoint.pth",
    device="cuda"
)

# 2. 准备数据
images = torch.randn(2, 3, 1024, 1024).cuda()  # 图像批次
text_features = torch.randn(2, 768).cuda()  # 预提取的文本特征（如 CLIP 输出）

# 3. 前向传播（自动启用融合模式）
output = model(images, text_features=text_features)
logits = output['logits']  # (2, 1, 1024, 1024)

print(f"分割结果形状: {logits.shape}")
```

### 与数据加载器集成

```python
from torch.utils.data import DataLoader
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, text_feature_paths):
        self.image_paths = image_paths
        self.text_feature_paths = text_feature_paths

    def __getitem__(self, idx):
        # 加载图像
        image = self.load_image(self.image_paths[idx])

        # 加载预提取的文本特征（从 .npy 文件）
        text_features = np.load(self.text_feature_paths[idx])
        text_features = torch.from_numpy(text_features).float()

        return {
            'image': image,
            'text_features': text_features,
            'label': self.load_label(idx)
        }

# 创建数据加载器
dataset = MultimodalDataset(image_paths, text_feature_paths)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 训练循环
for batch in dataloader:
    images = batch['image'].cuda()
    text_features = batch['text_features'].cuda()
    labels = batch['label'].cuda()

    # 前向传播（自动融合）
    output = model(images, text_features=text_features)

    # 计算损失
    loss = criterion(output['logits'], labels)
    loss.backward()
    optimizer.step()
```

---

## 🛡️ 维度兼容性保证

### 支持的文本特征维度

| 文本编码器 | 输出维度 | 是否兼容 | 处理方式 |
|-----------|---------|---------|---------|
| CLIP ViT-B/32 | (B, 512) | ✅ | 自动投影到 SAM3 维度 |
| CLIP ViT-L/14 | (B, 768) | ✅ | 自动投影到 SAM3 维度 |
| BiomedCLIP | (B, 512) | ✅ | 自动投影到 SAM3 维度 |
| PubMedBERT | (B, 768) | ✅ | 自动投影到 SAM3 维度 |
| 自定义编码器 | (B, D) | ✅ | 动态创建投影层 D → C |
| 序列特征 | (B, L, D) | ✅ | 自动平均池化为 (B, D) |

### 支持的图像特征格式

| 格式 | 形状 | 来源 | 处理方式 |
|------|------|------|---------|
| ViT 序列 | (B, N, C) | Transformer 编码器 | 自动转换为 (B, C, H, W) |
| 空间特征 | (B, C, H, W) | CNN 编码器 | 直接使用 |
| 下采样特征 | (B, C, H', W') | SAM3 输出 | 自动上采样到原始尺寸 |

---

## 🐛 故障排查

### 常见错误 1：找不到 Image Encoder

```
RuntimeError: 无法找到 SAM3 的 Image Encoder！
```

**原因**：SAM3 的模型结构与预期不符

**解决方案**：
1. 检查 SAM3 模型的实际结构：
   ```python
   print(model.sam3_model)
   ```
2. 根据输出调整代码中的访问路径（第 1000-1020 行）

---

### 常见错误 2：维度不匹配

```
RuntimeError: The size of tensor a (768) must match the size of tensor b (1024)
```

**原因**：文本特征维度与图像通道数不一致（这应该不会发生，因为已经有自动投影）

**解决方案**：
1. 检查 `_apply_fusion` 中的投影层是否正常工作
2. 验证 text_features 的形状：
   ```python
   print(f"文本特征形状: {text_features.shape}")
   ```

---

### 常见错误 3：Mask Decoder 调用失败

```
RuntimeError: 无法调用 Mask Decoder！
```

**原因**：SAM3 的 Mask Decoder 接口与预期不符

**解决方案**：
1. 检查 SAM3 官方文档，了解正确的调用方式
2. 修改第 1100-1150 行的 decoder 调用逻辑
3. 如果需要，可以使用简化调用（代码已包含 fallback 逻辑）

---

## 🔍 调试技巧

### 启用详细日志

代码中已包含详细的调试日志，会在融合模式下自动打印：

```python
[Fusion Mode] 检测到文本特征张量 (shape: torch.Size([2, 768]))，启动底层融合...
[Fusion] 找到 Image Encoder: backbone.vision_backbone.trunk
[Fusion] Image Encoder 输出形状: torch.Size([2, 1024, 64, 64])
[Fusion] 创建文本投影层: 768 -> 1024
[Fusion] 创建门控融合网络（输入通道: 2048, 输出通道: 1024）
[Fusion] 融合后特征形状: torch.Size([2, 1024, 64, 64])
[Fusion] 使用 Prompt Encoder 生成空 Prompt
[Fusion] Mask Decoder 输出形状: torch.Size([2, 1, 256, 256])
```

### 验证融合是否生效

```python
# 测试：对比有/无文本特征的输出差异
output_no_text = model(images, text_features=None)
output_with_text = model(images, text_features=text_features)

diff = torch.abs(output_no_text['logits'] - output_with_text['logits']).mean()
print(f"融合前后差异: {diff.item():.4f}")
# 期望：diff > 0，说明文本特征确实影响了结果
```

---

## 🎨 高级配置

### 切换融合策略：从门控融合到 Cross-Attention

如果你希望使用更强大的 Cross-Attention 融合（计算量更大），可以修改 `_apply_fusion` 方法：

**步骤**：
1. 打开 `src/integrated_model.py`
2. 找到第 750 行左右的注释代码块
3. 取消注释，并注释掉门控融合部分

```python
# 方案 B：Cross-Attention（取消注释以启用）
if not hasattr(self, '_cross_attention'):
    num_heads = 8
    self._cross_attention = nn.MultiheadAttention(
        embed_dim=C,
        num_heads=num_heads,
        batch_first=False
    ).to(image_spatial.device)

image_seq = image_spatial.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
text_seq = text_aligned.unsqueeze(0)  # (1, B, C)

attn_output, _ = self._cross_attention(
    query=image_seq,
    key=text_seq,
    value=text_seq
)

fused_spatial = attn_output.permute(1, 2, 0).reshape(B, C, H, W)
fused_spatial = fused_spatial + image_spatial  # 残差连接
```

---

## 📈 性能优化建议

1. **混合精度训练**：使用 `torch.cuda.amp` 加速
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       output = model(images, text_features=text_features)
       loss = criterion(output['logits'], labels)

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **缓存文本投影层**：避免每次都动态创建（代码已实现懒初始化）

3. **批量推理**：尽可能增大 batch size，充分利用 GPU

---

## ✅ 测试验证

### 单元测试

```python
def test_fusion_with_different_dims():
    """测试不同文本维度的融合"""
    model = SAM3MedicalIntegrated(use_sam3=True, device="cuda")
    images = torch.randn(2, 3, 1024, 1024).cuda()

    # 测试不同的文本维度
    for text_dim in [512, 768, 1024, 2048]:
        text_features = torch.randn(2, text_dim).cuda()
        output = model(images, text_features=text_features)
        assert output['logits'].shape == (2, 1, 1024, 1024)
        print(f"✓ 文本维度 {text_dim} 测试通过")

test_fusion_with_different_dims()
```

---

## 📚 参考资料

- **SAM3 官方仓库**: [sam3-main](../../core_projects/sam3-main)
- **CLIP**: [OpenAI CLIP](https://github.com/openai/CLIP)
- **门控融合**: Gated Multimodal Networks (2017)
- **Cross-Attention**: Attention is All You Need (2017)

---

## 🆘 获取帮助

如果遇到问题：
1. 检查本文档的"故障排查"部分
2. 查看代码中的详细注释
3. 运行调试模式，查看日志输出
4. 验证 SAM3 模型的结构是否与预期一致

**祝你成功实现多模态医学影像分割！** 🎉
