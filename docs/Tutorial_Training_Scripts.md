# WandB 训练脚本快速使用指南

## 🚀 快速开始（3步）

### 步骤 1: 安装依赖

```bash
pip install wandb torch torchvision
```

### 步骤 2: 登录 WandB（首次使用）

```bash
wandb login
```

会提示输入 API key，在 https://wandb.ai/settings 获取。

### 步骤 3: 运行训练（使用虚拟数据测试）

```bash
python scripts/train_with_wandb.py --use_dummy --epochs 10 --learning_rate 0.001
```

## 📝 使用方法

### 方法 1: 命令行使用（最简单）

#### 基本用法

```bash
python scripts/train_with_wandb.py \
    --use_dummy \
    --epochs 20 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --wandb_project my-project
```

#### 完整参数示例

```bash
python scripts/train_with_wandb.py \
    --epochs 50 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --optimizer adam \
    --wandb_project my-research \
    --experiment_name resnet50-baseline \
    --save_path models/best_model.pth \
    --use_dummy
```

#### 不使用 WandB

```bash
python scripts/train_with_wandb.py \
    --use_dummy \
    --epochs 10 \
    --no_wandb
```

### 方法 2: 代码集成（推荐用于实际项目）

#### 示例 1: 基本使用

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.train_with_wandb import Trainer

# 1. 准备数据和模型
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
model = YourModel()

# 2. 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    use_wandb=True,
    wandb_project="my-project",
    experiment_name="experiment-1"
)

# 3. 开始训练
trainer.train(
    epochs=50,
    learning_rate=0.001,
    optimizer_type="adam",
    save_path="best_model.pth"
)
```

#### 示例 2: 使用自定义损失函数

```python
from scripts.train_with_wandb import Trainer
import torch.nn as nn

# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    use_wandb=True,
    wandb_project="my-project"
)

# 自定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练（使用自定义损失函数）
trainer.train(
    epochs=50,
    learning_rate=0.001,
    optimizer_type="adam",
    criterion=criterion,
    save_path="best_model.pth"
)
```

#### 示例 3: 数据格式支持

脚本支持多种数据格式：

**格式 1: 元组格式**
```python
# 数据集返回 (image, label)
dataset = YourDataset(...)  # __getitem__ 返回 (image, label)
```

**格式 2: 字典格式**
```python
# 数据集返回字典
class YourDataset:
    def __getitem__(self, idx):
        return {
            'image': image_tensor,
            'label': label_tensor
        }
        # 或者
        return {
            'inp': image_tensor,
            'gt': label_tensor
        }
```

## 📊 WandB 记录的内容

训练脚本会自动记录以下内容到 WandB：

### 超参数
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `optimizer`: 优化器类型
- `batch_size`: 批次大小
- `device`: 训练设备

### 训练指标（每个 epoch）
- `train/loss`: 训练损失
- `train/accuracy`: 训练准确率

### 验证指标（每个 epoch，如果有验证集）
- `val/loss`: 验证损失
- `val/accuracy`: 验证准确率

### 批次指标（每 10 个批次）
- `train/batch_loss`: 批次损失

### 最终结果
- `best_val_accuracy`: 最佳验证准确率
- `final_train_loss`: 最终训练损失
- `final_train_accuracy`: 最终训练准确率

## 🔧 常用命令

### 查看帮助

```bash
python scripts/train_with_wandb.py --help
```

### 所有可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 10 |
| `--learning_rate` | 学习率 | 0.001 |
| `--batch_size` | 批次大小 | 32 |
| `--optimizer` | 优化器 (adam/sgd/adamw) | adam |
| `--wandb_project` | WandB 项目名称 | pytorch-training |
| `--wandb_entity` | WandB entity（可选） | None |
| `--experiment_name` | 实验名称（可选） | None |
| `--no_wandb` | 禁用 WandB | False |
| `--use_dummy` | 使用虚拟数据 | False |
| `--device` | 训练设备 (cuda/cpu) | cuda |
| `--save_path` | 模型保存路径 | None |

## 💡 实际使用示例

### 示例 1: 训练分类模型

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scripts.train_with_wandb import Trainer

# 1. 准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 2. 创建模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

# 3. 创建训练器并训练
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    use_wandb=True,
    wandb_project="mnist-classification",
    experiment_name="simple-cnn"
)

trainer.train(
    epochs=10,
    learning_rate=0.001,
    optimizer_type="adam",
    save_path="mnist_model.pth"
)
```

### 示例 2: 训练分割模型

```python
from scripts.train_with_wandb import Trainer
import torch.nn as nn

# 创建训练器
trainer = Trainer(
    model=segmentation_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    use_wandb=True,
    wandb_project="segmentation"
)

# 使用分割损失函数
criterion = nn.BCEWithLogitsLoss()

trainer.train(
    epochs=50,
    learning_rate=0.0001,
    optimizer_type="adam",
    criterion=criterion,
    save_path="segmentation_model.pth"
)
```

## 📈 查看结果

### 在 WandB 网站查看

1. 打开浏览器访问: https://wandb.ai
2. 登录您的账户
3. 选择对应的项目（`wandb_project`）
4. 查看实验运行结果和图表

### 本地查看（如果使用 TensorBoard）

```bash
tensorboard --logdir logs/
```

## ⚠️ 注意事项

1. **首次使用 WandB**: 需要先运行 `wandb login` 登录
2. **数据格式**: 确保数据集返回 `(image, label)` 或字典格式
3. **分类任务**: 准确率会自动计算（模型输出 `(B, num_classes)`，标签 `(B,)`）
4. **分割任务**: 只记录损失，准确率需要自定义计算
5. **GPU 内存**: 如果内存不足，减小 `batch_size`

## 🐛 常见问题

### Q: WandB 登录失败？
```bash
# 重新登录
wandb login

# 或设置环境变量
export WANDB_API_KEY=your_api_key
```

### Q: 如何禁用 WandB？
```bash
# 方法 1: 命令行参数
python scripts/train_with_wandb.py --no_wandb --use_dummy

# 方法 2: 代码中设置
trainer = Trainer(..., use_wandb=False)
```

### Q: 如何保存模型？
```python
# 在 trainer.train() 中指定 save_path
trainer.train(..., save_path="best_model.pth")
```

### Q: 如何使用自己的数据集？
```python
# 创建自己的数据集类
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return image, label  # 或 {'image': ..., 'label': ...}

# 然后使用
train_dataset = MyDataset(...)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## 📚 更多信息

- 详细文档: `docs/WandB训练脚本使用指南.md`
- 使用示例: `examples/train_example.py`

