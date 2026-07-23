# Git + DVC + W&B 项目管理完整指南

本指南将详细介绍如何在 FedSAM3-Cream 项目中使用 Git、DVC 和 Weights & Biases (W&B) 的组合来管理代码、数据和实验。

## 📋 目录

1. [工具概述](#工具概述)
2. [初始设置](#初始设置)
3. [工作流程](#工作流程)
4. [最佳实践](#最佳实践)
5. [常见场景](#常见场景)

---

## 🔧 工具概述

### Git - 代码版本控制
- **用途**: 管理源代码、配置文件和脚本
- **存储**: 代码变更历史、分支管理、协作开发

### DVC (Data Version Control) - 数据版本控制
- **用途**: 管理大型数据集、模型检查点、实验结果
- **存储**: 数据文件元数据（.dvc 文件），实际数据存储在远程存储（S3、GCS、本地等）
- **优势**: 避免将大文件提交到 Git，但保持数据版本追踪

### W&B (Weights & Biases) - 实验跟踪与可视化
- **用途**: 跟踪训练指标、超参数、模型版本、系统资源
- **存储**: 云端实验记录、可视化图表、模型注册表
- **优势**: 实时监控、实验对比、团队协作

---

## 🚀 初始设置

### 1. Git 初始化（如果尚未初始化）

```bash
# 初始化 Git 仓库
git init

# 创建 .gitignore 文件（如果不存在）
```

### 2. DVC 初始化

```bash
# 安装 DVC
pip install dvc dvc-s3  # 如果使用 S3 存储
# 或
pip install dvc[all]    # 安装所有依赖

# 初始化 DVC
dvc init

# 配置远程存储（示例：使用本地目录作为远程存储）
dvc remote add -d myremote /path/to/remote/storage

# 或使用 S3（推荐用于生产环境）
dvc remote add -d myremote s3://your-bucket/dvc-storage
dvc remote modify myremote credentialpath ~/.aws/credentials
```

### 3. W&B 初始化

```bash
# 安装 W&B
pip install wandb

# 登录 W&B（首次使用）
wandb login

# 或使用 API key
export WANDB_API_KEY=your_api_key_here
```

### 4. 配置 .gitignore

确保 `.gitignore` 包含以下内容：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# DVC
.dvc/cache/
.dvc/tmp/
*.dvc.lock

# W&B
wandb/

# 数据目录（由 DVC 管理）
data/
DataBase/
*.pkl
*.nii
*.nii.gz
*.pth
*.ckpt
*.h5
*.hdf5

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# 日志
*.log
logs/
```

### 5. 配置 .dvcignore

`.dvcignore` 文件已存在，可以添加更多忽略模式：

```dvcignore
# 添加 DVC 应该忽略的文件模式
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/
```

---

## 📝 工作流程

### 场景 1: 添加新数据集

```bash
# 1. 使用 DVC 跟踪数据目录
dvc add data/train
dvc add data/val
dvc add data/test

# 2. 提交 DVC 元数据到 Git
git add data/train.dvc data/val.dvc data/test.dvc .gitignore
git commit -m "Add training, validation, and test datasets"

# 3. 推送数据到远程存储
dvc push

# 4. 推送代码到 Git 远程仓库
git push
```

### 场景 2: 开始新的训练实验

#### 步骤 1: 创建新的 Git 分支

```bash
git checkout -b experiment/sam3-cream-v2
```

#### 步骤 2: 修改代码并配置 W&B

在训练脚本中添加 W&B 集成（例如 `scripts/main_federated.py`）：

```python
import wandb

# 初始化 W&B 项目
wandb.init(
    project="fedsam3-cream",
    name="experiment-sam3-cream-v2",
    config={
        "batch_size": 6,
        "learning_rate": 2e-4,
        "img_size": 1024,
        "rounds": 50,
        "lambda_cream": 0.1,
        "alpha": 0.9,
        "temperature": 1.0,
        "num_clients": 3,
    },
    tags=["federated-learning", "sam3", "cream"],
)

# 在训练循环中记录指标
for round_num in range(ROUNDS):
    # ... 训练代码 ...
    
    # 记录指标
    wandb.log({
        "round": round_num,
        "loss": loss.item(),
        "dice_score": dice_score,
        "client_avg_loss": avg_client_loss,
        "global_rep_norm": global_rep_norm,
    })
    
    # 记录模型检查点
    if round_num % 10 == 0:
        checkpoint_path = f"checkpoints/round_{round_num}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        # 使用 DVC 跟踪检查点
        # dvc add checkpoint_path
        
        # 可选：将检查点上传到 W&B
        wandb.save(checkpoint_path)

# 结束 W&B 运行
wandb.finish()
```

#### 步骤 3: 运行实验

```bash
# 运行训练
python scripts/main_federated.py

# W&B 会自动记录所有指标和日志
```

#### 步骤 4: 提交代码和 DVC 文件

```bash
# 添加代码变更
git add scripts/main_federated.py

# 如果有新的检查点，使用 DVC 跟踪
dvc add data/checkpoints/final_model.pth
git add data/checkpoints/final_model.pth.dvc

# 提交
git commit -m "Add W&B integration and experiment config"

# 推送
git push origin experiment/sam3-cream-v2
dvc push
```

### 场景 3: 版本化模型检查点

```bash
# 1. 训练完成后，保存模型
# (代码中已保存)

# 2. 使用 DVC 跟踪检查点
dvc add data/checkpoints/final_model.pth

# 3. 提交 DVC 文件到 Git
git add data/checkpoints/final_model.pth.dvc .gitignore
git commit -m "Add final model checkpoint (round 50)"

# 4. 推送
dvc push
git push
```

### 场景 4: 恢复特定版本的数据

```bash
# 1. 从 Git 获取特定版本的 .dvc 文件
git checkout <commit-hash> -- data/train.dvc

# 2. 使用 DVC 下载对应的数据
dvc pull data/train.dvc

# 3. 恢复代码到对应版本
git checkout <commit-hash>
```

### 场景 5: 团队协作 - 克隆项目并获取数据

```bash
# 1. 克隆 Git 仓库
git clone <repository-url>
cd FedSAM3-Cream

# 2. 安装依赖
pip install -r config/requirements.txt
pip install dvc wandb

# 3. 配置 DVC 远程存储（首次）
dvc remote add -d myremote <remote-storage-url>

# 4. 拉取数据
dvc pull

# 5. 配置 W&B（如果需要）
wandb login
```

---

## 🎯 最佳实践

### 1. Git 最佳实践

- ✅ **提交频率**: 每个功能/实验完成后立即提交
- ✅ **提交信息**: 使用清晰的提交信息，描述做了什么和为什么
- ✅ **分支策略**: 
  - `main/master`: 稳定版本
  - `develop`: 开发分支
  - `feature/*`: 新功能
  - `experiment/*`: 实验分支
- ✅ **不要提交**: 大文件、数据文件、模型检查点（使用 DVC）

### 2. DVC 最佳实践

- ✅ **跟踪大文件**: 所有 > 10MB 的文件都应该用 DVC 管理
- ✅ **数据目录结构**: 保持清晰的数据目录结构
  ```
  data/
  ├── train/
  ├── val/
  ├── test/
  └── checkpoints/
  ```
- ✅ **.dvc 文件**: 始终将 `.dvc` 文件提交到 Git
- ✅ **远程存储**: 使用可靠的远程存储（S3、GCS、Azure Blob）
- ✅ **数据版本**: 每次数据变更都要提交新的 `.dvc` 文件

### 3. W&B 最佳实践

- ✅ **项目命名**: 使用清晰的项目名称，如 `fedsam3-cream`
- ✅ **运行命名**: 使用描述性的运行名称，包含关键超参数
  ```
  sam3-cream-lr2e4-bs6-rounds50
  ```
- ✅ **配置记录**: 记录所有超参数到 `wandb.config`
- ✅ **指标记录**: 记录所有重要指标（损失、准确率、IoU、Dice 等）
- ✅ **标签使用**: 使用标签组织实验（`baseline`, `ablation`, `production`）
- ✅ **模型注册**: 使用 W&B Model Registry 管理生产模型

### 4. 组合使用最佳实践

- ✅ **实验可复现性**: 
  - Git commit hash → 代码版本
  - DVC .dvc 文件 → 数据版本
  - W&B run ID → 实验配置和结果
  
- ✅ **实验记录模板**:
  ```python
  wandb.init(
      project="fedsam3-cream",
      name=f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
      config={
          # 超参数
          "batch_size": 6,
          "learning_rate": 2e-4,
          # ...
          # 版本信息
          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
          "dvc_data_version": "data/train.dvc",
      },
      tags=["federated", "sam3"],
  )
  ```

- ✅ **检查点管理**:
  - 定期保存检查点（每 N 轮）
  - 使用 DVC 跟踪重要检查点
  - 在 W&B 中记录检查点路径和指标

---

## 🔄 常见场景

### 场景 1: 开始新的一天工作

```bash
# 1. 拉取最新代码
git pull

# 2. 拉取最新数据（如果有更新）
dvc pull

# 3. 检查 W&B 最新实验
# 在浏览器中打开 https://wandb.ai/your-username/fedsam3-cream
```

### 场景 2: 运行实验并记录

```bash
# 1. 创建实验分支
git checkout -b experiment/new-aggregation-method

# 2. 修改代码
# ... 编辑代码 ...

# 3. 运行实验（自动记录到 W&B）
python scripts/main_federated.py

# 4. 查看 W&B 实时结果
# 浏览器会自动打开或访问 wandb.ai

# 5. 如果结果好，保存检查点
dvc add data/checkpoints/best_model.pth
git add data/checkpoints/best_model.pth.dvc

# 6. 提交代码
git add scripts/main_federated.py
git commit -m "Add new aggregation method, results logged to W&B run XYZ"
git push
dvc push
```

### 场景 3: 对比不同实验

```bash
# 1. 在 W&B 网页界面中对比实验
# - 选择多个运行
# - 对比指标曲线
# - 对比超参数

# 2. 或使用 W&B API
python -c "
import wandb
api = wandb.Api()
runs = api.runs('your-username/fedsam3-cream')
for run in runs:
    print(f'{run.name}: {run.summary.get(\"best_dice\", \"N/A\")}')
"
```

### 场景 4: 回退到之前的实验

```bash
# 1. 在 W&B 中找到之前的运行 ID
# 2. 查看该运行的配置和 Git commit

# 3. 恢复代码
git checkout <commit-hash>

# 4. 恢复数据（如果需要）
git checkout <commit-hash> -- data/train.dvc
dvc pull data/train.dvc

# 5. 恢复模型检查点（如果已保存）
git checkout <commit-hash> -- data/checkpoints/final_model.pth.dvc
dvc pull data/checkpoints/final_model.pth.dvc
```

### 场景 5: 分享实验结果

```bash
# 1. 确保代码已推送
git push

# 2. 确保数据已推送
dvc push

# 3. 在 W&B 中分享运行链接
# 例如: https://wandb.ai/your-username/fedsam3-cream/runs/abc123

# 4. 团队成员可以：
#    - 查看 W&B 中的指标和可视化
#    - 克隆代码: git clone <repo>
#    - 获取数据: dvc pull
#    - 复现实验: python scripts/main_federated.py
```

---

## 📊 项目结构建议

```
FedSAM3-Cream/
├── .git/                    # Git 仓库
├── .dvc/                    # DVC 配置
├── .gitignore              # Git 忽略文件
├── .dvcignore              # DVC 忽略文件
│
├── data/                    # 数据目录（DVC 管理）
│   ├── train/              # 训练数据
│   ├── val/                # 验证数据
│   ├── test/               # 测试数据
│   └── checkpoints/        # 模型检查点
│       ├── final_model.pth
│       └── final_model.pth.dvc  # DVC 元数据
│
├── scripts/                 # 训练脚本（Git 管理）
│   └── main_federated.py   # 主训练脚本（集成 W&B）
│
├── config/                  # 配置文件（Git 管理）
│   └── experiment_config.yaml
│
├── wandb/                   # W&B 本地缓存（.gitignore）
│   └── ...
│
└── docs/                    # 文档（Git 管理）
    └── Git_DVC_WandB使用指南.md
```

---

## 🔗 工具集成示例

### 完整的训练脚本模板

```python
import wandb
import subprocess
import torch
from pathlib import Path

# 获取 Git commit hash
def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return "unknown"

# 初始化 W&B
wandb.init(
    project="fedsam3-cream",
    name=f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    config={
        # 超参数
        "batch_size": 6,
        "learning_rate": 2e-4,
        "img_size": 1024,
        "rounds": 50,
        "lambda_cream": 0.1,
        
        # 版本信息
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
    },
    tags=["federated-learning", "sam3", "cream"],
)

# 训练循环
for round_num in range(ROUNDS):
    # ... 训练代码 ...
    
    # 记录指标
    metrics = {
        "round": round_num,
        "loss": loss.item(),
        "dice_score": dice_score,
        "client_avg_loss": avg_client_loss,
    }
    wandb.log(metrics)
    
    # 定期保存检查点
    if round_num % 10 == 0:
        checkpoint_path = f"data/checkpoints/round_{round_num}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        # 记录检查点信息
        wandb.log({
            "checkpoint_round": round_num,
            "checkpoint_path": checkpoint_path,
        })

# 保存最终模型
final_model_path = "data/checkpoints/final_model.pth"
torch.save(model.state_dict(), final_model_path)

# 记录最终指标
wandb.log({
    "final_loss": final_loss,
    "final_dice": final_dice,
    "final_model_path": final_model_path,
})

# 标记为最佳模型（如果适用）
wandb.run.summary["best_dice"] = best_dice
wandb.run.summary["best_round"] = best_round

wandb.finish()

# 使用 DVC 跟踪最终模型
# 在命令行中运行: dvc add data/checkpoints/final_model.pth
```

---

## 🛠️ 故障排除

### DVC 问题

**问题**: `dvc pull` 失败
```bash
# 检查远程存储配置
dvc remote list

# 重新配置远程存储
dvc remote add -d myremote <storage-url>
```

**问题**: 数据文件损坏
```bash
# 清理 DVC 缓存
dvc cache dir
dvc cache clean
dvc pull --force
```

### W&B 问题

**问题**: W&B 无法连接
```bash
# 检查登录状态
wandb login --relogin

# 检查网络连接
wandb status
```

**问题**: 实验记录不完整
```bash
# 确保在训练循环中正确调用 wandb.log()
# 确保在脚本结束时调用 wandb.finish()
```

### Git 问题

**问题**: 提交了不应该提交的大文件
```bash
# 从 Git 历史中移除文件（但保留本地文件）
git rm --cached large_file.pth

# 使用 DVC 跟踪
dvc add large_file.pth
git add large_file.pth.dvc

# 提交
git commit -m "Move large file to DVC"
```

---

## 📚 参考资源

- **Git**: https://git-scm.com/doc
- **DVC**: https://dvc.org/doc
- **W&B**: https://docs.wandb.ai/
- **DVC 最佳实践**: https://dvc.org/doc/user-guide/best-practices
- **W&B 最佳实践**: https://docs.wandb.ai/guides

---

## ✅ 检查清单

在开始新实验前，确保：

- [ ] Git 仓库已初始化
- [ ] DVC 已初始化并配置远程存储
- [ ] W&B 已登录
- [ ] `.gitignore` 和 `.dvcignore` 已正确配置
- [ ] 训练脚本已集成 W&B
- [ ] 数据已使用 DVC 跟踪
- [ ] 远程存储可访问

---

**最后更新**: 2024年

