# 快速评估指南

## 最简单的使用方法

### 方法 1: 直接运行（自动查找检查点）

```bash
python scripts/evaluate_model.py
```

脚本会自动在以下位置查找检查点：
- `data/checkpoints/final_model.pth`
- `data/checkpoints/best_model.pth`
- `data/checkpoints/latest_checkpoint.pth`
- `checkpoints/final_model.pth`
- `checkpoints/best_model.pth`
- `checkpoints/latest_checkpoint.pth`

### 方法 2: 指定检查点目录

```bash
python scripts/evaluate_model.py --checkpoint_dir data/checkpoints
```

### 方法 3: 指定检查点文件

```bash
python scripts/evaluate_model.py --checkpoint data/checkpoints/final_model.pth
```

## 完整示例

### 基本评估（不保存结果）

```bash
python scripts/evaluate_model.py
```

### 评估并保存结果

```bash
python scripts/evaluate_model.py \
    --save_dir results/evaluation \
    --save_predictions
```

### 评估测试集

```bash
python scripts/evaluate_model.py \
    --split test \
    --save_dir results/test_evaluation \
    --save_predictions
```

## 常见问题

### Q: 找不到检查点文件？

**解决方案**:
1. 检查文件是否存在：`ls data/checkpoints/`
2. 使用完整路径：`--checkpoint data/checkpoints/final_model.pth`
3. 如果检查点在别的位置，使用 `--checkpoint_dir` 指定目录

### Q: 想要评估特定客户端？

```bash
python scripts/evaluate_model.py --client_id client_1
```

### Q: 内存不足？

```bash
python scripts/evaluate_model.py --batch_size 1
```

## 输出说明

评估完成后，结果会保存在 `results/evaluation/` 目录下：

```
results/evaluation/
├── pred_masks/              # 预测掩码（如果使用 --save_predictions）
│   └── *.nii.gz
└── metrics/                 # 评估指标
    ├── evaluation_results.csv      # 每个样本的详细指标
    └── overall_metrics.csv         # 总体平均指标
```

## 查看帮助

```bash
python scripts/evaluate_model.py --help
```

