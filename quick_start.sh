#!/bin/bash
# ===============================================================
# FedSAM3-Cream 快速启动脚本（修复版）
# ===============================================================

echo "================================================================"
echo "  FedSAM3-Cream 快速启动向导"
echo "================================================================"
echo ""

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

echo "✓ Python版本: $(python --version)"
echo ""

# 步骤1: 数据验证
echo "[步骤 1/3] 验证数据..."
echo "----------------------------------------------------------------"
python scripts/quick_data_check.py --data_root data/federated_split

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 数据验证失败！"
    echo "   请确保数据目录存在: data/federated_split/"
    exit 1
fi

echo ""
echo "✓ 数据验证通过"
echo ""

# 步骤2: 安装依赖
echo "[步骤 2/3] 检查依赖..."
echo "----------------------------------------------------------------"

# 检查关键依赖
python -c "import torch; import yaml; import monai" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  缺少关键依赖，正在安装..."
    pip install pyyaml monai -q
    
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败！"
        exit 1
    fi
fi

echo "✓ 依赖检查通过"
echo ""

# 步骤3: 启动训练
echo "[步骤 3/3] 启动训练..."
echo "----------------------------------------------------------------"
echo "使用配置: configs/exp_production.yaml"
echo "训练模式: Mock模型（快速测试）"
echo "训练轮数: 3 轮"
echo "批次大小: 1"
echo ""
echo "开始训练..."
echo "================================================================"
echo ""

python main.py \
    --config configs/exp_production.yaml \
    --rounds 3 \
    --batch_size 1

EXIT_CODE=$?

echo ""
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成！"
    echo ""
    echo "查看结果:"
    echo "  - 日志: logs/production/"
    echo "  - 检查点: data/checkpoints/"
    echo "  - 训练曲线: data/checkpoints/plots/"
else
    echo "❌ 训练失败 (exit code: $EXIT_CODE)"
    echo ""
    echo "故障排查:"
    echo "  1. 检查错误日志"
    echo "  2. 确认数据完整性"
    echo "  3. 检查显存是否充足"
fi

echo "================================================================"

exit $EXIT_CODE
