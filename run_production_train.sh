#!/bin/bash
# ===============================================================
# FedSAM3-Cream 生产训练启动脚本（防断连版）
# 适用于 AutoDL / 阿里云 / 腾讯云等云服务器
# ===============================================================
#
# 使用方法：
#   1. 给脚本添加执行权限: chmod +x run_production_train.sh
#   2. 后台运行训练:        nohup ./run_production_train.sh &
#   3. 查看实时日志:        tail -f full_training.log
#   4. 检查训练进程:        ps aux | grep python
#
# ===============================================================

# ==================== 配置区域（根据实际情况修改）====================
CONFIG_FILE="configs/exp_production.yaml"
PYTHON_BIN="python"  # 如果使用 conda，可以写 conda run -n your_env python
LOG_FILE="full_training.log"
ERROR_LOG_FILE="training_errors.log"
PID_FILE="training.pid"

# ==================== 颜色输出（可选）====================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# ==================== 预检查 ====================
echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} FedSAM3-Cream 生产训练启动脚本"
echo "========================================================"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} 配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} 配置文件: $CONFIG_FILE"

# 检查 Python 环境
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python 未找到: $PYTHON_BIN"
    echo "请修改脚本中的 PYTHON_BIN 变量"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Python: $($PYTHON_BIN --version)"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    echo -e "${GREEN}[OK]${NC} 检测到 $GPU_COUNT 个 GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}[WARNING]${NC} 未检测到 NVIDIA GPU，将使用 CPU 训练（速度较慢）"
fi

# ==================== 清理旧日志（可选）====================
if [ -f "$LOG_FILE" ]; then
    BACKUP_LOG="${LOG_FILE}.$(date '+%Y%m%d_%H%M%S').bak"
    mv "$LOG_FILE" "$BACKUP_LOG"
    echo -e "${YELLOW}[INFO]${NC} 旧日志已备份: $BACKUP_LOG"
fi

# ==================== 启动训练 ====================
echo "========================================================"
echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} 开始训练..."
echo "配置文件: $CONFIG_FILE"
echo "日志文件: $LOG_FILE"
echo "错误日志: $ERROR_LOG_FILE"
echo "========================================================"

# 启动训练（后台运行，重定向输出）
nohup $PYTHON_BIN main.py \\
    --config "$CONFIG_FILE" \\
    > "$LOG_FILE" 2> "$ERROR_LOG_FILE" &

# 保存 PID
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo -e "${GREEN}[OK]${NC} 训练已启动（PID: $TRAIN_PID）"
echo ""
echo "监控训练进度："
echo "  实时日志: tail -f $LOG_FILE"
echo "  错误日志: tail -f $ERROR_LOG_FILE"
echo "  停止训练: kill -9 $TRAIN_PID  或  kill -9 \$(cat $PID_FILE)"
echo ""
echo "========================================================"

# ==================== 可选：实时显示前 50 行日志 ====================
sleep 2  # 等待日志文件生成
if [ -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}[INFO]${NC} 显示初始日志（前 50 行）："
    echo "--------------------------------------------------------"
    head -n 50 "$LOG_FILE"
    echo "--------------------------------------------------------"
    echo ""
    echo -e "${GREEN}[提示]${NC} 使用 'tail -f $LOG_FILE' 查看实时日志"
fi

exit 0
