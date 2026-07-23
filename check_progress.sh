#!/bin/bash
# =============================================================================
# 快速检查实验进度
# 在服务器上运行，实时查看实验状态
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear

echo -e "${BLUE}========================================"
echo "FedSAM3-Cream 实验进度检查"
echo "========================================"
echo -e "检查时间: $(date)${NC}"
echo ""

# =============================================================================
# 1. 检查进程状态
# =============================================================================

echo -e "${YELLOW}1. 进程状态${NC}"
PYTHON_PROCS=$(ps aux | grep "[p]ython main.py" | wc -l)

if [ $PYTHON_PROCS -gt 0 ]; then
    echo -e "${GREEN}✓ 检测到 ${PYTHON_PROCS} 个运行中的 Python 进程${NC}"
    ps aux | grep "[p]ython main.py" | awk '{print "  PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%"}'
else
    echo -e "${RED}✗ 没有检测到运行中的实验进程${NC}"
fi

echo ""

# =============================================================================
# 2. GPU 使用情况
# =============================================================================

echo -e "${YELLOW}2. GPU 使用情况${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU %s: %s | 利用率: %s%% | 显存: %s/%s MB | 温度: %s°C\n", $1, $2, $3, $4, $5, $6}'
else
    echo -e "${RED}  nvidia-smi 不可用${NC}"
fi

echo ""

# =============================================================================
# 3. 检查每组实验的进度
# =============================================================================

check_group_progress() {
    GROUP=$1
    GROUP_LOWER=$(echo $GROUP | tr '[:upper:]' '[:lower:]')
    LOG_FILE="logs/group_${GROUP_LOWER}/train.log"

    echo -e "${YELLOW}Group ${GROUP}:${NC}"

    if [ ! -f "$LOG_FILE" ]; then
        echo -e "  ${RED}✗ 日志文件不存在${NC}"
        return
    fi

    # 检查最后修改时间
    LAST_MODIFIED=$(stat -c %Y "$LOG_FILE" 2>/dev/null || stat -f %m "$LOG_FILE" 2>/dev/null)
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - LAST_MODIFIED))

    if [ $TIME_DIFF -lt 300 ]; then
        echo -e "  ${GREEN}✓ 正在运行 (日志更新于 ${TIME_DIFF}秒前)${NC}"
    else
        echo -e "  ${YELLOW}⚠ 可能已停止 (日志已 $((TIME_DIFF / 60)) 分钟未更新)${NC}"
    fi

    # 提取当前轮数
    CURRENT_ROUND=$(grep -oP "Round \K[0-9]+" "$LOG_FILE" | tail -1)
    TOTAL_ROUNDS=$(grep -oP "总轮数: \K[0-9]+" "$LOG_FILE" | head -1 || echo "60")

    if [ -n "$CURRENT_ROUND" ]; then
        PROGRESS=$((CURRENT_ROUND * 100 / TOTAL_ROUNDS))
        echo -e "  进度: Round ${CURRENT_ROUND}/${TOTAL_ROUNDS} (${PROGRESS}%)"
    fi

    # 提取最新 Dice
    LATEST_DICE=$(grep "验证集 Dice:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    if [ -n "$LATEST_DICE" ]; then
        echo -e "  最新 Dice: ${LATEST_DICE}"
    fi

    # 提取最新损失
    LATEST_LOSS=$(grep "loss=" "$LOG_FILE" | tail -1 | grep -oP "loss=\K[0-9.]+" | head -1)
    if [ -n "$LATEST_LOSS" ]; then
        echo -e "  最新 Loss: ${LATEST_LOSS}"
    fi

    # 检查错误
    ERROR_COUNT=$(grep -i "error\|exception\|traceback" "$LOG_FILE" | wc -l)
    if [ $ERROR_COUNT -gt 0 ]; then
        echo -e "  ${RED}⚠ 检测到 ${ERROR_COUNT} 个错误${NC}"
    fi

    # Group B/C: 检查梯度监控
    if [ "$GROUP" != "A" ]; then
        GRADIENT_COUNT=$(grep "GRADIENT" "$LOG_FILE" | wc -l)
        if [ $GRADIENT_COUNT -gt 0 ]; then
            echo -e "  ${GREEN}✓ 梯度监控已激活 (${GRADIENT_COUNT} 条记录)${NC}"

            # 显示最新梯度夹角
            LATEST_ANGLE=$(grep "angle=" "$LOG_FILE" | grep "adapter" | tail -1 | grep -oP "angle=\K[0-9.]+")
            if [ -n "$LATEST_ANGLE" ]; then
                echo -e "  最新梯度夹角: ${LATEST_ANGLE}°"
            fi
        else
            echo -e "  ${YELLOW}⚠ 未检测到梯度监控输出${NC}"
        fi
    fi

    echo ""
}

echo -e "${YELLOW}3. 实验进度${NC}"
echo ""

check_group_progress "A"
check_group_progress "B"
check_group_progress "C"

# =============================================================================
# 4. 磁盘使用情况
# =============================================================================

echo -e "${YELLOW}4. 磁盘使用情况${NC}"
if [ -d "logs" ]; then
    LOG_SIZE=$(du -sh logs 2>/dev/null | cut -f1)
    echo "  日志目录大小: ${LOG_SIZE}"
fi

if [ -d "data" ]; then
    DATA_SIZE=$(du -sh data 2>/dev/null | cut -f1)
    echo "  数据目录大小: ${DATA_SIZE}"
fi

echo ""

# =============================================================================
# 5. 快速操作提示
# =============================================================================

echo -e "${BLUE}========================================"
echo "快速操作"
echo "========================================"
echo -e "${NC}"

echo "查看实时日志:"
echo "  tail -f logs/group_a/train.log"
echo ""

echo "查看错误:"
echo "  grep -i error logs/group_a/train.log"
echo ""

echo "查看梯度监控 (Group B/C):"
echo "  grep GRADIENT logs/group_b/train.log | tail -20"
echo ""

echo "查看 TensorBoard:"
echo "  tensorboard --logdir=logs --port=6006"
echo ""

echo "终止实验:"
echo "  kill -9 \$(pgrep -f 'python main.py')"
echo ""

# =============================================================================
# 6. 预估完成时间
# =============================================================================

for GROUP in A B C; do
    GROUP_LOWER=$(echo $GROUP | tr '[:upper:]' '[:lower:]')
    LOG_FILE="logs/group_${GROUP_LOWER}/train.log"

    if [ -f "$LOG_FILE" ]; then
        # 获取第一轮和最后一轮的时间戳
        FIRST_LINE=$(grep "Round 1/" "$LOG_FILE" | head -1)
        LAST_LINE=$(grep "Round [0-9]*/" "$LOG_FILE" | tail -1)

        if [ -n "$FIRST_LINE" ] && [ -n "$LAST_LINE" ]; then
            # 提取轮数
            CURRENT_ROUND=$(echo "$LAST_LINE" | grep -oP "Round \K[0-9]+")
            TOTAL_ROUNDS=$(grep -oP "总轮数: \K[0-9]+" "$LOG_FILE" | head -1 || echo "60")

            if [ $CURRENT_ROUND -gt 1 ] && [ $CURRENT_ROUND -lt $TOTAL_ROUNDS ]; then
                # 计算时间（简化版，假设均匀速度）
                # 这里需要更复杂的时间戳提取，暂时简化
                echo -e "${YELLOW}Group ${GROUP} 预估剩余时间: 计算中...${NC}"
            fi
        fi
    fi
done

echo ""
echo -e "${BLUE}========================================"
echo "提示: 使用 'watch -n 30 bash check_progress.sh' 自动刷新"
echo -e "=======================================${NC}"
