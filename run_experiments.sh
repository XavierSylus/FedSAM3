#!/bin/bash
# =============================================================================
# FedSAM3-Cream 自动运行三组实验
# 用于在服务器上后台运行所有实验
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 配置项
# =============================================================================

# 日志目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiments_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo -e "${BLUE}=========================================="
echo "FedSAM3-Cream 实验自动运行脚本"
echo "=========================================="
echo -e "日志目录: ${LOG_DIR}${NC}"
echo ""

# =============================================================================
# 激活环境
# =============================================================================

echo -e "${YELLOW}激活 Python 环境...${NC}"

# 尝试激活 conda 环境
if command -v conda &> /dev/null; then
    echo "检测到 conda"
    # 初始化 conda（如果需要）
    eval "$(conda shell.bash hook)" 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

    # 激活环境
    if conda env list | grep -q "fedsam3"; then
        conda activate fedsam3
        echo -e "${GREEN}✓ 已激活 conda 环境: fedsam3${NC}"
    else
        echo -e "${RED}✗ conda 环境 'fedsam3' 不存在${NC}"
        exit 1
    fi
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ 已激活 venv 环境${NC}"
else
    echo -e "${YELLOW}⚠ 未检测到虚拟环境，使用系统 Python${NC}"
fi

# 验证 Python 和 GPU
echo ""
echo -e "${YELLOW}环境检查...${NC}"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Python 环境检查失败${NC}"
    exit 1
fi

echo ""

# =============================================================================
# 函数：运行单个实验
# =============================================================================

run_experiment() {
    GROUP=$1
    CONFIG=$2
    DESCRIPTION=$3

    echo -e "${BLUE}=========================================="
    echo "Group ${GROUP}: ${DESCRIPTION}"
    echo "=========================================="
    echo "配置: ${CONFIG}"
    echo "开始时间: $(date)"
    echo -e "${NC}"

    # 清空对应的日志目录
    rm -rf logs/group_${GROUP,,}/*

    # 运行实验
    START_TIME=$(date +%s)

    python main.py --config $CONFIG \
        > $LOG_DIR/group_${GROUP}.log 2>&1

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # 转换时间为 HH:MM:SS
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    SECONDS=$((DURATION % 60))
    TIME_STR=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ Group ${GROUP} 完成 (耗时: ${TIME_STR})${NC}"

        # 提取最终 Dice
        FINAL_DICE=$(grep "验证集 Dice:" logs/group_${GROUP,,}/train.log 2>/dev/null | tail -1 | awk '{print $NF}')
        if [ -n "$FINAL_DICE" ]; then
            echo -e "${GREEN}   最终 Dice: ${FINAL_DICE}${NC}"
        fi
    else
        echo -e "${RED}❌ Group ${GROUP} 失败 (退出码: ${EXIT_CODE}, 耗时: ${TIME_STR})${NC}"

        # 显示最后几行错误日志
        echo -e "${RED}最后10行日志:${NC}"
        tail -10 $LOG_DIR/group_${GROUP}.log
    fi

    echo ""
    return $EXIT_CODE
}

# =============================================================================
# 运行三组实验
# =============================================================================

OVERALL_START=$(date +%s)

echo -e "${BLUE}=========================================="
echo "开始运行实验..."
echo "=========================================="
echo -e "${NC}"

# Group A: 纯视觉基线
run_experiment "A" "configs/exp_group_a.yaml" "纯视觉基线 (image_only)"
GROUP_A_STATUS=$?
sleep 5  # 等待5秒，确保资源释放

# Group B: 文本污染对照
run_experiment "B" "configs/exp_group_b.yaml" "文本污染对照 (image_only + multimodal)"
GROUP_B_STATUS=$?
sleep 5

# Group C: 解耦蒸馏终极方案
run_experiment "C" "configs/exp_group_c.yaml" "解耦蒸馏终极方案 (三客户端)"
GROUP_C_STATUS=$?

OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(( (TOTAL_DURATION % 3600) / 60 ))
TOTAL_TIME_STR=$(printf "%d小时%d分钟" $TOTAL_HOURS $TOTAL_MINUTES)

# =============================================================================
# 生成摘要报告
# =============================================================================

echo -e "${BLUE}=========================================="
echo "实验完成摘要"
echo "=========================================="
echo -e "${NC}"

echo "运行结果:"
echo -e "  Group A (纯视觉基线):      $([ $GROUP_A_STATUS -eq 0 ] && echo -e '${GREEN}✅ 成功${NC}' || echo -e '${RED}❌ 失败${NC}')"
echo -e "  Group B (文本污染对照):    $([ $GROUP_B_STATUS -eq 0 ] && echo -e '${GREEN}✅ 成功${NC}' || echo -e '${RED}❌ 失败${NC}')"
echo -e "  Group C (解耦蒸馏):        $([ $GROUP_C_STATUS -eq 0 ] && echo -e '${GREEN}✅ 成功${NC}' || echo -e '${RED}❌ 失败${NC}')"

echo ""
echo "指标对比:"

# 提取 Dice
DICE_A=$(grep "验证集 Dice:" logs/group_a/train.log 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
DICE_B=$(grep "验证集 Dice:" logs/group_b/train.log 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
DICE_C=$(grep "验证集 Dice:" logs/group_c/train.log 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")

echo "  Group A Dice: ${DICE_A}"
echo "  Group B Dice: ${DICE_B}"
echo "  Group C Dice: ${DICE_C}"

echo ""
echo "时间信息:"
echo "  总耗时: ${TOTAL_TIME_STR}"
echo "  开始时间: $(date -d @${OVERALL_START} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r ${OVERALL_START} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '未知')"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "文件位置:"
echo "  运行日志: ${LOG_DIR}/"
echo "  训练日志: logs/group_*/train.log"
echo "  检查点: logs/group_*/checkpoint_best.pth"

echo ""
echo -e "${BLUE}=========================================="
echo "建议下一步:"
echo "=========================================="
echo -e "${NC}"

if [ $GROUP_A_STATUS -eq 0 ] && [ $GROUP_B_STATUS -eq 0 ] && [ $GROUP_C_STATUS -eq 0 ]; then
    echo "1. 查看 TensorBoard:"
    echo "   tensorboard --logdir=logs --port=6006"
    echo ""
    echo "2. 下载结果到本地:"
    echo "   rsync -avz server:/path/to/logs ./logs_from_server/"
    echo ""
    echo "3. 提取关键指标:"
    echo "   grep 'Dice' logs/group_*/train.log"
    echo "   grep 'GRADIENT' logs/group_b/train.log"
else
    echo -e "${RED}部分实验失败，请检查日志:${NC}"
    echo "  tail -100 ${LOG_DIR}/group_*.log"
fi

echo ""

# 退出码：0 = 全部成功，1 = 部分失败
if [ $GROUP_A_STATUS -eq 0 ] && [ $GROUP_B_STATUS -eq 0 ] && [ $GROUP_C_STATUS -eq 0 ]; then
    exit 0
else
    exit 1
fi
