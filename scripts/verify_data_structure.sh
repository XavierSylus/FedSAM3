#!/bin/bash
# ============================================================================
# 数据目录结构验证脚本
# 用于检查 FedSAM3-Cream 联邦学习项目的数据目录是否正确配置
# ============================================================================

echo "============================================================================"
echo "FedSAM3-Cream 数据结构验证"
echo "============================================================================"
echo ""

# 设置数据根目录
DATA_ROOT="data/federated_split/train"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# 1. 检查客户端目录结构
# ============================================================================
echo "========== 1. 客户端目录结构检查 =========="
echo ""

for client in client_1 client_2 client_3; do
    echo "[$client]"

    # 检查客户端目录是否存在
    if [ ! -d "$DATA_ROOT/$client" ]; then
        echo -e "  ${RED}✗ 错误: 客户端目录不存在${NC}"
        continue
    fi

    # 检查 private 目录
    if [ -d "$DATA_ROOT/$client/private" ]; then
        echo -e "  ${GREEN}✓ Private 目录存在${NC}"
    else
        echo -e "  ${RED}✗ Private 目录不存在${NC}"
    fi

    # 检查 public 目录
    if [ -d "$DATA_ROOT/$client/public" ]; then
        echo -e "  ${GREEN}✓ Public 目录存在${NC}"
    else
        echo -e "  ${YELLOW}! Public 目录不存在 (某些客户端可能不需要)${NC}"
    fi

    echo ""
done

# ============================================================================
# 2. 统计数据量
# ============================================================================
echo "========== 2. 数据量统计 =========="
echo ""

for client in client_1 client_2 client_3; do
    if [ ! -d "$DATA_ROOT/$client" ]; then
        continue
    fi

    echo "[$client]"

    # Private 数据统计
    if [ -d "$DATA_ROOT/$client/private" ]; then
        priv_count=$(find "$DATA_ROOT/$client/private/" -maxdepth 1 -type d -name "BraTS20_Training_*" 2>/dev/null | wc -l)
        echo "  Private: $priv_count 个病例"
    fi

    # Public 数据统计
    if [ -d "$DATA_ROOT/$client/public" ]; then
        pub_count=$(find "$DATA_ROOT/$client/public/" -maxdepth 1 -type d -name "BraTS20_Training_*" 2>/dev/null | wc -l)
        echo "  Public:  $pub_count 个病例"
    else
        echo "  Public:  N/A"
    fi

    echo ""
done

# ============================================================================
# 3. 检查错误的目录（BraTS2020_TrainingData）
# ============================================================================
echo "========== 3. 错误目录检查 =========="
echo ""

wrong_dirs=$(find "$DATA_ROOT" -name "BraTS2020_TrainingData" -type d 2>/dev/null)

if [ -z "$wrong_dirs" ]; then
    echo -e "${GREEN}✓ 未发现错误的目录名称${NC}"
else
    echo -e "${RED}✗ 发现错误的目录（需要删除）：${NC}"
    echo "$wrong_dirs"
    echo ""
    echo "建议执行以下命令删除："
    echo "find $DATA_ROOT -name 'BraTS2020_TrainingData' -type d -delete"
fi

echo ""

# ============================================================================
# 4. 验证病例目录内容
# ============================================================================
echo "========== 4. 病例内容验证 =========="
echo ""

for client in client_1 client_2 client_3; do
    if [ ! -d "$DATA_ROOT/$client/private" ]; then
        continue
    fi

    # 获取第一个病例目录
    first_case=$(find "$DATA_ROOT/$client/private/" -maxdepth 1 -type d -name "BraTS20_Training_*" 2>/dev/null | head -1)

    if [ -z "$first_case" ]; then
        echo -e "[$client] ${RED}✗ 未找到任何病例${NC}"
        continue
    fi

    echo "[$client] 示例病例: $(basename $first_case)"

    # 检查必需文件
    has_nii=false
    has_text=false
    has_seg=false

    if ls "$first_case"/*.nii &>/dev/null || ls "$first_case"/*.nii.gz &>/dev/null; then
        has_nii=true
    fi

    if ls "$first_case"/*_text.npy &>/dev/null; then
        has_text=true
    fi

    if ls "$first_case"/*_seg.nii &>/dev/null || ls "$first_case"/*_seg.nii.gz &>/dev/null; then
        has_seg=true
    fi

    # 打印检查结果
    if [ "$has_nii" = true ]; then
        nii_count=$(ls "$first_case"/*.nii "$first_case"/*.nii.gz 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓ 图像文件: $nii_count 个 .nii 文件${NC}"
    else
        echo -e "  ${RED}✗ 图像文件: 未找到 .nii 文件${NC}"
    fi

    if [ "$has_text" = true ]; then
        text_count=$(ls "$first_case"/*_text.npy 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓ 文本特征: $text_count 个 _text.npy 文件${NC}"
    else
        echo -e "  ${YELLOW}! 文本特征: 未找到 _text.npy 文件${NC}"
    fi

    if [ "$has_seg" = true ]; then
        echo -e "  ${GREEN}✓ 分割标签: 存在 _seg.nii 文件${NC}"
    else
        echo -e "  ${RED}✗ 分割标签: 未找到 _seg.nii 文件${NC}"
    fi

    echo ""
done

# ============================================================================
# 5. 详细目录树（如果安装了 tree 命令）
# ============================================================================
echo "========== 5. 目录树（前 3 层）=========="
echo ""

if command -v tree &> /dev/null; then
    tree -L 3 -d "$DATA_ROOT" 2>/dev/null | head -50
else
    echo "tree 命令未安装，使用 find 显示目录结构："
    find "$DATA_ROOT" -maxdepth 3 -type d 2>/dev/null | head -30
fi

echo ""

# ============================================================================
# 总结
# ============================================================================
echo "============================================================================"
echo "验证完成"
echo "============================================================================"
echo ""
echo "如果发现错误，请参考以下修复建议："
echo "1. 错误的目录名称 (BraTS2020_TrainingData)："
echo "   find $DATA_ROOT -name 'BraTS2020_TrainingData' -type d -delete"
echo ""
echo "2. 缺少 public 目录（client_1）："
echo "   如果不需要 public 数据，代码已经修复为兼容模式"
echo "   如果需要 public 数据，请手动创建并复制数据"
echo ""
echo "3. 缺少文本特征文件："
echo "   请运行文本特征提取脚本"
echo ""
