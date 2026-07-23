#!/bin/bash
# =============================================================================
# FedSAM3-Cream 服务器部署前检查脚本
# 在本地运行此脚本，确保所有文件都已准备好上传到服务器
# =============================================================================

echo "=========================================="
echo "FedSAM3-Cream 服务器部署前检查"
echo "=========================================="

# 检查计数器
PASSED=0
TOTAL=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_file() {
    TOTAL=$((TOTAL + 1))
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 存在"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗${NC} $1 不存在"
        return 1
    fi
}

check_dir() {
    TOTAL=$((TOTAL + 1))
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 目录存在"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗${NC} $1 目录不存在"
        return 1
    fi
}

# =============================================================================
# 1. 检查核心代码文件
# =============================================================================
echo -e "\n${YELLOW}1. 检查核心代码文件...${NC}"

check_file "main.py"
check_file "src/federated_trainer.py"
check_file "src/server.py"
check_file "src/client.py"
check_file "src/integrated_model.py"
check_file "src/cream_losses.py"
check_file "src/gradient_monitor.py"  # ⚠️ 新增
check_file "data/dataset_loader.py"

# =============================================================================
# 2. 检查配置文件
# =============================================================================
echo -e "\n${YELLOW}2. 检查配置文件...${NC}"

check_file "configs/exp_group_a.yaml"
check_file "configs/exp_group_b.yaml"
check_file "configs/exp_group_c.yaml"

# 检查 client_id 是否已修复
echo -e "\n检查 client_id 格式..."
if grep -q "client_id: client_" configs/exp_group_*.yaml; then
    echo -e "${GREEN}✓${NC} client_id 格式正确 (带下划线)"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} client_id 格式错误 (缺少下划线)"
fi
TOTAL=$((TOTAL + 1))

# =============================================================================
# 3. 检查数据目录结构
# =============================================================================
echo -e "\n${YELLOW}3. 检查数据目录结构...${NC}"

check_dir "data/federated_split/train"
check_dir "data/federated_split/val"
check_dir "data/federated_split/test"

# 检查客户端目录
check_dir "data/federated_split/val/client_1"
check_dir "data/federated_split/val/client_2"
check_dir "data/federated_split/val/client_3"

# =============================================================================
# 4. 检查验证集样本数
# =============================================================================
echo -e "\n${YELLOW}4. 检查验证集样本数...${NC}"

if [ -f "data/federated_split/val_split.json" ]; then
    VAL_COUNT=$(python3 -c "import json; print(len(json.load(open('data/federated_split/val_split.json'))['data']))" 2>/dev/null || echo "0")
    echo "验证集样本数: $VAL_COUNT"
    if [ "$VAL_COUNT" -gt 20 ]; then
        echo -e "${GREEN}✓${NC} 验证集样本充足"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠${NC} 验证集样本较少 (建议 >20)"
    fi
    TOTAL=$((TOTAL + 1))
fi

# =============================================================================
# 5. 检查依赖文件
# =============================================================================
echo -e "\n${YELLOW}5. 检查依赖文件...${NC}"

check_file "requirements.txt"

# =============================================================================
# 6. 检查脚本文件
# =============================================================================
echo -e "\n${YELLOW}6. 检查工具脚本...${NC}"

check_file "final_verification.py"
check_file "quick_test.py"
check_file "scripts/fix_client_ids.py"

# =============================================================================
# 7. 统计数据大小
# =============================================================================
echo -e "\n${YELLOW}7. 统计数据大小...${NC}"

if [ -d "data/federated_split" ]; then
    DATA_SIZE=$(du -sh data/federated_split 2>/dev/null | cut -f1)
    echo "数据目录大小: $DATA_SIZE"
fi

# =============================================================================
# 8. 生成文件清单
# =============================================================================
echo -e "\n${YELLOW}8. 生成上传文件清单...${NC}"

cat > upload_checklist.txt << 'EOF'
=============================================================================
FedSAM3-Cream 服务器上传文件清单
=============================================================================

必须上传的文件和目录：

1. 核心代码
   ✓ main.py
   ✓ src/
   ✓ data/dataset_loader.py
   ✓ configs/

2. 配置文件
   ✓ configs/exp_group_a.yaml
   ✓ configs/exp_group_b.yaml
   ✓ configs/exp_group_c.yaml

3. 数据（如果服务器上没有）
   ✓ data/federated_split/train/
   ✓ data/federated_split/val/
   ✓ data/federated_split/test/
   ✓ data/federated_split/*.json

4. 依赖文件
   ✓ requirements.txt

5. 工具脚本
   ✓ final_verification.py
   ✓ scripts/

可选但推荐：
   - docs/ (文档)
   - tests/ (测试)
   - tools/ (诊断工具)

不需要上传：
   ✗ logs/ (会在服务器上生成)
   ✗ results/ (会在服务器上生成)
   ✗ .git/ (太大)
   ✗ __pycache__/ (会自动生成)
   ✗ *.pyc (会自动生成)

=============================================================================
EOF

echo -e "${GREEN}✓${NC} 已生成 upload_checklist.txt"

# =============================================================================
# 9. 总结
# =============================================================================
echo -e "\n=========================================="
echo "检查摘要"
echo "=========================================="
echo "通过: $PASSED/$TOTAL"

if [ $PASSED -eq $TOTAL ]; then
    echo -e "${GREEN}✅ 所有检查通过！可以上传到服务器。${NC}"
    echo ""
    echo "下一步："
    echo "1. 查看 upload_checklist.txt 确认要上传的文件"
    echo "2. 使用 rsync 或 scp 上传到服务器"
    echo "3. 在服务器上运行 final_verification.py"
    exit 0
else
    echo -e "${RED}❌ 部分检查失败，请先修复问题。${NC}"
    exit 1
fi
