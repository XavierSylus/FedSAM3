#!/bin/bash

# FedSAM3-Cream 环境配置脚本 (Linux/Cloud)
# 用法: bash setup_env.sh

set -e  #遇到错误立即退出

echo "============================================================"
echo "      FedSAM3-Cream 环境自动配置脚本"
echo "============================================================"

# 1. 检查 Python 版本
echo "[1/4] 检查 Python 环境..."
python3 --version
PIP=pip
if ! command -v pip &> /dev/null; then
    echo "未找到 pip，尝试使用 pip3..."
    if command -v pip3 &> /dev/null; then
        PIP=pip3
    else
        echo "错误: 未找到 pip 或 pip3。请先安装 Python。"
        exit 1
    fi
fi
echo "使用 pip: $(which $PIP)"

# 2. 安装系统库 (如果需要，通常云服务器可能缺少一些图像库)
#这通常需要 sudo 权限，如果没有 root 权限可能会失败，所以加上 user 提示
echo "[2/4] 检查/安装基本系统依赖 (可选)..."
echo "注意: 如果不是 root 用户或没有 sudo 权限，您可以忽略下面的 apt-get 错误。"
if command -v apt-get &> /dev/null; then
    sudo apt-get update || true
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 || echo "系统依赖安装跳过 (无 sudo 权限)"
fi

# 3. 安装 Python 依赖
echo "[3/4] 安装 Python 依赖 (pip install)..."
$PIP install --upgrade pip
$PIP install -r requirements.txt

# 4. 安装本地包 (如果需要)
echo "[4/4] 检查项目完整性..."
if [ -d "core_projects/sam3-main" ]; then
    echo "SAM3 目录存在。"
    # 可选：如果 SAM3 需要本地安装
    # cd core_projects/sam3-main && $PIP install -e . && cd ../..
else
    echo "警告: core_projects/sam3-main 未找到，请确保已下载完整项目。"
fi

echo "============================================================"
echo "  配置完成! "
echo "  您现在可以运行: python scripts/train_brats_federated.py"
echo "============================================================"
