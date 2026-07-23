#!/bin/bash
# FedSAM3-Cream 环境修复脚本
# 修复 OMP_NUM_THREADS 和其他环境问题

echo "==============================================================="
echo "FedSAM3-Cream 环境修复"
echo "==============================================================="

# 修复 OMP_NUM_THREADS
echo ""
echo "[1/3] 修复 OMP_NUM_THREADS..."
# 检查当前值
if [ -n "$OMP_NUM_THREADS" ]; then
    echo "  当前 OMP_NUM_THREADS=$OMP_NUM_THREADS"
fi

# 设置为合法值（使用CPU核心数）
CPU_CORES=$(nproc 2>/dev/null || echo "4")
export OMP_NUM_THREADS=$CPU_CORES
echo "  ✅ 已设置 OMP_NUM_THREADS=$OMP_NUM_THREADS"

# 写入环境变量文件
echo "export OMP_NUM_THREADS=$CPU_CORES" >> ~/.bashrc
echo "  ✅ 已写入 ~/.bashrc"

# 修复 MKL 线程数（如果使用Intel MKL）
export MKL_NUM_THREADS=$CPU_CORES
echo "  ✅ 已设置 MKL_NUM_THREADS=$MKL_NUM_THREADS"

# 修复 PyTorch线程数
export TORCH_NUM_THREADS=$CPU_CORES
echo "  ✅ 已设置 TORCH_NUM_THREADS=$TORCH_NUM_THREADS"

echo ""
echo "[2/3] 验证PyTorch环境..."
python -c "
import torch
print(f'  PyTorch版本: {torch.__version__}')
print(f'  CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA版本: {torch.version.cuda}')
    print(f'  GPU数量: {torch.cuda.device_count()}')
    print(f'  GPU名称: {torch.cuda.get_device_name(0)}')
print(f'  CPU线程数: {torch.get_num_threads()}')
"

echo ""
echo "[3/3] 创建环境配置文件..."
cat > set_env.sh << 'EOF'
#!/bin/bash
# FedSAM3-Cream 运行环境配置
# 使用方法: source set_env.sh

# 线程数配置
CPU_CORES=$(nproc 2>/dev/null || echo "4")
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES
export TORCH_NUM_THREADS=$CPU_CORES

# 禁用不必要的警告
export PYTHONWARNINGS="ignore"

# CUDA优化
export CUDA_LAUNCH_BLOCKING=0  # 异步执行（更快）
# export CUDA_LAUNCH_BLOCKING=1  # 同步执行（调试时使用）

echo "✅ FedSAM3-Cream 环境已配置"
echo "   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "   MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "   TORCH_NUM_THREADS=$TORCH_NUM_THREADS"
EOF

chmod +x set_env.sh
echo "  ✅ 已创建 set_env.sh"

echo ""
echo "==============================================================="
echo "✅ 环境修复完成！"
echo "==============================================================="
echo ""
echo "使用方法:"
echo "  1. 当前会话生效: source set_env.sh"
echo "  2. 永久生效: 已自动添加到 ~/.bashrc"
echo "  3. 运行训练前: source set_env.sh && python main.py --config configs/proposed_method.yaml"
echo ""
