@echo off
REM ===============================================================
REM FedSAM3-Cream 快速启动脚本（Windows版）
REM ===============================================================

echo ================================================================
echo   FedSAM3-Cream 快速启动向导
echo ================================================================
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python
    exit /b 1
)

echo ✓ Python已安装
echo.

REM 步骤1: 数据验证
echo [步骤 1/3] 验证数据...
echo ----------------------------------------------------------------
python scripts/quick_data_check.py --data_root data/federated_split

if %errorlevel% neq 0 (
    echo.
    echo ❌ 数据验证失败！
    echo    请确保数据目录存在: data/federated_split/
    exit /b 1
)

echo.
echo ✓ 数据验证通过
echo.

REM 步骤2: 检查依赖
echo [步骤 2/3] 检查依赖...
echo ----------------------------------------------------------------
python -c "import torch; import yaml; import monai" 2>nul

if %errorlevel% neq 0 (
    echo ⚠️  缺少关键依赖，正在安装...
    pip install pyyaml monai -q
)

echo ✓ 依赖检查通过
echo.

REM 步骤3: 启动训练
echo [步骤 3/3] 启动训练...
echo ----------------------------------------------------------------
echo 使用配置: configs/exp_production.yaml
echo 训练模式: Mock模型（快速测试）
echo 训练轮数: 3 轮
echo 批次大小: 1
echo.
echo 开始训练...
echo ================================================================
echo.

python main.py --config configs/exp_production.yaml --rounds 3 --batch_size 1

set EXIT_CODE=%errorlevel%

echo.
echo ================================================================

if %EXIT_CODE% equ 0 (
    echo ✓ 训练完成！
    echo.
    echo 查看结果:
    echo   - 日志: logs/production/
    echo   - 检查点: data/checkpoints/
    echo   - 训练曲线: data/checkpoints/plots/
) else (
    echo ❌ 训练失败
    echo.
    echo 故障排查:
    echo   1. 检查错误日志
    echo   2. 确认数据完整性
    echo   3. 检查显存是否充足
)

echo ================================================================

exit /b %EXIT_CODE%
