# 🚀 FedSAM3-Cream 服务器部署完整指南

> **目标**: 在服务器上成功运行三组对比实验
> **预计时间**: 部署 30 分钟，每组实验 2-8 小时

---

## 📋 部署前检查清单

### Step 1: 本地检查（5分钟）

```bash
# Windows (Git Bash)
bash pre_deploy_check.sh

# 或者手动检查
python final_verification.py
```

**必须通过的检查**：
- ✅ client_id 格式正确 (`client_1`, `client_2`, `client_3`)
- ✅ 验证集样本数 ≥ 20
- ✅ 梯度监控模块存在
- ✅ 配置文件完整

---

## 🔧 Step 2: 集成梯度监控（重要！）

**在上传前必须完成这一步**，否则无法监控梯度夹角。

### 2.1 编辑 `src/server.py`

找到 `CreamAggregator` 类，添加梯度监控：

```python
# ============ 在文件开头添加导入 ============
from src.gradient_monitor import GradientMonitor

# ============ 在 CreamAggregator 类中 ============
class CreamAggregator:
    def __init__(self, global_model, config, device, logger=None):
        # ... 现有代码 ...

        # ✅ 新增：初始化梯度监控器
        self.gradient_monitor = GradientMonitor(logger=self.logger)
        self.logger.info("✓ 梯度监控器已初始化")

    def aggregate(self, clients_data, round_idx):
        """聚合客户端更新"""

        # ... 现有的聚合逻辑 ...

        # ✅ 新增：计算梯度余弦相似度
        if len(clients_data) >= 2:
            try:
                # 计算所有客户端之间的梯度相似度
                grad_similarities = self.gradient_monitor.compute_gradient_cosine_similarity(
                    clients_data,
                    filter_patterns=['adapter']  # 重点关注 Adapter 层
                )

                # 记录到 TensorBoard
                if hasattr(self, 'writer') and self.writer is not None:
                    for key, value in grad_similarities.items():
                        if 'cosine' in key or 'angle' in key:
                            if isinstance(value, (int, float)):
                                self.writer.add_scalar(f'gradient/{key}', value, round_idx)

                # 每5轮打印详细摘要
                if round_idx % 5 == 0:
                    self.gradient_monitor.log_summary(grad_similarities)

            except Exception as e:
                self.logger.warning(f"梯度相似度计算失败: {e}")

        # ... 返回聚合结果 ...
        return aggregated_weights
```

### 2.2 验证集成

```bash
# 检查是否正确导入
python -c "from src.server import CreamAggregator; print('✓ 集成成功')"

# 检查语法错误
python -m py_compile src/server.py
```

---

## 📤 Step 3: 上传到服务器

### 3.1 准备上传脚本

创建 `upload_to_server.sh`：

```bash
#!/bin/bash
# 上传脚本 - 根据你的服务器信息修改

# ============ 配置项（修改这里）============
SERVER_USER="your_username"           # 你的用户名
SERVER_HOST="your_server.edu.cn"     # 服务器地址
SERVER_PORT="22"                      # SSH 端口
REMOTE_DIR="/home/$SERVER_USER/FedSAM3-Cream"  # 服务器目录

# ============ 上传命令 ============
echo "开始上传到服务器..."

# 使用 rsync（推荐，支持断点续传）
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='logs/' \
  --exclude='results/' \
  --exclude='.pytest_cache' \
  -e "ssh -p $SERVER_PORT" \
  ./ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/

echo "✅ 上传完成！"

# 或者使用 scp（简单但慢）
# scp -r -P $SERVER_PORT \
#   ./ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/
```

### 3.2 执行上传

```bash
# 给脚本执行权限
chmod +x upload_to_server.sh

# 上传（会提示输入密码）
./upload_to_server.sh

# 或者手动上传
rsync -avz --exclude='.git' --exclude='logs/' \
  ./ username@server:/path/to/FedSAM3-Cream/
```

**数据上传**（如果服务器上没有数据）：

```bash
# 单独上传数据（比较大）
rsync -avz --progress \
  data/federated_split/ \
  username@server:/path/to/FedSAM3-Cream/data/federated_split/
```

---

## 🖥️ Step 4: 服务器环境配置

### 4.1 登录服务器

```bash
ssh username@your_server.edu.cn
cd /path/to/FedSAM3-Cream
```

### 4.2 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n fedsam3 python=3.8 -y
conda activate fedsam3

# 或使用 venv
python3.8 -m venv venv
source venv/bin/activate
```

### 4.3 安装依赖

```bash
# 安装 PyTorch（根据服务器 CUDA 版本）
# 查看 CUDA 版本
nvidia-smi

# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 验证 GPU 可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 4.4 最终验证

```bash
# 运行验证脚本
python final_verification.py

# 预期输出：
# ✅ 所有 client_id 格式正确
# ✅ 所有验证集目录存在
# ✅ 数据加载功能正常
# ✅ 梯度监控模块可用
```

---

## 🏃 Step 5: 后台运行实验

### 5.1 创建运行脚本

创建 `run_experiments.sh`：

```bash
#!/bin/bash
# =============================================================================
# 自动运行三组实验脚本
# =============================================================================

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh  # 或你的 conda 路径
conda activate fedsam3

# 设置日志目录
LOG_DIR="logs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# 函数：运行单个实验
run_experiment() {
    GROUP=$1
    CONFIG=$2

    echo "=========================================="
    echo "开始运行 Group $GROUP"
    echo "配置: $CONFIG"
    echo "时间: $(date)"
    echo "=========================================="

    # 运行实验，输出重定向到日志文件
    python main.py --config $CONFIG \
        > $LOG_DIR/group_${GROUP}.log 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Group $GROUP 完成"
    else
        echo "❌ Group $GROUP 失败 (退出码: $EXIT_CODE)"
    fi

    echo ""
    return $EXIT_CODE
}

# =============================================================================
# 顺序运行三组实验
# =============================================================================

echo "开始运行 FedSAM3-Cream 实验"
echo "日志目录: $LOG_DIR"
echo ""

# Group A: 纯视觉基线
run_experiment "A" "configs/exp_group_a.yaml"
GROUP_A_STATUS=$?

# Group B: 文本污染对照
run_experiment "B" "configs/exp_group_b.yaml"
GROUP_B_STATUS=$?

# Group C: 解耦蒸馏终极方案
run_experiment "C" "configs/exp_group_c.yaml"
GROUP_C_STATUS=$?

# =============================================================================
# 生成摘要报告
# =============================================================================

echo "=========================================="
echo "实验摘要"
echo "=========================================="
echo "Group A (纯视觉基线): $([ $GROUP_A_STATUS -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo "Group B (文本污染对照): $([ $GROUP_B_STATUS -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo "Group C (解耦蒸馏): $([ $GROUP_C_STATUS -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo ""
echo "日志位置: $LOG_DIR"
echo "完成时间: $(date)"
echo "=========================================="

# 发送通知（可选）
# echo "实验完成！" | mail -s "FedSAM3-Cream 实验完成" your_email@example.com
```

### 5.2 使用 tmux/screen 后台运行

**方案 1: 使用 tmux（推荐）**

```bash
# 安装 tmux（如果没有）
# sudo apt-get install tmux  # Ubuntu
# sudo yum install tmux       # CentOS

# 创建新会话
tmux new -s fedsam3

# 在 tmux 中运行
bash run_experiments.sh

# 退出 tmux（实验继续运行）
# 按 Ctrl+B，然后按 D

# 重新连接
tmux attach -t fedsam3

# 查看所有会话
tmux ls

# 终止会话（如果需要）
tmux kill-session -t fedsam3
```

**方案 2: 使用 screen**

```bash
# 创建新 screen
screen -S fedsam3

# 运行实验
bash run_experiments.sh

# 退出 screen
# 按 Ctrl+A，然后按 D

# 重新连接
screen -r fedsam3
```

**方案 3: 使用 nohup（最简单）**

```bash
# 后台运行，输出到 nohup.out
nohup bash run_experiments.sh &

# 查看输出
tail -f nohup.out

# 查看进程
ps aux | grep python

# 终止（如果需要）
kill -9 <PID>
```

---

## 📊 Step 6: 监控和日志

### 6.1 实时查看日志

```bash
# 查看 Group A 日志（实时）
tail -f logs/experiments_*/group_A.log

# 查看最近50行
tail -50 logs/group_a/train.log

# 搜索关键词
grep "Dice" logs/group_a/train.log
grep "梯度" logs/group_b/train.log
grep "GRADIENT" logs/group_b/train.log
```

### 6.2 查看 GPU 使用情况

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

### 6.3 查看 TensorBoard

```bash
# 在服务器上启动 TensorBoard
tensorboard --logdir=logs --port=6006 --bind_all

# 在本地浏览器访问（需要端口转发）
# 在本地终端运行：
ssh -L 6006:localhost:6006 username@server

# 然后在浏览器打开：
# http://localhost:6006
```

### 6.4 检查梯度相似度日志

```bash
# 查看梯度监控输出
grep "GRADIENT" logs/group_b/train.log | tail -20

# 预期看到类似输出：
# [GRADIENT] client_2(image_only) vs client_3(multimodal)_adapter: cosine=..., angle=...°
```

---

## 🔍 Step 7: 检查实验是否正常运行

### 7.1 运行后 10 分钟检查

```bash
# 1. 检查进程是否还在运行
ps aux | grep "python main.py"

# 2. 检查日志文件是否在增长
ls -lh logs/group_a/train.log

# 3. 检查是否有错误
grep -i "error\|traceback\|exception" logs/group_a/train.log | tail -20

# 4. 检查 Dice 是否更新
grep "Dice:" logs/group_a/train.log | tail -5
```

### 7.2 预期的正常输出

```
Round 1/60:
  Client client_2 训练完成: loss=1.234, seg_loss=1.100, cream_loss=0.000
  聚合完成
  验证集 Dice: 0.123

Round 5/60:
  Client client_2 训练完成: loss=0.876, seg_loss=0.850, cream_loss=0.000
  聚合完成
  验证集 Dice: 0.345
  [GRADIENT] ...  # Group B/C 才有

Round 10/60:
  ...
  验证集 Dice: 0.567
```

---

## ⚠️ Step 8: 常见问题和应急处理

### 问题 1: CUDA out of memory

**解决方案**：

```yaml
# 编辑配置文件，减小 batch_size
training:
  batch_size: 1  # 从 2 减到 1
  accumulation_steps: 8  # 增加累积步数保持有效批大小
```

### 问题 2: 数据加载失败

**检查**：

```bash
# 验证数据路径
ls -la data/federated_split/val/client_2/private/

# 检查样本数
find data/federated_split/val/client_2/private/ -type d | wc -l
```

### 问题 3: 梯度监控没有输出

**检查**：

```bash
# 1. 确认 gradient_monitor.py 存在
ls -la src/gradient_monitor.py

# 2. 检查 server.py 是否集成
grep "GradientMonitor" src/server.py

# 3. 检查是否有多个客户端
# Group A 只有 1 个客户端，不会输出梯度相似度
# Group B/C 有 2-3 个客户端，应该有输出
```

### 问题 4: 训练速度太慢

**优化**：

```yaml
# 减少验证频率
validation:
  val_interval: 5  # 从 1 改为 5，每 5 轮验证一次

# 减少数据加载 workers
system:
  num_workers: 2  # 从 4 减到 2
```

### 问题 5: 如何终止实验

```bash
# 方法 1: 在 tmux/screen 中 Ctrl+C

# 方法 2: 找到进程并 kill
ps aux | grep "python main.py"
kill -9 <PID>

# 方法 3: killall
killall -9 python
```

---

## 📈 Step 9: 实验完成后的结果收集

### 9.1 下载日志和结果

在本地运行：

```bash
# 下载所有日志
rsync -avz --progress \
  username@server:/path/to/FedSAM3-Cream/logs/ \
  ./logs_from_server/

# 下载 TensorBoard 文件
rsync -avz --progress \
  username@server:/path/to/FedSAM3-Cream/logs/*/events.out.tfevents.* \
  ./tensorboard_logs/

# 下载检查点
rsync -avz --progress \
  username@server:/path/to/FedSAM3-Cream/logs/*/checkpoint_best.pth \
  ./checkpoints/
```

### 9.2 提取关键指标

```bash
# 提取所有 Dice 值
grep "验证集 Dice:" logs/group_a/train.log | awk '{print $NF}' > group_a_dice.txt
grep "验证集 Dice:" logs/group_b/train.log | awk '{print $NF}' > group_b_dice.txt
grep "验证集 Dice:" logs/group_c/train.log | awk '{print $NF}' > group_c_dice.txt

# 提取梯度夹角
grep "angle=" logs/group_b/train.log | grep "adapter" > group_b_gradients.txt
grep "angle=" logs/group_c/train.log | grep "adapter" > group_c_gradients.txt
```

---

## ✅ 最终检查清单

在提交到服务器前，确认：

- [ ] 已运行 `final_verification.py` 并通过
- [ ] 已集成梯度监控到 `src/server.py`
- [ ] `client_id` 已修复为 `client_1`, `client_2`, `client_3`
- [ ] 配置文件中的路径正确
- [ ] requirements.txt 完整
- [ ] 数据已上传或在服务器上可用
- [ ] 服务器环境已配置（conda/venv + PyTorch + 依赖）
- [ ] GPU 可用（`nvidia-smi` 检查）
- [ ] 准备好后台运行脚本（tmux/screen/nohup）
- [ ] 了解如何查看日志和监控进度

---

## 🎯 预期实验时间

| 实验 | 轮数 | 预计时间 | 说明 |
|------|------|---------|------|
| Group A | 60轮 | 2-4小时 | 单客户端，最快 |
| Group B | 60轮 | 4-6小时 | 双客户端 |
| Group C | 60轮 | 6-8小时 | 三客户端，最慢 |
| **总计** | 180轮 | **12-18小时** | 建议夜间运行 |

---

## 📞 需要帮助？

如果遇到问题：

1. 检查日志中的错误信息
2. 运行 `python final_verification.py` 再次验证
3. 查看 `docs/DIAGNOSTIC_GUIDE.md`

**祝实验顺利！** 🚀
