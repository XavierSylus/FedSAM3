#!/bin/bash
# ===============================================================
# FedSAM3-Cream: 完成真实BERT文本特征生成
# 继续生成剩余的 240/258 个特征
# ===============================================================

echo "==============================================================="
echo "FedSAM3-Cream: 完成文本特征生成"
echo "当前状态: 18/258 已完成，剩余 240 个待生成"
echo "==============================================================="

# 检查API密钥
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo ""
    echo "❌ 错误: DEEPSEEK_API_KEY 未设置"
    echo ""
    echo "请先运行:"
    echo "  export DEEPSEEK_API_KEY='your_api_key_here'"
    echo ""
    echo "获取API密钥: https://platform.deepseek.com/"
    exit 1
fi

echo "✅ DeepSeek API密钥已设置"

# 安装依赖
echo ""
echo "检查依赖包..."
pip install -q openai transformers torch nibabel tqdm scipy

# 开始生成
echo ""
echo "开始生成剩余240个真实BERT文本特征..."
echo "预计时间: 20-30分钟"
echo "==============================================================="

python scripts/generate_semantic_text_features.py \
    --data_root data/source_images/BraTS2020/BraTS2020_TrainingData/BraTS2020_Training \
    --force_overwrite

# 验证结果
echo ""
echo "==============================================================="
echo "验证生成结果..."
echo "==============================================================="

python << 'VERIFY'
import numpy as np
from pathlib import Path

data_root = Path('data/source_images/BraTS2020/BraTS2020_TrainingData/BraTS2020_Training')
text_files = list(data_root.rglob('*_flair_text.npy'))

real_count = 0
for text_file in text_files:
    features = np.load(text_file)
    mean = np.abs(np.mean(features))
    std = np.std(features)
    
    # 判断是否为真实BERT特征
    if not (mean < 0.002 and 0.035 < std < 0.037):
        real_count += 1

print(f"\n✅ 真实BERT特征数量: {real_count}/{len(text_files)}")

if real_count == len(text_files):
    print("🎉 完美！所有特征都是真实BERT特征，可以开始训练！")
elif real_count >= len(text_files) * 0.95:
    print("✅ 很好！绝大部分是真实特征，可以开始训练")
else:
    print(f"⚠️  只有 {real_count/len(text_files)*100:.1f}% 是真实特征")
VERIFY

echo ""
echo "==============================================================="
echo "✅ 完成！现在可以运行训练了"
echo "==============================================================="
