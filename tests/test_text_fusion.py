"""
测试文本融合模块
验证文本特征加载、融合和模型集成功能
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.text_fusion import (
    TextFeatureProjector,
    GatedFusion,
    CrossModalAttention
)
from src.model import SAM3_Medical
from src.data.text_feature_loader import TextFeatureLoader


def test_text_feature_projector():
    """测试文本特征投影器"""
    print("=" * 60)
    print("测试 TextFeatureProjector")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    text_dim = 512
    embed_dim = 768
    
    projector = TextFeatureProjector(text_dim, embed_dim).to(device)
    
    # 创建虚拟文本特征
    text_features = torch.randn(batch_size, text_dim).to(device)
    
    # 投影
    projected = projector(text_features)
    
    print(f"输入形状: {text_features.shape}")
    print(f"输出形状: {projected.shape}")
    print(f"✓ 投影成功")
    
    return True


def test_gated_fusion():
    """测试门控融合"""
    print("\n" + "=" * 60)
    print("测试 GatedFusion")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    num_tokens = 64
    embed_dim = 768
    
    # 测试不同融合类型
    fusion_types = ["gated", "additive", "multiplicative", "concatenate"]
    
    for fusion_type in fusion_types:
        fusion = GatedFusion(embed_dim, fusion_type=fusion_type).to(device)
        
        image_features = torch.randn(batch_size, num_tokens, embed_dim).to(device)
        text_features = torch.randn(batch_size, embed_dim).to(device)
        
        fused = fusion(image_features, text_features)
        
        print(f"{fusion_type:15s}: 输入图像 {image_features.shape}, 文本 {text_features.shape} -> 输出 {fused.shape}")
    
    print("✓ 所有融合类型测试通过")
    return True


def test_cross_modal_attention():
    """测试跨模态注意力"""
    print("\n" + "=" * 60)
    print("测试 CrossModalAttention")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    num_tokens = 64
    embed_dim = 768
    
    attention = CrossModalAttention(embed_dim, num_heads=8).to(device)
    
    image_features = torch.randn(batch_size, num_tokens, embed_dim).to(device)
    text_features = torch.randn(batch_size, embed_dim).to(device)
    
    fused = attention(image_features, text_features)
    
    print(f"输入图像: {image_features.shape}")
    print(f"输入文本: {text_features.shape}")
    print(f"输出: {fused.shape}")
    print("✓ 跨模态注意力测试通过")
    
    return True


def test_model_with_text_fusion():
    """测试带文本融合的模型"""
    print("\n" + "=" * 60)
    print("测试 SAM3_Medical with Text Fusion")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    img_size = 256  # 使用较小的尺寸以加快测试
    text_dim = 512
    
    # 创建模型（启用文本融合）
    model = SAM3_Medical(
        img_size=img_size,
        embed_dim=768,
        use_text_fusion=True,
        text_dim=text_dim,
        fusion_type="gated"
    ).to(device)
    
    # 创建虚拟输入
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    text_features = torch.randn(batch_size, text_dim).to(device)
    
    # 测试前向传播（带文本特征）
    print("测试前向传播（带文本特征）...")
    logits = model(images, text_features=text_features)
    print(f"  输入图像: {images.shape}")
    print(f"  输入文本: {text_features.shape}")
    print(f"  输出logits: {logits.shape}")
    
    # 测试前向传播（不带文本特征）
    print("\n测试前向传播（不带文本特征）...")
    logits_no_text = model(images, text_features=None)
    print(f"  输出logits: {logits_no_text.shape}")
    
    # 测试特征提取（带文本特征）
    print("\n测试特征提取（带文本特征）...")
    features = model.extract_features(images, text_features=text_features)
    print(f"  输出特征: {features.shape}")
    
    # 检查可训练参数
    trainable_params = model.get_trainable_params()
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n可训练参数量: {total_trainable:,}")
    
    print("✓ 模型测试通过")
    return True


def test_text_feature_loader():
    """测试文本特征加载器"""
    print("\n" + "=" * 60)
    print("测试 TextFeatureLoader")
    print("=" * 60)
    
    # 创建临时测试文件
    test_dir = Path("test_text_features")
    test_dir.mkdir(exist_ok=True)
    
    # 创建虚拟文本特征文件
    test_feature = np.random.randn(512).astype(np.float32)
    test_file = test_dir / "test_feature.npy"
    np.save(test_file, test_feature)
    
    # 测试加载器
    loader = TextFeatureLoader(base_dir=str(test_dir), cache=True)
    
    # 加载特征
    loaded_feature = loader.load_text_feature("test_feature.npy")
    
    if loaded_feature is not None:
        print(f"✓ 成功加载文本特征")
        print(f"  特征形状: {loaded_feature.shape}")
        print(f"  特征类型: {type(loaded_feature)}")
    else:
        print("✗ 加载失败")
        return False
    
    # 清理
    test_file.unlink()
    test_dir.rmdir()
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("文本融合模块测试套件")
    print("=" * 60)
    
    tests = [
        ("文本特征投影器", test_text_feature_projector),
        ("门控融合", test_gated_fusion),
        ("跨模态注意力", test_cross_modal_attention),
        ("模型集成", test_model_with_text_fusion),
        ("文本特征加载器", test_text_feature_loader),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n[SUCCESS] 所有测试通过！")
    else:
        print(f"\n[WARNING] {total - passed} 个测试失败")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

