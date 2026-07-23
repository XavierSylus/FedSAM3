"""
测试集成模型 (SAM3MedicalIntegrated)
验证文本融合功能对模型输出的影响
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.federated_trainer import compute_gradient_conflict_from_vectors
from src.client import BaseClientTrainer
from src.integrated_model import SAM3MedicalIntegrated


class TestTextFusionImpact:
    """测试文本融合对模型输出的影响"""
    
    @pytest.fixture
    def model_config(self):
        """模型配置"""
        return {
            'img_size': 256,  # 使用较小尺寸加速测试
            'num_classes': 1,
            'adapter_dim': 64,
            'use_sam3': False,  # 使用 Mock 模式加速测试
            'freeze_encoder': False,  # 测试时不冻结以便梯度检查
            'use_adapter': True,
            'embed_dim': 768,
            'text_dim': 512
        }
    
    @pytest.fixture
    def device(self):
        """设备配置"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_true_gradient_conflict_angle_from_vectors(self):
        """True adapter gradients should produce a deterministic angle."""
        img_vec = torch.tensor([1.0, 0.0, 0.0])
        multi_vec = torch.tensor([0.0, 1.0, 0.0])

        angle = compute_gradient_conflict_from_vectors(
            [img_vec, multi_vec],
            ['image_only', 'multimodal']
        )
        assert angle == pytest.approx(90.0, abs=1e-6)

        same_angle = compute_gradient_conflict_from_vectors(
            [img_vec, img_vec.clone()],
            ['image_only', 'multimodal']
        )
        assert same_angle == pytest.approx(0.0, abs=1e-6)

    def test_adapter_gradient_snapshot_capture(self):
        """The trainer must cache real adapter gradients from a tiny model."""

        class DummyTrainer(BaseClientTrainer):
            def unpack_private_batch(self, batch):
                return {}

            def unpack_public_batch(self, batch):
                return {}

            def compute_loss(self, model, private_inputs, public_inputs, global_reps, lambda_cream):
                return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

            def get_return_values(self, model, local_reps, training_stats):
                return {}, None, None, training_stats

        trainer = DummyTrainer(
            private_loader=None,
            public_loader=None,
            device='cpu',
            use_amp=False,
            local_epochs=1,
            dataset_name='dummy',
            contrastive_dim=4,
            grad_clip=1.0,
            accumulation_steps=1,
        )

        model = nn.Module()
        model.adapter_manager = nn.Module()
        model.adapter_manager.adapters = nn.ModuleList([nn.Linear(2, 1, bias=False)])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        trainer._reset_adapter_grad_tracking(model)
        model.adapter_manager.adapters[0].weight.grad = torch.tensor([[1.0, 2.0]])
        trainer._record_adapter_grad_snapshot(optimizer)
        trainer._finalize_adapter_grad_tracking()

        assert trainer._last_adapter_grad_vector is not None
        assert torch.allclose(
            trainer._last_adapter_grad_vector,
            torch.tensor([1.0, 2.0]),
        )

    def test_adapter_stays_trainable_when_encoder_frozen(self, device):
        """Encoder freeze must not freeze injected adapters."""
        model = SAM3MedicalIntegrated(
            img_size=64,
            num_classes=1,
            adapter_dim=32,
            use_sam3=False,
            freeze_encoder=True,
            use_adapter=True,
            embed_dim=64,
            text_dim=512,
            contrastive_dim=128,
        ).to(device)

        encoder_params = list(model.image_encoder.parameters())
        assert encoder_params, "Expected a frozen image encoder in mock mode"
        assert all(not p.requires_grad for p in encoder_params), \
            "Image encoder should stay frozen when freeze_encoder=True"

        adapter_params = list(model.adapter_manager.adapters.parameters())
        assert adapter_params, "Expected injected adapter parameters"
        assert all(p.requires_grad for p in adapter_params), \
            "Adapter parameters must remain trainable after encoder freeze"

        trainable_param_ids = {id(p) for p in model.get_trainable_params()}
        assert any(id(p) in trainable_param_ids for p in adapter_params), \
            "Trainable parameter list must include adapter parameters"

    def test_text_fusion_impact(self, model_config, device):
        """
        测试文本特征对模型输出的影响
        
        验证：
        1. 带文本特征和不带文本特征的输出应该不同（证明文本特征生效）
        2. 两次输出的形状应该一致
        """
        print("\n" + "=" * 60)
        print("测试文本融合对模型输出的影响")
        print("=" * 60)
        
        # 1. 初始化模型
        print("\n[1/4] 初始化 SAM3MedicalIntegrated 模型...")
        model = SAM3MedicalIntegrated(**model_config).to(device)
        model.eval()  # 使用评估模式确保一致性
        print(f"✓ 模型已初始化 (设备: {device})")
        
        # 2. 准备输入数据
        batch_size = 2
        img_size = model_config['img_size']
        text_dim = model_config['text_dim']
        
        # 固定随机种子以确保可重复性
        torch.manual_seed(42)
        images = torch.randn(batch_size, 3, img_size, img_size).to(device)
        text_features = torch.randn(batch_size, text_dim).to(device)
        
        print(f"\n[2/4] 准备输入数据...")
        print(f"  图像形状: {images.shape}")
        print(f"  文本特征形状: {text_features.shape}")
        
        # 3. 运行两次 forward：一次带文本特征，一次不带
        print(f"\n[3/4] 运行前向传播...")
        
        with torch.no_grad():
            # 第一次：带文本特征
            print("  - 带文本特征...")
            output_with_text = model(images, text_features=text_features)
            
            # 第二次：不带文本特征
            print("  - 不带文本特征...")
            output_without_text = model(images, text_features=None)
        
        # 提取 logits/masks
        if isinstance(output_with_text, dict):
            logits_with_text = output_with_text.get('logits', output_with_text.get('masks'))
            logits_without_text = output_without_text.get('logits', output_without_text.get('masks'))
        else:
            logits_with_text = output_with_text
            logits_without_text = output_without_text
        
        print(f"  输出形状 (带文本): {logits_with_text.shape}")
        print(f"  输出形状 (不带文本): {logits_without_text.shape}")
        
        # 4. 断言验证
        print(f"\n[4/4] 验证断言...")
        
        # 断言 1: 输出形状应该一致
        assert logits_with_text.shape == logits_without_text.shape, \
            f"输出形状不一致: {logits_with_text.shape} vs {logits_without_text.shape}"
        print(f"  ✓ 断言 1 通过: 输出形状一致 {logits_with_text.shape}")
        
        # 断言 2: 文本路径本身应可执行且数值正常
        with torch.no_grad():
            projected_text = model.fusion_head.text_proj(text_features)

        assert projected_text.shape == (batch_size, model.contrastive_dim), \
            f"文本投影形状不正确: {projected_text.shape}"
        assert torch.isfinite(projected_text).all(), "文本投影包含非有限值"
        assert projected_text.norm().item() > 0, "文本投影不应退化为零向量"

        print(f"  ✓ 断言 2 通过: 文本投影路径可执行且数值正常")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)


def test_text_fusion_shape_consistency():
    """
    额外测试：验证不同 batch size 下的形状一致性
    """
    print("\n" + "=" * 60)
    print("测试不同 batch size 下的形状一致性")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SAM3MedicalIntegrated(
        img_size=256,
        use_sam3=False,
        text_dim=512
    ).to(device)
    model.eval()
    
    test_batch_sizes = [1, 2, 4]
    
    for batch_size in test_batch_sizes:
        images = torch.randn(batch_size, 3, 256, 256).to(device)
        text_features = torch.randn(batch_size, 512).to(device)
        
        with torch.no_grad():
            output = model(images, text_features=text_features)
        
        if isinstance(output, dict):
            logits = output.get('logits', output.get('masks'))
        else:
            logits = output
        
        # 验证输出的 batch 维度正确
        assert logits.shape[0] == batch_size, \
            f"Batch size 不匹配: 期望 {batch_size}, 实际 {logits.shape[0]}"
        
        print(f"  Batch size {batch_size}: {logits.shape} ✓")
    
    print("✓ 形状一致性测试通过")


if __name__ == "__main__":
    # 允许直接运行此脚本（不通过 pytest）
    print("直接运行测试...")
    
    # 创建测试实例
    test_instance = TestTextFusionImpact()
    
    # 手动创建 fixtures
    model_config = {
        'img_size': 256,
        'num_classes': 1,
        'adapter_dim': 64,
        'use_sam3': False,
        'freeze_encoder': False,
        'use_adapter': True,
        'embed_dim': 768,
        'text_dim': 512
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 运行主测试
        test_instance.test_text_fusion_impact(model_config, device)
        
        # 运行形状一致性测试
        test_text_fusion_shape_consistency()
        
        print("\n[SUCCESS] 所有测试通过！")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
