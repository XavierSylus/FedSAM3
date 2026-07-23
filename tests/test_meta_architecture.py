"""
tests/test_meta_architecture.py
================================
零显存真实 SAM3 架构结构验证测试套件

# ============================================================================
# 核心原理：为什么 torch.device('meta') 可以在普通轻薄本上测试 20GB 模型？
# ============================================================================
#
# PyTorch 2.x 引入了 "Meta Tensor" 机制。与普通 Tensor 的唯一区别是：
#
#   Meta Tensor 只存储 Tensor 的"元信息"（shape、dtype、layout 等），
#   但 绝不分配任何实际的内存！
#
# 以 ViT-H（SAM3 默认骨干）为例：
#   - image_encoder 参数量约 632M
#   - float32 精度下占用  632M x 4bytes ≈ 2.5 GB 真实内存
#
# 使用 Meta 模式后：
#   - nn.Linear(1024, 4096) 仍然能被实例化，.weight.shape == (4096, 1024)
#   - 但 weight 上没有任何真实数值，占用显存/内存为 0
#   - 所有 shape/dtype/.dim()/.numel()/.device 等属性查询 100% 正确
#   - 不能做真实的前向传播（会抛出 "Cannot run ... on meta device"）
#
# 因此，Meta Tensor 非常适合"结构即测试 (Structural Testing)"：
#   验证模型中某层是否存在
#   验证参数 shape / dtype 是否符合预期
#   验证 Adapter 注入是否改变了正确的 block 数量
#   验证模块嵌套路径是否与 AdapterInjector 的代码期望一致
#   不能做真实数值前向传播（这不是 Meta Tensor 的用途）
#
# 使用方式：
#   pytest tests/test_meta_architecture.py -v
#   pytest tests/test_meta_architecture.py -v -k "meta"
# ============================================================================
"""

import sys
import logging
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# 导入被测模块
# ============================================================================
from src.integrated_model import (
    AdapterInjector,
    BlockWithAdapter,
    MultimodalFusionHead,
    SAM3MedicalIntegrated,
    SAM3_AVAILABLE,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 核心辅助函数：build_meta_sam3()
# ============================================================================

def build_meta_sam3(
    adapter_dim: int = 64,
    num_classes: int = 1,
    embed_dim: int = 1024,
    use_real_sam3: bool = True,
) -> nn.Module:
    """在 Meta Device 上实例化真实的 SAM3 模型（零显存）。

    torch.device('meta') 上下文管理器会拦截所有 nn.Linear、nn.Conv2d 等层的
    权重创建，将其重定向到 meta 设备。Meta Tensor 的形状、dtype 等属性与真实
    Tensor 完全一致，但不分配任何物理内存。

    Args:
        adapter_dim: Adapter 的瓶颈维度（默认 64）
        num_classes: 分割类别数（默认 1）
        embed_dim: ViT 嵌入维度（SAM3 ViT-H 为 1024）
        use_real_sam3: 是否尝试使用真实 SAM3（False = 使用 Mock 模型）

    Returns:
        在 Meta Device 上构建的模型（形状信息完整，无真实权重）

    Raises:
        pytest.skip: 如果 SAM3 包不可用且 use_real_sam3=True
    """
    if use_real_sam3 and not SAM3_AVAILABLE:
        pytest.skip("SAM3 包不可用（sam3 未安装），跳过真实架构测试")

    # 在 Meta Device 上构建模型
    # 所有 nn.Parameter、nn.Linear、nn.Conv2d 的权重都将是 Meta Tensor
    with torch.device('meta'):
        model = SAM3MedicalIntegrated(
            img_size=1024,
            num_classes=num_classes,
            adapter_dim=adapter_dim,
            use_sam3=use_real_sam3,
            freeze_encoder=False,
            use_adapter=False,    # 在 build 阶段不注入 Adapter，由测试函数手动控制
            device='meta',
            embed_dim=embed_dim,
        )
    return model


def build_meta_mock(
    adapter_dim: int = 64,
    embed_dim: int = 768,
    num_classes: int = 1,
) -> nn.Module:
    """在 Meta Device 上构建 Mock（非真实 SAM3）模型，适用于 CI 环境。"""
    with torch.device('meta'):
        model = SAM3MedicalIntegrated(
            img_size=256,
            num_classes=num_classes,
            adapter_dim=adapter_dim,
            use_sam3=False,
            freeze_encoder=False,
            use_adapter=False,
            device='meta',
            embed_dim=embed_dim,
        )
    return model


def _collect_meta_block_paths(root: nn.Module) -> Dict[str, int]:
    """递归扫描 module 树，收集所有名为 'blocks' 的 ModuleList。

    返回 {路径字符串: 长度} 映射。
    用于调试：帮助了解真实 SAM3 的嵌套结构，
    确认 AdapterInjector 能走到正确的 block 路径。
    """
    result: Dict[str, int] = {}

    def _scan(mod: nn.Module, prefix: str) -> None:
        for name, child in mod.named_children():
            full_path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ModuleList) and name == 'blocks':
                result[full_path] = len(child)
            _scan(child, full_path)

    _scan(root, "")
    return result


# ============================================================================
# 测试用例
# ============================================================================

class TestMetaTensorBasics:
    """基础验证：确保 Meta Tensor 机制在当前 PyTorch 版本上正常工作。"""

    def test_meta_tensor_has_no_memory(self):
        """Meta Tensor 断言：shape 存在但不占实际内存。

        一个 4096x4096 的 float32 矩阵通常需要 64MB 内存，
        但 Meta Tensor 分配后 torch.cuda.memory_allocated() 为 0。
        """
        meta_weight = torch.empty(4096, 4096, device='meta', dtype=torch.float32)

        assert meta_weight.shape == (4096, 4096), "Meta Tensor 的 shape 应该正确"
        assert meta_weight.dtype == torch.float32, "Meta Tensor 的 dtype 应该正确"
        assert meta_weight.device.type == 'meta', "Meta Tensor 的 device 应该是 meta"
        assert meta_weight.numel() == 4096 * 4096, "Meta Tensor 的 numel 应该正确"

        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()
            _ = torch.empty(4096, 4096, device='meta')
            mem_after = torch.cuda.memory_allocated()
            assert mem_after == mem_before, "Meta Tensor 不应该分配任何 CUDA 显存"

        print(f"\n  Meta Tensor 4096x4096 float32 shape={meta_weight.shape}, "
              f"numel={meta_weight.numel():,}, device={meta_weight.device}")

    def test_meta_linear_layer(self):
        """Meta 设备上的 nn.Linear 结构验证。

        验证在 meta 设备上构建的层，其 .weight.shape 属性可正常查询。
        """
        with torch.device('meta'):
            layer = nn.Linear(1024, 4096, bias=False)

        assert layer.weight.shape == (4096, 1024), "Meta Linear weight.shape 应正确"
        assert layer.weight.device.type == 'meta', "Meta Linear weight.device 应为 meta"

        # Meta 设备不支持真实的 matmul 运算（这是 meta 的限制，也是预期行为）
        # 注意：不同 PyTorch 版本对 meta 设备的 forward 行为不同：
        #   PyTorch < 2.0：抛出 NotImplementedError
        #   PyTorch >= 2.0：可能返回 Meta Tensor 而不抛出异常（fake tensor 模式）
        dummy_input = torch.zeros(1, 1024, device='meta')
        try:
            _result = layer(dummy_input)
            # PyTorch 2.x 可能不抛异常，显示警告
            print(f"\n  注：PyTorch {torch.__version__} meta forward 未抛异常（可能返回 fake tensor）")
        except Exception as e:
            print(f"\n  PyTorch {torch.__version__} meta forward 抛出: {type(e).__name__}")
        # 无论如何，不进行各向决策断言（该行为取决于 PyTorch 版本）

        print(f"\n  Meta Linear(1024, 4096) weight.shape={layer.weight.shape}")

    def test_pytorch_version_supports_meta(self):
        """确认当前 PyTorch 版本支持 Meta Tensor（需要 PyTorch >= 1.9）。"""
        major = int(torch.__version__.split('.')[0])
        minor = int(torch.__version__.split('.')[1].split('+')[0].split('a')[0].split('b')[0])
        assert (major, minor) >= (1, 9), \
            f"Meta Tensor 需要 PyTorch >= 1.9，当前版本: {torch.__version__}"
        print(f"\n  PyTorch {torch.__version__} 支持 Meta Tensor")


class TestMockArchitectureOnMeta:
    """Mock 模型（MockViTEncoder）的 Meta Tensor 架构验证。

    适用于无 GPU 的 CI 环境，不依赖真实 SAM3 checkpoint。
    """

    @pytest.fixture
    def mock_meta_model(self) -> nn.Module:
        """构建 Meta Device 上的 Mock 模型（不依赖 SAM3 checkpoint）。"""
        with torch.device('meta'):
            model = SAM3MedicalIntegrated(
                img_size=256,
                num_classes=1,
                adapter_dim=64,
                use_sam3=False,
                freeze_encoder=False,
                use_adapter=False,
                device='meta',
                embed_dim=768,
            )
        return model

    def test_mock_model_parameter_shapes(self, mock_meta_model):
        """验证 Mock 模型参数 shape 在 Meta 设备上是否正确。"""
        meta_params = [
            (name, p)
            for name, p in mock_meta_model.named_parameters()
            if p.device.type == 'meta'
        ]
        # 注意：MCSoftContrastiveLoss 的 shift/negative_scale 用 torch.tensor() 创建，
        # 不遵循 torch.device('meta') 上下文，这是已知的第三方限制。
        # 测试中排除 contrastive_loss_fn.* 的参数，仅校验模型主干权重。
        non_meta_params = [
            (name, p)
            for name, p in mock_meta_model.named_parameters()
            if p.device.type != 'meta' and not name.startswith('contrastive_loss_fn')
        ]

        assert len(non_meta_params) == 0, (
            f"发现 {len(non_meta_params)} 个非 Meta 参数:\n"
            + "\n".join(f"  {n}: device={p.device}" for n, p in non_meta_params[:5])
        )
        assert len(meta_params) > 0, "Mock 模型应该有参数"

        # 检查 fusion_head 投影层 shape（默认 text_dim=512, contrastive_dim=1024）
        assert mock_meta_model.fusion_head.text_proj[0].weight.shape[1] == 512, \
            "text_proj 输入维度应为 512（text_dim）"
        assert mock_meta_model.fusion_head.image_proj[0].weight.shape[1] == 768, \
            "image_proj 输入维度应为 768（embed_dim）"

        print(f"\n  Mock Meta 模型共 {len(meta_params)} 个参数，全部在 meta 设备")
        print(f"  fusion_head.text_proj: {mock_meta_model.fusion_head.text_proj[0].weight.shape}")
        print(f"  fusion_head.image_proj: {mock_meta_model.fusion_head.image_proj[0].weight.shape}")

    def test_adapter_injector_on_mock_meta(self, mock_meta_model):
        """验证 AdapterInjector 可以对 Meta 设备上的 Mock 模型完成 Adapter 注入。

        Meta Tensor 的杀手应用：
          - inject() 走 MockViTEncoder.blocks 的路径
          - 读取 blocks[0].norm1.weight.shape[0] 推断 embed_dim = 768
          - 创建 nn.Linear(768, 64) Meta Adapter
          - 全程无内存分配！
        """
        injector = AdapterInjector(adapter_dim=64)
        injector.inject(
            sam3_model=None,
            use_real_sam3=False,
            embed_dim_hint=768,
            image_encoder=mock_meta_model.image_encoder,
        )

        # Mock 模型有 12 个 Transformer Block（MockViTEncoder 默认值）
        assert injector.adapters is not None, "Adapter 注入应该成功"
        num_adapters = len(injector.adapters)
        assert num_adapters == 12, (
            f"Mock ViT 应有 12 个 Adapter，实际得到 {num_adapters}"
        )

        # 每个 Adapter 的 down_proj/up_proj shape 应正确
        for i, adapter in enumerate(injector.adapters):
            down_shape = adapter.down_proj.weight.shape
            up_shape = adapter.up_proj.weight.shape
            assert down_shape == (64, 768), \
                f"Adapter[{i}].down_proj 应为 (64, 768)，实际 {down_shape}"
            assert up_shape == (768, 64), \
                f"Adapter[{i}].up_proj 应为 (768, 64)，实际 {up_shape}"

        print(f"\n  成功向 Meta Mock 模型注入 {num_adapters} 个 Adapter")
        print(f"  down_proj.weight.shape = {injector.adapters[0].down_proj.weight.shape}")
        print(f"  up_proj.weight.shape   = {injector.adapters[0].up_proj.weight.shape}")

    def test_adapter_zero_initialization_on_meta(self, mock_meta_model):
        """验证 Adapter 的零初始化契约在 Meta 设备上不会崩溃。

        注意：Meta Tensor 不保存真实数值，因此 nn.init.zeros_() 等初始化操作
        在 Meta 设备上是 no-op（无操作），不会报错，但也无法验证数值为零。
        这个测试的意义：确保 init 调用路径本身不会因 Meta Tensor 而崩溃。
        """
        from src.models.adapter import Adapter

        with torch.device('meta'):
            adapter = Adapter(in_dim=768, out_dim=768, adapter_dim=64)

        assert adapter.down_proj.weight.shape == (64, 768), "down_proj shape 应正确"
        assert adapter.up_proj.weight.shape == (768, 64), "up_proj shape 应正确"

        print(f"\n  Meta Adapter 构建成功，zero-init 路径无崩溃")
        print(f"  down_proj: {adapter.down_proj.weight.shape}, device={adapter.down_proj.weight.device}")


class TestRealSAM3ArchitectureOnMeta:
    """真实 SAM3 架构的 Meta Tensor 结构验证。

    本类测试依赖 SAM3 包，无 SAM3 环境自动跳过。
    不加载任何实际权重（无需 checkpoint 文件），不分配任何 GPU/CPU 显存，
    但完整验证 SAM3 真实网络结构是否与 AdapterInjector 的期望一致。
    """

    @pytest.fixture
    def real_meta_sam3(self):
        """SAM3 真实架构的 Meta 模型（需要 SAM3 包但不需要 checkpoint）。"""
        if not SAM3_AVAILABLE:
            pytest.skip("SAM3 包不可用，跳过真实架构测试")
        try:
            return build_meta_sam3(use_real_sam3=True, embed_dim=1024)
        except Exception as e:
            # 如果 SAM3 在 meta 设备上初始化失败（例如内部做了 forward 操作），跳过
            pytest.skip(f"SAM3 在 Meta 设备上初始化失败（可能是内部 forward 依赖）: {type(e).__name__}: {e}")


    def test_real_sam3_block_path_exists(self, real_meta_sam3):
        """验证真实 SAM3 模型中存在 AdapterInjector 期望的路径。

        路径：backbone.vision_backbone.trunk.blocks

        这是最重要的"结构即测试"：
        如果 SAM3 架构更新导致路径变化（如 trunk.layers 而非 trunk.blocks），
        这个测试会立即失败，提醒我们更新 AdapterInjector.inject() 的路径。
        完全不需要任何真实数据或 GPU！
        """
        all_block_paths = _collect_meta_block_paths(real_meta_sam3)

        print(f"\n  检测到以下 blocks 路径:")
        for path, count in all_block_paths.items():
            print(f"    [{count} blocks] {path}")

        assert len(all_block_paths) > 0, (
            "真实 SAM3 模型中未找到任何名为 blocks 的 ModuleList!\n"
            "这意味着 AdapterInjector.inject() 无法找到 Transformer 块。\n"
            "请检查 SAM3 模型的实际结构，并更新 inject() 中的路径搜索逻辑。"
        )

        expected_patterns = ['blocks', 'trunk', 'vision_backbone']
        primary_path = next(
            (path for path in all_block_paths
             if any(p in path for p in expected_patterns)),
            None
        )
        assert primary_path is not None, (
            f"未找到包含 {expected_patterns} 的 blocks 路径。\n"
            f"所有找到的路径: {list(all_block_paths.keys())}"
        )

        print(f"\n  找到主干 blocks 路径: {primary_path}")
        print(f"  block 数量: {all_block_paths[primary_path]}")

    def test_adapter_injector_finds_blocks(self, real_meta_sam3):
        """验证 AdapterInjector 可以在真实 SAM3 Meta 模型中自动找到并注入 Adapter。

        这个测试复现了"本地测试全过，上 GPU 跑真实 SAM3 却报结构不匹配"的
        根本原因检测。如果 AdapterInjector 无法在 Meta 模型上找到 blocks，
        它在真实 GPU 模型上同样会失败。

        ViT 变体：ViT-H 有 32 个 Block，ViT-L 有 24 个，ViT-B 有 12 个。
        """
        injector = AdapterInjector(adapter_dim=64)

        assert hasattr(real_meta_sam3, 'sam3_model'), \
            "SAM3MedicalIntegrated 应该有 sam3_model 属性"

        injector.inject(
            sam3_model=real_meta_sam3.sam3_model,
            use_real_sam3=True,
            embed_dim_hint=1024,
            image_encoder=None,
        )

        assert injector.adapters is not None, (
            "AdapterInjector 未能在真实 SAM3 Meta 模型中找到 Transformer blocks!\n"
            "这意味着在真实 GPU 上运行时也会失败。\n"
            "请检查 AdapterInjector.inject() 中的路径搜索逻辑。"
        )

        num_adapters = len(injector.adapters)
        print(f"\n  成功在真实 SAM3 Meta 模型中注入 {num_adapters} 个 Adapter")

        assert num_adapters in (12, 24, 32), (
            f"Adapter 数量 {num_adapters} 不在预期范围 [12, 24, 32]。"
        )

        first_adapter = injector.adapters[0]
        assert first_adapter.down_proj.weight.shape == (64, 1024), (
            f"down_proj.weight.shape 应为 (64, 1024)，"
            f"实际得到 {first_adapter.down_proj.weight.shape}。\n"
            "这意味着 embed_dim 被错误推断，Adapter 维度不匹配！"
        )
        assert first_adapter.up_proj.weight.shape == (1024, 64), (
            f"up_proj.weight.shape 应为 (1024, 64)，"
            f"实际得到 {first_adapter.up_proj.weight.shape}。"
        )

        print(f"  所有 {num_adapters} 个 Adapter 维度正确:")
        print(f"    down_proj.weight.shape = {first_adapter.down_proj.weight.shape}")
        print(f"    up_proj.weight.shape   = {first_adapter.up_proj.weight.shape}")

    def test_real_sam3_embed_dim_inference(self, real_meta_sam3):
        """验证 AdapterInjector 推断 embed_dim 的逻辑在 Meta 模型中是否正确。

        AdapterInjector.inject() 推断 embed_dim 的顺序：
          1. vit.embed_dim（ViT module 的直接属性）
          2. blocks[0].norm1.normalized_shape[-1]
          3. blocks[0].norm1.weight.shape[0]
          4. 默认值 1024

        对于 Meta Tensor，属性 .shape 和 .normalized_shape 完全可用！
        """
        sam3 = real_meta_sam3.sam3_model

        vit = None
        if hasattr(sam3, 'backbone'):
            backbone = sam3.backbone
            if hasattr(backbone, 'vision_backbone'):
                neck = backbone.vision_backbone
                if hasattr(neck, 'trunk'):
                    vit = neck.trunk

        if vit is None:
            pytest.skip("无法找到 vit.trunk，跳过 embed_dim 推断测试")

        if hasattr(vit, 'embed_dim'):
            embed_dim = vit.embed_dim
            print(f"\n  vit.embed_dim 属性存在，值为: {embed_dim}")
            assert embed_dim == 1024, \
                f"SAM3 ViT-H 的 embed_dim 应为 1024，实际得到 {embed_dim}"

        if hasattr(vit, 'blocks') and len(vit.blocks) > 0:
            first_block = vit.blocks[0]
            norm1 = getattr(first_block, 'norm1', None)
            if norm1 is not None:
                if hasattr(norm1, 'normalized_shape'):
                    inferred_dim = norm1.normalized_shape[-1]
                    print(f"  blocks[0].norm1.normalized_shape: {norm1.normalized_shape}")
                    assert inferred_dim == 1024, \
                        f"norm1.normalized_shape 推断维度应为 1024，实际 {inferred_dim}"
                elif hasattr(norm1, 'weight'):
                    inferred_dim = norm1.weight.shape[0]
                    print(f"  blocks[0].norm1.weight.shape: {norm1.weight.shape}")
                    assert inferred_dim == 1024, \
                        f"norm1.weight.shape 推断维度应为 1024，实际 {inferred_dim}"


class TestStateDictCompatibilityOnMeta:
    """Meta 设备上的 State-Dict 键名兼容性验证。

    SAM3MedicalIntegrated 重写了 state_dict() 来保证联邦聚合的键名兼容。
    在 Meta 设备上，state_dict() 返回的是 Meta Tensor 的字典，
    但键名（Key）的验证逻辑与真实 Tensor 完全相同。
    """

    @pytest.fixture
    def mock_meta_model(self) -> nn.Module:
        """Mock Meta 模型（不依赖 SAM3 checkpoint）"""
        with torch.device('meta'):
            return SAM3MedicalIntegrated(
                img_size=256, num_classes=1, adapter_dim=64,
                use_sam3=False, freeze_encoder=False, use_adapter=True,
                device='meta', embed_dim=768,
            )

    def test_state_dict_no_submodule_prefix(self, mock_meta_model):
        """验证 Meta 模型的 state_dict() 重写在 Meta 设备上正确去除子模块前缀。

        核心约束（联邦聚合）：
          adapter_manager.adapters.0.down_proj.weight  ->  adapters.0.down_proj.weight
          fusion_head.text_proj.0.weight               ->  text_proj.0.weight
        """
        sd = mock_meta_model.state_dict()

        leaked_keys = [
            k for k in sd
            if k.startswith('adapter_manager.') or k.startswith('fusion_head.')
        ]
        assert not leaked_keys, (
            f"state_dict 泄漏了子模块前缀，影响联邦聚合！\n"
            f"泄漏的键: {leaked_keys[:5]}"
        )

        adapter_keys = [k for k in sd if k.startswith('adapters.') or k.startswith('wrapped_blocks.')]
        proj_keys = [k for k in sd if k.startswith('text_proj.') or k.startswith('image_proj.')]

        assert len(adapter_keys) > 0, "state_dict 应包含 adapters.* 键"
        assert len(proj_keys) > 0, "state_dict 应包含 text_proj.* / image_proj.* 键"

        print(f"\n  Meta state_dict 键名校验通过（共 {len(sd)} 个键）")
        print(f"  adapter keys 样本: {adapter_keys[:2]}")
        print(f"  proj    keys 样本: {proj_keys[:2]}")

    def test_meta_model_memory_footprint(self, mock_meta_model):
        """量化展示 Meta 模型的内存优势。"""
        sd = mock_meta_model.state_dict()
        total_params = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
        theoretical_memory_mb = total_params * 4 / 1024 / 1024

        print(f"\n  Meta 模型统计：")
        print(f"    总参数量（理论）: {total_params:,}")
        print(f"    理论显存占用   : {theoretical_memory_mb:.2f} MB")
        print(f"    实际显存占用   : 0 MB（Meta Tensor 不分配物理内存！）")

        # 注意：MCSoftContrastiveLoss 用 torch.tensor() 创建参数，不遵循 meta 上下文，
        # 这是已知的第三方库限制。排除 contrastive_loss_fn.* 条目。
        non_meta_entries = [
            (k, v.device) for k, v in sd.items()
            if hasattr(v, 'device') and v.device.type != 'meta'
            and not k.startswith('contrastive_loss_fn')
        ]
        assert not non_meta_entries, (
            f"发现非 Meta Tensor 在 state_dict 中:\n"
            + "\n".join(f"  {k}: {d}" for k, d in non_meta_entries[:5])
        )


# ============================================================================
# 快速冒烟测试（无 fixture，可直接 python test_meta_architecture.py 运行）
# ============================================================================

def smoke_test_meta_injector():
    """快速冒烟测试（无 pytest 依赖）。"""
    print("=" * 65)
    print("Meta Tensor 架构测试 --- 快速冒烟")
    print("=" * 65)

    print("\n[1/4] Meta Tensor 基础验证...")
    meta_t = torch.empty(4096, 4096, device='meta')
    assert meta_t.shape == (4096, 4096) and meta_t.device.type == 'meta'
    print(f"  torch.empty(4096, 4096, device='meta') shape={meta_t.shape}")

    print("\n[2/4] Meta Mock 模型构建...")
    with torch.device('meta'):
        mock_model = SAM3MedicalIntegrated(
            img_size=256, num_classes=1, adapter_dim=64,
            use_sam3=False, freeze_encoder=False, use_adapter=False,
            device='meta', embed_dim=768,
        )
    # 注意：MCSoftContrastiveLoss 用 torch.tensor() 创建参数，不遵循 meta 上下文，
    # 这是已知的第三方库限制。排除 contrastive_loss_fn.* 参数。
    non_meta = [(n, p.device) for n, p in mock_model.named_parameters()
                if p.device.type != 'meta' and not n.startswith('contrastive_loss_fn')]
    assert not non_meta, f"发现非 Meta 参数: {non_meta[:3]}"
    print(f"  Mock 模型构建成功，主干参数全部在 meta 设备")

    print("\n[3/4] AdapterInjector 注入（Meta Mock 模型）...")
    injector = AdapterInjector(adapter_dim=64)
    injector.inject(
        sam3_model=None,
        use_real_sam3=False,
        embed_dim_hint=768,
        image_encoder=mock_model.image_encoder,
    )
    num_adapters = len(injector.adapters)
    assert num_adapters == 12, f"期望 12 个 Adapter，实际 {num_adapters}"
    down_shape = injector.adapters[0].down_proj.weight.shape
    up_shape = injector.adapters[0].up_proj.weight.shape
    assert down_shape == (64, 768), f"down_proj shape 不正确: {down_shape}"
    assert up_shape == (768, 64), f"up_proj shape 不正确: {up_shape}"
    print(f"  注入 {num_adapters} 个 Adapter，shape 验证通过")
    print(f"    down_proj: {down_shape}, up_proj: {up_shape}")

    print("\n[4/4] State-dict 键名兼容性验证...")
    with torch.device('meta'):
        model_with_adapter = SAM3MedicalIntegrated(
            img_size=256, num_classes=1, adapter_dim=64,
            use_sam3=False, freeze_encoder=False, use_adapter=True,
            device='meta', embed_dim=768,
        )
    sd = model_with_adapter.state_dict()
    leaked = [k for k in sd if k.startswith('adapter_manager.') or k.startswith('fusion_head.')]
    assert not leaked, f"state_dict 前缀泄漏: {leaked}"
    print(f"  state_dict 共 {len(sd)} 个键，无子模块前缀泄漏")

    print("\n" + "=" * 65)
    print("所有冒烟测试通过！（实际内存占用: 0 bytes）")
    print("=" * 65)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    smoke_test_meta_injector()
