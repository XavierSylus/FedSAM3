"""
tests/test_ema_decoupled_update.py
====================================
EMA 解耦更新单元测试

验证目标：
  - image_only 客户端的表征不影响 global_text_rep
  - text_only 客户端的表征不影响 global_image_rep
  - multimodal 客户端的表征同时更新两个全局表征
  - EMA 系数 (alpha) 按预期工作
  - 更新后表征的 L2 范数为 1（normalize 效果）
  - 维度不匹配时的自动对齐行为
  - 防污染机制：image_only 客户端上传的文本表征被拦截

运行方式：
  pytest tests/test_ema_decoupled_update.py -v
  python tests/test_ema_decoupled_update.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SAM3_Medical
from src.server import CreamAggregator


# ============================================================================
# 辅助函数
# ============================================================================

def _create_normalized_vector(dim: int, seed: Optional[int] = None) -> torch.Tensor:
    """创建一个归一化的随机向量（L2 范数 = 1）。"""
    if seed is not None:
        torch.manual_seed(seed)
    vec = torch.randn(dim)
    return nn.functional.normalize(vec, p=2, dim=0)


def _get_l2_norm(tensor: torch.Tensor) -> float:
    """计算张量的 L2 范数。"""
    return torch.norm(tensor, p=2).item()


# ============================================================================
# 测试套件
# ============================================================================

class TestEMADecoupledUpdate:
    """
    EMA 解耦更新单元测试。

    三种客户端模态：
      - text_only:  仅更新 global_text_rep
      - image_only: 仅更新 global_image_rep
      - multimodal: 同时更新两者
    """

    def setup_method(self):
        """构造测试环境：极小号 SAM3_Medical 和 CreamAggregator。"""
        # 使用极小配置加速测试（contrastive_dim = embed_dim = 64）
        self.global_model = SAM3_Medical(img_size=64, embed_dim=64, num_heads=4)
        self.aggregator = CreamAggregator(
            global_model=self.global_model,
            device='cpu',
            aggregation_method='fedavg',
            global_rep_alpha=0.9  # 显式设置 EMA 系数
        )

        # 表征维度（与 SAM3_Medical 的 contrastive_dim 一致）
        # 对于测试用的小模型：contrastive_dim = embed_dim = 64
        self.image_dim = self.aggregator.global_image_rep.shape[0]
        self.text_dim = self.aggregator.global_text_rep.shape[0]

        # 保存初始全局表征（用于验证是否被更新）
        self.initial_image_rep = self.aggregator.global_image_rep.clone()
        self.initial_text_rep = self.aggregator.global_text_rep.clone()

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 1：模态隔离 - image_only 客户端不影响 global_text_rep
    # ─────────────────────────────────────────────────────────────────────────

    def test_image_only_does_not_update_global_text_rep(self):
        """
        image_only 客户端上传图像表征时，global_text_rep 应保持不变。
        """
        # 准备 image_only 客户端表征（仅图像，无文本）
        image_rep = _create_normalized_vector(self.image_dim, seed=42)

        # 记录更新前的 global_text_rep
        text_rep_before = self.aggregator.global_text_rep.clone()

        # 直接调用解耦更新方法（image_reps 有值，text_reps 为空）
        self.aggregator._update_global_reps_decoupled(
            image_reps=[image_rep],
            text_reps=[]
        )

        # 验证 global_text_rep 未被更新
        text_rep_after = self.aggregator.global_text_rep
        assert torch.allclose(text_rep_before, text_rep_after, atol=1e-6), (
            f"[FAIL] image_only 客户端意外更新了 global_text_rep！\n"
            f"  更新前: {text_rep_before[:5].tolist()}\n"
            f"  更新后: {text_rep_after[:5].tolist()}"
        )

        # 验证 global_image_rep 被更新
        image_rep_after = self.aggregator.global_image_rep
        assert not torch.allclose(self.initial_image_rep, image_rep_after, atol=1e-6), (
            f"[FAIL] global_image_rep 未被更新！"
        )

        print(f"  ✓ image_only 客户端物理隔离验证通过：global_text_rep 未被污染")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 2：模态隔离 - text_only 客户端不影响 global_image_rep
    # ─────────────────────────────────────────────────────────────────────────

    def test_text_only_does_not_update_global_image_rep(self):
        """
        text_only 客户端上传文本表征时，global_image_rep 应保持不变。
        """
        # 准备 text_only 客户端表征（仅文本，无图像）
        text_rep = _create_normalized_vector(self.text_dim, seed=43)

        # 记录更新前的 global_image_rep
        image_rep_before = self.aggregator.global_image_rep.clone()

        # 直接调用解耦更新方法（text_reps 有值，image_reps 为空）
        self.aggregator._update_global_reps_decoupled(
            image_reps=[],
            text_reps=[text_rep]
        )

        # 验证 global_image_rep 未被更新
        image_rep_after = self.aggregator.global_image_rep
        assert torch.allclose(image_rep_before, image_rep_after, atol=1e-6), (
            f"[FAIL] text_only 客户端意外更新了 global_image_rep！\n"
            f"  更新前: {image_rep_before[:5].tolist()}\n"
            f"  更新后: {image_rep_after[:5].tolist()}"
        )

        # 验证 global_text_rep 被更新
        text_rep_after = self.aggregator.global_text_rep
        assert not torch.allclose(self.initial_text_rep, text_rep_after, atol=1e-6), (
            f"[FAIL] global_text_rep 未被更新！"
        )

        print(f"  ✓ text_only 客户端物理隔离验证通过：global_image_rep 未被污染")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 3：multimodal 客户端同时更新两个全局表征
    # ─────────────────────────────────────────────────────────────────────────

    def test_multimodal_updates_both_reps(self):
        """
        multimodal 客户端上传图像和文本表征时，两个全局表征应同时被更新。
        """
        # 准备 multimodal 客户端表征
        image_rep = _create_normalized_vector(self.image_dim, seed=44)
        text_rep = _create_normalized_vector(self.text_dim, seed=45)

        # 直接调用解耦更新方法（两者都有值）
        self.aggregator._update_global_reps_decoupled(
            image_reps=[image_rep],
            text_reps=[text_rep]
        )

        # 验证两个全局表征都被更新
        assert not torch.allclose(self.initial_image_rep, self.aggregator.global_image_rep, atol=1e-6), (
            f"[FAIL] multimodal 客户端未更新 global_image_rep！"
        )
        assert not torch.allclose(self.initial_text_rep, self.aggregator.global_text_rep, atol=1e-6), (
            f"[FAIL] multimodal 客户端未更新 global_text_rep！"
        )

        print(f"  ✓ multimodal 客户端同时更新两个全局表征验证通过")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 4：EMA 系数验证（alpha = 0.9）
    # ─────────────────────────────────────────────────────────────────────────

    def test_ema_alpha_coefficient(self):
        """
        验证 EMA 更新公式：new_rep = normalize(alpha * old_rep + (1 - alpha) * new_input)
        其中 alpha = 0.9。
        """
        alpha = 0.9

        # 创建可控的初始表征和新输入
        initial_rep = torch.ones(self.image_dim) / (self.image_dim ** 0.5)  # 归一化向量
        new_input = torch.zeros(self.image_dim)
        new_input[0] = 1.0  # [1, 0, 0, ...]

        # 手动设置初始全局表征
        self.aggregator.global_image_rep = initial_rep.clone()

        # 调用解耦更新
        self.aggregator._update_global_reps_decoupled(
            image_reps=[new_input],
            text_reps=[]
        )

        # 计算预期结果（手动实现 EMA 公式）
        expected = nn.functional.normalize(
            alpha * initial_rep + (1 - alpha) * new_input,
            p=2, dim=0
        )

        # 验证实际结果与预期一致
        actual = self.aggregator.global_image_rep
        assert torch.allclose(expected, actual, atol=1e-5), (
            f"[FAIL] EMA 更新公式不正确！\n"
            f"  预期: {expected[:5].tolist()}\n"
            f"  实际: {actual[:5].tolist()}"
        )

        print(f"  ✓ EMA 系数 (alpha=0.9) 验证通过")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 5：归一化效果验证（L2 范数 = 1）
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalization_l2_norm_equals_one(self):
        """
        验证 EMA 更新后的表征 L2 范数为 1。
        """
        # 准备非归一化的向量
        unnormalized_image_rep = torch.randn(self.image_dim) * 10  # 故意放大
        unnormalized_text_rep = torch.randn(self.text_dim) * 10

        # 直接调用解耦更新方法
        self.aggregator._update_global_reps_decoupled(
            image_reps=[unnormalized_image_rep],
            text_reps=[unnormalized_text_rep]
        )

        # 验证归一化效果
        image_norm = _get_l2_norm(self.aggregator.global_image_rep)
        text_norm = _get_l2_norm(self.aggregator.global_text_rep)

        assert abs(image_norm - 1.0) < 1e-5, (
            f"[FAIL] global_image_rep 的 L2 范数不为 1！实际: {image_norm}"
        )
        assert abs(text_norm - 1.0) < 1e-5, (
            f"[FAIL] global_text_rep 的 L2 范数不为 1！实际: {text_norm}"
        )

        print(f"  ✓ 归一化验证通过：L2 范数 = 1.0 (image={image_norm:.6f}, text={text_norm:.6f})")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 6：维度不匹配时的自动对齐（padding）
    # ─────────────────────────────────────────────────────────────────────────

    def test_dimension_mismatch_padding(self):
        """
        验证当客户端上传的表征维度小于全局表征时，自动 padding 到目标维度。
        """
        # 客户端上传维度不足的表征（一半维度）
        small_dim = self.image_dim // 2
        small_image_rep = _create_normalized_vector(small_dim, seed=46)

        # 直接调用解耦更新方法
        self.aggregator._update_global_reps_decoupled(
            image_reps=[small_image_rep],
            text_reps=[]
        )

        # 验证全局表征维度未改变
        assert self.aggregator.global_image_rep.shape[0] == self.image_dim, (
            f"[FAIL] global_image_rep 维度被意外改变！"
            f"预期: {self.image_dim}, 实际: {self.aggregator.global_image_rep.shape[0]}"
        )

        # 验证表征已被更新（说明 padding 生效）
        assert not torch.allclose(self.initial_image_rep, self.aggregator.global_image_rep, atol=1e-6), (
            f"[FAIL] 维度不匹配时表征未被更新！"
        )

        # 验证归一化仍然有效
        norm = _get_l2_norm(self.aggregator.global_image_rep)
        assert abs(norm - 1.0) < 1e-5, (
            f"[FAIL] padding 后的表征未正确归一化！L2 范数: {norm}"
        )

        print(f"  ✓ 维度自动对齐（padding）验证通过：{small_dim} -> {self.image_dim}")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 7：维度不匹配时的自动对齐（truncation）
    # ─────────────────────────────────────────────────────────────────────────

    def test_dimension_mismatch_truncation(self):
        """
        验证当客户端上传的表征维度大于全局表征时，自动截断到目标维度。
        """
        # 客户端上传维度过大的表征（两倍维度）
        large_dim = self.text_dim * 2
        large_text_rep = _create_normalized_vector(large_dim, seed=47)

        # 直接调用解耦更新方法
        self.aggregator._update_global_reps_decoupled(
            image_reps=[],
            text_reps=[large_text_rep]
        )

        # 验证全局表征维度未改变
        assert self.aggregator.global_text_rep.shape[0] == self.text_dim, (
            f"[FAIL] global_text_rep 维度被意外改变！"
            f"预期: {self.text_dim}, 实际: {self.aggregator.global_text_rep.shape[0]}"
        )

        # 验证表征已被更新（说明 truncation 生效）
        assert not torch.allclose(self.initial_text_rep, self.aggregator.global_text_rep, atol=1e-6), (
            f"[FAIL] 维度不匹配时表征未被更新！"
        )

        # 验证归一化仍然有效
        norm = _get_l2_norm(self.aggregator.global_text_rep)
        assert abs(norm - 1.0) < 1e-5, (
            f"[FAIL] truncation 后的表征未正确归一化！L2 范数: {norm}"
        )

        print(f"  ✓ 维度自动对齐（truncation）验证通过：{large_dim} -> {self.text_dim}")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 8：防污染机制 - image_only 客户端上传的文本表征被拦截
    # ─────────────────────────────────────────────────────────────────────────

    def test_pollution_guard_blocks_image_only_text_rep(self):
        """
        验证防污染机制：image_only 客户端意外上传的文本表征应被拦截，
        不纳入全局文本表征更新池。

        注意：此测试通过 aggregate_heterogeneous_clients 进行，
        因为污染拦截逻辑在该方法中实现（server.py:503-510）。
        """
        # 准备 image_only 客户端意外上传文本表征的情况
        image_rep = _create_normalized_vector(self.image_dim, seed=48)
        malicious_text_rep = _create_normalized_vector(self.text_dim, seed=49)

        client_updates = [
            (None, image_rep, malicious_text_rep)  # image_only 不应上传 text_rep
        ]

        # 记录更新前的 global_text_rep
        text_rep_before = self.aggregator.global_text_rep.clone()

        # 执行聚合（防污染机制应拦截 text_rep）
        self.aggregator.aggregate_heterogeneous_clients(
            client_updates=client_updates,
            client_modalities=['image_only']
        )

        # 验证 global_text_rep 未被污染
        text_rep_after = self.aggregator.global_text_rep
        assert torch.allclose(text_rep_before, text_rep_after, atol=1e-6), (
            f"[FAIL] 防污染机制失效！image_only 客户端的文本表征污染了 global_text_rep！\n"
            f"  更新前: {text_rep_before[:5].tolist()}\n"
            f"  更新后: {text_rep_after[:5].tolist()}"
        )

        # 验证 global_image_rep 正常更新
        assert not torch.allclose(self.initial_image_rep, self.aggregator.global_image_rep, atol=1e-6), (
            f"[FAIL] global_image_rep 未被正常更新！"
        )

        print(f"  ✓ 防污染机制验证通过：image_only 客户端的文本表征被成功拦截")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 9：多客户端混合更新
    # ─────────────────────────────────────────────────────────────────────────

    def test_mixed_clients_update(self):
        """
        验证混合场景：text_only + image_only + multimodal 客户端同时参与时，
        解耦更新逻辑正确工作。
        """
        # 准备三种客户端的表征
        text_rep_1 = _create_normalized_vector(self.text_dim, seed=50)
        image_rep_2 = _create_normalized_vector(self.image_dim, seed=51)
        image_rep_3 = _create_normalized_vector(self.image_dim, seed=52)
        text_rep_3 = _create_normalized_vector(self.text_dim, seed=53)

        # 模拟混合场景：
        # - text_only 贡献 text_rep
        # - image_only 贡献 image_rep
        # - multimodal 贡献两者
        self.aggregator._update_global_reps_decoupled(
            image_reps=[image_rep_2, image_rep_3],  # from image_only + multimodal
            text_reps=[text_rep_1, text_rep_3]      # from text_only + multimodal
        )

        # 验证两个全局表征都被更新
        assert not torch.allclose(self.initial_image_rep, self.aggregator.global_image_rep, atol=1e-6), (
            f"[FAIL] 混合场景下 global_image_rep 未被更新！"
        )
        assert not torch.allclose(self.initial_text_rep, self.aggregator.global_text_rep, atol=1e-6), (
            f"[FAIL] 混合场景下 global_text_rep 未被更新！"
        )

        # 验证归一化
        image_norm = _get_l2_norm(self.aggregator.global_image_rep)
        text_norm = _get_l2_norm(self.aggregator.global_text_rep)
        assert abs(image_norm - 1.0) < 1e-5
        assert abs(text_norm - 1.0) < 1e-5

        print(f"  ✓ 混合客户端场景验证通过（text_only + image_only + multimodal）")


# ============================================================================
# 快速冒烟测试（无 pytest 依赖）
# ============================================================================

def smoke_test_ema():
    """在无 pytest 的环境下直接运行本文件进行快速冒烟验证。"""
    print("=" * 70)
    print("EMA 解耦更新冒烟测试")
    print("=" * 70)

    test_suite = TestEMADecoupledUpdate()

    test_cases = [
        ("模态隔离 - image_only 不影响 text_rep", test_suite.test_image_only_does_not_update_global_text_rep),
        ("模态隔离 - text_only 不影响 image_rep", test_suite.test_text_only_does_not_update_global_image_rep),
        ("multimodal 同时更新两个表征", test_suite.test_multimodal_updates_both_reps),
        ("EMA 系数验证 (alpha=0.9)", test_suite.test_ema_alpha_coefficient),
        ("归一化验证 (L2 norm=1)", test_suite.test_normalization_l2_norm_equals_one),
        ("维度对齐 - padding", test_suite.test_dimension_mismatch_padding),
        ("维度对齐 - truncation", test_suite.test_dimension_mismatch_truncation),
        ("防污染机制", test_suite.test_pollution_guard_blocks_image_only_text_rep),
        ("混合客户端场景", test_suite.test_mixed_clients_update),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in test_cases:
        # 每个测试前重新初始化环境
        test_suite.setup_method()

        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}")
            print(f"  错误: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} (异常)")
            print(f"  错误: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"结果: {passed} 通过 / {failed} 失败")
    print("=" * 70)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    smoke_test_ema()
