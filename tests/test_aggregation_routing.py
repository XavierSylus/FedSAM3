"""
tests/test_aggregation_routing.py
===================================
非对称解耦聚合路由器单元测试（ADA-Router v2）

验证目标：
  - text_only 客户端绝对无法参与 IMAGE_PARAMS / VISION_ADAPTER 聚合
  - image_only 客户端绝对无法参与 TEXT_PARAMS / TEXT_ADAPTER 聚合
  - image_proj 归视觉侧（text_only 被拒绝）
  - text_adapter 仅 text_only + multimodal 参与
  - adapters.* 仅 image_only + multimodal 参与
  - safety guard：无合格客户端时返回空列表（不崩溃）

运行方式：
  pytest tests/test_aggregation_routing.py -v
  python  tests/test_aggregation_routing.py
"""

import sys
import torch
import copy
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SAM3_Medical, DEVICE
from src.parameter_groups import (
    FUSION_PARAMS,
    IMAGE_PARAMS,
    TEXT_ADAPTER,
    TEXT_PARAMS,
    VISION_ADAPTER,
    classify_parameter,
)
from src.server import CreamAggregator


# ============================================================================
# 辅助函数
# ============================================================================

def _build_dummy_state(param_keys: list) -> dict:
    """产生 dummy state_dict，每个键填一个随机张量（形状 (2,2)）。"""
    return {k: torch.randn(2, 2) for k in param_keys}


def _router(
    global_model,
    param_name: str,
    client_dicts: list,
    modalities: list
) -> list:
    """调用路由器的便捷包装，返回参与下标列表。"""
    aggregator = CreamAggregator(
        global_model=global_model,
        device='cpu',
        aggregation_method='fedavg',
    )
    return aggregator._get_participating_clients_dynamic(
        param_name=param_name,
        client_weights_list=client_dicts,
        client_modalities=modalities,
    )


def test_parameter_group_classifier_matches_router_contract():
    assert classify_parameter("text_adapter.down.weight") == TEXT_ADAPTER
    assert classify_parameter("adapters.0.down.weight") == VISION_ADAPTER
    assert classify_parameter("fusion_head.text_proj.weight") == TEXT_PARAMS
    assert classify_parameter("fusion_head._text_projection.weight") == FUSION_PARAMS
    assert classify_parameter("fusion_head._fusion_gate.0.weight") == FUSION_PARAMS
    assert classify_parameter("medical_seg_head.weight") == IMAGE_PARAMS
    with pytest.raises(ValueError, match="Unclassified"):
        classify_parameter("unknown_module.weight")


# ============================================================================
# 测试套件
# ============================================================================

class TestADARouterV2:
    """
    ADA-Router v2 核心路由断言。
    三种客户端：
      client_0 = text_only
      client_1 = image_only
      client_2 = multimodal
    """

    def setup_method(self):
        """构造极小号 SAM3_Medical 用作 global_model（img_size=64 以节省内存）。"""
        # img_size 缩小到 64 以加速测试
        self.global_model = SAM3_Medical(img_size=64, embed_dim=64, num_heads=4)
        self.modalities = ['text_only', 'image_only', 'multimodal']

        # 构造三种客户端的 dummy state_dict
        # 每个客户端只需包含「当前要测试的」参数键即可
        self.base_keys_per_client = [
            # client_0 (text_only) 上传的键
            [
                'text_encoder.layer.0.weight',
                'text_proj.0.weight',
                'text_adapter.down_proj.weight',
            ],
            # client_1 (image_only) 上传的键
            [
                'image_encoder.patch_embed.weight',
                'mask_decoder.proj.weight',
                'adapters.0.down_proj.weight',
                'image_proj.0.weight',
                '_output_conv.weight',
            ],
            # client_2 (multimodal) 上传的键
            [
                'text_encoder.layer.0.weight',
                'text_proj.0.weight',
                'text_adapter.down_proj.weight',
                'image_encoder.patch_embed.weight',
                'mask_decoder.proj.weight',
                'adapters.0.down_proj.weight',
                'image_proj.0.weight',
                '_output_conv.weight',
                'wrapped_blocks.0.adapter.down_proj.weight',
            ],
        ]

    def _make_dicts(self, target_param: str) -> list:
        """为每个客户端构造 state_dict，确保 target_param 出现在对应客户端的 dict 中。"""
        dicts = []
        for keys in self.base_keys_per_client:
            all_keys = list(keys)
            if target_param not in all_keys:
                # 只有实际拥有该参数的客户端才上传它
                # 此处为了测试路由逻辑，确保三方都"上传"该参数
                all_keys.append(target_param)
            dicts.append(_build_dummy_state(all_keys))
        return dicts

    def _route(self, param_name: str) -> set:
        """返回参与聚合的客户端模态集合。"""
        dicts = self._make_dicts(param_name)
        indices = _router(self.global_model, param_name, dicts, self.modalities)
        return {self.modalities[i] for i in indices}

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 1：视觉参数物理隔离
    # ─────────────────────────────────────────────────────────────────────────

    def test_image_encoder_excludes_text_only(self):
        """image_encoder.* 的聚合绝对不包含 text_only 客户端。"""
        participants = self._route('image_encoder.patch_embed.weight')
        assert 'text_only' not in participants, (
            f"[FAIL] image_encoder 聚合池意外出现 text_only！实际: {participants}"
        )
        assert 'image_only' in participants and 'multimodal' in participants, (
            f"[FAIL] image_encoder 聚合池缺少视觉侧客户端！实际: {participants}"
        )
        print(f"  ✓ image_encoder 聚合池: {participants}（text_only 已物理隔离）")

    def test_mask_decoder_excludes_text_only(self):
        """mask_decoder.* 的聚合绝对不包含 text_only 客户端。"""
        participants = self._route('mask_decoder.proj.weight')
        assert 'text_only' not in participants
        print(f"  ✓ mask_decoder 聚合池: {participants}")

    def test_backbone_excludes_text_only(self):
        """backbone.* 的聚合绝对不包含 text_only 客户端。"""
        participants = self._route('backbone.layer.0.weight')
        assert 'text_only' not in participants
        print(f"  ✓ backbone 聚合池: {participants}")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 2：image_proj 归视觉侧（关键修改验证）
    # ─────────────────────────────────────────────────────────────────────────

    def test_image_proj_excludes_text_only(self):
        """
        ★ 核心审阅意见验证：image_proj 已从全员改为视觉侧专属。
        text_only 客户端无图像输入，其 image_proj 梯度为 0/噪声，
        必须被拒绝参与聚合。
        """
        participants = self._route('image_proj.0.weight')
        assert 'text_only' not in participants, (
            f"[FAIL] image_proj 仍允许 text_only！路由白名单未正确更新！"
            f"\n  实际参与: {participants}"
        )
        assert 'image_only' in participants or 'multimodal' in participants, (
            f"[FAIL] image_proj 没有任何视觉侧客户端！实际: {participants}"
        )
        print(f"  ✓ image_proj 聚合池: {participants}（审阅意见已落地）")

    def test_output_conv_excludes_text_only(self):
        """_output_conv.* 归视觉侧，text_only 不参与。"""
        participants = self._route('_output_conv.weight')
        assert 'text_only' not in participants
        print(f"  ✓ _output_conv 聚合池: {participants}")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 3：视觉 Adapter 物理隔离（关键修改验证）
    # ─────────────────────────────────────────────────────────────────────────

    def test_adapters_excludes_text_only(self):
        """
        ★ 核心修改验证：adapters.* 从全员改为视觉侧专属。
        text_only 客户端不得参与视觉 Adapter 聚合。
        """
        participants = self._route('adapters.0.down_proj.weight')
        assert 'text_only' not in participants, (
            f"[FAIL] adapters.* 仍允许 text_only！VISION_ADAPTER 路由失效！"
            f"\n  实际参与: {participants}"
        )
        print(f"  ✓ adapters.0 聚合池: {participants}（text_only 已被截断）")

    def test_wrapped_blocks_excludes_text_only(self):
        """wrapped_blocks.* (BlockWithAdapter) 归视觉侧，text_only 不参与。"""
        participants = self._route('wrapped_blocks.0.adapter.down_proj.weight')
        assert 'text_only' not in participants
        print(f"  ✓ wrapped_blocks 聚合池: {participants}")

    def test_lora_excludes_text_only(self):
        """lora.* 归视觉侧，text_only 不参与。"""
        participants = self._route('lora.0.lora_A.weight')
        assert 'text_only' not in participants
        print(f"  ✓ lora 聚合池: {participants}")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 4：文本参数物理隔离
    # ─────────────────────────────────────────────────────────────────────────

    def test_text_encoder_excludes_image_only(self):
        """text_encoder.* 的聚合绝对不包含 image_only 客户端。"""
        participants = self._route('text_encoder.layer.0.weight')
        assert 'image_only' not in participants, (
            f"[FAIL] text_encoder 聚合池意外出现 image_only！实际: {participants}"
        )
        assert 'text_only' in participants and 'multimodal' in participants
        print(f"  ✓ text_encoder 聚合池: {participants}")

    def test_text_proj_excludes_image_only(self):
        """text_proj.* 的聚合绝对不包含 image_only 客户端。"""
        participants = self._route('text_proj.0.weight')
        assert 'image_only' not in participants
        print(f"  ✓ text_proj 聚合池: {participants}")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 5：text_adapter 专属池（优先级 1）
    # ─────────────────────────────────────────────────────────────────────────

    def test_text_adapter_excludes_image_only(self):
        """
        text_adapter.* 是优先级最高的单独路由：
        仅 text_only + multimodal 参与，image_only 全程禁入。
        """
        participants = self._route('text_adapter.down_proj.weight')
        assert 'image_only' not in participants, (
            f"[FAIL] text_adapter 聚合池出现 image_only！实际: {participants}"
        )
        assert 'text_only' in participants, (
            f"[FAIL] text_adapter 聚合池缺少 text_only！实际: {participants}"
        )
        print(f"  ✓ text_adapter 聚合池: {participants}（image_only 已物理截断）")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 6：优先级保证——text_adapter 不会落入 VISION_ADAPTER
    # ─────────────────────────────────────────────────────────────────────────

    def test_text_adapter_priority_over_any_adapter_keyword(self):
        """
        'text_adapter' 命中 TEXT_ADAPTER_PARAMS（优先级 1），
        虽然 'adapter' 字面上也是 VISION_ADAPTER 的子串，
        但由于优先级 1 先命中，必须路由到 text_only+multimodal，
        而非 image_only+multimodal。
        """
        participants = self._route('text_adapter.up_proj.weight')
        assert 'image_only' not in participants, (
            f"[FAIL] text_adapter 错误地被路由到了 VISION_ADAPTER 池！\n"
            f"  实际: {participants}（应为 text_only + multimodal）"
        )
        assert 'text_only' in participants
        print(f"  ✓ 优先级保证：text_adapter → TEXT_ADAPTER 池 ({participants})")

    def test_fusion_parameters_only_accept_multimodal_client(self):
        participants = self._route("fusion_head._fusion_gate.0.weight")
        assert participants == {"multimodal"}

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 7：安全守卫——无合格客户端返回空列表
    # ─────────────────────────────────────────────────────────────────────────

    def test_safety_guard_returns_empty_list(self):
        """
        当上传者存在但全被模态过滤时，路由器必须返回空列表，
        而不是抛出异常或返回所有人。
        """
        # 只让 text_only 客户端上传 image_encoder 参数（不合法情形）
        dicts = [
            _build_dummy_state(['image_encoder.patch_embed.weight']),  # text_only 上传
            None,  # image_only 本轮未参与
            None,  # multimodal 本轮未参与
        ]
        indices = _router(
            self.global_model,
            'image_encoder.patch_embed.weight',
            dicts,
            self.modalities
        )
        assert indices == [], (
            f"[FAIL] 安全守卫失效！应返回 []，实际返回: {indices}"
        )
        print(f"  ✓ 安全守卫触发，返回空列表（不回填全局权重）")

    # ─────────────────────────────────────────────────────────────────────────
    # 测试 8：无模态信息时向后兼容
    # ─────────────────────────────────────────────────────────────────────────

    def test_no_modality_info_returns_all_uploaders(self):
        """Unrestricted routing returns all uploaders after parameter classification."""
        dicts = [
            _build_dummy_state(['image_encoder.weight']),
            _build_dummy_state(['image_encoder.weight']),
            None,
        ]
        indices = _router(
            self.global_model,
            'image_encoder.weight',
            dicts,
            modalities=None  # 无模态信息
        )
        assert set(indices) == {0, 1}, (
            f"[FAIL] 无模态信息时应返回 {{0, 1}}，实际: {indices}"
        )
        print(f"  ✓ unrestricted routing：返回所有上传者 {indices}")

    def test_unclassified_parameter_fails_in_unrestricted_routing(self):
        dicts = [
            _build_dummy_state(["unknown_module.weight"]),
            None,
            None,
        ]
        with pytest.raises(ValueError, match="Unclassified"):
            _router(
                self.global_model,
                "unknown_module.weight",
                dicts,
                modalities=None,
            )


# ============================================================================
# 快速冒烟测试（无 pytest 依赖）
# ============================================================================

def smoke_test_routing():
    """在无 pytest 的环境下直接运行本文件进行快速冒烟验证。"""
    print("=" * 60)
    print("ADA-Router v2 路由冒烟测试")
    print("=" * 60)

    global_model = SAM3_Medical(img_size=64, embed_dim=64, num_heads=4)
    modalities = ['text_only', 'image_only', 'multimodal']

    def make_all_dicts(param_key):
        return [
            _build_dummy_state([param_key]),
            _build_dummy_state([param_key]),
            _build_dummy_state([param_key]),
        ]

    test_cases = [
        # (param_name, should_include, should_exclude)
        ('image_encoder.patch_embed.weight', {'image_only', 'multimodal'}, {'text_only'}),
        ('mask_decoder.proj.weight',         {'image_only', 'multimodal'}, {'text_only'}),
        ('adapters.0.down_proj.weight',      {'image_only', 'multimodal'}, {'text_only'}),
        ('wrapped_blocks.0.adapter.w',       {'image_only', 'multimodal'}, {'text_only'}),
        ('image_proj.0.weight',              {'image_only', 'multimodal'}, {'text_only'}),
        ('text_encoder.layer.0.weight',      {'text_only', 'multimodal'}, {'image_only'}),
        ('text_proj.0.weight',               {'text_only', 'multimodal'}, {'image_only'}),
        ('text_adapter.down_proj.weight',    {'text_only', 'multimodal'}, {'image_only'}),
    ]

    passed = 0
    failed = 0
    for param, must_include, must_exclude in test_cases:
        dicts = make_all_dicts(param)
        indices = _router(global_model, param, dicts, modalities)
        actual = {modalities[i] for i in indices}
        ok_include = must_include.issubset(actual)
        ok_exclude = must_exclude.isdisjoint(actual)
        if ok_include and ok_exclude:
            print(f"  ✓ {param[:45]:<45}  pool={actual}")
            passed += 1
        else:
            print(f"  ✗ {param[:45]:<45}  pool={actual}")
            if not ok_include:
                print(f"      ↳ 缺少: {must_include - actual}")
            if not ok_exclude:
                print(f"      ↳ 非法混入: {must_exclude & actual}")
            failed += 1

    print()
    print(f"结果: {passed} 通过 / {failed} 失败")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    smoke_test_routing()
