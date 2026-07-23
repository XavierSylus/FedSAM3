import pytest
import torch
import torch.nn as nn

from src.server import CreamAggregator


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.contrastive_dim = 8
        self.sam3_model = nn.Linear(2, 2, bias=False)  # vision-like key
        self.adapter_manager = nn.Linear(2, 2, bias=False)  # trainable key
        for p in self.sam3_model.parameters():
            p.requires_grad = False
        for p in self.adapter_manager.parameters():
            p.requires_grad = True

    def reset_rope_frequencies(self, verbose: bool = False):
        return 0


def test_strict_guard_skips_frozen_vision_keys_outside_expected_set():
    model = _DummyModel()
    agg = CreamAggregator(model, device="cpu", strict_aggregation_guard=True)

    partial_state = {
        "adapter_manager.weight": model.state_dict()["adapter_manager.weight"].clone()
    }
    out = agg._safe_fill_missing_params(
        partial_state,
        location_tag="test",
        expected_param_names={"adapter_manager.weight"},
    )

    assert "sam3_model.weight" in out
    assert agg._last_aggregation_audit["vision_missing"] == 0


def test_strict_guard_raises_when_expected_vision_key_missing():
    model = _DummyModel()
    agg = CreamAggregator(model, device="cpu", strict_aggregation_guard=True)

    partial_state = {
        "adapter_manager.weight": model.state_dict()["adapter_manager.weight"].clone()
    }
    with pytest.raises(RuntimeError, match="strict_aggregation_guard=True"):
        agg._safe_fill_missing_params(
            partial_state,
            location_tag="test",
            expected_param_names={"adapter_manager.weight", "sam3_model.weight"},
        )
