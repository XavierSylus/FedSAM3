import math

import pytest
import torch

from src.update_diagnostics import compute_parameter_group_diagnostics


def test_group_diagnostics_detect_opposite_and_aligned_updates():
    global_state = {
        "adapters.0.weight": torch.zeros(2),
        "fusion_head.text_proj.weight": torch.zeros(2),
    }
    client_updates = {
        "image": {
            "adapters.0.weight": torch.tensor([1.0, 0.0]),
            "fusion_head.text_proj.weight": torch.tensor([1.0, 0.0]),
        },
        "multi": {
            "adapters.0.weight": torch.tensor([-1.0, 0.0]),
            "fusion_head.text_proj.weight": torch.tensor([1.0, 0.0]),
        },
    }
    aggregated_state = {
        "adapters.0.weight": torch.zeros(2),
        "fusion_head.text_proj.weight": torch.tensor([1.0, 0.0]),
    }

    result = compute_parameter_group_diagnostics(
        round_global_state=global_state,
        client_updates=client_updates,
        client_modalities={"image": "image_only", "multi": "multimodal"},
        aggregated_state=aggregated_state,
    )

    conflicts = {item["parameter_group"]: item for item in result["pairwise_conflicts"]}
    assert conflicts["VISION_ADAPTER"]["cosine_similarity"] == pytest.approx(-1.0)
    assert conflicts["VISION_ADAPTER"]["angle_deg"] == pytest.approx(180.0)
    assert conflicts["TEXT_PARAMS"]["cosine_similarity"] == pytest.approx(1.0)
    assert conflicts["TEXT_PARAMS"]["angle_deg"] == pytest.approx(0.0)

    image_drift = result["client_drift"]["image"]["VISION_ADAPTER"]
    assert image_drift["update_l2"] == pytest.approx(1.0)
    assert math.isfinite(image_drift["relative_drift"])

    summary = result["conflict_summary"]["VISION_ADAPTER"]
    assert summary["negative_cosine_ratio"] == pytest.approx(1.0)


def test_group_diagnostics_require_shared_keys_for_conflict():
    result = compute_parameter_group_diagnostics(
        round_global_state={
            "adapters.0.weight": torch.zeros(1),
            "adapters.1.weight": torch.zeros(1),
        },
        client_updates={
            "a": {"adapters.0.weight": torch.ones(1)},
            "b": {"adapters.1.weight": torch.ones(1)},
        },
        client_modalities={"a": "image_only", "b": "multimodal"},
        aggregated_state={},
    )

    assert result["pairwise_conflicts"] == []
