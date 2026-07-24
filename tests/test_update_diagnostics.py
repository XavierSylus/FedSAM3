import math

import pytest
import torch

from src.update_diagnostics import (
    compute_parameter_group_diagnostics,
    flatten_parameter_group_diagnostics,
)
from src.parameter_groups import classify_parameter


def _audit(routing_mode, global_state, client_updates, client_sample_counts, eligibility):
    active_client_ids = list(client_updates)
    parameters = {}
    for name in global_state:
        eligible_client_ids = eligibility[name]
        denominator = sum(client_sample_counts[client_id] for client_id in eligible_client_ids)
        parameters[name] = {
            "parameter_group": classify_parameter(name),
            "eligible_client_ids": eligible_client_ids,
            "normalized_weights": {
                client_id: client_sample_counts[client_id] / denominator
                for client_id in eligible_client_ids
            },
        }
    return {
        "routing_mode": routing_mode,
        "active_client_ids": active_client_ids,
        "client_sample_counts": client_sample_counts,
        "parameters": parameters,
    }


def test_group_diagnostics_detect_opposite_and_aligned_updates():
    global_state = {
        "adapters.0.weight": torch.zeros(2),
        "fusion_head.text_proj.weight": torch.zeros(2),
        "fusion_head._fusion_gate.0.weight": torch.zeros(2),
    }
    client_updates = {
        "image": {
            "adapters.0.weight": torch.tensor([1.0, 0.0]),
            "fusion_head.text_proj.weight": torch.tensor([1.0, 0.0]),
            "fusion_head._fusion_gate.0.weight": torch.tensor([0.0, 1.0]),
        },
        "multi": {
            "adapters.0.weight": torch.tensor([-1.0, 0.0]),
            "fusion_head.text_proj.weight": torch.tensor([1.0, 0.0]),
            "fusion_head._fusion_gate.0.weight": torch.tensor([0.0, 1.0]),
        },
    }
    aggregated_state = {
        "adapters.0.weight": torch.zeros(2),
        "fusion_head.text_proj.weight": torch.tensor([1.0, 0.0]),
        "fusion_head._fusion_gate.0.weight": torch.tensor([0.0, 1.0]),
    }
    client_sample_counts = {"image": 1, "multi": 2}

    result = compute_parameter_group_diagnostics(
        round_global_state=global_state,
        client_updates=client_updates,
        client_modalities={"image": "image_only", "multi": "multimodal"},
        client_sample_counts=client_sample_counts,
        aggregation_audit=_audit(
            "unrestricted",
            global_state,
            client_updates,
            client_sample_counts,
            {name: ["image", "multi"] for name in global_state},
        ),
        aggregated_state=aggregated_state,
    )

    conflicts = {item["parameter_group"]: item for item in result["pairwise_conflicts"]}
    assert conflicts["VISION_ADAPTER"]["cosine_similarity"] == pytest.approx(-1.0)
    assert conflicts["VISION_ADAPTER"]["angle_deg"] == pytest.approx(180.0)
    assert conflicts["TEXT_PARAMS"]["cosine_similarity"] == pytest.approx(1.0)
    assert conflicts["TEXT_PARAMS"]["angle_deg"] == pytest.approx(0.0)
    assert conflicts["FUSION_PARAMS"]["cosine_similarity"] == pytest.approx(1.0)

    image_drift = result["client_drift"]["image"]["VISION_ADAPTER"]
    assert image_drift["update_l2"] == pytest.approx(1.0)
    assert image_drift["update_rms"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert image_drift["parameter_count"] == 1
    assert image_drift["nonzero_parameter_ratio"] == pytest.approx(1.0)
    assert image_drift["sample_weight"] == 1
    assert math.isfinite(image_drift["relative_drift"])

    summary = result["conflict_summary"]["VISION_ADAPTER"]
    assert summary["negative_cosine_ratio"] == pytest.approx(1.0)
    assert summary["conflict_rate"] == pytest.approx(1.0)
    server_drift = result["global_drift"]["VISION_ADAPTER"]
    assert server_drift["routing_mode"] == "unrestricted"
    assert server_drift["aggregation_client_ids"] == ["image", "multi"]

    rows = flatten_parameter_group_diagnostics(3, result)
    assert {row["row_type"] for row in rows} == {
        "client_drift",
        "pairwise_conflict",
        "conflict_summary",
        "server_drift",
    }
    assert all(row["round"] == 3 for row in rows)


def test_group_diagnostics_require_shared_keys_for_conflict():
    global_state = {
        "adapters.0.weight": torch.zeros(1),
        "adapters.1.weight": torch.zeros(1),
    }
    client_updates = {
        "a": {"adapters.0.weight": torch.ones(1)},
        "b": {"adapters.1.weight": torch.ones(1)},
    }
    client_sample_counts = {"a": 1, "b": 1}
    result = compute_parameter_group_diagnostics(
        round_global_state=global_state,
        client_updates=client_updates,
        client_modalities={"a": "image_only", "b": "multimodal"},
        client_sample_counts=client_sample_counts,
        aggregation_audit=_audit(
            "restricted",
            global_state,
            client_updates,
            client_sample_counts,
            {"adapters.0.weight": ["a"], "adapters.1.weight": ["b"]},
        ),
        aggregated_state=global_state,
    )

    conflict = result["pairwise_conflicts"][0]
    assert conflict["cosine_similarity"] is None
    assert conflict["angle_deg"] is None
    assert conflict["conflict_status"] == "no_shared_parameters"
    assert conflict["shared_parameter_count"] == 0
    summary = result["conflict_summary"]["VISION_ADAPTER"]
    assert summary["pair_count"] == 0
    assert summary["conflict_rate"] is None
    assert summary["no_shared_pair_count"] == 1


def test_unrestricted_zero_update_dilution_changes_only_server_drift():
    global_state = {"medical_seg_head.weight": torch.zeros(1)}
    client_updates = {
        "text": {},
        "image": {"medical_seg_head.weight": torch.tensor([6.0])},
        "multi": {},
    }
    client_sample_counts = {"text": 1, "image": 2, "multi": 3}
    result = compute_parameter_group_diagnostics(
        round_global_state=global_state,
        client_updates=client_updates,
        client_modalities={
            "text": "text_only",
            "image": "image_only",
            "multi": "multimodal",
        },
        client_sample_counts=client_sample_counts,
        aggregation_audit=_audit(
            "unrestricted",
            global_state,
            client_updates,
            client_sample_counts,
            {"medical_seg_head.weight": ["text", "image", "multi"]},
        ),
        aggregated_state={"medical_seg_head.weight": torch.tensor([2.0])},
    )

    image_drift = result["client_drift"]["image"]["IMAGE_PARAMS"]
    assert image_drift["update_l2"] == pytest.approx(6.0)
    assert image_drift["sample_weight"] == 2
    assert all(
        conflict["conflict_status"] == "no_shared_parameters"
        for conflict in result["pairwise_conflicts"]
    )
    assert all(
        conflict["cosine_similarity"] is None
        for conflict in result["pairwise_conflicts"]
    )
    server_drift = result["global_drift"]["IMAGE_PARAMS"]
    assert server_drift["update_l2"] == pytest.approx(2.0)
    assert server_drift["aggregation_client_ids"] == ["image", "multi", "text"]
    weights = server_drift["aggregation_participation"]["medical_seg_head.weight"]
    assert weights["normalized_weights"]["text"] == pytest.approx(1.0 / 6.0)
    assert weights["normalized_weights"]["image"] == pytest.approx(2.0 / 6.0)
    assert weights["normalized_weights"]["multi"] == pytest.approx(3.0 / 6.0)
