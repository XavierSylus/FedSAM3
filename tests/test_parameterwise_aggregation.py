import pytest
import torch
import torch.nn as nn

from src.server import CreamAggregator


class _FusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._fusion_gate = nn.Linear(1, 1, bias=False)


class _AggregationModel(nn.Module):
    def __init__(self, include_unclassified=False):
        super().__init__()
        self.contrastive_dim = 2
        self.text_proj = nn.Linear(1, 1, bias=False)
        self.medical_seg_head = nn.Linear(1, 1, bias=False)
        self.fusion_head = _FusionModule()
        if include_unclassified:
            self.unclassified = nn.Linear(1, 1, bias=False)
        self.register_buffer("running_stat", torch.zeros(1))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(10.0)


def _aggregator(model):
    return CreamAggregator(model, device="cpu", aggregation_method="fedavg")


def _round_global(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _update(round_global, name, value):
    return {name: torch.full_like(round_global[name], value)}


def _aggregate(aggregator, round_global, updates, routing_mode):
    return aggregator.aggregate_weights(
        round_global_parameters=round_global,
        client_updates=updates,
        client_modalities={
            "client_1": "text_only",
            "client_2": "image_only",
            "client_3": "multimodal",
        },
        client_sample_counts={"client_1": 1, "client_2": 2, "client_3": 3},
        routing_mode=routing_mode,
    )


def test_unrestricted_uses_all_active_sample_counts_for_missing_updates():
    model = _AggregationModel()
    aggregator = _aggregator(model)
    round_global = _round_global(model)
    name = "text_proj.weight"

    aggregated = _aggregate(
        aggregator,
        round_global,
        {
            "client_1": _update(round_global, name, 16.0),
            "client_2": {},
            "client_3": {},
        },
        routing_mode="unrestricted",
    )

    torch.testing.assert_close(
        aggregated[name], torch.full_like(round_global[name], 11.0)
    )
    assert set(aggregated) == set(round_global)
    assert "running_stat" not in aggregated
    audit = aggregator._last_aggregation_audit["parameters"][name]
    assert audit["eligible_client_ids"] == ["client_1", "client_2", "client_3"]
    assert audit["zero_update_client_ids"] == ["client_2", "client_3"]
    assert audit["normalized_weights"] == {
        "client_1": pytest.approx(1 / 6),
        "client_2": pytest.approx(2 / 6),
        "client_3": pytest.approx(3 / 6),
    }


def test_restricted_renormalizes_only_eligible_optimizer_uploaders():
    model = _AggregationModel()
    aggregator = _aggregator(model)
    round_global = _round_global(model)
    name = "text_proj.weight"
    updates = {
        "client_1": _update(round_global, name, 16.0),
        "client_2": _update(round_global, name, 100.0),
        "client_3": _update(round_global, name, 22.0),
    }

    unrestricted = _aggregate(
        aggregator, round_global, updates, routing_mode="unrestricted"
    )
    restricted = _aggregate(
        aggregator, round_global, updates, routing_mode="restricted"
    )

    torch.testing.assert_close(
        unrestricted[name], torch.full_like(round_global[name], 47.0)
    )
    torch.testing.assert_close(
        restricted[name], torch.full_like(round_global[name], 20.5)
    )
    audit = aggregator._last_aggregation_audit["parameters"][name]
    assert audit["eligible_client_ids"] == ["client_1", "client_3"]
    assert audit["normalized_weights"] == {
        "client_1": pytest.approx(1 / 4),
        "client_3": pytest.approx(3 / 4),
    }


@pytest.mark.parametrize(
    ("name", "expected_eligible"),
    [
        ("medical_seg_head.weight", ["client_2", "client_3"]),
        ("text_proj.weight", ["client_1", "client_3"]),
        ("fusion_head._fusion_gate.weight", ["client_3"]),
    ],
)
def test_restricted_routing_enforces_parameter_group_allowlists(
    name, expected_eligible
):
    model = _AggregationModel()
    aggregator = _aggregator(model)
    round_global = _round_global(model)
    updates = {
        "client_1": _update(round_global, name, 11.0),
        "client_2": _update(round_global, name, 12.0),
        "client_3": _update(round_global, name, 13.0),
    }

    _aggregate(aggregator, round_global, updates, routing_mode="restricted")

    audit = aggregator._last_aggregation_audit["parameters"][name]
    assert audit["eligible_client_ids"] == expected_eligible


def test_restricted_empty_eligibility_keeps_round_global_and_records_audit():
    model = _AggregationModel()
    aggregator = _aggregator(model)
    round_global = _round_global(model)
    name = "medical_seg_head.weight"

    aggregated = _aggregate(
        aggregator,
        round_global,
        {
            "client_1": _update(round_global, name, 99.0),
            "client_2": {},
            "client_3": {},
        },
        routing_mode="restricted",
    )

    torch.testing.assert_close(aggregated[name], round_global[name])
    audit = aggregator._last_aggregation_audit["parameters"][name]
    assert audit["empty_eligible"] is True
    assert audit["eligible_client_ids"] == []


def test_unclassified_trainable_parameter_and_misaligned_clients_fail_fast():
    unclassified_model = _AggregationModel(include_unclassified=True)
    with pytest.raises(ValueError, match="Unclassified"):
        _aggregate(
            _aggregator(unclassified_model),
            _round_global(unclassified_model),
            {"client_1": {}, "client_2": {}, "client_3": {}},
            routing_mode="unrestricted",
        )

    model = _AggregationModel()
    with pytest.raises(ValueError, match="same client IDs"):
        model_aggregator = _aggregator(model)
        model_aggregator.aggregate_weights(
            round_global_parameters=_round_global(model),
            client_updates={"client_1": {}},
            client_modalities={"client_1": "text_only"},
            client_sample_counts={"client_1": 1, "client_2": 2},
            routing_mode="unrestricted",
        )
