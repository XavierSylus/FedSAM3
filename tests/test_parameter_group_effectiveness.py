from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.client import TextOnlyTrainer
from src.federated_trainer import FederatedTrainer
from src.parameter_groups import IMAGE_PARAMS, PARAMETER_GROUPS


class _AdapterManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Linear(2, 2)])


class _FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self._text_projection = nn.Linear(2, 2)
        self._fusion_gate = nn.Linear(2, 2)


class _AllGroupsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_adapter = nn.Linear(2, 2)
        self.text_proj = nn.Linear(2, 2)
        self.adapter_manager = _AdapterManager()
        self.medical_seg_head = nn.Linear(2, 2)
        self.fusion_head = _FusionHead()


class _TextOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(2, 2)


def _trainer(modality="text_only"):
    trainer = TextOnlyTrainer(
        private_loader=None,
        public_loader=None,
        device="cpu",
        use_amp=False,
    )
    trainer.client_modality = modality
    return trainer


def _snapshot(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _reports(client_modalities):
    return {
        client_id: {
            "modality": modality,
            "groups": {
                parameter_group: {
                    "aggregation_eligible_count": 0,
                    "aggregated_count": 0,
                    "aggregation_eligible": False,
                    "aggregated": False,
                }
                for parameter_group in PARAMETER_GROUPS
            },
        }
        for client_id, modality in client_modalities.items()
    }


def _aggregation_audit(routing_mode, client_modalities, eligible_client_ids):
    return {
        "routing_mode": routing_mode,
        "active_client_ids": list(client_modalities),
        "parameters": {
            "medical_seg_head.weight": {
                "parameter_group": IMAGE_PARAMS,
                "eligible_client_ids": eligible_client_ids,
                "normalized_weights": {
                    client_id: 1.0 / len(eligible_client_ids)
                    for client_id in eligible_client_ids
                },
            }
        },
    }


def test_absent_groups_are_not_fabricated():
    model = _TextOnlyModel()
    trainer = _trainer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._activate_optimizer_parameter_scope(model, optimizer)
    trainer._reset_adapter_grad_tracking(model)
    trainer._initialize_parameter_group_effectiveness(model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer._record_parameter_group_gradients(optimizer)
    report = trainer.collect_parameter_group_effectiveness(
        model,
        _snapshot(model),
        trainer.get_uploadable_state(model),
    )

    assert report["groups"][IMAGE_PARAMS]["status"] == "not_present"
    assert report["groups"][IMAGE_PARAMS]["model_parameter_count"] == 0


def test_expected_group_with_all_none_gradients_fails():
    model = _TextOnlyModel()
    trainer = _trainer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._activate_optimizer_parameter_scope(model, optimizer)
    trainer._reset_adapter_grad_tracking(model)
    trainer._initialize_parameter_group_effectiveness(model)

    with pytest.raises(RuntimeError, match="all gradients None"):
        trainer.collect_parameter_group_effectiveness(
            model,
            _snapshot(model),
            trainer.get_uploadable_state(model),
        )


def test_nonzero_delta_missing_from_upload_fails():
    model = _TextOnlyModel()
    trainer = _trainer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._activate_optimizer_parameter_scope(model, optimizer)
    trainer._reset_adapter_grad_tracking(model)
    trainer._initialize_parameter_group_effectiveness(model)
    round_global = _snapshot(model)
    with torch.no_grad():
        model.text_proj.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="Nonzero optimizer delta is missing"):
        trainer.collect_parameter_group_effectiveness(model, round_global, {})


def test_multimodal_records_all_five_group_transitions():
    model = _AllGroupsModel()
    trainer = _trainer("multimodal")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._activate_optimizer_parameter_scope(model, optimizer)
    trainer._reset_adapter_grad_tracking(model)
    trainer._initialize_parameter_group_effectiveness(model)
    round_global = _snapshot(model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
        parameter.data.add_(0.5)
    trainer._record_parameter_group_gradients(optimizer)
    report = trainer.collect_parameter_group_effectiveness(
        model,
        round_global,
        trainer.get_uploadable_state(model),
    )

    for group_report in report["groups"].values():
        assert group_report["present_in_model"]
        assert group_report["forward_grad_seen"]
        assert group_report["in_optimizer"]
        assert group_report["nonzero_gradient"]
        assert group_report["nonzero_delta"]
        assert group_report["uploaded"]


def test_restricted_routing_rejects_disallowed_client():
    client_modalities = {"text": "text_only", "image": "image_only"}
    trainer = object.__new__(FederatedTrainer)
    trainer.config = SimpleNamespace(routing_mode="restricted")
    audit = _aggregation_audit("restricted", client_modalities, ["text"])

    with pytest.raises(RuntimeError, match="routing-ineligible"):
        trainer._finalize_parameter_group_effectiveness(
            _reports(client_modalities),
            client_modalities,
            audit,
            {"medical_seg_head.weight": torch.zeros(2, 2)},
        )


def test_unrestricted_marks_zero_update_client_eligible_and_aggregated():
    client_modalities = {"text": "text_only", "image": "image_only"}
    trainer = object.__new__(FederatedTrainer)
    trainer.config = SimpleNamespace(routing_mode="unrestricted")
    finalized = trainer._finalize_parameter_group_effectiveness(
        _reports(client_modalities),
        client_modalities,
        _aggregation_audit("unrestricted", client_modalities, ["text", "image"]),
        {"medical_seg_head.weight": torch.zeros(2, 2)},
    )

    text_report = finalized["text"]["groups"][IMAGE_PARAMS]
    assert text_report["aggregation_eligible_count"] == 1
    assert text_report["aggregated_count"] == 1
    assert text_report["aggregation_eligible"]
    assert text_report["aggregated"]
