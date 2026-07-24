import torch
import torch.nn as nn
import pytest

from src.client import BaseClientTrainer


class _ContractTrainer(BaseClientTrainer):
    def unpack_private_batch(self, batch):
        raise NotImplementedError

    def unpack_public_batch(self, batch):
        raise NotImplementedError

    def compute_loss(self, model, private_inputs, public_inputs, global_reps, lambda_cream):
        raise NotImplementedError

    def get_return_values(self, model, local_reps, training_stats):
        raise NotImplementedError


class _UploadContractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(2, 2, bias=False)
        self.vision_head = nn.Linear(2, 2, bias=False)
        self.fusion_gate = nn.Linear(2, 1, bias=False)
        self.not_optimized = nn.Parameter(torch.ones(1))
        self.register_buffer("running_stat", torch.zeros(1))


def _trainer_with_optimizer_scope(model, optimizer):
    trainer = object.__new__(_ContractTrainer)
    trainer.device = "cpu"
    trainer.baseline_method = "fedprox"
    trainer.fedprox_mu = 0.5
    trainer._activate_optimizer_parameter_scope(model, optimizer)
    return trainer


@pytest.mark.parametrize(
    ("optimizer_params", "expected_names"),
    [
        ("text", {"text_proj.weight"}),
        ("vision", {"vision_head.weight"}),
        (
            "multimodal",
            {"text_proj.weight", "vision_head.weight", "fusion_gate.weight"},
        ),
    ],
)
def test_upload_equals_explicit_optimizer_named_parameters(
    optimizer_params, expected_names
):
    model = _UploadContractModel()
    parameter_sets = {
        "text": list(model.text_proj.parameters()),
        "vision": list(model.vision_head.parameters()),
        "multimodal": [
            *model.text_proj.parameters(),
            *model.vision_head.parameters(),
            *model.fusion_gate.parameters(),
        ],
    }
    optimizer = torch.optim.SGD(parameter_sets[optimizer_params], lr=0.1)
    trainer = _trainer_with_optimizer_scope(model, optimizer)

    round_global = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        for parameter in parameter_sets[optimizer_params]:
            parameter.add_(0.25)

    upload = trainer.get_uploadable_state(model)

    assert set(trainer.get_active_optimizer_parameter_names()) == expected_names
    assert set(upload) == expected_names
    assert "running_stat" not in upload
    assert "not_optimized" not in upload
    for name, value in upload.items():
        delta = value - round_global[name]
        assert torch.isfinite(delta).all()
        assert torch.count_nonzero(delta).item() > 0


def test_optimizer_scope_rejects_unregistered_and_frozen_parameters():
    model = _UploadContractModel()
    unregistered = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="not a named parameter"):
        BaseClientTrainer.resolve_optimizer_parameter_names(
            model, torch.optim.SGD([unregistered], lr=0.1)
        )

    model.fusion_gate.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="does not require gradients"):
        BaseClientTrainer.resolve_optimizer_parameter_names(
            model, torch.optim.SGD(model.fusion_gate.parameters(), lr=0.1)
        )


def test_fedprox_uses_the_same_optimizer_scope_as_upload():
    model = _UploadContractModel()
    optimizer = torch.optim.SGD(model.text_proj.parameters(), lr=0.1)
    trainer = _trainer_with_optimizer_scope(model, optimizer)
    round_global = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        model.text_proj.weight.add_(2.0)
        model.vision_head.weight.add_(7.0)

    penalty = trainer._compute_fedprox_penalty(
        model=model,
        global_reference_state=round_global,
        fedprox_param_names=trainer.get_active_optimizer_parameter_names(),
    )
    expected = 0.5 * trainer.fedprox_mu * torch.sum(
        (model.text_proj.weight - round_global["text_proj.weight"]) ** 2
    )

    torch.testing.assert_close(penalty, expected)
