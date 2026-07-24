import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.client import BaseClientTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = PROJECT_ROOT / "src" / "client.py"


class _FedProxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(2, 1, bias=False)
        self.vision_head = nn.Linear(2, 1, bias=False)


def _trainer(model, optimizer, baseline_method, fedprox_mu):
    trainer = object.__new__(BaseClientTrainer)
    trainer.device = "cpu"
    trainer.baseline_method = baseline_method
    trainer.fedprox_mu = fedprox_mu
    trainer._activate_optimizer_parameter_scope(model, optimizer)
    return trainer


def _reference(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _task_loss(model):
    return torch.sum(model.text_proj.weight ** 2)


def test_fedavg_and_fedprox_have_identical_task_loss_and_initial_gradient():
    model = _FedProxModel()
    optimizer = torch.optim.SGD(model.text_proj.parameters(), lr=0.1)
    reference = _reference(model)
    fedavg = _trainer(model, optimizer, "none", 0.0)
    fedprox = _trainer(model, optimizer, "fedprox", 0.5)

    fedavg_task_loss = _task_loss(model)
    fedavg_objective = fedavg_task_loss + fedavg._compute_fedprox_penalty(
        model,
        None,
        set(),
    )
    fedavg_gradient = torch.autograd.grad(
        fedavg_objective,
        model.text_proj.weight,
    )[0]

    fedprox_task_loss = _task_loss(model)
    proximal_loss = fedprox._compute_fedprox_penalty(
        model,
        reference,
        fedprox.get_active_optimizer_parameter_names(),
    )
    fedprox_objective = fedprox_task_loss + proximal_loss
    fedprox_gradient = torch.autograd.grad(
        fedprox_objective,
        model.text_proj.weight,
    )[0]

    torch.testing.assert_close(fedavg_task_loss, fedprox_task_loss)
    torch.testing.assert_close(proximal_loss, torch.zeros_like(proximal_loss))
    torch.testing.assert_close(fedavg_objective, fedprox_objective)
    torch.testing.assert_close(fedavg_gradient, fedprox_gradient)


def test_fedprox_proximal_term_matches_formula_and_excludes_non_optimizer_parameters():
    model = _FedProxModel()
    optimizer = torch.optim.SGD(model.text_proj.parameters(), lr=0.1)
    trainer = _trainer(model, optimizer, "fedprox", 0.25)
    reference = _reference(model)
    with torch.no_grad():
        model.text_proj.weight.add_(2.0)
        model.vision_head.weight.add_(7.0)

    penalty = trainer._compute_fedprox_penalty(
        model,
        reference,
        trainer.get_active_optimizer_parameter_names(),
    )
    expected = 0.5 * trainer.fedprox_mu * torch.sum(
        (model.text_proj.weight - reference["text_proj.weight"]) ** 2
    )

    torch.testing.assert_close(penalty, expected)


def test_mu_zero_matches_fedavg_task_gradient_after_parameter_shift():
    model = _FedProxModel()
    optimizer = torch.optim.SGD(model.text_proj.parameters(), lr=0.1)
    reference = _reference(model)
    with torch.no_grad():
        model.text_proj.weight.add_(3.0)
    fedavg = _trainer(model, optimizer, "none", 0.0)
    fedprox_zero = _trainer(model, optimizer, "fedprox", 0.0)

    fedavg_task_loss = _task_loss(model)
    fedavg_gradient = torch.autograd.grad(
        fedavg_task_loss,
        model.text_proj.weight,
    )[0]
    fedprox_task_loss = _task_loss(model)
    proximal_loss = fedprox_zero._compute_fedprox_penalty(
        model,
        reference,
        fedprox_zero.get_active_optimizer_parameter_names(),
    )
    fedprox_gradient = torch.autograd.grad(
        fedprox_task_loss + proximal_loss,
        model.text_proj.weight,
    )[0]

    torch.testing.assert_close(fedavg_task_loss, fedprox_task_loss)
    torch.testing.assert_close(proximal_loss, torch.zeros_like(proximal_loss))
    torch.testing.assert_close(fedavg_gradient, fedprox_gradient)


def test_fedprox_rejects_non_optimizer_scope_and_missing_reference():
    model = _FedProxModel()
    optimizer = torch.optim.SGD(model.text_proj.parameters(), lr=0.1)
    trainer = _trainer(model, optimizer, "fedprox", 0.5)

    with pytest.raises(RuntimeError, match="must equal the active optimizer"):
        trainer._compute_fedprox_penalty(
            model,
            _reference(model),
            {"vision_head.weight"},
        )
    with pytest.raises(RuntimeError, match="requires a round-global reference"):
        trainer._compute_fedprox_penalty(
            model,
            None,
            trainer.get_active_optimizer_parameter_names(),
        )


def test_modality_task_loss_methods_do_not_reference_fedprox():
    tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
    target_classes = {"TextOnlyTrainer", "ImageOnlyTrainer", "MultimodalTrainer"}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in target_classes:
            continue
        method = next(
            child
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == "compute_loss"
        )
        names = {
            child.id for child in ast.walk(method) if isinstance(child, ast.Name)
        }
        attributes = {
            child.attr for child in ast.walk(method) if isinstance(child, ast.Attribute)
        }
        assert "fedprox" not in names
        assert "fedprox_mu" not in attributes
        assert "_compute_fedprox_penalty" not in attributes
        assert "routing_mode" not in attributes
