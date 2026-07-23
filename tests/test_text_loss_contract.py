import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.client import TextOnlyTrainer
from src.cream_losses import PrototypeLogisticTextLoss
from src.config_manager import FederatedConfig


class _TinyFusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(3, 2, bias=False)

    def project_text(self, text_features):
        return self.text_proj(text_features)


class _TinyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_head = _TinyFusionHead()

    def get_trainable_params(self):
        return list(self.fusion_head.text_proj.parameters())


def test_prototype_logistic_text_loss_matches_fixed_formula():
    projected = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    prototype = torch.tensor([0.25, 1.0], requires_grad=True)
    temperature = 0.2

    actual = PrototypeLogisticTextLoss(temperature)(projected, prototype)
    cosine = F.normalize(projected, dim=1) @ F.normalize(
        prototype.detach(), dim=0
    )
    expected = F.softplus(-cosine / temperature).mean()

    torch.testing.assert_close(actual, expected)


def test_text_projection_has_gradient_delta_and_upload_key():
    torch.manual_seed(3407)
    model = _TinyTextModel()
    trainer = TextOnlyTrainer(
        private_loader=None,
        public_loader=None,
        device="cpu",
        use_amp=False,
        text_loss_temperature=0.2,
    )
    optimizer = torch.optim.SGD(
        model.fusion_head.text_proj.parameters(), lr=0.1
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_ids == {
        id(parameter)
        for parameter in model.fusion_head.text_proj.parameters()
    }

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    prototype = torch.tensor([0.3, 1.0], requires_grad=True)
    total_loss, seg_loss, text_loss, projected_rep = trainer.compute_loss(
        model=model,
        private_inputs={
            "text_feat": torch.tensor(
                [[1.0, 0.5, -0.25], [-0.5, 1.0, 0.25]]
            )
        },
        public_inputs={},
        global_reps={"text": prototype, "image": torch.tensor([1.0, 0.0])},
        lambda_cream=0.1,
    )

    assert total_loss is text_loss
    assert seg_loss.item() == 0.0
    torch.testing.assert_close(
        torch.linalg.vector_norm(projected_rep, dim=1),
        torch.ones(projected_rep.shape[0]),
    )

    total_loss.backward()
    gradient = model.fusion_head.text_proj.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0
    assert prototype.grad is None

    optimizer.step()
    after = dict(model.named_parameters())["fusion_head.text_proj.weight"].detach()
    delta = after - before["fusion_head.text_proj.weight"]
    assert torch.count_nonzero(delta).item() > 0

    upload, image_rep, text_rep, _ = trainer.get_return_values(
        model=model,
        local_reps=projected_rep.mean(dim=0),
        training_stats={},
    )
    assert image_rep is None
    assert text_rep.shape == (2,)
    assert set(upload) == {"fusion_head.text_proj.weight"}
    torch.testing.assert_close(
        upload["fusion_head.text_proj.weight"]
        - before["fusion_head.text_proj.weight"],
        delta.cpu(),
    )


def test_text_client_requires_explicit_text_supervision_contract():
    clients = [
        {
            "client_id": "client_1",
            "modality": "text_only",
            "data_source": "unused.json",
            "enabled": True,
        }
    ]

    with pytest.raises(ValueError, match="text_supervision"):
        FederatedConfig(
            clients=clients,
            device="cpu",
            use_mock=True,
        )

    config = FederatedConfig(
        clients=clients,
        text_loss_name="prototype_logistic",
        text_loss_temperature=0.2,
        device="cpu",
        use_mock=True,
    )
    assert config.text_loss_name == "prototype_logistic"
    assert config.text_loss_temperature == 0.2
