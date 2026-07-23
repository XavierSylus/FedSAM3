from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.client import ImageOnlyTrainer
from src.config_manager import FederatedConfig
from src.federated_trainer import FederatedTrainer


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_manager = nn.Linear(2, 2)
        self.fusion_head = nn.Linear(2, 2)
        self.medical_seg_head = nn.Linear(2, 2)
        self._output_conv = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.text_proj = nn.Linear(2, 2)


def _make_loader():
    dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1))
    return DataLoader(dataset, batch_size=1)


def test_image_only_trainer_text_param_upload_toggle():
    model = _ToyModel()
    trainer = ImageOnlyTrainer(
        private_loader=_make_loader(),
        public_loader=_make_loader(),
        device="cpu",
        use_amp=False,
        allow_text_param_upload=False,
    )

    weights, _, _, _ = trainer.get_return_values(
        model=model,
        local_reps=torch.zeros(2),
        training_stats={"avg_loss": 0.0},
    )

    assert "adapter_manager.weight" in weights
    assert "text_encoder.weight" not in weights
    assert "text_proj.weight" not in weights

    trainer_upload = ImageOnlyTrainer(
        private_loader=_make_loader(),
        public_loader=_make_loader(),
        device="cpu",
        use_amp=False,
        allow_text_param_upload=True,
    )
    uploaded_weights, _, _, _ = trainer_upload.get_return_values(
        model=model,
        local_reps=torch.zeros(2),
        training_stats={"avg_loss": 0.0},
    )

    assert "adapter_manager.weight" in uploaded_weights
    assert "text_encoder.weight" in uploaded_weights
    assert "text_proj.weight" in uploaded_weights


def test_protocol_helpers_decouple_b_and_c_flags():
    trainer = FederatedTrainer.__new__(FederatedTrainer)

    trainer.config = SimpleNamespace(lambda_cream=0.02, use_decoupled_agg=False)
    assert trainer._should_allow_text_param_upload() is True

    trainer.config = SimpleNamespace(lambda_cream=0.1, use_decoupled_agg=True)
    assert trainer._should_allow_text_param_upload() is False
    assert trainer._should_enable_text_assist_in_seg() is True

    trainer.config = SimpleNamespace(
        lambda_cream=0.0,
        use_decoupled_agg=False,
        enable_text_assist_in_seg=False,
    )
    assert trainer._should_allow_text_param_upload() is False
    assert trainer._should_enable_text_assist_in_seg() is False


def test_strict_matrix_always_uses_round_global_state():
    trainer = FederatedTrainer.__new__(FederatedTrainer)
    round_global_state = {"weight": torch.tensor([1.0])}
    cached_client_state = {"weight": torch.tensor([9.0])}

    for baseline_method in ("none", "fedprox"):
        trainer.config = SimpleNamespace(
            baseline_method=baseline_method,
            client_init_policy="round_global",
            persist_client_optimizer=False,
        )

        selected = trainer._select_client_initial_state(
            round_global_state=round_global_state,
            cached_client_state=cached_client_state,
        )

        assert selected is round_global_state
        assert trainer._should_restore_client_optimizer() is False


def test_strict_matrix_rejects_stateful_client_protocol():
    with pytest.raises(ValueError, match="client_init_policy"):
        FederatedConfig(client_init_policy="local_cache")

    with pytest.raises(ValueError, match="persist_client_optimizer"):
        FederatedConfig(persist_client_optimizer=True)


def test_missing_modality_client_ratio_is_derived_from_client_composition():
    clients_half_missing = {
        "client_2": {"modality": "image_only"},
        "client_3": {"modality": "multimodal"},
    }
    clients_two_thirds_missing = {
        "client_1": {"modality": "text_only"},
        "client_2": {"modality": "image_only"},
        "client_3": {"modality": "multimodal"},
    }

    assert FederatedTrainer._missing_modality_client_ratio(
        clients_half_missing
    ) == pytest.approx(0.5)
    assert FederatedTrainer._missing_modality_client_ratio(
        clients_two_thirds_missing
    ) == pytest.approx(2.0 / 3.0)


def test_optimizer_hyperparameters_are_loaded_from_training_config(tmp_path):
    config_path = tmp_path / "optimizer_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                "  weight_decay: 0.02",
                "  seg_head_lr: 0.002",
                "  adapter_lr: 0.0002",
                "server:",
                "  proxy_client_id: client_3",
            ]
        ),
        encoding="utf-8",
    )

    config = FederatedConfig.from_yaml(str(config_path))

    assert config.weight_decay == pytest.approx(0.02)
    assert config.seg_head_lr == pytest.approx(0.002)
    assert config.adapter_lr == pytest.approx(0.0002)
    assert config.proxy_client_id == "client_3"


def test_round_global_state_is_restored_before_aggregation():
    trainer = FederatedTrainer.__new__(FederatedTrainer)
    trainer.global_model = object()
    trainer.device = "cpu"
    captured = {}

    def fake_load(model, state, device):
        captured["model"] = model
        captured["state"] = state
        captured["device"] = device

    trainer.load_trainable_state_dict = fake_load
    round_global_state = {"weight": torch.tensor([1.0])}

    trainer._restore_round_global_before_aggregation(round_global_state)

    assert captured == {
        "model": trainer.global_model,
        "state": round_global_state,
        "device": "cpu",
    }
