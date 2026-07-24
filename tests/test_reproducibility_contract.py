import hashlib

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from data.heterogeneous_dataset_loader import (
    HeterogeneousBraTSDataset,
    configure_loader_randomness,
)
from src.config_manager import FederatedConfig
from src.federated_trainer import FederatedTrainer


def _config_yaml(deterministic_algorithms=True, deterministic_warn_only=False):
    return "\n".join(
        [
            "aggregation:",
            "  method: fedavg",
            "  sample_weight_unit: private_cases",
            "  unoptimized_update_policy: include_zero",
            "federated:",
            "  routing_mode: unrestricted",
            "  client_init_policy: round_global",
            "  persist_client_optimizer: false",
            "options:",
            "  use_amp: false",
            "system:",
            "  seed: 3407",
            f"  deterministic_algorithms: {str(deterministic_algorithms).lower()}",
            f"  deterministic_warn_only: {str(deterministic_warn_only).lower()}",
            "logging:",
            "  log_type: none",
            "  experiment_name: reproducibility-contract",
        ]
    )


class _ManifestDataset(Dataset):
    def __init__(self, case_ids):
        self.case_ids = list(case_ids)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        return torch.tensor(index)

    def get_reproducibility_manifest(self):
        return {"case_ids": self.case_ids}


def test_yaml_config_records_raw_sha256_and_deterministic_settings(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    raw_yaml = _config_yaml().encode("utf-8")
    config_path.write_bytes(raw_yaml)

    config = FederatedConfig.from_yaml(str(config_path))

    assert config.config_source_path == str(config_path.resolve())
    assert config.config_source_sha256 == hashlib.sha256(raw_yaml).hexdigest()
    assert config.deterministic_algorithms is True
    assert config.deterministic_warn_only is False


def test_config_source_provenance_does_not_change_run_identity(tmp_path):
    raw_yaml = _config_yaml().encode("utf-8")
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_bytes(raw_yaml)
    second_path.write_bytes(raw_yaml)

    first_trainer = object.__new__(FederatedTrainer)
    first_trainer.config = FederatedConfig.from_yaml(str(first_path))
    first_trainer.client_configs = {}
    first_trainer.client_sample_counts = {}
    second_trainer = object.__new__(FederatedTrainer)
    second_trainer.config = FederatedConfig.from_yaml(str(second_path))
    second_trainer.client_configs = {}
    second_trainer.client_sample_counts = {}

    assert first_trainer._build_run_identity()["config_hash"] == (
        second_trainer._build_run_identity()["config_hash"]
    )


def test_loader_randomness_replays_the_same_sampler_order():
    loader = DataLoader(_ManifestDataset(["a", "b", "c", "d"]), batch_size=1, shuffle=True)

    first_state = configure_loader_randomness(
        loader,
        order_seed=41,
        slice_seed=None,
    )
    first_order = [int(batch.item()) for batch in loader]
    second_state = configure_loader_randomness(
        loader,
        order_seed=41,
        slice_seed=None,
    )
    second_order = [int(batch.item()) for batch in loader]

    assert first_order == second_order
    assert first_state["order_generator_state_sha256"] == (
        second_state["order_generator_state_sha256"]
    )
    assert first_state["augmentation"] == {"enabled": False, "state": None}


def test_image_loader_records_a_dedicated_slice_generator():
    dataset = object.__new__(HeterogeneousBraTSDataset)
    dataset.client_type = "image_only"
    dataset.load_mask = True
    dataset.samples = None
    dataset.case_folders = [object()]
    dataset._slice_generator = None
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    state = configure_loader_randomness(loader, order_seed=41, slice_seed=53)

    assert dataset._slice_generator is not None
    assert state["slice_seed"] == 53
    assert len(state["slice_generator_state_sha256"]) == 64


def test_data_manifest_and_stream_seeds_are_recorded(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(_config_yaml(), encoding="utf-8")
    trainer = object.__new__(FederatedTrainer)
    trainer.config = FederatedConfig.from_yaml(str(config_path))
    trainer.device = "cpu"
    trainer.client_sample_counts = {"client_1": 2}
    trainer._initial_loader_randomness = {"client_1": {"private_loader": {}}}
    trainer.val_loader = None
    trainer.client_configs = {
        "client_1": {
            "modality": "text_only",
            "private_loader": DataLoader(_ManifestDataset(["case_a", "case_b"])),
            "public_loader": DataLoader(_ManifestDataset(["proxy_a"])),
        }
    }

    metadata = trainer._collect_run_metadata()

    assert metadata["data_manifest"]["client_participation_order"] == ["client_1"]
    assert metadata["data_manifest"]["clients"][0]["private"]["case_ids"] == [
        "case_a",
        "case_b",
    ]
    assert len(metadata["data_manifest_sha256"]) == 64
    assert metadata["protocol_payload"]["data_manifest_sha256"] == (
        metadata["data_manifest_sha256"]
    )
    assert trainer._derive_reproducibility_seed(
        round_num=2,
        client_id="client_1",
        stream="private_train:order",
    ) == trainer._derive_reproducibility_seed(
        round_num=2,
        client_id="client_1",
        stream="private_train:order",
    )


@pytest.mark.parametrize(
    ("deterministic_algorithms", "deterministic_warn_only"),
    [(False, False), (True, True)],
)
def test_non_strict_determinism_settings_are_rejected(
    tmp_path,
    deterministic_algorithms,
    deterministic_warn_only,
):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        _config_yaml(deterministic_algorithms, deterministic_warn_only),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="deterministic"):
        FederatedConfig.from_yaml(str(config_path))


def test_run_metadata_contains_seed_determinism_amp_and_environment(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(_config_yaml(), encoding="utf-8")
    config = FederatedConfig.from_yaml(str(config_path))
    trainer = object.__new__(FederatedTrainer)
    trainer.config = config
    trainer.device = "cpu"
    trainer.client_configs = {}
    trainer.client_sample_counts = {}
    trainer._set_random_seed()

    metadata = trainer._collect_run_metadata()

    assert metadata["random_seeds"] == {
        "python": 3407,
        "numpy": 3407,
        "torch": 3407,
        "cuda": 3407,
    }
    assert metadata["amp"] == {"enabled": False}
    assert metadata["config_file_sha256"] == config.config_source_sha256
    assert metadata["determinism"]["configured_algorithms"] is True
    assert metadata["determinism"]["configured_warn_only"] is False
    assert "cudnn_version" in metadata
    assert "gpu_devices" in metadata


def test_metadata_rejects_a_config_without_raw_yaml_sha256():
    config = FederatedConfig(
        aggregation_method="fedavg",
        routing_mode="unrestricted",
        sample_weight_unit="private_cases",
        unoptimized_update_policy="include_zero",
    )
    trainer = object.__new__(FederatedTrainer)
    trainer.config = config
    trainer.device = "cpu"
    trainer.client_configs = {}
    trainer.client_sample_counts = {}

    with pytest.raises(RuntimeError, match="YAML configuration SHA256"):
        trainer._collect_run_metadata()
