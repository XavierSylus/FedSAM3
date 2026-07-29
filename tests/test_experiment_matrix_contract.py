import json
from pathlib import Path, PurePosixPath

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "fedsam3_experiment_manifest.json"

MATRIX_EXPECTATIONS = {
    "fedsam3_2x2_u_fedavg.yaml": ("unrestricted", "include_zero", "none", 0.0),
    "fedsam3_2x2_u_fedprox.yaml": ("unrestricted", "include_zero", "fedprox", 0.01),
    "fedsam3_2x2_r_fedavg.yaml": (
        "restricted",
        "exclude_and_renormalize",
        "none",
        0.0,
    ),
    "fedsam3_2x2_r_fedprox.yaml": (
        "restricted",
        "exclude_and_renormalize",
        "fedprox",
        0.01,
    ),
}

EXPECTED_DATA_ROOT = (
    "/autodl-fs/data/FedSAM3-Cream/datasets/federated_split"
)
EXPECTED_SAM3_CHECKPOINT = (
    "/autodl-fs/data/FedSAM3-Cream/datasets/checkpoints/sam3.pt"
)
EXPECTED_DATA_SOURCES = {
    "client_1": (
        f"{EXPECTED_DATA_ROOT}/client1_text_only/dataset.json"
    ),
    "client_2": (
        f"{EXPECTED_DATA_ROOT}/client2_image_only/dataset.json"
    ),
    "client_3": (
        f"{EXPECTED_DATA_ROOT}/client3_multimodal/dataset.json"
    ),
}
SERVER_ARTIFACT_CONFIGS = (
    "fedsam3_2x2_u_fedavg.yaml",
    "fedsam3_2x2_u_fedprox.yaml",
)
S2_CONFIG_FILENAME = "fedsam3_s2_three_client_preflight.yaml"
EXPECTED_S2_LOG_DIR = (
    "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
    "server_s2_three_client_preflight"
)
EXPECTED_LOG_DIRS = {
    "fedsam3_2x2_u_fedavg.yaml": (
        "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
        "fedsam3_2x2/u_fedavg"
    ),
    "fedsam3_2x2_u_fedprox.yaml": (
        "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
        "fedsam3_2x2/u_fedprox"
    ),
    "fedsam3_2x2_r_fedavg.yaml": (
        "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
        "fedsam3_2x2/r_fedavg"
    ),
    "fedsam3_2x2_r_fedprox.yaml": (
        "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
        "fedsam3_2x2/r_fedprox"
    ),
    "fedsam3_ratio_2of3_r_fedprox.yaml": (
        "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
        "fedsam3_ratio/2of3_r_fedprox"
    ),
}


def _load_yaml(filename: str) -> dict:
    path = PROJECT_ROOT / "configs" / filename
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _client_modalities(config: dict) -> dict:
    return {
        client["client_id"]: (client["modality"], client["enabled"])
        for client in config["federated"]["clients"]
    }


def _assert_server_artifact_paths(config: dict) -> None:
    assert config["model"]["sam3_checkpoint"] == EXPECTED_SAM3_CHECKPOINT
    assert {
        client["client_id"]: client["data_source"]
        for client in config["federated"]["clients"]
    } == EXPECTED_DATA_SOURCES


def test_server_storage_paths_are_explicit_and_isolated():
    for filename, expected_log_dir in EXPECTED_LOG_DIRS.items():
        config = _load_yaml(filename)
        assert config["data_root"] == EXPECTED_DATA_ROOT
        assert config["logging"]["log_dir"] == expected_log_dir


def test_u_configs_use_external_server_artifacts():
    for filename in SERVER_ARTIFACT_CONFIGS:
        _assert_server_artifact_paths(_load_yaml(filename))


def test_s2_config_uses_external_manifests_and_a_small_validation_window():
    config = _load_yaml(S2_CONFIG_FILENAME)

    assert config["data_root"] == EXPECTED_DATA_ROOT
    assert config["max_samples"] == 1
    assert config["training"]["rounds"] == 1
    assert config["training"]["local_epochs"] == 2
    assert config["training"]["accumulation_steps"] == 1
    assert config["model"]["sam3_checkpoint"] == EXPECTED_SAM3_CHECKPOINT
    assert config["logging"]["log_dir"] == EXPECTED_S2_LOG_DIR

    for client in config["federated"]["clients"]:
        data_source = PurePosixPath(client["data_source"])
        assert data_source.is_absolute()
        assert str(data_source).startswith(f"{EXPECTED_DATA_ROOT}/")


def test_s2_max_samples_is_loaded_into_the_runtime_config():
    from src.config_manager import FederatedConfig

    config_path = PROJECT_ROOT / "configs" / S2_CONFIG_FILENAME
    runtime_config = FederatedConfig.from_yaml(str(config_path))

    assert runtime_config.max_samples == 1
    assert runtime_config.sam3_checkpoint == EXPECTED_SAM3_CHECKPOINT


def test_main_matrix_exposes_only_routing_and_fedprox_variables():
    configs = {filename: _load_yaml(filename) for filename in MATRIX_EXPECTATIONS}

    for filename, expected in MATRIX_EXPECTATIONS.items():
        routing_mode, unoptimized_policy, baseline_method, fedprox_mu = expected
        config = configs[filename]

        assert config["federated"]["routing_mode"] == routing_mode
        assert config["aggregation"] == {
            "method": "fedavg",
            "sample_weight_unit": "private_cases",
            "unoptimized_update_policy": unoptimized_policy,
        }
        assert config["baseline"] == {"method": baseline_method, "mu": fedprox_mu}
        assert _client_modalities(config) == {
            "client_1": ("text_only", True),
            "client_2": ("image_only", True),
            "client_3": ("multimodal", True),
        }
        assert config["federated"]["client_init_policy"] == "round_global"
        assert config["federated"]["persist_client_optimizer"] is False
        assert config["system"] == {
            "reproducibility_mode": "best_effort_cuda",
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
        }

    first = configs["fedsam3_2x2_u_fedavg.yaml"]
    for config in configs.values():
        for key in (
            "seed",
            "data_root",
            "training",
            "cream",
            "text_supervision",
            "model",
            "segmentation",
            "server",
            "options",
            "device",
            "checkpoint",
            "validation",
            "system",
        ):
            assert config[key] == first[key]


def test_manifest_matches_main_matrix_and_has_no_legacy_routing_flag():
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert "use_decoupled_agg" not in manifest_text
    assert "restricted_routing" not in manifest_text
    assert manifest["manifest_schema_version"] == 3
    assert manifest["matrix_unique_variables"] == [
        "routing_mode",
        "unoptimized_update_policy",
        "baseline_method",
        "fedprox_mu",
    ]
    assert manifest["client_structure"] == [
        {"client_id": "client_1", "modality": "text_only"},
        {"client_id": "client_2", "modality": "image_only"},
        {"client_id": "client_3", "modality": "multimodal"},
    ]

    expected_entries = {
        f"configs/{filename}": values
        for filename, values in MATRIX_EXPECTATIONS.items()
    }
    actual_entries = {
        entry["config"]: (
            entry["routing_mode"],
            entry["unoptimized_update_policy"],
            entry["baseline_method"],
            entry["fedprox_mu"],
        )
        for entry in manifest["matrix"]
    }
    assert actual_entries == expected_entries

    aggregation = manifest["parameterwise_aggregation"]
    assert aggregation["unrestricted"]["unoptimized_parameter_rule"] == (
        "Delta_{k,p}=0 while n_k remains in the denominator"
    )
    assert aggregation["restricted"]["empty_eligible_rule"] == (
        "preserve theta_p^t and write an aggregation audit event"
    )
    assert aggregation["parameter_buffer_boundary"]["upload"] == (
        "optimizer named parameters only"
    )
    assert manifest["fixed_controls"]["reproducibility"] == {
        "mode": "best_effort_cuda",
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "bitwise_reproducible": False,
        "known_nondeterministic_operation": "grid_sampler_2d_backward_cuda",
        "comparison_rule": "all experiment cells use the same declared seed schedule",
        "num_workers": 0,
        "persistent_workers": False,
        "recorded_state": [
            "python_numpy_torch_cuda_seeds",
            "client_round_loader_and_slice_generators",
            "client_participation_order",
            "public_proxy_batch_order",
            "optimizer_and_scheduler_initial_state",
            "configuration_and_data_manifest_sha256",
            "git_and_runtime_environment",
        ],
    }


def test_ratio_configuration_matches_its_manifest_contract():
    config = _load_yaml("fedsam3_ratio_2of3_r_fedprox.yaml")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ratio = manifest["ratio_experiment"]

    assert _client_modalities(config) == {
        "client_1": ("text_only", False),
        "client_2": ("image_only", True),
        "client_3": ("multimodal", True),
    }
    assert config["federated"]["routing_mode"] == ratio["routing_mode"]
    assert config["aggregation"]["sample_weight_unit"] == "private_cases"
    assert config["aggregation"]["unoptimized_update_policy"] == ratio[
        "unoptimized_update_policy"
    ]
    assert config["baseline"] == {
        "method": ratio["baseline_method"],
        "mu": ratio["fedprox_mu"],
    }
    assert ratio["enabled_client_ids"] == ["client_2", "client_3"]
    assert ratio["client_participation_ratio"] == 2.0 / 3.0
    assert config["system"]["reproducibility_mode"] == "best_effort_cuda"
    assert config["system"]["deterministic_algorithms"] is True
    assert config["system"]["deterministic_warn_only"] is True
