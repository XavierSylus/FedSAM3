import json
from pathlib import Path

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


def _load_yaml(filename: str) -> dict:
    path = PROJECT_ROOT / "configs" / filename
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _client_modalities(config: dict) -> dict:
    return {
        client["client_id"]: (client["modality"], client["enabled"])
        for client in config["federated"]["clients"]
    }


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
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
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
