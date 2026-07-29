import ast
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "fedsam3_s4_reverse_order_smoke.yaml"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "server_s4_reverse_order_smoke.py"
DATA_ROOT = "/autodl-fs/data/FedSAM3-Cream/datasets/federated_split"
LOG_DIR = (
    "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
    "server_s4_reverse_order_smoke_rerun"
)
FORWARD_EVIDENCE_DIR = (
    "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
    "server_s2_three_client_preflight"
)
REVERSE_ORDER = ["client_3", "client_2", "client_1"]


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_s4_reverses_all_three_clients_under_the_s2_training_contract():
    config = _load_config()

    assert config["data_root"] == DATA_ROOT
    assert config["max_samples"] == 3
    assert config["federated"]["clients"] == [
        {
            "client_id": "client_3",
            "modality": "multimodal",
            "data_source": (
                f"{DATA_ROOT}/client3_multimodal/dataset.json"
            ),
            "enabled": True,
        },
        {
            "client_id": "client_2",
            "modality": "image_only",
            "data_source": (
                f"{DATA_ROOT}/client2_image_only/dataset.json"
            ),
            "enabled": True,
        },
        {
            "client_id": "client_1",
            "modality": "text_only",
            "data_source": (
                f"{DATA_ROOT}/client1_text_only/dataset.json"
            ),
            "enabled": True,
        },
    ]
    assert config["federated"]["routing_mode"] == "restricted"
    assert config["federated"]["client_init_policy"] == "round_global"
    assert config["training"]["rounds"] == 1
    assert config["training"]["local_epochs"] == 2
    assert config["server"]["proxy_k_batches"] == 3
    assert config["server"]["proxy_client_id"] == "client_3"
    assert config["baseline"] == {"method": "fedprox", "mu": 0.01}


def test_s4_paths_and_order_comparison_are_configuration_driven():
    config = _load_config()

    assert config["logging"]["log_dir"] == LOG_DIR
    assert PurePosixPath(config["logging"]["log_dir"]).is_absolute()
    assert config["s4_smoke"] == {
        "client_order": REVERSE_ORDER,
        "forward_evidence_dir": FORWARD_EVIDENCE_DIR,
    }


def test_s4_script_reorders_complete_client_state_and_audits_real_evidence():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT_PATH))

    for client_mapping in (
        "client_configs",
        "client_trainers",
        "client_states",
        "client_sample_counts",
        "_initial_loader_randomness",
    ):
        assert client_mapping in source
    for required_operation in (
        "_train_single_round",
        "_evaluate_validation",
        "_finalize_training",
        "client_participation_order",
        "buffer_distribution",
        "restore_events",
        "both_empty_count",
        "empty_fp_count",
        "empty_fn_count",
        "both_nonempty_count",
        "s4_smoke_result.json",
    ):
        assert required_operation in source

    assert "torch.randn" not in source
    assert "/autodl-fs/" not in source
    assert "--data_root" not in source
    assert "--log_dir" not in source
    assert ".unlink(" not in source
    assert "shutil.rmtree" not in source


def test_s4_protocol_comparison_canonicalizes_json_container_types():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SCRIPT_PATH))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_canonical_json_value"
    )
    namespace = {"Any": Any, "json": json}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            filename=str(SCRIPT_PATH),
            mode="exec",
        ),
        namespace,
    )
    canonicalize = namespace["_canonical_json_value"]

    from_yaml = {"thresholds": (0.5, 0.5, 0.5)}
    from_json = {"thresholds": [0.5, 0.5, 0.5]}
    assert canonicalize(from_yaml) == canonicalize(from_json)
