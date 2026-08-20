import csv
import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import yaml

from data_processing import collect_fedsam3_2x2_evidence as collector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "fedsam3_2x2_final_evidence_package.json"
)


def _load_package_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_yaml(relative_path: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def _identity_free(config: dict) -> dict:
    value = deepcopy(config)
    value["seed"] = "<seed>"
    value["federated"]["routing_mode"] = "<routing>"
    value["aggregation"]["unoptimized_update_policy"] = "<routing-policy>"
    value["baseline"] = {"method": "<baseline>", "mu": "<mu>"}
    value["logging"]["log_dir"] = "<log-dir>"
    value["logging"]["experiment_name"] = "<experiment-name>"
    return value


def test_package_declares_complete_unique_3_by_4_matrix():
    config = _load_package_config()
    observed = {(cell["seed"], cell["cell"]) for cell in config["cells"]}
    expected = {
        (seed, cell)
        for seed in config["expected"]["seeds"]
        for cell in config["expected"]["cells"]
    }

    assert observed == expected
    assert len(config["cells"]) == 12
    assert config["checkpoint_selection"]["paper_checkpoint"] == "final_model.pth"
    assert config["checkpoint_selection"]["paper_round"] == 60
    assert "best_model.pth" in config["checkpoint_selection"][
        "excluded_from_primary_comparison"
    ]


def test_all_12_configs_differ_only_in_declared_identity_fields():
    package = _load_package_config()
    configs = [(cell, _load_yaml(cell["config"])) for cell in package["cells"]]
    reference = _identity_free(configs[0][1])

    for cell, config in configs:
        assert _identity_free(config) == reference
        assert config["seed"] == cell["seed"]
        assert config["federated"]["routing_mode"] == (
            "unrestricted" if cell["routing"] == "U" else "restricted"
        )
        assert config["aggregation"]["unoptimized_update_policy"] == (
            "include_zero"
            if cell["routing"] == "U"
            else "exclude_and_renormalize"
        )
        assert config["baseline"] == (
            {"method": "none", "mu": 0.0}
            if cell["aggregation"] == "FedAvg"
            else {"method": "fedprox", "mu": 0.01}
        )


def test_collector_stops_before_checkpoint_payloads(tmp_path):
    prefix = "experiments/logs/test_cell"
    members = {
        "formal_verification/final_metrics.csv": (
            "seed,round,dice\n3407,60,0.5\n".encode("utf-8")
        ),
        "formal_verification/round_metrics.csv": b"round,dice\n1,0.1\n",
        "formal_verification/formal_verification.json": b'{"status":"PASS"}',
        "preflight.log": b"PASS\n",
        "console.log": b"complete\n",
        "verification_console.log": b"PASS\n",
        "parameter_group_diagnostics.csv": b"round,row_type\n1,summary\n",
    }
    archive_path = tmp_path / "cell.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for relative, payload in members.items():
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        checkpoint = b"must-not-be-read"
        info = tarfile.TarInfo(f"{prefix}/checkpoints/final_model.pth")
        info.size = len(checkpoint)
        archive.addfile(info, io.BytesIO(checkpoint))

    artifacts = collector.read_selected_members(
        archive_path,
        prefix,
        {name: relative for name, relative in zip(members, members)},
    )

    assert set(artifacts) == {*members, "_crossed_checkpoint_payloads"}
    assert artifacts["_crossed_checkpoint_payloads"] == b"false"
    rows = list(
        csv.DictReader(io.StringIO(artifacts["formal_verification/final_metrics.csv"].decode()))
    )
    assert rows == [{"seed": "3407", "round": "60", "dice": "0.5"}]


def test_optional_validation_console_after_checkpoint_is_not_required(tmp_path):
    prefix = "experiments/logs/test_cell"
    required_payload = b'{"status":"PASS"}'
    archive_path = tmp_path / "cell_optional_after_checkpoint.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(f"{prefix}/formal_verification/formal_verification.json")
        info.size = len(required_payload)
        archive.addfile(info, io.BytesIO(required_payload))
        checkpoint = b"checkpoint"
        info = tarfile.TarInfo(f"{prefix}/checkpoints/final_model.pth")
        info.size = len(checkpoint)
        archive.addfile(info, io.BytesIO(checkpoint))
        optional_payload = b"PASS\n"
        info = tarfile.TarInfo(f"{prefix}/verification_console.log")
        info.size = len(optional_payload)
        archive.addfile(info, io.BytesIO(optional_payload))

    artifacts = collector.read_selected_members(
        archive_path,
        prefix,
        {"formal_verification_json": "formal_verification/formal_verification.json"},
        {"verification_console_log": "verification_console.log"},
    )

    assert artifacts == {
        "formal_verification_json": required_payload,
        "_crossed_checkpoint_payloads": b"false",
    }


def test_required_training_log_after_checkpoint_is_collected(tmp_path):
    prefix = "experiments/logs/test_cell"
    formal_payload = b'{"status":"PASS"}'
    console_payload = b"training complete\n"
    archive_path = tmp_path / "cell_required_after_checkpoint.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(f"{prefix}/formal_verification/formal_verification.json")
        info.size = len(formal_payload)
        archive.addfile(info, io.BytesIO(formal_payload))
        checkpoint = b"checkpoint-not-extracted"
        info = tarfile.TarInfo(f"{prefix}/checkpoints/final_model.pth")
        info.size = len(checkpoint)
        archive.addfile(info, io.BytesIO(checkpoint))
        info = tarfile.TarInfo(f"{prefix}/console.log")
        info.size = len(console_payload)
        archive.addfile(info, io.BytesIO(console_payload))

    artifacts = collector.read_selected_members(
        archive_path,
        prefix,
        {
            "formal_verification_json": "formal_verification/formal_verification.json",
            "console_log": "console.log",
        },
    )

    assert artifacts == {
        "formal_verification_json": formal_payload,
        "console_log": console_payload,
        "_crossed_checkpoint_payloads": b"true",
    }
