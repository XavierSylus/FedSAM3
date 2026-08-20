import csv
import io
import json
import tarfile
from pathlib import Path

import pytest

from data_processing.export_aggregation_audit import export_aggregation_audit


def _write_archive(path: Path, members: dict[str, dict]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            data = json.dumps(payload).encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _parameter_entry(
    *,
    group: str,
    uploaded: list[str],
    eligible: list[str],
    zero_update: list[str],
    sample_weights: dict[str, int],
    normalized_weights: dict[str, float],
    empty: bool,
) -> dict:
    return {
        "parameter_group": group,
        "uploaded_client_ids": uploaded,
        "eligible_client_ids": eligible,
        "zero_update_client_ids": zero_update,
        "sample_weights": sample_weights,
        "normalized_weights": normalized_weights,
        "empty_eligible": empty,
    }


def _history(commit: str, routing_mode: str, parameter_name: str, entry: dict) -> dict:
    return {
        "run_metadata": {"seed": 3407, "git_commit": commit},
        "rounds": [1],
        "aggregation_audits": [
            {
                "round": 1,
                "aggregation_method": "fedavg",
                "routing_mode": routing_mode,
                "active_client_ids": ["client_1", "client_2", "client_3"],
                "client_sample_counts": {
                    "client_1": 1,
                    "client_2": 2,
                    "client_3": 3,
                },
                "parameters": {parameter_name: entry},
            }
        ],
    }


def test_export_separates_missing_upload_from_uploaded_numerical_zero(tmp_path: Path):
    commit = "a" * 40
    verification_u = {
        "status": "PASS",
        "training_git_commit": commit,
        "configuration": {"seed": 3407, "routing_mode": "unrestricted"},
    }
    verification_r = {
        "status": "PASS",
        "training_git_commit": commit,
        "configuration": {"seed": 3407, "routing_mode": "restricted"},
    }
    u_history = _history(
        commit,
        "unrestricted",
        "text_proj.weight",
        _parameter_entry(
            group="TEXT_PARAMS",
            uploaded=["client_1", "client_3"],
            eligible=["client_1", "client_2", "client_3"],
            zero_update=["client_2", "client_3"],
            sample_weights={"client_1": 1, "client_2": 2, "client_3": 3},
            normalized_weights={
                "client_1": 1 / 6,
                "client_2": 2 / 6,
                "client_3": 3 / 6,
            },
            empty=False,
        ),
    )
    r_history = _history(
        commit,
        "restricted",
        "fusion_head._fusion_gate.weight",
        _parameter_entry(
            group="FUSION_PARAMS",
            uploaded=["client_1"],
            eligible=[],
            zero_update=[],
            sample_weights={},
            normalized_weights={},
            empty=True,
        ),
    )

    u_archive = tmp_path / "u.tar.gz"
    r_archive = tmp_path / "r.tar.gz"
    _write_archive(
        u_archive,
        {"u/formal.json": verification_u, "u/history.json": u_history},
    )
    _write_archive(
        r_archive,
        {"r/formal.json": verification_r, "r/history.json": r_history},
    )
    output_dir = tmp_path / "audit"
    config = {
        "schema_version": 1,
        "seed": 3407,
        "expected_training_git_commit": commit,
        "archive_root": str(tmp_path),
        "clients": {
            "client_1": "text_only",
            "client_2": "image_only",
            "client_3": "multimodal",
        },
        "sources": [
            {
                "cell": "U-FedAvg",
                "archive": u_archive.name,
                "formal_verification_member": "u/formal.json",
                "training_history_member": "u/history.json",
            },
            {
                "cell": "R-FedAvg",
                "archive": r_archive.name,
                "formal_verification_member": "r/formal.json",
                "training_history_member": "r/history.json",
            },
        ],
        "output": {
            "directory": str(output_dir),
            "csv": "aggregation_audit.csv",
            "jsonl": "aggregation_audit.jsonl",
            "manifest": "manifest.json",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = export_aggregation_audit(config_path)

    with result.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    u_rows = {row["client_id"]: row for row in rows if row["cell"] == "U-FedAvg"}
    assert u_rows["client_2"]["actual_uploader"] == "false"
    assert u_rows["client_2"]["denominator_member"] == "true"
    assert u_rows["client_2"]["zero_update_reason"] == "missing_upload_eq4"
    assert u_rows["client_3"]["actual_uploader"] == "true"
    assert u_rows["client_3"]["zero_update_reason"] == "uploaded_numerical_zero"
    assert float(u_rows["client_2"]["normalized_weight"]) == pytest.approx(2 / 6)
    assert u_rows["client_1"]["formula_branch"] == "eq5_unrestricted"

    r_rows = [row for row in rows if row["cell"] == "R-FedAvg"]
    assert all(row["empty_eligible"] == "true" for row in r_rows)
    assert all(row["denominator_member"] == "false" for row in r_rows)
    assert all(row["formula_branch"] == "eq7_empty_preserve_global" for row in r_rows)

    jsonl_records = [
        json.loads(line)
        for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    u_record = next(record for record in jsonl_records if record["cell"] == "U-FedAvg")
    assert u_record["actual_uploaders"] == ["client_1", "client_3"]
    assert u_record["eligible_uploaders"] == ["client_1", "client_3"]
    assert u_record["missing_upload_zero_update_clients"] == ["client_2"]
    assert u_record["numerical_zero_delta_uploaders"] == ["client_3"]
    assert u_record["denominator_sample_count_total"] == 6

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["formula_validation"]["status"] == "PASS"
    assert manifest["parameter_record_count"] == 2
    assert manifest["csv_row_count"] == 6


def test_export_rejects_unrestricted_missing_upload_not_marked_zero(tmp_path: Path):
    commit = "b" * 40
    history = _history(
        commit,
        "unrestricted",
        "text_proj.weight",
        _parameter_entry(
            group="TEXT_PARAMS",
            uploaded=["client_1"],
            eligible=["client_1", "client_2", "client_3"],
            zero_update=[],
            sample_weights={"client_1": 1, "client_2": 2, "client_3": 3},
            normalized_weights={
                "client_1": 1 / 6,
                "client_2": 2 / 6,
                "client_3": 3 / 6,
            },
            empty=False,
        ),
    )
    archive = tmp_path / "invalid.tar.gz"
    _write_archive(
        archive,
        {
            "formal.json": {
                "status": "PASS",
                "training_git_commit": commit,
                "configuration": {"seed": 3407, "routing_mode": "unrestricted"},
            },
            "history.json": history,
        },
    )
    config = {
        "schema_version": 1,
        "seed": 3407,
        "expected_training_git_commit": commit,
        "archive_root": str(tmp_path),
        "clients": {
            "client_1": "text_only",
            "client_2": "image_only",
            "client_3": "multimodal",
        },
        "sources": [
            {
                "cell": "U-FedAvg",
                "archive": archive.name,
                "formal_verification_member": "formal.json",
                "training_history_member": "history.json",
            }
        ],
        "output": {
            "directory": str(tmp_path / "audit"),
            "csv": "aggregation_audit.csv",
            "jsonl": "aggregation_audit.jsonl",
            "manifest": "manifest.json",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="missing-upload client is not a zero update"):
        export_aggregation_audit(config_path)
