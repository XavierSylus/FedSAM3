import csv
import hashlib
import json
from pathlib import Path

import pytest

from data_processing import evaluate_fedsam3_2x2_evidence as evaluator
from data_processing import generate_fedsam3_2x2_results as generator


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_cell(tmp_path: Path) -> tuple[dict, dict]:
    cell_dir = tmp_path / "seed_3407" / "u_fedavg"
    cell_dir.mkdir(parents=True)
    config_path = cell_dir / "config.yaml"
    config_path.write_text("seed: 3407\n", encoding="utf-8")
    config_sha = _sha(config_path)
    data_sha = "d" * 64
    model_sha = "m" * 64
    final = {
        "experiment_name": "test",
        "seed": 3407,
        "round": 60,
        "routing_mode": "unrestricted",
        "baseline_method": "none",
        "fedprox_mu": 0.0,
        "training_git_commit": "commit",
        "config_file_sha256": config_sha,
        "data_manifest_sha256": data_sha,
        "final_model_sha256": model_sha,
        "dice": 0.7,
        "iou": 0.6,
        "hd95_mm": 20.0,
        "WT_dice": 0.71,
        "TC_dice": 0.70,
        "ET_dice": 0.69,
        "WT_hd95_mm": 19.0,
        "TC_hd95_mm": 20.0,
        "ET_hd95_mm": 21.0,
        "num_cases": 32,
        "hd95_unit": "mm",
        "hd95_dimension": "3d_case",
    }
    _write_csv(cell_dir / "final_metrics.csv", [final])
    rounds = [
        {
            "round": round_num,
            "dice": 0.7 if round_num == 60 else 0.1,
            "iou": 0.6 if round_num == 60 else 0.1,
            "hd95_mm": 20.0 if round_num == 60 else 50.0,
        }
        for round_num in range(1, 61)
    ]
    _write_csv(cell_dir / "round_metrics.csv", rounds)
    metrics = {
        "dice": 0.7,
        "iou": 0.6,
        "hd95": 20.0,
        "WT_dice": 0.71,
        "TC_dice": 0.70,
        "ET_dice": 0.69,
        "WT_hd95": 19.0,
        "TC_hd95": 20.0,
        "ET_hd95": 21.0,
        "num_cases": 32,
    }
    verification = {
        "status": "PASS",
        "metrics_match": True,
        "config_file_sha256": config_sha,
        "data_manifest_sha256": data_sha,
        "evaluation_contract": {"checkpoint": "final_model.pth", "round": 60},
        "checkpoint_audit": {"final_model_sha256": model_sha},
        "historical_final_metrics": metrics,
        "reevaluated_final_metrics": metrics,
    }
    (cell_dir / "formal_verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    for filename in (
        "preflight.log",
        "console.log",
        "verification_console.log",
        "parameter_group_diagnostics.csv",
        "experiment_manifest.json",
    ):
        (cell_dir / filename).write_text("evidence\n", encoding="utf-8")

    artifacts = {}
    for path in cell_dir.iterdir():
        artifacts[path.stem] = {
            "path": str(path.relative_to(tmp_path)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    record = {
        "seed": 3407,
        "cell": "U-FedAvg",
        "routing": "U",
        "aggregation": "FedAvg",
        "artifact_directory": "seed_3407/u_fedavg",
        "archive": "cell.tar.gz",
        "archive_path": "cell.tar.gz",
        "artifacts": artifacts,
    }
    package = {
        "expected": {
            "round": 60,
            "num_cases": 32,
            "data_manifest_sha256": data_sha,
            "hd95_unit": "mm",
            "hd95_dimension": "3d_case",
        },
        "checkpoint_selection": {
            "paper_checkpoint": "final_model.pth",
            "paper_round": 60,
        },
    }
    return package, record


def test_evaluator_matches_csv_to_original_gpu_reevaluation(tmp_path):
    package, record = _synthetic_cell(tmp_path)

    result = evaluator.evaluate_cell(package, tmp_path, record)

    assert result["status"] == "PASS"
    assert result["dice"] == 0.7
    assert result["final_model_sha256"] == "m" * 64


def test_evaluator_rejects_a_changed_paper_metric(tmp_path):
    package, record = _synthetic_cell(tmp_path)
    final_path = tmp_path / record["artifact_directory"] / "final_metrics.csv"
    rows = evaluator._read_csv(final_path)
    rows[0]["dice"] = "0.8"
    _write_csv(final_path, rows)
    record["artifacts"]["final_metrics"]["bytes"] = final_path.stat().st_size
    record["artifacts"]["final_metrics"]["sha256"] = _sha(final_path)

    with pytest.raises(ValueError, match="final metric dice mismatch"):
        evaluator.evaluate_cell(package, tmp_path, record)


def test_summary_uses_sample_sd_and_paired_hd95_direction():
    records = []
    values = {
        "U-FedAvg": (0.5, 0.4, 50.0),
        "U-FedProx": (0.6, 0.5, 40.0),
        "R-FedAvg": (0.7, 0.6, 30.0),
        "R-FedProx": (0.8, 0.7, 20.0),
    }
    for seed, offset in ((3407, 0.0), (3408, 0.1), (3409, 0.2)):
        for cell, (dice, iou, hd95) in values.items():
            routing, aggregation = cell.split("-")
            records.append(
                {
                    "seed": seed,
                    "cell": cell,
                    "routing": routing,
                    "aggregation": aggregation,
                    "dice": dice + offset,
                    "iou": iou + offset,
                    "hd95_mm": hd95 + offset,
                }
            )

    summaries = generator.summarize(records)
    effects = generator.paired_effects(records)

    assert summaries[0]["dice_mean"] == pytest.approx(0.6)
    assert summaries[0]["dice_sample_sd"] == pytest.approx(0.1)
    assert effects[0]["dice_improvement"] == pytest.approx(0.2)
    assert effects[0]["hd95_mm_improvement"] == pytest.approx(20.0)


def test_file_manifest_includes_status_but_excludes_manifest_itself(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    status_path = results_dir / "generation_status.json"
    status_path.write_text('{"status":"GENERATED"}\n', encoding="utf-8")
    manifest_csv = results_dir / "file_manifest.csv"
    manifest_json = results_dir / "file_manifest.json"

    rows = generator._file_manifest(tmp_path, (manifest_csv, manifest_json))

    assert [row["path"] for row in rows] == ["results/generation_status.json"]
