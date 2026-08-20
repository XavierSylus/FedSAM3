"""Evaluate the integrity and metric consistency of the formal 2x2 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "fedsam3_2x2_final_evidence_package.json"
)
METRIC_MAP = {
    "dice": "dice",
    "iou": "iou",
    "hd95_mm": "hd95",
    "WT_dice": "WT_dice",
    "TC_dice": "TC_dice",
    "ET_dice": "ET_dice",
    "WT_hd95_mm": "WT_hd95",
    "TC_hd95_mm": "TC_hd95",
    "ET_hd95_mm": "ET_hd95",
    "num_cases": "num_cases",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _require_float_equal(label: str, actual: Any, expected: Any) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def evaluate_cell(
    package: Mapping[str, Any],
    evidence_root: Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    expected = package["expected"]
    artifact_dir = evidence_root / record["artifact_directory"]
    artifacts = record["artifacts"]
    for logical_name, artifact in artifacts.items():
        path = evidence_root / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {logical_name}: {path}")
        _require_equal(f"{logical_name} size", path.stat().st_size, artifact["bytes"])
        _require_equal(f"{logical_name} SHA256", _sha256(path), artifact["sha256"])

    final_path = artifact_dir / "final_metrics.csv"
    round_path = artifact_dir / "round_metrics.csv"
    verification_path = artifact_dir / "formal_verification.json"
    config_path = artifact_dir / "config.yaml"
    final_rows = _read_csv(final_path)
    if len(final_rows) != 1:
        raise ValueError(f"Expected one final metric row: {final_path}")
    final = final_rows[0]
    round_rows = _read_csv(round_path)
    verification = _read_json(verification_path)

    _require_equal("verification status", verification.get("status"), "PASS")
    _require_equal("metrics_match", verification.get("metrics_match"), True)
    _require_equal(
        "historical/reevaluated metrics",
        verification.get("historical_final_metrics"),
        verification.get("reevaluated_final_metrics"),
    )
    contract = verification["evaluation_contract"]
    selection = package["checkpoint_selection"]
    _require_equal("checkpoint", contract["checkpoint"], selection["paper_checkpoint"])
    _require_equal("checkpoint round", int(contract["round"]), selection["paper_round"])

    _require_equal("seed", int(final["seed"]), int(record["seed"]))
    _require_equal("round", int(final["round"]), int(expected["round"]))
    _require_equal(
        "routing mode",
        final["routing_mode"],
        "unrestricted" if record["routing"] == "U" else "restricted",
    )
    _require_equal(
        "baseline method",
        final["baseline_method"],
        "none" if record["aggregation"] == "FedAvg" else "fedprox",
    )
    _require_float_equal(
        "FedProx mu",
        final["fedprox_mu"],
        0.0 if record["aggregation"] == "FedAvg" else 0.01,
    )
    _require_equal("number of cases", int(final["num_cases"]), expected["num_cases"])
    _require_equal("HD95 unit", final["hd95_unit"], expected["hd95_unit"])
    _require_equal(
        "HD95 dimension", final["hd95_dimension"], expected["hd95_dimension"]
    )
    _require_equal(
        "data manifest SHA256",
        final["data_manifest_sha256"],
        expected["data_manifest_sha256"],
    )
    _require_equal("config SHA256", _sha256(config_path), final["config_file_sha256"])
    _require_equal(
        "verification config SHA256",
        verification["config_file_sha256"],
        final["config_file_sha256"],
    )
    _require_equal(
        "verification data SHA256",
        verification["data_manifest_sha256"],
        final["data_manifest_sha256"],
    )
    _require_equal(
        "final model SHA256",
        verification["checkpoint_audit"]["final_model_sha256"],
        final["final_model_sha256"],
    )

    reevaluated = verification["reevaluated_final_metrics"]
    for csv_name, json_name in METRIC_MAP.items():
        _require_float_equal(
            f"final metric {csv_name}", final[csv_name], reevaluated[json_name]
        )

    expected_rounds = list(range(1, int(expected["round"]) + 1))
    actual_rounds = [int(row["round"]) for row in round_rows]
    _require_equal("round metric sequence", actual_rounds, expected_rounds)
    last_round = round_rows[-1]
    for metric in ("dice", "iou", "hd95_mm"):
        _require_float_equal(f"round-60 {metric}", last_round[metric], final[metric])

    return {
        "seed": int(record["seed"]),
        "cell": record["cell"],
        "routing": record["routing"],
        "aggregation": record["aggregation"],
        "status": "PASS",
        "round": int(final["round"]),
        "num_cases": int(final["num_cases"]),
        "training_git_commit": final["training_git_commit"],
        "config_file_sha256": final["config_file_sha256"],
        "data_manifest_sha256": final["data_manifest_sha256"],
        "final_model_sha256": final["final_model_sha256"],
        "dice": float(final["dice"]),
        "iou": float(final["iou"]),
        "hd95_mm": float(final["hd95_mm"]),
        "artifact_directory": record["artifact_directory"],
        "archive": record["archive"],
        "archive_path": record["archive_path"],
    }


def evaluate(
    package_path: Path,
    evidence_root: Path,
    output_override: Path | None = None,
) -> dict[str, Any]:
    package = _read_json(package_path.resolve())
    evidence_root = evidence_root.resolve()
    collection = _read_json(evidence_root / "collection_manifest.json")
    records = [evaluate_cell(package, evidence_root, row) for row in collection["records"]]
    if len(records) != 12 or any(row["status"] != "PASS" for row in records):
        raise ValueError("The unified evaluation requires 12 passing cells")

    output_dir = output_override or evidence_root / "evaluation"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "evaluation_type": "audit of completed per-cell GPU reevaluation evidence",
        "inference_performed_by_this_script": False,
        "checkpoint_selection": package["checkpoint_selection"],
        "expected": package["expected"],
        "records": records,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "seed_cell_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all 12 completed formal 2x2 evidence cells"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate(args.config, args.evidence_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
