"""Generate the declared single-seed U/N/R denominator diagnostic tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


METRICS = ("dice", "iou", "hd95_mm")
LABELS = ("U", "N", "R")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one CSV row: {path}")
    return rows[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _final_group_rows(
    *,
    label: str,
    routing_mode: str,
    history: Mapping[str, Any],
    expected_rounds: int,
) -> list[dict[str, Any]]:
    audits = history.get("aggregation_audits")
    if not isinstance(audits, list) or len(audits) != expected_rounds:
        raise ValueError(
            f"{label} must contain exactly {expected_rounds} aggregation audits"
        )
    final_parameters = audits[-1].get("parameters")
    if not isinstance(final_parameters, dict) or not final_parameters:
        raise ValueError(f"{label} final aggregation audit has no parameters")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for parameter_name, entry in final_parameters.items():
        if not isinstance(entry, Mapping):
            raise TypeError(f"Invalid aggregation audit entry: {parameter_name}")
        parameter_group = entry.get("parameter_group")
        if not isinstance(parameter_group, str) or not parameter_group:
            raise ValueError(f"Missing parameter group: {parameter_name}")
        grouped.setdefault(parameter_group, []).append(entry)

    rows = []
    for parameter_group, entries in sorted(grouped.items()):
        signatures = {
            (
                tuple(entry.get("uploaded_client_ids", [])),
                tuple(entry.get("eligible_client_ids", [])),
                tuple(sorted(entry.get("normalized_weights", {}).items())),
            )
            for entry in entries
        }
        if len(signatures) != 1:
            raise ValueError(
                f"{label} has inconsistent final weights within {parameter_group}"
            )
        uploaded, eligible, weights = next(iter(signatures))
        rows.append(
            {
                "label": label,
                "routing_mode": routing_mode,
                "final_round": expected_rounds,
                "parameter_group": parameter_group,
                "parameter_count": len(entries),
                "uploaded_client_ids": list(uploaded),
                "eligible_client_ids": list(eligible),
                "normalized_weights": dict(weights),
            }
        )
    return rows


def _csv_group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "uploaded_client_ids": json.dumps(
                row["uploaded_client_ids"], ensure_ascii=False
            ),
            "eligible_client_ids": json.dumps(
                row["eligible_client_ids"], ensure_ascii=False
            ),
            "normalized_weights": json.dumps(
                row["normalized_weights"], ensure_ascii=False, sort_keys=True
            ),
        }
        for row in rows
    ]


def generate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    cells = manifest.get("cells")
    if not isinstance(cells, list) or [cell.get("label") for cell in cells] != list(LABELS):
        raise ValueError("Diagnostic manifest cells must be ordered U, N, R")
    expected_rounds = manifest.get("fixed_controls", {}).get(
        "rounds", manifest.get("expected_rounds")
    )
    if not isinstance(expected_rounds, int) or expected_rounds <= 0:
        raise ValueError("Diagnostic manifest must declare a positive round count")

    metrics_by_label: dict[str, dict[str, float]] = {}
    group_rows: list[dict[str, Any]] = []
    sources = []
    for cell in cells:
        label = cell["label"]
        log_dir = Path(cell["log_dir"])
        metrics_path = log_dir / "formal_verification" / "final_metrics.csv"
        history_path = log_dir / "checkpoints" / "training_history.json"
        metrics_row = _read_single_csv_row(metrics_path)
        history = _read_json(history_path)
        metrics_by_label[label] = {
            metric: float(metrics_row[metric]) for metric in METRICS
        }
        group_rows.extend(
            _final_group_rows(
                label=label,
                routing_mode=cell["routing_mode"],
                history=history,
                expected_rounds=expected_rounds,
            )
        )
        sources.append(
            {
                "label": label,
                "routing_mode": cell["routing_mode"],
                "training_git_commit": metrics_row.get("training_git_commit"),
                "final_metrics_csv": str(metrics_path),
                "final_metrics_csv_sha256": _sha256(metrics_path),
                "training_history_json": str(history_path),
                "training_history_json_sha256": _sha256(history_path),
            }
        )

    metric_rows = []
    for metric in METRICS:
        u_value = metrics_by_label["U"][metric]
        n_value = metrics_by_label["N"][metric]
        r_value = metrics_by_label["R"][metric]
        metric_rows.append(
            {
                "seed": manifest["seed"],
                "metric": metric,
                "U": u_value,
                "N": n_value,
                "R": r_value,
                "n_minus_u": n_value - u_value,
                "r_minus_n": r_value - n_value,
                "r_minus_u": r_value - u_value,
            }
        )

    output_dir = Path(manifest["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_csv = output_dir / "final_metrics_u_n_r.csv"
    metrics_json = output_dir / "final_metrics_u_n_r.json"
    weights_csv = output_dir / "parameter_group_weights_final_round.csv"
    weights_json = output_dir / "parameter_group_weights_final_round.json"
    _write_csv(metrics_csv, metric_rows)
    _write_json(metrics_json, metric_rows)
    _write_csv(weights_csv, _csv_group_rows(group_rows))
    _write_json(weights_json, group_rows)

    result_manifest_path = output_dir / "result_manifest.json"
    output_records = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in (metrics_csv, metrics_json, weights_csv, weights_json)
    ]
    result = {
        "status": "GENERATED",
        "seed": manifest["seed"],
        "expected_rounds": expected_rounds,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "sources": sources,
        "outputs": output_records,
    }
    _write_json(result_manifest_path, result)
    return {**result, "result_manifest": str(result_manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the seed-level U/N/R denominator diagnostic tables"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
