"""Generate seed-level, grouped, paired, and traceability outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRICS = ("dice", "iou", "hd95_mm")
CELL_ORDER = ("U-FedAvg", "U-FedProx", "R-FedAvg", "R-FedProx")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def summarize(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        rows = [row for row in records if row["cell"] == cell]
        if len(rows) != 3:
            raise ValueError(f"Expected three seeds for {cell}, found {len(rows)}")
        result: dict[str, Any] = {
            "cell": cell,
            "routing": rows[0]["routing"],
            "aggregation": rows[0]["aggregation"],
            "n_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_sample_sd"] = statistics.stdev(values)
        summaries.append(result)
    return summaries


def paired_effects(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (int(row["seed"]), row["aggregation"], row["routing"]): row
        for row in records
    }
    effects: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in records}):
        for aggregation in ("FedAvg", "FedProx"):
            unrestricted = lookup[(seed, aggregation, "U")]
            restricted = lookup[(seed, aggregation, "R")]
            row: dict[str, Any] = {"seed": seed, "aggregation": aggregation}
            for metric in METRICS:
                u_value = float(unrestricted[metric])
                r_value = float(restricted[metric])
                row[f"{metric}_u"] = u_value
                row[f"{metric}_r"] = r_value
                row[f"{metric}_r_minus_u"] = r_value - u_value
                row[f"{metric}_improvement"] = (
                    u_value - r_value if metric == "hd95_mm" else r_value - u_value
                )
            effects.append(row)
    return effects


def paper_values(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for metric in METRICS:
            mean = float(summary[f"{metric}_mean"])
            sd = float(summary[f"{metric}_sample_sd"])
            decimals = 1 if metric == "hd95_mm" else 3
            rows.append(
                {
                    "cell": summary["cell"],
                    "metric": metric,
                    "n_seeds": summary["n_seeds"],
                    "mean": mean,
                    "sample_sd": sd,
                    "paper_value": f"{mean:.{decimals}f} ± {sd:.{decimals}f}",
                }
            )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_manifest(root: Path, exclusions: Iterable[Path]) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in exclusions}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def generate(evidence_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    report = _read_json(evidence_root / "evaluation" / "evaluation_report.json")
    if report.get("status") != "PASS" or len(report.get("records", [])) != 12:
        raise ValueError("Results require a passing 12-cell evaluation report")
    records = sorted(
        report["records"], key=lambda row: (int(row["seed"]), CELL_ORDER.index(row["cell"]))
    )
    results_dir = evidence_root / "results"
    if results_dir.exists():
        raise FileExistsError(f"Results output already exists: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=False)

    for record in records:
        cell_dir = evidence_root / record["artifact_directory"]
        round_json = cell_dir / "round_metrics.json"
        final_json = cell_dir / "final_metrics.json"
        if round_json.exists() or final_json.exists():
            raise FileExistsError(f"Generated metric JSON already exists: {cell_dir}")
        _write_json(round_json, _read_csv(cell_dir / "round_metrics.csv"))
        final_rows = _read_csv(cell_dir / "final_metrics.csv")
        if len(final_rows) != 1:
            raise ValueError(f"Expected one final metric row: {cell_dir}")
        _write_json(final_json, final_rows[0])

    summaries = summarize(records)
    effects = paired_effects(records)
    formatted = paper_values(summaries)
    outputs = {
        "seed_level_metrics": records,
        "group_summary": summaries,
        "paired_routing_effects": effects,
        "paper_values": formatted,
    }
    for name, rows in outputs.items():
        _write_csv(results_dir / f"{name}.csv", rows)
        _write_json(results_dir / f"{name}.json", rows)

    checkpoint = report["checkpoint_selection"]
    readme = [
        "# FedSAM3 formal 2x2 evidence package",
        "",
        "This package contains 12 completed cells: four routing/FedProx settings across seeds 3407, 3408, and 3409.",
        "",
        "## Checkpoint rule",
        "",
        f"Primary paper checkpoint: `{checkpoint['paper_checkpoint']}` at round {checkpoint['paper_round']}.",
        "`best_model.pth` is excluded from the primary comparison. The original per-cell verifier checked the final model against the round-60 and latest checkpoints and re-evaluated it on the complete validation/test contract.",
        "",
        "## Reproduction",
        "",
        "1. Read each cell's `final_metrics.csv` for seed-level paper values.",
        "2. Recompute group means and sample standard deviations from the three seed rows.",
        "3. Use `round_metrics.csv` or `round_metrics.json` for round-level curves.",
        "4. Use `formal_verification.json` to trace each value to the re-evaluated final checkpoint and hashes.",
        "",
    ]
    (results_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    manifest_csv = results_dir / "file_manifest.csv"
    manifest_json = results_dir / "file_manifest.json"
    files = _file_manifest(evidence_root, (manifest_csv, manifest_json))
    _write_csv(manifest_csv, files)
    _write_json(manifest_json, files)
    result = {
        "status": "GENERATED",
        "evidence_root": str(evidence_root),
        "seed_records": len(records),
        "group_rows": len(summaries),
        "paired_rows": len(effects),
        "paper_value_rows": len(formatted),
        "manifest_entries": len(files),
    }
    _write_json(results_dir / "generation_status.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate final 2x2 tables and traceability outputs"
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = generate(args.evidence_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
