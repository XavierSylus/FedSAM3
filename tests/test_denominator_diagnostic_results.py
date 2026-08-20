import csv
import json

import pytest

from data_processing.generate_denominator_diagnostic_results import generate


def _write_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_cell(root, label, metrics, eligible_clients, weights):
    final_metrics_path = root / "formal_verification" / "final_metrics.csv"
    _write_csv(
        final_metrics_path,
        {
            "training_git_commit": f"commit-{label}",
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "hd95_mm": metrics["hd95_mm"],
        },
    )
    history_path = root / "checkpoints" / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {
                "aggregation_audits": [
                    {
                        "parameters": {
                            "text_proj.weight": {
                                "parameter_group": "TEXT_PARAMS",
                                "uploaded_client_ids": ["client_1", "client_3"],
                                "eligible_client_ids": eligible_clients,
                                "normalized_weights": weights,
                            }
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_generate_writes_u_n_r_effects_and_group_weights(tmp_path):
    cells = {
        "U": ({"dice": 0.60, "iou": 0.50, "hd95_mm": 40.0}, ["client_1", "client_2", "client_3"], {"client_1": 0.2, "client_2": 0.3, "client_3": 0.5}),
        "N": ({"dice": 0.70, "iou": 0.60, "hd95_mm": 30.0}, ["client_1", "client_3"], {"client_1": 0.3, "client_3": 0.7}),
        "R": ({"dice": 0.70, "iou": 0.60, "hd95_mm": 30.0}, ["client_1", "client_3"], {"client_1": 0.3, "client_3": 0.7}),
    }
    manifest_cells = []
    for label, (metrics, eligible, weights) in cells.items():
        log_dir = tmp_path / label
        _write_cell(log_dir, label, metrics, eligible, weights)
        manifest_cells.append(
            {
                "label": label,
                "routing_mode": {
                    "U": "unrestricted",
                    "N": "uploader_renormalized",
                    "R": "restricted",
                }[label],
                "log_dir": str(log_dir),
            }
        )

    output_dir = tmp_path / "comparison"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": 3407,
                "cells": manifest_cells,
                "expected_rounds": 1,
                "results_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    result = generate(manifest_path)

    assert result["status"] == "GENERATED"
    with (output_dir / "final_metrics_u_n_r.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    dice = next(row for row in rows if row["metric"] == "dice")
    hd95 = next(row for row in rows if row["metric"] == "hd95_mm")
    assert float(dice["n_minus_u"]) == pytest.approx(0.1)
    assert float(dice["r_minus_n"]) == 0.0
    assert float(hd95["n_minus_u"]) == -10.0
    assert float(hd95["r_minus_n"]) == 0.0

    weights = json.loads(
        (output_dir / "parameter_group_weights_final_round.json").read_text(
            encoding="utf-8"
        )
    )
    n_row = next(row for row in weights if row["label"] == "N")
    assert n_row["eligible_client_ids"] == ["client_1", "client_3"]
    assert n_row["normalized_weights"] == {"client_1": 0.3, "client_3": 0.7}
