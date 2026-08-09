import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import server_run_2x2_matrix as matrix_runner
from scripts import server_verify_formal_cell as formal_verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "fedsam3_experiment_manifest.json"
SEED_3408_MANIFEST_PATH = (
    PROJECT_ROOT / "configs" / "fedsam3_experiment_manifest_seed3408.json"
)
RUNNER_PATH = PROJECT_ROOT / "scripts" / "server_run_2x2_matrix.py"
EXPECTED_CELLS = [
    "U-FedAvg",
    "U-FedProx",
    "R-FedAvg",
    "R-FedProx",
]


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_formal_matrix_runner_uses_the_declared_four_cell_order():
    manifest = _load_manifest()
    matrix = manifest["matrix"]

    assert [entry["cell"] for entry in matrix] == EXPECTED_CELLS
    for entry in matrix:
        config_path = PROJECT_ROOT / entry["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["training"]["rounds"] == 60
        assert config["federated"]["routing_mode"] == entry["routing_mode"]
        assert (
            config["aggregation"]["unoptimized_update_policy"]
            == entry["unoptimized_update_policy"]
        )
        assert config["baseline"]["method"] == entry["baseline_method"]
        assert config["baseline"]["mu"] == entry["fedprox_mu"]


def test_seed_3408_manifest_uses_isolated_four_cell_outputs():
    manifest = json.loads(SEED_3408_MANIFEST_PATH.read_text(encoding="utf-8"))
    matrix = manifest["matrix"]

    assert manifest["seed"] == 3408
    assert [entry["cell"] for entry in matrix] == EXPECTED_CELLS

    log_dirs = []
    for entry in matrix:
        config_path = PROJECT_ROOT / entry["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["seed"] == 3408
        assert config["training"]["rounds"] == 60
        assert config["federated"]["routing_mode"] == entry["routing_mode"]
        assert (
            config["aggregation"]["unoptimized_update_policy"]
            == entry["unoptimized_update_policy"]
        )
        assert config["baseline"]["method"] == entry["baseline_method"]
        assert config["baseline"]["mu"] == entry["fedprox_mu"]
        log_dirs.append(config["logging"]["log_dir"])

    assert len(set(log_dirs)) == 4
    assert {
        log_dir.rsplit("/", maxsplit=1)[0] for log_dir in log_dirs
    } == {
        "/root/autodl-tmp/FedSAM3-Cream/experiments/logs/"
        "fedsam3_2x2_seed3408"
    }


def test_formal_matrix_runner_persists_and_audits_submission_evidence():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(RUNNER_PATH))

    for required_text in (
        "fedsam3_experiment_manifest.json",
        'manifest["matrix"]',
        "sys.executable",
        '"-m"',
        '"pytest"',
        "server_preflight.py",
        "main.py",
        "subprocess.run",
        "check=True",
        "supervisor.log",
        "runner_result.json",
        "installed_packages.json",
        "final_model.pth",
        "latest_checkpoint.pth",
        "training_history.json",
        "run_metadata.json",
        "final_val_metrics",
    ):
        assert required_text in source

    for forbidden_text in (
        "/autodl-fs/",
        "torch.randn",
        "--data_root",
        "--rounds",
        "--log_dir",
        ".unlink(",
        "os.remove",
        "shutil.rmtree",
    ):
        assert forbidden_text not in source


def test_formal_matrix_runner_can_select_exactly_one_declared_cell():
    cells = [{"cell": cell} for cell in EXPECTED_CELLS]

    selected = matrix_runner._select_cells(cells, "R-FedProx")

    assert [cell["cell"] for cell in selected] == ["R-FedProx"]
    with pytest.raises(ValueError, match="Unknown formal cell"):
        matrix_runner._select_cells(cells, "not-a-cell")


def test_formal_matrix_runner_accepts_explicit_repo_manifest():
    manifest_relative = Path("configs/fedsam3_experiment_manifest.json")
    arguments = matrix_runner.build_parser().parse_args(
        ["--manifest", str(manifest_relative), "--cell", "U-FedAvg"]
    )

    assert arguments.manifest == manifest_relative
    assert (
        matrix_runner._resolve_manifest_path(arguments.manifest)
        == MANIFEST_PATH.resolve()
    )


def test_formal_verifier_rejects_incomplete_round_sequence():
    with pytest.raises(ValueError, match="exact rounds 1..60"):
        formal_verifier.require_exact_rounds(
            "training_history.rounds",
            list(range(1, 60)),
            expected_rounds=60,
        )


def test_formal_verifier_rejects_duplicate_diagnostic_csv():
    diagnostics = {
        "client_drift": {
            "client_1": {
                "TEXT_PARAMS": {
                    "update_l2": 1.0,
                    "reference_l2": 2.0,
                }
            }
        },
        "pairwise_conflicts": [],
        "conflict_summary": {},
        "global_drift": {},
    }
    expected_rows = formal_verifier.serialized_diagnostic_rows(
        [1],
        [diagnostics],
    )

    with pytest.raises(ValueError, match="diagnostic CSV does not exactly match"):
        formal_verifier.assert_diagnostic_export_records(
            rounds=[1],
            diagnostics=[diagnostics],
            jsonl_records=[{"round": 1, **diagnostics}],
            csv_fieldnames=formal_verifier.DIAGNOSTIC_FIELDNAMES,
            csv_rows=expected_rows + expected_rows,
        )


def test_formal_verifier_uses_the_training_validation_contract():
    verifier_path = PROJECT_ROOT / "scripts" / "server_verify_formal_cell.py"
    source = verifier_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(verifier_path))

    for required_text in (
        "final_model.pth",
        "checkpoint_round_60.pth",
        "latest_checkpoint.pth",
        "FederatedTrainer",
        "setup_environment",
        "setup_clients",
        "setup_validation",
        "compute_hd95=True",
        '"hd95_unit": "mm"',
        '"hd95_dimension": "3d_case"',
        "formal_verification.json",
        "final_metrics.csv",
        "round_metrics.csv",
    ):
        assert required_text in source

    for forbidden_text in (
        "evaluate_model",
        "best_model.pth",
        "--data_root",
        "--log_dir",
    ):
        assert forbidden_text not in source


def test_formal_verifier_direct_entry_bootstraps_project_root():
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import runpy; "
                "runpy.run_path('server_verify_formal_cell.py', "
                "run_name='formal_verifier_probe'); "
                "import src"
            ),
        ],
        cwd=PROJECT_ROOT / "scripts",
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_round_metrics_csv_has_one_row_per_validation_round():
    metric = {
        "dice": 0.7,
        "iou": 0.6,
        "hd95": 20.0,
        "WT_dice": 0.71,
        "TC_dice": 0.70,
        "ET_dice": 0.69,
        "WT_iou": 0.61,
        "TC_iou": 0.60,
        "ET_iou": 0.59,
        "WT_hd95": 19.0,
        "TC_hd95": 20.0,
        "ET_hd95": 21.0,
        "num_cases": 32,
    }
    history = {
        "rounds": [1, 2],
        "avg_losses": [0.9, 0.8],
        "avg_seg_losses": [0.8, 0.7],
        "avg_cream_losses": [0.1, 0.1],
        "lr_history": [1e-4, 9e-5],
        "gpu_mem_mb": [1000, 1000],
        "round_time_sec": [10, 11],
        "grad_conflict_deg": [90.0, 89.0],
        "val_metrics": [
            {"round": 1, **metric},
            {"round": 2, **metric},
        ],
    }

    rows = formal_verifier._build_round_metric_rows(history)

    assert [row["round"] for row in rows] == [1, 2]
    assert [row["hd95_mm"] for row in rows] == [20.0, 20.0]
