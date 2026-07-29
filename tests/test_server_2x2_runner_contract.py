import ast
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "fedsam3_experiment_manifest.json"
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
