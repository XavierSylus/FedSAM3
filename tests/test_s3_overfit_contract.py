import ast
from pathlib import Path, PurePosixPath

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "fedsam3_s3_image_only_overfit.yaml"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "server_s3_image_only_overfit.py"
DATA_ROOT = "/autodl-fs/data/FedSAM3-Cream/datasets/federated_split"
LOG_DIR = (
    "/autodl-fs/data/FedSAM3-Cream/experiments/logs/"
    "server_s3_image_only_overfit"
)


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_s3_uses_one_real_image_client_and_an_isolated_log_dir():
    config = _load_config()

    assert config["data_root"] == DATA_ROOT
    assert config["max_samples"] == 1
    assert config["federated"]["clients"] == [
        {
            "client_id": "client_2",
            "modality": "image_only",
            "data_source": (
                f"{DATA_ROOT}/client2_image_only/dataset.json"
            ),
            "enabled": True,
        }
    ]
    assert config["cream"]["lambda_cream"] == 0.0
    assert config["baseline"] == {"method": "none", "mu": 0.0}
    assert config["logging"]["log_dir"] == LOG_DIR
    assert PurePosixPath(config["logging"]["log_dir"]).is_absolute()


def test_s3_fixed_sample_and_pass_criteria_are_configuration_driven():
    config = _load_config()

    assert config["training"]["batch_size"] == 1
    assert config["training"]["local_epochs"] == 1
    assert config["training"]["accumulation_steps"] == 1
    assert config["training"]["rounds"] == 40
    assert config["s3_overfit"] == {
        "client_id": "client_2",
        "min_source_wt_pixels": 64,
        "min_loss_reduction_ratio": 0.2,
        "min_logit_std": 1e-06,
        "min_predicted_wt_pixels": 1,
    }


def test_s3_script_uses_the_main_round_path_without_synthetic_data():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT_PATH))

    assert "HeterogeneousBraTSDataset" in source
    assert "get_reproducibility_manifest" in source
    assert "_train_single_round" in source
    assert "s3_overfit_result.json" in source
    assert "torch.randn" not in source
    assert "--data_root" not in source
    assert "--log_dir" not in source
