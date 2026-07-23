from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import FederatedConfig


def test_fedprox_baseline_section_is_flattened():
    config_path = PROJECT_ROOT / "tests" / ".fedprox_config_flattened.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_root: data/federated_split",
                "training:",
                "  rounds: 1",
                "cream:",
                "  lambda_cream: 0.0",
                "server:",
                "  aggregation_method: fedavg",
                "baseline:",
                "  method: fedprox",
                "  mu: 0.01",
                "logging:",
                "  experiment_name: FedSAM3_BaselineD_FedProx",
            ]
        ),
        encoding="utf-8",
    )

    try:
        config = FederatedConfig.from_yaml(str(config_path))

        assert config.baseline_method == "fedprox"
        assert config.fedprox_mu == pytest.approx(0.01)
    finally:
        config_path.unlink(missing_ok=True)


def test_fedprox_requires_positive_mu():
    config_path = PROJECT_ROOT / "tests" / ".fedprox_config_invalid_mu.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_root: data/federated_split",
                "training:",
                "  rounds: 1",
                "baseline:",
                "  method: fedprox",
                "  mu: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError, match="fedprox_mu"):
            FederatedConfig.from_yaml(str(config_path))
    finally:
        config_path.unlink(missing_ok=True)


def test_invalid_baseline_method_is_rejected():
    config_path = PROJECT_ROOT / "tests" / ".fedprox_config_invalid_method.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_root: data/federated_split",
                "training:",
                "  rounds: 1",
                "baseline:",
                "  method: unsupported",
                "  mu: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError, match="baseline_method"):
            FederatedConfig.from_yaml(str(config_path))
    finally:
        config_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "config_name",
    [
        "exp_group_a.yaml",
        "exp_group_b.yaml",
        "exp_group_c.yaml",
    ],
)
def test_existing_group_configs_default_to_no_baseline(config_name):
    config_path = PROJECT_ROOT / "configs" / config_name

    config = FederatedConfig.from_yaml(str(config_path))

    assert config.baseline_method == "none"
    assert config.fedprox_mu == pytest.approx(0.0)
