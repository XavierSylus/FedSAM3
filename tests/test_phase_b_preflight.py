import importlib
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = "/autodl-fs/data/FedSAM3-Cream/datasets/federated_split"
SMOKE_LOG_DIR = (
    "/autodl-fs/data/FedSAM3-Cream/experiments/logs/tests/phase_b_smoke"
)
DIRECT_LOG_DIR = (
    "/autodl-fs/data/FedSAM3-Cream/experiments/logs/tests/phase_b_direct"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _drop_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def test_importing_main_does_not_import_trainer():
    _drop_modules("main", "src.federated_trainer")

    import main  # noqa: F401

    assert "src.federated_trainer" not in sys.modules


def test_importing_client_does_not_import_metrics():
    _drop_modules("src.client", "src.metrics")

    import src.client  # noqa: F401

    assert "src.metrics" not in sys.modules


def test_main_preserves_explicit_log_dir_before_trainer_init(monkeypatch):
    config_path = PROJECT_ROOT / "tests" / ".phase_b_smoke.yaml"
    captured = {}

    class FakeTrainer:
        def __init__(self, config):
            captured["config"] = config

        def train(self):
            return 0

    fake_module = types.ModuleType("src.federated_trainer")
    fake_module.FederatedTrainer = FakeTrainer

    _drop_modules("main", "src.federated_trainer")
    monkeypatch.setitem(sys.modules, "src.federated_trainer", fake_module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--config", str(config_path), "--device", "cpu"],
    )

    config_path.write_text(
        "\n".join(
            [
                f"data_root: {DATA_ROOT}",
                "federated:",
                "  routing_mode: unrestricted",
                "  client_init_policy: round_global",
                "  persist_client_optimizer: false",
                "aggregation:",
                "  method: fedavg",
                "  sample_weight_unit: private_cases",
                "  unoptimized_update_policy: include_zero",
                "training:",
                "  rounds: 1",
                "options:",
                "  use_dummy: true",
                "logging:",
                "  log_type: none",
                f"  log_dir: {SMOKE_LOG_DIR}",
            ]
        ),
        encoding="utf-8",
    )

    try:
        main_module = importlib.import_module("main")
        exit_code = main_module.main()

        assert exit_code == 0
        assert captured["config"].device == "cpu"
        assert captured["config"].log_dir == SMOKE_LOG_DIR
    finally:
        config_path.unlink(missing_ok=True)


def test_federated_trainer_uses_explicit_log_dir_on_direct_init():
    from src.config_manager import FederatedConfig
    from src.federated_trainer import FederatedTrainer

    config = FederatedConfig(
        data_root=DATA_ROOT,
        log_dir=DIRECT_LOG_DIR,
        use_mock=True,
        device="cpu",
        aggregation_method="fedavg",
        routing_mode="unrestricted",
        sample_weight_unit="private_cases",
        unoptimized_update_policy="include_zero",
    )

    trainer = FederatedTrainer(config)

    assert trainer.config.log_dir == DIRECT_LOG_DIR
    assert trainer.checkpoint_dir == Path(DIRECT_LOG_DIR) / "checkpoints"
