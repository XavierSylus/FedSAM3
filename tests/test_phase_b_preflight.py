import importlib
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def test_main_sets_default_log_dir_before_trainer_init(monkeypatch):
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
                "data_root: data/federated_split",
                "training:",
                "  rounds: 1",
                "options:",
                "  use_dummy: true",
                "logging:",
                "  log_type: none",
            ]
        ),
        encoding="utf-8",
    )

    try:
        main_module = importlib.import_module("main")
        exit_code = main_module.main()

        assert exit_code == 0
        assert captured["config"].device == "cpu"
        assert captured["config"].log_dir == str(Path("data/federated_split") / "logs")
    finally:
        config_path.unlink(missing_ok=True)


def test_federated_trainer_sets_default_log_dir_on_direct_init():
    from src.config_manager import FederatedConfig
    from src.federated_trainer import FederatedTrainer

    config = FederatedConfig(
        data_root="data/federated_split",
        log_dir=None,
        use_mock=True,
        device="cpu",
    )

    trainer = FederatedTrainer(config)

    assert trainer.config.log_dir == str(Path("data/federated_split") / "logs")
    assert trainer.checkpoint_dir == Path("data/federated_split") / "logs" / "checkpoints"
