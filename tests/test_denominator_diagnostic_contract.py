import copy
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
U_CONFIG_PATH = PROJECT_ROOT / "configs" / "fedsam3_2x2_u_fedavg.yaml"
R_CONFIG_PATH = PROJECT_ROOT / "configs" / "fedsam3_2x2_r_fedavg.yaml"
N_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "fedsam3_denominator_diagnostic_seed3407_n_fedavg.yaml"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "configs"
    / "fedsam3_denominator_diagnostic_manifest_seed3407.json"
)


def _read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fixed_config(config):
    value = copy.deepcopy(config)
    value["federated"].pop("routing_mode")
    value["aggregation"].pop("unoptimized_update_policy")
    value["logging"].pop("log_dir")
    value["logging"].pop("experiment_name")
    return value


def test_n_config_changes_only_parameter_eligibility_and_output_identity():
    unrestricted = _read_yaml(U_CONFIG_PATH)
    uploader_normalized = _read_yaml(N_CONFIG_PATH)

    assert _fixed_config(uploader_normalized) == _fixed_config(unrestricted)
    assert uploader_normalized["seed"] == 3407
    assert uploader_normalized["federated"]["routing_mode"] == (
        "uploader_renormalized"
    )
    assert uploader_normalized["aggregation"]["unoptimized_update_policy"] == (
        "exclude_and_renormalize"
    )
    assert uploader_normalized["logging"]["log_dir"].endswith(
        "/fedsam3_denominator_diagnostic_seed3407/n_fedavg"
    )


def test_manifest_defines_a_single_seed_u_n_r_decomposition():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cells = manifest["cells"]

    assert manifest["seed"] == 3407
    assert [cell["label"] for cell in cells] == ["U", "N", "R"]
    assert [cell["routing_mode"] for cell in cells] == [
        "unrestricted",
        "uploader_renormalized",
        "restricted",
    ]
    assert [cell["config"] for cell in cells] == [
        "configs/fedsam3_2x2_u_fedavg.yaml",
        "configs/fedsam3_denominator_diagnostic_seed3407_n_fedavg.yaml",
        "configs/fedsam3_2x2_r_fedavg.yaml",
    ]
    assert len({cell["log_dir"] for cell in cells}) == 3
    assert manifest["effect_decomposition"] == {
        "denominator_dilution": "N - U",
        "modality_whitelist": "R - N",
        "total_routing_difference": "R - U",
    }
    assert manifest["acceptance"]["valid_upload_contract_requires_n_equals_r"] is True

    assert _fixed_config(_read_yaml(R_CONFIG_PATH)) == _fixed_config(
        _read_yaml(N_CONFIG_PATH)
    )
