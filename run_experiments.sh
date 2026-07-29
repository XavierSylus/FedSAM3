#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_APPROVED="${FEDSAM3_TRAINING_APPROVED:-0}"

if [[ "$TRAINING_APPROVED" != "1" ]]; then
    echo "Formal training is blocked until server gates S1-S6 pass."
    echo "Set FEDSAM3_TRAINING_APPROVED=1 only after recording their evidence."
    exit 2
fi

CONFIGS=(
    "configs/fedsam3_2x2_u_fedavg.yaml"
    "configs/fedsam3_2x2_u_fedprox.yaml"
    "configs/fedsam3_2x2_r_fedavg.yaml"
    "configs/fedsam3_2x2_r_fedprox.yaml"
    "configs/fedsam3_ratio_2of3_r_fedprox.yaml"
)

for config_file in "${CONFIGS[@]}"; do
    if [[ ! -f "$config_file" ]]; then
        echo "Missing experiment config: $config_file"
        exit 1
    fi
    "$PYTHON_BIN" scripts/server_preflight.py --config "$config_file"
done

for config_file in "${CONFIGS[@]}"; do
    echo "Starting experiment: $config_file"
    "$PYTHON_BIN" main.py --config "$config_file"
    echo "Completed experiment: $config_file"
done
