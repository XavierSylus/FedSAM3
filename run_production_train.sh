#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: ./run_production_train.sh <config.yaml> [--smoke]"
    exit 2
fi
if [[ $# -eq 2 && "$2" != "--smoke" ]]; then
    echo "Unknown mode: $2"
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG_FILE="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="$(basename "$CONFIG_FILE" .yaml)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
TRAIN_ARGS=()
if [[ $# -eq 2 ]]; then
    RUN_NAME="${RUN_NAME}_smoke"
    TRAIN_ARGS=(
        --rounds 1
        --log_dir "logs/smoke/${RUN_NAME}/${TIMESTAMP}"
    )
fi
LAUNCH_DIR="logs/launcher/${RUN_NAME}"
LOG_FILE="${LAUNCH_DIR}/${TIMESTAMP}.log"
ERROR_LOG_FILE="${LAUNCH_DIR}/${TIMESTAMP}.error.log"
PID_FILE="${LAUNCH_DIR}/training.pid"

mkdir -p "$LAUNCH_DIR"

if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID="$(cat "$PID_FILE")"
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Training is already running for ${RUN_NAME}: PID ${EXISTING_PID}"
        exit 1
    fi
fi

"$PYTHON_BIN" scripts/server_preflight.py --config "$CONFIG_FILE"

nohup "$PYTHON_BIN" main.py \
    --config "$CONFIG_FILE" \
    "${TRAIN_ARGS[@]}" \
    >"$LOG_FILE" \
    2>"$ERROR_LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" >"$PID_FILE"

echo "Training started"
echo "Config: $CONFIG_FILE"
echo "PID: $TRAIN_PID"
echo "Log: $LOG_FILE"
echo "Error log: $ERROR_LOG_FILE"
