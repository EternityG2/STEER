#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the PRM checkpoint path.}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8009}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
AGGREGATE="${AGGREGATE:-mean}"

python "$(dirname "$0")/prm_server.py" \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --cuda_devices "${CUDA_DEVICES}" \
  --aggregate "${AGGREGATE}"
