#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${DATA_PATH:?Set DATA_PATH to the evaluation JSONL file.}"
API_BASE_URL="${API_BASE_URL:?Set API_BASE_URL for the OpenAI-compatible generation endpoint.}"
API_KEY="${API_KEY:?Set API_KEY for the generation endpoint.}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME for evaluation.}"

python "${SCRIPT_DIR}/infer.py" \
  --data_path "${DATA_PATH}" \
  --api_base_url "${API_BASE_URL}" \
  --api_key "${API_KEY}" \
  --model_name "${MODEL_NAME}" \
  --num_workers "${NUM_WORKERS:-32}" \
  --num_generations "${NUM_GENERATIONS:-1}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --max_tokens "${MAX_TOKENS:-2048}" \
  --run_timeout "${RUN_TIMEOUT:-3}" \
  --max_tasks "${MAX_TASKS:--1}"
