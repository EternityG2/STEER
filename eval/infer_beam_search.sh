#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${DATA_PATH:?Set DATA_PATH to the evaluation JSONL file.}"
API_BASE_URL="${API_BASE_URL:?Set API_BASE_URL for the OpenAI-compatible generation endpoint.}"
API_KEY="${API_KEY:?Set API_KEY for the generation endpoint.}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME for evaluation.}"
PRM_SERVER_URL="${PRM_SERVER_URL:?Set PRM_SERVER_URL for the PRM scoring server.}"

python "${SCRIPT_DIR}/infer_beam_search.py" \
  --b1 "${B1:-1}" \
  --b2 "${B2:-1}" \
  --data_path "${DATA_PATH}" \
  --api_base_url "${API_BASE_URL}" \
  --api_key "${API_KEY}" \
  --model_name "${MODEL_NAME}" \
  --prm_server_url "${PRM_SERVER_URL}" \
  --num_workers "${NUM_WORKERS:-32}" \
  --prm_timeout "${PRM_TIMEOUT:-60.0}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --final_temperature "${FINAL_TEMPERATURE:-0.0}" \
  --step_max_tokens "${STEP_MAX_TOKENS:-512}" \
  --final_max_tokens "${FINAL_MAX_TOKENS:-1024}" \
  --run_timeout "${RUN_TIMEOUT:-3}" \
  --max_tasks "${MAX_TASKS:--1}"
