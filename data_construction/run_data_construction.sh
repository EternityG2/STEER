#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/data_construction}"

TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:?Set TOKENIZER_MODEL_PATH.}"
GENERATION_API_BASE_URL="${GENERATION_API_BASE_URL:?Set GENERATION_API_BASE_URL.}"
GENERATION_API_KEY="${GENERATION_API_KEY:?Set GENERATION_API_KEY.}"
GENERATION_MODEL_NAME="${GENERATION_MODEL_NAME:?Set GENERATION_MODEL_NAME.}"
JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:?Set JUDGE_API_BASE_URL.}"
JUDGE_API_KEY="${JUDGE_API_KEY:?Set JUDGE_API_KEY.}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:?Set JUDGE_MODEL_NAME.}"

DIFFICULTY_API_BASE_URL="${DIFFICULTY_API_BASE_URL:-${GENERATION_API_BASE_URL}}"
DIFFICULTY_API_KEY="${DIFFICULTY_API_KEY:-${GENERATION_API_KEY}}"
DIFFICULTY_MODEL_NAME="${DIFFICULTY_MODEL_NAME:-${GENERATION_MODEL_NAME}}"

INIT_DIR="${OUT_DIR}/init"
BALANCED_DIR="${OUT_DIR}/balanced"
MC_DIR="${OUT_DIR}/mc_scores"
JUDGE_DIR="${OUT_DIR}/judge"
CONSENSUS_DIR="${OUT_DIR}/consensus"
OPENRLHF_DIR="${OUT_DIR}/openrlhf"

mkdir -p "${INIT_DIR}" "${BALANCED_DIR}" "${MC_DIR}" "${JUDGE_DIR}" "${CONSENSUS_DIR}" "${OPENRLHF_DIR}"

RAW_JSONL="${INIT_DIR}/raw_filtered.jsonl"
TOKEN_JSONL="${INIT_DIR}/token_filtered.jsonl"
TASK_JSONL="${INIT_DIR}/tasks.jsonl"
BALANCED_JSONL="${BALANCED_DIR}/balanced_tasks.jsonl"
MC_JSONL="${MC_DIR}/mc_tasks.jsonl"
HARD_LABEL_JSONL="${MC_DIR}/mc_hard_labels.jsonl"
JUDGE_JSONL="${JUDGE_DIR}/judge_tasks.jsonl"
CONSENSUS_RETAINED_JSONL="${CONSENSUS_DIR}/retained.jsonl"
CONSENSUS_ALL_JSONL="${CONSENSUS_DIR}/all_with_flags.jsonl"

python "${SCRIPT_DIR}/data_init_procerss.py" \
  --output_jsonl "${RAW_JSONL}" \
  --dataset_name "${DATASET_NAME:-ByteDance-Seed/Code-Contests-Plus}" \
  --dataset_config "${DATASET_CONFIG:-3x}" \
  --dataset_split "${DATASET_SPLIT:-train}" \
  --target_difficulties "${TARGET_DIFFICULTIES:-7,8}" \
  ${DATASET_CACHE_DIR:+--cache_dir "${DATASET_CACHE_DIR}"} \
  ${HF_ENDPOINT:+--hf_endpoint "${HF_ENDPOINT}"}

python "${SCRIPT_DIR}/filter_by_token.py" \
  --input_jsonl "${RAW_JSONL}" \
  --output_jsonl "${TOKEN_JSONL}" \
  --model_path "${TOKENIZER_MODEL_PATH}" \
  --max_tokens "${MAX_TOKENS:-2000}"

python "${SCRIPT_DIR}/filter_difficulty_by_llm.py" \
  --input_jsonl "${TOKEN_JSONL}" \
  --output_jsonl "${TASK_JSONL}" \
  --api_base_url "${DIFFICULTY_API_BASE_URL}" \
  --api_key "${DIFFICULTY_API_KEY}" \
  --model_name "${DIFFICULTY_MODEL_NAME}" \
  --num_generations "${DIFFICULTY_NUM_GENERATIONS:-2}" \
  --max_workers "${DIFFICULTY_MAX_WORKERS:-5}" \
  --batch_size "${DIFFICULTY_BATCH_SIZE:-10}" \
  --timeout "${DIFFICULTY_TIMEOUT:-2}"

python "${SCRIPT_DIR}/build_balanced_dataset.py" \
  --input_jsonl "${TASK_JSONL}" \
  --output_dir "${BALANCED_DIR}" \
  --api_base_url "${GENERATION_API_BASE_URL}" \
  --api_key "${GENERATION_API_KEY}" \
  --model_name "${GENERATION_MODEL_NAME}" \
  --temperature "${GENERATION_TEMPERATURE:-0.8}" \
  --max_tokens "${GENERATION_MAX_TOKENS:-2048}" \
  --max_retries "${GENERATION_MAX_RETRIES:-3}" \
  --exec_timeout "${EXEC_TIMEOUT:-5}" \
  --target_each "${TARGET_EACH:-8}" \
  --batch_schedule "${BATCH_SCHEDULE:-16,32,64,128,256}" \
  --ray_num_actors "${RAY_NUM_ACTORS:-8}"

python "${SCRIPT_DIR}/compute_mc_scores_ray.py" \
  --input_jsonl "${BALANCED_JSONL}" \
  --output_dir "${MC_DIR}" \
  --api_base_url "${GENERATION_API_BASE_URL}" \
  --api_key "${GENERATION_API_KEY}" \
  --model_name "${GENERATION_MODEL_NAME}" \
  --temperature "${GENERATION_TEMPERATURE:-0.8}" \
  --max_tokens "${GENERATION_MAX_TOKENS:-2048}" \
  --max_retries "${GENERATION_MAX_RETRIES:-3}" \
  --exec_timeout "${EXEC_TIMEOUT:-5}" \
  --ray_num_actors "${RAY_NUM_ACTORS:-8}" \
  --max_samples_per_task "${MAX_SAMPLES_PER_TASK:-16}" \
  --max_mc_steps "${MAX_MC_STEPS:-3}"

python "${SCRIPT_DIR}/convert_mc_scores_to_hard_labels.py" \
  --input_jsonl "${MC_JSONL}" \
  --output_jsonl "${HARD_LABEL_JSONL}" \
  --stats_json "${MC_DIR}/hard_label_stats.json" \
  --epsilon "${RPE_EPSILON:-0.8}"

python "${SCRIPT_DIR}/llm_as_judge.py" \
  --input_jsonl "${MC_JSONL}" \
  --output_dir "${JUDGE_DIR}" \
  --api_base_url "${JUDGE_API_BASE_URL}" \
  --api_key "${JUDGE_API_KEY}" \
  --model_name "${JUDGE_MODEL_NAME}" \
  --temperature "${JUDGE_TEMPERATURE:-0.2}" \
  --max_tokens "${JUDGE_MAX_TOKENS:-2048}" \
  --max_retries "${JUDGE_MAX_RETRIES:-3}" \
  --ray_num_actors "${JUDGE_RAY_NUM_ACTORS:-8}" \
  --max_samples_per_task "${MAX_SAMPLES_PER_TASK:-16}" \
  --max_judge_steps "${MAX_JUDGE_STEPS:-3}"

python "${SCRIPT_DIR}/strict_consensus_filter_mc_judge.py" \
  --mc_jsonl "${HARD_LABEL_JSONL}" \
  --judge_jsonl "${JUDGE_JSONL}" \
  --output_retained_jsonl "${CONSENSUS_RETAINED_JSONL}" \
  --output_all_jsonl "${CONSENSUS_ALL_JSONL}" \
  --stats_json "${CONSENSUS_DIR}/stats.json"

python "${SCRIPT_DIR}/convert_consensus_to_openrlhf_prm.py" \
  --src_path "${CONSENSUS_RETAINED_JSONL}" \
  --out_dir "${OPENRLHF_DIR}" \
  --model_path "${TOKENIZER_MODEL_PATH}" \
  --test_rows "${TEST_ROWS:-500}" \
  --seed "${SEED:-42}" \
  --label_source "${LABEL_SOURCE:-mc}"

printf 'Data construction finished: %s\n' "${OUT_DIR}"
