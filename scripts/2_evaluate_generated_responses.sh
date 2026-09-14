#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${LANGS:-}" ]]; then
  echo 'LANGS is required, for example: LANGS="ca de es"'
  exit 1
fi
read -r -a langs <<< "${LANGS}"
DATA_DIR="${REPO_ROOT}/data"
OUTPUT_RESPONSES_DIR="${REPO_ROOT}/outputs/generated_responses"
OUTPUT_RESULTS_DIR="${REPO_ROOT}/outputs/results"

model_name="${MODEL_NAME:-tiny-aya-global}"
EXP_NAME="${EXP_NAME:-base_generation}"
GLOTLID_MIN_CONFIDENCE="${GLOTLID_MIN_CONFIDENCE:-0.3}"
GLOTLID_MODEL_PATH="${GLOTLID_MODEL_PATH:-}"

for lang in "${langs[@]}"; do
  input_data="${DATA_DIR}/${lang}.jsonl"
  input_response_data="${OUTPUT_RESPONSES_DIR}/${lang}/${model_name}/${EXP_NAME}.jsonl"
  output_dir="${OUTPUT_RESULTS_DIR}/${lang}/${model_name}/${EXP_NAME}"

  if [[ ! -f "${input_data}" ]]; then
    echo "Missing input data: ${input_data}"
    exit 1
  fi

  if [[ ! -f "${input_response_data}" ]]; then
    echo "Missing generated responses: ${input_response_data}"
    exit 1
  fi

  mkdir -p "${output_dir}"

  eval_args=(
    --input_data="${input_data}"
    --input_response_data="${input_response_data}"
    --output_dir="${output_dir}"
    --glotlid_min_confidence="${GLOTLID_MIN_CONFIDENCE}"
  )
  if [[ -n "${GLOTLID_MODEL_PATH}" ]]; then
    eval_args+=(--glotlid_model_path="${GLOTLID_MODEL_PATH}")
  fi
  (
    cd "${REPO_ROOT}"
    python -m evaluation.evaluation_main "${eval_args[@]}"
  ) > "${output_dir}/eval_results.txt"
done
