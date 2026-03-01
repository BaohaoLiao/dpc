#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env vars)
TRAIN_SENTENCE_CSV="${TRAIN_SENTENCE_CSV:-v1/data/final_train_sentence.csv}"
MODEL_NAME="${MODEL_NAME:-/path/to/base_or_checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/v1}"
NUM_FOLDS="${NUM_FOLDS:-0}"
FOLD_INDEX="${FOLD_INDEX:-0}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# Required resources (override as needed)
LEXICON_PATH="${LEXICON_PATH:-/path/to/OA_Lexicon_eBL.csv}"
ONOMASTICON_PATH="${ONOMASTICON_PATH:-/path/to/onomasticon.csv}"
EBL_DICT_PATH="${EBL_DICT_PATH:-/path/to/eBL_Dictionary.csv}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --train-sentence-csv "${TRAIN_SENTENCE_CSV}"
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --lexicon-path "${LEXICON_PATH}"
  --onomasticon-path "${ONOMASTICON_PATH}"
  --ebl-dict-path "${EBL_DICT_PATH}"
  --report-to "${REPORT_TO}"
)

if [[ "${NUM_FOLDS}" != "0" ]]; then
  ARGS+=(--num-folds "${NUM_FOLDS}" --fold-index "${FOLD_INDEX}")
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_RUN_NAME}" ]]; then
  ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
fi

python v1/train.py "${ARGS[@]}" "$@"
