#!/usr/bin/env bash

cd /data/${KRYLOV_NAMESPACE}/data/baliao/dpc/dpc/v1

set -euo pipefail

EP="${EP:-10}"
BS="${BS:-4}"
LR="${LR:-2e-4}"
ACCUM=2

# Defaults (override via env vars)
TRAIN_SENTENCE_CSV="${TRAIN_SENTENCE_CSV:-data/train_sentence_clean.csv,data/final_train_sentence.csv}"
MODEL_NAME="${MODEL_NAME:-/mnt/nushare2/data/baliao/PLLMs/google/byt5-base}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/nushare2/data/baliao/dpc/v1-base-clean-final/fold0/ep${EP}bs${ACCUM}x${BS}lr${LR}}"
NUM_FOLDS="${NUM_FOLDS:-10}"
FOLD_INDEX="${FOLD_INDEX:-0}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# Required resources (override as needed)
DPC_EXTRA_DIR="${DPC_EXTRA_DIR:-extra}"
LEXICON_PATH="${LEXICON_PATH:-extra/OA_Lexicon_eBL.csv}"
ONOMASTICON_PATH="${ONOMASTICON_PATH:-extra/onomasticon.csv}"
EBL_DICT_PATH="${EBL_DICT_PATH:-extra/eBL_Dictionary.csv}"

mkdir -p "${OUTPUT_DIR}"

chown -R 110541254:110541254 "/mnt/nushare2/data/baliao/dpc"

ARGS=(
  --train-sentence-csv "${TRAIN_SENTENCE_CSV}"
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --dpc-extra-dir "${DPC_EXTRA_DIR}"
  --lexicon-path "${LEXICON_PATH}"
  --onomasticon-path "${ONOMASTICON_PATH}"
  --ebl-dict-path "${EBL_DICT_PATH}"
  --report-to "${REPORT_TO}"
  --batch-size "${BS}"
  --learning-rate "${LR}"
  --epochs "${EP}"
  --grad-accum "${ACCUM}"
  --no-early-stopping
  --warmup-ratio 0
  --set "LARSEN_LETTERS_PATH=${DPC_EXTRA_DIR}/larsen_letters.csv"
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

python3 train.py "${ARGS[@]}" "$@" > ${OUTPUT_DIR}/log.out 2> ${OUTPUT_DIR}/log.err


chown -R 110541254:110541254 "/mnt/nushare2/data/baliao/dpc"
