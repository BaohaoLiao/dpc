#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="${TRAIN_CSV:-data/train_processed_with_ori_flags.csv}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-10}"
RUN_NAME="${RUN_NAME:-lr${LR}_ep${EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}"

python train.py \
  --train-csv "${TRAIN_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --learning-rate "${LR}" \
  --epochs "${EPOCHS}" \
  "$@"
