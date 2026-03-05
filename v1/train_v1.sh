#!/usr/bin/env bash

set -euo pipefail

EP="${EP:-10}"
BS="${BS:-4}"
LR="${LR:-2e-4}"
ACCUM="${ACCUM:-2}"

# Keep compatibility with old env style: "train.csv,final.csv"
TRAIN_CSV_PATH="${TRAIN_CSV_PATH:-${TRAIN_CSV_PATH_DEFAULT:-data/train_sentence_clean.csv}}"
TRAIN_FINAL_CSV_PATH="${TRAIN_FINAL_CSV_PATH:-${TRAIN_FINAL_CSV_PATH_DEFAULT:-data/final_train_sentence.csv}}"

# Defaults (override via env vars)
MODEL_NAME="${MODEL_NAME:-/mnt/nushare2/data/baliao/PLLMs/google/byt5-base}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/nushare2/data/baliao/dpc/v1-base-clean-final/fold0/ep${EP}bs${ACCUM}x${BS}lr${LR}}"
NUM_FOLDS="${NUM_FOLDS:-10}"
FOLD_INDEX="${FOLD_INDEX:-0}"

# Required resources (override as needed)
DPC_EXTRA_DIR="${DPC_EXTRA_DIR:-extra}"

mkdir -p "${OUTPUT_DIR}"

# Keep similar behavior to train.sh (best effort).
chown -R 110541254:110541254 "/mnt/nushare2/data/baliao/dpc"

ARGS=(
  --train-csv-path "${TRAIN_CSV_PATH}"
  --train-final-csv-path "${TRAIN_FINAL_CSV_PATH}"
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --dpc-extra-dir "${DPC_EXTRA_DIR}"
  --batch-size "${BS}"
  --learning-rate "${LR}"
  --epochs "${EP}"
  --grad-accum "${ACCUM}"
  --no-early-stopping
  --warmup-ratio 0.05
  --num-folds "${NUM_FOLDS}"
  --fold-index "${FOLD_INDEX}"
  --ckpt-avg-k 3
  --no-early-stopping
  --set "USE_VAL_FOR_TRAINING=True"
)

python3 train_v1.py "${ARGS[@]}" "$@" > "${OUTPUT_DIR}/log.out" 2> "${OUTPUT_DIR}/log.err"


chown -R 110541254:110541254 "/mnt/nushare2/data/baliao/dpc" || true
