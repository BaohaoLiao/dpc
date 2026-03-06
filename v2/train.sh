#!/usr/bin/env bash

cd /data/${KRYLOV_NAMESPACE}/data/baliao/dpc/dpc/v2

set -euo pipefail

EP="${EP:-10}"
BS="${BS:-4}"
LR="${LR:-2e-4}"
ACCUM="${ACCUM:-2}"
NPROC="${NPROC:-16}"
MAP_BS="${MAP_BS:-8192}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PARALLEL="${MODEL_PARALLEL:-on}"

# Keep compatibility with old env style: "train.csv,final.csv"
TRAIN_CSV_PATH="${TRAIN_CSV_PATH:-${TRAIN_CSV_PATH_DEFAULT:-data/train_sentence_clean.csv}}"
TRAIN_FINAL_CSV_PATH="${TRAIN_FINAL_CSV_PATH:-${TRAIN_FINAL_CSV_PATH_DEFAULT:-data/final_train_sentence.csv}}"

# Defaults (override via env vars)
MODEL_NAME="${MODEL_NAME:-/mnt/nushare2/data/baliao/PLLMs/google/byt5-akkadian-optimized-34x}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/nushare2/data/baliao/dpc/test/fold0/ep${EP}bs${ACCUM}x${BS}lr${LR}}"
NUM_FOLDS="${NUM_FOLDS:-10}"
FOLD_INDEX="${FOLD_INDEX:-0}"

# Required resources (override as needed)
DPC_EXTRA_DIR="${DPC_EXTRA_DIR:-extra}"
LEXICON_PATH="${LEXICON_PATH:-${DPC_EXTRA_DIR}/OA_Lexicon_eBL.csv}"
ONOMASTICON_PATH="${ONOMASTICON_PATH:-${DPC_EXTRA_DIR}/onomasticon.csv}"
EBL_DICT_PATH="${EBL_DICT_PATH:-${DPC_EXTRA_DIR}/eBL_Dictionary.csv}"
SENTENCES_PATH="${SENTENCES_PATH:-${DPC_EXTRA_DIR}/Sentences_Oare_FirstWord_LinNum.csv}"
LARSEN_LETTERS_PATH="${LARSEN_LETTERS_PATH:-${DPC_EXTRA_DIR}/larsen_letters.csv}"

mkdir -p "${OUTPUT_DIR}"

# Keep similar behavior to train.sh (best effort).
chown -R 110541254:110541254 "/mnt/nushare2/data/baliao/dpc"

ARGS=(
  --train-csv-path "${TRAIN_CSV_PATH}"
  --train-final-csv-path "${TRAIN_FINAL_CSV_PATH}"
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --dpc-extra-dir "${DPC_EXTRA_DIR}"
  --lexicon-path "${LEXICON_PATH}"
  --onomasticon-path "${ONOMASTICON_PATH}"
  --ebl-dict-path "${EBL_DICT_PATH}"
  --sentences-path "${SENTENCES_PATH}"
  --larsen-letters-path "${LARSEN_LETTERS_PATH}"
  --batch-size "${BS}"
  --learning-rate "${LR}"
  --epochs "${EP}"
  --grad-accum "${ACCUM}"
  --nproc "${NPROC}"
  --map-bs "${MAP_BS}"
  --warmup-ratio 0.05
  --num-folds "${NUM_FOLDS}"
  --fold-index "${FOLD_INDEX}"
  --ckpt-avg-k 3
  --no-early-stopping
  --model-parallel "${MODEL_PARALLEL}"
  --model-parallel-devices "0,1"
  --set "USE_VAL_FOR_TRAINING=True"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python3 train.py "${ARGS[@]}" "$@" > "${OUTPUT_DIR}/log.out" 2> "${OUTPUT_DIR}/log.err"


chown -R 110541254:110541254 "/mnt/nushare2/data/baliao/dpc" || true
