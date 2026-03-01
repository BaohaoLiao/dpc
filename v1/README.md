# v1 Training

This folder contains a self-contained training script derived from the notebook.
By default it trains **only** on `v1/data/final_train_sentence.csv` and does **not** use any other datasets.

## Quick Start

Single GPU (uses `v1/data/final_train_sentence.csv` by default):
```bash
python v1/train.py \
  --output-dir /path/to/output \
  --model-name /path/to/base_or_checkpoint \
  --lexicon-path /path/to/OA_Lexicon_eBL.csv \
  --onomasticon-path /path/to/onomasticon.csv \
  --ebl-dict-path /path/to/eBL_Dictionary.csv
```

Multi-GPU (DDP, uses `v1/data/final_train_sentence.csv` by default):
```bash
torchrun --nproc_per_node=4 v1/train.py \
  --output-dir /path/to/output \
  --model-name /path/to/base_or_checkpoint \
  --lexicon-path /path/to/OA_Lexicon_eBL.csv \
  --onomasticon-path /path/to/onomasticon.csv \
  --ebl-dict-path /path/to/eBL_Dictionary.csv
```

## Folded Split (10 parts by `oare_id`)

Use `NUM_FOLDS`/`FOLD_INDEX` to train with one fold held out:
```bash
python v1/train.py \
  --num-folds 10 \
  --fold-index 0 \
  --output-dir /path/to/output \
  --model-name /path/to/base_or_checkpoint \
  --lexicon-path /path/to/OA_Lexicon_eBL.csv \
  --onomasticon-path /path/to/onomasticon.csv \
  --ebl-dict-path /path/to/eBL_Dictionary.csv
```

Change `--fold-index` from `0..9` to rotate the held-out fold.

## Notes

- `--train-sentence-csv` can be used to override the default dataset path, but the script does **not** use any other datasets by default.
- Defaults in `Config` point to Kaggle paths, so override them for local runs.
- You can override any `Config` key via `--set KEY=VALUE`.

## Eval Generations

No config changes are required. By default, eval generations are saved to the checkpoint folder (or `output-dir` if no checkpoint folder exists).

You can control this with:
- `--set SAVE_EVAL_GENERATIONS=false` to disable saving.
- `--set EVAL_GENERATIONS_PREFIX=custom_name` to change the filename prefix.

## Weights & Biases (wandb)

Enable logging by setting `REPORT_TO`:
```bash
python v1/train.py \
  --report-to wandb \
  --wandb-project your_project \
  --wandb-run-name run_01 \
  --output-dir /path/to/output \
  --model-name /path/to/base_or_checkpoint \
  --lexicon-path /path/to/OA_Lexicon_eBL.csv \
  --onomasticon-path /path/to/onomasticon.csv \
  --ebl-dict-path /path/to/eBL_Dictionary.csv
```

You can also use `--set REPORT_TO=wandb`.
