#!/usr/bin/env python3
"""
Train ByT5 on Akkadian transliteration (fractional) -> English.

Example:
  python dpc_training_frac.py \
    --train-csv /path/to/train_processed_with_ori_flags.csv \
    --output-dir ./byt5-base-akkadian
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import sacrebleu
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)


DEFAULT_PREFIX = "translate Akkadian to English: "


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ByT5 on DPC fractional data")
    parser.add_argument("--train-csv", required=True, help="Path to training CSV")
    parser.add_argument(
        "--transliteration-col",
        default="transliteration",
        help="Column name for source text",
    )
    parser.add_argument(
        "--translation-col",
        default="translation",
        help="Column name for target text",
    )
    parser.add_argument("--model-name", default="google/byt5-base")
    parser.add_argument("--output-dir", default="./byt5-base-akkadian")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--gen-max-length", type=int, default=None)
    parser.add_argument("--gen-num-beams", type=int, default=1)

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage",
    )
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-strategy", default="no", choices=["no", "epoch", "steps"])
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--device-map", default=None, help="Set to 'auto' for model parallelism")
    parser.add_argument("--ratio-min", type=float, default=0.5)
    parser.add_argument("--ratio-max", type=float, default=3.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save and load the best model at end using the eval metric.",
    )
    parser.add_argument("--metric-for-best-model", default="geo_mean")
    parser.add_argument("--greater-is-better", action="store_true", default=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )


def seed_everything(seed: int) -> None:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_function(
    examples,
    tokenizer,
    max_length: int,
    prefix: str,
    ratio_min: float,
    ratio_max: float,
):
    inputs = [prefix + str(ex) for ex in examples["transliteration"]]
    targets = [str(ex) for ex in examples["translation"]]

    input_enc = tokenizer(inputs, max_length=max_length, truncation=True)
    target_enc = tokenizer(targets, max_length=max_length, truncation=True)

    filtered_input_ids = []
    filtered_attention_mask = []
    filtered_labels = []

    for inp_ids, attn, tgt_ids in zip(
        input_enc["input_ids"],
        input_enc["attention_mask"],
        target_enc["input_ids"],
    ):
        tok_input = len(inp_ids)
        tok_output = len(tgt_ids)
        ratio = tok_output / tok_input if tok_input > 0 else 0

        if ratio_min <= ratio <= ratio_max:
            filtered_input_ids.append(inp_ids)
            filtered_attention_mask.append(attn)
            filtered_labels.append(tgt_ids)

    return {
        "input_ids": filtered_input_ids,
        "attention_mask": filtered_attention_mask,
        "labels": filtered_labels,
    }


def load_and_prepare_data(
    train_csv: str,
    src_col: str,
    tgt_col: str,
    eval_split: float,
    seed: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    train_df = pd.read_csv(train_csv)
    if src_col not in train_df.columns or tgt_col not in train_df.columns:
        raise ValueError(
            f"Missing columns: expected '{src_col}' and '{tgt_col}' in {train_csv}"
        )

    group_col = "oare_id"
    has_group_col = group_col in train_df.columns
    keep_cols = [group_col] if has_group_col else []

    if eval_split and eval_split > 0 and has_group_col:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=eval_split,
            random_state=seed,
        )
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df[group_col].astype(str)))
        train_raw = train_df.iloc[tr_idx].reset_index(drop=True)
        eval_raw = train_df.iloc[va_idx].reset_index(drop=True)

        tr_ids = set(train_raw[group_col].astype(str).tolist())
        va_ids = set(eval_raw[group_col].astype(str).tolist())
        overlap = tr_ids & va_ids
        if overlap:
            raise ValueError(
                f"Split leakage detected: {len(overlap)} {group_col} values appear in both train and eval."
            )

        train_expanded = train_raw
        eval_expanded = eval_raw
        logging.info(
            "Group split by %s -> train rows=%s eval rows=%s",
            group_col,
            len(train_raw),
            len(eval_raw),
        )
        logging.info(
            "Expanded split data -> train rows=%s eval rows=%s",
            len(train_expanded),
            len(eval_expanded),
        )
        return (
            Dataset.from_pandas(train_expanded, preserve_index=False),
            Dataset.from_pandas(eval_expanded, preserve_index=False),
        )

    if eval_split and eval_split > 0 and not has_group_col:
        logging.warning(
            "Column '%s' not found in %s; using row-level random split.",
            group_col,
            train_csv,
        )

    train_expanded = train_df
    logging.info("Expanded train data: %s rows", len(train_expanded))

    dataset = Dataset.from_pandas(train_expanded, preserve_index=False)
    if eval_split and eval_split > 0:
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        return split["train"], split["test"]

    return dataset, None


def sanitize_token_ids(token_ids, tokenizer, name: str, ignore_index: Optional[int] = None):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().numpy()
    else:
        token_ids = np.asarray(token_ids)

    if token_ids.ndim == 3:
        # Likely logits: convert to token ids.
        token_ids = np.argmax(token_ids, axis=-1)

    token_ids = token_ids.astype(np.int64, copy=False)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if ignore_index is not None:
        if token_ids.ndim > 0:
            token_ids = token_ids.copy()
            token_ids[token_ids == ignore_index] = pad_id
        elif token_ids == ignore_index:
            token_ids = np.int64(pad_id)

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        invalid = (token_ids < 0) | (token_ids >= vocab_size)
        if invalid.any():
            logging.warning(
                "Found %d invalid %s token ids; replacing with pad_token_id=%s",
                int(invalid.sum()),
                name,
                pad_id,
            )
            token_ids = token_ids.copy()
            token_ids[invalid] = pad_id
    else:
        token_ids = np.where(token_ids < 0, pad_id, token_ids)

    return token_ids


def build_compute_metrics(tokenizer):

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = sanitize_token_ids(preds, tokenizer, "prediction")
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = sanitize_token_ids(labels, tokenizer, "label", ignore_index=-100)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        hypotheses = [pred.strip() for pred in decoded_preds]
        references = [label.strip() for label in decoded_labels]

        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
        geom_mean = math.sqrt(bleu.score * chrf.score)
        return {
            "bleu": bleu.score,
            "chrf": chrf.score,
            "geo_mean": geom_mean,
        }

    return compute_metrics


class SaveEvalPredictionsCallback(TrainerCallback):
    def __init__(self, trainer: Seq2SeqTrainer, tokenizer, eval_ds: Dataset, output_dir: str):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_ds = eval_ds
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_ds is None:
            return control

        preds_output = self.trainer.predict(self.eval_ds)
        preds = preds_output.predictions
        labels = preds_output.label_ids

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = sanitize_token_ids(preds, self.tokenizer, "prediction")
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = sanitize_token_ids(labels, self.tokenizer, "label", ignore_index=-100)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        epoch = state.epoch if state.epoch is not None else 0
        epoch_idx = int(round(epoch)) if epoch is not None else 0
        step = state.global_step

        preds_dir = os.path.join(self.output_dir, "preds")
        os.makedirs(preds_dir, exist_ok=True)
        out_path = os.path.join(
            preds_dir, f"eval-epoch-{epoch_idx:03d}-step-{step:06d}.jsonl"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for pred, label in zip(decoded_preds, decoded_labels):
                record = {
                    "prediction": pred.strip(),
                    "reference": label.strip(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info("Saved eval predictions to %s", out_path)
        return control


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging()

    seed_everything(args.seed)

    train_ds, eval_ds = load_and_prepare_data(
        args.train_csv,
        args.transliteration_col,
        args.translation_col,
        args.eval_split,
        args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenize_kwargs = dict(
        tokenizer=tokenizer,
        max_length=args.max_length,
        prefix=args.prefix,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    train_ds = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
        fn_kwargs=tokenize_kwargs,
    )

    if eval_ds is not None:
        if args.max_eval_samples:
            eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
        eval_ds = eval_ds.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_ds.column_names,
            fn_kwargs=tokenize_kwargs,
        )

    logging.info("Tokenized train rows: %s", len(train_ds))
    if eval_ds is not None:
        logging.info("Tokenized eval rows: %s", len(eval_ds))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    use_metrics = not args.disable_metrics and eval_ds is not None

    save_strategy = args.save_strategy
    eval_strategy = "epoch" if eval_ds is not None else "no"
    load_best_model_at_end = False
    metric_for_best_model = None
    greater_is_better = None

    if args.save_best:
        if eval_ds is None:
            raise ValueError("--save-best requires an eval split (set --eval-split > 0).")
        if save_strategy == "no":
            save_strategy = "epoch"
        eval_strategy = "epoch"
        load_best_model_at_end = True
        metric_for_best_model = args.metric_for_best_model
        greater_is_better = args.greater_is_better

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing_factor,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        predict_with_generate=eval_ds is not None,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        disable_tqdm=args.disable_tqdm,
        save_only_model=args.save_only_model,
        generation_max_length=args.gen_max_length or args.max_length,
        generation_num_beams=args.gen_num_beams,
        ddp_find_unused_parameters=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer) if use_metrics else None,
    )
    if eval_ds is not None:
        trainer.add_callback(
            SaveEvalPredictionsCallback(trainer, tokenizer, eval_ds, args.output_dir)
        )

    logging.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logging.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
