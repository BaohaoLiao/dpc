#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


_WS_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_BAD_EMPTY = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def _clean_text(s: object) -> str:
    if s is None:
        return ""
    out = _ZW_RE.sub("", str(s))
    out = out.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    out = unicodedata.normalize("NFKC", out).strip()
    out = _WS_RE.sub(" ", out)
    return "" if out.lower() in _BAD_EMPTY else out


def _pick_col(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols_low = {c.lower(): c for c in columns}
    for cand in candidates:
        c = cols_low.get(cand.lower())
        if c is not None:
            return c
    return None


def _infer_schema(df: pd.DataFrame) -> tuple[str, str, str | None]:
    cols = list(df.columns)
    src_col = _pick_col(
        cols,
        [
            "new_transliteration_sentence",
            "new_transliteration",
            "transliteration",
            "translit",
            "source",
            "src",
            "akkadian",
        ],
    )
    tgt_col = _pick_col(
        cols,
        [
            "new_translation_sentence",
            "new_translation",
            "translation",
            "english",
            "target",
            "tgt",
            "en",
        ],
    )
    id_col = _pick_col(cols, ["oare_id", "text_uuid", "id", "uuid", "text_id"])
    if src_col is None or tgt_col is None:
        raise ValueError(f"Could not infer src/tgt columns. Columns: {cols}")
    return src_col, tgt_col, id_col


def load_and_sanitize_parallel(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    src_col, tgt_col, id_col = _infer_schema(df)

    keep_cols = [c for c in (id_col, src_col, tgt_col) if c is not None]
    out = df[keep_cols].copy()

    ren = {src_col: "transliteration", tgt_col: "translation"}
    if id_col is not None:
        ren[id_col] = "oare_id"
    out = out.rename(columns=ren)

    if "oare_id" not in out.columns:
        out["oare_id"] = [f"{p.stem}::{i}" for i in range(len(out))]

    out["oare_id"] = out["oare_id"].map(_clean_text)
    out["transliteration"] = out["transliteration"].map(_clean_text)
    out["translation"] = out["translation"].map(_clean_text)

    out = out[
        out["transliteration"].str.strip().ne("") & out["translation"].str.strip().ne("")
    ].copy()
    out = out[out["transliteration"] != out["translation"]].copy()
    out["is_extra"] = True
    out["source_file"] = p.name
    return out[["oare_id", "transliteration", "translation", "is_extra", "source_file"]]


def normalize_external_transliteration(text: object) -> str:
    return _clean_text(text)


def normalize_external_translation(text: object) -> str:
    return _clean_text(text)


def flag_incomplete(df: pd.DataFrame, ratio_max: float = 0.60) -> pd.Series:
    raw_src = df["transliteration"].astype(str).fillna("").str.strip()
    raw_tgt = df["translation"].astype(str).fillna("").str.strip()

    src_chars = raw_src.str.len()
    tgt_chars = raw_tgt.str.len()
    ratio = (tgt_chars / src_chars.replace(0, pd.NA)).fillna(0.0)

    header_only = raw_tgt.str.lower().str.startswith("to ") & (src_chars >= 80) & (tgt_chars <= 60)
    return header_only | (ratio <= float(ratio_max))


def _none_like(s: str | None) -> bool:
    if s is None:
        return True
    return str(s).strip().lower() in {"", "none", "null"}


def _make_split(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_df = pd.read_csv(args.train_csv_path).assign(is_extra=False)

    if _none_like(args.train_final_csv_path):
        final_comp_df = pd.DataFrame(columns=["oare_id", "transliteration", "translation", "is_extra"])
        print("[DATA] TRAIN_FINAL_CSV_PATH disabled; using only TRAIN_CSV_PATH.", flush=True)
    else:
        final_path = Path(str(args.train_final_csv_path))
        if not final_path.exists():
            raise FileNotFoundError(f"TRAIN_FINAL_CSV_PATH does not exist: {final_path}")
        final_comp_df = pd.read_csv(final_path).assign(is_extra=False)

    extras: list[pd.DataFrame] = []
    larsen_path = Path(args.larsen_letters_path)
    if not larsen_path.exists():
        raise FileNotFoundError(f"LARSEN_LETTERS_PATH does not exist: {larsen_path}")
    larsen_df = load_and_sanitize_parallel(larsen_path).assign(source="larsen")
    larsen_df = larsen_df[["transliteration", "translation", "is_extra", "source"]].copy()
    larsen_df["transliteration"] = larsen_df["transliteration"].map(normalize_external_transliteration)
    larsen_df["translation"] = larsen_df["translation"].map(normalize_external_translation)
    extras.append(larsen_df)

    extra_df = pd.concat(extras, ignore_index=True)
    extra_df = extra_df.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)
    extra_df["oare_id"] = [f"extra::{i}" for i in range(len(extra_df))]

    bad_comp = flag_incomplete(comp_df)
    comp_clean = comp_df.loc[~bad_comp].reset_index(drop=True)
    bad_final = flag_incomplete(final_comp_df)
    final_comp_clean = final_comp_df.loc[~bad_final].reset_index(drop=True)
    bad_extra = flag_incomplete(extra_df)
    extra_clean = extra_df.loc[~bad_extra].reset_index(drop=True)

    core_clean = pd.concat([comp_clean, final_comp_clean], ignore_index=True)
    core_clean = core_clean.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)

    num_folds = int(args.num_folds)
    fold_index = int(args.fold_index)

    if num_folds > 1:
        groups = core_clean["oare_id"].astype(str)
        unique_groups = groups.nunique()
        if unique_groups < num_folds:
            raise ValueError(
                f"[SPLIT] NUM_FOLDS={num_folds} but only {unique_groups} unique oare_id groups are available."
            )
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(f"[SPLIT] FOLD_INDEX must be in [0, {num_folds - 1}], got {fold_index}.")

        fold_splits = list(GroupKFold(n_splits=num_folds).split(core_clean, groups=groups))
        tr_idx, va_idx = fold_splits[fold_index]
        train_core_df = core_clean.iloc[tr_idx].reset_index(drop=True)
        val_split_df = core_clean.iloc[va_idx].reset_index(drop=True)
        train_split_df = pd.concat([train_core_df, extra_clean], ignore_index=True)
        train_split_df = train_split_df.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)
        print(
            f"[SPLIT] GroupKFold folds={num_folds} fold_index={fold_index} | "
            f"core_train: {len(train_core_df)}, extra_train: {len(extra_clean)}, "
            f"train_total: {len(train_split_df)}, val: {len(val_split_df)}",
            flush=True,
        )
    else:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=float(args.val_size),
            random_state=int(args.seed),
        )
        tr_idx, va_idx = next(gss.split(core_clean, groups=core_clean["oare_id"].astype(str)))
        train_core_df = core_clean.iloc[tr_idx].reset_index(drop=True)
        val_split_df = core_clean.iloc[va_idx].reset_index(drop=True)
        train_split_df = pd.concat([train_core_df, extra_clean], ignore_index=True)
        train_split_df = train_split_df.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)
        print(
            f"[SPLIT] GroupShuffleSplit val_size={float(args.val_size)} | "
            f"core_train: {len(train_core_df)}, extra_train: {len(extra_clean)}, "
            f"train_total: {len(train_split_df)}, val: {len(val_split_df)}",
            flush=True,
        )

    tr_ids = set(train_split_df["oare_id"].astype(str).tolist())
    va_ids = set(val_split_df["oare_id"].astype(str).tolist())
    overlap = tr_ids & va_ids
    if overlap:
        raise ValueError(f"[SPLIT] LEAKAGE DETECTED: {len(overlap)} oare_id appear in both splits.")

    before = len(val_split_df)
    val_split_df = val_split_df[
        val_split_df["translation"].astype(str).map(lambda s: len(str(s).split()) <= int(args.val_cutoff_word_thr))
    ].reset_index(drop=True)
    print(f"[VAL_LEN_CAP] kept={len(val_split_df)}/{before} | dropped={before-len(val_split_df)}", flush=True)
    return train_split_df, val_split_df


def _ngram_counts(texts: Iterable[str], n: int) -> Counter[tuple[str, ...]]:
    out: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        toks = str(text).split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            out[tuple(toks[i : i + n])] += 1
    return out


def _report_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, min_n: int, max_n: int) -> None:
    train_texts = train_df["transliteration"].astype(str).tolist()
    val_texts = val_df["transliteration"].astype(str).tolist()

    print(
        f"[DATASET] train_rows={len(train_texts)} val_rows={len(val_texts)} "
        f"(whitespace tokenization for n-grams)",
        flush=True,
    )

    for n in range(min_n, max_n + 1):
        train_cnt = _ngram_counts(train_texts, n)
        val_cnt = _ngram_counts(val_texts, n)

        train_set = set(train_cnt.keys())
        val_set = set(val_cnt.keys())
        overlap_set = train_set & val_set

        val_unique_total = len(val_set)
        val_unique_overlap = len(overlap_set)
        unique_overlap_pct = (100.0 * val_unique_overlap / val_unique_total) if val_unique_total else 0.0

        val_occ_total = sum(val_cnt.values())
        val_occ_overlap = sum(c for g, c in val_cnt.items() if g in train_set)
        occ_overlap_pct = (100.0 * val_occ_overlap / val_occ_total) if val_occ_total else 0.0

        print(
            f"[NGRAM] n={n} | "
            f"train_unique={len(train_set)} val_unique={val_unique_total} "
            f"overlap_unique={val_unique_overlap} ({unique_overlap_pct:.2f}%) | "
            f"val_occ_overlap={val_occ_overlap}/{val_occ_total} ({occ_overlap_pct:.2f}%)",
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute transliteration n-gram overlap between train/val using v2/train.py split strategy."
    )
    p.add_argument("--train-csv-path", default="data/train_sentence_clean.csv")
    p.add_argument("--train-final-csv-path", default="data/final_train_sentence.csv")
    p.add_argument("--larsen-letters-path", default="extra/larsen_letters.csv")
    p.add_argument("--seed", type=int, default=4213)
    p.add_argument("--val-size", type=float, default=0.001)
    p.add_argument("--num-folds", type=int, default=10)
    p.add_argument("--fold-index", type=int, default=0)
    p.add_argument("--val-cutoff-word-thr", type=int, default=60)
    p.add_argument("--min-n", type=int, default=1)
    p.add_argument("--max-n", type=int, default=4)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.min_n < 1:
        raise ValueError("--min-n must be >= 1")
    if args.max_n < args.min_n:
        raise ValueError("--max-n must be >= --min-n")

    train_split_df, val_split_df = _make_split(args)
    _report_overlap(train_split_df, val_split_df, args.min_n, args.max_n)


if __name__ == "__main__":
    main()
