# Auto-generated from v1/deep-pasta-training.ipynb
# Notes:
# - Converted notebook code cells to a Python script.
# - Added CLI config overrides (see --help and --set KEY=VALUE).
# - Wrapped execution in main() and added multi-GPU device + model-parallel support.

# ===== Notebook Cell 0 =====
from __future__ import annotations
import argparse
import ast

# -------------------------
# stdlib
# -------------------------
import builtins as _bt
import gc, shutil
import glob
import joblib, copy
import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import Value
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

# -------------------------
# third-party
# -------------------------
import huggingface_hub
import numpy as np
import pandas as pd
import sacrebleu
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets.utils.logging import disable_progress_bar
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import ConcatDataset, DataLoader, Dataset as TorchDataset, IterableDataset, get_worker_info
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    get_scheduler,
    set_seed,
)


# -------------------------
# misc config / logging
# -------------------------
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)

disable_progress_bar()

# -------------------------
# Perf knobs (GPU)
# -------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

class Config:
    # ============================================================
    # Core
    # ============================================================
    SEED = 4213

    MODEL_NAME = (
    "/home/mangeli/kaggle/deep-past-initiative-machine-translation/working/"
        "byt5-akkadian-optimized-34x"
    )

    INPUT_DIR = "/home/mangeli/kaggle/input/deep-past-initiative-machine-translation"
    OUTPUT_DIR = (
        "/home/mangeli/kaggle/deep-past-initiative-machine-translation/working/"
        "byt5-akkadian-optimized-34x-mbr-mtg-tbm-new-data-v4"
    )

    HF_CACHE_DIR = (
        "/kaggle/deep-past-initiative-machine-translation/working/hf-cache"
    )

    DPC_EXTRA_DIR = INPUT_DIR
    MANUAL_EXTRA_DIR = INPUT_DIR

    PREFIX = "translate Akkadian to English: "
    CLEAN_CHECKPOINTS = True  # whether to delete intermediate checkpoints after training (only keep best)
    
    # ============================================================
    # Lengths / tokenization / generation budget
    # ============================================================
    SRC_MAX_LENGTH = 512
    TGT_MAX_LENGTH = 512
    GEN_MAX_NEW_TOKENS = 512
    GEN_LENGTH_PENALTY = 1.25
    GEN_REPETITION_PENALTY = 1.15
    GEN_NO_REPEAT_NGRAM = 0

    # ============================================================
    # Training (HF Trainer)
    # ============================================================
    BATCH_SIZE = 32
    GRAD_ACCUM = 1
    EPOCHS = 5
    VAL_SIZE = 0.001
    NUM_FOLDS = 10
    FOLD_INDEX = 0
    LEARNING_RATE = 1e-4
    LABEL_SMOOTHING = 0.1
    WARMUP_RATIO = 0.05
    RESET_DECODER = False
    USE_VAL_FOR_TRAINING = False  # whether to include val rows in train
    NPROC = 8
    MAP_BS = 2048
    EARLY_STOPPING_ENABLE = True
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 3e-1
    
    # ============================================================
    # eval: MBR + gloss sampling
    # ============================================================
    MBR_GLOSS_VARIANTS = 1

    MBR_NUM_BEAMS = 4
    MBR_NUM_BEAM_CANDS = 2

    MBR_NUM_SAMPLE_CANDS = 8
    MBR_TEMPERATURE = 0.75
    MBR_TOP_P = 0.9

    MBR_BATCH_SIZE_INPUTS = 64
    MBR_POOL_CAP = 36

    # add TBM/PN to MBR pools
    TBM_ENABLE = True
    PN_ENABLE = True
    KNN_ENABLE = False  # Experimental
    CKPT_AVG_K = 5
    BEST_METRIC_KEY = "eval_geo_mean"
    CKPT_AVG_CLEANUP = False
    SAVE_EVAL_GENERATIONS = True
    EVAL_GENERATIONS_PREFIX = "eval_generations"
    EVAL_AVG_CHECKPOINT = True
    EVAL_AVG_METRIC_PREFIX = "eval_avg"
    MODEL_PARALLEL = "auto"  # one of: auto, on, off
    MODEL_PARALLEL_DEVICES = None  # comma-separated visible CUDA device ids, e.g. "0,1,2,3"

    # ============================================================
    #   Build K TRAIN ds variants:
    #     (raw + appended probes) -> PNGLOSS -> tokenize
    #   Torch dataset selects 1 variant per epoch.
    K_TRAIN_VARIANTS = int(EPOCHS * 1.5)
    GLOSS_MIX_P = 0.5
    PN_MIX_P = 0.5
    USE_PROBE_APPEND = True
  
    # ============================================================
    # Data paths
    # ============================================================
    TRAIN_CSV_PATH     = f"{INPUT_DIR}/train_sentence_clean.csv" # final_train_sentence.csv 
    TRAIN_FINAL_CSV_PATH     = f"{INPUT_DIR}/final_train_sentence.csv"
   
    LEXICON_PATH       = f"{DPC_EXTRA_DIR}/OA_Lexicon_eBL.csv"
    EBL_DICT_PATH      = f"{DPC_EXTRA_DIR}/eBL_Dictionary.csv"
    SENTENCES_PATH     = f"{DPC_EXTRA_DIR}/Sentences_Oare_FirstWord_LinNum.csv"
    LARSEN_LETTERS_PATH = f"{DPC_EXTRA_DIR}/larsen_letters.csv"
    ONOMASTICON_PATH   = f"{DPC_EXTRA_DIR}/onomasticon.csv"
    
    # ============================================================
    # TBM (Translation/Template Based Matching) for MBR pools
    #   - used in TRAIN MBR eval + INFER MBR (same thresholds/behavior)
    # ============================================================

    # Retrieval behavior
    TBM_TOPK = 3              # recommend 1 for safety; bump to 2–3 if you see gains on val
    TBM_MIN_SIM = 0.9        # strict by default; 0.92 is often too permissive
    TBM_HARD_SIM = 0.995      # near-duplicate override-like injection threshold

    # TF-IDF char ngram index params
    TBM_NGRAM_MIN = 3
    TBM_NGRAM_MAX = 6
    TBM_MAX_FEATURES = 250_000

    # ============================================================
    # PNGLOSS (PN canonicalization + glossary append)
    # ============================================================
    
    GLOSS_MAX_ITEMS = 4
    GLOSS_MAX_APPEND_CHARS = 240
    GLOSS_SEED = SEED + 2

    # ============================================================
    # PROBE (append-before-pngloss)
    # ============================================================

    PROBE_APPEND_P = 0.05
    VAL_PROBE_APPEND_P = 0.0
    PROBE_SEED = SEED + 3

    PROBE_ENABLE = {
        "COMMODITY_RARE_SWAP": True,
        "TITLE_CROSS_SWAP": True,
        "RELATIONSHIP_RARE_SWAP": True,
        "DOCUMENT_RARE_SWAP": True,
        "TITLE_COMMON_SWAP": True,
        "VERB_FRAME_SWAP": True,
        "RELATION_SWAP": True,
        "NUMERIC_MEASURE_SWAP": True,
        "TWO_ENTITY_ORDER_SWAP": True,
        "SLASH_OPTION_CHOICE_SWAP": True,
        "JOINED_HYPHEN_COMPOUND_SWAP": True,
    }

    PROBE_CAT_WEIGHTS = {
        "COMMODITY_RARE_SWAP": 1.0,
        "TITLE_CROSS_SWAP": 1.0,
        "RELATIONSHIP_RARE_SWAP": 1.0,
        "DOCUMENT_RARE_SWAP": 1.0,
        "TITLE_COMMON_SWAP": 0.0,
        "VERB_FRAME_SWAP": 1.0,
        "RELATION_SWAP": 0.8,
        "NUMERIC_MEASURE_SWAP": 0.0,
        "TWO_ENTITY_ORDER_SWAP": 0.5,
        "SLASH_OPTION_CHOICE_SWAP": 1.5,
        "JOINED_HYPHEN_COMPOUND_SWAP": 1.5,
    }



def _parse_override_value(raw: str):
    s = str(raw).strip()
    lo = s.lower()
    if lo in {"true", "false"}:
        return lo == "true"
    if lo in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _apply_overrides(cfg_cls, overrides: dict):
    for key, value in overrides.items():
        if not hasattr(cfg_cls, key):
            raise ValueError(f"Unknown Config key: {key}")
        setattr(cfg_cls, key, value)


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Train pipeline converted from deep-pasta-training.ipynb")
    p.add_argument("--input-dir")
    p.add_argument("--output-dir")
    p.add_argument("--hf-cache-dir")
    p.add_argument("--model-name")
    p.add_argument("--dpc-extra-dir")
    p.add_argument("--manual-extra-dir")
    p.add_argument("--train-csv-path")
    p.add_argument("--train-final-csv-path")
    p.add_argument("--lexicon-path")
    p.add_argument("--onomasticon-path")
    p.add_argument("--ebl-dict-path")
    p.add_argument("--larsen-letters-path")
    p.add_argument("--sentences-path")

    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--val-size", type=float)
    p.add_argument("--num-folds", type=int)
    p.add_argument("--fold-index", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--label-smoothing", type=float)
    p.add_argument("--warmup-ratio", type=float)
    p.add_argument("--nproc", type=int)
    p.add_argument("--map-bs", "--map-batch-size", dest="map_bs", type=int)
    p.add_argument("--early-stopping", dest="early_stopping", action="store_true")
    p.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    p.set_defaults(early_stopping=None)
    p.add_argument("--early-stopping-patience", type=int)
    p.add_argument("--early-stopping-threshold", type=float)
    p.add_argument("--ckpt-avg-k", type=int)
    p.add_argument("--best-metric-key")
    p.add_argument("--eval-generations-prefix")
    p.add_argument("--eval-avg-metric-prefix")
    p.add_argument("--save-eval-generations", dest="save_eval_generations", action="store_true")
    p.add_argument("--no-save-eval-generations", dest="save_eval_generations", action="store_false")
    p.set_defaults(save_eval_generations=None)
    p.add_argument("--local-rank", type=int, default=None)
    p.add_argument("--model-parallel", choices=["auto", "on", "off"])
    p.add_argument("--model-parallel-devices")

    p.add_argument("--set", action="append", default=[], help="Arbitrary override: --set KEY=VALUE")
    return p


def _apply_cli_overrides(cfg_cls):
    parser = _build_arg_parser()
    args, _unknown = parser.parse_known_args()

    if args.local_rank is not None:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    mapping = {
        "input_dir": "INPUT_DIR",
        "output_dir": "OUTPUT_DIR",
        "hf_cache_dir": "HF_CACHE_DIR",
        "model_name": "MODEL_NAME",
        "dpc_extra_dir": "DPC_EXTRA_DIR",
        "manual_extra_dir": "MANUAL_EXTRA_DIR",
        "train_csv_path": "TRAIN_CSV_PATH",
        "train_final_csv_path": "TRAIN_FINAL_CSV_PATH",
        "lexicon_path": "LEXICON_PATH",
        "onomasticon_path": "ONOMASTICON_PATH",
        "ebl_dict_path": "EBL_DICT_PATH",
        "larsen_letters_path": "LARSEN_LETTERS_PATH",
        "sentences_path": "SENTENCES_PATH",
        "seed": "SEED",
        "batch_size": "BATCH_SIZE",
        "grad_accum": "GRAD_ACCUM",
        "epochs": "EPOCHS",
        "val_size": "VAL_SIZE",
        "num_folds": "NUM_FOLDS",
        "fold_index": "FOLD_INDEX",
        "learning_rate": "LEARNING_RATE",
        "label_smoothing": "LABEL_SMOOTHING",
        "warmup_ratio": "WARMUP_RATIO",
        "nproc": "NPROC",
        "map_bs": "MAP_BS",
        "early_stopping_patience": "EARLY_STOPPING_PATIENCE",
        "early_stopping_threshold": "EARLY_STOPPING_THRESHOLD",
        "ckpt_avg_k": "CKPT_AVG_K",
        "best_metric_key": "BEST_METRIC_KEY",
        "eval_generations_prefix": "EVAL_GENERATIONS_PREFIX",
        "eval_avg_metric_prefix": "EVAL_AVG_METRIC_PREFIX",
        "model_parallel": "MODEL_PARALLEL",
        "model_parallel_devices": "MODEL_PARALLEL_DEVICES",
    }

    overrides = {}
    explicit_keys = set()
    for arg_name, key in mapping.items():
        val = getattr(args, arg_name)
        if val is not None:
            overrides[key] = val
            explicit_keys.add(key)
    if args.early_stopping is not None:
        overrides["EARLY_STOPPING_ENABLE"] = bool(args.early_stopping)
        explicit_keys.add("EARLY_STOPPING_ENABLE")
    if args.save_eval_generations is not None:
        overrides["SAVE_EVAL_GENERATIONS"] = bool(args.save_eval_generations)
        explicit_keys.add("SAVE_EVAL_GENERATIONS")

    explicit_hf_cache = args.hf_cache_dir is not None
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        key = k.strip()
        if key == "HF_CACHE_DIR":
            explicit_hf_cache = True
        overrides[key] = _parse_override_value(v)
        explicit_keys.add(key)

    if overrides:
        _apply_overrides(cfg_cls, overrides)
        print("[CONFIG] applied overrides:", ", ".join(sorted(overrides.keys())), flush=True)

    # Keep derived paths in sync when base dirs are overridden.
    if ("DPC_EXTRA_DIR" in explicit_keys) or ("INPUT_DIR" in explicit_keys):
        extra_base = str(getattr(cfg_cls, "DPC_EXTRA_DIR", ""))
        input_base = str(getattr(cfg_cls, "INPUT_DIR", ""))

        derived_extra = {
            "LEXICON_PATH": "OA_Lexicon_eBL.csv",
            "EBL_DICT_PATH": "eBL_Dictionary.csv",
            "SENTENCES_PATH": "Sentences_Oare_FirstWord_LinNum.csv",
            "LARSEN_LETTERS_PATH": "larsen_letters.csv",
            "ONOMASTICON_PATH": "onomasticon.csv",
        }
        for k, rel in derived_extra.items():
            if k not in explicit_keys:
                setattr(cfg_cls, k, os.path.join(extra_base, rel))

        derived_input = {
            "TRAIN_CSV_PATH": "train_sentence_clean.csv",
            "TRAIN_FINAL_CSV_PATH": "final_train_sentence.csv",
        }
        for k, rel in derived_input.items():
            if ("INPUT_DIR" in explicit_keys) and (k not in explicit_keys):
                setattr(cfg_cls, k, os.path.join(input_base, rel))

    if not explicit_hf_cache:
        cfg_cls.HF_CACHE_DIR = os.path.join(str(cfg_cls.OUTPUT_DIR), "hf-cache")


def _setup_distributed_device():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def _normalize_model_parallel_mode(raw: Any) -> str:
    if isinstance(raw, bool):
        return "on" if raw else "off"
    s = "auto" if raw is None else str(raw).strip().lower()
    aliases = {
        "1": "on",
        "true": "on",
        "yes": "on",
        "y": "on",
        "0": "off",
        "false": "off",
        "no": "off",
        "n": "off",
    }
    s = aliases.get(s, s)
    if s not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported MODEL_PARALLEL mode: {raw!r}. Expected one of auto/on/off.")
    return s


def _parse_model_parallel_devices(raw: Any) -> Optional[List[int]]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        raise ValueError(f"Unsupported MODEL_PARALLEL_DEVICES value: {raw!r}")

    device_ids = []
    for item in parts:
        s = str(item).strip()
        if not s:
            continue
        did = int(s)
        if did < 0:
            raise ValueError(f"CUDA device ids must be >= 0, got {did}.")
        device_ids.append(did)

    if not device_ids:
        return None
    return list(dict.fromkeys(device_ids))


def _get_model_stack(model: nn.Module, stack_name: str):
    stack = getattr(model, stack_name, None)
    if stack is not None:
        return stack
    inner = getattr(model, "model", None)
    if inner is not None:
        return getattr(inner, stack_name, None)
    return None


def _build_even_t5_device_map(num_layers: int, device_ids: Sequence[int]) -> Dict[int, List[int]]:
    active_devices = [int(d) for d in device_ids]
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}.")
    if not active_devices:
        raise ValueError("device_ids must not be empty.")

    if len(active_devices) > num_layers:
        active_devices = active_devices[:num_layers]

    base, rem = divmod(int(num_layers), len(active_devices))
    out: Dict[int, List[int]] = {}
    start = 0
    for pos, dev_id in enumerate(active_devices):
        take = base + (1 if pos < rem else 0)
        if take <= 0:
            continue
        out[int(dev_id)] = list(range(start, start + take))
        start += take

    if start != int(num_layers):
        raise RuntimeError(f"Failed to assign all layers for model parallel: assigned={start}, total={num_layers}.")
    return out


def _invert_layer_map(layer_map: Dict[int, List[int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for dev_id, layer_ids in layer_map.items():
        for layer_id in layer_ids:
            out[int(layer_id)] = int(dev_id)
    return out


def _resolve_t5_module_device_map(model_name: str, device_ids: Sequence[int]) -> Dict[str, int]:
    cfg = AutoConfig.from_pretrained(model_name)

    enc_layers = getattr(cfg, "num_layers", None)
    if enc_layers is None:
        enc_layers = getattr(cfg, "num_hidden_layers", None)
    dec_layers = getattr(cfg, "num_decoder_layers", None)
    if dec_layers is None:
        dec_layers = enc_layers

    if enc_layers is None or dec_layers is None:
        raise ValueError(
            f"Could not infer encoder/decoder layer counts from config for {model_name!r}."
        )

    enc_layers = int(enc_layers)
    dec_layers = int(dec_layers)
    if enc_layers < 1 or dec_layers < 1:
        raise ValueError(
            f"Invalid encoder/decoder layer counts for {model_name!r}: "
            f"encoder={enc_layers}, decoder={dec_layers}."
        )

    enc_assign = _invert_layer_map(_build_even_t5_device_map(enc_layers, device_ids))
    dec_assign = _invert_layer_map(_build_even_t5_device_map(dec_layers, device_ids))

    first_enc_device = enc_assign[0]
    last_enc_device = enc_assign[enc_layers - 1]
    first_dec_device = dec_assign[0]
    last_dec_device = dec_assign[dec_layers - 1]

    device_map: Dict[str, int] = {
        "shared": int(first_enc_device),
        "encoder.embed_tokens": int(first_enc_device),
        "encoder.dropout": int(first_enc_device),
        "encoder.final_layer_norm": int(last_enc_device),
        "decoder.embed_tokens": int(first_dec_device),
        "decoder.dropout": int(first_dec_device),
        "decoder.final_layer_norm": int(last_dec_device),
        "lm_head": int(last_dec_device),
    }
    for layer_id, dev_id in enc_assign.items():
        device_map[f"encoder.block.{int(layer_id)}"] = int(dev_id)
    for layer_id, dev_id in dec_assign.items():
        device_map[f"decoder.block.{int(layer_id)}"] = int(dev_id)
    return device_map


def _get_model_primary_device(model: nn.Module) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        for key in ("shared", "encoder.embed_tokens", "encoder", "decoder", "lm_head", ""):
            if key in hf_device_map:
                dev = hf_device_map[key]
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                if isinstance(dev, str) and dev not in {"cpu", "disk"}:
                    return torch.device(dev)
        for dev in hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev not in {"cpu", "disk"}:
                return torch.device(dev)
    for stack_name in ("encoder", "decoder"):
        stack = _get_model_stack(model, stack_name)
        first_device = getattr(stack, "first_device", None) if stack is not None else None
        if first_device is not None:
            return torch.device(str(first_device))
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_primary_device_str(model: nn.Module) -> str:
    dev = _get_model_primary_device(model)
    if isinstance(dev, torch.device):
        if dev.index is None:
            return dev.type
        return f"{dev.type}:{dev.index}"
    return str(dev)


def _configure_model_parallel(cfg_cls, *, rank: int, world_size: int) -> dict:
    mode = _normalize_model_parallel_mode(getattr(cfg_cls, "MODEL_PARALLEL", "auto"))
    info = {
        "enabled": False,
        "mode": mode,
        "device_ids": [],
        "device_map": None,
        "primary_device": None,
        "from_pretrained_kwargs": {},
    }

    if mode == "off":
        return info

    gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if world_size > 1:
        msg = (
            f"[MODEL_PARALLEL] requested mode={mode}, but WORLD_SIZE={world_size}. "
            "This script only supports model parallel in single-process training."
        )
        if mode == "on":
            raise ValueError(msg)
        if rank == 0:
            print(msg + " Skipping.", flush=True)
        return info

    if gpu_count < 2:
        msg = (
            f"[MODEL_PARALLEL] requested mode={mode}, but only {gpu_count} visible CUDA device(s) "
            "are available."
        )
        if mode == "on":
            raise ValueError(msg)
        if rank == 0:
            print(msg + " Skipping.", flush=True)
        return info

    device_ids = _parse_model_parallel_devices(getattr(cfg_cls, "MODEL_PARALLEL_DEVICES", None))
    if device_ids is None:
        device_ids = list(range(gpu_count))

    bad_ids = [d for d in device_ids if d >= gpu_count]
    if bad_ids:
        raise ValueError(
            f"MODEL_PARALLEL_DEVICES contains ids not visible to this process: {bad_ids}. "
            f"Visible devices are 0..{gpu_count - 1}."
        )
    if len(device_ids) < 2:
        msg = f"[MODEL_PARALLEL] need at least 2 devices, got {device_ids!r}."
        if mode == "on":
            raise ValueError(msg)
        if rank == 0:
            print(msg + " Skipping.", flush=True)
        return info

    device_map = _resolve_t5_module_device_map(str(getattr(cfg_cls, "MODEL_NAME", "")), device_ids)
    used_devices = sorted(
        {
            int(dev)
            for dev in device_map.values()
            if isinstance(dev, int)
        }
    )
    if len(used_devices) < 2:
        msg = (
            f"[MODEL_PARALLEL] resolved fewer than 2 active devices after layer sharding: {device_map}."
        )
        if mode == "on":
            raise ValueError(msg)
        if rank == 0:
            print(msg + " Skipping.", flush=True)
        return info

    if rank == 0:
        per_device_module_counts = Counter(int(dev) for dev in device_map.values() if isinstance(dev, int))
        printable_map = ", ".join(
            f"cuda:{dev}={int(per_device_module_counts[dev])} modules" for dev in sorted(per_device_module_counts)
        )
        print(f"[MODEL_PARALLEL] loading with device_map on {printable_map}", flush=True)

    info.update(
        enabled=True,
        device_ids=used_devices,
        device_map=device_map,
        primary_device=f"cuda:{used_devices[0]}",
        from_pretrained_kwargs={"device_map": device_map},
    )
    return info


def _dump_training_config(cfg_cls):
    keys = []
    for k in dir(cfg_cls):
        if k.startswith("_"):
            continue
        v = getattr(cfg_cls, k)
        if callable(v):
            continue
        keys.append(k)
    keys = sorted(keys)
    print("=" * 90, flush=True)
    print("[CONFIG] Effective training config", flush=True)
    for k in keys:
        try:
            v = getattr(cfg_cls, k)
            print(f"[CONFIG] {k} = {repr(v)}", flush=True)
        except Exception as e:
            print(f"[CONFIG] {k} = <error: {e}>", flush=True)
    print("=" * 90, flush=True)


def _ensure_writable_hf_cache_dir(path: str) -> str:
    p = str(path)
    try:
        os.makedirs(p, exist_ok=True)
        return p
    except PermissionError:
        fallback = os.path.join(os.getcwd(), ".cache", "hf-cache")
        os.makedirs(fallback, exist_ok=True)
        print(
            f"[CONFIG] HF_CACHE_DIR is not writable: {p}. Using fallback: {fallback}",
            flush=True,
        )
        return fallback


_apply_cli_overrides(Config)
Config.HF_CACHE_DIR = _ensure_writable_hf_cache_dir(
    getattr(Config, "HF_CACHE_DIR", ".cache/hf-cache")
)

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)    
os.makedirs(Config.HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", Config.HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", Config.HF_CACHE_DIR)

# -------------------------
# Repro
# -------------------------
def seed_everything(seed=Config.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ===== Notebook Cell 1 =====
# Preprocessing utilities (SOFT-HARMONIZED — behavior switches via mode="train"/"infer")
# - Host-aligned but NOT overly strict.
# - Controversial unit/grain rewrites removed.
# - Avoid breaking hyphen-attached gaps (U-<gap>, <gap>-Aššur).
# - Unicode fractions enforced ONLY in infer mode (host: hidden test has no decimals).

# ============================================================
# Canonical decimals (float-artifact squashing) — HOST: 4 decimals
# ============================================================
_ALLOWED_FRACS = [
    (1.0 / 6.0, "0.1666"),
    (1.0 / 4.0, "0.25"),
    (1.0 / 3.0, "0.3333"),
    (1.0 / 2.0, "0.5"),
    (2.0 / 3.0, "0.6666"),
    (3.0 / 4.0, "0.75"),
    (5.0 / 6.0, "0.8333"),
]
_FRAC_TOL = 2e-3
_FLOAT_ARTIFACT_RE = re.compile(r"(?<![\w/])(\d+\.\d{6,})(?![\w/])")


def _canon_decimal_str(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = None
    for v, dec in _ALLOWED_FRACS:
        d = abs(frac - v)
        if best is None or d < best[0]:
            best = (d, dec)
    if best and best[0] <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}{dec}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


def normalize_float_artifacts(text: str) -> str:
    s = "" if text is None else str(text)

    def repl(m):
        raw = m.group(1)
        try:
            return _canon_decimal_str(float(raw))
        except Exception:
            return raw

    return _FLOAT_ARTIFACT_RE.sub(repl, s)


# ============================================================
# Decimals -> unicode fractions (infer-only)
# ============================================================
_DEC2UNICODE_FRAC = {
    "0.5": "½",
    "0.25": "¼",
    "0.3333": "⅓",
    "0.8333": "⅚",
    "0.625": "⅝",
    "0.6666": "⅔",
    "0.75": "¾",
    "0.1666": "⅙",
}

_DEC_TOKEN_RE = re.compile(
    r"(?<![\w/])"
    r"(\d+)\.(1666|25|3333|5|625|6666|75|8333)"
    r"(?![\w/])"
)


def decimals_to_unicode_fractions(s: str) -> str:
    if s is None:
        return ""
    t = str(s)

    def repl(m):
        ip = int(m.group(1))
        dec = f"0.{m.group(2)}"
        frac = _DEC2UNICODE_FRAC.get(dec)
        if not frac:
            return m.group(0)
        if ip == 0:
            return frac
        return f"{ip} {frac}"

    return _DEC_TOKEN_RE.sub(repl, t)


# ============================================================
# Gaps — canonicalize to "<gap>"
# ============================================================
_TAG_GAP_RE = re.compile(r"<\s*gap\s*>", re.I)
_TAG_BIGGAP_RE = re.compile(r"<\s*big[\s_\-]*gap\s*>", re.I)
_BARE_BIGGAP_RE = re.compile(r"\bbig[\s_\-]*gap\b", re.I)

_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+|\[\.+\])")
_BRACKET_X_RE = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))", re.I)
_XTOKEN_RUN_RE = re.compile(r"\bx(?:\s+x)+\b", re.I)
_XRUN_RE = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_XTOK_RE = re.compile(r"(?<!\w)x(?!\w)", re.I)

_BRACKET_LACUNA_RE = re.compile(r"\[\s*(?:x|\.|\s)+\s*\]", re.I)
_STAR_X_RE = re.compile(r"\*\s*x\b", re.I)
_BREAK_RE = re.compile(
    r"\(\s*(?:break|large\s+break|n\s+broken\s+lines|\d+\s+broken\s+lines|broken\s+lines?)\s*\)",
    re.I,
)

_WS_RE = re.compile(r"\s+")


def normalize_gaps(text: str) -> str:
    if text is None:
        return ""
    t = str(text)

    t = _TAG_BIGGAP_RE.sub("<gap>", t)
    t = _TAG_GAP_RE.sub("<gap>", t)
    t = _BARE_BIGGAP_RE.sub("<gap>", t)

    t = _BREAK_RE.sub("<gap>", t)

    t = _BRACKET_LACUNA_RE.sub("<gap>", t)
    t = _STAR_X_RE.sub("<gap>", t)

    t = _XTOKEN_RUN_RE.sub("<gap>", t)
    t = _ELLIPSIS_RE.sub("<gap>", t)
    t = _BRACKET_X_RE.sub("<gap>", t)
    t = _XRUN_RE.sub("<gap>", t)
    t = _XTOK_RE.sub("<gap>", t)

    return t


_GAP_TOKEN_RE = re.compile(r"^-?<gap>-?$", re.I)


def collapse_gap_runs_tokens(tokens: List[str], mode: str) -> List[str]:
    mode = (mode or "none").lower().strip()
    if mode in ("none", ""):
        return tokens
    if mode in ("big_only", "any2big"):
        mode = "single"

    def is_gap_tok(tok: str) -> bool:
        return bool(_GAP_TOKEN_RE.match(str(tok)))

    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if is_gap_tok(tokens[i]):
            j = i
            while j < n and is_gap_tok(tokens[j]):
                j += 1
            out.append("<gap>")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def space_gap_token_hyphen_safe(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    t = re.sub(r"(?<![\s\-])<gap>", " <gap>", t)
    t = re.sub(r"<gap>(?![\s\-])", "<gap> ", t)
    return t


_GAP_RUN_STR_RE = re.compile(
    r"(?i)"
    r"(?:"
    r"(?:^|(?<=\s))"
    r"-?\s*<gap>\s*-?"
    r"(?:\s*[\.,;:]\s*)?"
    r"(?:\s+|\s*$)"
    r"){2,}"
)


def collapse_gap_runs_string(s: str) -> str:
    if s is None:
        return ""
    t = str(s)

    t = re.sub(r"(?i)<gap>\s*-\s*<gap>", "<gap>-<gap>", t)
    t = re.sub(r"(?i)<gap>\s*[\.,;:]\s*<gap>", "<gap> <gap>", t)

    t = _GAP_RUN_STR_RE.sub(" <gap> ", t)
    return _WS_RE.sub(" ", t).strip()


# ============================================================
# ASCII/Oracc/ATF -> host diacritics
# ============================================================
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"})


def ascii_to_diacritics(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s


# ============================================================
# Determinatives alignment (host)
# ============================================================
_DET_DKI_RE = re.compile(r"\(\s*(d|ki)\s*\)", re.I)
_TUG_PARENS_RE = re.compile(r"\(\s*TÚG\s*\)")


def normalize_determinatives(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    t = _DET_DKI_RE.sub(lambda m: "{%s}" % m.group(1).lower(), t)
    t = _TUG_PARENS_RE.sub("TÚG", t)
    return t


# ============================================================
# Transliteration char cleanup (host-aligned)
# ============================================================
TRANSLIT_SPECIAL_CHAR_MAP = {
    "ḫ": "h",
    "Ḫ": "H",
    "ʾ": "",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "—": "-",
    "–": "-",
}
TRANSLIT_SPECIAL_SEQ_MAP = {"mₓ": "m", "zₓ": "z"}
_SUB_X = "ₓ"
_CHAR_TRANS = str.maketrans(TRANSLIT_SPECIAL_CHAR_MAP)


def normalize_silver_abbrev(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    t = re.sub(r"\bKÙ\.B\.(?=\s|$)", "KÙ.BABBAR", t)
    t = re.sub(r"\bKÙ\.B\b", "KÙ.BABBAR", t)
    return t


def normalize_external_transliteration(text: str, *, kb_to_silver: bool = True) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    s = ascii_to_diacritics(s)
    s = normalize_determinatives(s)
    s = normalize_gaps(s)

    for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
        s = s.replace(k, v)

    s = s.translate(_CHAR_TRANS).replace(_SUB_X, "")
    s = normalize_float_artifacts(s)

    s = normalize_silver_abbrev(s)
    if kb_to_silver:
        s = re.sub(r"\bKB\b", "KÙ.BABBAR", s)

    s = space_gap_token_hyphen_safe(s)
    s = " ".join(collapse_gap_runs_tokens(s.split(), "single"))
    s = collapse_gap_runs_string(s)

    s = _WS_RE.sub(" ", s).strip()
    return s


# ============================================================
# Translation normalizer (soft)
# ============================================================
_PN_RE = re.compile(r"\bPN\b")

_QUOTE_NORM_TRANS = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
})

_SOFT_GRAM_PARENS_RE = re.compile(
    r"""
    \(
      \s*
      (?:
        fem(?:\.)? |
        sing(?:\.)? |
        plur(?:\.)? |
        pl(?:\.)? |
        singular |
        plural |
        \? |
        \!
      )
      (?:\s*(?:[.;,]?\s*(?:fem|sing|plur|pl|singular|plural)(?:\.)?)\s*)*
      \s*
    \)
    """,
    re.I | re.VERBOSE,
)

_RE_GOLD = re.compile(r"(?<!\w)-gold\b")
_RE_TAX = re.compile(r"(?<!\w)-tax\b")
_RE_TEXTILES_DASH = re.compile(r"(?i)(?<!\w)-textiles\b")


def apply_host_translation_rewrites(text: str, *, enable: bool = True) -> str:
    if not enable:
        return "" if text is None else str(text)
    s = "" if text is None else str(text)
    s = _RE_GOLD.sub("pašallum gold", s)
    s = _RE_TAX.sub("šadduātum tax", s)
    s = _RE_TEXTILES_DASH.sub("kutānum textiles", s)
    return s


def normalize_external_translation(
    text: str,
    *,
    gap_collapse: str = "single",
    enable_host_rewrites: bool = True,
) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    s = normalize_gaps(s)
    s = _PN_RE.sub("<gap>", s)

    s = _SOFT_GRAM_PARENS_RE.sub(" ", s)
    s = s.translate(_QUOTE_NORM_TRANS)

    s = normalize_float_artifacts(s)

    if enable_host_rewrites:
        s = apply_host_translation_rewrites(s, enable=True)

    if gap_collapse and gap_collapse.lower().strip() not in ("none", ""):
        toks = collapse_gap_runs_tokens(s.split(), gap_collapse)
        s = " ".join(toks)

    s = collapse_gap_runs_string(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ============================================================
# Main transliteration preprocessor (mode switch)
# ============================================================
class OptimizedPreprocessor:
    def __init__(self, mode: str = "train"):
        self.mode = (mode or "train").lower().strip()
        self._char_trans = _CHAR_TRANS

    def preprocess_input_text(self, text: str) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        s = str(text)

        s = ascii_to_diacritics(s)
        s = normalize_determinatives(s)
        s = normalize_gaps(s)

        for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
            s = s.replace(k, v)

        s = s.translate(self._char_trans).replace(_SUB_X, "")
        s = normalize_float_artifacts(s)

        s = normalize_silver_abbrev(s)
        s = re.sub(r"\bKB\b", "KÙ.BABBAR", s)

        s = space_gap_token_hyphen_safe(s)
        s = " ".join(collapse_gap_runs_tokens(s.split(), "single"))
        s = collapse_gap_runs_string(s)

        s = _WS_RE.sub(" ", s).strip()
        return s

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        ser = pd.Series(texts).fillna("").astype(str)

        ser = ser.apply(ascii_to_diacritics)
        ser = ser.apply(normalize_determinatives)
        ser = ser.apply(normalize_gaps)

        for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
            ser = ser.str.replace(k, v, regex=False)

        ser = ser.str.translate(self._char_trans)
        ser = ser.str.replace(_SUB_X, "", regex=False)

        ser = ser.str.replace(
            _FLOAT_ARTIFACT_RE,
            lambda m: _canon_decimal_str(float(m.group(1))),
            regex=True,
        )

        ser = ser.apply(normalize_silver_abbrev)
        ser = ser.str.replace(r"\bKB\b", "KÙ.BABBAR", regex=True)

        ser = ser.apply(space_gap_token_hyphen_safe)
        ser = ser.apply(lambda x: " ".join(collapse_gap_runs_tokens(str(x).split(), "single")))
        ser = ser.apply(collapse_gap_runs_string)

        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()


# ============================================================
# Vectorized postprocessor for translations (mode switch)
# ============================================================
class VectorizedPostprocessor:
    def __init__(
        self,
        mode: str = "infer",
        *,
        aggressive: bool = True,
        empty_fallback: str = "",
        fix_repeats: bool = False,
        enable_gap_run_collapse: bool = True,
        enable_host_rewrites: bool = True,
    ):
        self.mode = (mode or "infer").lower().strip()
        self.aggressive = bool(aggressive)
        self.fix_repeats = bool(fix_repeats)
        self.empty_fallback = "" if empty_fallback is None else str(empty_fallback)

        self.enable_unicode_fractions = (self.mode == "infer")
        self.enable_gap_run_collapse = bool(enable_gap_run_collapse)
        self.enable_host_rewrites = bool(enable_host_rewrites)

        self.gap_collapse = "single" if self.mode == "infer" else "single"
        self._pn_re = _PN_RE
        self._soft_gram_parens_re = _SOFT_GRAM_PARENS_RE
        self._quote_norm_trans = _QUOTE_NORM_TRANS

        self.forbidden_chars = "—–<>⌈⌋⌊+ʾ"
        self.forbidden_trans = str.maketrans("", "", self.forbidden_chars)

        self.patterns = {
            "gap_legacy": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap_legacy": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "whitespace": _WS_RE,
            "punct_space": re.compile(r"\s+([.,:;])"),
            "repeated_punct": re.compile(r"([.,:;])\1+"),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
        }

        self._month_roman_re = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.IGNORECASE)
        self._roman2int = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}

    def _month_repl(self, m):
        r = m.group(1).upper()
        return f"Month {self._roman2int.get(r, r)}"

    def _collapse_gaps_str(self, t: str) -> str:
        toks = collapse_gap_runs_tokens(str(t).split(), self.gap_collapse)
        return " ".join(toks)

    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations)

        valid_mask = s.apply(lambda x: isinstance(x, str) and (len(x.strip()) > 0))
        if not bool(valid_mask.all()):
            s.loc[~valid_mask] = self.empty_fallback

        s = s.apply(normalize_gaps)
        s = s.str.replace(self._pn_re, "<gap>", regex=True)
        s = s.apply(lambda x: ("" if x is None else str(x)).translate(self._quote_norm_trans))
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True).str.strip()

        if self.aggressive:
            s = s.str.replace(self.patterns["gap_legacy"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap_legacy"], "<gap>", regex=True)

            s = s.str.replace(self._soft_gram_parens_re, " ", regex=True)

            s = s.apply(self._collapse_gaps_str)

            s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
            s = s.str.translate(self.forbidden_trans)
            s = s.str.replace("\x00GAP\x00", "<gap>", regex=False)

            s = s.str.replace(
                _FLOAT_ARTIFACT_RE,
                lambda m: _canon_decimal_str(float(m.group(1))),
                regex=True,
            )

            if self.enable_unicode_fractions:
                s = s.apply(decimals_to_unicode_fractions)

            if self.enable_host_rewrites:
                s = s.apply(lambda x: apply_host_translation_rewrites(x, enable=True))

            s = s.str.replace(self._month_roman_re, self._month_repl, regex=True)

            if self.enable_gap_run_collapse:
                s = s.apply(collapse_gap_runs_string)

            s = s.apply(space_gap_token_hyphen_safe)

            if self.fix_repeats:
                s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)

            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True).str.strip()

        if self.empty_fallback:
            s = s.replace("", self.empty_fallback)

        return s.tolist()

# ===== Notebook Cell 2 =====
# UTILITIES SWAMP 

# 1) Partial decoder reset 
def reset_t5_decoder(
    model,
    *,
    seed: int = 1234,
    alpha: float = 0.25,
    noise_std: float = 3e-4,
    shrink: float = 0.997,
    n_dec_blocks: int = 2,
) -> dict:
    cfg = getattr(model, "config", None)
    dec = getattr(model, "decoder", None) or getattr(getattr(model, "model", None), "decoder", None)
    blocks = (getattr(dec, "block", None) or getattr(dec, "layers", None)) if dec is not None else None

    if cfg is None or blocks is None:
        return {"skipped": True, "reason": "no_config_or_no_decoder_blocks"}

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    d_model = float(getattr(cfg, "d_model", 512))
    init_fac = float(getattr(cfg, "initializer_factor", 1.0))
    std_lin = init_fac * (d_model ** -0.5)

    @torch.no_grad()
    def _t5ish_init(m: nn.Module):
        if isinstance(m, nn.Linear):
            m.weight.normal_(0.0, std_lin)
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.normal_(0.0, init_fac)
            if m.padding_idx is not None:
                m.weight[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm) or "layernorm" in m.__class__.__name__.lower():
            if getattr(m, "weight", None) is not None:
                m.weight.fill_(1.0)
            if getattr(m, "bias", None) is not None:
                m.bias.zero_()

    @torch.no_grad()
    def _blend(dst: nn.Module, src: nn.Module, a: float):
        sd = dict(dst.named_parameters())
        ss = dict(src.named_parameters())
        for k, p in sd.items():
            q = ss.get(k)
            if q is None or p.data.shape != q.data.shape or (not p.data.is_floating_point()):
                continue
            p.data.mul_(1.0 - a).add_(q.data, alpha=a)

    @torch.no_grad()
    def _perturb(mod: nn.Module, ns: float, sh: float):
        for p in mod.parameters():
            if not p.is_floating_point():
                continue
            if sh != 1.0:
                p.mul_(sh)
            if ns > 0.0:
                p.add_(torch.randn_like(p) * ns)

    n = len(blocks)
    k = max(0, min(int(n_dec_blocks), int(n)))
    touched = list(range(n - k, n))

    with torch.no_grad():
        for i in touched:
            blk = blocks[i]
            fresh = copy.deepcopy(blk)
            fresh.apply(_t5ish_init)
            _blend(blk, fresh, float(alpha))
            _perturb(blk, float(noise_std), float(shrink))

    return {
        "skipped": False,
        "dec_blocks": n,
        "dec_touched": tuple(touched),
        "alpha": float(alpha),
        "noise_std": float(noise_std),
        "shrink": float(shrink),
        "seed": int(seed),
    }


def _hash_u32(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return int(x) & 0xFFFFFFFF


def _u01(u32: int) -> float:
    return float(int(u32) & 0xFFFFFFFF) / float(2**32)


# ------------------------------------------------------------
# 2) Shared tiny normalization helpers (DEDUPED)
# ------------------------------------------------------------
# punctuation strip used by multiple helpers (single source of truth)
_PUNCT_STRIP = "[](){}<>.,;:!?\"'“”„`´"

_SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_H_HAT_TRANS     = str.maketrans("ḫḪ", "hH")

_ROMAN_TAIL_RE   = re.compile(r"\s+[IVX]+$")                      # trailing I/II/III...
_ROMAN_ANY_RE    = re.compile(r"\s+(I|II|III|IV|V|VI|VII|VIII|IX|X)\b.*$")

_QUOTED_RE       = re.compile(r'"([^"]{1,80})"')
_DET_RE          = re.compile(r"(^|\-)\((d|m|f)\)", re.I)
_ELLIPSIS_RE     = re.compile(r"(…|\.\.\.)")
_NUM_RE          = re.compile(r"^\d+(\.\d+)?$")

_SUB_MAP         = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# conservative suffixes used by NB-ish candidate generation
_AKK_SUFFIXES = [
    "šu-nu", "šunu", "šunū",
    "šu", "ša", "ši", "šū",
    "ī",
]

def _key(s: str) -> str:
    # Keep case, normalize subscripts + ḫ.
    s = "" if s is None else str(s)
    return s.translate(_SUBSCRIPT_TRANS).translate(_H_HAT_TRANS).strip()

def _split_spellings(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = re.split(r"[;,\|/]+", cell)
    return [p.strip() for p in parts if p.strip()]

def _lemma_part(x: str) -> Optional[str]:
    # stable lemma key: remove trailing roman numerals; keep leftmost chunk
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = _ROMAN_TAIL_RE.sub("", s).strip()
    s = re.split(r"[\s/;]", s, maxsplit=1)[0].strip()
    return s if s else None

def _first_quoted_gloss(defn: str) -> Optional[str]:
    if defn is None:
        return None
    s = str(defn).strip()
    if not s or s.lower() == "nan":
        return None
    m = _QUOTED_RE.search(s)
    if not m:
        return None
    g = m.group(1).strip()
    if not g:
        return None
    # keep short; avoid long prose
    g = g.replace(",", " ").replace("/", " ")
    g = " ".join(g.split()[:4])
    return g or None

def _stable_u32(s: str) -> int:
    b = ("" if s is None else str(s)).encode("utf-8", errors="ignore")
    return zlib.crc32(b) & 0xFFFFFFFF

def _is_num(tok: str) -> bool:
    return bool(_NUM_RE.match("" if tok is None else str(tok).strip()))

def _norm(tok: str) -> str:
    tok = "" if tok is None else str(tok)
    tok = tok.strip().lower()
    tok = unicodedata.normalize("NFKC", tok)
    tok = tok.strip(".,;:()[]{}")
    if tok == "...":
        tok = "…"
    return tok

# ------------------------------------------------------------
# 3) Simple eBL dictionary -> glossary injection (same public names)
# ------------------------------------------------------------
EBL_PATH = Config.EBL_DICT_PATH

def _clean_word(w: str) -> str:
    w = "" if w is None else str(w).strip()
    return _ROMAN_TAIL_RE.sub("", w).strip()

def _short_gloss(defn: str) -> str | None:
    if defn is None or (isinstance(defn, float) and pd.isna(defn)):
        return None
    s = str(defn).strip()
    if not s:
        return None
    m = _QUOTED_RE.search(s)
    if m:
        g = m.group(1).strip()
        if g:
            return g
    s = re.split(r"[;(]", s, maxsplit=1)[0].strip()
    s = re.sub(r"\s+", " ", s).lstrip("= ").strip()
    words = s.split()
    return " ".join(words[:6]) if words else None

def load_ebl_lexicon(path=EBL_PATH, min_len=2):
    df = pd.read_csv(path)
    lex = defaultdict(list)

    for w, d, der in zip(df.get("word", []), df.get("definition", []), df.get("derived_from", [])):
        w = _clean_word(w)
        if len(w) < int(min_len):
            continue
        g = _short_gloss(d)
        if g:
            lex[w].append(g)

        if der is not None and not (isinstance(der, float) and pd.isna(der)):
            der = _clean_word(der)
            if der and g:
                lex[der].append(g)

    return {k: list(dict.fromkeys(v)) for k, v in lex.items()}

def normalize_src_token(tok: str) -> str:
    t = ("" if tok is None else str(tok)).strip().strip(_PUNCT_STRIP)
    return re.sub(r"\d+$", "", t)

def find_glossary_terms(src: str, lex: dict, max_terms=8):
    toks = [normalize_src_token(t) for t in ("" if src is None else str(src)).split()]
    hits, seen = [], set()
    for t in toks:
        if not t or t in seen:
            continue
        if t in lex:
            hits.append(t)
            seen.add(t)
            if len(hits) >= int(max_terms):
                break
    return hits

def add_glossary_to_source(src: str, lex: dict, max_terms=8, drop_prob=0.5):
    # IMPORTANT: never include literal </s> for ByT5 (EOS). Use <extra_id_0>.
    src = "" if src is None else str(src)
    if drop_prob and random.random() < float(drop_prob):
        return src

    terms = find_glossary_terms(src, lex, max_terms=max_terms)
    if not terms:
        return src

    items = []
    for t in terms:
        if t in lex and lex[t]:
            items.append(f"{t}={lex[t][0]}")
    if not items:
        return src

    return f"{src} <extra_id_0> GLOSSARY: " + " ; ".join(items)

# ------------------------------------------------------------
# 4) NB-ish surface normalization + candidate generation (kept)
# ------------------------------------------------------------
def _norm_form(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.strip().strip(_PUNCT_STRIP)
    s = _DET_RE.sub(r"\1", s)
    s = s.replace("[", "").replace("]", "")
    s = s.translate(_SUB_MAP)
    s = _ELLIPSIS_RE.sub("", s)
    return s.strip()

def _suffix_strips(s: str, *, max_depth: int = 2) -> list[str]:
    out = set()
    cur = {_norm_form(s)}
    for _ in range(int(max_depth)):
        nxt = set()
        for x in cur:
            for suf in _AKK_SUFFIXES:
                for lead in ("-", ""):
                    tail = lead + suf
                    if x.endswith(tail) and len(x) > len(tail) + 2:
                        base = x[:-len(tail)].rstrip("-")
                        if base and base not in out:
                            nxt.add(base)
        out |= nxt
        cur = nxt
        if not cur:
            break
    return list(out)

def _candidates_for_form(t: str) -> list[str]:
    raw = "" if t is None else str(t).strip()
    if not raw:
        return []

    cands, seen = [], set()
    def add(x):
        x = "" if x is None else str(x).strip()
        if x and x not in seen:
            seen.add(x)
            cands.append(x)

    add(raw)
    n0 = _norm_form(raw)
    add(n0)
    add(raw.translate(_SUB_MAP))
    for v in _suffix_strips(n0, max_depth=2):
        add(v)
        add(v.translate(_SUB_MAP))
    return cands

def _is_junk_surface(surface: str) -> bool:
    s = "" if surface is None else str(surface).strip()
    if not s:
        return True
    if s in {"<gap>", "<big_gap>", "x", "x.", "…", "...", "[...]", "…]", "[…"}:
        return True
    if re.fullmatch(r"\d+([.:]\d+)?", s):
        return True
    core = s.strip(_PUNCT_STRIP)
    return not bool(core)

def augment_multiview(
    df: pd.DataFrame,
    canonicalizer: SourceCanonicalizer,
    views: Tuple[str, ...] = ("original", "pn_norm"),
) -> pd.DataFrame:
    outs = []
    for view in views:
        tmp = df.copy()
        tmp["src_view"] = view
        tmp["transliteration"] = tmp["transliteration"].astype(str).map(lambda x: canonicalizer.canonicalize_source(x, mode=view))
        outs.append(tmp)
    return pd.concat(outs, ignore_index=True)

def split_by_oare_id(train_df: pd.DataFrame, val_ratio: float = 0.05, seed: int = 42) -> Tuple[Set[str], Set[str]]:
    ids = train_df["oare_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * float(val_ratio)))
    val_ids = set(ids[:n_val])
    tr_ids = set(ids[n_val:])
    return tr_ids, val_ids

def n_words(s: str) -> int:
    return 0 if pd.isna(s) else len(str(s).split())

# -------------------------
# HF type checks
# -------------------------
def _is_hf_dataset(x) -> bool:
    return (Dataset is not None) and isinstance(x, Dataset)

def _is_hf_datasetdict(x) -> bool:
    return (DatasetDict is not None) and isinstance(x, DatasetDict)

def _safe_str(x) -> str:
    return "" if x is None else str(x)

# -------------------------
# text cleaning / schema inference
# -------------------------
_WS_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_BAD_EMPTY = {"", "nan", "none", "null", "na", "n/a", "<na>"}

def _clean_text(s) -> str:
    if s is None:
        return ""
    s = _ZW_RE.sub("", str(s))
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = unicodedata.normalize("NFKC", s).strip()
    s = _WS_RE.sub(" ", s)
    return "" if s.lower() in _BAD_EMPTY else s

def _norm_key(s: str) -> str:
    s = _clean_text(s).lower()
    return s.strip(" .,:;\"'“”‘’()[]{}")

def _pick_col(columns, candidates):
    cols_low = {c.lower(): c for c in columns}
    for cand in candidates:
        c = cols_low.get(cand.lower())
        if c is not None:
            return c
    return None

def _infer_schema(df: pd.DataFrame):
    cols = list(df.columns)
    src_col = _pick_col(cols, ["new_transliteration_sentence","new_transliteration","transliteration", "translit", "source", "src", "akkadian"])
    tgt_col = _pick_col(cols, ["new_translation_sentence","new_translation","translation", "english", "target", "tgt", "en"])
    id_col  = _pick_col(cols, ["oare_id", "text_uuid", "id", "uuid", "text_id"])
    if src_col is None or tgt_col is None:
        raise ValueError(f"Could not infer src/tgt columns. Columns: {cols}")
    return src_col, tgt_col, id_col

# ============================================================
# 1) flag_incomplete (SINGLE FN: pandas + HF Dataset)
# ============================================================
def _make_incomplete_reasons_df(df: pd.DataFrame, *, ratio_max: float = 0.50) -> pd.DataFrame:
    pre = OptimizedPreprocessor()
    post = VectorizedPostprocessor(aggressive=True)

    raw_src = df["transliteration"].astype(str).fillna("").str.strip()
    raw_tgt = df["translation"].astype(str).fillna("").str.strip()
    pre_src  = pd.Series(pre.preprocess_batch(raw_src.tolist()), index=df.index)
    post_tgt = pd.Series(post.postprocess_batch(raw_tgt.tolist()), index=df.index)

    def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        ac, bc = a.str.len(), b.str.len()
        return (bc / ac.replace(0, np.nan)).fillna(0.0)

    rr_ratio = _ratio(raw_src, raw_tgt)
    pr_ratio = _ratio(pre_src, raw_tgt)
    rp_ratio = _ratio(raw_src, post_tgt)
    pp_ratio = _ratio(pre_src, post_tgt)

    raw_src_c = raw_src.str.len()
    raw_tgt_c = raw_tgt.str.len()
    header_only = raw_tgt.str.lower().str.startswith("to ") & (raw_src_c >= 80) & (raw_tgt_c <= 60)

    hit_rr = rr_ratio <= float(ratio_max)
    hit_pr = pr_ratio <= float(ratio_max)
    hit_rp = rp_ratio <= float(ratio_max)
    hit_pp = pp_ratio <= float(ratio_max)

    out = pd.DataFrame(
        {
            "src_chars": raw_src_c,
            "tgt_chars": raw_tgt_c,
            "rr_ratio": rr_ratio,
            "pr_ratio": pr_ratio,
            "rp_ratio": rp_ratio,
            "pp_ratio": pp_ratio,
            "header_only": header_only,
            "hit_rr": hit_rr,
            "hit_pr": hit_pr,
            "hit_rp": hit_rp,
            "hit_pp": hit_pp,
        },
        index=df.index,
    )
    out["flag"] = header_only | hit_rr | hit_pr | hit_rp | hit_pp
    return out

def flag_incomplete(
    data,
    *,
    ratio_max: float = 0.60,
):
    thr = float(ratio_max)

    # ---- pandas
    if isinstance(data, pd.DataFrame):
        reasons = _make_incomplete_reasons_df(data, ratio_max=thr)
        mask = reasons["flag"]
        return mask

    # ---- HF Dataset
    if _is_hf_dataset(data):
        pre = OptimizedPreprocessor()
        post = VectorizedPostprocessor(aggressive=True)

        srcs = [_safe_str(x).strip() for x in data["transliteration"]]
        tgts = [_safe_str(x).strip() for x in data["translation"]]
        pre_src  = pre.preprocess_batch(srcs)
        post_tgt = post.postprocess_batch(tgts)

        n = len(srcs)
        src_c = np.fromiter((len(s) for s in srcs), dtype=np.int64, count=n)
        tgt_c = np.fromiter((len(t) for t in tgts), dtype=np.int64, count=n)

        def _ratio(a_list, b_list):
            out = np.empty(n, dtype=np.float32)
            for i, (a, b) in enumerate(zip(a_list, b_list)):
                la = len(a)
                out[i] = (len(b) / la) if la > 0 else 0.0
            return out

        rr_ratio = _ratio(srcs, tgts)
        pr_ratio = _ratio(pre_src, tgts)
        rp_ratio = _ratio(srcs, post_tgt)
        pp_ratio = _ratio(pre_src, post_tgt)

        header_only = np.fromiter(
            ((t.lower().startswith("to ") and sc >= 80 and tc <= 60) for t, sc, tc in zip(tgts, src_c, tgt_c)),
            dtype=np.bool_,
            count=n,
        )

        hit_rr = rr_ratio <= thr
        hit_pr = pr_ratio <= thr
        hit_rp = rp_ratio <= thr
        hit_pp = pp_ratio <= thr
        flag = header_only | hit_rr | hit_pr | hit_rp | hit_pp
        mask = np.asarray(flag, dtype=np.bool_)
        return mask

    if _is_hf_datasetdict(data):
        raise TypeError("flag_incomplete got a DatasetDict. Call it on a split: ds['train'].")

    raise TypeError(f"Unsupported type: {type(data)}. Expected pd.DataFrame or datasets.Dataset.")

# ============================================================
# 2) load_and_sanitize_parallel (extra loader)
# ============================================================
def load_and_sanitize_parallel(
    paths: str | Path | Sequence[str | Path],
    *,
    train_df: Optional[pd.DataFrame] = None,
    drop_if_in_train: bool = True,
    in_train_match: str = "either",   # "pair" | "src" | "either"
    out_id_col: str = "oare_id",
    out_src_col: str = "transliteration",
    out_tgt_col: str = "translation",
    drop_same_src_tgt: bool = True,
    dedupe_on: Sequence[str] = ("transliteration", "translation"),
    drop_incomplete: bool = False,
    incomplete_kwargs: Optional[Dict[str, Any]] = None,
    add_source_col: bool = True,
) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]

    pre = OptimizedPreprocessor()
    post = VectorizedPostprocessor(aggressive=True)

    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        src_col, tgt_col, id_col = _infer_schema(df)

        keep_cols = [c for c in (id_col, src_col, tgt_col) if c is not None]
        out = df[keep_cols].copy()

        ren = {src_col: out_src_col, tgt_col: out_tgt_col}
        if id_col is not None:
            ren[id_col] = out_id_col
        out = out.rename(columns=ren)

        if out_id_col not in out.columns:
            out[out_id_col] = [f"{p.stem}::{i}" for i in range(len(out))]

        out[out_id_col] = out[out_id_col].map(_clean_text)
        out[out_src_col] = out[out_src_col].map(_clean_text)
        out[out_tgt_col] = out[out_tgt_col].map(_clean_text)

        out = out[out[out_src_col].str.strip().ne("") & out[out_tgt_col].str.strip().ne("")].copy()

        if drop_same_src_tgt:
            out = out[out[out_src_col] != out[out_tgt_col]].copy()

        out["is_extra"] = True
        if add_source_col:
            out["source_file"] = p.name

        frames.append(out)

    extra = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=[out_id_col, out_src_col, out_tgt_col, "is_extra"] + (["source_file"] if add_source_col else [])
    )
    if len(extra) == 0:
        cols = [out_id_col, out_src_col, out_tgt_col, "is_extra"] + (["source_file"] if add_source_col else [])
        return extra[cols]

    raw_src = extra[out_src_col].astype(str).fillna("").str.strip()
    raw_tgt = extra[out_tgt_col].astype(str).fillna("").str.strip()
    extra["_pre_src"] = pd.Series(pre.preprocess_batch(raw_src.tolist()), index=extra.index)
    extra["_post_tgt"] = pd.Series(post.postprocess_batch(raw_tgt.tolist()), index=extra.index)

    if dedupe_on:
        extra["_tgt_len"] = raw_tgt.str.len().fillna(0).astype(int)
        extra = extra.sort_values("_tgt_len", ascending=False).drop_duplicates(list(dedupe_on), keep="first")
        extra = extra.drop(columns=["_tgt_len"])

    if drop_if_in_train and train_df is not None and len(train_df) > 0:
        tr_raw_src = train_df[out_src_col].astype(str).map(_norm_key)
        tr_raw_tgt = train_df[out_tgt_col].astype(str).map(_norm_key)

        tr_pre_src  = pd.Series(pre.preprocess_batch(train_df[out_src_col].astype(str).tolist())).map(_norm_key)
        tr_post_tgt = pd.Series(post.postprocess_batch(train_df[out_tgt_col].astype(str).tolist())).map(_norm_key)

        ex_raw_src  = extra[out_src_col].astype(str).map(_norm_key)
        ex_raw_tgt  = extra[out_tgt_col].astype(str).map(_norm_key)
        ex_pre_src  = extra["_pre_src"].astype(str).map(_norm_key)
        ex_post_tgt = extra["_post_tgt"].astype(str).map(_norm_key)

        if in_train_match == "pair":
            train_pairs = set(zip(tr_raw_src, tr_raw_tgt)) \
                        | set(zip(tr_pre_src, tr_raw_tgt)) \
                        | set(zip(tr_raw_src, tr_post_tgt)) \
                        | set(zip(tr_pre_src, tr_post_tgt))
            in_train = pd.Series(
                [((a, b) in train_pairs) or ((c, b) in train_pairs) or ((a, d) in train_pairs) or ((c, d) in train_pairs)
                 for a, b, c, d in zip(ex_raw_src, ex_raw_tgt, ex_pre_src, ex_post_tgt)],
                index=extra.index
            )
        elif in_train_match == "src":
            train_src = set(tr_raw_src) | set(tr_pre_src)
            in_train = ex_raw_src.isin(train_src) | ex_pre_src.isin(train_src)
        elif in_train_match == "either":
            train_src = set(tr_raw_src) | set(tr_pre_src)
            train_tgt = set(tr_raw_tgt) | set(tr_post_tgt)
            in_train = (
                ex_raw_src.isin(train_src) | ex_pre_src.isin(train_src) |
                ex_raw_tgt.isin(train_tgt) | ex_post_tgt.isin(train_tgt)
            )
        else:
            raise ValueError("in_train_match must be one of: 'pair', 'src', 'either'")

        extra = extra.loc[~in_train].reset_index(drop=True)

    if drop_incomplete and len(extra) > 0:
        kwargs = incomplete_kwargs or {}
        bad = flag_incomplete(extra, **kwargs)
        extra = extra.loc[~bad].reset_index(drop=True)

    extra = extra.drop(columns=[c for c in ("_pre_src", "_post_tgt") if c in extra.columns])

    cols = [out_id_col, out_src_col, out_tgt_col, "is_extra"] + (["source_file"] if add_source_col else [])
    return extra[cols]

# ============================================================
# 3) drop_duplicates_hf (NO report arg)
# ============================================================
def drop_duplicates_hf(
    ds: Union["Dataset", "DatasetDict"],
    *,
    src_col: str = "transliteration",
    tgt_col: str = "translation",
    rule: Literal["tgt", "pair"] = "tgt",
    keep: Literal["first", "last", "longest_src"] = "longest_src",
    normalize: bool = True,
    lowercase: bool = True,
) -> Union["Dataset", "DatasetDict"]:
    if isinstance(ds, DatasetDict):
        return DatasetDict({
            split: drop_duplicates_hf(
                d, src_col=src_col, tgt_col=tgt_col, rule=rule, keep=keep,
                normalize=normalize, lowercase=lowercase
            )
            for split, d in ds.items()
        })

    if src_col not in ds.column_names:
        raise ValueError(f"src_col='{src_col}' not in columns: {ds.column_names}")
    if tgt_col not in ds.column_names:
        raise ValueError(f"tgt_col='{tgt_col}' not in columns: {ds.column_names}")

    src_list = ds[src_col]
    tgt_list = ds[tgt_col]

    def _norm_local(x):
        s = _clean_text(x)
        return s.lower() if lowercase else s

    chosen, best = {}, {}
    for i in range(ds.num_rows):
        s = _norm_local(src_list[i]) if normalize else _safe_str(src_list[i])
        t = _norm_local(tgt_list[i]) if normalize else _safe_str(tgt_list[i])
        key = t if rule == "tgt" else (s, t)

        if key not in chosen:
            chosen[key] = i
            if keep == "longest_src":
                best[key] = len(_safe_str(src_list[i]))
            continue

        if keep == "first":
            continue
        if keep == "last":
            chosen[key] = i
            if keep == "longest_src":
                best[key] = len(_safe_str(src_list[i]))
            continue

        sc = len(_safe_str(src_list[i]))
        if sc > best[key]:
            chosen[key] = i
            best[key] = sc

    return ds.select(sorted(chosen.values()))


def sanitize_generation_config_for_saving(model, *, default_num_beams: int = 8, default_len_pen: float = 1.0):
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return

    gen_cfg = model.generation_config
    nb = int(getattr(gen_cfg, "num_beams", 1) or 1)

    if default_num_beams is not None and int(default_num_beams) > 1:
        nb = int(default_num_beams)

    gen_cfg.num_beams = nb
    if nb > 1:
        gen_cfg.length_penalty = float(getattr(gen_cfg, "length_penalty", default_len_pen) or default_len_pen)
        gen_cfg.early_stopping = bool(getattr(gen_cfg, "early_stopping", True))
    else:
        gen_cfg.length_penalty = 1.0
        gen_cfg.early_stopping = False

    model.generation_config = gen_cfg


def compute_warmup_steps(num_examples, per_device_bs, grad_accum, epochs, warmup_ratio=0.05):
    steps_per_epoch = math.ceil(num_examples / (per_device_bs * grad_accum))
    total_steps = steps_per_epoch * epochs
    return int(total_steps * warmup_ratio)

# ===== Notebook Cell 3 =====
# GlossAugmenter and PN canon
# ------------------------------------------------------------
class GlossAugmenter:
    """
    Uses OA_Lexicon to map source 'form' -> 'lexeme',
    then uses eBL_Dictionary to map lemma(lexeme) -> short English gloss.

    Updates:
    - glossary key uses the ORIGINAL src token that matched (not the normalized candidate)
    - gloss is shortened to 1 head-gloss (no long synonym strings)
    - unseen df==0 is NOT clamped by rare_df_floor (enhance unseen, still capped by idf_cap)
    - safe delimiter: "<extra_id_0>"
    """
    def __init__(
        self,
        oa_lexicon_path: str,
        ebl_dict_path: str,
        *,
        train_texts: Optional[list[str]] = None,
        idf_cap: float = 3.5,
        rare_df_floor: int = 3,
        df1_penalty: float = 0.65,
        base_weight: float = 0.01,

        # new (safe defaults)
        gloss_max_chars: int = 48,     # cap per-gloss text
        unseen_boost: float = 1.15,    # >1 boosts df==0 slightly
    ):
        lex = pd.read_csv(oa_lexicon_path)
        dic = pd.read_csv(ebl_dict_path)

        def _short_gloss(g: str) -> str:
            g = "" if g is None else str(g).strip()
            if not g:
                return ""
            # drop leading parenthetical: "(mythical ...) snake" -> "snake"
            g = re.sub(r"^\([^)]*\)\s*", "", g).strip()

            # keep only first sense
            # - split on ';' first (often separates senses)
            # - then split on ',' (often adds extra detail)
            g = g.split(";", 1)[0].strip()
            g = g.split(",", 1)[0].strip()

            # drop remaining parentheticals (optional but helps)
            g = re.sub(r"\([^)]*\)", "", g).strip()

            g = _norm_ws(g)
            if int(gloss_max_chars) > 0 and len(g) > int(gloss_max_chars):
                g = g[: int(gloss_max_chars)].rstrip()
            return g

        lemma2gloss: Dict[str, str] = {}
        for w, d in zip(dic["word"].astype(str), dic["definition"].astype(str)):
            lemma = _lemma_part(w)
            g0 = _first_quoted_gloss(d)
            gloss = _short_gloss(g0)
            if lemma and gloss and lemma not in lemma2gloss:
                lemma2gloss[lemma] = gloss

        self.form2lex = dict(zip(lex["form"].astype(str), lex["lexeme"].astype(str)))
        self.lex2gloss = lemma2gloss

        self._df: Dict[str, int] = {}
        self._N: int = 0

        if train_texts:
            from collections import Counter
            dfc = Counter()
            N = 0
            for s in train_texts:
                toks = ("" if s is None else str(s)).split()
                seen = set()
                for t in toks:
                    for cand in _candidates_for_form(t):
                        if cand:
                            seen.add(cand)
                for x in seen:
                    dfc[x] += 1
                N += 1
            self._df = dict(dfc)
            self._N = int(N)
        else:
            vc = lex["form"].astype(str).value_counts()
            self._df = {k: int(v) for k, v in vc.items()}
            self._N = int(vc.sum())

        self._idf_cap = float(idf_cap)
        self._rare_df_floor = int(rare_df_floor)
        self._df1_penalty = float(df1_penalty)
        self._base_weight = float(base_weight)
        self._unseen_boost = float(unseen_boost)

    def _weight_for_surface(self, surface: str) -> float:
        if _is_junk_surface(surface):
            return 0.0
        s = _norm_form(surface)
        df = int(self._df.get(s, 0))

        # IMPORTANT:
        # - unseen df==0: do NOT clamp to rare_df_floor (enhance unseen)
        # - seen df>0: you may clamp low dfs to avoid rare garbage
        if df <= 0:
            df_eff = 1
        else:
            df_eff = max(df, int(self._rare_df_floor))

        idf = math.log((self._N + 1.0) / (df_eff + 1.0))
        idf = max(0.0, min(float(idf), float(self._idf_cap)))

        w = float(self._base_weight) + idf

        # penalize df==1 (rare-but-seen) only; don't penalize unseen
        if df == 1:
            w *= float(self._df1_penalty)

        # boost unseen a bit (still capped by idf_cap)
        if df <= 0:
            w *= float(self._unseen_boost)

        # numbers tend to be noisy
        if re.search(r"\d{2,}", str(surface)):
            w *= 0.6

        return max(0.0, float(w))

    def append_gloss(
        self,
        src_text: str,
        max_items: int = 6,
        max_append_chars: int = 240,
        *,
        seed: int = 0,
        epoch: int = 0,
        example_id: Optional[int] = None,
        keep_order: bool = True,
    ) -> str:
        src_text = "" if src_text is None else str(src_text)

        # never include literal </s>
        if "</s>" in src_text:
            src_text = src_text.replace("</s>", "<eos>")

        if (not src_text.strip()) or int(max_items) <= 0 or int(max_append_chars) <= 0:
            return src_text

        toks = src_text.split()
        candidates = []  # (pos, src_tok, match_surface, gloss, weight)
        seen = set()

        for pos, t in enumerate(toks):
            if _is_junk_surface(t):
                continue

            lexeme = None
            match_surface = None
            for cand in _candidates_for_form(t):
                lexeme = self.form2lex.get(cand)
                if lexeme:
                    match_surface = cand
                    break
            if not lexeme:
                continue

            lemma = _lemma_part(lexeme)
            if not lemma:
                continue

            g = self.lex2gloss.get(lemma)
            if not g:
                continue

            # dedupe by (normalized src token, lemma, gloss)
            key = (_norm_form(t), lemma, g)
            if key in seen:
                continue
            seen.add(key)

            w = self._weight_for_surface(match_surface)
            if w <= 0:
                continue

            candidates.append((pos, t, match_surface, g, w))

        if not candidates:
            return src_text

        ex = int(example_id) if example_id is not None else 0
        mix = (_stable_u32(src_text) ^ int(seed) ^ (int(epoch) * 1000003) ^ (ex * 9176)) & 0xFFFFFFFF
        rng = random.Random(mix)

        k = min(int(max_items), len(candidates))

        # weighted sampling without replacement (Efraimidis–Spirakis)
        keys = []
        for j, (_, _, _, _, w) in enumerate(candidates):
            u = max(1e-12, rng.random())
            keys.append((-math.log(u) / max(1e-6, float(w)), j))
        keys.sort(key=lambda x: x[0])
        picked = [candidates[j] for _, j in keys[:k]]

        if keep_order:
            picked.sort(key=lambda x: x[0])

        parts, used = [], 0
        for _, src_tok, _match_surface, gloss, _w in picked:
            # KEY FIX: emit the ORIGINAL src token that matched
            part = f"{src_tok}={gloss}"
            add_len = len(part) + (3 if parts else 0)
            if used + add_len > int(max_append_chars):
                break
            parts.append(part)
            used += add_len

        if not parts:
            return src_text

        return src_text + " <extra_id_0> GLOSSARY: " + " ; ".join(parts)
    
# ------------------------------------------------------------
# 6) SourceCanonicalizer (same class + methods)
# ------------------------------------------------------------
@dataclass
class SourceCanonicalizer:
    pn_gn_map: Dict[str, str]
    ono_map: Dict[str, str]

    @classmethod
    def from_csvs(
        cls,
        lexicon_path: str,
        onomasticon_path: str,
        use_norm: bool = True,
    ) -> "SourceCanonicalizer":
        lex = pd.read_csv(lexicon_path)
        ono = pd.read_csv(onomasticon_path)

        lex = lex[lex["type"].isin(["PN", "GN"])].copy()
        lex["form"] = lex["form"].astype(str)
        lex["canon"] = (lex["norm"].astype(str) if use_norm else lex["lexeme"].astype(str))

        lex["canon_len"] = lex["canon"].str.len()
        lex = lex.sort_values(["form", "canon_len"]).drop_duplicates("form", keep="first")

        pn_gn_map: Dict[str, str] = {}
        for form, canon in zip(lex["form"].tolist(), lex["canon"].tolist()):
            pn_gn_map[_key(form)] = canon

        if "Alt_lex" in lex.columns:
            alt_src = pd.read_csv(lexicon_path)
            alt_src = alt_src[alt_src["type"].isin(["PN", "GN"])].copy()
            alt_src["form"] = alt_src["form"].astype(str)
            alt_src["canon"] = (alt_src["norm"].astype(str) if use_norm else alt_src["lexeme"].astype(str))
            for form, canon, alt in zip(
                alt_src["form"],
                alt_src["canon"],
                alt_src.get("Alt_lex", pd.Series([None] * len(alt_src))),
            ):
                for v in _split_spellings(alt):
                    pn_gn_map.setdefault(_key(v), canon)

        ono_map: Dict[str, str] = {}
        if {"Name", "Spellings_semicolon_separated"}.issubset(set(ono.columns)):
            for name, cell in zip(ono["Name"].astype(str), ono["Spellings_semicolon_separated"]):
                for v in _split_spellings(cell):
                    ono_map[_key(v)] = name
            for name in ono["Name"].astype(str).tolist():
                ono_map.setdefault(_key(name), name)

        return cls(pn_gn_map=pn_gn_map, ono_map=ono_map)

    def canonicalize_source(self, text: str, mode: str = "pn_norm") -> str:
        if mode == "original" or not isinstance(text, str) or not text.strip():
            return text if isinstance(text, str) else ""
        out = []
        for t in text.split():
            kt = _key(t)
            out.append(self.pn_gn_map.get(kt) or self.ono_map.get(kt) or t)
        return " ".join(out)

    def extract_canonical_names_from_source(self, text: str) -> Set[str]:
        names: Set[str] = set()
        if not isinstance(text, str) or not text.strip():
            return names
        for t in text.split():
            kt = _key(t)
            if kt in self.pn_gn_map:
                names.add(self.pn_gn_map[kt])
            elif kt in self.ono_map:
                names.add(self.ono_map[kt])
        return names

# ===== Notebook Cell 4 =====
# PROBE SWAPS + PROBE VIEW BUILDERS (FULL BLOCK, UPDATED)

def _det_choice(seq, h: int):
    return None if not seq else seq[int(h % len(seq))]

_WS_PAT_CACHE: dict[str, re.Pattern] = {}

def _ws_tok_pat(tok: str):
    t = ("" if tok is None else str(tok)).strip()
    key = t.casefold()
    pat = _WS_PAT_CACHE.get(key)
    if pat is None:
        pat = re.compile(r"(?<!\S)" + re.escape(t) + r"(?!\S)", flags=re.IGNORECASE)
        _WS_PAT_CACHE[key] = pat
    return pat

def _has_tok(src: str, tok: str) -> bool:
    return _ws_tok_pat(tok).search(src) is not None

def _case_like(new_tok: str, template_tok: str) -> str:
    if template_tok.isupper():
        return new_tok.upper()
    if template_tok.islower():
        return new_tok.lower()
    return new_tok

def _swap_src_tok(src: str, old: str, new: str) -> str:
    pat = _ws_tok_pat(old)
    def _repl(m):
        return _case_like(new, m.group(0))
    return pat.sub(_repl, src)

def _swap_tgt_word(tgt: str, old_word: str, new_word: str):
    """
    Replace old_word in English tgt, preserving plural if matched.
    Returns None if old_word not found.
    """
    m = re.search(r"\b" + re.escape(old_word) + r"s?\b", tgt, flags=re.IGNORECASE)
    if not m:
        return None
    matched = m.group(0)
    rep = (new_word + "s") if matched.lower().endswith("s") else new_word
    return tgt[:m.start()] + rep + tgt[m.end():]

# -------------------------
# dataset overlay
# -------------------------
class OverlayDataset(torch.utils.data.Dataset):
    """Full-length overlay: return overlay[idx] else base[idx]."""
    def __init__(self, base_ds, overlay_ds, full_to_elig_map):
        self.base = base_ds
        self.ov = overlay_ds
        self.map = full_to_elig_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        j = self.map[int(idx)]
        return self.ov[j] if j >= 0 else self.base[int(idx)]

# ============================================================
# Canonical paired swap lists
# ============================================================
RARE_COMMODITY_PAIRS = [
    ("SÍG.HI.A", "wool"),
    ("SZE", "barley"),
    ("Ì.GISz", "oil"),
    ("(TÚG)", "textile"),
]

COMMON_TITLE_PAIRS = [
    ("DAM.GÀR", "merchant"),
    ("LUGAL", "king"),
    ("DUB.SAR", "scribe"),
    ("UGULA", "overseer"),
    ("GUDU₄", "priest"),
]

RARE_TITLE_PAIRS = [
    ("rabi", "chief"),
    ("ummianum", "craftsman"),
    ("szamallum", "assistant"),
    ("šāpirum", "messenger"),
]

COMMON_RELATIONSHIP_PAIRS = [
    ("DUMU", "son"),
    ("DAM", "wife"),
]

RARE_RELATIONSHIP_PAIRS = [
    ("DUMU.MUNUS", "daughter"),
    ("AD", "father"),
    ("AMA", "mother"),
    ("SZESZ", "brother"),
]

COMMON_DOCUMENT_PAIRS = [
    ("DUB", "tablet"),
    ("KI", "place"),
]

RARE_DOCUMENT_PAIRS = [
    ("KIŠIB", "seal"),
    ("KIŠIB3", "seal"),
    ("IM", "tablet"),
]

VERB_FRAME_PAIRS = [
    ("a-dí-in", "gave"),
    ("a-dí-na", "give"),
    ("ù-ša-qal", "pay"),
    ("il₅-qé-ú", "received"),
    ("tal-qé", "took"),
]

RELATION_PAIRS = [
    ("DUMU", "son"),
    ("DAM", "wife"),
    ("ARAD", "servant"),
]

MEASURE_UNIT_PAIRS = [
    ("GÍN", "shekel"),
    ("ma-na", "mina"),
    ("ITU.KAM", "month"),
]

# Alias map (kept; not applied automatically to avoid behavior change)
TITLE_ALIASES = {
    "overseer": ["wakil", "wa-ak-lum"],
    "priest": ["ku-um-ru-um"],
    "chief": ["rabûm"],
    "scribe": ["ṭupšarrum"],
}

# ============================================================
# Term swap (paired, deterministic)
# ============================================================
def apply_term_swap_det(src: str, tgt: str, idx: int, seed: int, from_pairs, to_pairs):
    cands = []
    for old_src, old_tgt in from_pairs:
        if _has_tok(src, old_src) and re.search(r"\b" + re.escape(old_tgt) + r"s?\b", tgt, flags=re.IGNORECASE):
            cands.append((old_src, old_tgt))

    if not cands:
        return src, tgt, False

    h0 = _hash_u32(seed ^ (idx * 9176) ^ 0xA1B2C3)
    old_src, old_tgt = _det_choice(cands, h0)
    old_cf = str(old_src).casefold()

    reps = [(a, e) for (a, e) in to_pairs if str(a).casefold() != old_cf and not _has_tok(src, a)]
    if not reps:
        return src, tgt, False

    h1 = _hash_u32((seed + 1337) ^ (idx * 2654435761) ^ 0xD00DFEED)
    new_src, new_tgt = _det_choice(reps, h1)

    src2 = _swap_src_tok(src, old_src, new_src)
    tgt2 = _swap_tgt_word(tgt, old_tgt, new_tgt)
    if tgt2 is None:
        return src, tgt, False

    ok = (src2 != src) and (tgt2 != tgt)
    return (src2, tgt2, ok) if ok else (src, tgt, False)

# ============================================================
# NUMERIC_MEASURE_SWAP (paired)
# ============================================================
_NUM_POOL_DEFAULT = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,25,30,34,37,40,50,60,80,100]

def _find_ints(s: str):
    return re.findall(r"\b\d+\b", str(s))

def _swap_first_int(s: str, old: str, new: str) -> str:
    return re.sub(r"\b" + re.escape(old) + r"\b", str(new), str(s), count=1)

def apply_numeric_measure_swap_det(src: str, tgt: str, idx: int, seed: int) -> tuple[str, str, bool]:
    s2, t2, ok = apply_term_swap_det(src, tgt, idx, seed, MEASURE_UNIT_PAIRS, MEASURE_UNIT_PAIRS)
    if ok:
        return s2, t2, True

    src_nums = _find_ints(src)
    tgt_nums = _find_ints(tgt)
    common = [n for n in src_nums if n in set(tgt_nums)]
    if not common:
        return src, tgt, False

    h0 = _hash_u32(seed ^ (idx * 9176) ^ 0x13579BDF)
    old_n = _det_choice(common, h0)

    pool = [n for n in _NUM_POOL_DEFAULT if str(n) != str(old_n)]
    if not pool:
        return src, tgt, False

    h1 = _hash_u32((seed + 1337) ^ (idx * 2654435761) ^ 0x2468ACE0)
    new_n = _det_choice(pool, h1)

    src3 = _swap_first_int(src, str(old_n), str(new_n))
    tgt3 = _swap_first_int(tgt, str(old_n), str(new_n))
    ok = (src3 != src) and (tgt3 != tgt)
    return (src3, tgt3, ok) if ok else (src, tgt, False)

# ============================================================
# TWO_ENTITY_ORDER_SWAP (paired)
# ============================================================
_IGI_NAME_PAT = re.compile(r"\bIGI\s+([^\s]+)", flags=re.IGNORECASE)

def _swap_first_two_igi(src: str) -> tuple[str, bool]:
    ms = list(_IGI_NAME_PAT.finditer(src))
    if len(ms) < 2:
        return src, False
    name1 = ms[0].group(1)
    name2 = ms[1].group(1)

    s = src
    s = re.sub(r"\bIGI\s+" + re.escape(name1) + r"\b", "IGI __TMP__", s, count=1, flags=re.IGNORECASE)
    s = re.sub(r"\bIGI\s+" + re.escape(name2) + r"\b", f"IGI {name1}", s, count=1, flags=re.IGNORECASE)
    s = re.sub(r"\bIGI\s+__TMP__\b", f"IGI {name2}", s, count=1, flags=re.IGNORECASE)
    return s, s != src

_TGT_WIT_1 = re.compile(r"(in the presence of)\s+([^.,;]+?)\s+(and of)\s+([^.,;]+?)([.,;]|$)", flags=re.IGNORECASE)
_TGT_WIT_2 = re.compile(r"(Witnessed by)\s+([^.,;]+?)\s*,\s*(by)\s+([^.,;]+?)([.,;]|$)", flags=re.IGNORECASE)

def _swap_witness_tgt(tgt: str) -> tuple[str, bool]:
    m = _TGT_WIT_1.search(tgt)
    if m:
        pre, a, mid, b, tail = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        out = tgt[:m.start()] + f"{pre} {b} {mid} {a}{tail}" + tgt[m.end():]
        return out, out != tgt

    m = _TGT_WIT_2.search(tgt)
    if m:
        pre, a, mid, b, tail = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        out = tgt[:m.start()] + f"{pre} {b}, {mid} {a}{tail}" + tgt[m.end():]
        return out, out != tgt

    return tgt, False

def apply_two_entity_order_swap_det(src: str, tgt: str, idx: int, seed: int) -> tuple[str, str, bool]:
    src2, ok_s = _swap_first_two_igi(src)
    tgt2, ok_t = _swap_witness_tgt(tgt)
    ok = ok_s and ok_t
    return (src2, tgt2, ok) if ok else (src, tgt, False)

# ============================================================
# NEW PROBE 1: SLASH_OPTION_CHOICE_SWAP (tgt-only; survives postproc)
# ============================================================
_SLASH_OPT_RE = re.compile(r"\b([^/\n]{1,40})\s*/\s*([^/\n]{1,40})\b")

def _choose_slash_option(t: str, choose_right: bool) -> tuple[str, bool]:
    m = _SLASH_OPT_RE.search(t)
    if not m:
        return t, False
    left = m.group(1).strip()
    right = m.group(2).strip()
    rep = right if choose_right else left
    out = t[:m.start()] + rep + t[m.end():]
    return out, out != t

def apply_slash_option_choice_swap_det(src: str, tgt: str, idx: int, seed: int) -> tuple[str, str, bool]:
    tgt0 = "" if tgt is None else str(tgt)
    h = _hash_u32(seed ^ (idx * 9176) ^ 0xA55A5AA5)
    choose_right = bool(h & 1)
    tgt2, ok = _choose_slash_option(tgt0, choose_right=choose_right)
    return (src, tgt2, ok) if ok else (src, tgt, False)

# ============================================================
# NEW PROBE 2: JOINED_HYPHEN_COMPOUND_SWAP (tgt-only; survives postproc)
# ============================================================
_COMPOUNDS = [
    ("kutānu-textiles", "kutānu textiles"),
    ("import-tax", "import tax"),
    ("pre-emption", "pre emption"),
]

def _swap_compound(t: str, a: str, b: str) -> tuple[str, bool]:
    pat_a = re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)", flags=re.IGNORECASE)
    pat_b = re.compile(r"(?<!\w)" + re.escape(b) + r"(?!\w)", flags=re.IGNORECASE)

    if pat_a.search(t):
        out = pat_a.sub(b, t, count=1)
        return out, out != t
    if pat_b.search(t):
        out = pat_b.sub(a, t, count=1)
        return out, out != t
    return t, False

def apply_joined_hyphen_compound_swap_det(src: str, tgt: str, idx: int, seed: int) -> tuple[str, str, bool]:
    tgt0 = "" if tgt is None else str(tgt)
    h = _hash_u32(seed ^ (idx * 9176) ^ 0x1EE7C0DE)
    start = int(h % len(_COMPOUNDS))

    for j in range(len(_COMPOUNDS)):
        a, b = _COMPOUNDS[(start + j) % len(_COMPOUNDS)]
        tgt2, ok = _swap_compound(tgt0, a, b)
        if ok:
            return src, tgt2, True

    return src, tgt, False

# ============================================================
# Probe dataset builders
# ============================================================
def _build_probe_variants_core(
    *,
    train_text_ds,
    tokenizer,
    pre,
    prefix: str,
    src_max_length: int,
    tgt_max_length: int,
    K: int,
    base_seed: int,
    MAP_BS: int,
    NPROC: int,
    apply_fn,  # (src,tgt,idx,seed)->(src2,tgt2,ok)
):
    N = len(train_text_ds)
    probe_ds_list = []
    eligible_any = np.zeros((N,), dtype=np.bool_)

    for v in range(int(K)):
        v_seed = int(base_seed) + 1009 * int(v)

        def _map_fn(examples, indices):
            src_clean = pre.preprocess_batch(examples["transliteration"])
            tgt_clean = (
                pre.preprocess_batch(examples["translation"])
                if bool(getattr(Config, "POSTPROCESS_TARGETS", True))
                else examples["translation"]
            )

            inps, tgts, ok_list = [], [], []
            for s, t, idx in zip(src_clean, tgt_clean, indices):
                s2, t2, ok = apply_fn(str(s), str(t), int(idx), int(v_seed))
                inps.append(prefix + s2)
                tgts.append(t2)
                ok_list.append(bool(ok))

            out = tokenizer(inps, max_length=src_max_length, truncation=True, padding=False)
            lab = tokenizer(tgts, max_length=tgt_max_length, truncation=True, padding=False)
            out["labels"] = lab["input_ids"]
            out["input_length"] = [len(x) for x in out["input_ids"]]
            out["_probe_ok"] = ok_list
            return out

        tokenized = train_text_ds.map(
            _map_fn,
            with_indices=True,
            batched=True,
            batch_size=MAP_BS,
            num_proc=NPROC,
            remove_columns=train_text_ds.column_names,
        )

        ok = np.asarray(tokenized["_probe_ok"], dtype=np.bool_)
        eligible_any |= ok
        tokenized = tokenized.remove_columns(["_probe_ok"])
        probe_ds_list.append(tokenized)

    return probe_ds_list, eligible_any

# ============================================================
# Unified builder (MERGED)
# ============================================================
def build_probe_variants(
    *,
    train_text_ds,
    tokenizer,
    pre,
    prefix: str,
    src_max_length: int,
    tgt_max_length: int,
    category_name: str,
    K: int,
    base_seed: int,
    MAP_BS: int,
    NPROC: int,
    apply_fn=None,          # (src,tgt,idx,seed)->(src2,tgt2,ok)
    from_pairs=None,
    to_pairs=None,
):
    if apply_fn is None:
        if from_pairs is None or to_pairs is None:
            raise ValueError("build_probe_variants: provide apply_fn OR (from_pairs and to_pairs).")
        fp = list(from_pairs)
        tp = list(to_pairs)
        def apply_fn(src, tgt, idx, seed):
            return apply_term_swap_det(src, tgt, idx, seed, fp, tp)

    return _build_probe_variants_core(
        train_text_ds=train_text_ds,
        tokenizer=tokenizer,
        pre=pre,
        prefix=prefix,
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        K=K,
        base_seed=base_seed,
        MAP_BS=MAP_BS,
        NPROC=NPROC,
        apply_fn=apply_fn,
    )

# ============================================================
# Backward-compatible wrappers
# ============================================================
def build_probe_term_swap_variants(
    *,
    train_text_ds,
    tokenizer,
    pre,
    prefix: str,
    src_max_length: int,
    tgt_max_length: int,
    category_name: str,
    from_pairs,
    to_pairs,
    K: int,
    base_seed: int,
    MAP_BS: int,
    NPROC: int,
):
    return build_probe_variants(
        train_text_ds=train_text_ds,
        tokenizer=tokenizer,
        pre=pre,
        prefix=prefix,
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        category_name=category_name,
        K=K,
        base_seed=base_seed,
        MAP_BS=MAP_BS,
        NPROC=NPROC,
        from_pairs=from_pairs,
        to_pairs=to_pairs,
    )

def build_probe_custom_variants(
    *,
    train_text_ds,
    tokenizer,
    pre,
    prefix: str,
    src_max_length: int,
    tgt_max_length: int,
    category_name: str,
    K: int,
    base_seed: int,
    MAP_BS: int,
    NPROC: int,
    apply_fn,
):
    return build_probe_variants(
        train_text_ds=train_text_ds,
        tokenizer=tokenizer,
        pre=pre,
        prefix=prefix,
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        category_name=category_name,
        K=K,
        base_seed=base_seed,
        MAP_BS=MAP_BS,
        NPROC=NPROC,
        apply_fn=apply_fn,
    )

# ============================================================
# Build all probe views (merged add helper)
# ============================================================
def build_all_probe_views(*, train_orig, tokenizer, pre, MAP_BS, NPROC):
    tgt_max = int(getattr(Config, "TGT_MAX_LENGTH", getattr(Config, "GEN_MAX_NEW_TOKENS", 768)))

    probe_views = {}
    probe_elig  = {}

    pe = getattr(Config, "PROBE_ENABLE", {}) or {}

    def _add(cat_name: str, seed_off: int, *, from_pairs=None, to_pairs=None, apply_fn=None):
        ds_list, elig = build_probe_variants(
            train_text_ds=train_orig,
            tokenizer=tokenizer,
            pre=pre,
            prefix=Config.PREFIX,
            src_max_length=Config.SRC_MAX_LENGTH,
            tgt_max_length=tgt_max,
            category_name=cat_name,
            K=int(getattr(Config, "PROBE_VARIANTS", 2)),
            base_seed=int(getattr(Config, "PROBE_SEED", 2027)) + int(seed_off),
            MAP_BS=MAP_BS,
            NPROC=NPROC,
            from_pairs=from_pairs,
            to_pairs=to_pairs,
            apply_fn=apply_fn,
        )
        probe_views[cat_name] = ds_list
        probe_elig[cat_name]  = elig

    if pe.get("COMMODITY_RARE_SWAP", False):
        _add("COMMODITY_RARE_SWAP", 11, from_pairs=RARE_COMMODITY_PAIRS, to_pairs=RARE_COMMODITY_PAIRS)

    if pe.get("TITLE_CROSS_SWAP", False):
        _add("TITLE_CROSS_SWAP", 21, from_pairs=COMMON_TITLE_PAIRS, to_pairs=RARE_TITLE_PAIRS)

    if pe.get("RELATIONSHIP_RARE_SWAP", False):
        _add("RELATIONSHIP_RARE_SWAP", 31, from_pairs=RARE_RELATIONSHIP_PAIRS, to_pairs=RARE_RELATIONSHIP_PAIRS)

    if pe.get("DOCUMENT_RARE_SWAP", False):
        _add("DOCUMENT_RARE_SWAP", 41, from_pairs=RARE_DOCUMENT_PAIRS, to_pairs=RARE_DOCUMENT_PAIRS)

    if pe.get("TITLE_COMMON_SWAP", False):
        _add("TITLE_COMMON_SWAP", 51, from_pairs=COMMON_TITLE_PAIRS, to_pairs=COMMON_TITLE_PAIRS)

    if pe.get("VERB_FRAME_SWAP", False):
        _add("VERB_FRAME_SWAP", 61, from_pairs=VERB_FRAME_PAIRS, to_pairs=VERB_FRAME_PAIRS)

    if pe.get("RELATION_SWAP", False):
        _add("RELATION_SWAP", 71, from_pairs=RELATION_PAIRS, to_pairs=RELATION_PAIRS)

    if pe.get("NUMERIC_MEASURE_SWAP", False):
        _add("NUMERIC_MEASURE_SWAP", 81, apply_fn=apply_numeric_measure_swap_det)

    if pe.get("TWO_ENTITY_ORDER_SWAP", False):
        _add("TWO_ENTITY_ORDER_SWAP", 91, apply_fn=apply_two_entity_order_swap_det)

    # ---- NEW "not washed out" probes
    if pe.get("SLASH_OPTION_CHOICE_SWAP", False):
        _add("SLASH_OPTION_CHOICE_SWAP", 121, apply_fn=apply_slash_option_choice_swap_det)

    if pe.get("JOINED_HYPHEN_COMPOUND_SWAP", False):
        _add("JOINED_HYPHEN_COMPOUND_SWAP", 131, apply_fn=apply_joined_hyphen_compound_swap_det)

    return probe_views, probe_elig

# ===== Notebook Cell 5 =====
# variants builder 

# -------------------------
# Probe op registry
# -------------------------
def _get_enabled_probe_ops(Config):
    ops = []  # list of (name, fn(src,tgt,idx,seed)->(src2,tgt2,ok))
    pe = getattr(Config, "PROBE_ENABLE", {}) or {}

    def add_term(name, from_pairs, to_pairs):
        def fn(src, tgt, idx, seed):
            return apply_term_swap_det(src, tgt, idx, seed, from_pairs, to_pairs)
        ops.append((name, fn))

    def add_custom(name, apply_fn):
        def fn(src, tgt, idx, seed):
            return apply_fn(src, tgt, idx, seed)
        ops.append((name, fn))

    if pe.get("COMMODITY_RARE_SWAP", False):
        add_term("COMMODITY_RARE_SWAP", RARE_COMMODITY_PAIRS, RARE_COMMODITY_PAIRS)
    if pe.get("TITLE_CROSS_SWAP", False):
        add_term("TITLE_CROSS_SWAP", COMMON_TITLE_PAIRS, RARE_TITLE_PAIRS)
    if pe.get("RELATIONSHIP_RARE_SWAP", False):
        add_term("RELATIONSHIP_RARE_SWAP", RARE_RELATIONSHIP_PAIRS, RARE_RELATIONSHIP_PAIRS)
    if pe.get("DOCUMENT_RARE_SWAP", False):
        add_term("DOCUMENT_RARE_SWAP", RARE_DOCUMENT_PAIRS, RARE_DOCUMENT_PAIRS)
    if pe.get("TITLE_COMMON_SWAP", False):
        add_term("TITLE_COMMON_SWAP", COMMON_TITLE_PAIRS, COMMON_TITLE_PAIRS)
    if pe.get("VERB_FRAME_SWAP", False):
        add_term("VERB_FRAME_SWAP", VERB_FRAME_PAIRS, VERB_FRAME_PAIRS)
    if pe.get("RELATION_SWAP", False):
        add_term("RELATION_SWAP", RELATION_PAIRS, RELATION_PAIRS)
    if pe.get("NUMERIC_MEASURE_SWAP", False):
        add_custom("NUMERIC_MEASURE_SWAP", apply_numeric_measure_swap_det)
    if pe.get("TWO_ENTITY_ORDER_SWAP", False):
        add_custom("TWO_ENTITY_ORDER_SWAP", apply_two_entity_order_swap_det)

    # optional probes (only if you defined them elsewhere)
    if pe.get("SLASH_OPTION_CHOICE_SWAP", False):
        add_custom("SLASH_OPTION_CHOICE_SWAP", apply_slash_option_choice_swap_det)
    if pe.get("JOINED_HYPHEN_COMPOUND_SWAP", False):
        add_custom("JOINED_HYPHEN_COMPOUND_SWAP", apply_joined_hyphen_compound_swap_det)

    return ops


def _normalize_weights(names, wdict):
    if not names:
        return None
    w = np.asarray([float((wdict or {}).get(n, 1.0)) for n in names], dtype=np.float64)
    s = float(w.sum())
    return (w / (s + 1e-12)) if s > 0 else (np.ones_like(w) / len(w))


def _hash_str_u32(s: str) -> int:
    return int(zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF)


# -------------------------
# Deterministic hash helpers
# -------------------------
def _hash_u32(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x & 0xFFFFFFFF


def _u01(h: int) -> float:
    return (int(h) & 0xFFFFFFFF) / 4294967296.0


# -------------------------
# OK-only probe append (TEXT-level)
# -------------------------
def build_probe_append_text_ds(
    *,
    base_text_ds,
    pre,
    Config,
    p_probe: float,
    ops,
    seed: int,
    cat_weights: dict | None = None,
    attempt_mult: int = 40,
    enforce_unique: bool = True,
    debug_label: str | None = None,
):
    N = len(base_text_ds)
    M = int(round(float(p_probe) * N))
    label = str(debug_label or "probe")
    if N <= 0:
        print(f"[PROBE][{label}] skipped: empty base dataset (N=0).", flush=True)
        return None
    if M <= 0:
        print(f"[PROBE][{label}] skipped: target append size M=0 (p_probe={float(p_probe):.4f}, N={N}).", flush=True)
        return None
    if not ops:
        print(f"[PROBE][{label}] skipped: no enabled probe ops.", flush=True)
        return None

    names = [n for n, _ in ops]
    w = _normalize_weights(names, cat_weights)
    cw = np.cumsum(w)

    # SRC clean for swap matching + output (probe rows will store preprocessed SRC)
    src_clean = pre.preprocess_batch(list(base_text_ds["transliteration"]))

    # TGT cleaned for swap matching + output (ONLY if enabled; collapse gaps for stability)
    tgt_clean = list(base_text_ds["translation"])
    tgt_clean = ["" if x is None else str(x) for x in tgt_clean]

    oare_ids = list(base_text_ds["oare_id"]) if "oare_id" in base_text_ds.column_names else [None] * N
    pair_ids = list(base_text_ds["pair_id"]) if "pair_id" in base_text_ds.column_names else [None] * N
    is_sent  = list(base_text_ds["is_sentence"]) if "is_sentence" in base_text_ds.column_names else [False] * N

    rows = []
    seen = set()
    cat_try = defaultdict(int)
    cat_ok = defaultdict(int)
    dedup_skip = 0

    total_attempts = max(1, int(M) * int(attempt_mult))
    for a in range(total_attempts):
        i = int(_hash_u32((seed + 999) ^ (a * 9176) ^ 0x51CE) % N)

        u = _u01(_hash_u32((seed + 12345) ^ (a * 2654435761) ^ 0xABCD))
        j = int(np.searchsorted(cw, u, side="right"))
        j = min(max(j, 0), len(ops) - 1)
        cat, fn = ops[j]
        cat_try[cat] += 1

        s0 = str(src_clean[i])
        t0 = str(tgt_clean[i])

        mix_seed = (int(seed) ^ _hash_str_u32(cat) ^ (int(a) * 1009)) & 0xFFFFFFFF
        s2, t2, ok = fn(s0, t0, int(i), int(mix_seed))
        if not ok:
            continue
        cat_ok[cat] += 1

        if enforce_unique:
            key = (_hash_str_u32(s2), _hash_str_u32(t2))
            if key in seen:
                dedup_skip += 1
                continue
            seen.add(key)

        oid = oare_ids[i]
        pid = pair_ids[i]
        rows.append({
            "oare_id": oid,
            "pair_id": (
                f"{str(oid)}::probe::{cat}::{i}::{len(rows)}"
                if oid is not None else f"probe::{cat}::{i}::{len(rows)}"
            ),
            "transliteration": s2,          # already SRC-preprocessed
            "translation": t2,              # conformance-postprocessed iff POSTPROCESS_TARGETS=True
            "src_is_preprocessed": True,    # IMPORTANT FIX
            "is_sentence": bool(is_sent[i]),
            "src_view": f"probe::{cat}",
            "base_pair_id": pid,
        })

        if len(rows) >= M:
            break

    if not rows:
        tried = ", ".join(f"{k}:{cat_try[k]}/{cat_ok.get(k, 0)}" for k in names)
        print(
            f"[PROBE][{label}] fallback: no rows produced | "
            f"target={M} attempts={total_attempts} dedup_skip={dedup_skip} ops={tried}",
            flush=True,
        )
        return None

    if len(rows) < M:
        tried = ", ".join(f"{k}:{cat_try[k]}/{cat_ok.get(k, 0)}" for k in names)
        print(
            f"[PROBE][{label}] partial fill: produced={len(rows)}/{M} "
            f"attempts={total_attempts} dedup_skip={dedup_skip} ops={tried}",
            flush=True,
        )
    else:
        print(
            f"[PROBE][{label}] produced={len(rows)}/{M} attempts={total_attempts} dedup_skip={dedup_skip}",
            flush=True,
        )

    ds_probe = Dataset.from_list(rows)

    # Align overlapping column dtypes to base dataset to avoid HF concat schema errors
    # (e.g., Value("string") vs Value("large_string") on oare_id).
    if hasattr(base_text_ds, "features") and hasattr(ds_probe, "features"):
        base_feats = base_text_ds.features
        for col in ds_probe.column_names:
            if col not in base_feats:
                continue
            src_feat = ds_probe.features[col]
            tgt_feat = base_feats[col]
            if src_feat == tgt_feat:
                continue
            try:
                ds_probe = ds_probe.cast_column(col, tgt_feat)
                print(
                    f"[PROBE][{label}] cast column '{col}' from {src_feat} to {tgt_feat}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[PROBE][{label}] warning: failed to cast '{col}' from {src_feat} to {tgt_feat}: {e}",
                    flush=True,
                )

    return ds_probe


# -------------------------
# Tokenization maps (labels gated by POSTPROCESS_TARGETS)
# -------------------------
def _get_src_clean_batch(pre, examples):
    if "src_is_preprocessed" in examples:
        flags = examples["src_is_preprocessed"]
        if isinstance(flags, list) and any(bool(x) for x in flags):
            out = []
            for s, f in zip(examples["transliteration"], flags):
                if bool(f):
                    out.append("" if s is None else str(s))
                else:
                    out.append(pre.preprocess_input_text(s))
            return out
    return pre.preprocess_batch(examples["transliteration"])


def make_map_raw(Config, tokenizer, pre, *, tgt_max: int):
    def _fn(examples):
        src_clean = _get_src_clean_batch(pre, examples)
        inputs = [Config.PREFIX + str(x) for x in src_clean]
        targets = list(examples["translation"])
        targets = ["" if x is None else str(x) for x in targets]

        out = tokenizer(inputs, max_length=int(Config.SRC_MAX_LENGTH), truncation=True, padding=False)
        lab = tokenizer(targets, max_length=int(tgt_max), truncation=True, padding=False)
        out["labels"] = lab["input_ids"]
        out["input_length"] = [len(x) for x in out["input_ids"]]
        return out
    return _fn


def make_map_pn(Config, tokenizer, pre, canon, *, tgt_max: int):
    def _fn(examples):
        src_clean = _get_src_clean_batch(pre, examples)
        src_pn = [canon.canonicalize_source(s, mode="pn_norm") for s in src_clean]

        inputs = [Config.PREFIX + str(x) for x in src_pn]
        targets = list(examples["translation"])
        targets = ["" if x is None else str(x) for x in targets]

        out = tokenizer(inputs, max_length=int(Config.SRC_MAX_LENGTH), truncation=True, padding=False)
        lab = tokenizer(targets, max_length=int(tgt_max), truncation=True, padding=False)
        out["labels"] = lab["input_ids"]
        out["input_length"] = [len(x) for x in out["input_ids"]]
        return out
    return _fn


def make_map_gloss_raw(Config, tokenizer, pre, glosser, *, seed_for_variant: int, tgt_max: int):
    def _fn(examples, indices):
        src_clean = _get_src_clean_batch(pre, examples)
        src_g = [
            glosser.append_gloss(
                s,
                max_items=int(getattr(Config, "GLOSS_MAX_ITEMS", 6)),
                max_append_chars=int(getattr(Config, "GLOSS_MAX_APPEND_CHARS", 240)),
                seed=int(seed_for_variant),
                epoch=0,
                example_id=int(idx),
                keep_order=True,
            )
            for s, idx in zip(src_clean, indices)
        ]

        inputs = [Config.PREFIX + str(x) for x in src_g]
        targets = list(examples["translation"])
        targets = ["" if x is None else str(x) for x in targets]

        out = tokenizer(inputs, max_length=int(Config.SRC_MAX_LENGTH), truncation=True, padding=False)
        lab = tokenizer(targets, max_length=int(tgt_max), truncation=True, padding=False)
        out["labels"] = lab["input_ids"]
        out["input_length"] = [len(x) for x in out["input_ids"]]
        return out
    return _fn

class EpochVariantMinViewMix3(torch.utils.data.Dataset):
    """
    MIN length across K variants + per-epoch variant + per-sample view mix:
      raw / pn / raw+gloss
    Also supports column access like HF Dataset: ds["input_length"].
    """
    def __init__(
        self,
        raws, pns, glosses,
        *,
        shared_epoch: Value,
        seed: int = 42,
        p_pn: float = 0.5,
        p_gloss: float = 0.5,
        mix_seed: int | None = None,
        length_key: str = "input_length",
        length_from: str = "raw",   # "raw" or "min" (raw is fastest)
    ):
        self.raws   = list(raws)
        self.pns    = list(pns)
        self.glosses= list(glosses)

        assert len(self.raws) >= 1
        K = len(self.raws)
        assert len(self.pns) == K and len(self.glosses) == K
        self.K = K

        # ---- IMPORTANT: safe min length across ALL views ----
        self.Ls = []
        for v in range(K):
            L = min(len(self.raws[v]), len(self.pns[v]), len(self.glosses[v]))
            self.Ls.append(int(L))
        self.L = int(min(self.Ls))

        self.shared_epoch = shared_epoch
        self.seed = int(seed)
        self.p_pn = float(p_pn)
        self.p_gl = float(p_gloss)
        self.mix_seed = int(mix_seed if mix_seed is not None else (self.seed + 991))

        # ---- cache lengths so Trainer length-grouping is instant ----
        self._length_key = str(length_key)
        self._lengths = None
        try:
            if length_from == "raw":
                # fastest: just take raw variant 0 lengths (slice to self.L)
                self._lengths = list(self.raws[0][self._length_key])[: self.L]
            else:
                # conservative: min across views for variant 0
                a = list(self.raws[0][self._length_key])[: self.L]
                b = list(self.pns[0][self._length_key])[: self.L]
                c = list(self.glosses[0][self._length_key])[: self.L]
                self._lengths = [min(x, y, z) for x, y, z in zip(a, b, c)]
        except Exception:
            self._lengths = None

    def __len__(self):
        return self.L

    def _epoch(self):
        try:
            return int(self.shared_epoch.value)
        except Exception:
            return 0

    def __getitem__(self, idx):
        # ---- KEY FIX: allow Trainer to do dataset["input_length"] ----
        if isinstance(idx, str):
            if idx == self._length_key and self._lengths is not None:
                return self._lengths
            # allow passthrough of other columns from raw[0] if you ever need it
            return self.raws[0][idx][: self.L]

        idx = int(idx)
        if idx < 0 or idx >= self.L:
            raise IndexError(idx)

        ep = self._epoch()

        v = int(_hash_u32(self.seed ^ (ep * 1000003) ^ 0xA5A5A5A5) % self.K)

        # (extra safety if some variant is shorter)
        if idx >= self.Ls[v]:
            # map into valid range deterministically
            idx = int(idx % self.Ls[v])

        u_pn = _u01(_hash_u32(self.mix_seed ^ (ep * 1000003) ^ (idx * 9176) ^ 0xC0FFEE))
        u_gl = _u01(_hash_u32((self.mix_seed + 1337) ^ (ep * 19217) ^ (idx * 2654435761) ^ 0xA53C))

        use_pn = (self.p_pn > 0.0) and (u_pn < self.p_pn)
        use_gl = (self.p_gl > 0.0) and (u_gl < self.p_gl)

        if use_gl:
            return self.glosses[v][idx]
        if use_pn:
            return self.pns[v][idx]
        return self.raws[v][idx]
    
# -------------------------
# Main builder: K train variants + val (raw+probe only)
# -------------------------
def build_probe_then_pngloss_variants(
    *,
    Config,
    train_text_ds,
    val_text_ds,
    tokenizer,
    pre,
    canon,
    glosser,
    NPROC: int = 8,
    MAP_BS: int = 1024,
):
    K = int(getattr(Config, "K_TRAIN_VARIANTS", 4))
    p_probe_tr = float(getattr(Config, "PROBE_APPEND_P", 0.0))
    p_probe_va = float(getattr(Config, "VAL_PROBE_APPEND_P", 0.0))

    base_seed  = int(getattr(Config, "SEED", 42))
    probe_seed = int(getattr(Config, "PROBE_SEED", base_seed + 3))
    gloss_seed = int(getattr(Config, "GLOSS_SEED", base_seed + 777))
    tgt_max = int(getattr(Config, "TGT_MAX_LENGTH", getattr(Config, "GEN_MAX_NEW_TOKENS", 768)))

    # mixing knobs
    p_pn = float(getattr(Config, "PN_MIX_P", 0.5))
    p_gl = float(getattr(Config, "GLOSS_MIX_P", 0.5))
    mix_seed = int(getattr(Config, "MIX_SEED", base_seed + 991))

    ops = _get_enabled_probe_ops(Config)
    cat_w = getattr(Config, "PROBE_CAT_WEIGHTS", None)
    attempt_mult = int(getattr(Config, "PROBE_ATTEMPT_MULT", 40))
    op_names = [n for n, _ in ops]
    print(
        f"[PROBE] enabled_ops={len(op_names)} "
        f"({', '.join(op_names) if op_names else 'none'}) | "
        f"p_train={p_probe_tr:.4f} p_val={p_probe_va:.4f} attempt_mult={attempt_mult}",
        flush=True,
    )

    shared_epoch = Value("i", 0)

    # ---- TRAIN: build K variants; each variant is (raw + probe_okonly), then 3 views ----
    raws, pns, glosses = [], [], []

    for v in tqdm(range(K), desc="Build train variants (probe→views)"):
        ds_base = train_text_ds

        ds_probe = build_probe_append_text_ds(
            base_text_ds=ds_base,
            pre=pre,
            Config=Config,
            p_probe=p_probe_tr,
            ops=ops,
            seed=int(probe_seed + 1009 * v),
            cat_weights=cat_w,
            attempt_mult=attempt_mult,
            enforce_unique=True,
            debug_label=f"train:v{v}",
        )

        ds_text = concatenate_datasets([ds_base, ds_probe]) if ds_probe is not None else ds_base
        if ds_probe is None:
            print(f"[PROBE][train:v{v}] fallback to base-only dataset (no appended probe rows).", flush=True)
        else:
            print(f"[PROBE][train:v{v}] base={len(ds_base)} + probe={len(ds_probe)} => total={len(ds_text)}", flush=True)

        ds_raw = ds_text.map(
            make_map_raw(Config, tokenizer, pre, tgt_max=int(tgt_max)),
            batched=True,
            batch_size=int(MAP_BS),
            num_proc=int(NPROC),
            remove_columns=[c for c in ds_text.column_names if c not in ()],
        )

        ds_pn = ds_text.map(
            make_map_pn(Config, tokenizer, pre, canon, tgt_max=int(tgt_max)),
            batched=True,
            batch_size=int(MAP_BS),
            num_proc=int(NPROC),
            remove_columns=[c for c in ds_text.column_names if c not in ()],
        )

        ds_gl = ds_text.map(
            make_map_gloss_raw(
                Config, tokenizer, pre, glosser,
                seed_for_variant=int(gloss_seed + 7919 * v),
                tgt_max=int(tgt_max),
            ),
            with_indices=True,
            batched=True,
            batch_size=int(MAP_BS),
            num_proc=int(NPROC),
            remove_columns=[c for c in ds_text.column_names if c not in ()],
        )

        raws.append(ds_raw)
        pns.append(ds_pn)
        glosses.append(ds_gl)

    tokenized_train = EpochVariantMinViewMix3(
        raws, pns, glosses,
        shared_epoch=shared_epoch,
        seed=base_seed,
        p_pn=p_pn,
        p_gloss=p_gl,
        mix_seed=mix_seed,
    )

    # ---- VAL: raw + probe_okonly only ----
    ds_probe_va = build_probe_append_text_ds(
        base_text_ds=val_text_ds,
        pre=pre,
        Config=Config,
        p_probe=p_probe_va,
        ops=ops,
        seed=int(probe_seed + 555),
        cat_weights=cat_w,
        attempt_mult=attempt_mult,
        enforce_unique=True,
        debug_label="val",
    )
    val_text_plus = concatenate_datasets([val_text_ds, ds_probe_va]) if ds_probe_va is not None else val_text_ds
    if ds_probe_va is None:
        print("[PROBE][val] fallback to base-only validation dataset (no appended probe rows).", flush=True)
    else:
        print(
            f"[PROBE][val] base={len(val_text_ds)} + probe={len(ds_probe_va)} => total={len(val_text_plus)}",
            flush=True,
        )

    tokenized_val = val_text_plus.map(
        make_map_raw(Config, tokenizer, pre, tgt_max=int(tgt_max)),
        batched=True,
        batch_size=int(MAP_BS),
        num_proc=int(NPROC),
        remove_columns=[c for c in val_text_plus.column_names if c not in ()],
    )

    return tokenized_train, tokenized_val, shared_epoch

# ================================================================
# Checkpoint averaging
# ================================================================
def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def _list_checkpoints(output_dir):
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=_natural_key)
    return [c for c in ckpts if os.path.isdir(c)]


def _load_state_dict_any(ckpt_dir, map_location="cpu"):
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(st_path):
        from safetensors.torch import load_file

        return load_file(st_path, device=map_location)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location=map_location)
    raise FileNotFoundError(f"No model weights found in {ckpt_dir}")


def _choose_best_k_by_metric(output_dir, k, metric_key="eval_geo_mean"):
    ckpts = _list_checkpoints(output_dir)
    if not ckpts:
        return []
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return ckpts[-k:]
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        scores = {}
        for log in state.get("log_history", []):
            if metric_key in log and "step" in log:
                step = int(log["step"])
                ckpt = os.path.join(output_dir, f"checkpoint-{step}")
                if os.path.isdir(ckpt):
                    scores[ckpt] = float(log[metric_key])
        if not scores:
            return ckpts[-k:]
        chosen = [p for p, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]
        return chosen
    except Exception:
        return ckpts[-k:]


def average_checkpoints_and_save(
    model, output_dir, save_dir, *, k=8, metric_key="eval_geo_mean",
    prefer_best=True, base_ckpt_for_config=None, cleanup_checkpoints=True,
):
    ckpts = _list_checkpoints(output_dir)
    if not ckpts:
        raise ValueError(f"No checkpoints found under {output_dir}")

    k = max(1, int(k))
    if prefer_best:
        chosen = _choose_best_k_by_metric(output_dir, k=k, metric_key=metric_key)
    else:
        chosen = ckpts[-k:]
    if not chosen:
        chosen = ckpts[-k:]

    base_ckpt = (
        base_ckpt_for_config
        if (base_ckpt_for_config and os.path.isdir(base_ckpt_for_config))
        else chosen[0]
    )

    avg_sd = None
    n = 0
    for ckpt in chosen:
        sd = _load_state_dict_any(ckpt, map_location="cpu")
        if avg_sd is None:
            avg_sd = {kk: vv.float().clone() for kk, vv in sd.items() if torch.is_tensor(vv)}
        else:
            for kk, vv in sd.items():
                if kk in avg_sd and torch.is_tensor(vv):
                    avg_sd[kk] += vv.float()
        n += 1

    for kk in avg_sd:
        avg_sd[kk] /= float(n)

    save_dir = str(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    missing, unexpected = model.load_state_dict(avg_sd, strict=False)

    try:
        model.save_pretrained(save_dir, safe_serialization=True)
    except Exception:
        model.save_pretrained(save_dir)

    meta = {
        "averaged_k": n,
        "metric_key": metric_key,
        "prefer_best": prefer_best,
        "chosen_checkpoints": chosen,
        "base_ckpt": base_ckpt,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "cleanup_checkpoints": bool(cleanup_checkpoints),
    }
    with open(os.path.join(save_dir, "ckpt_avg_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if cleanup_checkpoints:
        save_dir_resolved = Path(save_dir).resolve()
        for p in Path(output_dir).glob("checkpoint*"):
            if p.is_dir() and p.resolve() != save_dir_resolved:
                shutil.rmtree(p, ignore_errors=True)

    return save_dir, chosen

# ===== Notebook Cell 6 =====
# TBM (Translation/Template Based Matching) for MBR pool
# =====================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class TBMIndex:
    """
    Char n-gram TF-IDF over normalized SOURCE (transliteration).
    Retrieves TRAIN translations; returns (translation, cosine_sim).
    """
    def __init__(
        self,
        train_src_norm: list[str],
        train_tgt: list[str],
        *,
        ngram=(3, 6),
        max_features=250_000,
        n_neighbors: int = 8,
    ):
        self.train_tgt = list(map(str, train_tgt))
        self.n_neighbors = int(n_neighbors)

        self.vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(ngram),
            min_df=1,
            max_features=int(max_features),
        )
        X = self.vec.fit_transform(list(map(str, train_src_norm)))
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine", algorithm="brute")
        self.nn.fit(X)

    def query(self, src_norm: str, k: int = 3):
        k = max(1, min(int(k), self.n_neighbors))
        q = self.vec.transform([str(src_norm)])
        dists, idxs = self.nn.kneighbors(q, n_neighbors=k, return_distance=True)
        sims = 1.0 - dists[0]
        idxs = idxs[0]
        return [(self.train_tgt[i], float(s)) for i, s in zip(idxs, sims)]

def _tbm_build_from_lists(
    pre: "OptimizedPreprocessor",
    src_raw: list[str],
    tgt_raw: list[str],
    *,
    ngram=(3, 6),
    max_features=250_000,
    n_neighbors: int = 8,
    min_pairs: int = 10,
    dedupe_src: bool = True,
):
    # normalize sources using same preprocessor as inputs
    src_norm = pre.preprocess_batch(list(map(str, src_raw)))

    # drop empties
    pairs = [(str(s).strip(), str(t).strip()) for s, t in zip(src_norm, tgt_raw)]
    pairs = [(s, t) for (s, t) in pairs if s and t]
    if len(pairs) < int(min_pairs):
        return None

    # optional: dedupe by src_norm (keeps first translation)
    if dedupe_src:
        seen = set()
        src2, tgt2 = [], []
        for s, t in pairs:
            if s in seen:
                continue
            seen.add(s)
            src2.append(s)
            tgt2.append(t)
        pairs = list(zip(src2, tgt2))
        if len(pairs) < int(min_pairs):
            return None

    src_norm, tgt_raw = zip(*pairs)
    return TBMIndex(
        list(src_norm),
        list(tgt_raw),
        ngram=ngram,
        max_features=max_features,
        n_neighbors=n_neighbors,
    )


def try_build_tbm_index(
    *,
    pre: "OptimizedPreprocessor",
    train_csv: str,
    input_dir: str,
    ngram=(3, 6),
    max_features=250_000,
    n_neighbors: int = 8,
    verbose: bool = True,
) -> TBMIndex | None:
    """
    Builds TBM on sentence-level pairs if sentence file exists.
    Falls back to full-row train pairs if not.
    """
    if not train_csv or not os.path.exists(train_csv):
        if verbose:
            print("[TBM] train_csv not found -> TBM disabled", flush=True)
        return None

    tr = pd.read_csv(train_csv)
    need = {"oare_id", "transliteration", "translation"}
    if not need.issubset(set(tr.columns)):
        if verbose:
            print("[TBM] train_csv missing required columns -> TBM disabled", flush=True)
        return None

    src_raw = tr["transliteration"].astype(str).tolist()
    tgt_raw = tr["translation"].astype(str).tolist()

    tbm = _tbm_build_from_lists(
        pre,
        src_raw,
        tgt_raw,
        ngram=ngram,
        max_features=max_features,
        n_neighbors=n_neighbors,
        min_pairs=10,
        dedupe_src=True,
    )
    if tbm is None and verbose:
        print("[TBM] too few pairs after filtering -> TBM disabled", flush=True)
    return tbm


def build_tbm_from_pairs(
    pre,
    pairs_df: pd.DataFrame,
    *,
    ngram=(3, 6),
    max_features=250_000,
    n_neighbors: int = 8,
    src_col="transliteration",
    tgt_col="translation",
) -> TBMIndex | None:
    """
    Leak-safe builder when you pass ONLY the train-split pairs_df (sentence pairs recommended).
    """
    if pairs_df is None or len(pairs_df) < 10:
        return None
    if (src_col not in pairs_df.columns) or (tgt_col not in pairs_df.columns):
        return None

    src_raw = pairs_df[src_col].astype(str).tolist()
    tgt_raw = pairs_df[tgt_col].astype(str).tolist()

    return _tbm_build_from_lists(
        pre,
        src_raw,
        tgt_raw,
        ngram=ngram,
        max_features=max_features,
        n_neighbors=n_neighbors,
        min_pairs=10,
        dedupe_src=True,
    )

# ===== Notebook Cell 7 =====
# kNN machinery 
# - Embedding = mean-pooled encoder last_hidden_state (cosine)
# - Index = sklearn NearestNeighbors(metric="cosine", algorithm="brute")
# - Cache to disk (embeds.npy + src/tgt jsonl + meta.json), reload with mmap
# - Optional: rebuild cache if meta mismatches (prefix/max_length/dim) or on best val
# ============================================================

from sklearn.neighbors import NearestNeighbors

# -------------------------
# Encoder embedding: mean-pool + L2 norm
# (optimized: preallocate output array, avoid list+concat)
# -------------------------
@torch.inference_mode()
def encode_src_meanpool(
    model,
    tokenizer,
    texts,
    *,
    prefix: str,
    device: str,
    batch_size: int = 64,
    max_length: int = 512,
    use_bf16: bool = True,
    empty_cache_every: int = 0,   # 0=never (recommended)
):
    """
    Mean-pooled encoder embeddings for a list of (already-preprocessed) source strings.
    Returns float16 numpy array of shape (N, H), L2-normalized (cosine-ready).
    """

    # unwrap common wrappers (DDP / compile / etc.)
    m = getattr(model, "_orig_mod", model)
    m = getattr(m, "module", m)

    # get encoder
    if hasattr(m, "get_encoder") and callable(getattr(m, "get_encoder")):
        encoder = m.get_encoder()
    else:
        encoder = getattr(m, "encoder", None)
        if encoder is None:
            raise AttributeError("Model has no get_encoder() or .encoder")

    dev = torch.device(device)

    # preserve state
    was_train = bool(getattr(encoder, "training", False))
    encoder.eval()

    # autocast bf16 only if supported
    is_bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    amp = (dev.type == "cuda") and bool(use_bf16) and torch.cuda.is_available() and is_bf16_supported

    N = len(texts)
    out_np = None
    out_i = 0

    bs = int(max(1, batch_size))
    ml = int(max(8, max_length))
    pref = str(prefix)

    for bi, i0 in enumerate(range(0, N, bs)):
        batch = [pref + str(t) for t in texts[i0 : i0 + bs]]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ml,
        )
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = encoder(**enc, return_dict=True)
                h = out.last_hidden_state
        else:
            out = encoder(**enc, return_dict=True)
            h = out.last_hidden_state

        # mean-pool (mask padding)
        msk = enc["attention_mask"].unsqueeze(-1).to(dtype=h.dtype)  # (B,T,1)
        denom = msk.sum(dim=1).clamp_min(1.0)                        # (B,1)
        pooled = (h * msk).sum(dim=1) / denom                        # (B,H)

        # L2-normalize in fp32, store as fp16
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1).to(torch.float16)
        arr = pooled.cpu().numpy()  # (B,H) float16

        if out_np is None:
            H = int(arr.shape[1])
            out_np = np.empty((N, H), dtype=np.float16)

        bsz = int(arr.shape[0])
        out_np[out_i : out_i + bsz] = arr
        out_i += bsz

        # free GPU refs ASAP
        del enc, out, h, msk, denom, pooled, arr

        # avoid empty_cache spam (only every N batches, not at bi=0)
        if empty_cache_every and dev.type == "cuda" and ((bi + 1) % int(empty_cache_every) == 0):
            torch.cuda.empty_cache()

    if was_train:
        encoder.train()

    return out_np if out_np is not None else np.zeros((0, 0), dtype=np.float16)


# -------------------------
# Cache IO
# -------------------------
def save_knn_cache(
    cache_dir: str,
    embeds_f16: np.ndarray,
    src_texts,
    tgt_texts,
    meta: Optional[dict] = None,
):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "embeds.npy"), np.asarray(embeds_f16, dtype=np.float16))

    with open(os.path.join(cache_dir, "src.jsonl"), "w", encoding="utf-8") as f:
        for s in src_texts:
            f.write(json.dumps(str(s), ensure_ascii=False) + "\n")

    with open(os.path.join(cache_dir, "tgt.jsonl"), "w", encoding="utf-8") as f:
        for t in tgt_texts:
            f.write(json.dumps(str(t), ensure_ascii=False) + "\n")

    if meta is None:
        meta = {}
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_knn_cache(cache_dir: str):
    embeds = np.load(os.path.join(cache_dir, "embeds.npy"), mmap_mode="r")
    srcs, tgts = [], []

    with open(os.path.join(cache_dir, "src.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            srcs.append(json.loads(line))

    with open(os.path.join(cache_dir, "tgt.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            tgts.append(json.loads(line))

    meta_path = os.path.join(cache_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return embeds, srcs, tgts, meta


def _knn_meta_mismatch(meta: dict, *, prefix_for_encode: str, max_length: int) -> bool:
    if not isinstance(meta, dict):
        return True
    if str(meta.get("prefix_for_encode", "")) != str(prefix_for_encode):
        return True
    if int(meta.get("max_length", -1)) != int(max_length):
        return True
    # dim is checked after load (if present)
    return False


# -------------------------
# sklearn kNN index + search
# -------------------------
def build_sklearn_knn_index(embeds_f16: np.ndarray) -> NearestNeighbors:
    """
    Cosine kNN:
      sklearn returns cosine *distance*; we convert to similarity as 1 - dist.
    """
    x = np.asarray(embeds_f16, dtype=np.float32)  # small bank => acceptable
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(x)
    return nn


def knn_search_sklearn(nn: NearestNeighbors, query_emb_f16: np.ndarray, k: int = 8):
    q = np.asarray(query_emb_f16, dtype=np.float32)
    dist, idx = nn.kneighbors(q, n_neighbors=int(k), return_distance=True)
    sims = 1.0 - dist
    return idx.astype(np.int64), sims.astype(np.float32)


# -------------------------
# Build or load memory (with optional rebuild-on-mismatch)
# -------------------------
def build_or_load_knn_memory_sklearn(
    *,
    cache_dir: str,
    model,
    tokenizer,
    bank_src_texts,          # list[str] (already preprocessed / normalized)
    bank_tgt_texts,          # list[str] (normalized targets)
    prefix_for_encode: str,  # must match how you'll encode queries later ("" or Config.PREFIX)
    device: str,
    max_length: int = 512,
    batch_size: int = 64,
    use_bf16: bool = True,
    rebuild: bool = False,
    rebuild_on_mismatch: bool = True,
):
    """
    Returns dict:
      {
        "embeds": mmap np array,
        "srcs": list[str],
        "tgts": list[str],
        "nn": sklearn NearestNeighbors,
        "cache_dir": str,
        "meta": dict,
      }
    """
    cache_dir = str(cache_dir)
    embeds_path = os.path.join(cache_dir, "embeds.npy")

    need = bool(rebuild) or (not os.path.exists(embeds_path))

    if (not need) and bool(rebuild_on_mismatch):
        # quick meta check before trusting cache
        try:
            _embeds_tmp, _srcs_tmp, _tgts_tmp, _meta_tmp = load_knn_cache(cache_dir)
            if _knn_meta_mismatch(_meta_tmp, prefix_for_encode=prefix_for_encode, max_length=max_length):
                need = True
        except Exception:
            need = True

    if need:
        print(f"[kNN/sklearn] building cache -> {cache_dir}", flush=True)
        embeds = encode_src_meanpool(
            model, tokenizer, bank_src_texts,
            prefix=str(prefix_for_encode),
            device=str(device),
            batch_size=int(batch_size),
            max_length=int(max_length),
            use_bf16=bool(use_bf16),
            empty_cache_every=0,  # keep fast; enable only for debugging fragmentation
        )
        meta = {
            "prefix_for_encode": str(prefix_for_encode),
            "max_length": int(max_length),
            "n": int(len(bank_src_texts)),
            "dim": int(embeds.shape[1]) if embeds.ndim == 2 else 0,
            "dtype": "float16",
            "metric": "cosine",
        }
        save_knn_cache(cache_dir, embeds, bank_src_texts, bank_tgt_texts, meta=meta)
        embeds, srcs, tgts, meta = load_knn_cache(cache_dir)
    else:
        print(f"[kNN/sklearn] loading cache <- {cache_dir}", flush=True)
        embeds, srcs, tgts, meta = load_knn_cache(cache_dir)

    nn = build_sklearn_knn_index(embeds)

    return {
        "embeds": embeds,
        "srcs": srcs,
        "tgts": tgts,
        "nn": nn,
        "cache_dir": cache_dir,
        "meta": meta,
    }


# -------------------------
# Optional: attach helper (works with your bigger Trainer that uses knn_nn/knn_tgts)
# -------------------------
def attach_knn_mem_to_trainer(trainer, knn_mem: dict):
    """
    Safe attach for trainers that expect:
      trainer.knn_nn / trainer.knn_tgts (or trainer.knn_mem)
    """
    if trainer is None or knn_mem is None:
        return
    try:
        trainer.knn_mem = knn_mem
    except Exception:
        pass
    try:
        if hasattr(trainer, "knn_nn"):
            trainer.knn_nn = knn_mem.get("nn", None)
        if hasattr(trainer, "knn_tgts"):
            trainer.knn_tgts = knn_mem.get("tgts", None)
    except Exception:
        pass


# -------------------------
# Callback: rebuild bank only when monitored eval metric improves
# -------------------------
class RebuildKNNOnBestEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        trainer_ref,                 # bind the real trainer here (most reliable)
        bank_src_texts,
        bank_tgt_texts,
        cache_dir: str,
        prefix_for_encode: str,
        monitor_key: str = "eval_geo_mean",
        mode: str = "max",           # "max" for geo_mean, "min" for loss
        eps: float = 1e-9,
        max_length: int = 512,
        batch_size: int = 64,
        use_bf16: bool = True,
        rebuild_on_mismatch: bool = True,
        also_attach_to_trainer: bool = True,
    ):
        self.trainer_ref = trainer_ref
        self.bank_src_texts = list(bank_src_texts)
        self.bank_tgt_texts = list(bank_tgt_texts)
        self.cache_dir = str(cache_dir)
        self.prefix_for_encode = str(prefix_for_encode)

        self.monitor_key = str(monitor_key)
        self.mode = str(mode).lower().strip()
        self.eps = float(eps)

        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.use_bf16 = bool(use_bf16)
        self.rebuild_on_mismatch = bool(rebuild_on_mismatch)
        self.also_attach_to_trainer = bool(also_attach_to_trainer)

        self.best = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        # HF provides state.is_world_process_zero (bool)
        if not bool(getattr(state, "is_world_process_zero", True)):
            return

        cur = metrics.get(self.monitor_key, None)
        if cur is None:
            return

        cur = float(cur)

        improved = False
        if self.best is None:
            improved = True
        else:
            if self.mode == "min":
                improved = cur < (float(self.best) - self.eps)
            else:
                improved = cur > (float(self.best) + self.eps)

        if not improved:
            return

        self.best = cur

        # rebuild using CURRENT weights (best-so-far)
        tr = self.trainer_ref
        model = getattr(tr, "model", None)
        tok = getattr(tr, "processing_class", None) or getattr(tr, "tokenizer", None)
        if model is None or tok is None:
            return

        dev = _get_model_primary_device_str(model)
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"[kNN] {self.monitor_key} improved -> rebuild bank @ {self.cache_dir}", flush=True)

        knn_mem = build_or_load_knn_memory_sklearn(
            cache_dir=self.cache_dir,
            model=model,
            tokenizer=tok,
            bank_src_texts=self.bank_src_texts,
            bank_tgt_texts=self.bank_tgt_texts,
            prefix_for_encode=self.prefix_for_encode,
            device=dev,
            max_length=self.max_length,
            batch_size=self.batch_size,
            use_bf16=self.use_bf16,
            rebuild=True,
            rebuild_on_mismatch=self.rebuild_on_mismatch,
        )

        if self.also_attach_to_trainer:
            attach_knn_mem_to_trainer(tr, knn_mem)

        # optional: barrier so other ranks don't read half-written cache
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

# ===== Notebook Cell 8 =====
# MBR EVAL helpers
# ==================================

def _stable_int_id(s: str) -> int:
    return int(zlib.adler32(str(s).encode("utf-8")) & 0x7FFFFFFF)

def _norm_ws(s: str) -> str:
    return " ".join(str(s).strip().split())

def _dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        x = str(x)
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _val_unique_examples(val_source, *, prefer_original: bool = True):
    """
    Returns (ex_ids, srcs, refs_raw, oare_ids).
    Dedup by ex_id preferring src_view=='original' (stable order).
    Supports HF Dataset-like (has .column_names) and pandas DataFrame.
    """
    # --- HF Dataset-like
    if hasattr(val_source, "column_names"):
        cols = set(val_source.column_names)
        N = len(val_source)
        oare_all = list(val_source["oare_id"]) if ("oare_id" in cols) else [None] * N

        if "ex_id" not in cols:
            ex_ids = [f"row{i}" for i in range(N)]
            return ex_ids, list(val_source["transliteration"]), list(val_source["translation"]), list(oare_all)

        ex_all   = [str(x) for x in val_source["ex_id"]]
        src_all  = list(val_source["transliteration"])
        ref_all  = list(val_source["translation"])
        view_all = list(val_source["src_view"]) if ("src_view" in cols) else ["original"] * N

        best = {}
        for i, (eid, v) in enumerate(zip(ex_all, view_all)):
            if eid not in best:
                best[eid] = i
            elif prefer_original and (str(v) == "original") and (str(view_all[best[eid]]) != "original"):
                best[eid] = i

        seen = set()
        picked = []
        for eid in ex_all:  # preserve first occurrence order of ex_ids
            if eid in seen:
                continue
            seen.add(eid)
            picked.append(best[eid])

        return (
            [ex_all[i] for i in picked],
            [src_all[i] for i in picked],
            [ref_all[i] for i in picked],
            [oare_all[i] for i in picked],
        )

    # --- pandas
    if isinstance(val_source, pd.DataFrame):
        df = val_source
        oare_all = df["oare_id"].tolist() if ("oare_id" in df.columns) else [None] * len(df)
        if "ex_id" not in df.columns:
            ex_ids = [f"row{i}" for i in range(len(df))]
            return ex_ids, df["transliteration"].astype(str).tolist(), df["translation"].astype(str).tolist(), list(oare_all)

        if prefer_original and ("src_view" in df.columns):
            # stable: keep first ex_id order from original df, prefer src_view=="original"
            order = pd.Series(df["ex_id"].astype(str).tolist()).drop_duplicates().tolist()
            df2 = df.copy()
            df2["__is_orig__"] = (df2["src_view"].astype(str) == "original").astype(int)
            df2 = (
                df2.sort_values(["ex_id", "__is_orig__"], ascending=[True, False])
                   .drop_duplicates("ex_id", keep="first")
            )
            df2["__ord__"] = df2["ex_id"].astype(str).map({k: i for i, k in enumerate(order)})
            df2 = df2.sort_values("__ord__").drop(columns=["__is_orig__", "__ord__"])
        else:
            df2 = df.drop_duplicates("ex_id", keep="first")

        return (
            df2["ex_id"].astype(str).tolist(),
            df2["transliteration"].astype(str).tolist(),
            df2["translation"].astype(str).tolist(),
            df2["oare_id"].tolist() if ("oare_id" in df2.columns) else [None] * len(df2),
        )

    raise TypeError("val_source must be HF Dataset or pandas DataFrame.")


def _official_geo_mean(refs: list[str], preds: list[str]) -> tuple[float, float, float]:
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    geo = float(math.sqrt(float(bleu) * float(chrfpp)))
    return geo, float(bleu), float(chrfpp)


def _geo_sim_sentence(a: str, b: str) -> float:
    bleu = sacrebleu.sentence_bleu(a, [b]).score
    chrf = sacrebleu.sentence_chrf(a, [b], word_order=2).score
    return float(math.sqrt(max(0.0, bleu) * max(0.0, chrf)))

# ===== Notebook Cell 9 =====
class MBRGlossSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        val_text_ds=None,
        pre=None,
        prefix="",
        post=None,            # preds post (should match inference)
        post_ref=None,        # DEBUG only (NOT used for scoring)

        glosser=None,
        gloss_variants=1,
        gloss_seed=12345,
        gloss_max_items=6,
        gloss_max_append_chars=240,

        mbr_batch_size_inputs=16,
        src_max_length=512,
        max_new_tokens=512,
        num_beams=8,
        num_beam_cands=1,
        num_sample_cands=4,
        length_penalty=1.3,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        mbr_pool_cap=10,
        show_progress=True,

        # ---- PN view (optional) ----
        canon=None,                  # SourceCanonicalizer-like, must implement canonicalize_source(text, mode=...)
        pn_enable=False,
        pn_mode="pn_norm",

        # ---- TBM (optional) ----
        tbm_index=None,              # TBMIndex, if already built
        tbm_pairs=None,              # pd.DataFrame of train-split pairs to build TBM (leak-safe)
        tbm_ngram=(3, 6),
        tbm_max_features=250_000,
        tbm_topk=3,
        tbm_min_sim=0.92,
        tbm_hard_sim=0.97,
        tbm_enable=True,

        # ---- kNN (optional; sklearn NN over encoder meanpool) ----
        knn_mem=None,                # dict from build_or_load_knn_memory_sklearn(...), optional
        knn_nn=None,                 # sklearn NearestNeighbors, optional
        knn_tgts=None,               # list[str] same order as knn_nn bank, optional
        knn_enable: bool = False,
        knn_topk: int = 8,
        knn_hint_k: int = 2,
        knn_hint_max_chars: int = 240,
        knn_ret_k: int = 1,          # inject top-R retrieved targets directly into pool
        knn_prefix_for_encode: str | None = None,  # must match how bank was embedded
        knn_query_bs: int | None = None,
        knn_min_sim: float = 0.90,
        knn_hard_sim: float = 0.94,

        # ---- MSA polish (optional) ----
        msa_enable: bool = False,          # default disabled
        msa_gap_thr: float | None = None,  # if set, only run MSA when MBR gap <= thr
        msa_min_pool: int = 3,             # minimum pool size to attempt MSA

        # ---- tag-aware MBR biases (optional) ----
        mbr_tag_prior=0.35,
        mbr_beam_penalty=0.00,
        mbr_samp_bonus=0.00,
        mbr_gloss_penalty=0.2,
        mbr_pn_penalty=0.25,
        mbr_knn_penalty=0.2,
        mbr_tbm_bonus=0.5,
        mbr_raw_bonus=0.00,

        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if val_text_ds is None:
            raise ValueError("MBRGlossSeq2SeqTrainer requires val_text_ds.")
        if pre is None:
            raise ValueError("MBRGlossSeq2SeqTrainer requires pre.")

        self.val_text_ds = val_text_ds
        self.pre = pre
        self.prefix = str(prefix)

        self.post = post
        self.post_ref = post_ref  # debug only

        # PN
        self.canon = canon
        self.pn_enable = bool(pn_enable)
        self.pn_mode = str(pn_mode)

        # Gloss
        self.glosser = glosser
        self.gloss_variants = int(gloss_variants)
        self.gloss_seed = int(gloss_seed)
        self.gloss_max_items = int(gloss_max_items)
        self.gloss_max_append_chars = int(gloss_max_append_chars)

        # Decode
        self.mbr_batch_size_inputs = int(mbr_batch_size_inputs)
        self.src_max_length = int(src_max_length)
        self.max_new_tokens = int(max_new_tokens)

        self.num_beams = int(num_beams)
        self.num_beam_cands = int(num_beam_cands)
        self.num_sample_cands = int(num_sample_cands)
        self.length_penalty = float(length_penalty)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)

        self.mbr_pool_cap = None if mbr_pool_cap is None else int(mbr_pool_cap)
        self.show_progress = bool(show_progress)

        # ---- TBM config ----
        self.tbm_enable = bool(tbm_enable)
        self.tbm_topk = int(tbm_topk)
        self.tbm_min_sim = float(tbm_min_sim)
        self.tbm_hard_sim = float(tbm_hard_sim)

        self.tbm_index = None
        if self.tbm_enable:
            if tbm_index is not None:
                self.tbm_index = tbm_index
            elif tbm_pairs is not None:
                try:
                    self.tbm_index = build_tbm_from_pairs(
                        self.pre, tbm_pairs,
                        ngram=tuple(tbm_ngram),
                        max_features=int(tbm_max_features),
                    )
                except Exception as e:
                    self.tbm_index = None
                    if self.show_progress:
                        print("[TBM] build failed -> TBM disabled:", repr(e), flush=True)

        # ---- kNN config ----
        self.knn_enable = bool(knn_enable)
        self.knn_topk = int(knn_topk)
        self.knn_hint_k = int(knn_hint_k)
        self.knn_hint_max_chars = int(knn_hint_max_chars)
        self.knn_ret_k = int(knn_ret_k)
        self.knn_min_sim = float(knn_min_sim)
        self.knn_hard_sim = float(knn_hard_sim)

        if knn_prefix_for_encode is None:
            knn_prefix_for_encode = self.prefix
        self.knn_prefix_for_encode = str(knn_prefix_for_encode)

        if knn_query_bs is None:
            knn_query_bs = 128  # train default; override in Config if needed
        self.knn_query_bs = int(knn_query_bs)

        self.knn_nn = None
        self.knn_tgts = None
        if self.knn_enable:
            if knn_mem is not None:
                self.knn_nn = knn_mem.get("nn", None)
                self.knn_tgts = knn_mem.get("tgts", None)
            if self.knn_nn is None and knn_nn is not None:
                self.knn_nn = knn_nn
            if self.knn_tgts is None and knn_tgts is not None:
                self.knn_tgts = knn_tgts

            if self.knn_nn is None or self.knn_tgts is None:
                self.knn_enable = False
                if self.show_progress:
                    print("[kNN] disabled: missing knn_nn/knn_tgts (or knn_mem).", flush=True)

        # ---- MSA config ----
        self.msa_enable = bool(msa_enable)
        self.msa_gap_thr = None if msa_gap_thr is None else float(msa_gap_thr)
        self.msa_min_pool = int(msa_min_pool)

        # ---- tag-aware MBR biases ----
        self.mbr_tag_prior = float(mbr_tag_prior)
        self.mbr_beam_penalty = float(mbr_beam_penalty)
        self.mbr_samp_bonus = float(mbr_samp_bonus)
        self.mbr_gloss_penalty = float(mbr_gloss_penalty)
        self.mbr_pn_penalty = float(mbr_pn_penalty)
        self.mbr_knn_penalty = float(mbr_knn_penalty)
        self.mbr_tbm_bonus = float(mbr_tbm_bonus)
        self.mbr_raw_bonus = float(mbr_raw_bonus)

        # prefer caching in generate
        try:
            self.model.config.use_cache = True
        except Exception:
            pass
        try:
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.use_cache = True
        except Exception:
            pass

        self.has_gloss = (self.glosser is not None) and (self.gloss_variants > 0)
        self.has_pn = self.pn_enable and (self.canon is not None)
        self.has_tbm = bool(tbm_enable) and (self.tbm_index is not None)
        self.has_knn = bool(self.knn_enable and (self.knn_nn is not None) and (self.knn_tgts is not None))

    # -------------------------
    # small helpers
    # -------------------------
    def _is_dist(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _broadcast_metrics(self, metrics: dict):
        if not self._is_dist():
            return metrics
        obj = [metrics if self.is_world_process_zero() else None]
        torch.distributed.broadcast_object_list(obj, src=0)
        return obj[0]

    def _bf16_eval_context(self):
        if torch.cuda.is_available() and bool(getattr(self.args, "bf16", False)):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _tqdm(self, total: int, desc: str):
        if (not self.show_progress) or (tqdm is None):
            return None
        return tqdm(total=total, desc=desc, leave=False)

    # -------------------------
    # LEAN generator helpers (NO globals)
    # -------------------------
    def _encode_for_generate(self, tok, batch_in, *, device, src_max_length: int):
        enc = tok(
            batch_in,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(src_max_length),
        )
        return {k: v.to(device, non_blocking=True) for k, v in enc.items()}

    @torch.inference_mode()
    def _generate_multi_decode(self, model, tok, batch_in, *, device):
        """
        Uses self.* decode params. Returns list[list[str]]: beams first then samples.
        """
        enc = self._encode_for_generate(tok, batch_in, device=device, src_max_length=self.src_max_length)
        B = len(batch_in)
        outs = [[] for _ in range(B)]

        Rb = int(max(0, self.num_beam_cands))
        if Rb > 0:
            nb = int(max(1, int(self.num_beams), Rb))
            seq = model.generate(
                **enc,
                do_sample=False,
                num_beams=nb,
                num_return_sequences=Rb,
                max_new_tokens=int(self.max_new_tokens),
                length_penalty=float(self.length_penalty),
                repetition_penalty=float(self.repetition_penalty),
                no_repeat_ngram_size=int(self.no_repeat_ngram_size),
                use_cache=True,
                early_stopping=True,
            )
            txt = tok.batch_decode(seq, skip_special_tokens=True)
            for i in range(B):
                outs[i].extend(txt[i * Rb : (i + 1) * Rb])
            del seq, txt

        Rs = int(max(0, self.num_sample_cands))
        if Rs > 0:
            seq = model.generate(
                **enc,
                do_sample=True,
                num_beams=1,
                num_return_sequences=Rs,
                max_new_tokens=int(self.max_new_tokens),
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                repetition_penalty=float(self.repetition_penalty),
                no_repeat_ngram_size=int(self.no_repeat_ngram_size),
                use_cache=True,
            )
            txt = tok.batch_decode(seq, skip_special_tokens=True)
            for i in range(B):
                outs[i].extend(txt[i * Rs : (i + 1) * Rs])
            del seq, txt

        return outs

    @torch.inference_mode()
    def _generate_multi_decode_tagged(self, model, tok, batch_in, batch_view, *, device):
        """
        Returns (cand_txt_lists, cand_tag_lists) with tags beam|<view>, samp|<view>.
        """
        views = list(batch_view)
        cands = self._generate_multi_decode(model, tok, batch_in, device=device)
        Rb = int(max(0, self.num_beam_cands))

        out_txt, out_tag = [], []
        for vw, xs in zip(views, cands):
            xs = list(xs or [])
            tags = [("beam" if k < Rb else "samp") + "|" + str(vw) for k in range(len(xs))]
            out_txt.append(xs)
            out_tag.append(tags)
        return out_txt, out_tag

    # ---- tag parsing (local) ----
    def _origin_from_tag_local(self, tag: str) -> str:
        t = "" if tag is None else str(tag)
        o = t.split("|", 1)[0] if "|" in t else t
        if o in ("beam", "samp", "tbm", "knn"):
            return o
        return o or "other"

    def _view_from_tag_local(self, tag: str) -> str:
        t = "" if tag is None else str(tag)
        if "|" in t:
            o, v = t.split("|", 1)
        else:
            o, v = t, ""
        if o == "tbm":
            return "raw"
        if o == "knn" and (not v):
            return "raw"  # retrieval inject
        v = v or "raw"
        if v.startswith("gloss"):
            return "gloss"
        if v.startswith("knn"):
            return "knn"
        if v == "pn":
            return "pn"
        if v == "raw":
            return "raw"
        return v or "other"

    def _tags_for_pick(self, ts: list[str]) -> list[str]:
        out = []
        for t in ts:
            t = "" if t is None else str(t)
            if t.startswith("tbm|"):
                out.append("tbm|raw")
            elif t.startswith("knn|"):
                out.append("knn|raw")
            else:
                o = self._origin_from_tag_local(t)
                v = self._view_from_tag_local(t)
                out.append(f"{o}|{v}")
        return out

    def _pick_mbr(self, xs: list[str], ts_pick: list[str]):
        # keep global tagged picker optional, but no globals lookup required for generation anymore
        fn = globals().get("_mbr_pick_geo_tagged", None)
        if callable(fn):
            try:
                return fn(
                    xs, ts_pick,
                    tag_prior=self.mbr_tag_prior,
                    beam_penalty=self.mbr_beam_penalty,
                    samp_bonus=self.mbr_samp_bonus,
                    gloss_penalty=self.mbr_gloss_penalty,
                    pn_penalty=self.mbr_pn_penalty,
                    knn_penalty=self.mbr_knn_penalty,
                    tbm_bonus=self.mbr_tbm_bonus,
                    raw_bonus=self.mbr_raw_bonus,
                )
            except TypeError:
                return fn(
                    xs, ts_pick,
                    tag_prior=self.mbr_tag_prior,
                    beam_penalty=self.mbr_beam_penalty,
                    samp_bonus=self.mbr_samp_bonus,
                    gloss_penalty=self.mbr_gloss_penalty,
                    pn_penalty=self.mbr_pn_penalty,
                    tbm_bonus=self.mbr_tbm_bonus,
                    raw_bonus=self.mbr_raw_bonus,
                )

        n = len(xs)
        if n <= 1:
            return 0, 0.0, 0.0
        sums = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            ai = xs[i]
            for j in range(i + 1, n):
                s = _geo_sim_sentence(ai, xs[j])
                sums[i] += s
                sums[j] += s
        avg = sums / float(n - 1)
        jbest = int(np.argmax(avg))
        best = float(avg[jbest])
        tmp = avg.copy()
        tmp[jbest] = -1e9
        gap = float(best - float(np.max(tmp)))
        return jbest, gap, gap

    # -------------------------
    # kNN helpers (unchanged)
    # -------------------------
    def _knn_compute(self, src_clean: list[str]):
        if (not self.has_knn) or (self.knn_nn is None) or (self.knn_tgts is None):
            return None, None, None
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tok is None:
            return None, None, None
        device_str = _get_model_primary_device_str(self.model)
        with torch.inference_mode(), self._bf16_eval_context():
            q_emb = encode_src_meanpool(
                self.model, tok, src_clean,
                prefix=self.knn_prefix_for_encode,
                device=device_str,
                batch_size=int(self.knn_query_bs),
                max_length=int(self.src_max_length),
                use_bf16=bool(getattr(self.args, "bf16", False)),
            )
        knn_ids, knn_sims = knn_search_sklearn(self.knn_nn, q_emb, k=int(self.knn_topk))

        hints = [""] * len(src_clean)
        cap = int(self.knn_hint_max_chars)
        min_sim = float(self.knn_min_sim)
        hard_sim = float(self.knn_hard_sim)
        K = int(self.knn_hint_k)

        for i in range(len(src_clean)):
            if float(knn_sims[i, 0]) < min_sim:
                continue
            if float(knn_sims[i, 0]) >= hard_sim:
                j0 = int(knn_ids[i, 0])
                if 0 <= j0 < len(self.knn_tgts):
                    tt = str(self.knn_tgts[j0]).strip()
                    if tt:
                        hints[i] = tt[:cap]
                continue
            ts = []
            for j, sim in zip(knn_ids[i, :K], knn_sims[i, :K]):
                if float(sim) < min_sim:
                    continue
                jj = int(j)
                if 0 <= jj < len(self.knn_tgts):
                    tt = str(self.knn_tgts[jj]).strip()
                    if tt and tt not in ts:
                        ts.append(tt)
            if ts:
                hints[i] = (" || ".join(ts))[:cap]

        return knn_ids, knn_sims, hints

    def _knn_inject_retrieval(self, pools_txt, pools_tag, knn_ids, knn_sims):
        if (not self.has_knn) or knn_ids is None or knn_sims is None:
            return 0
        if int(self.knn_ret_k) <= 0:
            return 0
        hit = 0
        R = int(self.knn_ret_k)
        min_sim = float(self.knn_min_sim)
        for i in range(len(pools_txt)):
            if float(knn_sims[i, 0]) < min_sim:
                continue
            added = 0
            for j, sim in zip(knn_ids[i, :R], knn_sims[i, :R]):
                if float(sim) < min_sim:
                    continue
                jj = int(j)
                if 0 <= jj < len(self.knn_tgts):
                    t = str(self.knn_tgts[jj]).strip()
                    if t:
                        pools_txt[i].insert(0, t)
                        pools_tag[i].insert(0, "knn|raw")
                        added += 1
            if added:
                hit += 1
        return hit

    # -------------------------
    # public: eval hook (unchanged)
    # -------------------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        t0 = time.time()
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("evaluate() needs an eval_dataset (even a placeholder) to produce eval_loss.")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        metrics = dict(output.metrics)
        if self.is_world_process_zero():
            mbr_metrics = self._evaluate_mbr(metric_key_prefix=metric_key_prefix)
        else:
            mbr_metrics = None
        mbr_metrics = self._broadcast_metrics(mbr_metrics)
        if mbr_metrics is not None:
            metrics.update(mbr_metrics)
        metrics[f"{metric_key_prefix}_runtime"] = float(time.time() - t0)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    # -------------------------
    # shared core: build pools + pick MBR
    # -------------------------
    @torch.inference_mode()
    def _mbr_from_sources(
        self,
        src_texts: list[str],
        *,
        ex_ids: list[str] | None = None,
        add_tbm: bool = True,
        use_knn: bool = True,
        return_pools: bool = False,
        return_tags: bool = False,
    ):
        model = self.model
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tok is None:
            raise ValueError("Tokenizer/processor not found (trainer.processing_class or trainer.tokenizer).")
        device = _get_model_primary_device(model)

        src_clean = self.pre.preprocess_batch(list(map(str, src_texts)))

        knn_ids = knn_sims = knn_hints = None
        knn_on = bool(use_knn) and bool(self.has_knn)
        if knn_on:
            try:
                knn_ids, knn_sims, knn_hints = self._knn_compute(list(src_clean))
            except Exception as e:
                knn_ids = knn_sims = knn_hints = None
                knn_on = False
                if self.show_progress and self.is_world_process_zero():
                    print("[kNN] compute failed -> disabled for this call:", repr(e), flush=True)

        flat_inputs, flat_exi, flat_view = [], [], []
        for ex_i, base in enumerate(src_clean):
            base = str(base)
            seen_inp = set()

            def _add(txt: str, view: str):
                inp = self.prefix + str(txt)
                if inp in seen_inp:
                    return
                seen_inp.add(inp)
                flat_inputs.append(inp)
                flat_exi.append(ex_i)
                flat_view.append(view)

            _add(base, "raw")

            if self.has_pn:
                try:
                    pn = self.canon.canonicalize_source(base, mode=self.pn_mode)
                except Exception:
                    pn = base
                pn = "" if pn is None else str(pn)
                if pn and pn != base:
                    _add(pn, "pn")

            if self.has_gloss:
                if ex_ids is not None:
                    ex_int = _stable_int_id(str(ex_ids[ex_i]))
                else:
                    ex_int = _stable_int_id(f"infer::{ex_i}::{base[:48]}")
                for v in range(self.gloss_variants):
                    vseed = int(self.gloss_seed) + 1009 * int(v)
                    s_gl = self.glosser.append_gloss(
                        base,
                        max_items=self.gloss_max_items,
                        max_append_chars=self.gloss_max_append_chars,
                        seed=vseed,
                        epoch=0,
                        example_id=int(ex_int),
                        keep_order=True,
                    )
                    s_gl = "" if s_gl is None else str(s_gl)
                    if s_gl and s_gl != base:
                        _add(s_gl, "gloss")

            if knn_on and (knn_hints is not None):
                hint = str(knn_hints[ex_i] or "").strip()
                if hint:
                    s_knn = f"{base} <extra_id_1> TM: {hint}"
                    _add(s_knn, "knn0")

        lens = np.fromiter((len(x.split()) for x in flat_inputs), dtype=np.int32, count=len(flat_inputs))
        order = np.argsort(lens, kind="mergesort")

        pools_txt = [[] for _ in range(len(src_texts))]
        pools_tag = [[] for _ in range(len(src_texts))]

        pbar = self._tqdm(len(order), "MBR/generate")
        with self._bf16_eval_context():
            for a in range(0, len(order), self.mbr_batch_size_inputs):
                idx = order[a:a + self.mbr_batch_size_inputs]
                batch_in = [flat_inputs[i] for i in idx]
                batch_ex = [flat_exi[i] for i in idx]
                batch_vw = [flat_view[i] for i in idx]

                cand_txt_lists, cand_tag_lists = self._generate_multi_decode_tagged(
                    model, tok, batch_in, batch_vw,
                    device=device,
                )

                for ex_i2, vw, cands, tags in zip(batch_ex, batch_vw, cand_txt_lists, cand_tag_lists):
                    ex_i2 = int(ex_i2)
                    cands = list(cands or [])
                    tags = list(tags or [])
                    if len(tags) != len(cands):
                        tags = (tags + [f"other|{vw}"] * len(cands))[: len(cands)]
                    pools_txt[ex_i2].extend(cands)
                    pools_tag[ex_i2].extend(tags)

                if pbar is not None:
                    pbar.update(len(idx))
        if pbar is not None:
            pbar.close()

        pool_raw_mean = float(np.mean([len(p) for p in pools_txt])) if pools_txt else 0.0

        tbm_hit = 0
        if add_tbm and self.has_tbm:
            for ex_i, base_src in enumerate(src_clean):
                try:
                    res = self.tbm_index.query(str(base_src), k=self.tbm_topk)
                except Exception:
                    res = None
                if not res:
                    continue
                if float(res[0][1]) >= float(self.tbm_hard_sim):
                    prepend = [res[0][0]]
                else:
                    prepend = [t for (t, sim) in res if float(sim) >= float(self.tbm_min_sim)]
                if prepend:
                    pools_txt[ex_i] = prepend + pools_txt[ex_i]
                    pools_tag[ex_i] = (["tbm|raw"] * len(prepend)) + pools_tag[ex_i]
                    tbm_hit += 1

        knn_hit = 0
        if knn_on and (knn_ids is not None) and (knn_sims is not None):
            try:
                knn_hit = self._knn_inject_retrieval(pools_txt, pools_tag, knn_ids, knn_sims)
            except Exception:
                knn_hit = 0

        # NOTE: keep your existing tagged-dedupe helper if present; otherwise local fallback.
        dedup_tagged = globals().get("_dedup_keep_order_tagged", None)

        def _dedup_keep_order_tagged_local(xs, ts):
            if callable(dedup_tagged):
                return dedup_tagged(xs, ts)
            seen = set()
            xo, to = [], []
            for x, t in zip(xs, ts):
                if x in seen:
                    continue
                seen.add(x)
                xo.append(x)
                to.append(t)
            return xo, to

        pools_txt2, pools_tag2 = [], []
        for xs, ts in zip(pools_txt, pools_tag):
            xs2, ts2 = _dedup_keep_order_tagged_local(xs, ts)
            if self.mbr_pool_cap is not None:
                xs2 = xs2[: self.mbr_pool_cap]
                ts2 = ts2[: self.mbr_pool_cap]
            pools_txt2.append(xs2)
            pools_tag2.append(ts2)
        pools_txt, pools_tag = pools_txt2, pools_tag2

        flat_all, flat_tag, sizes = [], [], []
        for xs, ts in zip(pools_txt, pools_tag):
            sizes.append(len(xs))
            flat_all.extend(xs)
            flat_tag.extend(ts)

        if self.post is not None and flat_all:
            flat_all = self.post.postprocess_batch([str(x) for x in flat_all])
        flat_all = [_norm_ws(str(x)) for x in flat_all]

        pools_txt2, pools_tag2 = [], []
        k0 = 0
        for sz in sizes:
            xs = flat_all[k0:k0 + sz]
            ts = flat_tag[k0:k0 + sz]
            k0 += sz
            xs2, ts2 = _dedup_keep_order_tagged_local(xs, ts)
            pools_txt2.append(xs2)
            pools_tag2.append(ts2)
        pools_txt, pools_tag = pools_txt2, pools_tag2

        preds, chosen_tags, gaps = [], [], []
        pb_mbr = self._tqdm(len(pools_txt), "MBR/select")
        for xs, ts in zip(pools_txt, pools_tag):
            xs = list(xs)
            ts = list(ts)
            ts_pick = self._tags_for_pick(ts)
            n = len(xs)

            if n == 0:
                preds.append("")
                chosen_tags.append("")
                gaps.append(0.0)
            elif n == 1:
                preds.append(xs[0])
                chosen_tags.append(ts[0] if ts else "")
                gaps.append(0.0)
            else:
                best_i, gap_m, _ = self._pick_mbr(xs, ts_pick)
                bi = int(best_i)
                preds.append(xs[bi])
                chosen_tags.append(ts[bi] if bi < len(ts) else "")
                gaps.append(float(gap_m))

            if pb_mbr is not None:
                pb_mbr.update(1)
        if pb_mbr is not None:
            pb_mbr.close()

        preds = [_norm_ws(x) for x in preds]
        diag = {
            "pool_raw_mean": float(pool_raw_mean),
            "tbm_hit_rate": float(tbm_hit) / max(1, len(src_texts)),
            "knn_hit_rate": float(knn_hit) / max(1, len(src_texts)),
            "mbr_gap_mean": float(np.mean(gaps)) if len(gaps) else 0.0,
        }

        if return_pools and return_tags:
            return preds, pools_txt, pools_tag, chosen_tags, diag
        if return_pools:
            return preds, pools_txt, diag
        if return_tags:
            return preds, pools_tag, chosen_tags, diag
        return preds, diag

    # -------------------------
    # eval: uses val_text_ds and computes official metrics
    # -------------------------
    def _evaluate_mbr(
        self,
        metric_key_prefix="eval",
        *,
        save_dir_override: Optional[str] = None,
        file_tag_override: Optional[str] = None,
    ):
        ex_ids, srcs, refs_raw, oare_ids = _val_unique_examples(self.val_text_ds, prefer_original=True)
        refs = [_norm_ws(r) for r in refs_raw]

        preds, pools_txt, pools_tag, chosen_tags, diag = self._mbr_from_sources(
            list(srcs),
            ex_ids=list(ex_ids),
            add_tbm=True,
            use_knn=True,
            return_pools=True,
            return_tags=True,
        )

        if bool(getattr(Config, "SAVE_EVAL_GENERATIONS", False)):
            try:
                prefix = str(getattr(Config, "EVAL_GENERATIONS_PREFIX", "eval_generations"))
                step = int(getattr(self.state, "global_step", 0) or 0)
                epoch = getattr(self.state, "epoch", None)
                if file_tag_override is not None and str(file_tag_override).strip():
                    tag = str(file_tag_override).strip()
                elif epoch is None:
                    tag = f"step{step}"
                else:
                    try:
                        epoch_f = float(epoch)
                        epoch_tag = f"epoch{epoch_f:.2f}".replace(".", "p")
                    except Exception:
                        epoch_tag = f"epoch{epoch}"
                    tag = f"{epoch_tag}_step{step}"

                if save_dir_override is not None and str(save_dir_override).strip():
                    save_dir = str(save_dir_override)
                else:
                    out_dir = str(self.args.output_dir)
                    ckpt_dir = os.path.join(out_dir, f"checkpoint-{step}")
                    save_dir = ckpt_dir if os.path.isdir(ckpt_dir) else out_dir

                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"{prefix}_{tag}.csv")
                df_out = pd.DataFrame(
                    {
                        "oare_id": [str(x) if x is not None else "" for x in oare_ids],
                        "ex_id": [str(x) for x in ex_ids],
                        "transliteration": [str(x) for x in srcs],
                        "translation": [str(x) for x in refs_raw],
                        "generation": [str(x) for x in preds],
                    }
                )
                df_out.to_csv(out_path, index=False)
                print(f"[EVAL_SAVE] wrote {out_path}", flush=True)
            except Exception as e:
                print(f"[EVAL_SAVE] failed: {e}", flush=True)

        geo, bleu, chrfpp = _official_geo_mean(refs, preds)

        out = {
            f"{metric_key_prefix}_bleu": float(bleu),
            f"{metric_key_prefix}_chrfpp": float(chrfpp),
            f"{metric_key_prefix}_geo_mean": float(geo),

            # f"{metric_key_prefix}_pool_raw_mean": float(diag["pool_raw_mean"]),
            f"{metric_key_prefix}_tbm_hit_rate": float(diag.get("tbm_hit_rate", 0.0)),
            f"{metric_key_prefix}_knn_hit_rate": float(diag.get("knn_hit_rate", 0.0)),
            f"{metric_key_prefix}_mbr_gap_mean": float(diag["mbr_gap_mean"]),
        }

        # pick fractions + used-precision (adds knn view)
        denom = float(max(1, len(preds)))
        pick_origin = {"beam": 0, "samp": 0, "tbm": 0, "knn": 0, "other": 0}
        pick_view   = {"raw": 0, "pn": 0, "gloss": 0, "knn": 0, "other": 0}

        tbm_used_n = tbm_used_good = 0
        pn_used_n  = pn_used_good  = 0
        gloss_used_n = gloss_used_good = 0
        knn_used_n = knn_used_good = 0
        EPS = 1e-6

        for ref_i, pred_i, xs, ts, ctag in zip(refs, preds, pools_txt, pools_tag, chosen_tags):
            ref_i = _norm_ws(ref_i)
            pred_i = _norm_ws(pred_i)

            co = self._origin_from_tag_local(ctag)
            cv = self._view_from_tag_local(ctag)
            pick_origin[co if co in pick_origin else "other"] += 1
            pick_view[cv if cv in pick_view else "other"] += 1

            chosen_geo = float(_geo_sim_sentence(pred_i, ref_i)) if (pred_i and ref_i) else 0.0

            best_non_tbm = -1.0
            best_non_pn = -1.0
            best_non_gloss = -1.0
            best_non_knn = -1.0

            if ref_i:
                for c, tg in zip(xs, ts):
                    c = _norm_ws(str(c))
                    if not c:
                        continue
                    g = float(_geo_sim_sentence(c, ref_i))
                    o2 = self._origin_from_tag_local(tg)
                    v2 = self._view_from_tag_local(tg)

                    if o2 != "tbm":
                        best_non_tbm = max(best_non_tbm, g)
                    if v2 != "pn":
                        best_non_pn = max(best_non_pn, g)
                    if v2 != "gloss":
                        best_non_gloss = max(best_non_gloss, g)
                    if v2 != "knn":
                        best_non_knn = max(best_non_knn, g)

            if self.has_tbm and co == "tbm":
                tbm_used_n += 1
                if (best_non_tbm < 0) or (chosen_geo + EPS >= best_non_tbm):
                    tbm_used_good += 1

            if self.has_pn and cv == "pn":
                pn_used_n += 1
                if (best_non_pn < 0) or (chosen_geo + EPS >= best_non_pn):
                    pn_used_good += 1

            if self.has_gloss and cv == "gloss":
                gloss_used_n += 1
                if (best_non_gloss < 0) or (chosen_geo + EPS >= best_non_gloss):
                    gloss_used_good += 1

            if self.has_knn and cv == "knn":
                knn_used_n += 1
                if (best_non_knn < 0) or (chosen_geo + EPS >= best_non_knn):
                    knn_used_good += 1

        if self.has_tbm:
            out[f"{metric_key_prefix}_pick_tbm_frac"] = float(pick_origin["tbm"]) / denom
            out[f"{metric_key_prefix}_tbm_used_precision"] = float(tbm_used_good) / float(max(1, tbm_used_n))

        if self.has_pn:
            out[f"{metric_key_prefix}_pick_pn_frac"] = float(pick_view["pn"]) / denom
            out[f"{metric_key_prefix}_pn_used_precision"] = float(pn_used_good) / float(max(1, pn_used_n))

        if self.has_gloss:
            out[f"{metric_key_prefix}_pick_gloss_frac"] = float(pick_view["gloss"]) / denom
            out[f"{metric_key_prefix}_gloss_used_precision"] = float(gloss_used_good) / float(max(1, gloss_used_n))

        if self.has_knn:
            out[f"{metric_key_prefix}_pick_knn_frac"] = float(pick_view["knn"]) / denom
            out[f"{metric_key_prefix}_knn_used_precision"] = float(knn_used_good) / float(max(1, knn_used_n))

        return out

    # -------------------------
    # inference
    # -------------------------
    @torch.inference_mode()
    def mbr_predict(
        self,
        src_texts: list[str],
        *,
        add_tbm: bool = True,
        use_knn: bool = True,
    ):
        """
        Inference-time MBR (+ optional MSA polish).
        Returns ONLY preds (list[str]).
        """
        need_pool = bool(self.msa_enable)

        if not need_pool:
            preds, _diag = self._mbr_from_sources(
                list(src_texts),
                ex_ids=None,
                add_tbm=bool(add_tbm),
                use_knn=bool(use_knn),
                return_pools=False,
                return_tags=False,
            )
            return [_norm_ws(x) for x in preds]

        preds, pools, _diag = self._mbr_from_sources(
            list(src_texts),
            ex_ids=None,
            add_tbm=bool(add_tbm),
            use_knn=bool(use_knn),
            return_pools=True,
            return_tags=False,
        )

        _msa = globals().get("msa_consensus", None)
        if _msa is None:
            return [_norm_ws(x) for x in preds]

        def _mbr_gap_plain(p2: list[str]) -> float:
            n = len(p2)
            if n <= 1:
                return 0.0
            sums = np.zeros((n,), dtype=np.float32)
            for i in range(n):
                ai = p2[i]
                for j in range(i + 1, n):
                    s = _geo_sim_sentence(ai, p2[j])
                    sums[i] += s
                    sums[j] += s
            avg = sums / float(n - 1)
            jbest = int(np.argmax(avg))
            best = float(avg[jbest])
            tmp = avg.copy()
            tmp[jbest] = -1e9
            return float(best - float(np.max(tmp)))

        out = []
        for best, pool in zip(preds, pools):
            best = _norm_ws(str(best))
            pool = [_norm_ws(str(p)) for p in (pool or []) if str(p).strip()]

            if (not best) or (len(pool) < int(self.msa_min_pool)):
                out.append(best)
                continue

            if self.msa_gap_thr is not None:
                try:
                    gap = _mbr_gap_plain(_dedup_keep_order(pool))
                except Exception:
                    gap = 0.0
                if float(gap) > float(self.msa_gap_thr):
                    out.append(best)
                    continue

            try:
                out.append(_norm_ws(_msa(best, pool)))
            except Exception:
                out.append(best)

        return out

# ===== Notebook Cell 11 =====
# VAL INSPECTION (MBR) — adapted for MBRGlossSeq2SeqTrainer
# - NO custom generation: reuses trainer._mbr_from_sources() to build the SAME pools
# - Handles RAW / PN / GLOSS + TBM (no kNN in this trainer)
# - Uses trainer tags (beam|raw, samp|pn, tbm|raw, beam|gloss, ...)
# - Computes per-mechanism best-vs-REF diagnostics + oracle + global summary
# ============================================================

def inspect_val_predictions_mbr_glosstrainer(
    trainer,
    *,
    val_text_ds,
    n_show: int = 20,
    sample_seed: int = 1234,
    save_csv_path: str | None = None,

    show_pool: bool = True,
    show_pool_max: int = 12,

    show_src_views: bool = True,
    show_src_views_max: int = 10,

    oracle_pick: bool = True,
    show_oracle_pred: bool = True,

    global_diag: bool = True,
    global_diag_eps: float = 1e-6,
):
 
    def _norm_ws(s: str) -> str:
        return " ".join(str(s).strip().split())

    def _stable_int_id(x) -> int:
        return int(zlib.adler32(str(x).encode("utf-8")) & 0x7fffffff)

    # Prefer your predef helper if present (matches trainer’s MBR sim)
    _geo_sim_sentence_fn = globals().get("_geo_sim_sentence", None)

    # Fallback sentence GEO using sacrebleu (inspection only)
    bleu_metric = chrfpp_metric = None
    if not callable(_geo_sim_sentence_fn):
        try:
            from sacrebleu.metrics import BLEU, CHRF
            try:
                bleu_metric = BLEU(effective_order=True)
            except TypeError:
                bleu_metric = BLEU()
            chrfpp_metric = CHRF(word_order=2)
        except Exception:
            bleu_metric = None
            chrfpp_metric = None

    def _geo_sim(a: str, b: str) -> float:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0
        if callable(_geo_sim_sentence_fn):
            return float(_geo_sim_sentence_fn(a, b))
        if bleu_metric is None or chrfpp_metric is None:
            return 0.0
        bleu = float(bleu_metric.sentence_score(a, [b]).score)
        chrf = float(chrfpp_metric.sentence_score(a, [b]).score)
        return float(math.sqrt(max(0.0, bleu) * max(0.0, chrf)))

    def _sent_metrics(pred: str, ref: str):
        pred = (pred or "").strip()
        ref  = (ref or "").strip()
        if not pred or not ref:
            return float("nan"), float("nan"), float("nan")
        if bleu_metric is None or chrfpp_metric is None:
            # if we don’t have sacrebleu, at least report GEO via _geo_sim_sentence
            return float("nan"), float("nan"), float(_geo_sim(pred, ref))
        b = float(bleu_metric.sentence_score(pred, [ref]).score)
        c = float(chrfpp_metric.sentence_score(pred, [ref]).score)
        g = math.sqrt(max(b, 0.0) * max(c, 0.0))
        return b, c, g

    def _oracle_pick(cands: list[str], ref: str) -> tuple[int, float]:
        if not cands:
            return 0, float("nan")
        best_i, best_g = 0, -1e9
        for i, c in enumerate(cands):
            g = _geo_sim(str(c), ref) if ref else -1e9
            if g > best_g:
                best_g, best_i = float(g), int(i)
        return int(best_i), float(best_g)

    # ---------------------------
    # tag parsing consistent with our trainer
    # ---------------------------
    def _origin(tag: str) -> str:
        t = "" if tag is None else str(tag)
        o = t.split("|", 1)[0] if "|" in t else t
        if o in ("beam", "samp", "tbm"):
            return o
        return o or "other"

    def _view(tag: str) -> str:
        t = "" if tag is None else str(tag)
        if "|" in t:
            o, v = t.split("|", 1)
        else:
            o, v = t, ""
        if o == "tbm":
            return "raw"  # TBM treated as raw view
        v = v or "raw"
        if v.startswith("gloss"):
            return "gloss"
        if v == "pn":
            return "pn"
        if v == "raw":
            return "raw"
        return v or "other"

    def _is_raw_non_tbm(tag: str) -> bool:
        return (_view(tag) == "raw") and (_origin(tag) != "tbm")

    def _is_pn(tag: str) -> bool:
        return _view(tag) == "pn"

    def _is_gloss(tag: str) -> bool:
        return _view(tag) == "gloss"

    def _is_tbm(tag: str) -> bool:
        return _origin(tag) == "tbm"

    # ---------------------------
    # dataset access (datasets.Dataset or pandas.DataFrame)
    # ---------------------------
    if hasattr(val_text_ds, "column_names"):
        cols = set(val_text_ds.column_names)
        get_row = lambda i: val_text_ds[int(i)]
        n_rows = len(val_text_ds)
        src_col = "transliteration" if "transliteration" in cols else ("src" if "src" in cols else None)
        tgt_col = "translation" if "translation" in cols else ("tgt" if "tgt" in cols else None)

        if src_col is None:
            raise ValueError(f"val_text_ds must have transliteration/src. columns={list(val_text_ds.column_names)}")

        if "ex_id" in cols:
            ex_ids_all = list(map(str, val_text_ds["ex_id"]))
        elif "id" in cols:
            ex_ids_all = list(map(str, val_text_ds["id"]))
        else:
            ex_ids_all = [str(i) for i in range(n_rows)]

    else:
        # pandas DataFrame fallback
        dfv = val_text_ds
        cols = set(dfv.columns)
        get_row = lambda i: dfv.iloc[int(i)].to_dict()
        n_rows = len(dfv)
        src_col = "transliteration" if "transliteration" in cols else ("src" if "src" in cols else None)
        tgt_col = "translation" if "translation" in cols else ("tgt" if "tgt" in cols else None)

        if src_col is None:
            raise ValueError(f"val_text_ds must have transliteration/src. columns={list(dfv.columns)}")

        if "ex_id" in cols:
            ex_ids_all = list(map(str, dfv["ex_id"].values))
        elif "id" in cols:
            ex_ids_all = list(map(str, dfv["id"].values))
        else:
            ex_ids_all = [str(i) for i in range(n_rows)]

    rng = random.Random(int(sample_seed))
    all_idxs = list(range(n_rows))
    rng.shuffle(all_idxs)
    show_ds_idxs = all_idxs[: min(int(n_show), n_rows)]

    # sources + refs (raw)
    src_raw_list = [str(get_row(i).get(src_col, "")) for i in show_ds_idxs]
    refs_raw = []
    if tgt_col is not None:
        for i in show_ds_idxs:
            tv = get_row(i).get(tgt_col, None)
            refs_raw.append("" if tv is None else str(tv))
    else:
        refs_raw = [""] * len(show_ds_idxs)

    refs = [_norm_ws(r) for r in refs_raw]
    ex_ids_sel = [str(ex_ids_all[i]) for i in show_ds_idxs]

    # ---------------------------
    # Build pools via trainer internals (the whole point)
    # ---------------------------
    preds, pools_txt, pools_tag, chosen_tags, diag = trainer._mbr_from_sources(
        list(src_raw_list),
        ex_ids=list(ex_ids_sel),
        add_tbm=True,
        return_pools=True,
        return_tags=True,
    )

    preds = [_norm_ws(p) for p in preds]

    # ---------------------------
    # Optional: reconstruct SRC views (for display only)
    # (this does NOT affect pools; just helps you debug)
    # ---------------------------
    src_clean_disp = trainer.pre.preprocess_batch(list(map(str, src_raw_list)))
    ex_src_views: dict[int, dict[str, str]] = {}

    def _is_effective_gloss(base: str, s_gl: str) -> bool:
        if s_gl is None:
            return False
        s_gl = str(s_gl)
        if not s_gl.strip():
            return False
        if _norm_ws(s_gl) == _norm_ws(base):
            return False
        if ("<extra_id_0>" not in s_gl) and ("TERMS:" not in s_gl):
            return False
        return True

    for pos, (eid, base) in enumerate(zip(ex_ids_sel, src_clean_disp)):
        base = str(base)
        ex_src_views[pos] = {"raw": base}

        if getattr(trainer, "has_pn", False):
            pn = base
            try:
                pn = trainer.canon.canonicalize_source(base, mode=getattr(trainer, "pn_mode", "pn_norm"))
            except Exception:
                pn = base
            pn = "" if pn is None else str(pn).strip()
            if pn and _norm_ws(pn) != _norm_ws(base):
                ex_src_views[pos]["pn"] = pn

        if getattr(trainer, "has_gloss", False):
            ex_int = _stable_int_id(eid)
            V = int(getattr(trainer, "gloss_variants", 0) or 0)
            for v in range(V):
                vseed = int(getattr(trainer, "gloss_seed", 0)) + 1009 * int(v)
                try:
                    s_gl = trainer.glosser.append_gloss(
                        base,
                        max_items=int(getattr(trainer, "gloss_max_items", 6)),
                        max_append_chars=int(getattr(trainer, "gloss_max_append_chars", 240)),
                        seed=int(vseed),
                        epoch=0,
                        example_id=int(ex_int),
                        keep_order=True,
                    )
                except Exception:
                    s_gl = None
                if _is_effective_gloss(base, s_gl):
                    ex_src_views[pos][f"gloss{v}"] = str(s_gl)

    # ---------------------------
    # per example: diagnostics
    # ---------------------------
    from collections import Counter

    def _best_geo_for(xs, ts, ref, filt):
        if not ref:
            return float("nan")
        best_g = -1e9
        hit = False
        for p, tg in zip(xs, ts):
            if not filt(str(tg)):
                continue
            hit = True
            g = _geo_sim(str(p), ref)
            if g > best_g:
                best_g = float(g)
        return float(best_g) if hit else float("nan")

    rows = []
    for pos, ds_i in enumerate(show_ds_idxs):
        ref = refs[pos] if pos < len(refs) else ""
        pred = preds[pos] if pos < len(preds) else ""
        xs = list(pools_txt[pos] or [])
        ts = list(pools_tag[pos] or [])
        chosen_tag = str(chosen_tags[pos] if pos < len(chosen_tags) else "")

        bleu_s, chrfpp_s, geo_s = _sent_metrics(pred, ref) if ref else (float("nan"), float("nan"), float("nan"))
        chosen_geo = float(_geo_sim(pred, ref)) if (pred and ref) else float("nan")

        cnt = Counter(ts)
        majority_tag = ""
        if cnt:
            bestc = max(cnt.values())
            for tg in ts:  # stable tie-break
                if cnt.get(tg, 0) == bestc:
                    majority_tag = tg
                    break

        oracle_pred = oracle_tag = ""
        oracle_geo = oracle_gain_geo = float("nan")
        if bool(oracle_pick) and ref and xs:
            oi, og = _oracle_pick(xs, ref)
            oracle_pred = _norm_ws(xs[int(oi)])
            oracle_tag = ts[int(oi)] if int(oi) < len(ts) else ""
            oracle_geo = float(og)
            oracle_gain_geo = (oracle_geo - float(chosen_geo)) if np.isfinite(oracle_geo) and np.isfinite(chosen_geo) else float("nan")

        best_raw_geo   = _best_geo_for(xs, ts, ref, _is_raw_non_tbm)
        best_pn_geo    = _best_geo_for(xs, ts, ref, _is_pn)
        best_gloss_geo = _best_geo_for(xs, ts, ref, _is_gloss)
        best_tbm_geo   = _best_geo_for(xs, ts, ref, _is_tbm)

        gain_raw_geo   = (best_raw_geo   - float(chosen_geo)) if np.isfinite(best_raw_geo)   and np.isfinite(chosen_geo) else float("nan")
        gain_pn_geo    = (best_pn_geo    - float(chosen_geo)) if np.isfinite(best_pn_geo)    and np.isfinite(chosen_geo) else float("nan")
        gain_gloss_geo = (best_gloss_geo - float(chosen_geo)) if np.isfinite(best_gloss_geo) and np.isfinite(chosen_geo) else float("nan")
        gain_tbm_geo   = (best_tbm_geo   - float(chosen_geo)) if np.isfinite(best_tbm_geo)   and np.isfinite(chosen_geo) else float("nan")

        pool_show = []
        if bool(show_pool):
            for p, tg in list(zip(xs, ts))[: int(show_pool_max)]:
                marks = []
                if _norm_ws(p) == pred:
                    marks.append("CHOSEN")
                if oracle_pred and (_norm_ws(p) == oracle_pred):
                    marks.append("ORACLE")
                mark = (" <== " + "&".join(marks)) if marks else ""
                pool_show.append(f"[{tg}] {p}{mark}")

        src_views = ex_src_views.get(pos, {"raw": str(src_clean_disp[pos])})
        view_order = []
        for vv in ["raw", "pn"]:
            if vv in src_views:
                view_order.append(vv)
        for vv in sorted([k for k in src_views.keys() if k.startswith("gloss")],
                         key=lambda x: int(x.replace("gloss", ""))):
            view_order.append(vv)

        src_view_lines = []
        if bool(show_src_views):
            for vv in view_order[: int(show_src_views_max)]:
                src_view_lines.append(f"SRC_{vv.upper()}: {src_views.get(vv,'')}")

        rows.append({
            "ds_idx": int(ds_i),
            "ex_id": str(ex_ids_sel[pos]),
            "pool_n": int(len(xs)),
            "bleu": float(bleu_s),
            "chrfpp": float(chrfpp_s),
            "geo": float(geo_s),
            "ref": str(ref),
            "pred": str(pred),

            "pred_tag": str(chosen_tag),
            "majority_tag": str(majority_tag),
            "tag_counts_json": json.dumps(cnt, ensure_ascii=False),

            "oracle_pred": str(oracle_pred),
            "oracle_tag": str(oracle_tag),
            "oracle_geo": float(oracle_geo),
            "oracle_gain_geo": float(oracle_gain_geo),

            "best_raw_geo": float(best_raw_geo),
            "best_pn_geo": float(best_pn_geo),
            "best_gloss_geo": float(best_gloss_geo),
            "best_tbm_geo": float(best_tbm_geo),

            "gain_raw_geo": float(gain_raw_geo),
            "gain_pn_geo": float(gain_pn_geo),
            "gain_gloss_geo": float(gain_gloss_geo),
            "gain_tbm_geo": float(gain_tbm_geo),

            "pool": xs,
            "pool_tags": ts,
            "pool_show": pool_show,
            "src_view_lines": src_view_lines,
        })

    df_show = pd.DataFrame(rows)

    def _trunc(s, m=180):
        s = str(s).replace("\n", " ")
        return s if len(s) <= m else s[:m] + " …"

    def _fmt(x):
        return "nan" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3f}"

    for _, r in df_show.iterrows():
        print("=" * 90)
        print(
            f"ds_idx: {int(r['ds_idx'])} | ex_id: {r['ex_id']} | pool={int(r['pool_n'])} | "
            f"BLEU={_fmt(r['bleu'])} | chrF++={_fmt(r['chrfpp'])} | GEO={_fmt(r['geo'])}"
        )
        print(f"CHOSEN_TAG: {r['pred_tag']} | MAJORITY_TAG: {r['majority_tag']} | tag_counts={r['tag_counts_json']}")

        if bool(oracle_pick) and str(r.get("oracle_pred", "")).strip():
            print(f"ORACLE_TAG: {r['oracle_tag']} | ORACLE_GEO={_fmt(r['oracle_geo'])} | GAIN_GEO={_fmt(r['oracle_gain_geo'])}")
            if bool(show_oracle_pred):
                print("ORACLE_PRED:", _trunc(r["oracle_pred"], m=220))

        if bool(show_src_views):
            for line in (r["src_view_lines"] or []):
                print(_trunc(line, m=260))

        print("REF :", _trunc(r["ref"]))
        print("PRED:", _trunc(r["pred"]))

        if bool(show_pool):
            print("POOL(after post+dedupe+cap):")
            for line in (r["pool_show"] or []):
                print("  ", _trunc(line, m=260))

    # ---------------------------
    # global diagnostics
    # ---------------------------
    if bool(global_diag) and len(df_show):
        def _summ(best_col, gain_col):
            x = df_show[best_col].astype(float).values
            g = df_show[gain_col].astype(float).values
            cov = np.isfinite(x)
            cov_frac = float(np.mean(cov)) if len(cov) else 0.0
            mean_best = float(np.nanmean(x)) if np.any(cov) else float("nan")
            mean_gain = float(np.nanmean(g)) if np.any(np.isfinite(g)) else float("nan")
            win_frac = float(np.mean((g > float(global_diag_eps)) & np.isfinite(g))) if len(g) else 0.0
            return cov_frac, mean_best, win_frac, mean_gain

        raw_cov, raw_best, raw_win, raw_gain = _summ("best_raw_geo", "gain_raw_geo")
        pn_cov,  pn_best,  pn_win,  pn_gain  = _summ("best_pn_geo", "gain_pn_geo")
        gl_cov,  gl_best,  gl_win,  gl_gain  = _summ("best_gloss_geo", "gain_gloss_geo")
        tb_cov,  tb_best,  tb_win,  tb_gain  = _summ("best_tbm_geo", "gain_tbm_geo")

        tags = df_show["oracle_tag"].astype(str).values if "oracle_tag" in df_show.columns else np.array([], dtype=str)
        tags = [t for t in tags if str(t).strip()]

        def _oracle_frac_origin(origin: str) -> float:
            if not tags:
                return 0.0
            return float(np.mean([str(t).startswith(origin + "|") for t in tags]))

        def _oracle_frac_view(view: str) -> float:
            if not tags:
                return 0.0
            if view == "gloss":
                return float(np.mean([("|" in t and _view(t) == "gloss") for t in tags]))
            if view == "pn":
                return float(np.mean([("|" in t and _view(t) == "pn") for t in tags]))
            if view == "raw":
                return float(np.mean([("|" in t and _view(t) == "raw" and (not str(t).startswith("tbm|"))) for t in tags]))
            return 0.0

        oracle_tbm  = _oracle_frac_origin("tbm")
        oracle_gl   = _oracle_frac_view("gloss")
        oracle_pn   = _oracle_frac_view("pn")
        oracle_raw  = _oracle_frac_view("raw")

        print("\n" + "=" * 90)
        print("GLOBAL VAL DIAGNOSTICS (on inspected subset)")
        print(f"n={len(df_show)} | tbm_hit_rate={float(diag.get('tbm_hit_rate',0.0)):.3f} | mbr_gap_mean={float(diag.get('mbr_gap_mean',0.0)):.3f}")
        print("- Mechanism coverage / best-vs-ref / wins vs chosen / mean gain (sentence GEO)")
        print(f"RAW  : cov={raw_cov:.3f}  bestGEO={raw_best:.3f}  win={raw_win:.3f}  gain={raw_gain:.3f}")
        print(f"PN   : cov={pn_cov:.3f}   bestGEO={pn_best:.3f}   win={pn_win:.3f}   gain={pn_gain:.3f}")
        print(f"GLOSS: cov={gl_cov:.3f}  bestGEO={gl_best:.3f}  win={gl_win:.3f}  gain={gl_gain:.3f}")
        print(f"TBM  : cov={tb_cov:.3f}   bestGEO={tb_best:.3f}   win={tb_win:.3f}   gain={tb_gain:.3f}")
        print("- Oracle source fractions (best candidate vs REF within pool)")
        print(f"oracle_raw={oracle_raw:.3f}  oracle_pn={oracle_pn:.3f}  oracle_gloss={oracle_gl:.3f}  oracle_tbm={oracle_tbm:.3f}")
        print("=" * 90 + "\n")

    return df_show

def main():
    rank, local_rank, world_size = _setup_distributed_device()
    if world_size > 1 and rank == 0:
        print(f"[DIST] world_size={world_size} local_rank={local_rank}", flush=True)
    if rank == 0:
        _dump_training_config(Config)

    # Training

    model_parallel_info = _configure_model_parallel(
        Config,
        rank=rank,
        world_size=world_size,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        Config.MODEL_NAME,
        **dict(model_parallel_info.get("from_pretrained_kwargs", {})),
    )
    if bool(model_parallel_info.get("enabled")):
        model_parallel_info["primary_device"] = _get_model_primary_device_str(model)
        if rank == 0:
            print(f"[MODEL_PARALLEL] primary_device={model_parallel_info['primary_device']}", flush=True)

    if Config.RESET_DECODER:
        _ = reset_t5_decoder(model, seed=Config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    pre = OptimizedPreprocessor()
    post_out = VectorizedPostprocessor(
        aggressive=True,               
        mode='train',
    )

    # -------------------------
    # Load comp train/test
    # -------------------------
    comp_df = pd.read_csv(Config.TRAIN_CSV_PATH).assign(is_extra=False)

    final_path_raw = getattr(Config, "TRAIN_FINAL_CSV_PATH", "")
    final_path = "" if final_path_raw is None else str(final_path_raw).strip()
    if final_path.lower() in {"", "none", "null"}:
        print("[DATA] TRAIN_FINAL_CSV_PATH disabled; using only TRAIN_CSV_PATH.", flush=True)
        final_comp_df = pd.DataFrame(
            columns=["oare_id", "transliteration", "translation", "is_extra"]
        )
    else:
        if not os.path.exists(final_path):
            raise FileNotFoundError(f"TRAIN_FINAL_CSV_PATH does not exist: {final_path}")
        final_comp_df = pd.read_csv(final_path).assign(is_extra=False)

    # -------------------------
    # Load + normalize ALL extras (Larsen + hybrid + train1..3)
    # -------------------------
    extras = []

    # Larsen
    df = load_and_sanitize_parallel(Config.LARSEN_LETTERS_PATH).assign(is_extra=True, source="larsen")
    df = df[["transliteration", "translation", "is_extra", "source"]].copy()
    df["transliteration"] = df["transliteration"].astype(str).map(normalize_external_transliteration)
    df["translation"]     = df["translation"].astype(str).map(normalize_external_translation)
    extras.append(df)

    extra_df = pd.concat(extras, ignore_index=True)
    extra_df = extra_df.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)
    extra_df["oare_id"] = [f"extra::{i}" for i in range(len(extra_df))]

    # -------------------------
    # Filter incomplete separately
    # -------------------------
    bad_comp   = flag_incomplete(comp_df)
    comp_clean = comp_df.loc[~bad_comp].reset_index(drop=True)
    bad_final_comp   = flag_incomplete(final_comp_df)
    final_comp_clean = final_comp_df.loc[~bad_final_comp].reset_index(drop=True)

    bad_extra   = flag_incomplete(extra_df)
    extra_clean = extra_df.loc[~bad_extra].reset_index(drop=True)

    core_clean = pd.concat([comp_clean, final_comp_clean], ignore_index=True)
    core_clean = core_clean.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)

    num_folds = int(getattr(Config, "NUM_FOLDS", 10))
    fold_index = int(getattr(Config, "FOLD_INDEX", 0))

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
            test_size=float(getattr(Config, "VAL_SIZE", 0.1)),
            random_state=int(getattr(Config, "SEED", 42)),
        )
        tr_idx, va_idx = next(gss.split(core_clean, groups=core_clean["oare_id"].astype(str)))
        train_core_df = core_clean.iloc[tr_idx].reset_index(drop=True)
        val_split_df = core_clean.iloc[va_idx].reset_index(drop=True)
        train_split_df = pd.concat([train_core_df, extra_clean], ignore_index=True)
        train_split_df = train_split_df.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)
        print(
            f"[SPLIT] GroupShuffleSplit val_size={float(getattr(Config, 'VAL_SIZE', 0.1))} | "
            f"core_train: {len(train_core_df)}, extra_train: {len(extra_clean)}, "
            f"train_total: {len(train_split_df)}, val: {len(val_split_df)}",
            flush=True,
        )

    # leakage guard
    tr_ids = set(train_split_df["oare_id"].astype(str).tolist())
    va_ids = set(val_split_df["oare_id"].astype(str).tolist())
    overlap = tr_ids & va_ids
    if overlap:
        raise ValueError(f"[SPLIT] LEAKAGE DETECTED: {len(overlap)} oare_id appear in both splits.")

    # HF conversion  
    train_for_tokenize = Dataset.from_pandas(train_split_df, preserve_index=False)
    val_for_tokenize   = Dataset.from_pandas(val_split_df,   preserve_index=False)

    drop_inc_cols = ["src_chars","tgt_chars","tgt_over_src","header_only","length_rule","flag"]
    train_for_tokenize = train_for_tokenize.remove_columns([c for c in drop_inc_cols if c in train_for_tokenize.column_names])
    val_for_tokenize   = val_for_tokenize.remove_columns([c for c in drop_inc_cols if c in val_for_tokenize.column_names])

    # stable ex_id (MBR grouping / sentence tracking)
    def _add_ex_id(examples, indices):
        out = []
        for idx, oid in zip(indices, examples["oare_id"]):
            out.append(f"{str(oid)}::row{int(idx)}")
        return {"ex_id": out}

    train_for_tokenize = train_for_tokenize.map(_add_ex_id, batched=True, with_indices=True)
    val_for_tokenize   = val_for_tokenize.map(_add_ex_id,   batched=True, with_indices=True)

    before = len(val_for_tokenize)
    val_for_tokenize = val_for_tokenize.filter(
        lambda batch, thr: [len(str(s).split()) <= int(thr) for s in batch["translation"]],
        batched=True,
        fn_kwargs={"thr": int(getattr(Config, "VAL_CUTOFF_WORD_THR", 60))},
        num_proc=1,
    )
    print(f"[VAL_LEN_CAP] kept={len(val_for_tokenize)}/{before} | dropped={before-len(val_for_tokenize)}", flush=True)

    # -------------------------
    # PN+gloss resources (built AFTER cleaning)
    # -------------------------
    LEXICON_PATH     = str(getattr(Config, "LEXICON_PATH", ""))
    ONOMASTICON_PATH = str(getattr(Config, "ONOMASTICON_PATH", ""))
    EBL_DICT_PATH    = str(getattr(Config, "EBL_DICT_PATH", ""))

    canon = SourceCanonicalizer.from_csvs(LEXICON_PATH, ONOMASTICON_PATH, use_norm=True)

    train_texts = pre.preprocess_batch(list(train_for_tokenize["transliteration"]))
    glosser = GlossAugmenter(
        LEXICON_PATH,
        EBL_DICT_PATH,
        train_texts=train_texts,
    )
    joblib.dump(glosser, os.path.join(Config.OUTPUT_DIR, "glosser.joblib"))

    # -------------------------
    # Build TRAIN/VAL variants
    # -------------------------
    tokenized_train, tokenized_val, shared_epoch = build_probe_then_pngloss_variants(
        Config=Config,
        train_text_ds=train_for_tokenize,
        val_text_ds=val_for_tokenize,
        tokenizer=tokenizer,
        pre=pre,
        canon=canon,
        glosser=glosser,
        NPROC=int(getattr(Config, "NPROC", 8)),
        MAP_BS=int(getattr(Config, "MAP_BS", 2048)),
    )

    # -------------------------
    # TBM pairs for MBR eval
    # -------------------------
    tbm_pairs_df = train_split_df[["transliteration","translation"]].copy()
    tbm_pairs_df.to_csv(os.path.join(Config.OUTPUT_DIR, "tbm_pairs.csv"), index=False)  
    print(f"[TBM] from train pairs (normalized): {len(tbm_pairs_df)}", flush=True)
    # -------------------------
    # kNN bank (optional) + rebuild-on-best-eval callback (LEAN)
    # -------------------------
    KNN_ENABLE = bool(getattr(Config, "KNN_ENABLE", True)) and Config.USE_VAL_FOR_TRAINING and (rank == 0)

    knn_mem = None
    if KNN_ENABLE:
        bank_src_knn = list(train_src_for_gloss)
        bank_tgt_knn = list(train_tgt_for_gloss)

        KNN_CACHE_DIR = str(getattr(
            Config,
            "KNN_CACHE_DIR",
            os.path.join(str(getattr(Config, "OUTPUT_DIR", "/kaggle/working")), "knn_bank_cache"),
        ))

        dev_str = _get_model_primary_device_str(model)
        tmp_model = model if bool(model_parallel_info.get("enabled")) else (model.to(dev_str) if str(dev_str).startswith("cuda") else model)
        tmp_model.eval()

        knn_mem = build_or_load_knn_memory_sklearn(
            cache_dir=KNN_CACHE_DIR,
            model=tmp_model,
            tokenizer=tokenizer,
            bank_src_texts=bank_src_knn,
            bank_tgt_texts=bank_tgt_knn,
            prefix_for_encode=Config.PREFIX,
            device=dev_str,
            max_length=int(getattr(Config, "SRC_MAX_LENGTH", 512)),
            batch_size=int(getattr(Config, "KNN_BANK_BS", 128)),
            use_bf16=bool(getattr(Config, "KNN_USE_BF16", True)),
            rebuild_on_mismatch=True,
        )
        print(f"[kNN] ready: n={len(knn_mem['tgts'])} | cache={KNN_CACHE_DIR}", flush=True)

    # ============================================================
    # Warmup
    # ============================================================
    warmup_steps = compute_warmup_steps(
        num_examples=len(tokenized_train),
        per_device_bs=Config.BATCH_SIZE,
        grad_accum=getattr(Config, "GRAD_ACCUM", 1),
        epochs=Config.EPOCHS,
        warmup_ratio=getattr(Config, "WARMUP_RATIO", 0.05),
    )

    # ============================================================
    # Model + collator + Trainer
    # ============================================================
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    sanitize_generation_config_for_saving(
        model,
        default_num_beams=int(getattr(Config, "NUM_BEAMS", 8)),
        default_len_pen=float(getattr(Config, "GEN_LENGTH_PENALTY", 1.0)),
    )

    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,

        eval_strategy="epoch" if Config.USE_VAL_FOR_TRAINING else "no",
        save_strategy="epoch",

        save_total_limit=20,
        save_only_model=True,
        load_best_model_at_end=True if Config.USE_VAL_FOR_TRAINING else False,
        metric_for_best_model="eval_geo_mean",
        greater_is_better=True,

        bf16=use_bf16,
        fp16=False,

        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=getattr(Config, "GRAD_ACCUM", 1),

        group_by_length=True,
        length_column_name="input_length",

        learning_rate=Config.LEARNING_RATE,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=Config.EPOCHS,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=warmup_steps,
        prediction_loss_only=False,

        optim="adamw_torch_fused",
        label_smoothing_factor=Config.LABEL_SMOOTHING,

        predict_with_generate=False,  # generate inside evaluate()
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,

        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
    )

    gen_cfg = model.generation_config
    gen_cfg.repetition_penalty = float(getattr(Config, "GEN_REPETITION_PENALTY", 1.0))
    gen_cfg.no_repeat_ngram_size = int(getattr(Config, "GEN_NO_REPEAT_NGRAM", 0)) or 0
    model.generation_config = gen_cfg
    model.config.use_cache = False

    eval_placeholder = tokenized_val.select(range(min(64, len(tokenized_val)))) if len(tokenized_val) else tokenized_val
    trainer_callbacks = []
    if Config.USE_VAL_FOR_TRAINING and bool(getattr(Config, "EARLY_STOPPING_ENABLE", True)):
        trainer_callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(getattr(Config, "EARLY_STOPPING_PATIENCE", 3)),
                early_stopping_threshold=float(getattr(Config, "EARLY_STOPPING_THRESHOLD", 3e-1)),
            )
        )
    if Config.USE_VAL_FOR_TRAINING:
        print(
            f"[TRAIN] early_stopping={'on' if trainer_callbacks else 'off'} "
            f"(patience={int(getattr(Config, 'EARLY_STOPPING_PATIENCE', 3))}, "
            f"threshold={float(getattr(Config, 'EARLY_STOPPING_THRESHOLD', 3e-1))})",
            flush=True,
        )

    trainer = MBRGlossSeq2SeqTrainer(
        model=model,
        args=args,

        train_dataset=tokenized_train,
        eval_dataset=eval_placeholder if Config.USE_VAL_FOR_TRAINING else None,

        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=None,

        callbacks=trainer_callbacks,

        val_text_ds=val_for_tokenize,
        pre=pre,
        prefix=Config.PREFIX,

        post=post_out,
        post_ref=None,

        glosser=glosser,
        gloss_variants=int(getattr(Config, "MBR_GLOSS_VARIANTS", 2)),
        gloss_seed=int(getattr(Config, "GLOSS_SEED", int(getattr(Config, "SEED", 42)) + 777)),
        gloss_max_items=int(getattr(Config, "GLOSS_MAX_ITEMS", 6)),
        gloss_max_append_chars=int(getattr(Config, "GLOSS_MAX_APPEND_CHARS", 240)),

        canon=canon,
        pn_enable=bool(getattr(Config, "PN_ENABLE", False)),

        mbr_batch_size_inputs=int(getattr(Config, "MBR_BATCH_SIZE_INPUTS", 16)),
        src_max_length=Config.SRC_MAX_LENGTH,
        max_new_tokens=Config.GEN_MAX_NEW_TOKENS,

        num_beams=int(getattr(Config, "MBR_NUM_BEAMS", int(getattr(Config, "NUM_BEAMS", 8)))),
        num_beam_cands=int(getattr(Config, "MBR_NUM_BEAM_CANDS", 2)),
        num_sample_cands=int(getattr(Config, "MBR_NUM_SAMPLE_CANDS", 4)),
        mbr_pool_cap=int(getattr(Config, "MBR_POOL_CAP", 32)),
        length_penalty=float(getattr(Config, "GEN_LENGTH_PENALTY", 1.3)),
        temperature=float(getattr(Config, "MBR_TEMPERATURE", 0.7)),
        top_p=float(getattr(Config, "MBR_TOP_P", 0.9)),
        repetition_penalty=float(getattr(Config, "GEN_REPETITION_PENALTY", 1.0)),
        no_repeat_ngram_size=int(getattr(Config, "GEN_NO_REPEAT_NGRAM", 0)) or 0,

        tbm_pairs=tbm_pairs_df,
        tbm_enable=bool(getattr(Config, "TBM_ENABLE", True)),
        tbm_topk=int(getattr(Config, "TBM_TOPK", 3)),
        tbm_min_sim=float(getattr(Config, "TBM_MIN_SIM", 0.92)),
        tbm_hard_sim=float(getattr(Config, "TBM_HARD_SIM", 0.97)),
        tbm_ngram=(int(getattr(Config, "TBM_NGRAM_MIN", 3)), int(getattr(Config, "TBM_NGRAM_MAX", 6))),
        tbm_max_features=int(getattr(Config, "TBM_MAX_FEATURES", 250_000)),

        knn_enable=bool(getattr(Config, "KNN_ENABLE", False)) and (knn_mem is not None),
        knn_mem=knn_mem,  # from your kNN build/load block (or None)
        knn_topk=int(getattr(Config, "KNN_TOPK", 8)),
        knn_hint_k=int(getattr(Config, "KNN_HINT_K", 2)),
        knn_hint_max_chars=int(getattr(Config, "KNN_HINT_MAX_CHARS", 240)),
        knn_ret_k=int(getattr(Config, "KNN_RET_K", 1)),
        knn_prefix_for_encode=str(getattr(Config, "KNN_PREFIX_FOR_ENCODE", Config.PREFIX)),
        knn_query_bs=int(getattr(Config, "KNN_QUERY_BS", 32)),
        knn_min_sim=float(getattr(Config, "KNN_MIN_SIM", 0.90)),
        knn_hard_sim=float(getattr(Config, "KNN_HARD_SIM", 0.94)),
    )

    class _SetSharedEpochCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            try:
                shared_epoch.value = int(state.epoch or 0)
            except Exception:
                pass
            return control

    trainer.add_callback(_SetSharedEpochCallback())

    if Config.KNN_ENABLE and Config.USE_VAL_FOR_TRAINING and (knn_mem is not None):
        if hasattr(trainer, "knn_nn"):
            trainer.knn_nn = knn_mem["nn"]
        if hasattr(trainer, "knn_tgts"):
            trainer.knn_tgts = knn_mem["tgts"]

        # rebuild only when eval improves
        trainer.add_callback(RebuildKNNOnBestEvalCallback(
            trainer_ref=trainer,
            bank_src_texts=bank_src_knn,
            bank_tgt_texts=bank_tgt_knn,
            cache_dir=KNN_CACHE_DIR,
            prefix_for_encode=Config.PREFIX,
            monitor_key=str(getattr(Config, "KNN_MONITOR_KEY", "eval_geo_mean")),
            mode=str(getattr(Config, "KNN_MONITOR_MODE", "max")),
            eps=float(getattr(Config, "KNN_MONITOR_EPS", 1e-6)),
            max_length=int(getattr(Config, "SRC_MAX_LENGTH", 512)),
            batch_size=int(getattr(Config, "KNN_BANK_BS", 128)),
            use_bf16=bool(getattr(Config, "KNN_USE_BF16", True)),
            rebuild_on_mismatch=True,
            also_attach_to_trainer=True,
        ))
        print("[kNN] callback added: rebuild on best eval", flush=True)

    print("Starting Training...")
    trainer.train()

    print("\nBest checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric:", trainer.state.best_metric)

    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    print("Saved BEST model to:", Config.OUTPUT_DIR)

    ckpt_avg_k = int(getattr(Config, "CKPT_AVG_K", 5))
    best_metric_key = str(getattr(Config, "BEST_METRIC_KEY", "eval_geo_mean"))
    avg_dir_default = os.path.join(Config.OUTPUT_DIR, f"ckpt_avg_best{ckpt_avg_k}")
    do_avg_eval = bool(getattr(Config, "EVAL_AVG_CHECKPOINT", True))

    avg_dir = None
    if trainer.is_world_process_zero():
        try:
            avg_dir, chosen = average_checkpoints_and_save(
                model=trainer.model,
                output_dir=Config.OUTPUT_DIR,
                save_dir=avg_dir_default,
                k=ckpt_avg_k,
                metric_key=best_metric_key,
                prefer_best=True,
                base_ckpt_for_config=trainer.state.best_model_checkpoint,
                cleanup_checkpoints=bool(getattr(Config, "CKPT_AVG_CLEANUP", False)),
            )
            tokenizer.save_pretrained(avg_dir)
            print(f"Saved AVERAGED model to: {avg_dir}", flush=True)
            print(f"[CKPT_AVG] chosen checkpoints: {chosen}", flush=True)
        except Exception as e:
            print(f"[CKPT_AVG] failed: {e}", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if do_avg_eval and Config.USE_VAL_FOR_TRAINING and trainer.is_world_process_zero():
        try:
            avg_eval_dir = avg_dir or avg_dir_default
            print("[AVG_EVAL] Loading averaged checkpoint and running validation...", flush=True)
            avg_sd = _load_state_dict_any(avg_eval_dir, map_location="cpu")
            missing, unexpected = trainer.model.load_state_dict(avg_sd, strict=False)
            print(
                f"[AVG_EVAL] state_dict loaded: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
            trainer.model.eval()
            avg_prefix = str(getattr(Config, "EVAL_AVG_METRIC_PREFIX", "eval_avg"))
            avg_metrics = trainer._evaluate_mbr(
                metric_key_prefix=avg_prefix,
                save_dir_override=avg_eval_dir,
                file_tag_override="avg",
            )
            avg_metrics_path = os.path.join(avg_eval_dir, f"{avg_prefix}_metrics.json")
            with open(avg_metrics_path, "w", encoding="utf-8") as f:
                json.dump(avg_metrics, f, indent=2)
            print(f"[AVG_EVAL] metrics: {avg_metrics}", flush=True)
            print(f"[AVG_EVAL] wrote {avg_metrics_path}", flush=True)
        except Exception as e:
            print(f"[AVG_EVAL] failed: {e}", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if getattr(Config, "CLEAN_CHECKPOINTS", True) and Config.USE_VAL_FOR_TRAINING:
        [shutil.rmtree(os.path.join(Config.OUTPUT_DIR, d), ignore_errors=True)
        for d in os.listdir(Config.OUTPUT_DIR)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(Config.OUTPUT_DIR, d))]

    if Config.USE_VAL_FOR_TRAINING:
        _ = inspect_val_predictions_mbr_glosstrainer(
            trainer=trainer,
            val_text_ds=val_for_tokenize,
            n_show=int(getattr(Config, "N_SHOW", 20)),
            sample_seed=int(getattr(Config, "VAL_SAMPLE_SEED", 1234)),
            save_csv_path=getattr(Config, "VAL_INSPECT_CSV", None),
            show_pool=bool(getattr(Config, "SHOW_MBR_POOL", True)),
            show_pool_max=int(getattr(Config, "SHOW_MBR_POOL_MAX", 12)),
            show_src_views=bool(getattr(Config, "SHOW_MBR_SOURCES", True)),
            show_src_views_max=int(getattr(Config, "SHOW_MBR_SOURCES_MAX", 10)),
            oracle_pick=bool(getattr(Config, "SHOW_ORACLE_PICK", True)),
            show_oracle_pred=bool(getattr(Config, "SHOW_ORACLE_PRED", True)),
            global_diag=bool(getattr(Config, "SHOW_GLOBAL_DIAG", True)),
            global_diag_eps=float(getattr(Config, "GLOBAL_DIAG_EPS", 1e-6)),
        )


if __name__ == "__main__":
    main()
