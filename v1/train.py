from __future__ import annotations
import argparse

# -------------------------
# stdlib
# -------------------------
import builtins as _bt
import gc
import glob
import hashlib, copy
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
from sklearn.model_selection import GroupShuffleSplit
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
    NB_MODE = "infer"  # "train" or "infer"
    SEED = 4213

    MODEL_NAME = (
        "/kaggle/input/models/mattiaangeli/akkadian-model-mind-the-gap/pytorch/default/8/checkpoint-1548"
    )

    INPUT_DIR = "/kaggle/input/competitions/deep-past-initiative-machine-translation"
    OUTPUT_DIR = (
        "/kaggle/deep-past-initiative-machine-translation/working/"
        "byt5-akkadian-optimized-34x-mbr-mtg-tbm"
    )

    HF_CACHE_DIR = (
        "/kaggle/deep-past-initiative-machine-translation/working/hf-cache"
    )

    DPC_EXTRA_DIR = '/kaggle/input/datasets/mattiaangeli/dpc-extra'
    MANUAL_EXTRA_DIR = '/kaggle/input/datasets/honganzhu/manual'

    PREFIX = "translate Akkadian to English: "

    # ============================================================
    # Lengths / tokenization / generation budget
    # ============================================================
    SRC_MAX_LENGTH = 512
    TGT_MAX_LENGTH = 512
    GEN_MAX_NEW_TOKENS = 512

    NUM_BEAMS = 8
    GEN_LENGTH_PENALTY = 1.2
    GEN_REPETITION_PENALTY = 1.1
    GEN_NO_REPEAT_NGRAM = 0

    VAL_CUTOFF_WORD_THR = 60

    # ============================================================
    # Training (HF Trainer)
    # ============================================================
    RESET_WEIGHTS = True
    RESET_MODE = "medium"

    POSTPROCESS_TARGETS = False
    AGGRESSIVE_POSTPROCESS_TARGETS = False

    BATCH_SIZE = 1
    GRAD_ACCUM = 4
    GRADIENT_CHECKPOINTING = False
    EPOCHS = 5
    VAL_SIZE = 0.1
    NUM_FOLDS = 0   # if >1, use fold split by oare_id instead of VAL_SIZE
    FOLD_INDEX = 0  # which fold is used as validation/test
    LEARNING_RATE = 1e-4
    LABEL_SMOOTHING = 0.1
    WARMUP_RATIO = 0.05

    MAP_BATCH_SIZE = 2048
    TORCH_COMPILE = False
    REPORT_TO = "none"  # set to "wandb" to enable Weights & Biases

    # ============================================================
    # Data paths
    # ============================================================
    TRAIN_CSV_PATH     = f"{INPUT_DIR}/train.csv"
    TEST_CSV_PATH      = f"{INPUT_DIR}/test.csv"
    TRAIN_SENTENCE_CSV = "v1/data/final_train_sentence.csv"  # optional: prebuilt sentence-level train csv

    LEXICON_PATH       = f"{DPC_EXTRA_DIR}/OA_Lexicon_eBL.csv"
    EBL_DICT_PATH      = f"{DPC_EXTRA_DIR}/eBL_Dictionary.csv"
    SENTENCES_PATH     = f"{DPC_EXTRA_DIR}/Sentences_Oare_FirstWord_LinNum.csv"
    LARSEN_LETTERS_PATH = f"{DPC_EXTRA_DIR}/larsen_letters.csv"
    ONOMASTICON_PATH   = f"{DPC_EXTRA_DIR}/onomasticon.csv"
    MTM24_DATASET_PATH = f"{DPC_EXTRA_DIR}/mtm24_transliterated.csv"
    HYBRID_CSV_PATH     = f"{MANUAL_EXTRA_DIR}/aligned_hybrid_keep_highsim_fallback512_FINAL.csv"

    # ============================================================
    # Cleaning / dedupe knobs
    # ============================================================
    INCOMPLETE_RATIO_MAX = 0.55
    INCOMPLETE_KEEP = False

    DEDUPE_NORMALIZE = True
    DEDUPE_RULE = "tgt"
    DEDUPE_KEEP = "longest_src"

    USE_SENTENCE_AUG = True
    SENT_MIN_TOKENS = 3
    SENT_MAX_TOKENS = SRC_MAX_LENGTH

    # ============================================================
    # EXTRA data
    # ============================================================
    USE_EXTRA_IN_TRAIN = True

    # ============================================================
    # TBM (Translation/Template Based Matching) for MBR pools
    # ============================================================
    TBM_ENABLE = True

    TBM_TOPK = 3
    TBM_MIN_SIM = 0.9
    TBM_HARD_SIM = 0.995

    TBM_NGRAM_MIN = 3
    TBM_NGRAM_MAX = 6
    TBM_MAX_FEATURES = 250_000

    TBM_INCLUDE_EXTRA_FALLBACK = True

    # ============================================================
    # K train variants
    # ============================================================
    K_TRAIN_VARIANTS = 32

    # ============================================================
    # PNGLOSS (PN canonicalization + glossary append)
    # ============================================================
    USE_PNGLOSS = True

    GLOSS_MAX_ITEMS = 16
    GLOSS_MAX_APPEND_CHARS = 240
    GLOSS_SEED = SEED + 2

    # ============================================================
    # PROBE (append-before-pngloss)
    # ============================================================
    USE_PROBE_APPEND = True
    PROBE_APPEND_P = 0.15
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
    }

    # ============================================================
    # REAL eval: MBR + gloss sampling
    # ============================================================
    MBR_GLOSS_VARIANTS = 1

    MBR_NUM_BEAMS = 4
    MBR_NUM_BEAM_CANDS = 1

    MBR_NUM_SAMPLE_CANDS = 6
    MBR_TEMPERATURE = 0.75
    MBR_TOP_P = 0.9

    MBR_BATCH_SIZE_INPUTS = 64
    MBR_POOL_CAP = 36

    CKPT_AVG_K = 5
    BEST_METRIC_KEY = "eval_geo_mean"
    SAVE_EVAL_GENERATIONS = True
    EVAL_GENERATIONS_PREFIX = "eval_generations"
    CKPT_AVG_CLEANUP = False
    EVAL_AVG_CHECKPOINT = True
    EVAL_AVG_METRIC_PREFIX = "eval_avg"

    # ============================================================
    # (MTM24) pretrain knobs
    # ============================================================
    PRETRAIN_OUTPUT_DIR = (
        "/kaggle/deep-past-initiative-machine-translation/working/"
        "byt5-akkadian-34x-pretrain"
    )

    PRETRAIN_MAX_STEPS = 10000
    PRETRAIN_LR = 1e-4
    PRETRAIN_WD = 0.01
    PRETRAIN_WARMUP_RATIO = 0.03
    PRETRAIN_SCHED = "cosine"

    PRETRAIN_BS = 32
    PRETRAIN_GRAD_ACCUM = 1
    PRETRAIN_TGT_MAX_LENGTH = 256

    PRETRAIN_SAVE_STEPS = 2000
    PRETRAIN_LOG_STEPS = 50

    PRETRAIN_MIN_SRC_CHARS = 16
    PRETRAIN_MIN_TGT_CHARS = 10

    # MSA consensus polishing (post-MBR)
    MSA_ENABLE = True
    MSA_MIN_POOL = 3            # skip MSA if pool smaller than this
    MSA_MIN_AGREEMENT = 0.35    # fraction of candidates that must agree to override ref
    MSA_TIE_BIAS = 2            # ref token wins ties unless challenger leads by this many votes
    MSA_FUZZY_THR = 0.5         # char-bigram similarity threshold for NW partial credit
    MSA_MAX_INSERT_LEN = 3      # ignore insertion runs longer than this (likely garbage)

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


Config.HF_CACHE_DIR = _ensure_writable_hf_cache_dir(getattr(Config, "HF_CACHE_DIR", ".cache/hf-cache"))

# -------------------------
# Repro
# -------------------------
def seed_everything(seed=Config.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()


# ================================================================
# HF Partial Reset Helper (T5 / ByT5-friendly)
# ================================================================


def _get_encoder(model):
    if hasattr(model, "encoder"):
        return model.encoder
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder
    return None

def _get_decoder(model):
    if hasattr(model, "decoder"):
        return model.decoder
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder
    return None

def _get_blocks(stack) -> Optional[nn.ModuleList]:
    if stack is None:
        return None
    if hasattr(stack, "block"):
        return stack.block
    if hasattr(stack, "layers"):
        return stack.layers
    return None

def _is_t5_like(model) -> bool:
    name = type(model).__name__.lower()
    if "t5" in name:
        return True
    cfg = getattr(model, "config", None)
    if cfg is not None:
        mtype = str(getattr(cfg, "model_type", "")).lower()
        if "t5" in mtype:
            return True
    return False

def _seed_all(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def _t5ish_init_module(module: nn.Module, cfg) -> None:
    d_model = float(getattr(cfg, "d_model", 512))
    init_fac = float(getattr(cfg, "initializer_factor", 1.0))

    if isinstance(module, nn.Linear):
        std = init_fac * (d_model ** -0.5)
        module.weight.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.zero_()
        return

    if isinstance(module, nn.Embedding):
        module.weight.normal_(mean=0.0, std=init_fac)
        if module.padding_idx is not None:
            module.weight[module.padding_idx].zero_()
        return

    cls = module.__class__.__name__.lower()
    if "layernorm" in cls or isinstance(module, nn.LayerNorm):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.fill_(1.0)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.zero_()
        return

def _fresh_copy_and_init(mod: nn.Module, cfg, init_fn) -> nn.Module:
    fresh = copy.deepcopy(mod)
    fresh.apply(lambda m: init_fn(m, cfg))
    return fresh

@torch.no_grad()
def _blend_params(dst: nn.Module, src: nn.Module, alpha: float) -> None:
    sd = dict(dst.named_parameters())
    ss = dict(src.named_parameters())
    for k, p in sd.items():
        q = ss.get(k, None)
        if q is None:
            continue
        if p.data.shape != q.data.shape:
            continue
        if not p.data.is_floating_point():
            continue
        p.data.mul_(1.0 - alpha).add_(q.data, alpha=alpha)

@torch.no_grad()
def _perturb_params(mod: nn.Module, noise_std: float, shrink: float) -> None:
    noise_std = float(noise_std)
    shrink = float(shrink)
    for p in mod.parameters():
        if not p.is_floating_point():
            continue
        if shrink != 1.0:
            p.mul_(shrink)
        if noise_std > 0.0:
            p.add_(torch.randn_like(p) * noise_std)

def _pick_block_parts(block: nn.Module, which: str, is_decoder: bool) -> Sequence[nn.Module]:
    if not hasattr(block, "layer"):
        return [block] if which == "all" else []

    layers = list(block.layer)
    if which == "all":
        return layers
    if which == "ffn":
        return [layers[-1]] if layers else []
    if which == "self_attn":
        return [layers[0]] if layers else []
    if which == "cross_attn":
        if is_decoder and len(layers) >= 3:
            return [layers[1]]
        return []
    return []

ResetWhich = Literal["ffn", "self_attn", "cross_attn", "all"]
ResetMode  = Literal["hallucinations", "light", "medium", "hard"]

@dataclass
class ResetReport:
    mode: str
    enc_blocks: int
    dec_blocks: int
    enc_touched: Tuple[int, ...]
    dec_touched: Tuple[int, ...]
    notes: str


class HFPartialResetHelper:
    def __init__(self, model: nn.Module):
        self.model = model
        self.cfg = getattr(model, "config", None)
        if self.cfg is None:
            raise ValueError("Model has no .config; expected a HF model with config.")
        self._t5ish = _is_t5_like(model)

        self.enc = _get_encoder(model)
        self.dec = _get_decoder(model)
        self.enc_blocks = _get_blocks(self.enc)
        self.dec_blocks = _get_blocks(self.dec)

        if self.enc_blocks is None or self.dec_blocks is None:
            raise ValueError(
                "Could not locate encoder/decoder blocks."
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "model_class": type(self.model).__name__,
            "encoder_blocks": len(self.enc_blocks),
            "decoder_blocks": len(self.dec_blocks),
            "d_model": getattr(self.cfg, "d_model", None),
            "num_heads": getattr(self.cfg, "num_heads", None),
            "d_ff": getattr(self.cfg, "d_ff", None),
            "num_layers": getattr(self.cfg, "num_layers", None),
            "num_decoder_layers": getattr(self.cfg, "num_decoder_layers", None),
        }

    @torch.no_grad()
    def reinit_blocks(self, *, enc_ids: Sequence[int] = (), dec_ids: Sequence[int] = (), which: ResetWhich = "all") -> None:
        for i in enc_ids:
            parts = _pick_block_parts(self.enc_blocks[int(i)], which=which, is_decoder=False)
            for p in parts:
                p.apply(lambda m: _t5ish_init_module(m, self.cfg))
        for i in dec_ids:
            parts = _pick_block_parts(self.dec_blocks[int(i)], which=which, is_decoder=True)
            for p in parts:
                p.apply(lambda m: _t5ish_init_module(m, self.cfg))

    @torch.no_grad()
    def soft_reset_blocks(self, *, enc_ids: Sequence[int] = (), dec_ids: Sequence[int] = (), which: ResetWhich = "all", alpha: float = 0.3) -> None:
        a = float(alpha)
        if not (0.0 <= a <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        for i in enc_ids:
            block = self.enc_blocks[int(i)]
            for part in _pick_block_parts(block, which=which, is_decoder=False):
                fresh = _fresh_copy_and_init(part, self.cfg, _t5ish_init_module)
                _blend_params(part, fresh, alpha=a)
        for i in dec_ids:
            block = self.dec_blocks[int(i)]
            for part in _pick_block_parts(block, which=which, is_decoder=True):
                fresh = _fresh_copy_and_init(part, self.cfg, _t5ish_init_module)
                _blend_params(part, fresh, alpha=a)

    @torch.no_grad()
    def perturb_blocks(self, *, enc_ids: Sequence[int] = (), dec_ids: Sequence[int] = (), which: ResetWhich = "all", noise_std: float = 5e-4, shrink: float = 0.995, seed: int = 1234) -> None:
        _seed_all(int(seed))
        for i in enc_ids:
            block = self.enc_blocks[int(i)]
            for part in _pick_block_parts(block, which=which, is_decoder=False):
                _perturb_params(part, noise_std=float(noise_std), shrink=float(shrink))
        for i in dec_ids:
            block = self.dec_blocks[int(i)]
            for part in _pick_block_parts(block, which=which, is_decoder=True):
                _perturb_params(part, noise_std=float(noise_std), shrink=float(shrink))

    @torch.no_grad()
    def reset_trainable_only(self, *, noise_std: float = 1e-3, seed: int = 1234) -> None:
        _seed_all(int(seed))
        for _, p in self.model.named_parameters():
            if p.requires_grad and p.is_floating_point():
                p.normal_(mean=0.0, std=float(noise_std))

    def apply(self, mode: ResetMode = "hallucinations", *, seed: int = 1234) -> ResetReport:
        n_enc = len(self.enc_blocks)
        n_dec = len(self.dec_blocks)

        def tail_ids(n, k):
            k = max(0, min(int(k), int(n)))
            return tuple(range(n - k, n)) if k > 0 else tuple()

        def head_ids(n, k):
            k = max(0, min(int(k), int(n)))
            return tuple(range(0, k)) if k > 0 else tuple()

        def all_ids(n):
            return tuple(range(0, int(n)))

        if mode == "hallucinations":
            dec_ids = tail_ids(n_dec, 2)
            self.soft_reset_blocks(dec_ids=dec_ids, which="ffn", alpha=0.3)
            self.perturb_blocks(dec_ids=dec_ids, which="ffn", noise_std=5e-4, shrink=0.995, seed=seed)
            self.perturb_blocks(dec_ids=tail_ids(n_dec, 1), which="self_attn", noise_std=2e-4, shrink=0.999, seed=seed + 17)
            notes = "hallucinations: soft-reset last 2 decoder FFNs + tiny perturb; encoder untouched."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=tuple(), dec_touched=dec_ids, notes=notes)

        if mode == "light":
            dec_ids = tail_ids(n_dec, 1)
            self.soft_reset_blocks(dec_ids=dec_ids, which="ffn", alpha=0.10)
            notes = "light: soft-reset last decoder FFN only."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=tuple(), dec_touched=dec_ids, notes=notes)

        if mode == "medium":
            dec_ids = tail_ids(n_dec, 2)
            self.soft_reset_blocks(dec_ids=dec_ids, which="all", alpha=0.25)
            self.perturb_blocks(dec_ids=dec_ids, which="all", noise_std=3e-4, shrink=0.997, seed=seed)
            notes = "medium: soft-reset last 2 decoder blocks (all parts) + small perturb."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=tuple(), dec_touched=dec_ids, notes=notes)

        if mode == "hard":
            dec_ids = tail_ids(n_dec, 3)
            self.reinit_blocks(dec_ids=dec_ids, which="all")
            notes = "hard: hard reinit last 3 decoder blocks."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=tuple(), dec_touched=dec_ids, notes=notes)

        if mode == "encoder_light":
            enc_ids = head_ids(n_enc, 1)
            self.soft_reset_blocks(enc_ids=enc_ids, which="ffn", alpha=0.20)
            notes = "encoder_light: soft-reset encoder[0] FFN."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=tuple(), notes=notes)

        if mode == "encoder_medium":
            enc_ids = tail_ids(n_enc, 2)
            self.soft_reset_blocks(enc_ids=enc_ids, which="all", alpha=0.25)
            self.perturb_blocks(enc_ids=enc_ids, which="all", noise_std=3e-4, shrink=0.997, seed=seed)
            notes = "encoder_medium: soft-reset last 2 encoder blocks (all parts) + small perturb."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=tuple(), notes=notes)

        if mode == "encoder_hard":
            enc_ids = tail_ids(n_enc, 3)
            self.reinit_blocks(enc_ids=enc_ids, which="all")
            notes = "encoder_hard: hard reinit last 3 encoder blocks."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=tuple(), notes=notes)

        if mode == "encoder_attn":
            enc_ids = tail_ids(n_enc, 2)
            self.soft_reset_blocks(enc_ids=enc_ids, which="self_attn", alpha=0.25)
            self.perturb_blocks(enc_ids=enc_ids, which="self_attn", noise_std=2e-4, shrink=0.999, seed=seed + 7)
            notes = "encoder_attn: soft-reset last 2 encoder self-attn + tiny perturb."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=tuple(), notes=notes)

        if mode == "encoder_ffn":
            enc_ids = tail_ids(n_enc, 3)
            self.soft_reset_blocks(enc_ids=enc_ids, which="ffn", alpha=0.30)
            self.perturb_blocks(enc_ids=enc_ids, which="ffn", noise_std=4e-4, shrink=0.996, seed=seed + 9)
            notes = "encoder_ffn: soft-reset last 3 encoder FFNs + small FFN perturb."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=tuple(), notes=notes)

        if mode == "sandwich":
            enc_ids = head_ids(n_enc, 1)
            dec_ids = tail_ids(n_dec, 2)
            self.soft_reset_blocks(enc_ids=enc_ids, which="ffn", alpha=0.18)
            self.soft_reset_blocks(dec_ids=dec_ids, which="ffn", alpha=0.25)
            self.perturb_blocks(dec_ids=tail_ids(n_dec, 1), which="self_attn", noise_std=2e-4, shrink=0.999, seed=seed + 19)
            notes = "sandwich: encoder[0] FFN + last 2 decoder FFNs (+ tiny last self-attn perturb)."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=dec_ids, notes=notes)

        if mode == "alignment":
            dec_ids = head_ids(n_dec, 2)
            self.soft_reset_blocks(dec_ids=dec_ids, which="cross_attn", alpha=0.25)
            self.perturb_blocks(dec_ids=dec_ids, which="cross_attn", noise_std=3e-4, shrink=0.997, seed=seed + 11)
            notes = "alignment: soft-reset early decoder cross-attn (0..1) + small perturb."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=tuple(), dec_touched=dec_ids, notes=notes)

        if mode == "all_soft":
            enc_ids = all_ids(n_enc)
            dec_ids = all_ids(n_dec)
            self.soft_reset_blocks(enc_ids=enc_ids, dec_ids=dec_ids, which="all", alpha=0.10)
            self.perturb_blocks(enc_ids=enc_ids, dec_ids=dec_ids, which="all", noise_std=1e-4, shrink=0.999, seed=seed)
            notes = "all_soft: soft-reset ALL encoder+decoder blocks (all parts) + tiny global perturb."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=dec_ids, notes=notes)

        if mode == "nuke_all":
            _seed_all(int(seed))
            with torch.no_grad():
                self.model.apply(lambda m: _t5ish_init_module(m, self.cfg))
            enc_ids = tuple(range(n_enc))
            dec_ids = tuple(range(n_dec))
            notes = "nuke_all: hard reinit ENTIRE model via t5ish init (all modules)."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=dec_ids, notes=notes)

        if mode == "all_hard_tail":
            enc_ids = tail_ids(n_enc, 3)
            dec_ids = tail_ids(n_dec, 3)
            self.reinit_blocks(enc_ids=enc_ids, dec_ids=dec_ids, which="all")
            notes = "all_hard_tail: hard reinit last 3 encoder + last 3 decoder blocks."
            return ResetReport(mode=mode, enc_blocks=n_enc, dec_blocks=n_dec, enc_touched=enc_ids, dec_touched=dec_ids, notes=notes)

        raise ValueError(f"Unknown mode: {mode}")


# ================================================================
# Augmentation and canonicalization utilities
# ================================================================

_SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_H_HAT_TRANS = str.maketrans("ḫḪ", "hH")


def _key(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, float):
        try:
            if math.isnan(s):
                return ""
        except Exception:
            pass
    s = s if isinstance(s, str) else str(s)
    if s.lower() == "nan":
        return ""
    return s.translate(_SUBSCRIPT_TRANS).translate(_H_HAT_TRANS).strip()


def _split_spellings(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = re.split(r"[;,\|/]+", cell)
    return [p.strip() for p in parts if p.strip()]


# -------------------------
# Helpers (robust, short)
# -------------------------
_QUOTED = re.compile(r'"([^"]{1,80})"')
_ROMAN  = re.compile(r"\s+[IVX]+$")

def _first_quoted_gloss(defn: str) -> Optional[str]:
    if defn is None:
        return None
    s = str(defn).strip()
    if not s or s.lower() == "nan":
        return None
    m = _QUOTED.search(s)
    if m:
        g = m.group(1).strip()
        return g if g else None
    return None

def _lemma_part(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = _ROMAN.sub("", s).strip()
    s = re.split(r"[\s/;]", s, maxsplit=1)[0].strip()
    return s if s else None


_PUNCT_STRIP = "[](){}<>.,;:!?\"'""„`´"
_SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_DET_RE = re.compile(r"(^|\-)\((d|m|f)\)", re.I)
_ELLIPSIS_RE_AUG = re.compile(r"(…|\.\.\.)")

_AKK_SUFFIXES = [
    "šu-nu", "šunu", "šunū",
    "šu", "ša", "ši", "šū",
    "ī",
]

def _norm_form(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("-", "-").replace("–", "-").replace("—", "-")
    s = s.strip().strip(_PUNCT_STRIP)
    s = _DET_RE.sub(r"\1", s)
    s = s.replace("[", "").replace("]", "")
    s = s.translate(_SUB_MAP)
    s = _ELLIPSIS_RE_AUG.sub("", s)
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

    cands = []
    seen = set()
    def add(x):
        x = "" if x is None else str(x).strip()
        if not x:
            return
        if x not in seen:
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

def _stable_u32(s: str) -> int:
    b = ("" if s is None else str(s)).encode("utf-8", errors="ignore")
    return zlib.crc32(b) & 0xFFFFFFFF

def _is_junk_surface(surface: str) -> bool:
    s = "" if surface is None else str(surface).strip()
    if not s:
        return True
    if s in {"<gap>", "<big_gap>", "x", "x.", "…", "...", "[...]", "…]", "[…"}:
        return True
    if re.fullmatch(r"\d+([.:]\d+)?", s):
        return True
    core = s.strip(_PUNCT_STRIP)
    if not core:
        return True
    return False

class GlossAugmenter:
    def __init__(
        self,
        oa_lexicon_path: str,
        ebl_dict_path: str,
        *,
        train_texts: Optional[list[str]] = None,
        idf_cap: float = 3.5,
        rare_df_floor: int = 3,
        df1_penalty: float = 0.65,
        base_weight: float = 0.6,
    ):
        lex = pd.read_csv(oa_lexicon_path)
        dic = pd.read_csv(ebl_dict_path)

        lemma2gloss = {}
        for w, d in zip(dic["word"].astype(str), dic["definition"].astype(str)):
            lemma = _lemma_part(w)
            gloss = _first_quoted_gloss(d)
            if lemma and gloss and lemma not in lemma2gloss:
                lemma2gloss[lemma] = gloss

        self.form2lex = dict(zip(lex["form"].astype(str), lex["lexeme"].astype(str)))
        self.lex2gloss = lemma2gloss

        self._df: Dict[str, int] = {}
        self._N: int = 0

        if train_texts is not None and len(train_texts) > 0:
            from collections import Counter
            dfc = Counter()
            N = 0
            for s in train_texts:
                s = "" if s is None else str(s)
                toks = s.split()
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

    def _weight_for_surface(self, surface: str) -> float:
        if _is_junk_surface(surface):
            return 0.0
        s = _norm_form(surface)
        df = int(self._df.get(s, 0))
        df_eff = max(df, int(self._rare_df_floor))
        idf = math.log((self._N + 1.0) / (df_eff + 1.0))
        idf = max(0.0, min(float(idf), float(self._idf_cap)))
        w = float(self._base_weight) + idf
        if df <= 1:
            w *= float(self._df1_penalty)
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

        if "</s>" in src_text:
            src_text = src_text.replace("</s>", "<eos>")

        if (not src_text.strip()) or (int(max_items) <= 0) or (int(max_append_chars) <= 0):
            return src_text

        toks = src_text.split()
        candidates = []
        seen = set()

        for pos, t in enumerate(toks):
            if _is_junk_surface(t):
                continue
            lexeme = None
            surface = None
            for cand in _candidates_for_form(t):
                lexeme = self.form2lex.get(cand)
                if lexeme:
                    surface = cand
                    break
            if not lexeme:
                continue
            lemma = _lemma_part(lexeme)
            if not lemma:
                continue
            g = self.lex2gloss.get(lemma)
            if not g:
                continue
            key = (surface, lemma, g)
            if key in seen:
                continue
            seen.add(key)
            w = self._weight_for_surface(surface)
            if w <= 0:
                continue
            candidates.append((pos, surface, g, w))

        if not candidates:
            return src_text

        ex = int(example_id) if example_id is not None else 0
        mix = (_stable_u32(src_text) ^ int(seed) ^ (int(epoch) * 1000003) ^ (ex * 9176)) & 0xFFFFFFFF
        rng = random.Random(mix)

        k = min(int(max_items), len(candidates))

        keys = []
        for j, (_, _, _, w) in enumerate(candidates):
            u = max(1e-12, rng.random())
            key = -math.log(u) / max(1e-6, float(w))
            keys.append((key, j))
        keys.sort(key=lambda x: x[0])
        picked = [candidates[j] for _, j in keys[:k]]

        if keep_order:
            picked.sort(key=lambda x: x[0])

        parts = []
        used = 0
        for _, surface, gloss, _w in picked:
            part = f"{surface}={gloss}"
            add_len = len(part) + (3 if parts else 0)
            if used + add_len > int(max_append_chars):
                break
            parts.append(part)
            used += add_len

        if not parts:
            return src_text

        gloss_str = " ; ".join(parts)
        return src_text + " <extra_id_0> GLOSSARY: " + gloss_str


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
        if use_norm:
            lex["canon"] = lex["norm"].astype(str)
        else:
            lex["canon"] = lex["lexeme"].astype(str)

        lex["canon_len"] = lex["canon"].str.len()
        lex = lex.sort_values(["form", "canon_len"]).drop_duplicates("form", keep="first")

        pn_gn_map: Dict[str, str] = {}
        for form, canon in zip(lex["form"].tolist(), lex["canon"].tolist()):
            pn_gn_map[_key(form)] = canon

        if "Alt_lex" in lex.columns:
            alt_src = pd.read_csv(lexicon_path)
            alt_src = alt_src[alt_src["type"].isin(["PN", "GN"])].copy()
            alt_src["form"] = alt_src["form"].astype(str)
            alt_src["canon"] = alt_src["norm"].astype(str) if use_norm else alt_src["lexeme"].astype(str)
            for form, canon, alt in zip(alt_src["form"], alt_src["canon"], alt_src.get("Alt_lex", pd.Series([None]*len(alt_src)))):
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
        toks = text.split()
        out = []
        for t in toks:
            kt = _key(t)
            if kt in self.pn_gn_map:
                out.append(self.pn_gn_map[kt])
            elif kt in self.ono_map:
                out.append(self.ono_map[kt])
            else:
                out.append(t)
        return " ".join(out)


_NUM_RE = re.compile(r"^\d+(\.\d+)?$")

def _is_num(tok: str) -> bool:
    return bool(_NUM_RE.match(tok))

def _norm(tok: str) -> str:
    tok = "" if tok is None else str(tok)
    tok = tok.strip().lower()
    tok = unicodedata.normalize("NFKC", tok)
    tok = tok.strip(".,;:()[]{}")
    if tok == "...":
        tok = "…"
    return tok

def build_sentence_pairs(
    train_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    allowed_text_ids: Optional[Set[str]] = None,
    min_tokens: int = 3,
    max_tokens: int = 200,
    spelling_window: int = 3,
    max_numeric_prefix: int = 2,
) -> pd.DataFrame:
    train_map = {
        str(r.oare_id): str(r.transliteration)
        for r in train_df.itertuples(index=False)
    }

    s = sentences_df.copy()
    s = s.dropna(subset=["text_uuid", "first_word_number", "translation", "first_word_spelling"])
    s["text_uuid"] = s["text_uuid"].astype(str)
    s["first_word_number"] = pd.to_numeric(s["first_word_number"], errors="coerce")
    s = s.dropna(subset=["first_word_number"])
    s["first_word_number"] = s["first_word_number"].astype(int)

    if allowed_text_ids is not None:
        allowed = {str(x) for x in allowed_text_ids}
        s = s[s["text_uuid"].isin(allowed)]

    rows = []

    for text_uuid, g in s.groupby("text_uuid", sort=False):
        if text_uuid not in train_map:
            continue
        toks = train_map[text_uuid].split()
        if not toks:
            continue
        g = g.sort_values("first_word_number")
        starts_1b = g["first_word_number"].tolist()
        spellings = g["first_word_spelling"].tolist()
        translations = g["translation"].astype(str).tolist()
        sent_ids = g.get("sentence_uuid", pd.Series([None] * len(g))).astype(str).tolist()

        start_idxs = []
        meta_idxs = []

        for i, start_1based in enumerate(starts_1b):
            idx0 = start_1based - 1
            if idx0 < 0 or idx0 >= len(toks):
                continue
            want = _norm(spellings[i])
            if not want:
                continue
            candidates = []
            lo = max(0, idx0 - spelling_window)
            hi = min(len(toks) - 1, idx0 + spelling_window)
            for j in range(lo, hi + 1):
                if _norm(toks[j]) == want:
                    candidates.append(j)
                elif _is_num(toks[j]) and (j + 1 < len(toks)) and _norm(toks[j + 1]) == want:
                    candidates.append(j)
            if not candidates:
                continue
            candidates = sorted(candidates, key=lambda j: (abs(j - idx0), j))
            if len(candidates) >= 2 and abs(candidates[0] - idx0) == abs(candidates[1] - idx0):
                continue
            start_idx = candidates[0]

            pulled = 0
            while start_idx > 0 and pulled < max_numeric_prefix:
                prev_tok = toks[start_idx - 1]
                if not _is_num(prev_tok):
                    break
                if _norm(toks[start_idx]) == want or (
                    _is_num(toks[start_idx]) and start_idx + 1 < len(toks) and _norm(toks[start_idx + 1]) == want
                ):
                    if start_idxs and (start_idx - 1) <= start_idxs[-1]:
                        break
                    start_idx -= 1
                    pulled += 1
                else:
                    break

            if start_idxs and start_idx <= start_idxs[-1]:
                continue
            start_idxs.append(start_idx)
            meta_idxs.append(i)

        if not start_idxs:
            continue

        for k, start in enumerate(start_idxs):
            end = start_idxs[k + 1] if (k + 1 < len(start_idxs)) else len(toks)
            if end <= start:
                continue
            span = toks[start:end]
            if not (min_tokens <= len(span) <= max_tokens):
                continue
            tgt = translations[meta_idxs[k]].strip()
            if not tgt:
                continue
            sid = sent_ids[meta_idxs[k]]
            rows.append(
                dict(
                    oare_id=text_uuid,
                    pair_id=f"{text_uuid}::sent::{sid if sid != 'nan' else meta_idxs[k]}",
                    transliteration=" ".join(span),
                    translation=tgt,
                    is_sentence=True,
                )
            )

    return pd.DataFrame(rows)


# ================================================================
# Preprocessing utilities
# ================================================================

_ALLOWED_FRACS = [
    (1.0 / 6.0, "0.16666"),
    (1.0 / 4.0, "0.25"),
    (1.0 / 3.0, "0.33333"),
    (1.0 / 2.0, "0.5"),
    (2.0 / 3.0, "0.66666"),
    (3.0 / 4.0, "0.75"),
    (5.0 / 6.0, "0.83333"),
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
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")

def normalize_float_artifacts(text: str) -> str:
    s = "" if text is None else str(text)
    def repl(m):
        raw = m.group(1)
        try:
            return _canon_decimal_str(float(raw))
        except Exception:
            return raw
    return _FLOAT_ARTIFACT_RE.sub(repl, s)


# Gaps
_TAG_GAP_RE      = re.compile(r"<\s*gap\s*>", re.I)
_TAG_BIGGAP_RE   = re.compile(r"<\s*big[\s_\-]*gap\s*>", re.I)
_BARE_BIGGAP_RE  = re.compile(r"\bbig[\s_\-]*gap\b", re.I)

_ELLIPSIS_RE     = re.compile(r"(?:\.{3,}|…+|……|\[\.+\])")
_BRACKET_X_RE    = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))", re.I)
_XTOKEN_RUN_RE   = re.compile(r"\bx(?:\s+x)+\b", re.I)
_XRUN_RE         = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_XTOK_RE         = re.compile(r"(?<!\w)x(?!\w)", re.I)

_WS_RE           = re.compile(r"\s+")

def normalize_gaps(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = _TAG_BIGGAP_RE.sub("<gap>", t)
    t = _TAG_GAP_RE.sub("<gap>", t)
    t = _BARE_BIGGAP_RE.sub("<gap>", t)
    t = _XTOKEN_RUN_RE.sub("<gap>", t)
    t = _ELLIPSIS_RE.sub("<gap>", t)
    t = _BRACKET_X_RE.sub("<gap>", t)
    t = _XRUN_RE.sub("<gap>", t)
    t = _XTOK_RE.sub("<gap>", t)
    return t

def collapse_gap_runs_tokens(tokens: List[str], mode: str) -> List[str]:
    mode = (mode or "none").lower().strip()
    if mode in ("none", ""):
        return tokens
    if mode in ("big_only", "any2big"):
        mode = "single"
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] == "<gap>":
            j = i
            while j < n and tokens[j] == "<gap>":
                j += 1
            out.append("<gap>")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out

def space_gap_token(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    t = re.sub(r"(?<![\s\-])<gap>", " <gap>", t)
    t = re.sub(r"<gap>(?![\s\-])", "<gap> ", t)
    return t


# ASCII/Oracc/ATF -> host diacritics
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

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


# Transliteration char cleanup
TRANSLIT_SPECIAL_CHAR_MAP = {
    "ḫ": "h", "Ḫ": "H",
    "ʾ": "",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "—": "-", "–": "-",
}
TRANSLIT_SPECIAL_SEQ_MAP = {"mₓ": "m", "zₓ": "z"}
_SUB_X = "ₓ"
_CHAR_TRANS = str.maketrans(TRANSLIT_SPECIAL_CHAR_MAP)

_DET_PARENS_RE = re.compile(r"\(([A-Za-z0-9]{1,4})\)")


def normalize_external_transliteration(text: str, *, kb_to_silver: bool = True) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = ascii_to_diacritics(s)
    s = _DET_PARENS_RE.sub(r"{\1}", s)
    s = normalize_gaps(s)
    for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
        s = s.replace(k, v)
    s = s.translate(_CHAR_TRANS).replace(_SUB_X, "")
    s = normalize_float_artifacts(s)
    if kb_to_silver:
        s = re.sub(r"\bKB\b", "KÙ.BABBAR", s)
    s = space_gap_token(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# External-rich translation normalizer
_PN_RE = re.compile(r"\bPN\b")
_QUOTES_RE = re.compile(r'["""'']')
_SOFT_GRAM_PARENS_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)

def normalize_external_translation(text: str, *, gap_collapse: str = "single") -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = normalize_gaps(s)
    s = _PN_RE.sub("<gap>", s)
    s = _SOFT_GRAM_PARENS_RE.sub(" ", s)
    s = _QUOTES_RE.sub("", s)
    s = normalize_float_artifacts(s)
    s = space_gap_token(s)
    if gap_collapse and gap_collapse.lower().strip() not in ("none", ""):
        toks = collapse_gap_runs_tokens(s.split(), gap_collapse)
        s = " ".join(toks)
    s = s.replace("<gap>", " <gap> ")
    s = _WS_RE.sub(" ", s).strip()
    return s


class OptimizedPreprocessor:
    def __init__(self):
        self._char_trans = _CHAR_TRANS

    def preprocess_input_text(self, text: str) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        s = str(text)
        s = ascii_to_diacritics(s)
        s = _DET_PARENS_RE.sub(r"{\1}", s)
        s = normalize_gaps(s)
        for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
            s = s.replace(k, v)
        s = s.translate(self._char_trans).replace(_SUB_X, "")
        s = normalize_float_artifacts(s)
        s = _WS_RE.sub(" ", s).strip()
        return s

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        ser = pd.Series(texts).fillna("").astype(str)
        ser = ser.apply(ascii_to_diacritics)
        ser = ser.str.replace(_DET_PARENS_RE, r"{\1}", regex=True)
        ser = ser.apply(normalize_gaps)
        for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
            ser = ser.str.replace(k, v, regex=False)
        ser = ser.str.translate(self._char_trans)
        ser = ser.str.replace(_SUB_X, "", regex=False)
        ser = ser.str.replace(_FLOAT_ARTIFACT_RE, lambda m: _canon_decimal_str(float(m.group(1))), regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()


class VectorizedPostprocessor:
    def __init__(
        self,
        aggressive: bool = True,
        infer_defaults: bool = True,
        gap_collapse: Optional[str] = None,
        empty_fallback: str = "",
        fix_repeats: bool = False,
    ):
        self.aggressive = bool(aggressive)
        self.infer_defaults = bool(infer_defaults)
        self.fix_repeats = bool(fix_repeats)

        if gap_collapse is None:
            self.gap_collapse = "single" if self.infer_defaults else "none"
        else:
            self.gap_collapse = str(gap_collapse).lower().strip()

        self.empty_fallback = "" if empty_fallback is None else str(empty_fallback)

        self._pn_re = _PN_RE
        self._soft_gram_parens_re = _SOFT_GRAM_PARENS_RE
        self._quotes_re = _QUOTES_RE

        self.forbidden_chars = "()""''—–<>⌈⌋⌊+ʾ"
        self.forbidden_trans = str.maketrans("", "", self.forbidden_chars)

        self.patterns = {
            "gap_legacy": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap_legacy": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
            "whitespace": _WS_RE,
            "punct_space": re.compile(r"\s+([.,:;])"),
            "repeated_punct": re.compile(r"([.,:;])\1+"),
        }

        self._month_roman_re = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.IGNORECASE)
        self._roman2int = {
            "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
            "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
        }

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
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True).str.strip()

        if self.aggressive:
            s = s.str.replace(self.patterns["gap_legacy"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap_legacy"], "<gap>", regex=True)
            s = s.str.replace(self._soft_gram_parens_re, " ", regex=True)
            s = s.str.replace(self._quotes_re, "", regex=True)

            if self.gap_collapse not in ("none", ""):
                s = s.apply(self._collapse_gaps_str)

            s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
            s = s.str.translate(self.forbidden_trans)
            s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
            s = s.str.replace(_FLOAT_ARTIFACT_RE, lambda m: _canon_decimal_str(float(m.group(1))), regex=True)
            s = s.str.replace(self._month_roman_re, self._month_repl, regex=True)

            if self.fix_repeats:
                s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)
                for n in range(4, 1, -1):
                    pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                    s = s.str.replace(pattern, r"\1", regex=True)

            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True).str.strip()

        if self.empty_fallback:
            s = s.replace("", self.empty_fallback)

        return s.tolist()


# ================================================================
# Extra data cleaning utilities
# ================================================================

_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_BAD_EMPTY = {"", "nan", "none", "null", "na", "n/a", "<na>"}

def _clean_text(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _ZW_RE.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = unicodedata.normalize("NFKC", s).strip()
    s = _WS_RE.sub(" ", s)
    if s.lower() in _BAD_EMPTY:
        return ""
    return s

def _norm_key(s: str) -> str:
    s = _clean_text(s).lower()
    return s.strip(" .,:;\"'""''()[]{}")

def _pick_col(columns, candidates):
    cols_low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_low:
            return cols_low[cand.lower()]
    return None

def _infer_schema(df: pd.DataFrame):
    cols = list(df.columns)
    src_col = _pick_col(cols, ["new_transliteration_sentence","new_transliteration","transliteration", "translit", "source", "src", "akkadian"])
    tgt_col = _pick_col(cols, ["new_translation_sentence","new_translation","translation", "english", "target", "tgt", "en"])
    id_col  = _pick_col(cols, ["oare_id", "text_uuid", "id", "uuid", "text_id"])
    if src_col is None or tgt_col is None:
        raise ValueError(f"Could not infer src/tgt columns. Columns: {cols}")
    return src_col, tgt_col, id_col


def load_and_sanitize_parallel(
    paths: str | Path | Sequence[str | Path],
    *,
    train_df: Optional[pd.DataFrame] = None,
    drop_if_in_train: bool = True,
    in_train_match: str = "either",
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

        keep_cols = [c for c in [id_col, src_col, tgt_col] if c is not None]
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

        out = out[
            out[out_src_col].astype(str).str.strip().ne("") &
            out[out_tgt_col].astype(str).str.strip().ne("")
        ].copy()

        if drop_same_src_tgt:
            out = out[out[out_src_col] != out[out_tgt_col]].copy()

        out["is_extra"] = True
        if add_source_col:
            out["source_file"] = p.name

        frames.append(out)

    extra = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=[out_id_col, out_src_col, out_tgt_col, "is_extra"] + (["source_file"] if add_source_col else []))
    )

    if len(extra) == 0:
        cols = [out_id_col, out_src_col, out_tgt_col, "is_extra"]
        if add_source_col:
            cols.append("source_file")
        return extra[cols]

    raw_src = extra[out_src_col].astype(str).fillna("").str.strip()
    raw_tgt = extra[out_tgt_col].astype(str).fillna("").str.strip()
    pre_src = pd.Series(pre.preprocess_batch(raw_src.tolist()), index=extra.index)
    post_tgt = pd.Series(post.postprocess_batch(raw_tgt.tolist()), index=extra.index)

    extra["_pre_src"] = pre_src
    extra["_post_tgt"] = post_tgt

    if dedupe_on:
        extra["_tgt_len"] = raw_tgt.str.len().fillna(0).astype(int)
        extra = extra.sort_values("_tgt_len", ascending=False).drop_duplicates(list(dedupe_on), keep="first")
        extra = extra.drop(columns=["_tgt_len"])

    if drop_if_in_train and train_df is not None and len(train_df) > 0:
        tr_raw_src = train_df[out_src_col].astype(str).map(_norm_key)
        tr_raw_tgt = train_df[out_tgt_col].astype(str).map(_norm_key)
        tr_pre_src = pd.Series(pre.preprocess_batch(train_df[out_src_col].astype(str).tolist())).map(_norm_key)
        tr_post_tgt = pd.Series(post.postprocess_batch(train_df[out_tgt_col].astype(str).tolist())).map(_norm_key)

        ex_raw_src = extra[out_src_col].astype(str).map(_norm_key)
        ex_raw_tgt = extra[out_tgt_col].astype(str).map(_norm_key)
        ex_pre_src = extra["_pre_src"].astype(str).map(_norm_key)
        ex_post_tgt = extra["_post_tgt"].astype(str).map(_norm_key)

        if in_train_match == "pair":
            train_pairs = set(zip(tr_raw_src.tolist(), tr_raw_tgt.tolist()))
            train_pairs |= set(zip(tr_pre_src.tolist(), tr_raw_tgt.tolist()))
            train_pairs |= set(zip(tr_raw_src.tolist(), tr_post_tgt.tolist()))
            train_pairs |= set(zip(tr_pre_src.tolist(), tr_post_tgt.tolist()))

            ex_pairs = list(zip(ex_raw_src.tolist(), ex_raw_tgt.tolist()))
            ex_pairs2 = list(zip(ex_pre_src.tolist(), ex_raw_tgt.tolist()))
            ex_pairs3 = list(zip(ex_raw_src.tolist(), ex_post_tgt.tolist()))
            ex_pairs4 = list(zip(ex_pre_src.tolist(), ex_post_tgt.tolist()))

            in_train = [
                (a in train_pairs) or (b in train_pairs) or (c in train_pairs) or (d in train_pairs)
                for a, b, c, d in zip(ex_pairs, ex_pairs2, ex_pairs3, ex_pairs4)
            ]
            in_train = pd.Series(in_train, index=extra.index)

        elif in_train_match == "src":
            train_src = set(tr_raw_src.tolist()) | set(tr_pre_src.tolist())
            in_train = ex_raw_src.isin(train_src) | ex_pre_src.isin(train_src)

        elif in_train_match == "either":
            train_src = set(tr_raw_src.tolist()) | set(tr_pre_src.tolist())
            train_tgt = set(tr_raw_tgt.tolist()) | set(tr_post_tgt.tolist())
            in_train = (
                ex_raw_src.isin(train_src) | ex_pre_src.isin(train_src) |
                ex_raw_tgt.isin(train_tgt) | ex_post_tgt.isin(train_tgt)
            )
        else:
            raise ValueError("in_train_match must be one of: 'pair', 'src', 'either'")

        extra = extra.loc[~in_train].reset_index(drop=True)

    if drop_incomplete and len(extra) > 0:
        if "flag_incomplete" not in globals():
            raise NameError("drop_incomplete=True but flag_incomplete(...) is not defined.")
        kwargs = incomplete_kwargs or {}
        bad = flag_incomplete(extra, **kwargs)
        extra = extra.loc[~bad].reset_index(drop=True)

    for c in ("_pre_src", "_post_tgt"):
        if c in extra.columns:
            extra = extra.drop(columns=[c])

    cols = [out_id_col, out_src_col, out_tgt_col, "is_extra"]
    if add_source_col:
        cols.append("source_file")
    return extra[cols]

def _norm_text(s: str, *, lowercase: bool = True) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _ZW_RE.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = unicodedata.normalize("NFKC", s).strip()
    s = _WS_RE.sub(" ", s)
    if lowercase:
        s = s.lower()
    return s

def drop_duplicates_hf(
    ds: Union[Dataset, DatasetDict],
    *,
    src_col: str = "transliteration",
    tgt_col: str = "translation",
    rule: Literal["tgt", "pair"] = "tgt",
    keep: Literal["first", "last", "longest_src"] = "longest_src",
    normalize: bool = True,
    lowercase: bool = True,
    report: bool = True,
) -> Union[Dataset, DatasetDict]:
    if isinstance(ds, DatasetDict):
        return DatasetDict({
            split: drop_duplicates_hf(
                d, src_col=src_col, tgt_col=tgt_col,
                rule=rule, keep=keep,
                normalize=normalize, lowercase=lowercase, report=report,
            )
            for split, d in ds.items()
        })

    if src_col not in ds.column_names:
        raise ValueError(f"src_col='{src_col}' not in columns: {ds.column_names}")
    if tgt_col not in ds.column_names:
        raise ValueError(f"tgt_col='{tgt_col}' not in columns: {ds.column_names}")

    n0 = ds.num_rows
    src_list = ds[src_col]
    tgt_list = ds[tgt_col]

    chosen_idx = {}
    best_score = {}

    def norm_src(i):
        s = src_list[i]
        return _norm_text(s, lowercase=lowercase) if normalize else ("" if s is None else str(s))

    def norm_tgt(i):
        t = tgt_list[i]
        return _norm_text(t, lowercase=lowercase) if normalize else ("" if t is None else str(t))

    for i in range(n0):
        s = norm_src(i)
        t = norm_tgt(i)
        key = t if rule == "tgt" else (s, t)

        if key not in chosen_idx:
            chosen_idx[key] = i
            if keep == "longest_src":
                best_score[key] = len(src_list[i] or "")
            continue

        if keep == "first":
            continue
        if keep == "last":
            chosen_idx[key] = i
            continue

        score = len(src_list[i] or "")
        if score > best_score[key]:
            chosen_idx[key] = i
            best_score[key] = score

    keep_indices = sorted(chosen_idx.values())
    ds2 = ds.select(keep_indices)

    if report:
        n1 = ds2.num_rows
        print(f"[drop_duplicates_hf] rule={rule} keep={keep} normalize={normalize} :: {n0} -> {n1} (dropped {n0-n1})")

    return ds2

def _is_hf_dataset(x) -> bool:
    return (Dataset is not None) and isinstance(x, Dataset)

def _is_hf_datasetdict(x) -> bool:
    return (DatasetDict is not None) and isinstance(x, DatasetDict)


# Incomplete detection (pandas)
def _make_incomplete_reasons_df(
    df: pd.DataFrame,
    *,
    ratio_max: float = 0.50,
) -> pd.DataFrame:
    pre = OptimizedPreprocessor()
    post = VectorizedPostprocessor(aggressive=True)

    raw_src = df["transliteration"].astype(str).fillna("").str.strip()
    raw_tgt = df["translation"].astype(str).fillna("").str.strip()

    pre_src = pd.Series(pre.preprocess_batch(raw_src.tolist()), index=df.index)
    post_tgt = pd.Series(post.postprocess_batch(raw_tgt.tolist()), index=df.index)

    def _ratio(src_s, tgt_s):
        src_c = src_s.str.len()
        tgt_c = tgt_s.str.len()
        return (tgt_c / src_c.replace(0, np.nan)).fillna(0.0)

    rr_ratio = _ratio(raw_src, raw_tgt)
    pr_ratio = _ratio(pre_src, raw_tgt)
    rp_ratio = _ratio(raw_src, post_tgt)
    pp_ratio = _ratio(pre_src, post_tgt)

    raw_src_c = raw_src.str.len()
    raw_tgt_c = raw_tgt.str.len()
    header_only = raw_tgt.str.lower().str.startswith("to ") & (raw_src_c >= 80) & (raw_tgt_c <= 60)

    rr_len = rr_ratio <= float(ratio_max)
    pr_len = pr_ratio <= float(ratio_max)
    rp_len = rp_ratio <= float(ratio_max)
    pp_len = pp_ratio <= float(ratio_max)

    flag = header_only | rr_len | pr_len | rp_len | pp_len

    out = pd.DataFrame(
        {
            "src_chars": raw_src_c, "tgt_chars": raw_tgt_c,
            "rr_ratio": rr_ratio, "pr_ratio": pr_ratio,
            "rp_ratio": rp_ratio, "pp_ratio": pp_ratio,
            "header_only": header_only,
            "hit_rr": rr_len, "hit_pr": pr_len,
            "hit_rp": rp_len, "hit_pp": pp_len,
        },
        index=df.index,
    )
    out["flag"] = flag
    return out


def _safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def add_incomplete_reasons_hf(
    ds, *, ratio_max: float = 0.50, batch_size: int = 2048, num_proc: int | None = None,
):
    if not _is_hf_dataset(ds):
        raise TypeError("add_incomplete_reasons_hf expects a HuggingFace datasets.Dataset")

    pre = OptimizedPreprocessor()
    post = VectorizedPostprocessor(aggressive=True)

    def _map(batch):
        raw_src = [_safe_str(x).strip() for x in batch["transliteration"]]
        raw_tgt = [_safe_str(x).strip() for x in batch["translation"]]
        pre_src = pre.preprocess_batch(raw_src)
        post_tgt = post.postprocess_batch(raw_tgt)

        src_c = [len(s) for s in raw_src]
        tgt_c = [len(t) for t in raw_tgt]

        def _ratios(src_list, tgt_list):
            return [float(len(t) / len(s)) if len(s) > 0 else 0.0 for s, t in zip(src_list, tgt_list)]

        rr_ratio = _ratios(raw_src, raw_tgt)
        pr_ratio = _ratios(pre_src, raw_tgt)
        rp_ratio = _ratios(raw_src, post_tgt)
        pp_ratio = _ratios(pre_src, post_tgt)

        header_only = [bool(t.lower().startswith("to ") and sc >= 80 and tc <= 60)
                       for s, t, sc, tc in zip(raw_src, raw_tgt, src_c, tgt_c)]

        hit_rr = [bool(r <= float(ratio_max)) for r in rr_ratio]
        hit_pr = [bool(r <= float(ratio_max)) for r in pr_ratio]
        hit_rp = [bool(r <= float(ratio_max)) for r in rp_ratio]
        hit_pp = [bool(r <= float(ratio_max)) for r in pp_ratio]

        flag = [bool(ho or a or b or c or d) for ho, a, b, c, d in zip(header_only, hit_rr, hit_pr, hit_rp, hit_pp)]

        return {
            "src_chars": src_c, "tgt_chars": tgt_c,
            "rr_ratio": rr_ratio, "pr_ratio": pr_ratio,
            "rp_ratio": rp_ratio, "pp_ratio": pp_ratio,
            "header_only": header_only,
            "hit_rr": hit_rr, "hit_pr": hit_pr,
            "hit_rp": hit_rp, "hit_pp": hit_pp,
            "flag": flag,
        }

    return ds.map(_map, batched=True, batch_size=batch_size, num_proc=num_proc)


def filter_incomplete_hf(
    ds, *, ratio_max: float = 0.50, keep: bool = False,
    batch_size: int = 2048, num_proc: int | None = None,
):
    ds2 = add_incomplete_reasons_hf(ds, ratio_max=ratio_max, batch_size=batch_size, num_proc=num_proc)
    return ds2.filter(lambda ex: bool(ex["flag"]) if keep else (not bool(ex["flag"])))


def flag_incomplete(
    data, *, ratio_max: float = 0.50, print: bool = False, max_rows: int | None = None,
    snippet_chars: int = 160, hf_batch_size: int = 2048, hf_num_proc: int | None = None,
    return_reasons: bool = False,
):
    def _snip(s: str) -> str:
        s = _safe_str(s).replace("\n", " ").strip()
        return s if len(s) <= snippet_chars else (s[: snippet_chars - 1] + "…")

    # pandas path
    if isinstance(data, pd.DataFrame):
        reasons = _make_incomplete_reasons_df(data, ratio_max=ratio_max)
        mask = reasons["flag"]

        if print:
            total = int(mask.sum())
            _bt.print(f"[flag_incomplete] flagged {total}/{len(data)} rows (rr|pr|rp|pp)")
            if total:
                counts = reasons.loc[mask, ["header_only", "hit_rr", "hit_pr", "hit_rp", "hit_pp"]].sum().astype(int)
                _bt.print("Reason counts:", counts.to_dict())

                raw_src = data["transliteration"].astype(str).fillna("").str.strip()
                raw_tgt = data["translation"].astype(str).fillna("").str.strip()

                report = reasons.loc[mask].copy()
                if "oare_id" in data.columns:
                    report.insert(0, "oare_id", data.loc[mask, "oare_id"].astype(str).values)
                else:
                    report.insert(0, "row_id", report.index.astype(str))

                def _pairs_row(r):
                    hit = []
                    if bool(r["header_only"]): hit.append("header_only")
                    if bool(r["hit_rr"]): hit.append("raw/raw")
                    if bool(r["hit_pr"]): hit.append("pre/raw")
                    if bool(r["hit_rp"]): hit.append("raw/post")
                    if bool(r["hit_pp"]): hit.append("pre/post")
                    return ",".join(hit)

                report["reasons"] = report.apply(_pairs_row, axis=1)
                report["src_snip"] = raw_src.loc[mask].map(lambda x: _snip(x)).values
                report["tgt_snip"] = raw_tgt.loc[mask].map(lambda x: _snip(x)).values

                cols = [
                    report.columns[0],
                    "src_chars", "tgt_chars",
                    "rr_ratio", "pr_ratio", "rp_ratio", "pp_ratio",
                    "reasons", "src_snip", "tgt_snip",
                ]
                report = report[cols].sort_values(["rr_ratio", "src_chars"], ascending=[True, False])
                if max_rows is not None:
                    report = report.head(max_rows)

                with pd.option_context("display.max_colwidth", None, "display.width", 200):
                    _bt.print(report.to_string(index=False))

        return mask

    # HF Dataset path
    if _is_hf_dataset(data):
        ds_with = add_incomplete_reasons_hf(data, ratio_max=ratio_max, batch_size=hf_batch_size, num_proc=hf_num_proc)
        mask = np.asarray(ds_with["flag"], dtype=np.bool_)

        if print:
            total = int(mask.sum())
            _bt.print(f"[flag_incomplete] flagged {total}/{len(ds_with)} rows (rr|pr|rp|pp)")
            if total:
                counts = {
                    "header_only": int(np.asarray(ds_with["header_only"], dtype=np.bool_).sum()),
                    "hit_rr": int(np.asarray(ds_with["hit_rr"], dtype=np.bool_).sum()),
                    "hit_pr": int(np.asarray(ds_with["hit_pr"], dtype=np.bool_).sum()),
                    "hit_rp": int(np.asarray(ds_with["hit_rp"], dtype=np.bool_).sum()),
                    "hit_pp": int(np.asarray(ds_with["hit_pp"], dtype=np.bool_).sum()),
                }
                _bt.print("Reason counts:", counts)

        if return_reasons:
            return mask, ds_with
        return mask

    if _is_hf_datasetdict(data):
        raise TypeError("flag_incomplete got a DatasetDict. Call it on a split: ds['train'].")

    raise TypeError(f"Unsupported type: {type(data)}. Expected pd.DataFrame or datasets.Dataset.")


# --- bounded rolling dedupe (stream-safe) ---
from collections import deque

class RollingDedupe:
    __slots__ = ("max_items", "q", "s")
    def __init__(self, max_items: int = 1_000_000):
        self.max_items = int(max_items)
        self.q = deque()
        self.s = set()

    def seen_or_add(self, key: str) -> bool:
        if key in self.s:
            return True
        self.s.add(key)
        self.q.append(key)
        if len(self.q) > self.max_items:
            old = self.q.popleft()
            self.s.discard(old)
        return False


# ================================================================
# Deterministic term swapping for probing
# ================================================================

def _hash_u32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return int(x & 0xFFFFFFFF)

def _u01(u: int) -> float:
    return (u & 0xFFFFFFFF) / 4294967296.0


# Canonical paired swap lists
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


def _case_like(new_tok: str, template_tok: str) -> str:
    if template_tok.isupper():
        return new_tok.upper()
    if template_tok.islower():
        return new_tok.lower()
    return new_tok

def _ws_tok_pat(tok: str):
    return re.compile(r"(?<!\S)" + re.escape(tok) + r"(?!\S)", flags=re.IGNORECASE)

def _has_tok(src: str, tok: str) -> bool:
    return _ws_tok_pat(tok).search(src) is not None

def _swap_src_tok(src: str, old: str, new: str) -> str:
    pat = _ws_tok_pat(old)
    def _repl(m):
        matched = m.group(0)
        return _case_like(new, matched)
    return pat.sub(_repl, src)

def _swap_tgt_word(tgt: str, old_word: str, new_word: str):
    m = re.search(r"\b" + re.escape(old_word) + r"s?\b", tgt, flags=re.IGNORECASE)
    if not m:
        return None
    matched = m.group(0)
    rep = (new_word + "s") if matched.lower().endswith("s") else new_word
    return tgt[:m.start()] + rep + tgt[m.end():]

def _det_choice(seq, h: int):
    if not seq:
        return None
    return seq[int(h % len(seq))]


def apply_term_swap_det(src, tgt, idx, seed, from_pairs, to_pairs):
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


# NUMERIC_MEASURE_SWAP
_NUM_POOL_DEFAULT = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,25,30,34,37,40,50,60,80,100]

def _find_ints(s: str):
    return re.findall(r"\b\d+\b", str(s))

def _swap_first_int(s: str, old: str, new: str) -> str:
    return re.sub(r"\b" + re.escape(old) + r"\b", str(new), str(s), count=1)

def apply_numeric_measure_swap_det(src, tgt, idx, seed):
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


# TWO_ENTITY_ORDER_SWAP
_IGI_NAME_PAT = re.compile(r"\bIGI\s+([^\s]+)", flags=re.IGNORECASE)

def _swap_first_two_igi(src: str):
    ms = list(_IGI_NAME_PAT.finditer(src))
    if len(ms) < 2:
        return src, False
    m1, m2 = ms[0], ms[1]
    name1 = m1.group(1)
    name2 = m2.group(1)
    s = src
    s = re.sub(r"\bIGI\s+" + re.escape(name1) + r"\b", f"IGI __TMP__", s, count=1, flags=re.IGNORECASE)
    s = re.sub(r"\bIGI\s+" + re.escape(name2) + r"\b", f"IGI {name1}", s, count=1, flags=re.IGNORECASE)
    s = re.sub(r"\bIGI\s+__TMP__\b", f"IGI {name2}", s, count=1, flags=re.IGNORECASE)
    return (s, s != src)

_TGT_WIT_1 = re.compile(r"(in the presence of)\s+([^.,;]+?)\s+(and of)\s+([^.,;]+?)([.,;]|$)", flags=re.IGNORECASE)
_TGT_WIT_2 = re.compile(r"(Witnessed by)\s+([^.,;]+?)\s*,\s*(by)\s+([^.,;]+?)([.,;]|$)", flags=re.IGNORECASE)

def _swap_witness_tgt(tgt: str):
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

def apply_two_entity_order_swap_det(src, tgt, idx, seed):
    src2, ok_s = _swap_first_two_igi(src)
    tgt2, ok_t = _swap_witness_tgt(tgt)
    ok = ok_s and ok_t
    return (src2, tgt2, ok) if ok else (src, tgt, False)


# ================================================================
# Probing/pn/gloss variants builder
# ================================================================

def _get_enabled_probe_ops(Config):
    ops = []
    pe = getattr(Config, "PROBE_ENABLE", {}) or {}

    def add_term(name, from_pairs, to_pairs):
        def fn(src, tgt, idx, seed):
            return apply_term_swap_det(src, tgt, idx, seed, from_pairs, to_pairs)
        ops.append((name, fn))

    def add_custom(name, apply_fn):
        ops.append((name, apply_fn))

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

    return ops


def _normalize_weights(names, wdict):
    if not names:
        return None
    w = np.asarray([float((wdict or {}).get(n, 1.0)) for n in names], dtype=np.float64)
    s = float(w.sum())
    return (w / (s + 1e-12)) if s > 0 else (np.ones_like(w) / len(w))


def _hash_str_u32(s: str) -> int:
    return int(zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF)


# Conformance-only TGT postprocessor
_UFRAC = {
    "½": "0.5", "¼": "0.25", "¾": "0.75",
    "⅓": "0.33333", "⅔": "0.66666",
    "⅙": "0.16666", "⅚": "0.83333",
}
_MIXED_UFRAC_RE = re.compile(r"\b(\d+)\s*([½¼¾⅓⅔⅙⅚])\b")
_SLASH_FRAC_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
_DEN_MAP = {2: 0.5, 3: 1.0/3.0, 4: 0.25, 6: 1.0/6.0}
_PARENS_RE = re.compile(r"\([^)]*\)")
_GRAM_TAG_RE = re.compile(r"\((fem|plur|pl|sing|singular|plural)\.?\s*\w*\)", re.I)


def _normalize_gaps_str(t: str) -> str:
    if t is None:
        return ""
    s = str(t)
    s = _TAG_BIGGAP_RE.sub("<gap>", s)
    s = _TAG_GAP_RE.sub("<gap>", s)
    s = _XTOKEN_RUN_RE.sub("<gap>", s)
    s = _ELLIPSIS_RE.sub("<gap>", s)
    s = _BRACKET_X_RE.sub("<gap>", s)
    s = _XRUN_RE.sub("<gap>", s)
    s = _XTOK_RE.sub("<gap>", s)
    return s


def _collapse_gap_runs_str(s: str) -> str:
    toks = str(s).split()
    out = []
    i = 0
    n = len(toks)
    while i < n:
        if toks[i] == "<gap>":
            j = i
            while j < n and toks[j] == "<gap>":
                j += 1
            out.append("<gap>")
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def _ufrac_to_decimals(s: str) -> str:
    def repl(m):
        ip = int(m.group(1))
        uf = m.group(2)
        dec = _UFRAC.get(uf)
        if dec is None:
            return m.group(0)
        try:
            return _canon_decimal_str(ip + float(dec))
        except Exception:
            return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    s = _MIXED_UFRAC_RE.sub(repl, s)
    for uf, dec in _UFRAC.items():
        s = s.replace(uf, dec)
    return s


def _slashfrac_to_decimals(s: str) -> str:
    def repl(m):
        n = int(m.group(1)); d = int(m.group(2))
        if d not in _DEN_MAP:
            return m.group(0)
        return _canon_decimal_str(n * _DEN_MAP[d])
    return _SLASH_FRAC_RE.sub(repl, s)


def conformance_postprocess_tgt_batch(xs, *, collapse_gaps: bool = True) -> List[str]:
    out = []
    for x in (xs or []):
        s = "" if x is None else str(x)
        s = _normalize_gaps_str(s)
        s = _PN_RE.sub("<gap>", s)
        s = _GRAM_TAG_RE.sub(" ", s)
        s = _PARENS_RE.sub(" ", s)
        s = _QUOTES_RE.sub("", s)
        s = _ufrac_to_decimals(s)
        s = _slashfrac_to_decimals(s)
        s = space_gap_token(s)
        if collapse_gaps:
            s = _collapse_gap_runs_str(s)
        s = _WS_RE.sub(" ", s).strip()
        out.append(s)
    return out


def _maybe_post_tgt_batch(Config, texts, *, collapse_gaps: bool):
    xs = list(texts) if texts is not None else []
    xs = ["" if x is None else str(x) for x in xs]
    if not bool(getattr(Config, "POSTPROCESS_TARGETS", True)):
        return xs
    return conformance_postprocess_tgt_batch(xs, collapse_gaps=bool(collapse_gaps))


# OK-only probe append (TEXT-level)
def build_probe_append_text_ds(
    *, base_text_ds, pre, Config, p_probe, ops, seed, cat_weights=None,
    attempt_mult=40, enforce_unique=True,
):
    N = len(base_text_ds)
    M = int(round(float(p_probe) * N))
    if M <= 0 or (not ops) or N <= 0:
        return None

    names = [n for n, _ in ops]
    w = _normalize_weights(names, cat_weights)
    cw = np.cumsum(w)

    src_clean = pre.preprocess_batch(list(base_text_ds["transliteration"]))
    tgt_clean = _maybe_post_tgt_batch(Config, list(base_text_ds["translation"]), collapse_gaps=True)

    oare_ids = list(base_text_ds["oare_id"]) if "oare_id" in base_text_ds.column_names else [None] * N
    pair_ids = list(base_text_ds["pair_id"]) if "pair_id" in base_text_ds.column_names else [None] * N
    is_sent  = list(base_text_ds["is_sentence"]) if "is_sentence" in base_text_ds.column_names else [False] * N

    rows = []
    seen = set()

    total_attempts = max(1, int(M) * int(attempt_mult))
    for a in range(total_attempts):
        i = int(_hash_u32((seed + 999) ^ (a * 9176) ^ 0x51CE) % N)

        u = _u01(_hash_u32((seed + 12345) ^ (a * 2654435761) ^ 0xABCD))
        j = int(np.searchsorted(cw, u, side="right"))
        j = min(max(j, 0), len(ops) - 1)
        cat, fn = ops[j]

        s0 = str(src_clean[i])
        t0 = str(tgt_clean[i])

        mix_seed = (int(seed) ^ _hash_str_u32(cat) ^ (int(a) * 1009)) & 0xFFFFFFFF
        s2, t2, ok = fn(s0, t0, int(i), int(mix_seed))
        if not ok:
            continue

        if enforce_unique:
            key = (_hash_str_u32(s2), _hash_str_u32(t2))
            if key in seen:
                continue
            seen.add(key)

        oid = oare_ids[i]
        rows.append({
            "oare_id": oid,
            "pair_id": (f"{str(oid)}::probe::{cat}::{i}::{len(rows)}" if oid is not None else f"probe::{cat}::{i}::{len(rows)}"),
            "transliteration": s2,
            "translation": t2,
            "src_is_preprocessed": True,
            "is_sentence": bool(is_sent[i]),
            "src_view": f"probe::{cat}",
            "base_pair_id": pair_ids[i],
        })

        if len(rows) >= M:
            break

    return Dataset.from_list(rows) if rows else None


# Tokenization maps
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


def make_map_raw(Config, tokenizer, pre, *, tgt_max):
    def _fn(examples):
        src_clean = _get_src_clean_batch(pre, examples)
        inputs = [Config.PREFIX + str(x) for x in src_clean]
        collapse_lbl = bool(getattr(Config, "LABEL_COLLAPSE_GAPS", True))
        targets = _maybe_post_tgt_batch(Config, examples["translation"], collapse_gaps=collapse_lbl)
        out = tokenizer(inputs, max_length=int(Config.SRC_MAX_LENGTH), truncation=True, padding=False)
        lab = tokenizer(targets, max_length=int(tgt_max), truncation=True, padding=False)
        out["labels"] = lab["input_ids"]
        out["input_length"] = [len(x) for x in out["input_ids"]]
        return out
    return _fn


def make_map_pn(Config, tokenizer, pre, canon, *, tgt_max):
    def _fn(examples):
        src_clean = _get_src_clean_batch(pre, examples)
        src_pn = [canon.canonicalize_source(s, mode="pn_norm") for s in src_clean]
        inputs = [Config.PREFIX + str(x) for x in src_pn]
        collapse_lbl = bool(getattr(Config, "LABEL_COLLAPSE_GAPS", True))
        targets = _maybe_post_tgt_batch(Config, examples["translation"], collapse_gaps=collapse_lbl)
        out = tokenizer(inputs, max_length=int(Config.SRC_MAX_LENGTH), truncation=True, padding=False)
        lab = tokenizer(targets, max_length=int(tgt_max), truncation=True, padding=False)
        out["labels"] = lab["input_ids"]
        out["input_length"] = [len(x) for x in out["input_ids"]]
        return out
    return _fn


def make_map_gloss_raw(Config, tokenizer, pre, glosser, *, seed_for_variant, tgt_max):
    def _fn(examples, indices):
        src_clean = _get_src_clean_batch(pre, examples)
        src_g = [
            glosser.append_gloss(
                s,
                max_items=int(getattr(Config, "GLOSS_MAX_ITEMS", 6)),
                max_append_chars=int(getattr(Config, "GLOSS_MAX_APPEND_CHARS", 240)),
                seed=int(seed_for_variant), epoch=0, example_id=int(idx), keep_order=True,
            )
            for s, idx in zip(src_clean, indices)
        ]
        inputs = [Config.PREFIX + str(x) for x in src_g]
        collapse_lbl = bool(getattr(Config, "LABEL_COLLAPSE_GAPS", True))
        targets = _maybe_post_tgt_batch(Config, examples["translation"], collapse_gaps=collapse_lbl)
        out = tokenizer(inputs, max_length=int(Config.SRC_MAX_LENGTH), truncation=True, padding=False)
        lab = tokenizer(targets, max_length=int(tgt_max), truncation=True, padding=False)
        out["labels"] = lab["input_ids"]
        out["input_length"] = [len(x) for x in out["input_ids"]]
        return out
    return _fn


def _align_features_for_concat(base_ds, other_ds):
    """
    Align shared column feature types so `concatenate_datasets` does not fail
    on string vs large_string mismatches.
    """
    if other_ds is None:
        return None
    out = other_ds
    for col in base_ds.column_names:
        if col in out.column_names:
            base_feat = base_ds.features[col]
            other_feat = out.features[col]
            if base_feat != other_feat:
                out = out.cast_column(col, base_feat)
    return out


class EpochVariantMinViewMix3(torch.utils.data.Dataset):
    def __init__(self, raws, pns, glosses, *, shared_epoch, seed=42, p_pn=0.5, p_gloss=0.5, mix_seed=None):
        self.raws = list(raws)
        self.pns = list(pns)
        self.glosses = list(glosses)

        assert len(self.raws) >= 1
        K = len(self.raws)
        assert len(self.pns) == K and len(self.glosses) == K
        self.K = K

        for v in range(K):
            L = len(self.raws[v])
            assert len(self.pns[v]) == L and len(self.glosses[v]) == L

        self.shared_epoch = shared_epoch
        self.seed = int(seed)
        self.p_pn = float(p_pn)
        self.p_gl = float(p_gloss)
        self.mix_seed = int(mix_seed if mix_seed is not None else (self.seed + 991))
        self.L = min(len(self.raws[v]) for v in range(K))

    def __len__(self):
        return self.L

    def _epoch(self):
        try:
            return int(self.shared_epoch.value)
        except Exception:
            return 0

    def __getitem__(self, idx):
        idx = int(idx)
        ep = self._epoch()
        v = _hash_u32(self.seed ^ (ep * 1000003) ^ 0xA5A5A5A5) % self.K
        v = int(v)
        u_pn = _u01(_hash_u32(self.mix_seed ^ (ep * 1000003) ^ (idx * 9176) ^ 0xC0FFEE))
        u_gl = _u01(_hash_u32((self.mix_seed + 1337) ^ (ep * 19217) ^ (idx * 2654435761) ^ 0xA53C))
        use_pn = (self.p_pn > 0.0) and (u_pn < self.p_pn)
        use_gl = (self.p_gl > 0.0) and (u_gl < self.p_gl)
        if use_gl:
            return self.glosses[v][idx]
        if use_pn:
            return self.pns[v][idx]
        return self.raws[v][idx]


# Main builder: K train variants + val
def build_probe_then_pngloss_variants(
    *, Config, train_text_ds, val_text_ds, tokenizer, pre, canon, glosser, NPROC=8, MAP_BS=1024,
    use_pn_view=True, use_gloss_view=True,
):
    K = int(getattr(Config, "K_TRAIN_VARIANTS", 4))
    p_probe_tr = float(getattr(Config, "PROBE_APPEND_P", 0.0))
    p_probe_va = float(getattr(Config, "VAL_PROBE_APPEND_P", 0.0))

    base_seed  = int(getattr(Config, "SEED", 42))
    probe_seed = int(getattr(Config, "PROBE_SEED", base_seed + 3))
    gloss_seed = int(getattr(Config, "GLOSS_SEED", base_seed + 777))
    tgt_max = int(getattr(Config, "TGT_MAX_LENGTH", getattr(Config, "GEN_MAX_NEW_TOKENS", 768)))

    p_pn = float(getattr(Config, "PN_MIX_P", 0.5))
    p_gl = float(getattr(Config, "GLOSS_MIX_P", 0.5))
    mix_seed = int(getattr(Config, "MIX_SEED", base_seed + 991))

    ops = _get_enabled_probe_ops(Config)
    cat_w = getattr(Config, "PROBE_CAT_WEIGHTS", None)
    attempt_mult = int(getattr(Config, "PROBE_ATTEMPT_MULT", 40))

    shared_epoch = Value("i", 0)

    raws, pns, glosses = [], [], []

    for v in tqdm(range(K), desc="Build train variants (probe→views)"):
        ds_base = train_text_ds
        ds_probe = build_probe_append_text_ds(
            base_text_ds=ds_base, pre=pre, Config=Config,
            p_probe=p_probe_tr, ops=ops,
            seed=int(probe_seed + 1009 * v), cat_weights=cat_w,
            attempt_mult=attempt_mult, enforce_unique=True,
        )
        ds_probe = _align_features_for_concat(ds_base, ds_probe)
        ds_text = concatenate_datasets([ds_base, ds_probe]) if ds_probe is not None else ds_base

        ds_raw = ds_text.map(
            make_map_raw(Config, tokenizer, pre, tgt_max=int(tgt_max)),
            batched=True, batch_size=int(MAP_BS), num_proc=int(NPROC),
            remove_columns=[c for c in ds_text.column_names if c not in ()],
        )
        if bool(use_pn_view):
            ds_pn = ds_text.map(
                make_map_pn(Config, tokenizer, pre, canon, tgt_max=int(tgt_max)),
                batched=True, batch_size=int(MAP_BS), num_proc=int(NPROC),
                remove_columns=[c for c in ds_text.column_names if c not in ()],
            )
        else:
            ds_pn = ds_raw
        if bool(use_gloss_view):
            ds_gl = ds_text.map(
                make_map_gloss_raw(Config, tokenizer, pre, glosser, seed_for_variant=int(gloss_seed + 7919 * v), tgt_max=int(tgt_max)),
                with_indices=True, batched=True, batch_size=int(MAP_BS), num_proc=int(NPROC),
                remove_columns=[c for c in ds_text.column_names if c not in ()],
            )
        else:
            ds_gl = ds_raw

        raws.append(ds_raw)
        pns.append(ds_pn)
        glosses.append(ds_gl)

    tokenized_train = EpochVariantMinViewMix3(
        raws, pns, glosses,
        shared_epoch=shared_epoch, seed=base_seed,
        p_pn=p_pn, p_gloss=p_gl, mix_seed=mix_seed,
    )

    ds_probe_va = build_probe_append_text_ds(
        base_text_ds=val_text_ds, pre=pre, Config=Config,
        p_probe=p_probe_va, ops=ops,
        seed=int(probe_seed + 555), cat_weights=cat_w,
        attempt_mult=attempt_mult, enforce_unique=True,
    )
    ds_probe_va = _align_features_for_concat(val_text_ds, ds_probe_va)
    val_text_plus = concatenate_datasets([val_text_ds, ds_probe_va]) if ds_probe_va is not None else val_text_ds

    tokenized_val = val_text_plus.map(
        make_map_raw(Config, tokenizer, pre, tgt_max=int(tgt_max)),
        batched=True, batch_size=int(MAP_BS), num_proc=int(NPROC),
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
    st_path  = os.path.join(ckpt_dir, "model.safetensors")
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
        state = json.load(open(state_path, "r", encoding="utf-8"))
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
    output_dir, save_dir, *, k=8, metric_key="eval_geo_mean",
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

    base_ckpt = base_ckpt_for_config if (base_ckpt_for_config and os.path.isdir(base_ckpt_for_config)) else chosen[0]

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
        "averaged_k": n, "metric_key": metric_key, "prefer_best": prefer_best,
        "chosen_checkpoints": chosen, "base_ckpt": base_ckpt,
        "missing_keys_count": len(missing), "unexpected_keys_count": len(unexpected),
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


# ================================================================
# TBM (Translation/Template Based Matching)
# ================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class TBMIndex:
    def __init__(self, train_src_norm, train_tgt, *, ngram=(3,6), max_features=250_000):
        self.train_tgt = list(map(str, train_tgt))
        self.vec = TfidfVectorizer(
            analyzer="char", ngram_range=tuple(ngram),
            min_df=1, max_features=int(max_features),
        )
        X = self.vec.fit_transform(list(map(str, train_src_norm)))
        self.nn = NearestNeighbors(n_neighbors=8, metric="cosine", algorithm="brute")
        self.nn.fit(X)

    def query(self, src_norm, k=3):
        q = self.vec.transform([str(src_norm)])
        dists, idxs = self.nn.kneighbors(q, n_neighbors=int(k), return_distance=True)
        sims = 1.0 - dists[0]
        idxs = idxs[0]
        return [(self.train_tgt[i], float(s)) for i, s in zip(idxs, sims)]


def build_tbm_from_pairs(pre, pairs_df, *, ngram=(3,6), max_features=250_000,
                         src_col="transliteration", tgt_col="translation"):
    if pairs_df is None or len(pairs_df) < 10:
        return None
    if (src_col not in pairs_df.columns) or (tgt_col not in pairs_df.columns):
        return None
    src_raw = pairs_df[src_col].astype(str).tolist()
    tgt_raw = pairs_df[tgt_col].astype(str).tolist()
    src_norm = pre.preprocess_batch(src_raw)
    keep_src, keep_tgt = [], []
    for s, t in zip(src_norm, tgt_raw):
        s = str(s).strip()
        t = str(t).strip()
        if s and t:
            keep_src.append(s)
            keep_tgt.append(t)
    if len(keep_src) < 10:
        return None
    return TBMIndex(keep_src, keep_tgt, ngram=ngram, max_features=max_features)


# ================================================================
# MBR EVAL + Trainer
# ================================================================

def _stable_int_id(s: str) -> int:
    return int(zlib.adler32(str(s).encode("utf-8")) & 0x7FFFFFFF)

def _norm_ws(s: str) -> str:
    return " ".join(str(s).strip().split())

def _dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        x = str(x)
        if x.strip() and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _val_unique_examples(val_source, *, prefer_original=True):
    if hasattr(val_source, "column_names"):
        cols = set(val_source.column_names)
        N = len(val_source)
        ex_all   = [str(x) for x in (val_source["ex_id"] if "ex_id" in cols else [f"row{i}" for i in range(N)])]
        src_all  = list(val_source["transliteration"])
        ref_all  = list(val_source["translation"])
        oare_all = list(val_source["oare_id"]) if ("oare_id" in cols) else [None] * N
        view_all = list(val_source["src_view"]) if ("src_view" in cols) else ["original"] * N

        if "ex_id" not in cols:
            return ex_all, src_all, ref_all, oare_all

        best = {}
        for i, (eid, v) in enumerate(zip(ex_all, view_all)):
            if eid not in best:
                best[eid] = i
            elif prefer_original and str(v) == "original":
                best[eid] = i

        seen = set()
        picked = []
        for eid in ex_all:
            if eid in seen:
                continue
            seen.add(eid)
            picked.append(best[eid])

        return [ex_all[i] for i in picked], [src_all[i] for i in picked], [ref_all[i] for i in picked], [oare_all[i] for i in picked]

    if isinstance(val_source, pd.DataFrame):
        df = val_source
        if "ex_id" not in df.columns:
            oare_list = df["oare_id"].astype(str).tolist() if "oare_id" in df.columns else [None] * len(df)
            return [f"row{i}" for i in range(len(df))], df["transliteration"].astype(str).tolist(), df["translation"].astype(str).tolist(), oare_list

        if prefer_original and ("src_view" in df.columns):
            df2 = df.copy()
            df2["__is_orig__"] = (df2["src_view"].astype(str) == "original").astype(int)
            df2 = df2.sort_values(["ex_id", "__is_orig__"], ascending=[True, False]).drop_duplicates("ex_id", keep="first")
            order = pd.Series(df["ex_id"].astype(str).tolist()).drop_duplicates().tolist()
            df2["__ord__"] = df2["ex_id"].astype(str).map({k: i for i, k in enumerate(order)})
            df2 = df2.sort_values("__ord__").drop(columns=["__is_orig__", "__ord__"])
        else:
            df2 = df.drop_duplicates("ex_id", keep="first")

        oare_list = df2["oare_id"].astype(str).tolist() if "oare_id" in df2.columns else [None] * len(df2)
        return df2["ex_id"].astype(str).tolist(), df2["transliteration"].astype(str).tolist(), df2["translation"].astype(str).tolist(), oare_list

    raise TypeError("val_source must be HF Dataset or pandas DataFrame.")


def _official_geo_mean(refs, preds):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    geo = float(math.sqrt(float(bleu) * float(chrfpp)))
    return geo, float(bleu), float(chrfpp)


def _geo_sim_sentence(a, b):
    bleu = sacrebleu.sentence_bleu(a, [b]).score
    chrf = sacrebleu.sentence_chrf(a, [b], word_order=2).score
    return float(math.sqrt(max(0.0, bleu) * max(0.0, chrf)))


def _mbr_pick_geo(cands):
    cands = _dedup_keep_order(cands)
    n = len(cands)
    if n == 0:
        return ""
    if n == 1:
        return cands[0]
    sims = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _geo_sim_sentence(cands[i], cands[j])
            sims[i, j] = s
            sims[j, i] = s
    avg = (sims.sum(axis=1) - np.diag(sims)) / float(n - 1)
    return cands[int(np.argmax(avg))]


@torch.inference_mode()
def _generate_multi_decode(
    model, tokenizer, inputs, *,
    device,
    src_max_length=512,
    max_new_tokens=512,
    num_beams=4,
    num_beam_cands=1,
    length_penalty=1.0,
    num_sample_cands=8,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    no_repeat_ngram_size=0,
):
    enc = tokenizer(
        inputs, return_tensors="pt", padding=True,
        truncation=True, max_length=int(src_max_length),
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    B = len(inputs)
    out_lists = [[] for _ in range(B)]

    if int(num_beam_cands) > 0:
        nb = int(max(1, int(num_beams), int(num_beam_cands)))
        beam_kwargs = dict(
            do_sample=False, num_beams=nb,
            num_return_sequences=int(num_beam_cands),
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            use_cache=True,
        )
        if nb > 1:
            beam_kwargs["length_penalty"] = float(length_penalty)
            beam_kwargs["early_stopping"] = True

        beam_out = model.generate(**enc, **beam_kwargs)
        beam_txt = tokenizer.batch_decode(beam_out, skip_special_tokens=True)
        R = int(num_beam_cands)
        for i in range(B):
            out_lists[i].extend(beam_txt[i * R:(i + 1) * R])

    if int(num_sample_cands) > 0:
        samp_out = model.generate(
            **enc, do_sample=True, num_beams=1,
            temperature=float(temperature), top_p=float(top_p),
            num_return_sequences=int(num_sample_cands),
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            use_cache=True,
        )
        samp_txt = tokenizer.batch_decode(samp_out, skip_special_tokens=True)
        R = int(num_sample_cands)
        for i in range(B):
            out_lists[i].extend(samp_txt[i * R:(i + 1) * R])

    return out_lists


class MBRGlossSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self, *args,
        val_text_ds=None, pre=None, prefix="",
        post=None, post_ref=None,
        glosser=None, gloss_variants=1, gloss_seed=12345,
        gloss_max_items=6, gloss_max_append_chars=240,
        mbr_batch_size_inputs=16,
        src_max_length=512, max_new_tokens=512,
        num_beams=8, num_beam_cands=1, num_sample_cands=4,
        length_penalty=1.3, temperature=0.7, top_p=0.9,
        repetition_penalty=1.0, no_repeat_ngram_size=0,
        mbr_pool_cap=10, show_progress=True,
        tbm_index=None, tbm_pairs=None,
        tbm_ngram=(3,6), tbm_max_features=250_000,
        tbm_topk=3, tbm_min_sim=0.92, tbm_hard_sim=0.97,
        tbm_enable=True,
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
        self.post_ref = post_ref
        self.glosser = glosser
        self.gloss_variants = int(gloss_variants)
        self.gloss_seed = int(gloss_seed)
        self.gloss_max_items = int(gloss_max_items)
        self.gloss_max_append_chars = int(gloss_max_append_chars)
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
                        ngram=tuple(tbm_ngram), max_features=int(tbm_max_features),
                    )
                except Exception as e:
                    self.tbm_index = None

        if not bool(getattr(self.args, "gradient_checkpointing", False)):
            try:
                self.model.config.use_cache = True
            except Exception:
                pass
            try:
                if getattr(self.model, "generation_config", None) is not None:
                    self.model.generation_config.use_cache = True
            except Exception:
                pass

    def _is_dist(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _broadcast_metrics(self, metrics):
        if not self._is_dist():
            return metrics
        obj = [metrics if self.is_world_process_zero() else None]
        torch.distributed.broadcast_object_list(obj, src=0)
        return obj[0]

    def _bf16_eval_context(self):
        if torch.cuda.is_available() and bool(getattr(self.args, "bf16", False)):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _tqdm(self, total, desc):
        if (not self.show_progress) or (tqdm is None):
            return None
        return tqdm(total=total, desc=desc, leave=False)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        t0 = time.time()
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("evaluate() needs an eval_dataset.")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.evaluation_loop(
            eval_dataloader, description="Evaluation",
            prediction_loss_only=True, ignore_keys=ignore_keys,
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

    def _evaluate_mbr(
        self,
        metric_key_prefix="eval",
        *,
        save_dir_override: Optional[str] = None,
        file_tag_override: Optional[str] = None,
    ):
        model = self.model
        tok = self.processing_class
        device = model.device

        ex_ids, srcs, refs_raw, oare_ids = _val_unique_examples(self.val_text_ds, prefer_original=True)
        refs = [_norm_ws(r) for r in refs_raw]
        src_clean = self.pre.preprocess_batch(list(srcs))

        flat_inputs, flat_exi = [], []
        for ex_i, (eid, s) in enumerate(zip(ex_ids, src_clean)):
            base = str(s)
            flat_inputs.append(self.prefix + base)
            flat_exi.append(ex_i)
            if (self.glosser is not None) and (self.gloss_variants > 0):
                ex_int = _stable_int_id(eid)
                for v in range(self.gloss_variants):
                    vseed = int(self.gloss_seed) + 1009 * int(v)
                    s_gl = self.glosser.append_gloss(
                        base, max_items=self.gloss_max_items,
                        max_append_chars=self.gloss_max_append_chars,
                        seed=vseed, epoch=0, example_id=int(ex_int), keep_order=True,
                    )
                    flat_inputs.append(self.prefix + str(s_gl))
                    flat_exi.append(ex_i)

        lens = np.array([len(x.split()) for x in flat_inputs], dtype=np.int32)
        order = np.argsort(lens, kind="mergesort")
        n_inputs = int(len(order))
        pools = [[] for _ in range(len(ex_ids))]

        pbar = self._tqdm(n_inputs, "EVAL/generate")
        with self._bf16_eval_context():
            for a in range(0, n_inputs, self.mbr_batch_size_inputs):
                idx = order[a:a + self.mbr_batch_size_inputs]
                batch_in = [flat_inputs[i] for i in idx]
                batch_ex = [flat_exi[i] for i in idx]
                cand_lists = _generate_multi_decode(
                    model, tok, batch_in, device=device,
                    src_max_length=self.src_max_length, max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams, num_beam_cands=self.num_beam_cands,
                    length_penalty=self.length_penalty,
                    num_sample_cands=self.num_sample_cands,
                    temperature=self.temperature, top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                )
                for ex_i, cands in zip(batch_ex, cand_lists):
                    pools[int(ex_i)].extend(cands)
                if pbar is not None:
                    pbar.update(len(idx))
        if pbar is not None:
            pbar.close()

        pool_raw_mean = float(np.mean([len(p) for p in pools])) if pools else 0.0

        # TBM inject
        tbm_hit = 0
        tbm_added_total = 0
        if self.tbm_index is not None:
            for ex_i, base_src in enumerate(src_clean):
                try:
                    res = self.tbm_index.query(str(base_src), k=self.tbm_topk)
                except Exception:
                    res = None
                if not res:
                    continue
                added = 0
                if float(res[0][1]) >= float(self.tbm_hard_sim):
                    pools[ex_i].insert(0, res[0][0])
                    added = 1
                else:
                    for t, sim in res:
                        if float(sim) >= float(self.tbm_min_sim):
                            pools[ex_i].insert(0, t)
                            added += 1
                if added > 0:
                    tbm_hit += 1
                    tbm_added_total += added

        # Postprocess + cap
        flat_all, sizes = [], []
        for p in pools:
            p = _dedup_keep_order(p)
            if self.mbr_pool_cap is not None:
                p = p[: self.mbr_pool_cap]
            sizes.append(len(p))
            flat_all.extend(p)

        if self.post is not None:
            flat_all = self.post.postprocess_batch([str(x) for x in flat_all])
        flat_all = [_norm_ws(x) for x in flat_all]

        pools2, k = [], 0
        for sz in sizes:
            p = flat_all[k:k+sz]
            k += sz
            p = _dedup_keep_order(p)
            pools2.append(p)
        pools = pools2

        pool_final_mean = float(np.mean([len(p) for p in pools])) if pools else 0.0

        # MBR pick
        pb_mbr = self._tqdm(len(pools), "EVAL/MBR")
        preds = []
        gaps = []
        for p in pools:
            p2 = _dedup_keep_order(p)
            n = len(p2)
            if n <= 1:
                preds.append(p2[0] if n == 1 else "")
                gaps.append(0.0)
            else:
                sims = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    for j in range(i + 1, n):
                        s = _geo_sim_sentence(p2[i], p2[j])
                        sims[i, j] = s
                        sims[j, i] = s
                avg = (sims.sum(axis=1) - np.diag(sims)) / float(n - 1)
                jbest = int(np.argmax(avg))
                preds.append(p2[jbest])
                best = float(avg[jbest])
                tmp = avg.copy()
                tmp[jbest] = -1e9
                second = float(np.max(tmp))
                gaps.append(float(best - second))
            if pb_mbr is not None:
                pb_mbr.update(1)
        if pb_mbr is not None:
            pb_mbr.close()

        preds = [_norm_ws(x) for x in preds]
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
        chrf1 = float(sacrebleu.corpus_chrf(preds, [refs], word_order=0).score)

        out = {
            f"{metric_key_prefix}_bleu": float(bleu),
            f"{metric_key_prefix}_chrf1": float(chrf1),
            f"{metric_key_prefix}_chrfpp": float(chrfpp),
            f"{metric_key_prefix}_geo_mean": float(geo),
            f"{metric_key_prefix}_mbr_pool_mean": float(pool_final_mean),
            f"{metric_key_prefix}_pool_raw_mean": float(pool_raw_mean),
            f"{metric_key_prefix}_tbm_hit_rate": float(tbm_hit) / max(1, len(ex_ids)),
            f"{metric_key_prefix}_tbm_added_mean": float(tbm_added_total) / max(1, len(ex_ids)),
            f"{metric_key_prefix}_mbr_gap_mean": float(np.mean(gaps)) if len(gaps) else 0.0,
        }

        if self.post_ref is not None:
            refs_dbg = self.post_ref.postprocess_batch([str(x) for x in refs_raw])
            refs_dbg = [_norm_ws(r) for r in refs_dbg]
            geo_dbg, _, _ = _official_geo_mean(refs_dbg, preds)
            out[f"{metric_key_prefix}_geo_mean_refpost_dbg"] = float(geo_dbg)

        return out

def sanitize_generation_config_for_saving(model, *, default_num_beams=8, default_len_pen=1.0):
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


def _as_path_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        items = list(x)
    else:
        items = [x]
    out: List[str] = []
    for it in items:
        s = "" if it is None else str(it).strip()
        if not s:
            continue
        parts = re.split(r"[,\n]+", s)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def split_group_folds(df: pd.DataFrame, group_col: str, num_folds: int, seed: int, fold_index: int):
    if num_folds <= 1:
        raise ValueError("num_folds must be > 1")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}]")

    groups = df[group_col].astype(str)
    uniq = groups.unique()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)

    fold_map = {g: (i % int(num_folds)) for i, g in enumerate(uniq)}
    fold_ids = groups.map(fold_map)
    val_mask = fold_ids == int(fold_index)

    train_df = df.loc[~val_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    return train_df, val_df


# ================================================================
# Pretraining
# ================================================================
# ================================================================
# Training
# ================================================================

def run_training():

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    pre = OptimizedPreprocessor()

    # Log Config at start of training
    try:
        cfg_items = {
            k: getattr(Config, k)
            for k in dir(Config)
            if k.isupper() and not k.startswith("_")
        }
        print("[CONFIG]", json.dumps(cfg_items, indent=2, sort_keys=True, default=str), flush=True)
    except Exception as e:
        print(f"[CONFIG] Failed to serialize Config: {e}", flush=True)

    post_ref = VectorizedPostprocessor(
        aggressive=bool(getattr(Config, "AGGRESSIVE_POSTPROCESS_TARGETS", False))
    ) if getattr(Config, "POSTPROCESS_TARGETS", True) else None

    post_out = VectorizedPostprocessor(aggressive=True)

    HF_CACHE_DIR = str(getattr(Config, "HF_CACHE_DIR", "/local_nvme/hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

    NPROC  = int(getattr(Config, "NPROC", 16))
    MAP_BS = int(getattr(Config, "MAP_BATCH_SIZE", 2048))

    train_sentence_csv_paths = _as_path_list(getattr(Config, "TRAIN_SENTENCE_CSV", ""))
    train_sentence_csv_exists = [p for p in train_sentence_csv_paths if os.path.exists(p)]
    train_sentence_csv_missing = [p for p in train_sentence_csv_paths if not os.path.exists(p)]
    if train_sentence_csv_paths and not train_sentence_csv_exists:
        raise FileNotFoundError(
            f"TRAIN_SENTENCE_CSV provided but no files exist: {train_sentence_csv_paths}"
        )
    if train_sentence_csv_missing:
        print(f"[WARN] TRAIN_SENTENCE_CSV missing files ignored: {train_sentence_csv_missing}", flush=True)
    use_sentence_csv = bool(train_sentence_csv_exists)

    if use_sentence_csv:
        required = {"oare_id", "transliteration", "translation"}
        frames = []
        for p in train_sentence_csv_exists:
            df = pd.read_csv(p)
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"TRAIN_SENTENCE_CSV file '{p}' missing columns: {sorted(missing)}"
                )
            df = df.copy()
            df["oare_id"] = df["oare_id"].astype(str)
            df["is_extra"] = False
            if "pair_id" not in df.columns:
                df["pair_id"] = df["oare_id"].map(lambda x: f"{x}::full")
            frames.append(df)
        base_df = pd.concat(frames, ignore_index=True)
        print(
            f"[TRAIN_SENTENCE_CSV] files={len(train_sentence_csv_exists)} rows={len(base_df)}",
            flush=True,
        )
        all_clean = base_df
    else:
        comp_df = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv")).assign(is_extra=False)
        _ = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"))

        extras = []

        df = load_and_sanitize_parallel(Config.LARSEN_LETTERS_PATH).assign(is_extra=True, source="larsen")
        if "transliteration" in df.columns and "translation" in df.columns:
            df = df[["transliteration", "translation", "is_extra", "source"]].copy()
            df["transliteration"] = df["transliteration"].astype(str).map(normalize_external_transliteration)
            df["translation"]     = df["translation"].astype(str).map(normalize_external_translation)
            extras.append(df)

        df = load_and_sanitize_parallel(Config.HYBRID_CSV_PATH).assign(is_extra=True, source="hybrid")
        if "new_transliteration_sentence" in df.columns and "new_translation_sentence" in df.columns:
            df = df.rename(columns={"new_transliteration_sentence": "transliteration", "new_translation_sentence": "translation"})
        df = df[["transliteration", "translation", "is_extra", "source"]].copy()
        df["transliteration"] = df["transliteration"].astype(str).map(normalize_external_transliteration)
        df["translation"]     = df["translation"].astype(str).map(normalize_external_translation)
        extras.append(df)

        for p in range(1, 4):
            extra_train_path = os.path.join(str(Config.MANUAL_EXTRA_DIR), f"train{p}.csv")
            df = load_and_sanitize_parallel(extra_train_path).assign(is_extra=True, source=f"train{p}")
            if "new_transliteration_sentence" in df.columns and "new_translation_sentence" in df.columns:
                df = df.rename(columns={"new_transliteration_sentence": "transliteration", "new_translation_sentence": "translation"})
            df = df[["transliteration", "translation", "is_extra", "source"]].copy()
            df["transliteration"] = df["transliteration"].astype(str).map(normalize_external_transliteration)
            df["translation"]     = df["translation"].astype(str).map(normalize_external_translation)
            extras.append(df)

        extra_df = pd.concat(extras, ignore_index=True)
        extra_df = extra_df.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)
        extra_df["oare_id"] = [f"extra::{i}" for i in range(len(extra_df))]
        extra_df["pair_id"] = [f"extra::{i}::full" for i in range(len(extra_df))]

        bad_comp   = flag_incomplete(comp_df)
        comp_clean = comp_df.loc[~bad_comp].reset_index(drop=True)
        bad_extra   = flag_incomplete(extra_df)
        extra_clean = extra_df.loc[~bad_extra].reset_index(drop=True)

        print(f"[EXTRA] clean={len(extra_clean)}/{len(extra_df)}", flush=True)

        comp2 = comp_clean.copy()
        comp2["oare_id"] = comp2["oare_id"].astype(str)
        comp2["is_extra"] = False
        if "pair_id" not in comp2.columns:
            comp2["pair_id"] = comp2["oare_id"].astype(str).map(lambda x: f"{x}::full")

        extra2 = extra_clean.copy()
        extra2["oare_id"] = extra2["oare_id"].astype(str)
        extra2["is_extra"] = True
        if "pair_id" not in extra2.columns:
            extra2["pair_id"] = extra2["oare_id"].astype(str).map(lambda x: f"{x}::full")

        all_clean = pd.concat([comp2, extra2], ignore_index=True)

    num_folds = int(getattr(Config, "NUM_FOLDS", 0))
    fold_index = int(getattr(Config, "FOLD_INDEX", 0))
    if num_folds and num_folds > 1:
        train_split_df, val_split_df = split_group_folds(
            all_clean, "oare_id",
            num_folds=num_folds,
            seed=int(getattr(Config, "SEED", 42)),
            fold_index=fold_index,
        )
        print(
            f"[SPLIT_FOLDS] folds={num_folds} fold={fold_index} "
            f"train={len(train_split_df)} val={len(val_split_df)}",
            flush=True,
        )
    else:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=float(getattr(Config, "VAL_SIZE", 0.1)),
            random_state=int(getattr(Config, "SEED", 42)),
        )
        tr_idx, va_idx = next(gss.split(all_clean, groups=all_clean["oare_id"].astype(str)))

        train_split_df = all_clean.iloc[tr_idx].reset_index(drop=True)
        val_split_df   = all_clean.iloc[va_idx].reset_index(drop=True)

    tr_ids = set(train_split_df["oare_id"].astype(str).tolist())
    va_ids = set(val_split_df["oare_id"].astype(str).tolist())
    overlap = tr_ids & va_ids
    if overlap:
        raise ValueError(f"[SPLIT] LEAKAGE DETECTED: {len(overlap)} oare_id appear in both splits.")

    print(f"[SPLIT_ALL] train={len(train_split_df)} val={len(val_split_df)}", flush=True)

    dsd = DatasetDict({
        "train": Dataset.from_pandas(train_split_df, preserve_index=False),
        "val":   Dataset.from_pandas(val_split_df,   preserve_index=False),
    })

    def _set_meta(examples, view="original", is_sentence=False):
        n = len(examples["transliteration"])
        return {"src_view": [view]*n, "is_sentence": [bool(is_sentence)]*n}

    train_for_tokenize = dsd["train"].map(_set_meta, batched=True, fn_kwargs={"view":"original","is_sentence":False})
    val_for_tokenize   = dsd["val"].map(_set_meta,   batched=True, fn_kwargs={"view":"original","is_sentence":False})

    TRAIN_CSV_PATH = os.path.join(str(getattr(Config, "INPUT_DIR", "")), "train.csv")
    SENT_PATH      = str(getattr(Config, "SENTENCES_PATH", ""))

    if (not use_sentence_csv) and os.path.exists(SENT_PATH) and os.path.exists(TRAIN_CSV_PATH):
        raw_train_df = pd.read_csv(TRAIN_CSV_PATH)
        sent_df = pd.read_csv(SENT_PATH)

        sent_min = int(getattr(Config, "SENT_MIN_TOKENS", 3))
        sent_max = int(getattr(Config, "SRC_MAX_LENGTH", 512))

        comp_oare_ids = set(raw_train_df["oare_id"].astype(str).tolist()) if "oare_id" in raw_train_df.columns else set()
        allowed_tr = set(map(str, train_split_df["oare_id"].tolist())) & comp_oare_ids
        allowed_va = set(map(str, val_split_df["oare_id"].tolist())) & comp_oare_ids

        sent_train_df = build_sentence_pairs(raw_train_df, sent_df, allowed_text_ids=allowed_tr, min_tokens=sent_min, max_tokens=sent_max)
        sent_val_df = build_sentence_pairs(raw_train_df, sent_df, allowed_text_ids=allowed_va, min_tokens=sent_min, max_tokens=sent_max)

        if len(sent_train_df):
            ds_sent = Dataset.from_pandas(sent_train_df[["oare_id","transliteration","translation","pair_id"]], preserve_index=False)
            ds_sent = ds_sent.map(_set_meta, batched=True, fn_kwargs={"view":"original","is_sentence":True})
            train_for_tokenize = concatenate_datasets([train_for_tokenize, ds_sent])

        if len(sent_val_df):
            ds_sent = Dataset.from_pandas(sent_val_df[["oare_id","transliteration","translation","pair_id"]], preserve_index=False)
            ds_sent = ds_sent.map(_set_meta, batched=True, fn_kwargs={"view":"original","is_sentence":True})
            val_for_tokenize = concatenate_datasets([val_for_tokenize, ds_sent])

        print(f"[SENT] train_added={len(sent_train_df)} | val_added={len(sent_val_df)}", flush=True)

    train_for_tokenize = drop_duplicates_hf(train_for_tokenize, src_col="transliteration", tgt_col="translation", rule="tgt", keep="longest_src", normalize=True, report=True)
    val_for_tokenize = drop_duplicates_hf(val_for_tokenize, src_col="transliteration", tgt_col="translation", rule="tgt", keep="longest_src", normalize=True, report=True)

    train_for_tokenize = filter_incomplete_hf(train_for_tokenize, ratio_max=0.6, keep=False, batch_size=2048, num_proc=1)
    val_for_tokenize   = filter_incomplete_hf(val_for_tokenize,   ratio_max=0.6, keep=False, batch_size=2048, num_proc=1)

    drop_inc_cols = ["src_chars","tgt_chars","tgt_over_src","header_only","length_rule","flag"]
    train_for_tokenize = train_for_tokenize.remove_columns([c for c in drop_inc_cols if c in train_for_tokenize.column_names])
    val_for_tokenize   = val_for_tokenize.remove_columns([c for c in drop_inc_cols if c in val_for_tokenize.column_names])

    def _keep_val_len_cutoff(batch, thr):
        if bool(getattr(Config, "POSTPROCESS_TARGETS", True)) and (post_ref is not None):
            tgt_clean = post_ref.postprocess_batch(batch["translation"])
        else:
            tgt_clean = batch["translation"]
        return [len(str(s).split()) <= int(thr) for s in tgt_clean]

    before = len(val_for_tokenize)
    val_for_tokenize = val_for_tokenize.filter(_keep_val_len_cutoff, batched=True, fn_kwargs={"thr": int(getattr(Config, "VAL_CUTOFF_WORD_THR", 50))}, num_proc=1)
    print(f"[VAL_LEN_CAP] kept={len(val_for_tokenize)}/{before}", flush=True)

    def _add_ex_id(examples):
        pair = examples.get("pair_id", [None] * len(examples["oare_id"]))
        out = []
        for oid, pid in zip(examples["oare_id"], pair):
            out.append(f"{str(oid)}::full" if (pid is None or str(pid) == "nan") else str(pid))
        return {"ex_id": out}

    train_for_tokenize = train_for_tokenize.map(_add_ex_id, batched=True)
    val_for_tokenize   = val_for_tokenize.map(_add_ex_id,   batched=True)

    LEXICON_PATH     = str(getattr(Config, "LEXICON_PATH", ""))
    ONOMASTICON_PATH = str(getattr(Config, "ONOMASTICON_PATH", ""))
    EBL_DICT_PATH    = str(getattr(Config, "EBL_DICT_PATH", ""))

    class _IdentityCanonicalizer:
        def canonicalize_source(self, text: str, mode: str = "pn_norm") -> str:
            return text if isinstance(text, str) else ""

    class _NoopGlossAugmenter:
        def append_gloss(self, src_text: str, **kwargs) -> str:
            return src_text if isinstance(src_text, str) else ""

    train_texts = pre.preprocess_batch(list(train_for_tokenize["transliteration"]))

    pn_enabled = False
    if os.path.exists(LEXICON_PATH) and os.path.exists(ONOMASTICON_PATH):
        try:
            canon = SourceCanonicalizer.from_csvs(LEXICON_PATH, ONOMASTICON_PATH, use_norm=True)
            pn_enabled = True
        except Exception as e:
            print(f"[WARN] Failed to load canonicalizer, using identity fallback: {e}", flush=True)
            canon = _IdentityCanonicalizer()
    else:
        print("[WARN] Missing lexicon/onomasticon; PN canonicalization disabled.", flush=True)
        canon = _IdentityCanonicalizer()

    gloss_enabled = False
    if os.path.exists(LEXICON_PATH) and os.path.exists(EBL_DICT_PATH):
        try:
            glosser = GlossAugmenter(
                LEXICON_PATH,
                EBL_DICT_PATH,
                train_texts=train_texts,
                idf_cap=3.5,
                rare_df_floor=3,
                df1_penalty=0.65,
                base_weight=0.6,
            )
            gloss_enabled = True
        except Exception as e:
            print(f"[WARN] Failed to load glossary augmenter, using noop fallback: {e}", flush=True)
            glosser = _NoopGlossAugmenter()
    else:
        print("[WARN] Missing lexicon/dictionary; glossary augmentation disabled.", flush=True)
        glosser = _NoopGlossAugmenter()

    tokenized_train, tokenized_val, shared_epoch = build_probe_then_pngloss_variants(
        Config=Config, train_text_ds=train_for_tokenize, val_text_ds=val_for_tokenize,
        tokenizer=tokenizer, pre=pre, canon=canon, glosser=glosser, NPROC=NPROC, MAP_BS=MAP_BS,
        use_pn_view=pn_enabled, use_gloss_view=gloss_enabled,
    )

    tbm_pairs_df = train_split_df[["transliteration","translation"]].copy()

    warmup_steps = compute_warmup_steps(len(tokenized_train), Config.BATCH_SIZE, getattr(Config, "GRAD_ACCUM", 1), Config.EPOCHS, getattr(Config, "WARMUP_RATIO", 0.05))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
    grad_ckpt = bool(getattr(Config, "GRADIENT_CHECKPOINTING", False))
    if grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[WARN] Failed to enable gradient checkpointing: {e}", flush=True)
        try:
            model.config.use_cache = False
        except Exception:
            pass
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    sanitize_generation_config_for_saving(model, default_num_beams=int(getattr(Config, "NUM_BEAMS", 8)), default_len_pen=float(getattr(Config, "GEN_LENGTH_PENALTY", 1.0)))

    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=20, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model="eval_geo_mean", greater_is_better=True,
        bf16=use_bf16, fp16=False,
        per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=32,
        gradient_accumulation_steps=getattr(Config, "GRAD_ACCUM", 1),
        gradient_checkpointing=grad_ckpt,
        group_by_length=True, length_column_name="input_length",
        learning_rate=Config.LEARNING_RATE, weight_decay=0.01, max_grad_norm=1.0,
        num_train_epochs=Config.EPOCHS, lr_scheduler_type="cosine_with_restarts",
        warmup_steps=warmup_steps, prediction_loss_only=False,
        optim="adamw_torch_fused", label_smoothing_factor=Config.LABEL_SMOOTHING,
        predict_with_generate=False,
        dataloader_num_workers=16, dataloader_pin_memory=True, dataloader_persistent_workers=True,
        logging_strategy="steps", logging_steps=10, report_to=str(getattr(Config, "REPORT_TO", "none")),
    )

    gen_cfg = model.generation_config
    gen_cfg.repetition_penalty = float(getattr(Config, "GEN_REPETITION_PENALTY", 1.0))
    gen_cfg.no_repeat_ngram_size = int(getattr(Config, "GEN_NO_REPEAT_NGRAM", 0)) or 0
    model.generation_config = gen_cfg

    eval_placeholder = tokenized_val.select(range(min(64, len(tokenized_val)))) if len(tokenized_val) else tokenized_val

    trainer = MBRGlossSeq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized_train, eval_dataset=eval_placeholder,
        data_collator=data_collator, processing_class=tokenizer, compute_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-1)],
        val_text_ds=val_for_tokenize, pre=pre, prefix=Config.PREFIX,
        post=post_out, post_ref=None,
        glosser=glosser,
        gloss_variants=int(getattr(Config, "MBR_GLOSS_VARIANTS", 2)),
        gloss_seed=int(getattr(Config, "GLOSS_SEED", int(getattr(Config, "SEED", 42)) + 777)),
        gloss_max_items=int(getattr(Config, "GLOSS_MAX_ITEMS", 6)),
        gloss_max_append_chars=int(getattr(Config, "GLOSS_MAX_APPEND_CHARS", 240)),
        mbr_batch_size_inputs=int(getattr(Config, "MBR_BATCH_SIZE_INPUTS", 16)),
        src_max_length=Config.SRC_MAX_LENGTH, max_new_tokens=Config.GEN_MAX_NEW_TOKENS,
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
    )

    class _SetSharedEpochCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            try:
                shared_epoch.value = int(state.epoch or 0)
            except Exception:
                pass
            return control

    trainer.add_callback(_SetSharedEpochCallback())

    print("Starting Training...")
    trainer.train()

    print("\nBest checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric:", trainer.state.best_metric)

    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)

    CKPT_AVG_K = int(getattr(Config, "CKPT_AVG_K", 5))
    AVG_DIR = os.path.join(Config.OUTPUT_DIR, f"ckpt_avg_best{CKPT_AVG_K}")
    do_avg_eval = bool(getattr(Config, "EVAL_AVG_CHECKPOINT", True))

    avg_dir = None
    if trainer.is_world_process_zero():
        avg_dir, chosen = average_checkpoints_and_save(
            output_dir=Config.OUTPUT_DIR, save_dir=AVG_DIR,
            k=CKPT_AVG_K, metric_key="eval_geo_mean", prefer_best=True,
            base_ckpt_for_config=trainer.state.best_model_checkpoint,
            cleanup_checkpoints=bool(getattr(Config, "CKPT_AVG_CLEANUP", False)),
        )
        tokenizer.save_pretrained(avg_dir)
        print(f"Saved AVERAGED model to: {avg_dir}", flush=True)
        print(f"[CKPT_AVG] chosen checkpoints: {chosen}", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if do_avg_eval and trainer.is_world_process_zero():
        try:
            if avg_dir is None:
                avg_dir = AVG_DIR
            print("[AVG_EVAL] Loading averaged checkpoint and running validation generation...", flush=True)
            avg_sd = _load_state_dict_any(avg_dir, map_location="cpu")
            missing, unexpected = trainer.model.load_state_dict(avg_sd, strict=False)
            print(
                f"[AVG_EVAL] state_dict loaded: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
            trainer.model.eval()
            avg_prefix = str(getattr(Config, "EVAL_AVG_METRIC_PREFIX", "eval_avg"))
            avg_metrics = trainer._evaluate_mbr(
                metric_key_prefix=avg_prefix,
                save_dir_override=avg_dir,
                file_tag_override="avg",
            )
            avg_metrics_path = os.path.join(avg_dir, f"{avg_prefix}_metrics.json")
            with open(avg_metrics_path, "w", encoding="utf-8") as f:
                json.dump(avg_metrics, f, indent=2)
            print(f"[AVG_EVAL] metrics: {avg_metrics}", flush=True)
            print(f"[AVG_EVAL] wrote {avg_metrics_path}", flush=True)
        except Exception as e:
            print(f"[AVG_EVAL] failed: {e}", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()



# ============================================================
# 2. NEW MODULE: paste BEFORE the INFER block
# ============================================================
def _maybe_tqdm(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc)
    except Exception:
        return None
# --- Word-level MSA consensus (bioinformatics-inspired polishing) ---

def _char_bigram_sim(a: str, b: str) -> float:
    """Char-bigram Jaccard for fuzzy word matching in NW alignment."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    def _bg(s):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) > 1 else {s}
    ba, bb = _bg(a), _bg(b)
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union > 0 else 0.0


def _nw_word_align(
    ref_toks: list[str],
    hyp_toks: list[str],
    *,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap: float = -0.5,
    fuzzy_thr: float = 0.5,
) -> list[tuple]:
    """
    Needleman-Wunsch at word level with optional fuzzy partial credit.
    Returns list of (ref_pos|None, ref_tok|None, hyp_tok|None).
    """
    n, m = len(ref_toks), len(hyp_toks)

    # dp[i][j] = best score aligning ref[:i] to hyp[:j]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        bt[i][0] = 1  # up = deletion in hyp
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        bt[0][j] = 2  # left = insertion in hyp

    for i in range(1, n + 1):
        ri = ref_toks[i - 1]
        for j in range(1, m + 1):
            hj = hyp_toks[j - 1]
            if ri == hj:
                s = match
            elif _char_bigram_sim(ri, hj) >= fuzzy_thr:
                s = match * 0.5
            else:
                s = mismatch

            diag = dp[i - 1][j - 1] + s
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap

            if diag >= up and diag >= left:
                dp[i][j] = diag; bt[i][j] = 0
            elif up >= left:
                dp[i][j] = up; bt[i][j] = 1
            else:
                dp[i][j] = left; bt[i][j] = 2

    # traceback
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 0:
            aligned.append((i - 1, ref_toks[i - 1], hyp_toks[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or bt[i][j] == 1):
            aligned.append((i - 1, ref_toks[i - 1], None))
            i -= 1
        else:
            aligned.append((None, None, hyp_toks[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned
# Replace your existing msa_consensus function with this one.
# _char_bigram_sim, _nw_word_align, _maybe_tqdm stay the same.

def _build_ngram_counts(pool: list[str], max_n: int = 4) -> dict[tuple, int]:
    """
    Build n-gram frequency table from pool candidates.
    This is the "language model" derived from what the model actually generates.
    Analogous to k-mer spectrum in assembly.
    """
    counts: dict[tuple, int] = Counter()
    for cand in pool:
        toks = cand.split()
        for n in range(1, min(max_n, len(toks)) + 1):
            for i in range(len(toks) - n + 1):
                counts[tuple(toks[i:i+n])] += 1
    return counts


def _local_ngram_score(
    tokens: list[str],
    pos: int,
    ngram_counts: dict[tuple, int],
    max_n: int = 4,
) -> float:
    """
    Score how well position `pos` fits its context, using pool n-gram frequencies.
    Sums log(count+1) for all n-grams that include position `pos`.
    
    Higher = more candidates produced this exact n-gram context.
    """
    n_toks = len(tokens)
    score = 0.0
    for n in range(1, min(max_n, n_toks) + 1):
        # all n-grams that include position `pos`
        start_lo = max(0, pos - n + 1)
        start_hi = min(pos, n_toks - n)
        for start in range(start_lo, start_hi + 1):
            ng = tuple(tokens[start:start + n])
            cnt = ngram_counts.get(ng, 0)
            # weight higher n-grams more (they're more informative)
            w = float(n)
            score += w * math.log1p(cnt)
    return score


def msa_consensus(
    ref_text: str,
    pool: list[str],
    *,
    min_agreement: float = 0.35,
    min_pool: int = 3,
    tie_bias: int = 2,
    fuzzy_thr: float = 0.5,
    max_insert_len: int = 3,
    # --- n-gram aware params ---
    ngram_verify: bool = True,
    ngram_max_n: int = 4,
    ngram_min_gain: float = 0.0,  # accept change only if ngram score improves by at least this
    mbr_verify: bool = True,      # final check: only accept if pool-agreement doesn't drop
) -> str:
    """
    Word-level MSA consensus with n-gram-aware edit acceptance.

    Phase 1: Standard NW alignment + per-position voting (same as before)
    Phase 2: N-gram verification — for each proposed word change, check whether
             the surrounding n-grams have better support in the pool.
             Rejects edits that break high-frequency n-gram contexts.
    Phase 3: MBR verification — score final output against pool, reject if worse.

    The n-gram table acts as a "k-mer spectrum" — it captures which word sequences
    the model actually co-generates, not just individual word frequencies.
    """
    pool = [str(c).strip() for c in pool if str(c).strip()]
    ref_text = str(ref_text).strip()
    ref_toks = ref_text.split()

    if not ref_toks or len(pool) < int(min_pool):
        return ref_text

    n_ref = len(ref_toks)
    n_cands = len(pool)
    threshold = max(2, int(n_cands * float(min_agreement)))

    # ================================================================
    # Phase 1: Standard alignment + voting (unchanged logic)
    # ================================================================
    position_votes = [Counter() for _ in range(n_ref)]
    insert_counter = Counter()

    for cand in pool:
        cand_toks = cand.split()
        if not cand_toks:
            continue

        aligned = _nw_word_align(ref_toks, cand_toks, fuzzy_thr=float(fuzzy_thr))
        insert_buf = []

        for ref_pos, ref_tok, hyp_tok in aligned:
            if ref_pos is not None:
                if insert_buf and len(insert_buf) <= int(max_insert_len):
                    insert_counter[(ref_pos, tuple(insert_buf))] += 1
                insert_buf = []
                position_votes[ref_pos][hyp_tok] += 1
            else:
                insert_buf.append(hyp_tok)

        if insert_buf and len(insert_buf) <= int(max_insert_len):
            insert_counter[(n_ref, tuple(insert_buf))] += 1

    # ================================================================
    # Build proposed edits (what standard MSA would produce)
    # ================================================================
    # For each position, determine: keep ref, substitute, or delete
    proposed = []  # list of (action, token_or_None)
    #   action: "keep", "sub", "del"

    for i in range(n_ref):
        votes = position_votes[i]
        if not votes:
            proposed.append(("keep", ref_toks[i]))
            continue

        best_tok, best_count = votes.most_common(1)[0]

        if best_tok is None:
            # deletion candidate
            if best_count >= threshold:
                proposed.append(("del", None))
            else:
                proposed.append(("keep", ref_toks[i]))
        else:
            ref_count = votes.get(ref_toks[i], 0)
            if best_tok != ref_toks[i] and (best_count - ref_count) >= int(tie_bias):
                proposed.append(("sub", best_tok))
            else:
                proposed.append(("keep", ref_toks[i]))

    # Collect proposed insertions (same as before)
    proposed_inserts = {}  # pos -> list of tokens
    for i in range(n_ref + 1):
        best_ins = None
        best_ins_count = 0
        for (pos, toks), cnt in insert_counter.items():
            if pos == i and cnt > best_ins_count:
                best_ins = toks
                best_ins_count = cnt
        if best_ins is not None and best_ins_count >= threshold:
            proposed_inserts[i] = list(best_ins)

    # ================================================================
    # Phase 2: N-gram verification of each edit
    # ================================================================
    if ngram_verify and pool:
        ngram_counts = _build_ngram_counts(pool, max_n=int(ngram_max_n))

        # Check each substitution/deletion against n-gram context
        for i in range(n_ref):
            action, tok = proposed[i]

            if action == "keep":
                continue  # no change to verify

            # Build ref version (with ref token at position i)
            ref_version = []
            for j in range(n_ref):
                if j == i:
                    ref_version.append(ref_toks[j])
                else:
                    _, t = proposed[j]
                    if t is not None:
                        ref_version.append(t)
                    else:
                        ref_version.append(ref_toks[j])  # placeholder for scoring

            # Build proposed version
            prop_version = list(ref_version)
            if action == "sub":
                prop_version[i] = tok
            elif action == "del":
                # for scoring, treat deletion as removing the token
                # score both: with token present vs absent
                pass

            # Find position in the built sequence (accounting for earlier deletions)
            # Simpler: just score with token present in both versions at position i
            if action == "sub":
                ref_score = _local_ngram_score(ref_version, i, ngram_counts, max_n=int(ngram_max_n))
                prop_score = _local_ngram_score(prop_version, i, ngram_counts, max_n=int(ngram_max_n))

                if prop_score < ref_score + float(ngram_min_gain):
                    # n-gram context is worse -> revert to ref token
                    proposed[i] = ("keep", ref_toks[i])

            elif action == "del":
                # For deletion: compare n-grams with vs without the token
                with_tok = list(ref_version)  # has ref token at i
                without_tok = [t for j, t in enumerate(ref_version) if j != i]

                # Score the bigram/trigram spanning the deletion point
                if i > 0 and i < len(without_tok):
                    # n-grams that would bridge the gap
                    bridge_score = 0.0
                    for n in range(2, min(int(ngram_max_n), len(without_tok)) + 1):
                        start = max(0, i - n + 1)
                        end = min(i + 1, len(without_tok) - n + 1)
                        for s in range(start, end):
                            ng = tuple(without_tok[s:s+n])
                            bridge_score += float(n) * math.log1p(ngram_counts.get(ng, 0))

                    # n-grams with the token present
                    keep_score = _local_ngram_score(with_tok, i, ngram_counts, max_n=int(ngram_max_n))

                    if bridge_score < keep_score + float(ngram_min_gain):
                        proposed[i] = ("keep", ref_toks[i])

        # Verify insertions: check if inserted n-grams appear in pool
        inserts_to_remove = []
        for pos, ins_toks in proposed_inserts.items():
            # Build local context around insertion point
            before = []
            for j in range(max(0, pos - 2), pos):
                _, t = proposed[j]
                if t is not None:
                    before.append(t)

            after = []
            for j in range(pos, min(n_ref, pos + 2)):
                _, t = proposed[j]
                if t is not None:
                    after.append(t)

            # Check if the inserted phrase + context appears in pool n-grams
            test_seq = before + ins_toks + after
            if len(test_seq) >= 2:
                ins_support = 0.0
                for n in range(2, min(int(ngram_max_n), len(test_seq)) + 1):
                    for s in range(len(test_seq) - n + 1):
                        ng = tuple(test_seq[s:s+n])
                        ins_support += float(n) * math.log1p(ngram_counts.get(ng, 0))

                # Compare to context without insertion
                no_ins_seq = before + after
                no_ins_support = 0.0
                if len(no_ins_seq) >= 2:
                    for n in range(2, min(int(ngram_max_n), len(no_ins_seq)) + 1):
                        for s in range(len(no_ins_seq) - n + 1):
                            ng = tuple(no_ins_seq[s:s+n])
                            no_ins_support += float(n) * math.log1p(ngram_counts.get(ng, 0))

                if ins_support < no_ins_support + float(ngram_min_gain):
                    inserts_to_remove.append(pos)

        for pos in inserts_to_remove:
            del proposed_inserts[pos]

    # ================================================================
    # Emit final consensus
    # ================================================================
    result = []
    for i in range(n_ref):
        # insertions before position i
        if i in proposed_inserts:
            result.extend(proposed_inserts[i])

        action, tok = proposed[i]
        if action == "del":
            continue
        result.append(tok if tok is not None else ref_toks[i])

    # trailing insertions
    if n_ref in proposed_inserts:
        result.extend(proposed_inserts[n_ref])

    consensus = " ".join(result)

    # ================================================================
    # Phase 3: MBR verification (optional safety net)
    # ================================================================
    if mbr_verify and consensus != ref_text and pool:
        # Score both against pool using same geo_sim as MBR
        ref_pool_score = sum(_geo_sim_sentence(ref_text, c) for c in pool) / len(pool)
        con_pool_score = sum(_geo_sim_sentence(consensus, c) for c in pool) / len(pool)

        if con_pool_score < ref_pool_score:
            return ref_text

    # Safety fallback
    if not consensus.strip() or len(consensus.split()) < len(ref_toks) * 0.3:
        return ref_text

    return consensus
# ================================================================
# INFER MBR
# ================================================================

def run_inference_mbr_and_make_submission(
    model_dir, test_csv_path, output_csv_path, *,
    prefix="translate Akkadian to English: ",
    src_max_length=512, max_new_tokens=256,
    num_beams=4, num_beam_cands=1, num_sample_cands=8,
    length_penalty=1.0, temperature=0.7, top_p=0.9,
    repetition_penalty=1.1, no_repeat_ngram_size=0,
    batch_size_inputs=16, use_bucket_sort=True,
    use_gloss=True, gloss_src_word_thr=160,
    gloss_variants=1, gloss_seed=12345,
    gloss_max_items=6, gloss_max_append_chars=240,
    pool_cap=32, show_progress=True, verbose_every_batches=20,
    device=None,
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pre = OptimizedPreprocessor()
    post = VectorizedPostprocessor()

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    mdl.eval()
    try:
        mdl.config.use_cache = True
    except Exception:
        pass

    test_df = pd.read_csv(test_csv_path)

    # PN
    canon = None
    lex_path = str(getattr(Config, "LEXICON_PATH", ""))
    ono_path = str(getattr(Config, "ONOMASTICON_PATH", ""))
    ebl_path = str(getattr(Config, "EBL_DICT_PATH", ""))

    if lex_path and ono_path and os.path.exists(lex_path) and os.path.exists(ono_path):
        try:
            canon = SourceCanonicalizer.from_csvs(lex_path, ono_path, use_norm=True)
        except Exception:
            canon = None

    # TBM bank
    TBM_ENABLE    = bool(getattr(Config, "TBM_ENABLE", True))
    TBM_TOPK      = int(getattr(Config, "TBM_TOPK", 3))
    TBM_MIN_SIM   = float(getattr(Config, "TBM_MIN_SIM", 0.90))
    TBM_HARD_SIM  = float(getattr(Config, "TBM_HARD_SIM", 0.995))
    TBM_NGRAM_MIN = int(getattr(Config, "TBM_NGRAM_MIN", 3))
    TBM_NGRAM_MAX = int(getattr(Config, "TBM_NGRAM_MAX", 6))
    TBM_MAX_FEATURE = int(getattr(Config, "TBM_MAX_FEATURES", 250_000))

    tbm_parts = []
    comp_train_csv = os.path.join(str(getattr(Config, "INPUT_DIR", "")), "train.csv")

    if os.path.exists(comp_train_csv):
        tr = pd.read_csv(comp_train_csv)
        if "transliteration" in tr.columns and "translation" in tr.columns:
            tbm_parts.append(tr[["transliteration", "translation"]].copy())

    SENT_PATH = str(getattr(Config, "SENTENCES_PATH", ""))
    if os.path.exists(comp_train_csv) and SENT_PATH and os.path.exists(SENT_PATH):
        try:
            raw_train_df = pd.read_csv(comp_train_csv)
            sent_df_infer = pd.read_csv(SENT_PATH)
            sp = build_sentence_pairs(raw_train_df, sent_df_infer, allowed_text_ids=None, min_tokens=int(getattr(Config, "SENT_MIN_TOKENS", 3)), max_tokens=int(getattr(Config, "SRC_MAX_LENGTH", 512)))
            if len(sp):
                tbm_parts.append(sp[["transliteration", "translation"]].copy())
        except Exception:
            pass

    def _add_extra_df(df, tag):
        if df is None or len(df) == 0:
            return
        if "new_transliteration_sentence" in df.columns and "new_translation_sentence" in df.columns:
            df = df.rename(columns={"new_transliteration_sentence": "transliteration", "new_translation_sentence": "translation"})
        if not ("transliteration" in df.columns and "translation" in df.columns):
            return
        df = df[["transliteration", "translation"]].copy()
        df["transliteration"] = df["transliteration"].astype(str).map(normalize_external_transliteration)
        df["translation"]     = df["translation"].astype(str).map(normalize_external_translation)
        tbm_parts.append(df)

    try:
        if hasattr(Config, "LARSEN_LETTERS_PATH") and os.path.exists(Config.LARSEN_LETTERS_PATH):
            _add_extra_df(load_and_sanitize_parallel(Config.LARSEN_LETTERS_PATH), "larsen")
    except Exception:
        pass
    try:
        if hasattr(Config, "HYBRID_CSV_PATH") and os.path.exists(Config.HYBRID_CSV_PATH):
            _add_extra_df(load_and_sanitize_parallel(Config.HYBRID_CSV_PATH), "hybrid")
    except Exception:
        pass
    for p in range(1, 4):
        try:
            path = os.path.join(str(getattr(Config, "MANUAL_EXTRA_DIR", "")), f"train{p}.csv")
            if os.path.exists(path):
                _add_extra_df(load_and_sanitize_parallel(path), f"train{p}")
        except Exception:
            pass

    tbm_pairs_df = None
    if tbm_parts:
        tbm_pairs_df = pd.concat(tbm_parts, ignore_index=True)
        tbm_pairs_df["transliteration"] = tbm_pairs_df["transliteration"].astype(str).map(normalize_external_transliteration)
        tbm_pairs_df["translation"]     = tbm_pairs_df["translation"].astype(str).map(normalize_external_translation)
        tbm_pairs_df = tbm_pairs_df.dropna(subset=["transliteration", "translation"])
        tbm_pairs_df = tbm_pairs_df[tbm_pairs_df["transliteration"].str.strip().astype(bool) & tbm_pairs_df["translation"].str.strip().astype(bool)]
        bad = flag_incomplete(tbm_pairs_df)
        tbm_pairs_df = tbm_pairs_df.loc[~bad].reset_index(drop=True)
        tbm_pairs_df = tbm_pairs_df.drop_duplicates(subset=["transliteration", "translation"], keep="first").reset_index(drop=True)

    if show_progress:
        print(f"[TBM] bank_total_pairs={0 if tbm_pairs_df is None else len(tbm_pairs_df)}", flush=True)

    # Glosser
    glosser = None
    if use_gloss and lex_path and ebl_path and os.path.exists(lex_path) and os.path.exists(ebl_path):
        try:
            if tbm_pairs_df is not None and len(tbm_pairs_df):
                train_texts_all = pre.preprocess_batch(tbm_pairs_df["transliteration"].astype(str).tolist())
                train_texts_all = list(dict.fromkeys([t for t in train_texts_all if str(t).strip()]))
                glosser = GlossAugmenter(lex_path, ebl_path, train_texts=train_texts_all)
            else:
                glosser = GlossAugmenter(lex_path, ebl_path)
        except Exception:
            glosser = None

    # TBM index
    tbm_index = None
    if TBM_ENABLE and tbm_pairs_df is not None and len(tbm_pairs_df):
        try:
            tbm_index = build_tbm_from_pairs(pre, tbm_pairs_df, ngram=(TBM_NGRAM_MIN, TBM_NGRAM_MAX), max_features=TBM_MAX_FEATURE)
        except Exception:
            tbm_index = None

    pre_texts = pre.preprocess_batch(test_df["transliteration"].astype(str).tolist())
    word_lens = [len(str(t).split()) for t in pre_texts]
    N = len(pre_texts)

    pn_texts = None
    if canon is not None:
        pn_texts = [canon.canonicalize_source(str(t), mode="pn_norm") for t in pre_texts]

    # Build flat inputs
    flat_inputs, flat_exi, flat_lens = [], [], []
    thr = int(gloss_src_word_thr)
    K = int(gloss_variants)
    gseed = int(gloss_seed)

    for ex_i, base0 in enumerate(pre_texts):
        base = str(base0)
        try:
            exid = int(test_df["id"].iloc[ex_i])
        except Exception:
            exid = int(ex_i)
        ex_int = _stable_int_id(exid)

        flat_inputs.append(prefix + base)
        flat_exi.append(ex_i)
        flat_lens.append(word_lens[ex_i])

        if pn_texts is not None:
            pn = str(pn_texts[ex_i])
            if pn and pn != base:
                flat_inputs.append(prefix + pn)
                flat_exi.append(ex_i)
                flat_lens.append(len(pn.split()))

        if (glosser is not None) and (K > 0) and (word_lens[ex_i] <= thr):
            for v in range(K):
                vseed = int(gseed) + 1009 * int(v)
                s_gl = glosser.append_gloss(base, max_items=int(gloss_max_items), max_append_chars=int(gloss_max_append_chars), seed=int(vseed), epoch=0, example_id=int(ex_int), keep_order=True)
                s_gl = str(s_gl)
                if s_gl and s_gl != base:
                    flat_inputs.append(prefix + s_gl)
                    flat_exi.append(ex_i)
                    flat_lens.append(len(s_gl.split()))

    n_inputs = len(flat_inputs)
    if show_progress:
        print(f"[INFER_MBR] n_examples={N} | flat_inputs={n_inputs}", flush=True)

    pools = [[] for _ in range(N)]
    order = np.argsort(np.array(flat_lens, dtype=np.int32), kind="mergesort") if use_bucket_sort else np.arange(n_inputs, dtype=np.int32)

    t_gen0 = time.time()
    pbar = tqdm(total=n_inputs, desc="INFER generate") if show_progress else None

    for bi, a in enumerate(range(0, n_inputs, int(batch_size_inputs))):
        idx = order[a:a + int(batch_size_inputs)]
        batch_in = [flat_inputs[i] for i in idx]
        batch_ex = [flat_exi[i] for i in idx]

        cand_lists = _generate_multi_decode(
            mdl, tok, batch_in, device=mdl.device,
            src_max_length=int(src_max_length), max_new_tokens=int(max_new_tokens),
            num_beams=int(num_beams), num_beam_cands=int(num_beam_cands),
            length_penalty=float(length_penalty),
            num_sample_cands=int(num_sample_cands),
            temperature=float(temperature), top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size) or 0,
        )

        for ex_i, cands in zip(batch_ex, cand_lists):
            pools[int(ex_i)].extend(cands)

        if pbar is not None:
            pbar.update(len(idx))

    if pbar is not None:
        pbar.close()

    # TBM inject
    if tbm_index is not None:
        for i, base_src in enumerate(pre_texts):
            try:
                res = tbm_index.query(str(base_src), k=TBM_TOPK)
            except Exception:
                continue
            if not res:
                continue
            if float(res[0][1]) >= TBM_HARD_SIM:
                pools[i].insert(0, res[0][0])
            else:
                for t, sim in res:
                    if float(sim) >= TBM_MIN_SIM:
                        pools[i].insert(0, t)

    # Postprocess + cap
    flat_all, sizes = [], []
    for p in pools:
        p = _dedup_keep_order(p)
        if pool_cap is not None:
            p = p[: int(pool_cap)]
        sizes.append(len(p))
        flat_all.extend(p)

    flat_all = post.postprocess_batch([str(x) for x in flat_all])
    flat_all = [_norm_ws(x) for x in flat_all]

    pools2, k = [], 0
    for sz in sizes:
        pools2.append(flat_all[k:k+sz])
        k += sz
    pools = pools2

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#
# REPLACE IT with:
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ---- MBR select + MSA polish ----
    MSA_ON       = bool(getattr(Config, "MSA_ENABLE", True))
    MSA_MIN_POOL = int(getattr(Config, "MSA_MIN_POOL", 3))
    MSA_MIN_AGR  = float(getattr(Config, "MSA_MIN_AGREEMENT", 0.35))
    MSA_BIAS     = int(getattr(Config, "MSA_TIE_BIAS", 2))
    MSA_FUZZY    = float(getattr(Config, "MSA_FUZZY_THR", 0.5))
    MSA_MAX_INS  = int(getattr(Config, "MSA_MAX_INSERT_LEN", 3))

    t_mbr0 = time.time()
    pbar2 = _maybe_tqdm(total=N, desc="INFER MBR+MSA") if show_progress else None

    preds = []
    n_polished = 0
    for i, p in enumerate(pools):
        mbr_winner = _mbr_pick_geo(p)

        if MSA_ON and len(p) >= MSA_MIN_POOL:
            polished = msa_consensus(
                mbr_winner, p,
                min_agreement=MSA_MIN_AGR,
                min_pool=MSA_MIN_POOL,
                tie_bias=MSA_BIAS,
                fuzzy_thr=MSA_FUZZY,
                max_insert_len=MSA_MAX_INS,
            )
            if polished != mbr_winner:
                n_polished += 1
            mbr_winner = polished

        preds.append(mbr_winner)
        if pbar2 is not None:
            pbar2.update(1)
    if pbar2 is not None:
        pbar2.close()

    if show_progress:
        print(
            f"[INFER] MBR+MSA done in {time.time()-t_mbr0:.2f}s"
            f" | MSA polished {n_polished}/{N} examples",
            flush=True,
        )

    preds = post.postprocess_batch(preds)
    preds = [str(p).strip() for p in preds]

    sub = pd.DataFrame({"id": test_df["id"].values, "translation": preds})
    sub.to_csv(output_csv_path, index=False)
    print("Wrote:", output_csv_path, flush=True)
    print(sub.head(), flush=True)
    return sub



# ================================================================
# CLI
# ================================================================

def _coerce_value(old_val, new_val: str):
    if old_val is None:
        return new_val
    if isinstance(old_val, bool):
        return new_val.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(old_val, int) and not isinstance(old_val, bool):
        return int(new_val)
    if isinstance(old_val, float):
        return float(new_val)
    return new_val


def _apply_overrides(cfg, overrides):
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Config has no attribute: {k}")
        old = getattr(cfg, k)
        setattr(cfg, k, _coerce_value(old, v))


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Train ByT5 (self-contained, from notebook)")
    p.add_argument("--input-dir", help="Overrides Config.INPUT_DIR")
    p.add_argument("--output-dir", help="Overrides Config.OUTPUT_DIR")
    p.add_argument("--model-name", help="Overrides Config.MODEL_NAME")
    p.add_argument("--hf-cache-dir", help="Overrides Config.HF_CACHE_DIR")
    p.add_argument("--dpc-extra-dir", help="Overrides Config.DPC_EXTRA_DIR")
    p.add_argument("--manual-extra-dir", help="Overrides Config.MANUAL_EXTRA_DIR")
    p.add_argument(
        "--train-sentence-csv",
        nargs="+",
        help="Overrides Config.TRAIN_SENTENCE_CSV with one or more csv paths.",
    )
    p.add_argument("--lexicon-path", help="Overrides Config.LEXICON_PATH")
    p.add_argument("--onomasticon-path", help="Overrides Config.ONOMASTICON_PATH")
    p.add_argument("--ebl-dict-path", help="Overrides Config.EBL_DICT_PATH")
    p.add_argument("--sentences-path", help="Overrides Config.SENTENCES_PATH")
    p.add_argument("--hybrid-csv-path", help="Overrides Config.HYBRID_CSV_PATH")

    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    p.set_defaults(gradient_checkpointing=None)
    p.add_argument("--label-smoothing", type=float)
    p.add_argument("--warmup-ratio", type=float)
    p.add_argument("--val-size", type=float)
    p.add_argument("--num-folds", type=int)
    p.add_argument("--fold-index", type=int)
    p.add_argument("--report-to", help="Overrides Config.REPORT_TO (e.g. wandb)")
    p.add_argument("--wandb-project", help="Sets WANDB_PROJECT env var")
    p.add_argument("--wandb-run-name", help="Sets WANDB_RUN_NAME env var")
    p.add_argument("--src-max-length", type=int)
    p.add_argument("--tgt-max-length", type=int)
    p.add_argument("--gen-max-new-tokens", type=int)
    p.add_argument("--num-beams", type=int)
    p.add_argument("--nproc", type=int)
    p.add_argument("--map-batch-size", type=int)

    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Arbitrary Config override: --set KEY=VALUE",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()

    overrides = {}
    if args.input_dir:
        overrides["INPUT_DIR"] = args.input_dir
    if args.output_dir:
        overrides["OUTPUT_DIR"] = args.output_dir
    if args.model_name:
        overrides["MODEL_NAME"] = args.model_name
    if args.hf_cache_dir:
        overrides["HF_CACHE_DIR"] = args.hf_cache_dir
    if args.dpc_extra_dir:
        overrides["DPC_EXTRA_DIR"] = args.dpc_extra_dir
    if args.manual_extra_dir:
        overrides["MANUAL_EXTRA_DIR"] = args.manual_extra_dir
    if args.train_sentence_csv:
        overrides["TRAIN_SENTENCE_CSV"] = args.train_sentence_csv
    if args.lexicon_path:
        overrides["LEXICON_PATH"] = args.lexicon_path
    if args.onomasticon_path:
        overrides["ONOMASTICON_PATH"] = args.onomasticon_path
    if args.ebl_dict_path:
        overrides["EBL_DICT_PATH"] = args.ebl_dict_path
    if args.sentences_path:
        overrides["SENTENCES_PATH"] = args.sentences_path
    if args.hybrid_csv_path:
        overrides["HYBRID_CSV_PATH"] = args.hybrid_csv_path

    if args.seed is not None:
        overrides["SEED"] = str(args.seed)
    if args.epochs is not None:
        overrides["EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        overrides["BATCH_SIZE"] = str(args.batch_size)
    if args.grad_accum is not None:
        overrides["GRAD_ACCUM"] = str(args.grad_accum)
    if args.learning_rate is not None:
        overrides["LEARNING_RATE"] = str(args.learning_rate)
    if args.gradient_checkpointing is not None:
        overrides["GRADIENT_CHECKPOINTING"] = str(args.gradient_checkpointing)
    if args.label_smoothing is not None:
        overrides["LABEL_SMOOTHING"] = str(args.label_smoothing)
    if args.warmup_ratio is not None:
        overrides["WARMUP_RATIO"] = str(args.warmup_ratio)
    if args.val_size is not None:
        overrides["VAL_SIZE"] = str(args.val_size)
    if args.num_folds is not None:
        overrides["NUM_FOLDS"] = str(args.num_folds)
    if args.fold_index is not None:
        overrides["FOLD_INDEX"] = str(args.fold_index)
    if args.report_to:
        overrides["REPORT_TO"] = str(args.report_to)
    if args.src_max_length is not None:
        overrides["SRC_MAX_LENGTH"] = str(args.src_max_length)
    if args.tgt_max_length is not None:
        overrides["TGT_MAX_LENGTH"] = str(args.tgt_max_length)
    if args.gen_max_new_tokens is not None:
        overrides["GEN_MAX_NEW_TOKENS"] = str(args.gen_max_new_tokens)
    if args.num_beams is not None:
        overrides["NUM_BEAMS"] = str(args.num_beams)
    if args.nproc is not None:
        overrides["NPROC"] = str(args.nproc)
    if args.map_batch_size is not None:
        overrides["MAP_BATCH_SIZE"] = str(args.map_batch_size)

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    _apply_overrides(Config, overrides)

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = str(args.wandb_project)
    if args.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = str(args.wandb_run_name)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_training()


if __name__ == "__main__":
    main()
