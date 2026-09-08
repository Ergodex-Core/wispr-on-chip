"""Paths and the small text corpora used for calibration and evaluation (no external datasets)."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_CACHE = Path(os.environ.get("MINICPM_SI_DATA", "/home/user/data-cache"))
MODEL_DIR = DATA_CACHE / "minicpm5-2b"
CHECKPOINT = MODEL_DIR / "model-00000-of-00001.safetensors"
TOKENIZER = MODEL_DIR / "tokenizer.json"
CONFIG = MODEL_DIR / "config.json"

BOS, EOS, IM_START, IM_END = 0, 1, 130072, 130073
EOS_IDS = (EOS, IM_END)


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))


def chat_ids(tok, user: str, think: bool = False) -> list[int]:
    """<s><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n[<think>\n\n</think>\n\n]  (the model's chat template,
    no tools; think=False appends the empty thinking block the template emits for enable_thinking=False)."""
    # encode the pieces separately so the special tokens are exact
    ids = [BOS, IM_START] + tok.encode("user\n", add_special_tokens=False).ids + tok.encode(user, add_special_tokens=False).ids
    ids += [IM_END] + tok.encode("\n", add_special_tokens=False).ids + [IM_START] + tok.encode("assistant\n", add_special_tokens=False).ids
    if not think:
        ids += tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False).ids
    return ids


def text_ids(tok, text: str, max_len: int | None = None) -> list[int]:
    ids = [BOS] + tok.encode(text, add_special_tokens=False).ids
    return ids[:max_len] if max_len else ids


def prompts(kind: str) -> list[dict]:
    """kind: 'calib' or 'eval' -> [{name, text, mode}] from data/prompts.json."""
    return json.load(open(REPO / "data" / "prompts.json"))[kind]
