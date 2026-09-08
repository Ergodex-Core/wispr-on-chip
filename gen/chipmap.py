"""Chip memory map shared by gen/emit_microcode.py and the tests (mirrors ChipMap / KVRegion in
MiniCPMConfig.scala and KVCache.scala). Activation banks are addressed in 256-bit words, KV regions
in 2048-bit words. Everything is a function of (max_ctx, chunk) so a small program can be emitted for
the layer-level Verilator test alongside the full 42-layer one."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

D_MODEL, N_HEAD, N_KV_HEAD, HEAD_DIM, D_FF = 2048, 16, 2, 128, 6144
KV_DIM = N_KV_HEAD * HEAD_DIM
MAX_CTX = 2048            # default (full) configuration
CHUNK = 512
N_LAYERS = 42
HEAD_TILES = HEAD_DIM // 32

# words per row in each format (256-bit words)
D32, D16, D8, FF16, FF8, KV16 = D_MODEL // 8, D_MODEL // 16, D_MODEL // 32, D_FF // 16, D_FF // 32, KV_DIM // 16

# activation banks: id -> name (sizes from bank_words)
BANK_NAMES = ["X", "A8", "QK16", "Q8", "T32", "G16", "U16", "A8X"]
ROWFAC_BANKS = [1, 7]


def bank_words(max_ctx: int = MAX_CTX, chunk: int = CHUNK) -> list[int]:
    """0 X    int32 residual, all positions (256 words/row)
       1 A8   int8 norm / attention-out rows, chunk-local (64 words/row)
       2 QK16 int16 q rows @0 (128 words/row), int16 k rows @k16_base (16 words/row)
       3 Q8   int8 rotated q rows
       4 T32  scratch: int16 attention out (128 words/row) / int32 o out, down out (256 words/row)
       5 G16  int16 gate rows (384 words/row)   6 U16 int16 up rows   7 A8X int8 gated rows (192 words/row)"""
    return [D32 * max_ctx, D8 * chunk, D16 * chunk + KV16 * chunk, D8 * chunk,
            D32 * chunk, FF16 * chunk, FF16 * chunk, FF8 * chunk]


def k16_base(chunk: int = CHUNK) -> int:
    return D16 * chunk


# KV cache regions (must mirror KVCache.scala): [L0 K][L0 V][L1 K][L1 V]...
def per_head(max_ctx: int = MAX_CTX) -> int:
    return HEAD_TILES * (max_ctx // 32) * 4      # words per (layer, kv head), K and V alike


def kv_kbase(layer: int, max_ctx: int = MAX_CTX) -> int:
    return layer * 2 * N_KV_HEAD * per_head(max_ctx)


def kv_vbase(layer: int, max_ctx: int = MAX_CTX) -> int:
    return kv_kbase(layer, max_ctx) + N_KV_HEAD * per_head(max_ctx)


def kv_words(kv_layers: int, max_ctx: int = MAX_CTX) -> int:
    return kv_kbase(kv_layers, max_ctx)


SPACES = {"w8": 2048, "i32vec": 1024, "i8mat": 256, "i16mat": 256}
SPACE_OF = {"w8": "w", "i32vec": "p", "i8mat": "t", "i16mat": "t"}


def tensor_bases(manifest=None) -> dict[str, int]:
    """Word base of every tensor in its space (same rule as gen/emit_weights.py: sorted by name)."""
    man = manifest or json.load(open(REPO / "weights" / "MANIFEST.json"))
    T = man["tensors"]
    bases = {"w": 0, "p": 0, "t": 0}
    out = {}
    for name in sorted(T):
        t = T[name]
        sp = SPACE_OF[t["kind"]]
        out[name] = bases[sp]
        bases[sp] += t["depth"]
    return out
