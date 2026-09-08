"""Chip memory map shared by gen/emit_microcode.py and the tests (mirrors ChipMap / KVRegion in
MiniCPMConfig.scala and KVCache.scala). Activation banks are addressed in 256-bit words, KV regions
in 2048-bit words."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

D_MODEL, N_HEAD, N_KV_HEAD, HEAD_DIM, D_FF = 2048, 16, 2, 128, 6144
KV_DIM = N_KV_HEAD * HEAD_DIM
MAX_CTX = 2048
CHUNK = 512
N_LAYERS = 42
HEAD_TILES = HEAD_DIM // 32

D32, D16, D8, FF16, FF8, KV16 = D_MODEL // 8, D_MODEL // 16, D_MODEL // 32, D_FF // 16, D_FF // 32, KV_DIM // 16
K16_BASE = D16 * CHUNK
# activation banks: id -> (name, words256)
BANKS = {
    0: ("X", D32 * MAX_CTX),                 # int32 residual, all positions (256 words/row)
    1: ("A8", D8 * CHUNK),                   # int8 norm / attention-out rows, chunk-local (64 words/row)
    2: ("QK16", D16 * CHUNK + KV16 * CHUNK), # int16 q rows @0 (128 words/row), k rows @K16_BASE (16 words/row)
    3: ("Q8", D8 * CHUNK),                   # int8 rotated q rows
    4: ("T32", D32 * CHUNK),                 # scratch: int16 attention out (128 words/row) / int32 o out, down out (256 words/row)
    5: ("G16", FF16 * CHUNK),                # int16 gate rows (384 words/row)
    6: ("U16", FF16 * CHUNK),                # int16 up rows
    7: ("A8X", FF8 * CHUNK),                 # int8 gated rows (192 words/row)
}
BANK_WORDS = [BANKS[i][1] for i in range(8)]
ROWFAC_BANKS = [1, 7]

# KV cache regions (must mirror KVCache.scala): [L0 K][L0 V][L1 K][L1 V]...
PER_HEAD = HEAD_TILES * (MAX_CTX // 32) * 4      # 1024 words per (layer, kv head), K and V alike


def kv_kbase(layer: int) -> int:
    return layer * 2 * N_KV_HEAD * PER_HEAD


def kv_vbase(layer: int) -> int:
    return kv_kbase(layer) + N_KV_HEAD * PER_HEAD


def kv_words(kv_layers: int) -> int:
    return kv_kbase(kv_layers)


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
