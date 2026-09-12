"""Chip memory map shared by gen/emit_microcode.py, tests/e2e and the Scala config (mirrored constants).
Activation banks are addressed in 256-bit words. KV regions in 2048-bit words."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# activation banks: id -> (name, words256)
BANKS = {
    0: ("MEL", 9216),     # 3000 frames x 3 words (96 int8)
    1: ("A8", 36864),     # int8 rows (12 words): conv1-gelu (3000 rows), LN outputs; decoder EM8@0, AD8@32
    2: ("X", 36864),      # int16 residual (24 words), 1500 rows; decoder XD@0
    3: ("Q8", 18432),     # int8 q rows; decoder QD8@0
    4: ("T16", 36864),    # int16 scratch (24 words): conv1 out chunk / attention out / o+fc2 out; dec TD16@0, OD16@64
    5: ("H16", 49152),    # int16 FFN hidden chunk (96 words x 512 rows); dec HD16@0
    6: ("A8X", 24576),    # int8 FFN hidden chunk (48 words x 512 rows); dec AD8X@0
    7: ("E8", 18432),     # encoder output int8 rows (persistent through decoding)
}
ROWFAC_BANKS = [1, 6, 7]    # banks holding dynamically-quantised int8 rows
ROWS_MAX = 4096             # rowfac table depth
CONV1_CHUNK = 1536          # rows per conv1 chunk (== accRows)
FFN_CHUNK = 512             # rows per FFN chunk
DEC = dict(EM8=(1, 0), AD8=(1, 32), XD=(2, 0), QD8=(3, 0), TD16=(4, 0), OD16=(4, 64), HD16=(5, 0), AD8X=(6, 0))

# KV cache regions (must mirror KVCache.scala)
ENC_KEYS, DEC_KEYS, H = 1536, 448, 6
PH_ENC = 2 * (ENC_KEYS // 32) * 4     # 384 words per head
PH_DEC = 2 * (DEC_KEYS // 32) * 4     # 112
KV = {}
KV["encSelfK"] = 0
KV["encSelfV"] = KV["encSelfK"] + H * PH_ENC
KV["decSelfK"] = KV["encSelfV"] + H * PH_ENC
KV["decSelfV"] = KV["decSelfK"] + 4 * H * PH_DEC
KV["crossK"] = KV["decSelfV"] + 4 * H * PH_DEC
KV["crossV"] = KV["crossK"] + 4 * H * PH_ENC
KV["words"] = KV["crossV"] + 4 * H * PH_ENC


def tensor_bases() -> dict[str, int]:
    """Word base of every tensor in its kind's address space (same rule as gen/emit_weights.py)."""
    man = json.load(open(REPO / "weights" / "MANIFEST.json"))
    T = man["tensors"]
    bases = {"w8": 0, "i32vec": 0, "i8mat": 0}
    out = {}
    for name in sorted(T):
        t = T[name]
        out[name] = bases[t["kind"]]
        bases[t["kind"]] += t["depth"]
    return out
