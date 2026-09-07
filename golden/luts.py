"""Every lookup table in the design. Emitted verbatim into Scala by gen/emit_luts.py."""
from __future__ import annotations

import math

import numpy as np

# ---- exp LUT: p = exp(-d/256) in Q1.15 for d in [0, 4096); index i = d>>4 (256 entries), linear interp on d&15
EXP_D_BITS = 12          # d in [0, 4096): 4 integer bits (0..16 nats) + 8 fractional bits
EXP_FRAC_BITS = 8
EXP_LUT_BITS = 8         # 256 entries
EXP_INTERP_BITS = 4
EXP_OUT_ONE = 1 << 15    # Q1.15


def exp_table() -> list[int]:
    """T[i] = round(exp(-i/16) * 2^15) for i in 0..255; T[256] := 0 implicitly."""
    return [int(round(math.exp(-i / 16.0) * EXP_OUT_ONE)) for i in range(256)]


def exp_lut(d):
    """d: int array in [0, 4095]. Returns p in [0, 32768] (Q1.15)."""
    T = np.array(exp_table() + [0], dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    assert d.min() >= 0 and d.max() < (1 << EXP_D_BITS)
    i = d >> EXP_INTERP_BITS
    f = d & ((1 << EXP_INTERP_BITS) - 1)
    hi = T[i]
    lo = T[i + 1]
    # p = hi - rsr((hi - lo) * f, 4); (hi-lo)>=0 so plain rounding shift
    return hi - (((hi - lo) * f + (1 << (EXP_INTERP_BITS - 1))) >> EXP_INTERP_BITS)


# ---- rsqrt LUT: y0 = round(2^30 / sqrt((i+0.5) * 2^22)) for normalized m in [2^28, 2^30), i = m >> 22 in [64, 255]
RSQRT_M_LO = 1 << 28
RSQRT_M_HI = 1 << 30
RSQRT_IDX_SHIFT = 22
RSQRT_Y_SHIFT = 30


def rsqrt_table() -> list[int]:
    t = [0] * 256
    for i in range(64, 256):
        t[i] = int(round((1 << RSQRT_Y_SHIFT) / math.sqrt((i + 0.5) * (1 << RSQRT_IDX_SHIFT))))
    return t


def rsqrt_newton(m):
    """m: int64 in [2^28, 2^30). Returns y1 ~= 2^30/sqrt(m) after one Newton step (17-bit)."""
    T = np.array(rsqrt_table(), dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    assert m.min() >= RSQRT_M_LO and m.max() < RSQRT_M_HI
    y0 = T[m >> RSQRT_IDX_SHIFT]
    t = m * y0 * y0                      # <= 2^30 * 2^32 = 2^62
    d = ((3 << 60) - t + (1 << 29)) >> 30  # ~2^31
    y1 = (y0 * d + (1 << 30)) >> 31        # ~y0
    return y1


# ---- GELU LUT (per FFN/conv, depends on the static input/output scales)
def gelu_f(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_table(s_in: float, s_out: float) -> list[int]:
    """table[i + 128] = sat8(round(gelu(i * s_in) / s_out)) for i in -128..127 (index = i & 0xff order below)."""
    t = []
    for i in range(-128, 128):
        v = int(round(gelu_f(i * s_in) / s_out))
        t.append(max(-128, min(127, v)))
    return t  # index k corresponds to input i = k - 128


def gelu_lut(x8, table):
    T = np.asarray(table, dtype=np.int64)
    x = np.asarray(x8, dtype=np.int64)
    return T[x + 128].astype(np.int8)
