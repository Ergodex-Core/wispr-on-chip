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


# ---- sigmoid LUT (SiLU = x * sigmoid(x)): SIG[i] = round(sigmoid((i-128)/16) * 2^15), i = 0..255, x in [-8, 8)
SIG_ONE = 1 << 15


def sigmoid_table() -> list[int]:
    return [int(round(SIG_ONE / (1.0 + math.exp(-((i - 128) / 16.0))))) for i in range(256)]


def sigmoid_last() -> int:
    """SIG[256] (x = +8), used by the interpolation of the last segment."""
    return int(round(SIG_ONE / (1.0 + math.exp(-8.0))))


def sigmoid_lut(xf):
    """xf: x in Q.8 clamped to [-2048, 2047]. Returns sigmoid(x) in Q15, linear interpolation."""
    T = np.array(sigmoid_table() + [sigmoid_last()], dtype=np.int64)
    xf = np.clip(np.asarray(xf, dtype=np.int64), -2048, 2047)
    i = (xf >> 4) + 128
    f = xf & 15
    lo, hi = T[i], T[i + 1]
    return lo + (((hi - lo) * f + 8) >> 4)


# ---- RoPE table: cos/sin in Q15 (scale 32767) per position and frequency, HF Llama convention
ROPE_ONE = 32767


def rope_table(n_pos: int, head_dim: int, theta: float) -> np.ndarray:
    """int16 [n_pos, head_dim]: row = [cos_0..cos_{hd/2-1}, sin_0..sin_{hd/2-1}] with
    inv_freq[i] = theta^(-2i/hd) computed in float32 like HF, angle = pos * inv_freq in float32."""
    half = head_dim // 2
    inv = (1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))).astype(np.float32)
    ang = (np.arange(n_pos, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    c = np.round(np.cos(ang.astype(np.float64)) * ROPE_ONE)
    s = np.round(np.sin(ang.astype(np.float64)) * ROPE_ONE)
    out = np.concatenate([c, s], axis=1)
    assert out.shape == (n_pos, head_dim) and np.abs(out).max() <= ROPE_ONE
    return out.astype(np.int16)
