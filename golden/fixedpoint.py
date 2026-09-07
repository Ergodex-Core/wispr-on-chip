"""Fixed-point primitives shared by every golden op. These ARE the spec: RTL must match them bit-exactly.

All arithmetic is on numpy int64 (or Python int) with explicit widths. No floats.
Rounding convention everywhere: round-half-up via (x + 2^(s-1)) >> s (arithmetic shift), s >= 1.
"""
from __future__ import annotations

import numpy as np

I64 = np.int64


def rsr(x, s: int):
    """Rounding arithmetic right shift: floor((x + 2^(s-1)) / 2^s); identity for s == 0."""
    x = np.asarray(x, dtype=I64)
    if s == 0:
        return x
    assert 0 < s < 63
    return (x + (I64(1) << I64(s - 1))) >> I64(s)


def sat(x, bits: int):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.asarray(x, dtype=I64), lo, hi)


def sat8(x):
    return sat(x, 8)


def sat16(x):
    return sat(x, 16)


def sat32(x):
    return sat(x, 32)


def check_range(x, bits: int, what: str = ""):
    x = np.asarray(x, dtype=I64)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if x.size and (x.min() < lo or x.max() > hi):
        raise OverflowError(f"{what}: value out of {bits}-bit range [{x.min()}, {x.max()}]")
    return x


# ----------------------------------------------------------------------------------------------
# Dynamic per-row quantisation (producer side). Input x16: int16 rows with a static scale S_in.
# Output a8 (int8) and rowfac = maxabs (uint16, >= 1). Real value of a8 = a8 * maxabs * S_in / 127.
DYNQ_RECIP_SHIFT = 16


def dyn_quant_rows(x16):
    x16 = check_range(x16, 16, "dyn_quant input")
    maxabs = np.maximum(np.abs(x16).max(axis=-1), 1).astype(I64)  # [rows]
    recip = ((I64(127) << I64(DYNQ_RECIP_SHIFT)) + (maxabs >> I64(1))) // maxabs  # 24-bit
    a8 = sat8(rsr(x16 * recip[..., None], DYNQ_RECIP_SHIFT))
    return a8.astype(np.int8), maxabs.astype(np.uint16)


# ----------------------------------------------------------------------------------------------
# Matmul-edge requantisation (engine side).
#   t   = sat32( rsr(acc * M[n], s1) )
#   y8  = sat8( rsr( t * rowfac[m] + B[n], s2 ) )          s2 == REQ_S2 for int8 outputs
REQ_S2 = 24


def requant_int8(acc, mult, bias, s1: int, rowfac):
    """acc [M,N] int32, mult/bias [N] int32, rowfac [M] uint16 (all ones for static inputs)."""
    acc = check_range(acc, 32, "acc")
    mult = check_range(mult, 32, "mult")
    bias = check_range(bias, 32, "bias")
    t = sat32(rsr(acc * mult[None, :], s1))
    rf = np.asarray(rowfac, dtype=I64)
    u = t * rf[:, None] + bias[None, :]
    return sat8(rsr(u, REQ_S2)).astype(np.int8)


def requant_wide(acc, mult, s1: int):
    """LM-head mode: t = sat32(rsr(acc * M[n], s1)), no row factor, no bias (argmax only)."""
    acc = check_range(acc, 32, "acc")
    return sat32(rsr(acc * check_range(mult, 32, "mult")[None, :], s1)).astype(np.int32)


def matmul_i8(a, w):
    """Exact int32 accumulation of int8/uint8 activations [M,K] against int8 weights [K,N].

    Uses float64 BLAS which is exact while |sum| < 2^53 (true for K*255*128 < 2^53)."""
    a = np.asarray(a)
    w = np.asarray(w)
    assert a.shape[-1] == w.shape[0], (a.shape, w.shape)
    acc = a.astype(np.float64) @ w.astype(np.float64)
    out = acc.astype(I64)
    assert np.array_equal(out.astype(np.float64), acc)
    return check_range(out, 32, "matmul acc").astype(np.int32)


# ----------------------------------------------------------------------------------------------
# Scaled add (residual / pos-emb / embedding): out = sat_w( rsr(a * Ma + b * Mb, SADD_SHIFT) )
SADD_SHIFT = 16


def scaled_add(a, ma, b, mb, out_bits: int):
    a = np.asarray(a, dtype=I64)
    b = np.asarray(b, dtype=I64)
    ma = np.asarray(ma, dtype=I64)
    mb = np.asarray(mb, dtype=I64)
    return sat(rsr(a * ma + b * mb, SADD_SHIFT), out_bits)


def mult_for(ratio: float, shift: int, bits: int = 31) -> int:
    m = int(round(ratio * (1 << shift)))
    assert 0 <= m < (1 << bits), (ratio, shift, m)
    return m
