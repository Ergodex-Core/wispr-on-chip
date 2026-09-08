"""Fixed-point primitives shared by every golden op. These ARE the spec: RTL must match them bit-exactly.

All arithmetic is on numpy int64 (or Python int) with explicit widths. No floats in the datapath.
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
# Dynamic per-row quantisation (producer side). Input x: int16 or int32 rows with a static scale S_in.
# Output a8 (int8) and rowfac = (m16, b): maxabs = max(1, max|x|), b = max(bitlen(maxabs) - 16, 0),
# m16 = maxabs >> b (uint16 >= 1), recip = floor((127*2^16 + m16/2) / m16), a8 = sat8(rsr(x * recip, 16 + b)).
# rowfac is packed as m16 | (b << 16) (21 bits); the effective row factor is m16 << b.
# Real value of a8 = a8 * (m16 << b) * S_in / 127. For int16 rows b = 0 and this is the whisper-si rule.
DYNQ_RECIP_SHIFT = 16
ROWFAC_B_SHIFT = 16


def rowfac_pack(m16, b):
    return (np.asarray(m16, dtype=I64) | (np.asarray(b, dtype=I64) << I64(ROWFAC_B_SHIFT))).astype(np.int32)


def rowfac_eff(rowfac):
    rf = np.asarray(rowfac, dtype=I64)
    return (rf & I64((1 << 16) - 1)) << (rf >> I64(ROWFAC_B_SHIFT))


def dyn_quant_rows(x):
    x = check_range(x, 32, "dyn_quant input")
    maxabs = np.maximum(np.abs(x).max(axis=-1), 1).astype(I64)  # [rows]
    bl = np.array([int(v).bit_length() for v in maxabs], dtype=I64)
    b = np.maximum(bl - 16, 0)
    m16 = maxabs >> b
    recip = ((I64(127) << I64(DYNQ_RECIP_SHIFT)) + (m16 >> I64(1))) // m16  # 24-bit
    a8 = np.empty(x.shape, dtype=I64)
    for i in range(x.shape[0]):
        a8[i] = rsr(x[i] * recip[i], DYNQ_RECIP_SHIFT + int(b[i]))
    return sat8(a8).astype(np.int8), rowfac_pack(m16, b)


# ----------------------------------------------------------------------------------------------
# Matmul-edge requantisation (engine side). rowfac = (m16, b), effective row factor m16 << b.
#   t   = sat48( rsr(acc * M[n], s1) )
#   u   = t * m16[m] + (B[n] << (s2 - 16))                        (B = 0 whenever b > 0: int32 outputs; MiniCPM has no biases)
#   y   = sat_w( rsr(u, s2 - b[m]) )                              s2 = 24 (int8 out) / 20 (int16) / 24 (int32)
# i.e. y = rsr(t * (m16 << b) + bias_term, s2) computed with the power-of-two part of the row factor folded into
# the shift, so u stays below 2^64 (t < 2^47, m16 < 2^16). For int16 rows b = 0 and this is the whisper-si rule.
REQ_S2_BY_BITS = {8: 24, 16: 20, 32: 24}
REQ_B_FRAC = 16
REQ_T_BITS = 48


def requant(acc, mult, bias, s1: int, rowfac, out_bits: int):
    """acc [M,N] int32, mult/bias [N] int32, rowfac [M] packed (m16 | b<<16; all ones for static inputs)."""
    acc = check_range(acc, 32, "acc")
    mult = check_range(mult, 32, "mult")
    bias = check_range(bias, 32, "bias")
    s2 = REQ_S2_BY_BITS[out_bits]
    t = sat(rsr(acc * mult[None, :], s1), REQ_T_BITS)         # |t| < 2^47 (acc*mult < 2^62 fits int64)
    rf = np.asarray(rowfac, dtype=I64)
    m16 = (rf & I64(0xFFFF))[:, None]
    b = (rf >> I64(ROWFAC_B_SHIFT))[:, None]
    if out_bits != 32:
        assert not b.any(), "int8/int16 outputs expect int16 input rows (b = 0)"
        u = t * m16 + (bias[None, :] << I64(s2 - REQ_B_FRAC))
        y = sat(rsr(u, s2), out_bits)
    else:
        assert not bias.any(), "int32 outputs carry no bias"
        u = t * m16                                           # < 2^63
        y = np.empty(u.shape, dtype=I64)
        for i in range(u.shape[0]):                           # per-row shift s2 - b
            y[i] = rsr(u[i], s2 - int(b[i, 0]))
        y = sat(y, out_bits)
    return y.astype({8: np.int8, 16: np.int16, 32: np.int32}[out_bits])


def requant_wide(acc, mult, s1: int):
    """LM-head mode: t = sat32(rsr(acc * M[n], s1)), no row factor, no bias (argmax only)."""
    acc = check_range(acc, 32, "acc")
    return sat32(rsr(acc * check_range(mult, 32, "mult")[None, :], s1)).astype(np.int32)


_TORCH = None


def matmul_i8(a, w):
    """Exact int32 accumulation of int8/uint8 activations [M,K] against int8 weights [K,N].

    Uses torch._int_mm (exact int8 x int8 -> int32 on CPU) when the shapes qualify, else float64 BLAS
    (exact while |sum| < 2^53). Both paths give identical results (golden/tests/test_ops.py)."""
    global _TORCH
    a = np.asarray(a)
    w = np.asarray(w)
    assert a.shape[-1] == w.shape[0], (a.shape, w.shape)
    M, K = a.shape
    N = w.shape[1]
    if a.dtype == np.int8 and w.dtype == np.int8 and K % 8 == 0 and N % 8 == 0 and N > 16 and K > 16:
        if _TORCH is None:
            import torch
            _TORCH = torch
        torch = _TORCH
        rows = max(M, 17)                                    # _int_mm wants M > 16
        ap = np.zeros((rows, K), dtype=np.int8)
        ap[:M] = a
        out = torch._int_mm(torch.from_numpy(ap), torch.from_numpy(np.ascontiguousarray(w))).numpy()[:M]
        return out.astype(np.int32)
    acc = a.astype(np.float64) @ w.astype(np.float64)
    out = acc.astype(I64)
    assert np.array_equal(out.astype(np.float64), acc)
    return check_range(out, 32, "matmul acc").astype(np.int32)


# ----------------------------------------------------------------------------------------------
# Scaled add (residual, int32 operands): out = sat_w( rsr(a * Ma + b * Mb, SADD_SHIFT) )
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


def mult_shift(ratio: float, bits: int = 31) -> tuple[int, int]:
    """(mult, shift) with mult in [2^(bits-2), 2^(bits-1)) such that mult / 2^shift ~= ratio."""
    import math
    assert ratio > 0
    sh = (bits - 1) - 1 - int(math.floor(math.log2(ratio)))
    m = int(round(ratio * 2.0 ** sh))
    if m >= (1 << (bits - 1)):
        m >>= 1
        sh -= 1
    assert 0 <= sh < 63 and 0 < m < (1 << (bits - 1)), (ratio, m, sh)
    return m, sh
