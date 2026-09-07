"""Analytic cycle model of WhisperTop, built only from unit-level measurements (Phase 2/3) and the
micro-program structure, for comparison with the measured e2e cycle counts (docs/status.md).

Model constants (all measured, see docs/status.md Phase 2/3):
  matmul   : KT*NT*(M+4) + 10          (99.7 % util at M=1500; 5 cycles/tile at M=1)
  vector   : cycles per row: LN 123, dynq 85, gelu+dynq(1536) 229, gelu static8 59, add 32, embed 83
  attention: per (head, 64-query block, 64-key tile): S = 1 + 4*(rows+4) + 13 ; softmax 12*rows ;
             PV hi/lo 2*(1 + 4*(rows+4) + 13) ; final 31 cycles per query row (divider latency 21)
Usage: uv run python tests/e2e/cycle_model.py <n_frames> <n_generated_tokens> [measured_cycles]
"""
from __future__ import annotations

import math
import sys

D = 384
FF = 1536
HEADS = 6
LAYERS = 4
LM_N = 51872
PROMPT = 4
VEC = dict(ln=123, dynq=85, gelu_dynq=229, gelu8=59, add=32, embed=83)
ENG_DRAIN = 13


def tiles(n):
    return (n + 31) // 32


def matmul(m, k, n):
    return tiles(k) * tiles(n) * (m + 4) + 10


def attention(n_q, n_keys):
    total = 0
    for qb in range(math.ceil(n_q / 64)):
        rows = min(64, n_q - qb * 64)
        n_kt = math.ceil(n_keys / 64)
        s = 1 + 4 * (rows + 4) + ENG_DRAIN
        per_kt = s + 12 * rows + 2 * s
        total += n_kt * per_kt + rows * 31
    return total * HEADS


def encoder(n_frames):
    n_ctx = n_frames // 2
    c = {}
    c["conv1 (im2col 288->384, gelu8)"] = matmul(n_frames, 288, D) + n_frames * VEC["gelu8"]
    c["conv2 (im2col 1152->384) + x0 add"] = matmul(n_ctx, 1152, D) + n_ctx * VEC["add"]
    per_layer = (n_ctx * VEC["ln"] + 4 * matmul(n_ctx, D, D) + attention(n_ctx, n_ctx) + n_ctx * VEC["dynq"]
                 + n_ctx * VEC["add"] + n_ctx * VEC["ln"] + matmul(n_ctx, D, FF) + n_ctx * VEC["gelu_dynq"]
                 + matmul(n_ctx, FF, D) + n_ctx * VEC["add"])
    c["encoder layers x4 (of which attention)"] = LAYERS * per_layer
    c["  attention only x4"] = LAYERS * attention(n_ctx, n_ctx)
    c["ln_post + cross K/V x4"] = n_ctx * VEC["ln"] + 8 * matmul(n_ctx, D, D)
    return c


def decoder_step(pos, n_ctx, sample):
    t = VEC["embed"] + VEC["add"]
    for _ in range(LAYERS):
        t += VEC["ln"] + 3 * matmul(1, D, D) + attention(1, pos + 1) + VEC["dynq"] + matmul(1, D, D) + VEC["add"]
        t += VEC["ln"] + matmul(1, D, D) + attention(1, n_ctx) + VEC["dynq"] + matmul(1, D, D) + VEC["add"]
        t += VEC["ln"] + matmul(1, D, FF) + VEC["gelu_dynq"] + matmul(1, FF, D) + VEC["add"]
    if sample:
        t += VEC["ln"] + matmul(1, D, LM_N)
    return t


def model(n_frames, n_gen):
    c = encoder(n_frames)
    n_ctx = n_frames // 2
    n_pos = PROMPT + n_gen            # positions decoded; LM head from position PROMPT-1 onwards
    dec = sum(decoder_step(p, n_ctx, p >= PROMPT - 1) for p in range(n_pos))
    c[f"decoder ({n_pos} positions, {n_gen + 1} LM-head evaluations)"] = dec
    total = sum(v for k, v in c.items() if not k.startswith("  "))
    return c, total


def main():
    n_frames, n_gen = int(sys.argv[1]), int(sys.argv[2])
    meas = int(sys.argv[3]) if len(sys.argv) > 3 else None
    c, total = model(n_frames, n_gen)
    print(f"n_frames={n_frames} n_ctx={n_frames // 2} generated tokens={n_gen}")
    for k, v in c.items():
        print(f"  {k:52s} {v:12,d}  ({100 * v / total:5.1f} %)")
    print(f"  {'model total':52s} {total:12,d}")
    if meas:
        print(f"  {'measured':52s} {meas:12,d}   model/measured = {total / meas:.3f}")


if __name__ == "__main__":
    main()
