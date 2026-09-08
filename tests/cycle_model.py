"""Analytic cycle model of the chip from unit-level measurements (no full-chip simulation).

  uv run python tests/cycle_model.py [--prompt 128] [--new 32]

Engine: KT*NT*(M+4) + 10 cycles per job (weight-port bound at M = 1: 4 cycles per 32x32 tile), measured in
MatmulEngineSpec. Vector unit: per-row costs measured in VectorUnitSpec. Attention: per engine job
1 + 4*(rows+4) + 13, 14 cycles per query row per key tile for the softmax pass, 16 per query row for the
final pass (AttentionSpec). Sequencer overhead: ~4 cycles per instruction.
"""
from __future__ import annotations

import argparse
import math

D, H, HKV, HD, FF, V, LAYERS = 2048, 16, 2, 128, 6144, 130560, 42
CHUNK = 512
# vector unit, cycles per row (VectorUnitSpec)
VU = dict(rmsnorm=691, dynq=295, add=262, embed=139, rope_q=410, rope_k=74, silumul=807)


def engine(M, K, N):
    return (K // 32) * (N // 32) * (M + 4) + 10


def attention(nq, nk):
    """all 16 heads over the query blocks of 64 and key tiles of 64"""
    total = 0
    for qb in range(math.ceil(nq / 64)):
        rows = min(64, nq - 64 * qb)
        for kt in range(math.ceil(nk / 64)):
            job = 1 + 4 * (rows + 4) + 13
            total += 3 * job + 14 * rows            # S job, softmax pass, PV hi, PV lo
        total += 16 * rows                          # final pass
    return H * total


def layer(M, n_keys):
    c = {}
    c["rmsnorm"] = 2 * M * VU["rmsnorm"]
    c["qkv"] = engine(M, D, D) + engine(M, D, HKV * HD) * 2
    c["rope"] = M * (VU["rope_q"] + VU["rope_k"])
    c["attention"] = attention(M, n_keys)
    c["dynq"] = M * VU["dynq"]
    c["o"] = engine(M, D, D)
    c["gate_up"] = 2 * engine(M, D, FF)
    c["silumul"] = M * VU["silumul"]
    c["down"] = engine(M, FF, D)
    c["add"] = 2 * M * VU["add"]
    c["seq"] = 22 * 4
    return c


def run(prompt, new):
    def total(M, n_keys):
        t = {}
        for l in range(LAYERS):
            for k, v in layer(M, n_keys).items():
                t[k] = t.get(k, 0) + v
        t["embed"] = M * VU["embed"]
        t["lm_head"] = VU["rmsnorm"] + engine(1, D, V)
        return t

    # prefill in chunks of 512 rows (keys grow with the chunk base)
    pre = {}
    base = 0
    while base < prompt:
        rows = min(CHUNK, prompt - base)
        for k, v in total(rows, base + rows).items():
            pre[k] = pre.get(k, 0) + (v if k != "lm_head" else 0)
        base += rows
    pre["lm_head"] = VU["rmsnorm"] + engine(1, D, V)
    dec = total(1, prompt + new // 2)   # average decode step
    return pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--new", type=int, default=32)
    a = ap.parse_args()
    pre, dec = run(a.prompt, a.new)
    tp, td = sum(pre.values()), sum(dec.values())
    print(f"prefill of {a.prompt} tokens: {tp / 1e6:.1f} M cycles")
    for k, v in sorted(pre.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v / 1e6:8.2f} M  ({100 * v / tp:4.1f} %)")
    print(f"one decode step (~{a.prompt + a.new // 2} keys): {td / 1e6:.2f} M cycles")
    for k, v in sorted(dec.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v / 1e6:8.3f} M  ({100 * v / td:4.1f} %)")
    macs = 2.52e9  # MACs per token (weights)
    print(f"engine MAC utilisation in decode: {100 * macs / (td * 1024):.1f} %  (weight-port bound: 4 cycles per 32x32 tile)")
    print(f"at 1 GHz: prefill {tp / 1e6:.0f} ms, {td / 1e6:.1f} ms per generated token")


if __name__ == "__main__":
    main()
