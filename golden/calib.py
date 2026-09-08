"""Calibration: run the fp32 reference over data/prompts.json['calib'] and record activation maxima
(per tensor, per channel for norm outputs, per head for Q/K). Output: weights/calib_stats.json.

  uv run python golden/calib.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import REPO, chat_ids, load_tokenizer, prompts, text_ids  # noqa: E402
from golden.reference_cpu import LlamaRef  # noqa: E402

OUT = REPO / "weights" / "calib_stats.json"


def prompt_ids(tok, p: dict, max_len: int) -> list[int]:
    return chat_ids(tok, p["text"]) if p["mode"] == "chat" else text_ids(tok, p["text"], max_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--gen", type=int, default=24, help="greedy tokens generated per chat prompt (also calibrated)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    tok = load_tokenizer()
    m = LlamaRef()
    m.stats = dict(max={}, chan={}, head={})
    n_tokens = 0
    for p in prompts("calib"):
        ids = prompt_ids(tok, p, a.max_len)
        t0 = time.time()
        if p["mode"] == "chat":
            out = m.generate(ids, a.gen)
            n_tokens += len(ids) + len(out)
            print(f"{p['name']:12s} {len(ids):4d} prompt + {len(out):3d} generated tokens  {time.time() - t0:5.1f} s  {tok.decode(out)[:60]!r}")
        else:
            m.forward(ids)
            n_tokens += len(ids)
            print(f"{p['name']:12s} {len(ids):4d} tokens  {time.time() - t0:5.1f} s")
    st = m.stats
    # per-head maxima over both pre- and post-RoPE values (the int16 Q/K scale must cover both)
    L = m.L
    for l in range(L):
        st["head"][f"q.{l}"] = np.maximum(st["head"][f"q.{l}"], st["head"].pop(f"q_pre.{l}"))
        st["head"][f"k.{l}"] = np.maximum(st["head"][f"k.{l}"], st["head"].pop(f"k_pre.{l}"))
    js = dict(n_tokens=n_tokens, n_prompts=len(prompts("calib")), n_layers=L,
              max={k: float(v) for k, v in st["max"].items()},
              chan={k: [float(x) for x in v] for k, v in st["chan"].items()},
              head={k: [float(x) for x in v] for k, v in st["head"].items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(js, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}: {n_tokens} tokens")
    for l in (0, 1, 10, 20, 30, 41):
        print(f"layer {l}: x_in {js['max'][f'x_in.{l}']:.2f} x_mid {js['max'][f'x_mid.{l}']:.2f} q {max(js['head'][f'q.{l}']):.2f} "
              f"k {max(js['head'][f'k.{l}']):.2f} v {js['max'][f'v.{l}']:.2f} o {js['max'][f'o.{l}']:.2f} gate {js['max'][f'gate.{l}']:.2f} "
              f"up {js['max'][f'up.{l}']:.2f} h {js['max'][f'h.{l}']:.2f} down {js['max'][f'down.{l}']:.2f} norm1 {max(js['chan'][f'norm1.{l}']):.2f}")


if __name__ == "__main__":
    main()
