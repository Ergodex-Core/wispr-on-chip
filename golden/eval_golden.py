"""Accuracy of the integer golden model against the fp32 reference on data/prompts.json['eval'].

  uv run python golden/eval_golden.py [--max-new 12] [--max-len 96] [--out out/eval.json]

Text prompts: teacher-forced next-token top-1 agreement (int argmax == fp32 argmax) and the rank of the fp32
argmax under the int logits. Chat prompts: greedy generations of both models (token-level match length).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import EOS_IDS, REPO, chat_ids, load_tokenizer, prompts, text_ids  # noqa: E402
from golden.quant import QConfig, build  # noqa: E402
from golden.minicpm_int import IntMiniCPM  # noqa: E402
from golden.reference_cpu import LlamaRef  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--out", default=str(REPO / "out" / "eval.json"))
    a = ap.parse_args()
    tok = load_tokenizer()
    ref = LlamaRef()
    t0 = time.time()
    qm = build(QConfig())
    print(f"quantised model built in {time.time() - t0:.0f} s")
    gm = IntMiniCPM(qm)
    results = []
    agree = total = 0
    for p in prompts("eval"):
        if p["mode"] == "text":
            ids = text_ids(tok, p["text"], a.max_len)
            t0 = time.time()
            lf = ref.forward(ids).numpy()                      # [T, V]
            gm.reset_cache()
            li = gm.forward(ids, 0, "all")                     # [T, V] int32
            af, ai = lf.argmax(axis=1), li.argmax(axis=1)
            n = len(ids) - 1
            match = int((af[:n] == ai[:n]).sum())
            # rank of the fp32 argmax under the int logits (1 = agrees)
            ranks = [int((li[t] > li[t, af[t]]).sum()) + 1 for t in range(n)]
            agree += match; total += n
            results.append(dict(name=p["name"], mode="text", tokens=n, top1_agree=match, mean_rank=float(np.mean(ranks)), max_rank=int(max(ranks)),
                                seconds=round(time.time() - t0, 1)))
            print(f"{p['name']:12s} {n:3d} positions: top-1 agreement {match}/{n} ({100 * match / n:.1f} %), fp32-argmax rank under int: mean {np.mean(ranks):.2f} max {max(ranks)}  [{time.time() - t0:.0f} s]")
        else:
            ids = chat_ids(tok, p["text"])
            t0 = time.time()
            gf = ref.generate(ids, a.max_new)
            gi = gm.generate(ids, a.max_new)
            k = 0
            while k < min(len(gf), len(gi)) and gf[k] == gi[k]:
                k += 1
            results.append(dict(name=p["name"], mode="chat", fp32=tok.decode(gf), int=tok.decode(gi), fp32_ids=gf, int_ids=gi,
                                match_prefix=k, identical=gf == gi, seconds=round(time.time() - t0, 1)))
            print(f"{p['name']:12s} fp32: {tok.decode(gf)!r}\n{'':12s} int : {tok.decode(gi)!r}  (first {k} tokens identical{', all' if gf == gi else ''})  [{time.time() - t0:.0f} s]")
    summary = dict(top1_agree=agree, positions=total, top1_pct=round(100 * agree / max(total, 1), 2),
                   chat_identical=sum(1 for r in results if r["mode"] == "chat" and r["identical"]),
                   chat_total=sum(1 for r in results if r["mode"] == "chat"), results=results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"\ntop-1 agreement {agree}/{total} = {summary['top1_pct']} %; chat generations identical {summary['chat_identical']}/{summary['chat_total']}")


if __name__ == "__main__":
    main()
