"""Sweep quantisation configurations against the fp32 reference (teacher-forced next-token agreement).

  uv run python golden/sweep_quant.py [--max-len 96] [--configs default]

The fp32 argmax of every eval text prompt is computed once and cached in out/ref_argmax.npz, so each
configuration costs one integer forward per prompt. Reports top-1 agreement, the rank the fp32 choice has
under the integer logits, and the fraction of saturated values in the int16/int32 static tensors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import REPO, load_tokenizer, prompts, text_ids  # noqa: E402
from golden.fixedpoint import SAT, sat_stats_reset  # noqa: E402
from golden.quant import QConfig, build  # noqa: E402
from golden.minicpm_int import IntMiniCPM  # noqa: E402

CACHE = REPO / "out" / "ref_argmax.npz"


def text_prompts(which: str):
    """eval: the two eval texts (184 positions); all: + the calibration texts (~510 positions)."""
    ps = [p for p in prompts("eval") if p["mode"] == "text"]
    if which == "all":
        ps += [p for p in prompts("calib") if p["mode"] == "text"]
    return ps

CONFIGS = {
    "default": [
        ("baseline (no smoothing)", QConfig(smooth_alpha=None)),
        ("smooth a=0.5", QConfig(smooth_alpha=0.5)),
        ("smooth a=0.7", QConfig(smooth_alpha=0.7)),
        ("smooth a=0.85", QConfig(smooth_alpha=0.85)),
    ],
    "margin": [
        ("smooth a=0.5, m16=1.5", QConfig(smooth_alpha=0.5, margin_int16=1.5)),
        ("smooth a=0.5, m16=1.0", QConfig(smooth_alpha=0.5, margin_int16=1.0)),
    ],
    "isolate": [   # is the gain the smoothing or the margin?
        ("no smoothing, m16=1.5", QConfig(smooth_alpha=None, margin_int16=1.5)),
        ("smooth a=0.5, m16=1.25", QConfig(smooth_alpha=0.5, margin_int16=1.25)),
    ],
    "final": [   # the decision, on ~3x the positions
        ("baseline: no smoothing, m16=2.0", QConfig(smooth_alpha=None)),
        ("smooth a=0.5, m16=2.0", QConfig(smooth_alpha=0.5)),
        ("smooth a=0.5, m16=1.5", QConfig(smooth_alpha=0.5, margin_int16=1.5)),
    ],
}


def reference(tok, ps, max_len):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if {p["name"] for p in ps} <= set(d.files):
            ids = json.loads(str(d["ids"]))
            return {p["name"]: d[p["name"]] for p in ps}, {p["name"]: ids[p["name"]] for p in ps}
    from golden.reference_cpu import LlamaRef
    ref = LlamaRef()
    out, ids_all = {}, {}
    if CACHE.exists():          # keep what is already cached
        d = np.load(CACHE, allow_pickle=True)
        old_ids = json.loads(str(d["ids"]))
        out = {k: d[k] for k in d.files if k != "ids"}
        ids_all = dict(old_ids)
    for p in ps:
        if p["name"] in out:
            continue
        ids = text_ids(tok, p["text"], max_len)
        t0 = time.time()
        out[p["name"]] = ref.forward(ids).numpy().argmax(axis=1).astype(np.int32)
        ids_all[p["name"]] = ids
        print(f"fp32 {p['name']:12s} {len(ids):3d} positions in {time.time() - t0:.0f} s")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ids=json.dumps(ids_all), **out)
    return {p["name"]: out[p["name"]] for p in ps}, {p["name"]: ids_all[p["name"]] for p in ps}


def evaluate(qm, ids_all, refs):
    gm = IntMiniCPM(qm)
    agree = total = 0
    ranks = []
    sat_stats_reset(True)
    for name, ids in ids_all.items():
        gm.reset_cache()
        li = gm.forward(ids, 0, "all")
        n = len(ids) - 1
        af, ai = refs[name][:n], li[:n].argmax(axis=1)
        agree += int((af == ai).sum())
        total += n
        ranks += [int((li[t] > li[t, af[t]]).sum()) + 1 for t in range(n)]
    sat = 100.0 * SAT["count"] / max(SAT["total"], 1)
    sat_stats_reset(False)
    return agree, total, float(np.mean(ranks)), int(max(ranks)), sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--configs", default="default")
    ap.add_argument("--prompts", default="eval", choices=["eval", "all"])
    ap.add_argument("--out", default=str(REPO / "out" / "sweep.json"))
    a = ap.parse_args()
    tok = load_tokenizer()
    ps = text_prompts(a.prompts)
    refs, ids_all = reference(tok, ps, a.max_len)
    rows = []
    for label, cfg in CONFIGS[a.configs]:
        t0 = time.time()
        qm = build(cfg)
        agree, total, mean_rank, max_rank, sat = evaluate(qm, ids_all, refs)
        rows.append(dict(label=label, cfg=cfg.__dict__, agree=agree, total=total,
                         pct=round(100 * agree / total, 2), mean_rank=round(mean_rank, 3), max_rank=max_rank,
                         saturated_pct=round(sat, 4), seconds=round(time.time() - t0, 1)))
        print(f"{label:34s} top-1 {agree:3d}/{total} = {100 * agree / total:5.2f} %   mean rank {mean_rank:.3f}  max {max_rank}"
              f"  requant saturation {sat:.4f} %  [{time.time() - t0:.0f} s]")
        del qm
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    best = max(rows, key=lambda r: r["agree"])
    print(f"\nbest: {best['label']} ({best['pct']} %)")


if __name__ == "__main__":
    main()
