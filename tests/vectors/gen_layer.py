"""Golden mirror of the layer-level Verilator test: run the chip's micro-program in Python and dump what
the chip must produce (docs/decisions.md #12).

The chip runs `MicrocodeLayer` — the same generated program as the full model but emitted for `--layers 2
--max-ctx 64 --chunk 8 --vocab-tiles 256`: embed the prompt in chunks of 8, two transformer layers per
chunk, final norm + LM head over the first 8192 vocabulary columns, sample, then one row per decode step.
This mirror executes exactly that sequence with the full model's quantisation constants, so the residual
bank and the emitted token ids can be compared bit for bit.

  uv run python tests/vectors/gen_layer.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.data import EOS_IDS, REPO, load_tokenizer, prompts, text_ids  # noqa: E402
from golden.layout import act32_words, write_hex  # noqa: E402
from golden.quant import EMB_SLICE, LM_SLICE, build, manifest_config  # noqa: E402
from golden.minicpm_int import IntMiniCPM  # noqa: E402

OUT = REPO / "out" / "vectors" / "layer"


def pick_prompt(tok, n: int, vocab_limit: int) -> list[int]:
    """The first n token ids of the calibration prose that the restricted embedding table covers."""
    ids = text_ids(tok, prompts("calib")[0]["text"], 512)
    out = [i for i in ids if i < vocab_limit][:n]
    assert len(out) == n, f"only {len(out)} tokens below {vocab_limit}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-ctx", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--lm-cols", type=int, default=LM_SLICE)
    ap.add_argument("--prompt-len", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=2)
    a = ap.parse_args()
    assert a.lm_cols <= EMB_SLICE, "sampled ids must stay inside the instantiated embedding slice"
    t0 = time.time()
    cfg = manifest_config()
    qm = build(cfg)
    print(f"quantised model built in {time.time() - t0:.0f} s ({cfg.tag()})")
    tok = load_tokenizer()
    ids = pick_prompt(tok, a.prompt_len, a.lm_cols)
    print(f"prompt ({len(ids)} tokens): {ids} = {tok.decode(ids)!r}")

    m = IntMiniCPM(qm, n_layers=a.layers, lm_cols=a.lm_cols, max_ctx=a.max_ctx)
    m.reset_cache()
    X = np.zeros((a.max_ctx, qm.tables["embed"].shape[1]), np.int32)   # the chip's residual bank
    toks, gen, pos = list(ids), [], 0
    t0 = time.time()
    while True:
        # embed + layers for rows [pos, len(toks)) in chunks, exactly as CHUNK_BEGIN/CHUNK_NEXT do
        for c0 in range(pos, len(toks), a.chunk):
            rows = np.arange(c0, min(c0 + a.chunk, len(toks)))
            x = m.embed([toks[int(i)] for i in rows])
            for l in range(a.layers):
                x = m.layer(l, x, rows)
            X[rows] = x
        logits = m.lm_head(X[len(toks) - 1:len(toks)])[0]
        nxt = int(np.argmax(logits))
        print(f"  pos {len(toks) - 1:2d}: argmax {nxt} ({tok.decode([nxt])!r}), max logit {logits.max()}")
        if nxt in EOS_IDS or len(gen) >= a.max_new:
            break
        gen.append(nxt)
        toks.append(nxt)
        pos = len(toks) - 1
    print(f"mirror ran in {time.time() - t0:.0f} s: {len(toks)} rows, generated {gen} = {tok.decode(gen)!r}")

    d = OUT / f"L{a.layers}_ctx{a.max_ctx}"
    d.mkdir(parents=True, exist_ok=True)
    write_hex(d / "x_exp.hex", act32_words(X[:len(toks)]))
    (d / "prompt.txt").write_text("\n".join(str(i) for i in ids) + "\n")
    (d / "tokens_exp.txt").write_text(("\n".join(str(i) for i in gen) + "\n") if gen else "")
    json.dump(dict(layers=a.layers, max_ctx=a.max_ctx, chunk=a.chunk, lm_cols=a.lm_cols, prompt_len=len(ids),
                   max_new=a.max_new, rows=len(toks), n_gen=len(gen)), open(d / "meta.json", "w"), indent=1)
    print(f"wrote {d}")


if __name__ == "__main__":
    main()
