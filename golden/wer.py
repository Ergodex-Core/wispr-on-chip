"""WER computation.

Usage:
  uv run python golden/wer.py --hyp data/ref            # fp32 reference vs ground truth
  uv run python golden/wer.py --hyp out/golden_int      # int golden vs ground truth (+ vs fp32 tokens)
Hypothesis dir holds <uid>.json files with fields uid, language, text, tokens.
Prints per-set WER using whisper's normalizers (English normalizer for en; basic otherwise).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import DATA, clip_by_uid, normalize_text, ref_path, testclean_list, varied_clips  # noqa: E402


def load_hyp(d: Path, uid: str) -> dict | None:
    p = d / (uid.replace("/", "__") + ".json")
    return json.load(open(p)) if p.exists() else None


def wer_of(pairs: list[tuple[str, str]]) -> float:
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    return jiwer.wer(refs, hyps) if refs else float("nan")


def score(hyp_dir: Path, uids: list[str], label: str, compare_ref: bool = True, verbose: bool = False) -> dict:
    pairs, tok_match, tok_total, missing, ref_pairs = [], 0, 0, 0, []
    for uid in uids:
        h = load_hyp(hyp_dir, uid)
        if h is None:
            missing += 1
            continue
        c = clip_by_uid(uid)
        ref = normalize_text(c.text, c.language)
        hyp = normalize_text(h["text"], c.language)
        pairs.append((ref, hyp))
        if compare_ref and ref_path(uid).exists():
            r = json.load(open(ref_path(uid)))
            tok_total += 1
            tok_match += int(list(r["tokens"]) == list(h["tokens"]))
            ref_pairs.append((normalize_text(r["text"], c.language), hyp))
        if verbose:
            print(f"  {uid:40s} wer={jiwer.wer(ref, hyp) if ref else float('nan'):.3f} | {hyp}")
    out = dict(label=label, n=len(pairs), missing=missing, wer=wer_of(pairs))
    if compare_ref and tok_total:
        out["token_identity"] = tok_match / tok_total
        out["wer_vs_fp32_text"] = wer_of(ref_pairs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", type=Path, default=DATA / "ref")
    ap.add_argument("--sets", default="testclean_200,varied,rtl_20")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    is_ref = a.hyp.resolve() == (DATA / "ref").resolve()
    for s in a.sets.split(","):
        if s == "varied":
            uids = [c.uid for c in varied_clips()]
        else:
            uids = testclean_list(s + ".txt")
        r = score(a.hyp, uids, s, compare_ref=not is_ref, verbose=a.v)
        print(json.dumps(r))


if __name__ == "__main__":
    main()
