"""Run the int golden model over evaluation sets and write hypothesis JSONs (same schema as data/ref).

Usage: uv run python golden/run_golden.py --set rtl_default [--frames var|full] [--residual-bits 16]
       [--gelu-mode phi16|lut8] [--smooth-alpha 0.5] [--smooth-fc2] [--out out/golden_int] [--dump DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import REPO, clip_by_uid, load_audio, testclean_list, varied_clips  # noqa: E402
from golden.quant import QConfig, build  # noqa: E402
from golden.whisper_int import IntWhisper, n_frames_for  # noqa: E402

_model = None
_args = None


def get_model(cfg: QConfig):
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)
        _model = IntWhisper(build(cfg), dump=False)
    return _model


def mel_of(audio: np.ndarray) -> np.ndarray:
    import torch
    import whisper
    return whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)), n_mels=80).numpy()


def run_one(args) -> dict:
    uid, cfg, frames_mode, dump_dir, pad, minf = args
    m = get_model(cfg)
    c = clip_by_uid(uid)
    audio = load_audio(c.path)
    mel = mel_of(audio)
    nf = n_frames_for(len(audio), frames_mode, pad, minf)
    if dump_dir:
        m.dump.enabled = True
    t0 = time.time()
    toks = m.transcribe(mel, nf, c.language)
    dt = time.time() - t0
    if dump_dir:
        m.dump.save(Path(dump_dir) / (uid.replace("/", "__") + ".npz"))
        m.dump.d.clear()
    import whisper
    tok = whisper.tokenizer.get_tokenizer(True, num_languages=99, language=c.language, task="transcribe")
    text = tok.decode(toks).strip()
    return dict(uid=uid, language=c.language, n_frames=nf, tokens=toks, text=text, ref_text=c.text, seconds=round(dt, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="rtl_default")
    ap.add_argument("--uids", default=None, help="comma-separated explicit uids")
    ap.add_argument("--frames", default="var", choices=["var", "full"])
    ap.add_argument("--pad-frames", type=int, default=0, help="silence frames appended before rounding (var mode)")
    ap.add_argument("--min-frames", type=int, default=0, help="minimum context in frames (var mode)")
    ap.add_argument("--residual-bits", type=int, default=16)
    ap.add_argument("--gelu-mode", default="phi16")
    ap.add_argument("--smooth-alpha", type=float, default=None)
    ap.add_argument("--smooth-fc2", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    cfg = QConfig(residual_bits=a.residual_bits, gelu_mode=a.gelu_mode, smooth_alpha=a.smooth_alpha, smooth_fc2=a.smooth_fc2)
    rule = "full" if a.frames == "full" else f"var_p{a.pad_frames}_m{a.min_frames}"
    out = a.out or (REPO / "out" / f"golden_{cfg.tag()}_{rule}")
    out.mkdir(parents=True, exist_ok=True)
    uids = []
    if a.uids:
        uids = a.uids.split(",")
    else:
        for s in a.set.split(","):
            uids += [c.uid for c in varied_clips()] if s == "varied" else testclean_list(s + ".txt")
    jobs = [(u, cfg, a.frames, str(a.dump) if a.dump else None, a.pad_frames, a.min_frames) for u in uids]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_one, jobs):
            with open(out / (r["uid"].replace("/", "__") + ".json"), "w") as f:
                json.dump(r, f, indent=1, ensure_ascii=False, sort_keys=True)
            print(f"{r['uid']:40s} nf={r['n_frames']:4d} {r['seconds']:6.1f}s {len(r['tokens']):3d} tok | {r['text']}")
    print("out:", out)


if __name__ == "__main__":
    main()
