"""Calibration statistics from the fp32 model on LibriSpeech dev-clean (disjoint from the eval set).

Collects max-abs (tensor-level, per-head, per-channel where needed) of every activation whose scale is
static in the int datapath. Deterministic: sorted utterance list, single-threaded torch.
Output: weights/calib_stats.json  (committed; `make calib` regenerates it).
Usage: uv run python golden/calib.py [--n 128]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import CACHE, REPO, WHISPER_ROOT, load_audio  # noqa: E402

DEV = CACHE / "librispeech" / "LibriSpeech" / "dev-clean"


def calib_clips(n: int) -> list[Path]:
    files = sorted(DEV.glob("*/*/*.flac"))
    return files[:n]


class Stats:
    def __init__(self):
        self.max = defaultdict(float)          # tensor-level max-abs
        self.chan = {}                         # per-channel max-abs (last dim)
        self.head = {}                         # per-head max-abs

    def tensor(self, name, x):
        self.max[name] = max(self.max[name], float(x.abs().max()))

    def channels(self, name, x):
        m = x.reshape(-1, x.shape[-1]).abs().max(dim=0).values.double().numpy()
        self.chan[name] = np.maximum(self.chan[name], m) if name in self.chan else m

    def heads(self, name, x, n_head):
        m = x.reshape(-1, n_head, x.shape[-1] // n_head).abs().amax(dim=(0, 2)).double().numpy()
        self.head[name] = np.maximum(self.head[name], m) if name in self.head else m


def attach(model, st: Stats):
    hooks = []
    H = model.dims.n_audio_head

    def out_hook(name, per_chan=False, per_head=False):
        def f(mod, inp, out):
            st.tensor(name, out)
            if per_chan:
                st.channels(name, out)
            if per_head:
                st.heads(name, out, H)
        return f

    def in_hook(name, per_chan=False):
        def f(mod, inp):
            st.tensor(name, inp[0])
            if per_chan:
                st.channels(name, inp[0])
        return f

    enc, dec = model.encoder, model.decoder
    hooks.append(enc.conv1.register_forward_hook(out_hook("enc.conv1")))
    hooks.append(enc.conv2.register_forward_hook(out_hook("enc.conv2")))
    hooks.append(enc.conv1.register_forward_pre_hook(in_hook("enc.mel")))
    for l, b in enumerate(enc.blocks):
        p = f"enc.{l}"
        hooks.append(b.register_forward_pre_hook(in_hook(f"{p}.x_in", per_chan=True)))
        hooks.append(b.attn_ln.register_forward_hook(out_hook(f"{p}.ln1", per_chan=True)))
        hooks.append(b.attn.query.register_forward_hook(out_hook(f"{p}.q", per_head=True)))
        hooks.append(b.attn.key.register_forward_hook(out_hook(f"{p}.k", per_head=True)))
        hooks.append(b.attn.value.register_forward_hook(out_hook(f"{p}.v", per_head=True)))
        hooks.append(b.attn.out.register_forward_pre_hook(in_hook(f"{p}.wv", per_chan=True)))
        hooks.append(b.attn.out.register_forward_hook(out_hook(f"{p}.o")))
        hooks.append(b.mlp_ln.register_forward_pre_hook(in_hook(f"{p}.x_mid", per_chan=True)))
        hooks.append(b.mlp_ln.register_forward_hook(out_hook(f"{p}.ln2", per_chan=True)))
        hooks.append(b.mlp[0].register_forward_hook(out_hook(f"{p}.fc1")))
        hooks.append(b.mlp[2].register_forward_pre_hook(in_hook(f"{p}.gelu", per_chan=True)))
        hooks.append(b.mlp[2].register_forward_hook(out_hook(f"{p}.fc2")))
        hooks.append(b.register_forward_hook(out_hook(f"{p}.x_out", per_chan=True)))
    hooks.append(enc.ln_post.register_forward_hook(out_hook("enc.ln_post", per_chan=True)))
    for l, b in enumerate(dec.blocks):
        p = f"dec.{l}"
        hooks.append(b.register_forward_pre_hook(in_hook(f"{p}.x_in", per_chan=True)))
        hooks.append(b.attn_ln.register_forward_hook(out_hook(f"{p}.ln1", per_chan=True)))
        hooks.append(b.attn.query.register_forward_hook(out_hook(f"{p}.q", per_head=True)))
        hooks.append(b.attn.key.register_forward_hook(out_hook(f"{p}.k", per_head=True)))
        hooks.append(b.attn.value.register_forward_hook(out_hook(f"{p}.v", per_head=True)))
        hooks.append(b.attn.out.register_forward_pre_hook(in_hook(f"{p}.wv", per_chan=True)))
        hooks.append(b.attn.out.register_forward_hook(out_hook(f"{p}.o")))
        hooks.append(b.cross_attn_ln.register_forward_pre_hook(in_hook(f"{p}.x_mid1", per_chan=True)))
        hooks.append(b.cross_attn_ln.register_forward_hook(out_hook(f"{p}.lnc", per_chan=True)))
        hooks.append(b.cross_attn.query.register_forward_hook(out_hook(f"{p}.cq", per_head=True)))
        hooks.append(b.cross_attn.key.register_forward_hook(out_hook(f"{p}.ck", per_head=True)))
        hooks.append(b.cross_attn.value.register_forward_hook(out_hook(f"{p}.cv", per_head=True)))
        hooks.append(b.cross_attn.out.register_forward_pre_hook(in_hook(f"{p}.cwv", per_chan=True)))
        hooks.append(b.cross_attn.out.register_forward_hook(out_hook(f"{p}.co")))
        hooks.append(b.mlp_ln.register_forward_pre_hook(in_hook(f"{p}.x_mid2", per_chan=True)))
        hooks.append(b.mlp_ln.register_forward_hook(out_hook(f"{p}.ln2", per_chan=True)))
        hooks.append(b.mlp[0].register_forward_hook(out_hook(f"{p}.fc1")))
        hooks.append(b.mlp[2].register_forward_pre_hook(in_hook(f"{p}.gelu", per_chan=True)))
        hooks.append(b.mlp[2].register_forward_hook(out_hook(f"{p}.fc2")))
        hooks.append(b.register_forward_hook(out_hook(f"{p}.x_out", per_chan=True)))
    hooks.append(dec.ln.register_forward_hook(out_hook("dec.ln", per_chan=True)))
    return hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", type=Path, default=REPO / "weights" / "calib_stats.json")
    a = ap.parse_args()
    import whisper

    torch.set_num_threads(4)
    model = whisper.load_model("tiny", device="cpu", download_root=str(WHISPER_ROOT)).eval()
    st = Stats()
    attach(model, st)
    opts = whisper.DecodingOptions(task="transcribe", language="en", without_timestamps=True, temperature=0.0, fp16=False)
    files = calib_clips(a.n)
    for i, f in enumerate(files):
        audio = whisper.pad_or_trim(load_audio(f))
        mel = whisper.log_mel_spectrogram(torch.from_numpy(audio), n_mels=80)
        with torch.no_grad():
            res = whisper.decode(model, mel, opts)
        if i % 16 == 0:
            print(i, f.name, res.text[:60])
    # GELU outputs of conv1/conv2 are not module outputs; derive from pre-GELU max (gelu is monotone for x>0)
    out = {
        "n_clips": len(files), "clips": [f.stem for f in files],
        "max": dict(sorted(st.max.items())),
        "chan": {k: v.tolist() for k, v in sorted(st.chan.items())},
        "head": {k: v.tolist() for k, v in sorted(st.head.items())},
    }
    a.out.parent.mkdir(exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
        f.write("\n")
    for k, v in out["max"].items():
        print(f"{k:20s} {v:10.4f}")


if __name__ == "__main__":
    main()
