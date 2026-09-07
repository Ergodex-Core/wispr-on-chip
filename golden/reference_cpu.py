"""fp32 CPU reference: openai-whisper tiny, greedy, no timestamps, one 30 s window per clip.

Writes data/ref/<uid>.json with generated tokens + text, and data/ref/_meta.json with the
decoding configuration (prompt, suppress lists) that the int golden model must replicate.
Deterministic: single-threaded torch per worker.

Usage: uv run python golden/reference_cpu.py [--set testclean_200|varied|all] [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import DATA, WHISPER_ROOT, Clip, load_audio, ref_path, testclean_clips, varied_clips  # noqa: E402

_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        import whisper

        torch.set_num_threads(1)
        _model = whisper.load_model("tiny", device="cpu", download_root=str(WHISPER_ROOT))
        _model.eval()
    return _model


def decode_options(language: str):
    import whisper

    return whisper.DecodingOptions(
        task="transcribe", language=language, without_timestamps=True, temperature=0.0, fp16=False,
        beam_size=None, best_of=None, sample_len=None,
    )


def transcribe(clip: Clip) -> dict:
    import torch
    import whisper

    model = get_model()
    audio = load_audio(clip.path)
    n_samples = len(audio)
    audio30 = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio30), n_mels=80)
    with torch.no_grad():
        res = whisper.decode(model, mel, decode_options(clip.language))
    return dict(
        uid=clip.uid, language=clip.language, n_samples=n_samples, duration_s=n_samples / 16000,
        tokens=[int(t) for t in res.tokens], text=res.text, ref_text=clip.text,
        avg_logprob=float(res.avg_logprob), no_speech_prob=float(res.no_speech_prob),
    )


def dump_meta():
    import torch
    import whisper
    from whisper.decoding import DecodingTask

    model = get_model()
    meta = {"model": "tiny", "checkpoint_sha256": hashlib.sha256(open(WHISPER_ROOT / "tiny.pt", "rb").read()).hexdigest(),
            "dims": model.dims.__dict__, "languages": {}}
    for lang in ["en", "de", "fr"]:
        task = DecodingTask(model, decode_options(lang))
        meta["languages"][lang] = dict(
            initial_tokens=list(task.initial_tokens), sample_begin=task.sample_begin, sample_len=task.sample_len,
            eot=task.tokenizer.eot, suppress_tokens=sorted(int(t) for t in task._get_suppress_tokens()),
            suppress_blank=[task.tokenizer.encode(" ")[0], task.tokenizer.eot],
            logit_filters=[type(f).__name__ for f in task.logit_filters],
        )
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=["testclean_200", "varied", "all"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=DATA / "ref")
    a = ap.parse_args()
    clips: list[Clip] = []
    if a.set in ("testclean_200", "all"):
        clips += testclean_clips()
    if a.set in ("varied", "all"):
        clips += varied_clips()
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "_meta.json", "w") as f:
        json.dump(dump_meta(), f, indent=1, sort_keys=True)
        f.write("\n")
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(transcribe, clips, chunksize=4):
            with open(a.out / ref_path(r["uid"]).name, "w") as f:
                json.dump(r, f, indent=1, sort_keys=True, ensure_ascii=False)
                f.write("\n")
            print(f"{r['uid']:40s} {r['duration_s']:6.2f}s  {len(r['tokens']):3d} tok  {r['text']}")


if __name__ == "__main__":
    main()
