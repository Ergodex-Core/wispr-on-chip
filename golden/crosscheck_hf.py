"""Cross-check: transformers WhisperForConditionalGeneration (HF mirror) vs openai-whisper reference tokens.

Both are fp32 greedy on the same 30 s window; front-ends differ slightly (HF numpy STFT vs torch STFT),
so agreement is expected to be high but not necessarily 100 %. Reported in docs/status.md.
Usage: uv run python golden/crosscheck_hf.py [--n 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import CACHE, load_audio, normalize_text, ref_path, testclean_clips, varied_clips  # noqa: E402

HF = CACHE / "hf-whisper-tiny"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(4)
    proc = WhisperProcessor.from_pretrained(str(HF))
    model = WhisperForConditionalGeneration.from_pretrained(str(HF), torch_dtype=torch.float32).eval()
    # weight cross-check against the openai checkpoint
    import whisper
    om = whisper.load_model("tiny", device="cpu", download_root=str(CACHE / "whisper"))
    d = (om.decoder.token_embedding.weight - model.model.decoder.embed_tokens.weight).abs().max().item()
    d2 = (om.encoder.blocks[0].mlp[0].weight - model.model.encoder.layers[0].fc1.weight).abs().max().item()
    print(f"weight max-abs diff: token_embedding={d:.3e} enc0.fc1={d2:.3e}")
    clips = testclean_clips()[: a.n] + varied_clips()
    same, total, wer_pairs = 0, 0, []
    for c in clips:
        audio = load_audio(c.path)
        feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, language=c.language, task="transcribe", do_sample=False, num_beams=1,
                                 return_timestamps=False, max_new_tokens=224)[0].tolist()
        ref = json.load(open(ref_path(c.uid)))
        hf_gen = [t for t in ids if t < 50257]  # strip special tokens for comparison with generated text tokens
        ref_gen = [t for t in ref["tokens"] if t < 50257]
        total += 1
        same += int(hf_gen == ref_gen)
        wer_pairs.append((normalize_text(ref["text"], c.language), normalize_text(proc.tokenizer.decode(ids, skip_special_tokens=True), c.language)))
        if hf_gen != ref_gen:
            print(f"DIFF {c.uid}: hf={proc.tokenizer.decode(ids, skip_special_tokens=True)!r} ref={ref['text']!r}")
    import jiwer
    print(json.dumps(dict(n=total, token_identity=same / total, wer_hf_vs_openai=jiwer.wer([r for r, _ in wer_pairs], [h for _, h in wer_pairs]))))


if __name__ == "__main__":
    main()
