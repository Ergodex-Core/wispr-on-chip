"""End-to-end harness: mel -> int8 (as the golden does) -> Verilator WhisperTop -> tokens, compared with
the golden int model (must be identical) and with the fp32 CPU reference (WER).

  uv run python tests/vectors/../e2e/run_e2e.py --set default|long|<uids> [--frames var|full] [--prepare-only] [--compare-only]

Flow: 1) prepare out/e2e/<set>/<uid>/{mel.hex, meta.json, golden_tokens.txt} and out/e2e/<set>/clips.txt
      2) sbt "testOnly whisper.E2ESpec" with E2E_SET=<set> (one Verilator build, all clips in one sim)
      3) compare rtl_tokens.txt vs golden, decode text, WER vs ground truth and vs fp32
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.data import REPO, clip_by_uid, load_audio, normalize_text, ref_path, testclean_list, varied_clips  # noqa: E402
from golden.layout import write_hex  # noqa: E402
from golden.quant import QConfig, build  # noqa: E402
from golden.whisper_int import IntWhisper, n_frames_for  # noqa: E402

SETS = {
    "default": ["varied/en_2s_f", "varied/en_5s_m", "varied/en_12s_f"] + testclean_list("rtl_default.txt"),
    "long": ["varied/en_29s_m"] + testclean_list("rtl_20.txt"),
    "smoke": ["varied/en_2s_f"],
}
FRAME_RULE = dict(pad_frames=512, min_frames=1024)   # see docs/decisions.md


def prepare(set_name: str, uids: list[str], frames_mode: str, out: Path):
    import torch
    import whisper
    qm = build(QConfig())
    m = IntWhisper(qm, dump=False)
    lines = []
    for uid in uids:
        c = clip_by_uid(uid)
        audio = load_audio(c.path)
        mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)), n_mels=80).numpy()
        nf = n_frames_for(len(audio), frames_mode, **(FRAME_RULE if frames_mode == "var" else {}))
        mel8 = m.mel_quant(mel, nf)                                     # [nf, 96]
        d = out / uid.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        # mel.hex: per frame 80 bytes as 20 x 32-bit words (little-endian), frames consecutive
        words = mel8[:, :80].astype(np.int8).view(np.uint8).reshape(-1).view("<u4")
        write_hex(d / "mel.hex", words)
        t0 = time.time()
        toks = m.transcribe(mel, nf, c.language)
        lang = qm.meta["decoding"][c.language]["initial_tokens"][1]
        meta = dict(uid=uid, language=c.language, lang_token=int(lang), n_frames=int(nf), n_samples=len(audio),
                    golden_seconds=round(time.time() - t0, 1))
        json.dump(meta, open(d / "meta.json", "w"), indent=1)
        (d / "golden_tokens.txt").write_text(" ".join(map(str, toks)) + "\n")
        lines.append(d.name)
        for stale in ("rtl_tokens.txt", "rtl_stats.json"):   # never compare against a previous run's output
            (d / stale).unlink(missing_ok=True)
        print(f"prepared {uid}: n_frames={nf} golden tokens={len(toks)} ({meta['golden_seconds']} s)")
    (out / "clips.txt").write_text("\n".join(lines) + "\n")


def compare(set_name: str, uids: list[str], out: Path) -> dict:
    import jiwer
    import whisper
    rows = []
    for uid in uids:
        d = out / uid.replace("/", "__")
        meta = json.load(open(d / "meta.json"))
        gold = [int(x) for x in (d / "golden_tokens.txt").read_text().split()]
        rf = d / "rtl_tokens.txt"
        if not rf.exists():
            rows.append(dict(uid=uid, status="missing")); continue
        rtl = [int(x) for x in rf.read_text().split()]
        stats = json.load(open(d / "rtl_stats.json")) if (d / "rtl_stats.json").exists() else {}
        c = clip_by_uid(uid)
        tok = whisper.tokenizer.get_tokenizer(True, num_languages=99, language=c.language, task="transcribe")
        rtl_text = tok.decode(rtl).strip()
        ref = json.load(open(ref_path(uid)))
        wer = jiwer.wer(normalize_text(c.text, c.language), normalize_text(rtl_text, c.language)) if c.text.strip() else float("nan")
        rows.append(dict(uid=uid, status="ok", n_frames=meta["n_frames"], identical=(rtl == gold), n_tokens=len(rtl),
                         cpu_text=ref["text"], rtl_text=rtl_text, wer=wer, cycles=stats.get("cycles"), sim_seconds=stats.get("seconds"),
                         cpu_match=normalize_text(ref["text"], c.language) == normalize_text(rtl_text, c.language)))
    ok = [r for r in rows if r["status"] == "ok"]
    agg = dict(set=set_name, n=len(rows), ran=len(ok), identical=sum(r["identical"] for r in ok),
               wer=jiwer.wer([normalize_text(clip_by_uid(r["uid"]).text, clip_by_uid(r["uid"]).language) for r in ok],
                             [normalize_text(r["rtl_text"], clip_by_uid(r["uid"]).language) for r in ok]) if ok else None,
               rows=rows)
    json.dump(agg, open(out / "results.json", "w"), indent=1, ensure_ascii=False)
    for r in rows:
        if r["status"] == "ok":
            print(f"{r['uid']:36s} nf={r['n_frames']:4d} tok={r['n_tokens']:3d} identical={r['identical']!s:5s} wer={r['wer']:.3f} "
                  f"cycles={r['cycles']} sim={r['sim_seconds']}s | {r['rtl_text']}")
        else:
            print(f"{r['uid']:36s} {r['status']}")
    print(json.dumps({k: v for k, v in agg.items() if k != "rows"}))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="default")
    ap.add_argument("--frames", default="var", choices=["var", "full"])
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    uids = SETS.get(a.set) or a.set.split(",")
    set_name = a.set if a.set in SETS else "custom"
    out = REPO / "out" / "e2e" / f"{set_name}_{a.frames}"
    if not a.compare_only:
        prepare(set_name, uids, a.frames, out)
    if a.prepare_only:
        return
    rc = 0
    if not a.compare_only:
        env = dict(os.environ, E2E_DIR=str(out), WHISPER_SIM_THREADS=str(a.threads), PATH="/opt/sbt/bin:" + os.environ["PATH"])
        rc = subprocess.call(["sbt", "-batch", "testOnly whisper.E2ESpec"], cwd=REPO, env=env)
        print("sbt rc", rc)
    agg = compare(set_name, uids, out)
    # the harness fails unless every requested clip ran and its tokens are identical to the golden model
    ok = rc == 0 and agg["ran"] == agg["n"] and agg["identical"] == agg["n"]
    print("E2E", "PASS" if ok else "FAIL", f"({agg['identical']}/{agg['n']} identical, sbt rc {rc})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
