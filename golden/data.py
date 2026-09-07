"""Shared data helpers: paths, audio loading, LibriSpeech/FLEURS access, text normalisation.

Large corpora live outside the repo in WHISPER_SI_DATA (default /home/user/data-cache);
small clips are committed under data/varied.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CACHE = Path(os.environ.get("WHISPER_SI_DATA", "/home/user/data-cache"))
LIBRI = CACHE / "librispeech" / "LibriSpeech" / "test-clean"
WHISPER_ROOT = CACHE / "whisper"
SR = 16000


@dataclass(frozen=True)
class Clip:
    uid: str          # unique id, e.g. "1089-134686-0000" or "varied/en_5s_m"
    path: Path        # audio file (flac/wav, 16 kHz)
    text: str         # reference transcript (raw)
    language: str     # whisper language code, e.g. "en", "de"


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    return audio.astype(np.float32)


def libri_transcripts(spk: str, chap: str) -> dict[str, str]:
    d = LIBRI / spk / chap
    out = {}
    with open(d / f"{spk}-{chap}.trans.txt") as f:
        for line in f:
            uid, text = line.strip().split(" ", 1)
            out[uid] = text
    return out


def libri_clip(uid: str) -> Clip:
    spk, chap, _ = uid.split("-")
    text = libri_transcripts(spk, chap)[uid]
    return Clip(uid, LIBRI / spk / chap / f"{uid}.flac", text, "en")


def libri_gender() -> dict[str, str]:
    g = {}
    with open(LIBRI.parent / "SPEAKERS.TXT") as f:
        for line in f:
            if line.startswith(";"):
                continue
            p = [x.strip() for x in line.split("|")]
            if len(p) >= 3 and p[2] == "test-clean":
                g[p[0]] = p[1]
    return g


def testclean_list(name: str = "testclean_200.txt") -> list[str]:
    return [l.strip() for l in open(DATA / name) if l.strip()]


def testclean_clips(name: str = "testclean_200.txt") -> list[Clip]:
    return [libri_clip(u) for u in testclean_list(name)]


def varied_clips() -> list[Clip]:
    man = json.load(open(DATA / "varied" / "manifest.json"))
    return [Clip("varied/" + m["id"], DATA / "varied" / m["file"], m["text"], m["language"]) for m in man]


def clip_by_uid(uid: str) -> Clip:
    if uid.startswith("varied/"):
        for c in varied_clips():
            if c.uid == uid:
                return c
        raise KeyError(uid)
    return libri_clip(uid)


def ref_path(uid: str) -> Path:
    return DATA / "ref" / (uid.replace("/", "__") + ".json")


_normalizers: dict[str, object] = {}


def normalize_text(text: str, language: str) -> str:
    """Whisper's normalizers: English one for en (standard for LibriSpeech WER), basic otherwise."""
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    key = "en" if language == "en" else "basic"
    if key not in _normalizers:
        _normalizers[key] = EnglishTextNormalizer() if key == "en" else BasicTextNormalizer()
    return re.sub(r"\s+", " ", _normalizers[key](text)).strip()
