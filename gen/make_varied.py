"""Build data/varied/: >=10 committed 16 kHz clips of different character.

Deterministic: sources are fixed LibriSpeech test-clean / FLEURS utterances; noise is seeded.
Run: uv run python gen/make_varied.py
"""
from __future__ import annotations

import csv
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import CACHE, DATA, SR, libri_clip, load_audio  # noqa: E402

OUT = DATA / "varied"
OUT.mkdir(parents=True, exist_ok=True)

# (id, librispeech uid); durations measured: 2.0 s F, 2.4 s M, 5.0 s M, 5.0 s F, 12.0 s F, 12.0 s M, 29.1 s M
LIBRI_SEL = [
    ("en_2s_f", "121-127105-0021"),
    ("en_5s_m", "2830-3980-0068"),
    ("en_12s_f", "1284-1180-0018"),
    ("en_29s_m", "5105-28241-0015"),
    ("en_12s_m", "4077-13751-0012"),
]
NOISE_SEL = [
    ("en_5s_f_white10db", "4992-23283-0010", "white"),
    ("en_5s_m_babble10db", "6930-81414-0020", "babble"),
]
BABBLE_SRC = ["1089-134686-0000", "1188-133604-0000", "1221-135766-0000", "1320-122617-0000"]
SILENCE_SEL = ("en_silence", "1580-141084-0011", "121-127105-0021")  # clip A + 8 s silence + clip B
FLEURS_SEL = [("de", "de_de"), ("fr", "fr_fr")]
# FLEURS raw transcription typos fixed by hand (the corpus text has a stray trailing character).
TEXT_FIX = {
    "10229344228128634115.wav": "Jeder nimmt an der Gesellschaft teil und benutzt Transportsysteme. Fast jeder beklagt sich über die Transportsysteme.",
}


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise[: len(x)] if len(noise) >= len(x) else np.resize(noise, len(x))
    ps = np.mean(x ** 2)
    pn = np.mean(noise ** 2)
    g = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    y = x + g * noise
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def write(name: str, audio: np.ndarray) -> str:
    fn = f"{name}.flac"
    sf.write(str(OUT / fn), audio, SR, subtype="PCM_16")
    return fn


def fleurs_clip(lang: str, dirname: str, rng: np.random.Generator):
    """Pick the first FLEURS test utterance with duration in [4, 8] s (sorted by id) for determinism."""
    tsv = CACHE / "fleurs" / f"{dirname}-test.tsv"
    rows = list(csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    # columns: id, file_name, raw transcription, transcription, chars, num_samples, gender
    rows.sort(key=lambda r: r[1])
    tar = tarfile.open(CACHE / "fleurs" / f"{dirname}-test.tar.gz")
    names = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
    for r in rows:
        n = int(r[5])
        if 4 * SR <= n <= 8 * SR and r[1] in names:
            f = tar.extractfile(names[r[1]])
            audio, sr = sf.read(f, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == SR
            text = TEXT_FIX.get(r[1], r[2])
            return f"{lang}_fleurs_{Path(r[1]).stem}", audio.astype(np.float32), text, r[1]
    raise RuntimeError("no fleurs clip found")


def main():
    rng = np.random.default_rng(20260907)
    man = []
    for cid, uid in LIBRI_SEL:
        c = libri_clip(uid)
        fn = write(cid, load_audio(c.path))
        man.append(dict(id=cid, file=fn, text=c.text, language="en", source=uid, kind="clean"))
    babble = np.concatenate([load_audio(libri_clip(u).path) for u in BABBLE_SRC])
    for cid, uid, kind in NOISE_SEL:
        c = libri_clip(uid)
        x = load_audio(c.path)
        if kind == "white":
            noise = rng.standard_normal(len(x)).astype(np.float32)
        else:
            # babble = sum of 4 other speakers, random offsets
            noise = np.zeros(len(x), np.float32)
            for _ in range(4):
                off = int(rng.integers(0, len(babble) - len(x)))
                noise += babble[off: off + len(x)]
        fn = write(cid, add_noise(x, noise, 10.0))
        man.append(dict(id=cid, file=fn, text=c.text, language="en", source=uid, kind=f"{kind}_10dB", seed=20260907))
    cid, ua, ub = SILENCE_SEL
    a, b = libri_clip(ua), libri_clip(ub)
    audio = np.concatenate([load_audio(a.path), np.zeros(8 * SR, np.float32), load_audio(b.path)])
    fn = write(cid, audio)
    man.append(dict(id=cid, file=fn, text=a.text + " " + b.text, language="en", source=f"{ua}+8s_silence+{ub}", kind="silence"))
    for lang, dirname in FLEURS_SEL:
        cid, audio, text, src = fleurs_clip(lang, dirname, rng)
        fn = write(cid, audio)
        man.append(dict(id=cid, file=fn, text=text, language=lang, source=f"fleurs/{dirname}/{src}", kind="non_english"))
    for m in man:
        m["duration_s"] = round(sf.info(str(OUT / m["file"])).duration, 3)
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
