"""Compare RTL bank dumps (E2EDebugSpec: rtl_banks.txt) with golden per-op dumps (run_golden --dump).

Bank contents after a full run (encoder rows >= 4 are untouched by the decoder scratch rows):
  bank 7 E8  = enc.out (+ rowfac enc.out.rf)        bank 2 X = enc.3.add2.out
  bank 1 A8  = enc.3.fc1.in (+rf)                    bank 3 Q8 = enc.3.attn.q
  bank 5 H16 = enc.3.fc1.out (last FFN chunk, local rows)   bank 6 A8X = enc.3.fc2.in (local)
  bank 4 T16 = enc.3.fc2.out (local rows)
Usage: uv run python tests/e2e/compare_dumps.py out/e2e/smoke_var/varied__en_2s_f out/dumps/smoke/varied__en_2s_f.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.layout import act8_words, act16_words  # noqa: E402


def main():
    cd = Path(sys.argv[1]); npz = np.load(sys.argv[2])
    words = {}; rf = {}
    for line in open(cd / "rtl_banks.txt"):
        p = line.split()
        if p[0] == "rf":
            rf[(int(p[1]), int(p[2]))] = int(p[3])
        else:
            words[(int(p[0]), int(p[1]), int(p[2]))] = int(p[3], 16)
    rows = max(r for (_, r, _) in words) + 1
    gold_tok = [int(x) for x in (cd / "golden_tokens.txt").read_text().split()]
    rtl_tok = [int(x) for x in (cd / "rtl_tokens.txt").read_text().split()] if (cd / "rtl_tokens.txt").exists() else []
    print("golden tokens:", gold_tok[:20]); print("rtl tokens   :", rtl_tok[:20])
    nctx = npz["enc.out"].shape[0]
    def bank_words(bank, stride, r):
        return [words[(bank, r, w)] for w in range(stride)]
    def cmp(name, bank, stride, arr, bits16, local=False, first_row=4):
        g = npz[name]
        n = min(rows, g.shape[0])
        bad = []
        for r in range(first_row, n):
            gw = (act16_words(g[r:r+1]) if bits16 else act8_words(g[r:r+1]))
            gwords = [int.from_bytes(gw[8*i:8*i+8].tobytes(), "little") for i in range(len(gw)//8)]
            got = bank_words(bank, stride, r)
            if got[:len(gwords)] != gwords:
                bad.append(r)
        print(f"{name:20s} bank {bank}: rows {first_row}..{n-1}: {'OK' if not bad else f'{len(bad)} rows differ, first {bad[:5]}'}")
        return bad
    cmp("enc.out", 7, 12, npz["enc.out"], False)
    rfb = [r for r in range(4, min(rows, nctx)) if rf.get((7, r)) != int(npz["enc.out.rf"][r])]
    print(f"{'enc.out.rf':20s} rowfac 7: {'OK' if not rfb else f'{len(rfb)} rows differ, first {rfb[:5]}'}")
    cmp("enc.3.add2.out", 2, 24, npz["enc.3.add2.out"], True)
    cmp("enc.3.fc1.in", 1, 12, npz["enc.3.fc1.in"], False)
    cmp("enc.3.attn.q", 3, 12, npz["enc.3.attn.q"], False)
    # earlier layers cannot be checked (overwritten); x0 / conv outputs likewise
    # last FFN chunk (local rows): chunk base = largest multiple of 512 below nctx
    cb = ((nctx - 1) // 512) * 512
    for name, bank, stride, b16 in [("enc.3.fc1.out", 5, 96, True), ("enc.3.fc2.in", 6, 48, False), ("enc.3.fc2.out", 4, 24, True)]:
        g = npz[name][cb:]
        cmp(name, bank, stride, g, b16, first_row=4)


if __name__ == "__main__":
    main()
