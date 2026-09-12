"""Compare RTL bank dumps taken at sequencer breakpoints (E2EDebugSpec) with golden per-op dumps.
Usage: uv run python tests/e2e/compare_dumps.py <clip dir> <golden npz>
Bank contents *before* executing pc (see docs/microcode.txt):
  pc 4 : bank1 A8 = enc.conv1.gelu (frames)          pc 5 : bank4 T16 = enc.conv2.out (rows)
  pc 6 : bank2 X = enc.x0                            pc 7 : bank1 A8 = enc.0.attn.in (+rf)
  pc 8 : bank3 Q8 = enc.0.attn.q                     pc 13: bank4 T16 = enc.0.attn.out
  pc 15: bank4 T16 = enc.0.attn.o.out ; bank1 A8 = enc.0.attn.o.in (+rf)
  pc 16: bank2 X = enc.0.add1.out                    pc 17: bank1 A8 = enc.0.fc1.in (+rf)
  pc 22: bank5 H16 = enc.0.fc1.out ; bank6 A8X = enc.0.fc2.in (+rf) ; bank4 T16 = enc.0.fc2.out ; bank2 X = enc.0.add2.out (first chunk rows)
  end  : bank7 E8 = enc.out (+rf)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.layout import act8_words, act16_words  # noqa: E402

TABLE = {
    "pc4": [("enc.conv1.gelu", 1, 12, False, None)],
    "pc5": [("enc.conv2.out", 4, 24, True, None)],
    "pc6": [("enc.x0", 2, 24, True, None)],
    "pc7": [("enc.0.attn.in", 1, 12, False, "enc.0.attn.rf")],
    "pc8": [("enc.0.attn.q", 3, 12, False, None)],
    "pc13": [("enc.0.attn.out", 4, 24, True, None)],
    "pc15": [("enc.0.attn.o.out", 4, 24, True, None), ("enc.0.attn.o.in", 1, 12, False, "enc.0.attn.o.rf")],
    "pc16": [("enc.0.add1.out", 2, 24, True, None)],
    "pc17": [("enc.0.fc1.in", 1, 12, False, "enc.0.fc1.rf")],
    "pc22": [("enc.0.fc1.out", 5, 96, True, None), ("enc.0.fc2.in", 6, 48, False, "enc.0.fc2.rf"), ("enc.0.fc2.out", 4, 24, True, None), ("enc.0.add2.out", 2, 24, True, None)],
    "end": [("enc.out", 7, 12, False, "enc.out.rf")],
}


# decoder scratch rows (absolute 256-bit word addresses): (bank, base, stride)
DEC = dict(XD=(2, 0, 24), AD8=(1, 32, 12), QD8=(3, 0, 12), TD16=(4, 0, 24), OD16=(4, 64, 24), HD16=(5, 0, 96), AD8X=(6, 0, 48))
DEC_TABLE = {   # tag -> [(golden name, scratch, bits16, rowfac name or None)]
    "pc94_pos0": [("dec.pos0.x0", "XD", True, None)],
    "pc95_pos0": [("dec.pos0.0.attn.in", "AD8", False, "dec.pos0.0.attn.rf")],
    "pc96_pos0": [("dec.pos0.0.attn.q", "QD8", False, None)],
    "pc101_pos0": [("dec.pos0.0.attn.out", "TD16", True, None)],
    "pc103_pos0": [("dec.pos0.0.attn.o.out", "OD16", True, None)],
    "pc104_pos0": [("dec.pos0.0.add1.out", "XD", True, None)],
    "pc105_pos0": [("dec.pos0.0.xattn.in", "AD8", False, "dec.pos0.0.xattn.rf")],
    "pc106_pos0": [("dec.pos0.0.xattn.q", "QD8", False, None)],
    "pc107_pos0": [("dec.pos0.0.xattn.out", "TD16", True, None)],
    "pc109_pos0": [("dec.pos0.0.xattn.o.out", "OD16", True, None)],
    "pc110_pos0": [("dec.pos0.0.add2.out", "XD", True, None)],
    "pc111_pos0": [("dec.pos0.0.fc1.in", "AD8", False, "dec.pos0.0.fc1.rf")],
    "pc112_pos0": [("dec.pos0.0.fc1.out", "HD16", True, None)],
    "pc113_pos0": [("dec.pos0.0.fc2.in", "AD8X", False, "dec.pos0.0.fc2.rf")],
    "pc114_pos0": [("dec.pos0.0.fc2.out", "OD16", True, None)],
    "pc115_pos0": [("dec.pos0.0.add3.out", "XD", True, None)],
    "pc180_pos3": [("dec.pos3.lm.in", "AD8", False, None)],
}


def load(path):
    words, rf = {}, {}
    for line in open(path):
        p = line.split()
        if p[0] == "rf":
            rf[(int(p[1]), int(p[2]))] = int(p[3])
        else:
            words[(int(p[0]), int(p[1]), int(p[2]))] = int(p[3], 16)
    return words, rf


def main():
    cd = Path(sys.argv[1]); npz = np.load(sys.argv[2])
    for tag, items in TABLE.items():
        f = cd / f"rtl_banks_{tag}.txt"
        if not f.exists():
            continue
        words, rf = load(f)
        rows = max(r for (_, r, _) in words) + 1
        for name, bank, stride, b16, rfname in items:
            g = npz[name]
            n = min(rows, g.shape[0])
            bad = []
            for r in range(n):
                gw = act16_words(g[r:r + 1]) if b16 else act8_words(g[r:r + 1])
                gwords = [int.from_bytes(gw[8 * i:8 * i + 8].tobytes(), "little") for i in range(len(gw) // 8)]
                got = [words[(bank, r, w)] for w in range(stride)]
                if got[:len(gwords)] != gwords:
                    bad.append(r)
            msg = "OK" if not bad else f"{len(bad)}/{n} rows differ, first {bad[:6]}"
            print(f"{tag:5s} {name:20s} bank {bank}: {msg}")
            if bad:
                r = bad[0]
                gw = act16_words(g[r:r + 1]) if b16 else act8_words(g[r:r + 1])
                gwords = [int.from_bytes(gw[8 * i:8 * i + 8].tobytes(), "little") for i in range(len(gw) // 8)]
                got = [words[(bank, r, w)] for w in range(stride)]
                print(f"        row {r} word0 got {got[0]:064x}\n        row {r} word0 exp {gwords[0]:064x}")
            if rfname:
                rfb = [r for r in range(n) if rf.get((bank, r)) != int(npz[rfname][r])]
                print(f"{tag:5s} {rfname:20s} rowfac {bank}: {'OK' if not rfb else f'{len(rfb)} rows differ, first {rfb[:6]} got {[rf.get((bank, r)) for r in rfb[:3]]} exp {[int(npz[rfname][r]) for r in rfb[:3]]}'}")


def decoder_compare(cd, npz):
    for tag, items in DEC_TABLE.items():
        f = cd / f"rtl_banks_{tag}.txt"
        if not f.exists():
            continue
        words, rf = load(f)
        # absolute address map
        absw = {}
        for (b, r, w), v in words.items():
            stride = {1: 12, 2: 24, 3: 12, 4: 24, 5: 96, 6: 48, 7: 12}[b]
            absw[(b, r * stride + w)] = v
        for name, scratch, b16, rfname in items:
            bank, base, stride = DEC[scratch]
            g = npz[name][0:1]
            gw = act16_words(g) if b16 else act8_words(g)
            gwords = [int.from_bytes(gw[8 * i:8 * i + 8].tobytes(), "little") for i in range(len(gw) // 8)]
            got = [absw.get((bank, base + w)) for w in range(len(gwords))]
            ok = got == gwords
            print(f"{tag:12s} {name:24s} {scratch:5s}: {'OK' if ok else 'DIFF'}")
            if not ok:
                print(f"        word0 got {got[0]:064x}\n        word0 exp {gwords[0]:064x}")
            if rfname:
                rrow = base // 12
                got_rf = rf.get((bank, rrow)); exp_rf = int(npz[rfname][0])
                print(f"{tag:12s} {rfname:24s} rowfac: {'OK' if got_rf == exp_rf else f'DIFF got {got_rf} exp {exp_rf}'}")


if __name__ == "__main__":
    main()
    decoder_compare(Path(sys.argv[1]), np.load(sys.argv[2]))
