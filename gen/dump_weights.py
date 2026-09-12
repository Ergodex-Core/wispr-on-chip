"""Emit the quantised model as text hex files under weights/ plus MANIFEST.json (see docs/tiling.md).

  uv run python gen/dump_weights.py            # (re)generate weights/*.hex + MANIFEST.json
  uv run python gen/dump_weights.py --check    # rebuild in memory, verify every committed file's sha256

Deterministic given tiny.pt + weights/calib_stats.json + the QConfig below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import REPO, WHISPER_ROOT  # noqa: E402
from golden.quant import D_MODEL, HEAD_DIM, N_HEAD, QConfig, QModel, build  # noqa: E402

WDIR = REPO / "weights"
DEFAULT_CFG = QConfig()   # the frozen configuration (see docs/decisions.md)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def words_to_hex(words: np.ndarray) -> bytes:
    """uint32 array -> text, one 8-digit lowercase hex word per line."""
    w = np.asarray(words, dtype=np.uint32)
    return ("\n".join(f"{int(x):08x}" for x in w) + "\n").encode()


def pack_w8(w8: np.ndarray) -> tuple[np.ndarray, int]:
    """W[K,N] int8 -> uint32 words in tile-row-group order (docs/tiling.md). Returns (words, n_mem_words)."""
    K, N = w8.shape
    assert K % 32 == 0 and N % 32 == 0
    KT, NT = K // 32, N // 32
    # reshape to [nt, kt, g, r, c] bytes
    t = w8.reshape(KT, 4, 8, NT, 32)                    # k = kt*32 + g*8 + r ; n = nt*32 + c
    t = t.transpose(3, 0, 1, 2, 4)                       # [nt, kt, g, r, c]
    b = np.ascontiguousarray(t).reshape(-1).view(np.uint8)
    words = b.view("<u4")                                # little-endian 4-byte groups
    return words, KT * NT * 4


def unpack_w8(words: np.ndarray, K: int, N: int) -> np.ndarray:
    KT, NT = K // 32, N // 32
    b = np.asarray(words, dtype="<u4").view(np.uint8).view(np.int8)
    t = b.reshape(NT, KT, 4, 8, 32).transpose(1, 2, 3, 0, 4)
    return np.ascontiguousarray(t).reshape(K, N)


def pack_i32vec(v: np.ndarray) -> tuple[np.ndarray, int]:
    v = np.asarray(v, dtype=np.int64)
    n = ((len(v) + 31) // 32) * 32
    out = np.zeros(n, dtype=np.int64)
    out[: len(v)] = v
    assert out.min() >= -(1 << 31) and out.max() < (1 << 31)
    return out.astype(np.int32).view(np.uint32), n // 32


def pack_i8mat(m: np.ndarray) -> tuple[np.ndarray, int]:
    m = np.asarray(m, dtype=np.int8)
    rows, cols = m.shape
    assert cols % 32 == 0
    b = np.ascontiguousarray(m).reshape(-1).view(np.uint8)
    return b.view("<u4"), rows * cols // 32


def collect(qm: QModel) -> dict[str, dict]:
    """name -> {kind, array, meta}"""
    T: dict[str, dict] = {}

    def add(name, kind, arr, **meta):
        assert name not in T, name
        T[name] = dict(kind=kind, arr=arr, meta=meta)

    for name, L in qm.linears.items():
        add(f"{name}.w", "w8", L.w8, shape=list(L.w8.shape), s_w=None, dynamic=L.dynamic, out_bits=L.out_bits,
            s1=L.s1, s_in=L.s_in)
        add(f"{name}.mult", "i32vec", L.mult, shape=[len(L.mult)])
        if L.out_bits != 0:
            add(f"{name}.bias", "i32vec", L.bias, shape=[len(L.bias)])
    for name, ln in qm.lns.items():
        add(f"{name}.g", "i32vec", ln.g, shape=[D_MODEL], F=ln.F, eps_q=ln.eps_q)
        add(f"{name}.b", "i32vec", ln.b, shape=[D_MODEL])
    for name, g in qm.gelus.items():
        if g.smooth is not None:
            add(f"{name}.smooth", "i32vec", g.smooth, shape=[len(g.smooth)])
        if g.table is not None:
            add(f"{name}.lut8", "i32vec", np.asarray(g.table), shape=[256])
    add("enc.pos", "i8mat", qm.tables["enc.pos"], shape=list(qm.tables["enc.pos"].shape), scale=qm.scalars["s_enc_pos"])
    add("dec.pos", "i8mat", qm.tables["dec.pos"], shape=list(qm.tables["dec.pos"].shape), scale=qm.scalars["s_dec_pos"])
    add("dec.emb.resmult", "i32vec", qm.tables["dec.emb.resmult"], shape=[len(qm.tables["dec.emb.resmult"])])
    return T


def encode(name: str, t: dict) -> tuple[bytes, dict]:
    kind, arr = t["kind"], t["arr"]
    if kind == "w8":
        words, depth = pack_w8(arr)
        width = 2048
    elif kind == "i32vec":
        words, depth = pack_i32vec(arr)
        width = 1024
    elif kind == "i8mat":
        words, depth = pack_i8mat(arr)
        width = 256
    else:
        raise ValueError(kind)
    data = words_to_hex(words)
    entry = dict(kind=kind, file=f"{name}.hex", width_bits=width, depth=depth, words32=len(words),
                 sha256=sha256_bytes(data), zero_point=0, **t["meta"])
    return data, entry


def op_params(qm: QModel) -> dict:
    """Scalar op parameters (consumed by gen/emit_microcode.py)."""
    P: dict = {"linears": {}, "lns": {}, "attns": {}, "adds": {}, "gelus": {}, "scalars": qm.scalars}
    for n, L in qm.linears.items():
        P["linears"][n] = dict(K=int(L.w8.shape[0]), N=int(L.w8.shape[1]), s1=int(L.s1), out_bits=int(L.out_bits),
                               dynamic=bool(L.dynamic), s_in=float(L.s_in), s_out=[float(x) for x in np.unique(L.s_out)][:8])
    for n, ln in qm.lns.items():
        P["lns"][n] = dict(F=int(ln.F), eps_q=int(ln.eps_q), s_in=float(ln.s_in))
    for n, a in qm.attns.items():
        P["attns"][n] = dict(mq=[int(x) for x in a.mq], sq=[int(x) for x in a.sq], s_q=[float(x) for x in a.s_q],
                             s_k=[float(x) for x in a.s_k], s_v=float(a.s_v))
    for n, a in qm.adds.items():
        P["adds"][n] = dict(ma=int(a.ma), mb=int(a.mb), out_bits=int(a.out_bits), s_a=float(a.s_a), s_b=float(a.s_b), s_out=float(a.s_out))
    for n, g in qm.gelus.items():
        P["gelus"][n] = dict(mode=g.mode, m_phi=int(g.m_phi), s_phi=int(g.s_phi), req_mult=int(g.req_mult),
                             req_shift=int(g.req_shift), s_in=float(g.s_in), s_out=float(g.s_out),
                             has_smooth=g.smooth is not None)
    P["decoding"] = qm.meta["decoding"]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    qm = build(DEFAULT_CFG)
    T = collect(qm)
    manifest = dict(
        version=1,
        checkpoint_sha256=sha256_bytes(open(WHISPER_ROOT / "tiny.pt", "rb").read()),
        calib_stats_sha256=sha256_bytes(open(WDIR / "calib_stats.json", "rb").read()),
        config=DEFAULT_CFG.__dict__,
        hex_format="one 32-bit little-endian word per line; memory words are width_bits/32 consecutive lines",
        tensors={},
        params=op_params(qm),
    )
    bad = 0
    total_bytes = 0
    for name in sorted(T):
        data, entry = encode(name, T[name])
        manifest["tensors"][name] = entry
        total_bytes += len(data)
        path = WDIR / entry["file"]
        if a.check:
            if not path.exists() or sha256_bytes(path.read_bytes()) != entry["sha256"]:
                print(f"MISMATCH {name}")
                bad += 1
        else:
            path.write_bytes(data)
    mtxt = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    if a.check:
        old = json.load(open(WDIR / "MANIFEST.json"))
        for k in ("tensors", "config", "checkpoint_sha256", "calib_stats_sha256"):
            if old[k] != manifest[k]:
                print(f"MANIFEST field differs: {k}")
                bad += 1
        print("weights check:", "OK" if bad == 0 else f"{bad} problems", f"({len(T)} tensors, {total_bytes/1e6:.1f} MB hex)")
        sys.exit(1 if bad else 0)
    (WDIR / "MANIFEST.json").write_bytes(mtxt)
    print(f"wrote {len(T)} tensors, {total_bytes/1e6:.1f} MB of hex to {WDIR}")
    # round-trip self-test on the largest and one small tensor
    for n in ("dec.lm.w", "enc.0.attn.q.w"):
        w8 = qm.linears[n[:-2]].w8
        words = np.array([int(l, 16) for l in open(WDIR / f"{n}.hex")], dtype=np.uint32)
        assert np.array_equal(unpack_w8(words, *w8.shape), w8), n
    print("round-trip OK")


if __name__ == "__main__":
    main()
