"""Emit the quantised model as binary word images under weights/ plus MANIFEST.json (docs/tiling.md).

  uv run python gen/dump_weights.py            # (re)generate weights/*.bin + MANIFEST.json (~2.4 GB)
  uv run python gen/dump_weights.py --check    # rebuild in memory, verify every file's sha256 + the manifest

Deterministic given the checkpoint + weights/calib_stats.json + the QConfig below. The .bin files are not
committed (2.4 GB of int8); MANIFEST.json is, so `--check` proves a regenerated set is identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import CHECKPOINT, EOS_IDS, REPO  # noqa: E402
from golden.quant import (D_MODEL, EMB_SLICE, HEAD_DIM, LM_SLICE, MAX_CTX, N_HEAD, N_KV_HEAD, N_LAYERS, N_VOCAB,  # noqa: E402
                          QConfig, QModel, build)

WDIR = REPO / "weights"
DEFAULT_CFG = QConfig()   # the frozen configuration (see docs/decisions.md)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_w8(w8: np.ndarray) -> tuple[np.ndarray, int]:
    """W[K,N] int8 -> uint32 words in tile-row-group order (docs/tiling.md). Returns (words, n_mem_words)."""
    K, N = w8.shape
    assert K % 32 == 0 and N % 32 == 0
    KT, NT = K // 32, N // 32
    t = w8.reshape(KT, 4, 8, NT, 32)                    # k = kt*32 + g*8 + r ; n = nt*32 + c
    t = t.transpose(3, 0, 1, 2, 4)                       # [nt, kt, g, r, c]
    b = np.ascontiguousarray(t).reshape(-1).view(np.uint8)
    return b.view("<u4"), KT * NT * 4


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


def pack_i16mat(m: np.ndarray) -> tuple[np.ndarray, int]:
    m = np.asarray(m, dtype="<i2")
    rows, cols = m.shape
    assert cols % 16 == 0
    b = np.ascontiguousarray(m).view(np.uint8).reshape(-1)
    return b.view("<u4"), rows * cols // 16


def collect(qm: QModel) -> dict[str, dict]:
    """name -> {kind, array, meta}"""
    T: dict[str, dict] = {}

    def add(name, kind, arr, **meta):
        assert name not in T, name
        T[name] = dict(kind=kind, arr=arr, meta=meta)

    for name, L in qm.linears.items():
        if name == "lm":
            for s in range(0, L.w8.shape[1], LM_SLICE):
                i = s // LM_SLICE
                add(f"lm.{i:02d}.w", "w8", L.w8[:, s:s + LM_SLICE], shape=[L.w8.shape[0], min(LM_SLICE, L.w8.shape[1] - s)],
                    dynamic=L.dynamic, out_bits=0, s1=L.s1, s_in=L.s_in, col0=s)
                add(f"lm.{i:02d}.mult", "i32vec", L.mult[s:s + LM_SLICE], shape=[min(LM_SLICE, L.w8.shape[1] - s)])
            continue
        add(f"{name}.w", "w8", L.w8, shape=list(L.w8.shape), dynamic=L.dynamic, out_bits=L.out_bits, s1=L.s1, s_in=L.s_in)
        add(f"{name}.mult", "i32vec", L.mult, shape=[len(L.mult)])
    for name, nq in qm.norms.items():
        add(f"{name}.g", "i32vec", nq.g, shape=[D_MODEL], F=nq.F, eps_q=nq.eps_q)
    e8 = qm.tables["embed"]
    for s in range(0, e8.shape[0], EMB_SLICE):
        i = s // EMB_SLICE
        add(f"embed.{i:02d}", "i8mat", e8[s:s + EMB_SLICE], shape=[min(EMB_SLICE, e8.shape[0] - s), D_MODEL], row0=s)
    add("embed.resmult", "i32vec", qm.tables["embed.resmult"], shape=[N_VOCAB])
    add("rope", "i16mat", qm.tables["rope"], shape=list(qm.tables["rope"].shape), one=32767)
    return T


def encode(name: str, t: dict) -> tuple[bytes, dict]:
    kind, arr = t["kind"], t["arr"]
    if kind == "w8":
        words, depth = pack_w8(arr); width = 2048
    elif kind == "i32vec":
        words, depth = pack_i32vec(arr); width = 1024
    elif kind == "i8mat":
        words, depth = pack_i8mat(arr); width = 256
    elif kind == "i16mat":
        words, depth = pack_i16mat(arr); width = 256
    else:
        raise ValueError(kind)
    data = np.ascontiguousarray(words.astype("<u4")).tobytes()
    entry = dict(kind=kind, file=f"{name}.bin", width_bits=width, depth=depth, words32=len(words),
                 sha256=sha256_bytes(data), zero_point=0, **t["meta"])
    return data, entry


def op_params(qm: QModel) -> dict:
    """Scalar op parameters (consumed by gen/emit_microcode.py and the RTL vector generators)."""
    P: dict = {"linears": {}, "norms": {}, "ropes": {}, "attns": {}, "silus": {}, "adds": {}, "scalars": qm.scalars}
    for n, L in qm.linears.items():
        P["linears"][n] = dict(K=int(L.w8.shape[0]), N=int(L.w8.shape[1]), s1=int(L.s1), out_bits=int(L.out_bits),
                               dynamic=bool(L.dynamic), s_in=float(L.s_in), s_out=[float(x) for x in np.unique(L.s_out)][:4])
    for n, q in qm.norms.items():
        P["norms"][n] = dict(F=int(q.F), eps_q=int(q.eps_q), s_in=float(q.s_in))
    for n, r in qm.ropes.items():
        P["ropes"][n] = dict(m_r=int(r.m_r), s_r=int(r.s_r), n_heads=int(r.n_heads), s16=[float(x) for x in r.s16], s8=[float(x) for x in r.s8])
    for n, a in qm.attns.items():
        P["attns"][n] = dict(mq=[int(x) for x in a.mq], sq=[int(x) for x in a.sq], s_q=[float(x) for x in a.s_q],
                             s_k=[float(x) for x in a.s_k], s_v=float(a.s_v))
    for n, s in qm.silus.items():
        P["silus"][n] = dict(m_sig=int(s.m_sig), s_sig=int(s.s_sig), s_gate=float(s.s_gate), s_up=float(s.s_up), s_out=float(s.s_out))
    for n, a in qm.adds.items():
        P["adds"][n] = dict(ma=int(a.ma), mb=int(a.mb), out_bits=int(a.out_bits), s_a=float(a.s_a), s_b=float(a.s_b), s_out=float(a.s_out))
    P["model"] = dict(d_model=D_MODEL, n_head=N_HEAD, n_kv_head=N_KV_HEAD, head_dim=HEAD_DIM, n_layers=qm.cfg.n_layers,
                      n_vocab=N_VOCAB, max_ctx=MAX_CTX, lm_slice=LM_SLICE, emb_slice=EMB_SLICE)
    P["decoding"] = dict(eos_ids=list(EOS_IDS), bos=0)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    qm = build(DEFAULT_CFG)
    T = collect(qm)
    manifest = dict(
        version=1,
        checkpoint_sha256=sha256_file(CHECKPOINT),
        calib_stats_sha256=sha256_file(WDIR / "calib_stats.json"),
        config=DEFAULT_CFG.__dict__,
        format="little-endian 32-bit words; memory words are width_bits/32 consecutive words",
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
            if not path.exists() or sha256_file(path) != entry["sha256"]:
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
        print("weights check:", "OK" if bad == 0 else f"{bad} problems", f"({len(T)} tensors, {total_bytes/1e6:.1f} MB)")
        sys.exit(1 if bad else 0)
    (WDIR / "MANIFEST.json").write_bytes(mtxt)
    print(f"wrote {len(T)} tensors, {total_bytes/1e6:.1f} MB to {WDIR}")
    # round-trip self-test on one small and one large tensor
    for n in ("L0.k", "L41.down"):
        w8 = qm.linears[n].w8
        words = np.fromfile(WDIR / f"{n}.w.bin", dtype="<u4")
        assert np.array_equal(unpack_w8(words, *w8.shape), w8), n
    print("round-trip OK")


if __name__ == "__main__":
    main()
