"""Generate every RTL unit-test vector set from one golden prefill (docs/status.md "verification ladder").

  uv run python tests/vectors/gen_all.py            # everything (builds the quantised model: ~5 min)
  uv run python tests/vectors/gen_all.py --only matmul,vector

Output: out/vectors/<unit>/<case>/{meta.json, *.hex, *.txt}. Hex files hold 32-bit words, one per line, in
the bank / ROM layouts of docs/tiling.md. Cases use layers 0, 20 and 41 of the real model on a real
128-token prompt, plus random shapes for the engine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gen.dump_weights import pack_i32vec, pack_w8  # noqa: E402
from golden.data import REPO, load_tokenizer, prompts, text_ids  # noqa: E402
from golden.fixedpoint import dyn_quant_rows, matmul_i8, requant, requant_wide, rowfac_pack  # noqa: E402
from golden.layout import act8_words, act16_words, act32_words, raw32_words, write_hex  # noqa: E402
from golden.quant import HEAD_DIM, KV_DIM, LM_SLICE, N_HEAD, N_KV_HEAD, QConfig, build  # noqa: E402
from golden.minicpm_int import IntMiniCPM, attention, dynq, embed_rows, rmsnorm, rope, scaled_add, silu_gate  # noqa: E402

OUT = REPO / "out" / "vectors"
DUMP_LAYERS = [0, 20, 41]
R = 40            # rows per vector-unit / real matmul case (keeps Verilator runs short)


def words(x, bits: int):
    return {8: act8_words, 16: act16_words, 32: act32_words}[bits](x)


def emit(unit: str, case: str, meta: dict, files: dict[str, np.ndarray], texts: dict[str, list] | None = None):
    d = OUT / unit / case
    d.mkdir(parents=True, exist_ok=True)
    for name, w in files.items():
        write_hex(d / name, w)
    for name, vals in (texts or {}).items():
        (d / name).write_text("\n".join(str(int(v)) for v in vals) + "\n")
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    print(f"  {unit}/{case}: " + ", ".join(f"{k}={v}" for k, v in meta.items() if k in ("M", "K", "N", "rows", "cols", "out_mode", "op", "n_queries", "n_keys")))


# ---------------------------------------------------------------------------------------------- matmul
def matmul_random(seed: int, M: int, K: int, N: int, out_mode: str, dynamic: bool, unsigned: bool = False, wide_rows: bool = False):
    rng = np.random.default_rng(seed)
    lo, hi = (0, 256) if unsigned else (-128, 128)
    a = rng.integers(lo, hi, size=(M, K)).astype(np.int64)
    w = rng.integers(-128, 128, size=(K, N)).astype(np.int8)
    acc = matmul_i8(a.astype(np.uint8).view(np.int8) if unsigned else a.astype(np.int8), w) if not unsigned else matmul_i8(a, w)
    bits = {"int8": 8, "int16": 16, "int32": 32, "wide": 0, "raw": 0}[out_mode]
    rowfac = np.ones(M, np.int64).astype(np.int32)
    if out_mode == "raw":
        mult = np.zeros(N, np.int32); s1 = 0
        y_words = raw32_words(acc.astype(np.int32))
    elif out_mode == "wide":
        mult = rng.integers(1 << 28, 1 << 31, N).astype(np.int32); s1 = 22
        y_words = raw32_words(requant_wide(acc, mult, s1))
    else:
        mult = rng.integers(1 << 26, 1 << 31, N).astype(np.int32)
        if dynamic:
            if wide_rows:   # 32-bit rows: exponent b > 0 in the row factor
                rowmax = rng.integers(1 << 17, 1 << 30, M).astype(np.int64)
            else:
                rowmax = rng.integers(1, 1 << 16, M).astype(np.int64)
            bl = np.array([int(v).bit_length() for v in rowmax]); b = np.maximum(bl - 16, 0)
            rowfac = rowfac_pack(rowmax >> b, b)
        s2 = {8: 24, 16: 20, 32: 24}[bits]
        eff = (rowfac & 0xFFFF).astype(np.int64) << (rowfac >> 16)
        mag = np.abs(acc).max() * float(mult.max()) * float(eff.max())
        s1 = max(0, min(63, int(np.ceil(np.log2(max(mag, 1)))) - s2 - (bits - 2)))
        y = requant(acc, mult, np.zeros(N, np.int32), s1, rowfac, bits)
        y_words = words(y, bits)
    a8 = a.astype(np.int8) if not unsigned else a.astype(np.uint8).view(np.int8)
    meta = dict(kind="random", seed=seed, M=M, K=K, N=N, out_mode=out_mode, dynamic=dynamic, unsigned=unsigned, s1=int(s1),
                a_stride=K // 32, y_stride={"int8": N // 32, "int16": N // 16, "int32": N // 8, "wide": N, "raw": N}[out_mode])
    emit("matmul", f"rand_{seed}_{M}x{K}x{N}_{out_mode}{'_dyn' if dynamic else ''}{'_u8' if unsigned else ''}{'_b' if wide_rows else ''}",
         meta, {"a.hex": act8_words(a8), "y.hex": y_words, "w.hex": pack_w8(w)[0], "mult.hex": pack_i32vec(mult)[0]},
         {"rowfac.txt": rowfac})


def matmul_real(qm, D):
    def case(name, tensor, a8, rowfac, y=None, w_name=None, mult_name=None, n_cols=None):
        L = qm.linears[tensor]
        w8 = L.w8 if n_cols is None else L.w8[:, :n_cols]
        mult = L.mult if n_cols is None else L.mult[:n_cols]
        K, N = w8.shape
        M = a8.shape[0]
        if y is None:
            acc = matmul_i8(a8, w8)
            y = requant_wide(acc, mult, L.s1) if L.out_bits == 0 else requant(acc, mult, np.zeros(N, np.int32), L.s1, rowfac, L.out_bits)
        mode = {8: "int8", 16: "int16", 32: "int32", 0: "wide"}[L.out_bits]
        yw = raw32_words(y) if L.out_bits == 0 else words(y, L.out_bits)
        meta = dict(kind="real", tensor=tensor, w_tensor=w_name or f"{tensor}.w", mult_tensor=mult_name or f"{tensor}.mult", M=M, K=K, N=N,
                    out_mode=mode, dynamic=bool(L.dynamic), unsigned=False, s1=int(L.s1), a_stride=K // 32,
                    y_stride={"int8": N // 32, "int16": N // 16, "int32": N // 8, "wide": N}[mode])
        emit("matmul", f"real_{name}", meta, {"a.hex": act8_words(a8), "y.hex": yw}, {"rowfac.txt": rowfac if rowfac is not None else np.ones(M, np.int32)})

    a, rf = D["L0.attn.in"][:R], D["L0.attn.rf"][:R]
    case("L0_q", "L0.q", a, rf, D["L0.q16"][:R])
    case("L0_k", "L0.k", a, rf, D["L0.k16"][:R])
    case("L0_v", "L0.v", a, rf, D["L0.v8"][:R])
    case("L0_o", "L0.o", D["L0.o.in"][:R], D["L0.o.rf"][:R], D["L0.o.out"][:R])
    case("L0_gate", "L0.gate", D["L0.mlp.in"][:R], D["L0.mlp.rf"][:R], D["L0.gate.out"][:R])
    case("L0_up", "L0.up", D["L0.mlp.in"][:R], D["L0.mlp.rf"][:R], D["L0.up.out"][:R])
    case("L0_down", "L0.down", D["L0.down.in"][:R], D["L0.down.rf"][:R], D["L0.down.out"][:R])
    case("L20_k", "L20.k", D["L20.attn.in"][:R], D["L20.attn.rf"][:R], D["L20.k16"][:R])
    case("L41_down", "L41.down", D["L41.down.in"][:R], D["L41.down.rf"][:R], D["L41.down.out"][:R])
    # LM head, first column slice, one row (the wide/argmax path)
    case("lm00", "lm", D["lm.in"][:1], None, D["logits"][:1, :LM_SLICE], w_name="lm.00.w", mult_name="lm.00.mult", n_cols=LM_SLICE)


# ---------------------------------------------------------------------------------------------- vector unit
def vector_cases(qm, D, ids):
    def vcase(case, meta, a, a_bits, y, y_bits, b=None, b_bits=None, rowfac=None, toks=None):
        files = {"a.hex": words(a, a_bits), "y.hex": words(y, y_bits)}
        if b is not None:
            files["b.hex"] = words(b, b_bits)
        texts = {}
        if rowfac is not None:
            texts["rowfac_exp.txt"] = rowfac
        if toks is not None:
            texts["toks.txt"] = toks
        meta.update(rows=int(y.shape[0]), cols=int(y.shape[1]), a_bits=a_bits, y_bits=y_bits, b_bits=b_bits or 0)
        emit("vector", case, meta, files, texts)

    # RMSNORM (int32 residual -> int8 + rowfac)
    for name, xkey, nkey, akey, rkey in (("rmsnorm_L0n1", "x0", "L0.norm1", "L0.attn.in", "L0.attn.rf"),
                                         ("rmsnorm_L41n2", "L41.add1.out", "L41.norm2", "L41.mlp.in", "L41.mlp.rf")):
        nq = qm.norms[nkey]
        x = D[xkey][:R]
        y16 = rmsnorm(nq, x); a8, rf = dynq(y16)
        assert np.array_equal(a8, D[akey][:R]) and np.array_equal(rf, D[rkey][:R])
        vcase(name, dict(op="RMSNORM", norm=nkey, eps_q=int(nq.eps_q)), x, 32, a8, 8, rowfac=rf)
    nq = qm.norms["norm_f"]
    xl = D["L41.add2.out"][-1:]
    a8, rf = dynq(rmsnorm(nq, xl))
    assert np.array_equal(a8, D["lm.in"]) and np.array_equal(rf, D["lm.rf"])
    vcase("rmsnorm_f_last", dict(op="RMSNORM", norm="norm_f", eps_q=int(nq.eps_q)), xl, 32, a8, 8, rowfac=rf)
    # DYNQ (attention out int16 -> int8 + rowfac)
    att = D["L0.attn.out"][:R]
    a8, rf = dynq(att)
    assert np.array_equal(a8, D["L0.o.in"][:R])
    vcase("dynq_attn_L0", dict(op="DYNQ"), att, 16, a8, 8, rowfac=rf)
    # SILUMUL (gate16, up16 -> h32 -> int8 + rowfac with b > 0)
    for l in (0, 41):
        sq = qm.silus[f"L{l}.silu"]
        g, u = D[f"L{l}.gate.out"][:R], D[f"L{l}.up.out"][:R]
        h = silu_gate(sq, g, u); a8, rf = dynq(h)
        assert np.array_equal(a8, D[f"L{l}.down.in"][:R]) and np.array_equal(rf, D[f"L{l}.down.rf"][:R])
        vcase(f"silumul_L{l}", dict(op="SILUMUL", m_sig=int(sq.m_sig), s_sig=int(sq.s_sig)), g, 16, a8, 8, b=u, b_bits=16, rowfac=rf)
    # ADD (int32 residual adds)
    ad = qm.adds["L0.add1"]
    y = scaled_add(ad.ma, D["x0"][:R], ad.mb, D["L0.o.out"][:R], 32)
    assert np.array_equal(y, D["L0.add1.out"][:R])
    vcase("add1_L0", dict(op="ADD", ma=int(ad.ma), mb=int(ad.mb)), D["x0"][:R], 32, y.astype(np.int32), 32, b=D["L0.o.out"][:R], b_bits=32)
    ad = qm.adds["L41.add2"]
    y = scaled_add(ad.ma, D["L41.add1.out"][:R], ad.mb, D["L41.down.out"][:R], 32)
    assert np.array_equal(y, D["L41.add2.out"][:R])
    vcase("add2_L41", dict(op="ADD", ma=int(ad.ma), mb=int(ad.mb)), D["L41.add1.out"][:R], 32, y.astype(np.int32), 32, b=D["L41.down.out"][:R], b_bits=32)
    # EMBED (token rows of the embedding table -> int32 residual rows); tokens from the token memory
    e8, rm = qm.tables["embed"], qm.tables["embed.resmult"]
    toks = ids[:R]
    y = embed_rows(e8, rm, toks)
    assert np.array_equal(y, D["x0"][:R])
    vcase("embed_prompt", dict(op="EMBED", slices=sorted({int(t) // 8192 for t in toks})), y, 32, y, 32, toks=toks)
    toks2 = [0, 1, 130072, 130073, 8448, 220, 130559, 8191, 8192, 122880]
    y2 = embed_rows(e8, rm, toks2)
    vcase("embed_special", dict(op="EMBED", slices=sorted({t // 8192 for t in toks2})), y2, 32, y2, 32, toks=toks2)
    # ROPE (int16 q/k rows -> rotated int8), positions from the row index + posBase
    tbl = qm.tables["rope"]
    rq = qm.ropes["L0.rope_q"]
    q16 = D["L0.q16"][:R]
    q8 = rope(rq, q16, np.arange(R), tbl)
    assert np.array_equal(q8, D["L0.q8"][:R])
    vcase("rope_q_L0", dict(op="ROPE", pos_base=0, m_r=int(rq.m_r), s_r=int(rq.s_r), to_kv=False), q16, 16, q8, 8)
    q8b = rope(rq, q16, np.arange(1000, 1000 + R), tbl)
    vcase("rope_q_L0_pos1000", dict(op="ROPE", pos_base=1000, m_r=int(rq.m_r), s_r=int(rq.s_r), to_kv=False), q16, 16, q8b, 8)
    rk = qm.ropes["L20.rope_k"]
    k16 = D["L20.k16"][:R]
    k8 = rope(rk, k16, np.arange(R), tbl)
    assert np.array_equal(k8, D["L20.k8"][:R])
    vcase("rope_k_L20", dict(op="ROPE", pos_base=0, m_r=int(rk.m_r), s_r=int(rk.s_r), to_kv=False), k16, 16, k8, 8)
    rk0 = qm.ropes["L0.rope_k"]
    k16 = D["L0.k16"][:R]
    k8 = rope(rk0, k16, np.arange(R), tbl)
    vcase("rope_k_L0_kv", dict(op="ROPE", pos_base=0, m_r=int(rk0.m_r), s_r=int(rk0.s_r), to_kv=True), k16, 16, k8, 8)


# ---------------------------------------------------------------------------------------------- attention / kv
def kv_words(k8: np.ndarray, v8: np.ndarray, keys_max: int):
    """K^T per kv head as tiled W[d=128][key=keys_max]; V per kv head as tiled W[key][d]. Concatenated by head."""
    n = k8.shape[0]
    kw, vw = [], []
    for h in range(N_KV_HEAD):
        hs = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        kt = np.zeros((HEAD_DIM, keys_max), np.int8); kt[:, :n] = k8[:, hs].T
        v = np.zeros((keys_max, HEAD_DIM), np.int8); v[:n] = v8[:, hs]
        kw.append(pack_w8(kt)[0]); vw.append(pack_w8(v)[0])
    return np.concatenate(kw), np.concatenate(vw)


def attention_cases(qm, D, m: IntMiniCPM):
    KEYS_MAX = 256

    def acase(case, layer, q8, k8, v8, n_keys, causal, q_pos0):
        at = qm.attns[f"L{layer}.attn"]
        out = attention(at, q8, k8, v8, n_keys, causal_offset=q_pos0 if causal else None)
        kw, vw = kv_words(k8, v8, KEYS_MAX)
        meta = dict(layer=layer, n_queries=int(q8.shape[0]), n_keys=n_keys, causal=causal, q_pos0=q_pos0, keys_max=KEYS_MAX,
                    mq=[int(x) for x in at.mq], sq=[int(x) for x in at.sq])
        emit("attention", case, meta, {"q.hex": act8_words(q8), "k.hex": kw, "v.hex": vw, "y.hex": act16_words(out)})
        return out

    q8, k8, v8 = D["L0.q8"], m.kcache[0][:128], m.vcache[0][:128]
    n = q8.shape[0]
    out = acase("L0_prefill128", 0, q8, k8, v8, n, True, 0)
    assert np.array_equal(out, D["L0.attn.out"])
    acase("L0_chunk64", 0, q8[64:], k8, v8, n, True, 64)
    acase("L20_ragged100x70", 20, D["L20.q8"][:100], m.kcache[20][:70], m.vcache[20][:70], 70, False, 0)
    acase("L41_decode_pos40", 41, D["L41.q8"][40:41], m.kcache[41][:41], m.vcache[41][:41], 41, True, 40)
    acase("L41_decode_pos127", 41, D["L41.q8"][127:128], m.kcache[41][:128], m.vcache[41][:128], 128, True, 127)
    # KV write path: V through the engine (L0.v on the real activations), K through the RoPE op (L0.k8 rows)
    kw, vw = kv_words(m.kcache[0][:R], m.vcache[0][:R], KEYS_MAX)
    emit("kv", "L0_rows40", dict(rows=R, keys_max=KEYS_MAX, k_rope="L0.rope_k", m_r=int(qm.ropes["L0.rope_k"].m_r), s_r=int(qm.ropes["L0.rope_k"].s_r)),
         {"k.hex": kw, "v.hex": vw, "k16.hex": act16_words(D["L0.k16"][:R]), "a.hex": act8_words(D["L0.attn.in"][:R])},
         {"rowfac.txt": D["L0.attn.rf"][:R]})


# ---------------------------------------------------------------------------------------------- sampler
def sampler_cases(qm, D):
    logits = D["logits"][0]
    L = qm.linears["lm"]
    emit("sampler", "lm00_real", dict(n_tiles=LM_SLICE // 32, expect=int(np.argmax(logits[:LM_SLICE])), eos=False, s1=int(L.s1)),
         {"a.hex": act8_words(D["lm.in"][:1]), "y.hex": raw32_words(logits[None, :LM_SLICE].astype(np.int32))})
    rng = np.random.default_rng(7)
    for i, (n_tiles, force) in enumerate(((50, None), (17, 1), (300, 130073 - 4080 * 32 + 300 * 32 - 7))):
        t = rng.integers(-(1 << 30), 1 << 30, n_tiles * 32).astype(np.int32)
        j = int(np.argmax(t))
        t[(j + 5) % len(t)] = t[j]       # a tie: the lower index wins
        if force is not None and 0 <= force < len(t):
            t[force] = (1 << 30) + 1
        exp = int(np.argmax(t))
        emit("sampler", f"rand_{i}", dict(n_tiles=n_tiles, expect=exp, eos=exp in (1, 130073), s1=0), {"y.hex": raw32_words(t[None, :])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="matmul,vector,attention,sampler")
    a = ap.parse_args()
    units = a.only.split(",")
    t0 = time.time()
    qm = build(QConfig())
    print(f"quantised model built in {time.time() - t0:.0f} s")
    tok = load_tokenizer()
    ids = text_ids(tok, prompts("calib")[0]["text"], 128)
    m = IntMiniCPM(qm, dump=True, dump_layers=DUMP_LAYERS)
    t0 = time.time()
    m.forward(ids, 0, "last")
    print(f"golden prefill of {len(ids)} tokens in {time.time() - t0:.0f} s")
    D = m.dump.d
    (OUT).mkdir(parents=True, exist_ok=True)
    json.dump(dict(prompt_ids=ids), open(OUT / "prompt.json", "w"))
    if "matmul" in units:
        matmul_random(1, 5, 32, 32, "raw", False)
        matmul_random(2, 70, 96, 64, "int8", True)
        matmul_random(3, 33, 2048, 256, "int8", False)
        matmul_random(4, 40, 1536, 384, "int16", True)
        matmul_random(5, 1, 384, 1536, "int16", True)
        matmul_random(6, 3, 64, 64, "raw", False, unsigned=True)
        matmul_random(7, 64, 64, 96, "wide", False)
        matmul_random(8, 130, 288, 384, "int8", True)
        matmul_random(9, 256, 256, 256, "int8", True)                   # utilisation measurement
        matmul_random(10, 2, 32, 32, "int8", True)
        matmul_random(11, 20, 6144, 64, "int32", True, wide_rows=True)  # 32-bit rows: b > 0
        matmul_random(12, 7, 2048, 32, "int32", True)                   # int32 out, b = 0
        matmul_real(qm, D)
    if "vector" in units:
        vector_cases(qm, D, ids)
    if "attention" in units:
        attention_cases(qm, D, m)
    if "sampler" in units:
        sampler_cases(qm, D)
    print("done")


if __name__ == "__main__":
    main()
