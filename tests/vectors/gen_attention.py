"""Attention test vectors from the golden model (encoder layer 0, all heads; decoder self+cross with a
40-token KV history). Output: out/vectors/attention/<case>/..."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import whisper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gen.dump_weights import pack_w8  # noqa: E402
from golden.data import REPO, clip_by_uid, load_audio  # noqa: E402
from golden.layout import act8_words, act16_words, write_hex  # noqa: E402
from golden.quant import HEAD_DIM, N_HEAD, QConfig, build  # noqa: E402
from golden.whisper_int import IntWhisper, attention, dynq  # noqa: E402

OUT = REPO / "out" / "vectors" / "attention"


def kv_words(k8: np.ndarray, v8: np.ndarray, keys_max: int):
    """K^T per head as tiled W[d=64][key=keys_max]; V per head as tiled W[key][d]. Concatenated by head."""
    n = k8.shape[0]
    kw, vw = [], []
    for h in range(N_HEAD):
        hs = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        kt = np.zeros((HEAD_DIM, keys_max), np.int8); kt[:, :n] = k8[:, hs].T
        v = np.zeros((keys_max, HEAD_DIM), np.int8); v[:n] = v8[:, hs]
        kw.append(pack_w8(kt)[0]); vw.append(pack_w8(v)[0])
    return np.concatenate(kw), np.concatenate(vw)


def emit(case, meta, q8, k8, v8, out16, keys_max):
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    kw, vw = kv_words(k8, v8, keys_max)
    write_hex(d / "q.hex", act8_words(q8))
    write_hex(d / "k.hex", kw)
    write_hex(d / "v.hex", vw)
    write_hex(d / "y.hex", act16_words(out16))
    meta.update(n_queries=int(q8.shape[0]), keys_max=keys_max, kv_words_per_head=int(len(kw) // N_HEAD // 64))
    json.dump(meta, open(d / "meta.json", "w"))
    print(f"{case:24s} queries={meta['n_queries']} keys={meta['n_keys']} causal={meta['causal']}")


def main():
    qm = build(QConfig())
    m = IntWhisper(qm, dump=True)
    c = clip_by_uid("varied/en_2s_f")
    audio = load_audio(c.path)
    mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)), n_mels=80).numpy()
    enc8, rf = m.encoder(m.mel_quant(mel, 256))
    D = m.dump.d
    at = qm.attns["enc.0.attn"]
    q8, k8, v8 = D["enc.0.attn.q"], D["enc.0.attn.k"], D["enc.0.attn.v"]
    n = q8.shape[0]   # 128
    out = attention(at, q8, k8, v8, n)
    assert np.array_equal(out, D["enc.0.attn.out"])
    common = dict(mq=[int(x) for x in at.mq], sq=[int(x) for x in at.sq])
    emit("enc0_full", dict(n_keys=n, causal=False, q_pos0=0, **common), q8, k8, v8, out, 1536)
    # a ragged case: 100 queries x 70 keys (partial tiles)
    out2 = attention(at, q8[:100], k8[:70], v8[:70], 70)
    emit("enc0_ragged", dict(n_keys=70, causal=False, q_pos0=0, **common), q8[:100], k8[:70], v8[:70], out2, 1536)
    # decoder: run the real decoder on the clip to get a 40-token history, then self-attn for token 40 and cross-attn
    ck, cv = m.cross_kv(enc8, rf)
    lm = qm.meta["decoding"]["en"]
    state = dict(n_ctx=enc8.shape[0], ck=ck, cv=cv, k=[np.zeros((448, 384), np.int8) for _ in range(4)],
                 v=[np.zeros((448, 384), np.int8) for _ in range(4)])
    prompt = list(lm["initial_tokens"])
    tok = prompt[0]
    toks = []
    m.dump.enabled = False
    for pos in range(41):
        sample = pos >= 3
        nxt, _ = m.decoder_step(tok, pos, state, sample, first_sample=(pos == 3), lang_meta=lm)
        if pos < 3:
            tok = prompt[pos + 1]
        else:
            toks.append(nxt); tok = nxt if nxt != lm["eot"] else 50257
    m.dump.enabled = True
    # position 41 (query), keys 0..41: self-attention layer 0
    m.dump.d.clear()
    x = m.embed(tok, 41)
    pt = "dec.pos41.0"
    from golden.whisper_int import layernorm, linear
    y16 = layernorm(qm.lns["dec.0.ln1"], x); a8, rfq = dynq(y16)
    q = linear(qm.linears["dec.0.attn.q"], a8, rfq); k = linear(qm.linears["dec.0.attn.k"], a8, rfq); v = linear(qm.linears["dec.0.attn.v"], a8, rfq)
    kc = state["k"][0].copy(); vc = state["v"][0].copy(); kc[41] = k[0]; vc[41] = v[0]
    atd = qm.attns["dec.0.attn"]
    outd = attention(atd, q, kc[:42], vc[:42], 42, causal_offset=41)
    emit("dec0_self_pos41", dict(n_keys=42, causal=True, q_pos0=41, mq=[int(x) for x in atd.mq], sq=[int(x) for x in atd.sq]),
         q, kc[:42], vc[:42], outd, 448)
    # cross attention layer 0 for the same token
    y16 = layernorm(qm.lns["dec.0.lnc"], x); a8, rfq = dynq(y16)
    qc = linear(qm.linears["dec.0.xattn.q"], a8, rfq)
    atx = qm.attns["dec.0.xattn"]
    outx = attention(atx, qc, ck[0], cv[0], enc8.shape[0])
    emit("dec0_cross_pos41", dict(n_keys=int(enc8.shape[0]), causal=False, q_pos0=0, mq=[int(x) for x in atx.mq], sq=[int(x) for x in atx.sq]),
         qc, ck[0], cv[0], outx, 1536)


if __name__ == "__main__":
    main()
