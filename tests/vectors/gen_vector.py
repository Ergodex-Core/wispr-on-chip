"""VectorUnit test vectors from the golden model (real activations, short clip, n_frames=256).
Output: out/vectors/vector/<case>/{meta.json, a.hex, b.hex?, y.hex, rowfac_exp.txt?}"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import whisper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.data import REPO, clip_by_uid, load_audio  # noqa: E402
from golden.layout import act8_words, act16_words, write_hex  # noqa: E402
from golden.quant import QConfig, build  # noqa: E402
from golden.whisper_int import IntWhisper, dynq, gelu16, layernorm, requant_static8, scaled_add  # noqa: E402

OUT = REPO / "out" / "vectors" / "vector"


def words(x, bits16: bool):
    return act16_words(x) if bits16 else act8_words(x)


def emit(case, meta, a, a16, y, y16, b=None, b16=None, rowfac=None):
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    write_hex(d / "a.hex", words(a, a16))
    write_hex(d / "y.hex", words(y, y16))
    if b is not None:
        write_hex(d / "b.hex", words(b, b16))
    if rowfac is not None:
        (d / "rowfac_exp.txt").write_text("\n".join(str(int(x)) for x in rowfac) + "\n")
    meta.update(rows=int(a.shape[0]), cols=int(y.shape[1]), a_bits16=a16, y_bits16=y16, b_bits16=bool(b16))
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    print(f"{case:24s} rows={meta['rows']} cols={meta['cols']}")


def main():
    qm = build(QConfig())
    m = IntWhisper(qm, dump=True)
    c = clip_by_uid("varied/en_2s_f")
    audio = load_audio(c.path)
    mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)), n_mels=80).numpy()
    m.encoder(m.mel_quant(mel, 256))
    D = m.dump.d
    R = 40   # rows per case (keeps sims short)

    # 1. LN + dynq (enc.0.ln1 on the residual x0)
    ln = qm.lns["enc.0.ln1"]
    x = D["enc.x0"][:R]
    y16 = layernorm(ln, x); a8, rf = dynq(y16)
    assert np.array_equal(a8, D["enc.0.attn.in"][:R])
    emit("ln_enc0", dict(op="LN", ln="enc.0.ln1", eps_q=int(ln.eps_q)), x, True, a8, False, rowfac=rf)
    # 2. DYNQ + GELU (fc1 out -> fc2 in)
    g = qm.gelus["enc.0.gelu"]
    h = D["enc.0.fc1.out"][:R]
    g16 = gelu16(g, h); a8, rf = dynq(g16, g.smooth)
    assert np.array_equal(a8, D["enc.0.fc2.in"][:R])
    emit("gelu_dynq_enc0", dict(op="DYNQ", gelu=True, m_phi=int(g.m_phi), s_phi=int(g.s_phi), smooth=g.smooth is not None,
                                smooth_tensor="enc.0.gelu.smooth" if g.smooth is not None else ""), h, True, a8, False, rowfac=rf)
    # 3. plain DYNQ (attention out)
    att = D["enc.0.attn.out"][:R]
    a8, rf = dynq(att)
    emit("dynq_attn_enc0", dict(op="DYNQ", gelu=False, m_phi=0, s_phi=0, smooth=False, smooth_tensor=""), att, True, a8, False, rowfac=rf)
    # 4. static8: conv1 GELU
    g1 = qm.gelus["enc.conv1"]
    c1 = D["enc.conv1.out"][:R]
    g8 = requant_static8(g1, gelu16(g1, c1))
    assert np.array_equal(g8, D["enc.conv1.gelu"][:R])
    emit("gelu_static8_conv1", dict(op="DYNQ", gelu=True, m_phi=int(g1.m_phi), s_phi=int(g1.s_phi), smooth=False, smooth_tensor="",
                                    static8=True, req_mult=int(g1.req_mult), req_shift=int(g1.req_shift)), c1, True, g8, False)
    # 5. ADD residual: x0 + o16 (both int16 banks)
    ad = qm.adds["enc.0.add1"]
    x0 = D["enc.x0"][:R]; o = D["enc.0.attn.o.out"][:R]
    y = scaled_add(ad.ma, x0, ad.mb, o, 16)
    assert np.array_equal(y, D["enc.0.add1.out"][:R])
    emit("add_resid_enc0", dict(op="ADD", ma=int(ad.ma), mb=int(ad.mb), b_src="bank"), x0, True, y.astype(np.int16), True, b=o, b16=True)
    # 6. ADD with ROM pos-emb: g2 (int16) + enc.pos rows (int8 ROM) -> x0
    ad = qm.adds["enc.x0"]
    g2 = D["enc.conv2.gelu"][:R]
    pos = qm.tables["enc.pos"][:R]
    y = scaled_add(ad.ma, g2, ad.mb, pos, 16)
    assert np.array_equal(y, D["enc.x0"][:R])
    emit("add_pos_enc", dict(op="ADD", ma=int(ad.ma), mb=int(ad.mb), b_src="rom", b_tensor="enc.pos", b_row0=0), g2, True, y.astype(np.int16), True)
    # 7. EMBED: token rows of the LM matrix, then ADD with per-token mult + dec.pos
    toks = [50258, 50259, 50359, 50363, 440, 1, 51864]
    L = qm.linears["dec.lm"]
    emb = np.stack([L.w8[:, t] for t in toks]).astype(np.int8)
    emit("embed", dict(op="EMBED", toks=toks), emb, False, emb, False)
    ad = qm.adds["dec.x0"]
    rows = []
    for i, t in enumerate(toks):
        rows.append(m.embed(t, i)[0])
    y = np.stack(rows).astype(np.int16)
    emit("add_embed_dec", dict(op="ADD", ma=0, mb=int(ad.mb), b_src="rom", b_tensor="dec.pos", b_row0=0, ma_from_table=True,
                               ma_table="dec.emb.resmult", toks=toks), emb, False, y, True)


if __name__ == "__main__":
    main()
