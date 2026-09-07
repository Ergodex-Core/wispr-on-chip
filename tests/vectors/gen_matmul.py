"""Generate MatmulEngine test vectors (random tiles with random requant params, and real tensors driven
by real activations from the golden model). Output: out/vectors/matmul/<case>/...

Files per case: meta.json, a.hex (activation words), y.hex (expected output words), rowfac.txt,
and for random cases w.hex/mult.hex/bias.hex (loaded into the Sram weight store by the test).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gen.dump_weights import pack_i32vec, pack_w8  # noqa: E402
from golden.data import REPO  # noqa: E402
from golden.fixedpoint import matmul_i8, requant, requant_wide  # noqa: E402
from golden.layout import act8_words, act16_words, raw32_words, write_hex  # noqa: E402

OUT = REPO / "out" / "vectors" / "matmul"


def emit(case: str, meta: dict, a_words, y_words, rowfac=None, w_words=None, mult=None, bias=None):
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    write_hex(d / "a.hex", a_words)
    write_hex(d / "y.hex", y_words)
    if rowfac is not None:
        (d / "rowfac.txt").write_text("\n".join(str(int(x)) for x in rowfac) + "\n")
    if w_words is not None:
        write_hex(d / "w.hex", w_words)
        write_hex(d / "mult.hex", pack_i32vec(mult)[0])
        write_hex(d / "bias.hex", pack_i32vec(bias)[0])
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    print(f"{case:40s} M={meta['M']:5d} K={meta['K']:5d} N={meta['N']:6d} mode={meta['out_mode']}")


def random_case(seed: int, M: int, K: int, N: int, out_mode: str, dynamic: bool, unsigned: bool = False):
    rng = np.random.default_rng(seed)
    lo, hi = (0, 256) if unsigned else (-128, 128)
    a = rng.integers(lo, hi, size=(M, K)).astype(np.int64)
    w = rng.integers(-128, 128, size=(K, N)).astype(np.int8)
    acc = matmul_i8(a, w)
    bits = {"int8": 8, "int16": 16, "wide": 0, "raw": 0}[out_mode]
    if out_mode == "raw":
        mult = np.zeros(N, np.int32); bias = np.zeros(N, np.int32); s1 = 0
        y = acc.astype(np.int32)
        y_words = raw32_words(y)
        rowfac = np.ones(M, np.int64)
    elif out_mode == "wide":
        mult = rng.integers(1 << 28, 1 << 31, N).astype(np.int32)
        bias = np.zeros(N, np.int32)
        s1 = 22
        y = requant_wide(acc, mult, s1)
        y_words = raw32_words(y)
        rowfac = np.ones(M, np.int64)
    else:
        mult = rng.integers(1 << 26, 1 << 31, N).astype(np.int32)
        bias = rng.integers(-(1 << 22), 1 << 22, N).astype(np.int32)
        rowfac = rng.integers(1, 1 << 16, M).astype(np.int64) if dynamic else np.ones(M, np.int64)
        s2 = {8: 24, 16: 20}[bits]
        mag = np.abs(acc).max() * float(mult.max()) * float(rowfac.max())
        s1 = max(0, min(63, int(np.ceil(np.log2(max(mag, 1)))) - s2 - (bits - 2)))
        y = requant(acc, mult, bias, s1, rowfac, bits)
        y_words = act8_words(y) if bits == 8 else act16_words(y)
    a8 = a.astype(np.int8) if not unsigned else a.astype(np.uint8).view(np.int8)
    meta = dict(kind="random", seed=seed, M=M, K=K, N=N, out_mode=out_mode, dynamic=dynamic, unsigned=unsigned,
                s1=int(s1), a_stride=K // 32, y_stride={"int8": N // 32, "int16": N // 16, "wide": N, "raw": N}[out_mode],
                has_bias=out_mode in ("int8", "int16"), im2col=False, tiles_per_frame=0, stride2=False, frames=0)
    emit(f"rand_{seed}_{M}x{K}x{N}_{out_mode}{'_dyn' if dynamic else ''}{'_u8' if unsigned else ''}", meta,
         act8_words(a8), y_words, rowfac, pack_w8(w)[0], mult, bias)


def real_cases():
    """Real tensors x real activations from the golden model on a short clip (n_frames=256)."""
    import torch
    import whisper
    from golden.data import clip_by_uid, load_audio
    from golden.quant import QConfig, build
    from golden.whisper_int import IntWhisper, linear

    qm = build(QConfig())
    m = IntWhisper(qm, dump=True)
    c = clip_by_uid("varied/en_2s_f")
    audio = load_audio(c.path)
    mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)), n_mels=80).numpy()
    mel8 = m.mel_quant(mel, 256)
    m.encoder(mel8)
    D = m.dump.d

    def case(name, tensor, a8, rowfac, out_mode, im2col=None):
        L = qm.linears[tensor]
        K, N = L.w8.shape
        if im2col:
            frames, tiles_per_frame, stride = im2col
            a_in = m.im2col(a8, stride)
            a_words = act8_words(a8)        # the raw frame buffer: frames x (tiles_per_frame*32) bytes
            a_stride = tiles_per_frame
            M = frames // stride
        else:
            a_in = a8
            a_words = act8_words(a8)
            a_stride = K // 32
            M = a8.shape[0]
        y = linear(L, a_in, rowfac if L.dynamic else None)
        if out_mode == "int8":
            yw, ys = act8_words(y), N // 32
        elif out_mode == "int16":
            yw, ys = act16_words(y), N // 16
        else:
            yw, ys = raw32_words(y), N
        meta = dict(kind="real", tensor=tensor, M=M, K=K, N=N, out_mode=out_mode, dynamic=bool(L.dynamic),
                    unsigned=False, s1=int(L.s1), a_stride=a_stride, y_stride=ys, has_bias=L.out_bits != 0,
                    im2col=bool(im2col), tiles_per_frame=im2col[1] if im2col else 0, stride2=bool(im2col and im2col[2] == 2),
                    frames=im2col[0] if im2col else 0)
        emit(f"real_{name}", meta, a_words, yw, rowfac if rowfac is not None else np.ones(M, np.int64))

    case("conv1", "enc.conv1", D["enc.mel8"], None, "int16", im2col=(256, 3, 1))
    case("conv2", "enc.conv2", D["enc.conv1.gelu"], None, "int16", im2col=(256, 12, 2))
    case("enc0_q", "enc.0.attn.q", D["enc.0.attn.in"], D["enc.0.attn.rf"], "int8")
    case("enc0_k", "enc.0.attn.k", D["enc.0.attn.in"], D["enc.0.attn.rf"], "int8")
    case("enc0_o", "enc.0.attn.o", D["enc.0.attn.o.in"], D["enc.0.attn.o.rf"], "int16")
    case("enc0_fc1", "enc.0.fc1", D["enc.0.fc1.in"], D["enc.0.fc1.rf"], "int16")
    case("enc0_fc2", "enc.0.fc2", D["enc.0.fc2.in"], D["enc.0.fc2.rf"], "int16")
    case("dec0_xk", "dec.0.xattn.k", D["enc.out"], D["enc.out.rf"], "int8")
    case("lm", "dec.lm", D["enc.out"][:1], None, "wide")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random_case(1, 5, 32, 32, "raw", False)
    random_case(2, 70, 96, 64, "int8", True)
    random_case(3, 33, 384, 384, "int8", False)
    random_case(4, 40, 1536, 384, "int16", True)
    random_case(5, 1, 384, 1536, "int16", True)
    random_case(6, 3, 64, 64, "raw", False, unsigned=True)
    random_case(7, 64, 64, 96, "wide", False)
    random_case(8, 130, 288, 384, "int8", True)
    random_case(9, 1500, 384, 384, "int8", True)      # utilisation measurement
    random_case(10, 2, 32, 32, "int8", True)
    real_cases()


if __name__ == "__main__":
    main()
