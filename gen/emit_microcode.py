"""Emit the sequencer program (generated/Microcode.scala) from weights/MANIFEST.json params + the chip map.
The program mirrors golden/whisper_int.py op for op."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gen.chipmap import (BANKS, CONV1_CHUNK, DEC, DEC_KEYS, ENC_KEYS, FFN_CHUNK, KV, PH_DEC, PH_ENC, REPO,  # noqa: E402
                         tensor_bases)

OUT = REPO / "src/main/scala/whisper/generated/Microcode.scala"
MM = dict(MATMUL=0, VEC=1, ATTN=2, KVSET=3, CHUNK_BEGIN=4, CHUNK_NEXT=5, JUMP=6, BR_PROMPT=7, SAMPLE=8, NEXTPOS=9, HALT=10, SETPOS=11, TOK_PROMPT=12)
ROWS = dict(Const=0, NFrames=1, NCtx=2, Chunk=3)
BASE = dict(Zero=0, Chunk=1, Pos=2)
KEYS = dict(Const=0, NCtx=1, PosPlus1=2)
VOP = dict(LN=0, DYNQ=1, ADD=2, EMBED=3)
MODE = dict(int8=0, int16=1, raw=2, wide=3)
SINK = dict(bank=0, attn=1, sampler=2, kv=3)

man = json.load(open(REPO / "weights" / "MANIFEST.json"))
P = man["params"]
B = tensor_bases()
prog: list[dict] = []
labels: dict[str, int] = {}


def op(opc, **kw):
    d = dict(opc=MM[opc], mm={}, vec={}, att={}, kv={}, rowsSel=0, framesSel=0, rowOffSel=0, keyOffSel=0, nKeysSel=0, nQSel=0,
             qPos0Sel=0, bBasePos=False, tokSel=False, flushK=False, imm=0, imm2=0, name="")
    d.update(kw)
    prog.append(d)
    return d


def label(name):
    labels[name] = len(prog)


def matmul(tensor, act, out, rows_sel="Const", rows=0, rowoff="Zero", im2col=None, frames_sel=0, out_local=False, dyn_bank=None,
           name=""):
    """act=(bank, base, stride[, koff]); out=(mode, sink, bank, base, stride[, tag])"""
    L = P["linears"][tensor]
    K, N = L["K"], L["N"]
    KT, NT = K // 32, N // 32
    has_bias = f"{tensor}.bias" in B
    mm = dict(rows=rows, kTiles=KT, nTiles=NT, actSrc=0, actBank=act[0], actBase=act[1], actStride=act[2], actKOff=0,
              wSrc=0, wBase=B[f"{tensor}.w"], wStrideN=KT * 4, wStrideK=4, wStrideG=1,
              outMode=MODE[out[0]], outSink=SINK[out[1]], outBank=out[2], outBase=out[3], outStride=out[4], outTag=out[5] if len(out) > 5 else 0,
              rowBase=0, rowOff=0, s1=L["s1"], multBase=B[f"{tensor}.mult"], biasBase=B.get(f"{tensor}.bias", 0), hasBias=int(has_bias),
              dynamic=int(L["dynamic"]), rowfacBank=dyn_bank if dyn_bank is not None else 0, outLocal=int(out_local))
    if im2col:
        tiles_per_frame, stride2 = im2col
        mm.update(im2col=1, tilesPerFrame=tiles_per_frame, stride2=int(stride2))
    return op("MATMUL", mm=mm, rowsSel=ROWS[rows_sel], rowOffSel=BASE[rowoff], framesSel=frames_sel, name=name or tensor)


def vec(vop, rows_sel="Const", rows=0, rowoff="Zero", name="", **fields):
    v = dict(op=VOP[vop], rows=rows, rowOff=0)
    v.update(fields)
    return op("VEC", vec=v, rowsSel=ROWS[rows_sel], rowOffSel=BASE[rowoff], name=name)


def ln(lnname, inb, outb, rows_sel, rowoff="Zero", cols=384):
    L = P["lns"][lnname]
    eps = L["eps_q"]
    return vec("LN", rows_sel, 0, rowoff, name=lnname, cols=cols, inBank=inb[0], inBase=inb[1], inStride=24, inBits16=1,
               outBank=outb[0], outBase=outb[1], outStride=12, rowfacBank=outb[0], rowBase=outb[1] // 12,
               gBase=B[f"{lnname}.g"], bParamBase=B[f"{lnname}.b"], epsLo=eps & 0xFFFFFFFF, epsHi=eps >> 32)


def dynq(inb, outb, rows_sel, cols, rowoff="Zero", gelu=None, out_local=False, name=""):
    """inb=(bank, base, stride), outb=(bank, base, stride); gelu = gelu param name or None"""
    f = dict(cols=cols, inBank=inb[0], inBase=inb[1], inStride=inb[2], inBits16=1, outBank=outb[0], outBase=outb[1], outStride=outb[2],
             rowfacBank=outb[0], rowBase=outb[1] // outb[2], outLocal=int(out_local))
    if gelu:
        g = P["gelus"][gelu]
        f.update(gelu=1, mPhi=g["m_phi"], sPhi=g["s_phi"])
        if g["has_smooth"]:
            f.update(smooth=1, smoothBase=B[f"{gelu}.smooth"])
        if g["req_mult"]:
            f.update(static8=1, reqMult=g["req_mult"], reqShift=g["req_shift"])
    return vec("DYNQ", rows_sel, 0, rowoff, name=name or f"dynq{'+' + gelu if gelu else ''}", **f)


def add(addname, a, b, outb, rows_sel, rowoff="Zero", b_rom=None, b_local=False, gelu_a=None, ma_table=None, b_pos=False, name=""):
    A = P["adds"][addname]
    f = dict(cols=384, inBank=a[0], inBase=a[1], inStride=a[2], inBits16=1, outBank=outb[0], outBase=outb[1], outStride=24,
             ma=A["ma"], mb=A["mb"], bLocal=int(b_local))
    if b_rom:
        f.update(bSrc=1, bBase=B[b_rom], bStride=12, bBits16=0)
    else:
        f.update(bSrc=0, bBank=b[0], bBase=b[1], bStride=b[2], bBits16=int(b[3]))
    if gelu_a:
        g = P["gelus"][gelu_a]
        f.update(geluA=1, mPhi=g["m_phi"], sPhi=g["s_phi"])
    if ma_table:
        f.update(maFromTable=1, maTableBase=B[ma_table])
    return op("VEC", vec=dict(op=VOP["ADD"], rows=0, rowOff=0, **f), rowsSel=ROWS[rows_sel], rowOffSel=BASE[rowoff], bBasePos=b_pos,
              tokSel=bool(ma_table), name=name or addname)


def kvset(is_k, base, keys_max, keyoff="Zero"):
    return op("KVSET", kv=dict(isK=int(is_k), base=base, keysMax=keys_max, keyOff=0, flush=0), keyOffSel=BASE[keyoff], name=f"kvset {'K' if is_k else 'V'}")


def attn(attname, q, kbase, vbase, keys_max, outb, nq_sel, nkeys_sel, causal=False, qpos_sel="Zero", name=""):
    A = P["attns"][attname]
    a = dict(nQueries=1, nKeys=0, causal=int(causal), qPos0=0, qBank=q[0], qBase=q[1], qStride=12, kBase=kbase, vBase=vbase, keysMax=keys_max,
             outBank=outb[0], outBase=outb[1], outStride=24)
    for h in range(6):
        a[f"mq.{h}"] = A["mq"][h]
        a[f"sq.{h}"] = A["sq"][h]
    return op("ATTN", att=a, nQSel=nq_sel, nKeysSel=KEYS[nkeys_sel], qPos0Sel=BASE[qpos_sel], name=name or attname)


def encoder():
    MEL, A8, X, Q8, T16, H16, A8X, E8 = range(8)
    # conv1 in chunks of CONV1_CHUNK rows: T16 <- conv1(mel) (int16, local rows) ; A8 <- gelu/static8 (global rows)
    op("CHUNK_BEGIN", imm=ROWS["NFrames"], imm2=CONV1_CHUNK, name="conv1 chunks")
    label("conv1_loop")
    matmul("enc.conv1", (MEL, 0, 3), ("int16", "bank", T16, 0, 24), "Chunk", rowoff="Chunk", im2col=(3, False), frames_sel=1, out_local=True)
    dynq((T16, 0, 24), (A8, 0, 12), "Chunk", 384, rowoff="Chunk", gelu="enc.conv1", name="gelu8 conv1")  # in local, out global
    prog[-1]["vec"]["inLocal"] = 1
    op("CHUNK_NEXT", imm=labels["conv1_loop"], name="conv1 next")
    # conv2 (stride 2) over all nCtx rows: T16 <- conv2(A8 frames) int16 ; X <- gelu(T16) + pos
    matmul("enc.conv2", (A8, 0, 12), ("int16", "bank", T16, 0, 24), "NCtx", im2col=(12, True), frames_sel=1)
    add("enc.x0", (T16, 0, 24), None, (X, 0), "NCtx", b_rom="enc.pos", gelu_a="enc.conv2", name="x0 = gelu(conv2)+pos")
    for l in range(4):
        p = f"enc.{l}"
        ln(f"{p}.ln1", (X, 0), (A8, 0), "NCtx")
        matmul(f"{p}.attn.q", (A8, 0, 12), ("int8", "bank", Q8, 0, 12), "NCtx", dyn_bank=A8)
        kvset(True, KV["encSelfK"], ENC_KEYS)
        matmul(f"{p}.attn.k", (A8, 0, 12), ("int8", "kv", 0, 0, 12), "NCtx", dyn_bank=A8, name=f"{p}.attn.k -> KV")
        prog[-1]["flushK"] = True
        kvset(False, KV["encSelfV"], ENC_KEYS)
        matmul(f"{p}.attn.v", (A8, 0, 12), ("int8", "kv", 0, 0, 12), "NCtx", dyn_bank=A8, name=f"{p}.attn.v -> KV")
        attn(f"{p}.attn", (Q8, 0), KV["encSelfK"], KV["encSelfV"], ENC_KEYS, (T16, 0), 2, "NCtx")
        dynq((T16, 0, 24), (A8, 0, 12), "NCtx", 384, name="dynq attn out")
        matmul(f"{p}.attn.o", (A8, 0, 12), ("int16", "bank", T16, 0, 24), "NCtx", dyn_bank=A8)
        add(f"{p}.add1", (X, 0, 24), (T16, 0, 24, True), (X, 0), "NCtx")
        ln(f"{p}.ln2", (X, 0), (A8, 0), "NCtx")
        op("CHUNK_BEGIN", imm=ROWS["NCtx"], imm2=FFN_CHUNK, name=f"{p} ffn chunks")
        label(f"ffn{l}")
        matmul(f"{p}.fc1", (A8, 0, 12), ("int16", "bank", H16, 0, 96), "Chunk", rowoff="Chunk", dyn_bank=A8, out_local=True)
        dynq((H16, 0, 96), (A8X, 0, 48), "Chunk", 1536, gelu=f"{p}.gelu", out_local=True, name="gelu+dynq")
        matmul(f"{p}.fc2", (A8X, 0, 48), ("int16", "bank", T16, 0, 24), "Chunk", dyn_bank=A8X, out_local=True)
        add(f"{p}.add2", (X, 0, 24), (T16, 0, 24, True), (X, 0), "Chunk", rowoff="Chunk", b_local=True)
        op("CHUNK_NEXT", imm=labels[f"ffn{l}"], name=f"{p} ffn next")
    ln("enc.ln_post", (X, 0), (E8, 0), "NCtx")
    for l in range(4):
        p = f"dec.{l}"
        kvset(True, KV["crossK"] + l * 6 * PH_ENC, ENC_KEYS)
        matmul(f"{p}.xattn.k", (E8, 0, 12), ("int8", "kv", 0, 0, 12), "NCtx", dyn_bank=E8, name=f"{p}.xattn.k -> KV")
        prog[-1]["flushK"] = True
        kvset(False, KV["crossV"] + l * 6 * PH_ENC, ENC_KEYS)
        matmul(f"{p}.xattn.v", (E8, 0, 12), ("int8", "kv", 0, 0, 12), "NCtx", dyn_bank=E8, name=f"{p}.xattn.v -> KV")


def decoder():
    MEL, A8, X, Q8, T16, H16, A8X, E8 = range(8)
    EM8, AD8, XD, QD8, TD16, OD16, HD16, AD8X = (DEC[k] for k in ["EM8", "AD8", "XD", "QD8", "TD16", "OD16", "HD16", "AD8X"])
    op("SETPOS", imm=0, name="pos = 0, tok = prompt[0]")
    label("dec_loop")
    op("VEC", vec=dict(op=VOP["EMBED"], rows=1, cols=384, outBank=EM8[0], outBase=EM8[1], outStride=12, lmBase=B["dec.lm.w"], static8=1, reqMult=1, reqShift=0),
       tokSel=True, name="embed tok")
    add("dec.x0", (EM8[0], EM8[1], 12), None, XD, "Const", b_rom="dec.pos", ma_table="dec.emb.resmult", b_pos=True, name="xd = emb*me + pos")
    prog[-1]["vec"]["rows"] = 1; prog[-1]["vec"]["inBits16"] = 0
    for l in range(4):
        p = f"dec.{l}"
        ln(f"{p}.ln1", XD, AD8, "Const"); prog[-1]["vec"]["rows"] = 1
        matmul(f"{p}.attn.q", (AD8[0], AD8[1], 12), ("int8", "bank", QD8[0], QD8[1], 12), rows=1, dyn_bank=AD8[0])
        prog[-1]["mm"]["rowBase"] = AD8[1] // 12
        kvset(True, KV["decSelfK"] + l * 6 * PH_DEC, DEC_KEYS, keyoff="Pos")
        matmul(f"{p}.attn.k", (AD8[0], AD8[1], 12), ("int8", "kv", 0, 0, 12), rows=1, dyn_bank=AD8[0], name=f"{p}.attn.k -> KV")
        prog[-1]["mm"]["rowBase"] = AD8[1] // 12; prog[-1]["flushK"] = True
        kvset(False, KV["decSelfV"] + l * 6 * PH_DEC, DEC_KEYS, keyoff="Pos")
        matmul(f"{p}.attn.v", (AD8[0], AD8[1], 12), ("int8", "kv", 0, 0, 12), rows=1, dyn_bank=AD8[0], name=f"{p}.attn.v -> KV")
        prog[-1]["mm"]["rowBase"] = AD8[1] // 12
        attn(f"{p}.attn", (QD8[0], QD8[1]), KV["decSelfK"] + l * 6 * PH_DEC, KV["decSelfV"] + l * 6 * PH_DEC, DEC_KEYS, TD16, 0, "PosPlus1", causal=True, qpos_sel="Pos")
        dynq((TD16[0], TD16[1], 24), (AD8[0], AD8[1], 12), "Const", 384, name="dynq attn out"); prog[-1]["vec"]["rows"] = 1
        matmul(f"{p}.attn.o", (AD8[0], AD8[1], 12), ("int16", "bank", OD16[0], OD16[1], 24), rows=1, dyn_bank=AD8[0]); prog[-1]["mm"]["rowBase"] = AD8[1] // 12
        add(f"{p}.add1", (XD[0], XD[1], 24), (OD16[0], OD16[1], 24, True), XD, "Const"); prog[-1]["vec"]["rows"] = 1
        ln(f"{p}.lnc", XD, AD8, "Const"); prog[-1]["vec"]["rows"] = 1
        matmul(f"{p}.xattn.q", (AD8[0], AD8[1], 12), ("int8", "bank", QD8[0], QD8[1], 12), rows=1, dyn_bank=AD8[0]); prog[-1]["mm"]["rowBase"] = AD8[1] // 12
        attn(f"{p}.xattn", (QD8[0], QD8[1]), KV["crossK"] + l * 6 * PH_ENC, KV["crossV"] + l * 6 * PH_ENC, ENC_KEYS, TD16, 0, "NCtx")
        dynq((TD16[0], TD16[1], 24), (AD8[0], AD8[1], 12), "Const", 384, name="dynq xattn out"); prog[-1]["vec"]["rows"] = 1
        matmul(f"{p}.xattn.o", (AD8[0], AD8[1], 12), ("int16", "bank", OD16[0], OD16[1], 24), rows=1, dyn_bank=AD8[0]); prog[-1]["mm"]["rowBase"] = AD8[1] // 12
        add(f"{p}.add2", (XD[0], XD[1], 24), (OD16[0], OD16[1], 24, True), XD, "Const"); prog[-1]["vec"]["rows"] = 1
        ln(f"{p}.ln2", XD, AD8, "Const"); prog[-1]["vec"]["rows"] = 1
        matmul(f"{p}.fc1", (AD8[0], AD8[1], 12), ("int16", "bank", HD16[0], HD16[1], 96), rows=1, dyn_bank=AD8[0]); prog[-1]["mm"]["rowBase"] = AD8[1] // 12
        dynq((HD16[0], HD16[1], 96), (AD8X[0], AD8X[1], 48), "Const", 1536, gelu=f"{p}.gelu", name="gelu+dynq"); prog[-1]["vec"]["rows"] = 1
        matmul(f"{p}.fc2", (AD8X[0], AD8X[1], 48), ("int16", "bank", OD16[0], OD16[1], 24), rows=1, dyn_bank=AD8X[0]); prog[-1]["mm"]["rowBase"] = AD8X[1] // 48
        add(f"{p}.add3", (XD[0], XD[1], 24), (OD16[0], OD16[1], 24, True), XD, "Const"); prog[-1]["vec"]["rows"] = 1
    op("BR_PROMPT", imm=0, name="skip LM head while feeding the prompt")   # target patched below
    br = len(prog) - 1
    ln("dec.ln", XD, AD8, "Const"); prog[-1]["vec"]["rows"] = 1
    matmul("dec.lm", (AD8[0], AD8[1], 12), ("wide", "sampler", 0, 0, 1621), rows=1, dyn_bank=AD8[0]); prog[-1]["mm"]["rowBase"] = AD8[1] // 12
    op("SAMPLE", name="tok = argmax (emit; halt on eot / limit)")
    op("NEXTPOS", imm=labels["dec_loop"], name="pos++ ; loop")
    prog[br]["imm"] = len(prog)
    op("TOK_PROMPT", name="tok = prompt[pos+1]")
    op("NEXTPOS", imm=labels["dec_loop"], name="pos++ ; loop")
    op("HALT", name="halt")


def emit():
    encoder()
    decoder()
    dec = P["decoding"]["en"]
    lines = ["// GENERATED by gen/emit_microcode.py — do not edit.", "package whisper.generated", "", "import whisper.UOp", "",
             "object Microcode {", f"  val promptLen = {len(dec['initial_tokens'])}", f"  val sampleLen = {dec['sample_len']}",
             f"  val eot = {dec['eot']}", f"  val sotSequence: Seq[Int] = Seq({', '.join(str(t) for t in dec['initial_tokens'])})",
             f"  val suppressTokens: Seq[Int] = Seq({', '.join(str(t) for t in dec['suppress_tokens'])})",
             f"  val suppressBlank: Seq[Int] = Seq({', '.join(str(t) for t in dec['suppress_blank'])})"]

    def m(d):
        return "Map(" + ", ".join(f'"{k}" -> BigInt("{int(v)}")' for k, v in d.items()) + ")"

    parts = [prog[i:i + 40] for i in range(0, len(prog), 40)]
    for pi, part in enumerate(parts):
        lines.append(f"  def part{pi}: Seq[UOp] = Seq(")
        for u in part:
            lines.append(f'    UOp({u["opc"]}, {m(u["mm"])}, {m(u["vec"])}, {m(u["att"])}, {m(u["kv"])}, {u["rowsSel"]}, {u["framesSel"]}, '
                         f'{u["rowOffSel"]}, {u["keyOffSel"]}, {u["nKeysSel"]}, {u["nQSel"]}, {u["qPos0Sel"]}, {str(u["bBasePos"]).lower()}, '
                         f'{str(u["tokSel"]).lower()}, {str(u["flushK"]).lower()}, BigInt({u["imm"]}), BigInt({u["imm2"]}), "{u["name"]}"),')
        lines.append("  )")
    lines.append("  val program: Seq[UOp] = " + " ++ ".join(f"part{i}" for i in range(len(parts))))
    lines.append("}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(prog)} instructions")
    # human-readable listing
    lst = REPO / "docs" / "microcode.txt"
    lst.write_text("\n".join(f"{i:4d}  {u['name']}" for i, u in enumerate(prog)) + "\n")


if __name__ == "__main__":
    emit()
