"""Emit the sequencer program (generated/Microcode.scala) from weights/MANIFEST.json params + the chip map.
The program mirrors golden/minicpm_int.py op for op: one row-chunk loop per layer that serves both the
prefill (chunk = up to 512 prompt rows) and decoding (chunk = the single new row at `pos`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gen.chipmap import (BANK_WORDS, CHUNK, D16, D32, D8, D_FF, D_MODEL, FF16, FF8, K16_BASE, KV16, KV_DIM, MAX_CTX,  # noqa: E402
                         N_LAYERS, REPO, kv_kbase, kv_vbase, tensor_bases)

OUT = REPO / "src/main/scala/minicpm/generated/Microcode.scala"
OPC = dict(MATMUL=0, VEC=1, ATTN=2, KVSET=3, CHUNK_BEGIN=4, CHUNK_NEXT=5, JUMP=6, SAMPLE=7, HALT=8)
ROWS = dict(Const=0, Chunk=1)
BASE = dict(Zero=0, Chunk=1, Last=2)
KEYS = dict(Const=0, ChunkEnd=1)
VOP = dict(RMSNORM=0, DYNQ=1, ADD=2, EMBED=3, ROPE=4, SILUMUL=5)
MODE = dict(int8=0, int16=1, raw=2, wide=3, int32=4)
SINK = dict(bank=0, attn=1, sampler=2, kv=3)
X, A8, QK16, Q8, T32, G16, U16, A8X = range(8)

man = json.load(open(REPO / "weights" / "MANIFEST.json"))
P = man["params"]
B = tensor_bases(man)
LAYERS = P["model"]["n_layers"]
prog: list[dict] = []
labels: dict[str, int] = {}


def op(opc, **kw):
    d = dict(opc=OPC[opc], mm={}, vec={}, att={}, kv={}, rowsSel=0, rowOffSel=0, posSel=0, keyOffSel=0, nKeysSel=0, nQSel=0,
             qPos0Sel=0, flushK=False, imm=0, imm2=0, name="")
    d.update(kw)
    prog.append(d)
    return d


def label(name):
    labels[name] = len(prog)


def matmul(tensor, act, out, rows_sel="Chunk", rows=0, rowoff="Zero", dyn_bank=None, name="", w_name=None, mult_name=None, n_tiles=None):
    """act=(bank, base, stride); out=(mode, sink, bank, base, stride)"""
    L = P["linears"][tensor]
    K, N = L["K"], L["N"]
    KT, NT = K // 32, (n_tiles or N // 32)
    mm = dict(rows=rows, kTiles=KT, nTiles=NT, actSrc=0, actBank=act[0], actBase=act[1], actStride=act[2], actKOff=0,
              wSrc=0, wBase=B[w_name or f"{tensor}.w"], wStrideN=KT * 4, wStrideK=4, wStrideG=1,
              outMode=MODE[out[0]], outSink=SINK[out[1]], outBank=out[2], outBase=out[3], outStride=out[4], outTag=0,
              rowBase=0, rowOff=0, s1=L["s1"], multBase=B[mult_name or f"{tensor}.mult"], biasBase=0, hasBias=0,
              dynamic=int(L["dynamic"]), rowfacBank=dyn_bank if dyn_bank is not None else 0, outLocal=1)
    return op("MATMUL", mm=mm, rowsSel=ROWS[rows_sel], rowOffSel=BASE[rowoff], name=name or tensor)


def vec(vop, rows_sel="Chunk", rows=0, rowoff="Zero", pos_sel="Zero", name="", flush_k=False, **fields):
    v = dict(op=VOP[vop], rows=rows, rowOff=0)
    v.update(fields)
    return op("VEC", vec=v, rowsSel=ROWS[rows_sel], rowOffSel=BASE[rowoff], posSel=1 if pos_sel == "Chunk" else 0, flushK=flush_k, name=name)


def rmsnorm(nname, rows_sel="Chunk", rowoff="Chunk"):
    N = P["norms"][nname]
    eps = N["eps_q"]
    return vec("RMSNORM", rows_sel, 1 if rows_sel == "Const" else 0, rowoff, name=nname, cols=D_MODEL,
               inBank=X, inBase=0, inStride=D32, inBits=2, inLocal=0,
               outBank=A8, outBase=0, outStride=D8, outLocal=1, rowfacBank=A8, rowBase=0, rfLocal=1,
               gBase=B[f"{nname}.g"], epsLo=eps & 0xFFFFFFFF, epsHi=eps >> 32)


def dynq(inb, outb, cols, name):
    return vec("DYNQ", name=name, cols=cols, inBank=inb[0], inBase=inb[1], inStride=inb[2], inBits=1, inLocal=1,
               outBank=outb[0], outBase=outb[1], outStride=outb[2], outLocal=1, rowfacBank=outb[0], rowBase=0, rfLocal=1)


def rope(rname, inb, cols, outb=None, to_kv=False, name=""):
    R = P["ropes"][rname]
    f = dict(cols=cols, inBank=inb[0], inBase=inb[1], inStride=inb[2], inBits=1, inLocal=1, outLocal=1,
             reqMult=R["m_r"], reqShift=R["s_r"], ropeBase=B["rope"], rowBase=0)
    if to_kv:
        f.update(outSink=1, outBank=0, outBase=0, outStride=cols // 32)
    else:
        f.update(outSink=0, outBank=outb[0], outBase=outb[1], outStride=outb[2])
    return vec("ROPE", pos_sel="Chunk", name=name or rname, flush_k=to_kv, **f)


def add(aname, b, name=""):
    A = P["adds"][aname]
    return vec("ADD", rowoff="Chunk", name=name or aname, cols=D_MODEL, inBank=X, inBase=0, inStride=D32, inBits=2, inLocal=0,
               bBank=b[0], bBase=b[1], bStride=b[2], bBits=2, bLocal=1, outBank=X, outBase=0, outStride=D32, outLocal=0,
               ma=A["ma"], mb=A["mb"])


def silumul(sname):
    S = P["silus"][sname]
    return vec("SILUMUL", name=sname, cols=D_FF, inBank=G16, inBase=0, inStride=FF16, inBits=1, inLocal=1,
               bBank=U16, bBase=0, bStride=FF16, bBits=1, bLocal=1, outBank=A8X, outBase=0, outStride=FF8, outLocal=1,
               rowfacBank=A8X, rowBase=0, rfLocal=1, mSig=S["m_sig"], sSig=S["s_sig"])


def kvset(is_k, base):
    return op("KVSET", kv=dict(isK=int(is_k), base=base, keysMax=MAX_CTX, keyOff=0, flush=0), keyOffSel=1, name=f"kvset {'K' if is_k else 'V'}")


def attn(aname, kbase, vbase):
    A = P["attns"][aname]
    a = dict(nQueries=1, nKeys=0, causal=1, qPos0=0, qBank=Q8, qBase=0, qStride=D8, kBase=kbase, vBase=vbase, keysMax=MAX_CTX,
             outBank=T32, outBase=0, outStride=D16)
    for h in range(16):
        a[f"mq.{h}"] = A["mq"][h]
        a[f"sq.{h}"] = A["sq"][h]
    return op("ATTN", att=a, nQSel=1, nKeysSel=KEYS["ChunkEnd"], qPos0Sel=1, name=aname)


def program():
    label("loop")
    # ---- embedding of the chunk rows (prompt rows, or the one new row): X[pos] = emb[tok[pos]] * resmult
    op("CHUNK_BEGIN", imm2=CHUNK, name="embed chunks")
    label("embed")
    vec("EMBED", rowoff="Chunk", name="embed", cols=D_MODEL, inBits=0, outBank=X, outBase=0, outStride=D32, outLocal=0,
        tokFromMem=1, maTableBase=B["embed.resmult"], embBase=B["embed.00"])
    op("CHUNK_NEXT", imm=labels["embed"], name="embed next")
    for l in range(LAYERS):
        p = f"L{l}"
        op("CHUNK_BEGIN", imm2=CHUNK, name=f"{p} chunks")
        label(p)
        rmsnorm(f"{p}.norm1")
        matmul(f"{p}.q", (A8, 0, D8), ("int16", "bank", QK16, 0, D16), dyn_bank=A8)
        matmul(f"{p}.k", (A8, 0, D8), ("int16", "bank", QK16, K16_BASE, KV16), dyn_bank=A8)
        kvset(False, kv_vbase(l))
        matmul(f"{p}.v", (A8, 0, D8), ("int8", "kv", 0, 0, KV_DIM // 32), dyn_bank=A8, name=f"{p}.v -> KV")
        rope(f"{p}.rope_q", (QK16, 0, D16), D_MODEL, outb=(Q8, 0, D8))
        kvset(True, kv_kbase(l))
        rope(f"{p}.rope_k", (QK16, K16_BASE, KV16), KV_DIM, to_kv=True, name=f"{p}.rope_k -> KV")
        attn(f"{p}.attn", kv_kbase(l), kv_vbase(l))
        dynq((T32, 0, D16), (A8, 0, D8), D_MODEL, "dynq attn out")
        matmul(f"{p}.o", (A8, 0, D8), ("int32", "bank", T32, 0, D32), dyn_bank=A8)
        add(f"{p}.add1", (T32, 0, D32))
        rmsnorm(f"{p}.norm2")
        matmul(f"{p}.gate", (A8, 0, D8), ("int16", "bank", G16, 0, FF16), dyn_bank=A8)
        matmul(f"{p}.up", (A8, 0, D8), ("int16", "bank", U16, 0, FF16), dyn_bank=A8)
        silumul(f"{p}.silu")
        matmul(f"{p}.down", (A8X, 0, FF8), ("int32", "bank", T32, 0, D32), dyn_bank=A8X)
        add(f"{p}.add2", (T32, 0, D32))
        op("CHUNK_NEXT", imm=labels[p], name=f"{p} next")
    # ---- final norm on the last row, LM head over the whole vocabulary (16 contiguous slices), sample
    rmsnorm("norm_f", rows_sel="Const", rowoff="Last")
    n_tiles = P["model"]["n_vocab"] // 32
    matmul("lm", (A8, 0, D8), ("wide", "sampler", 0, 0, 0), rows_sel="Const", rows=1, dyn_bank=A8, w_name="lm.00.w", mult_name="lm.00.mult", n_tiles=n_tiles)
    op("SAMPLE", name="tok = argmax (emit; halt on eos / limit)")
    op("JUMP", imm=labels["loop"], name="next position")
    op("HALT", name="halt")


def emit():
    program()
    dec = P["decoding"]
    lines = ["// GENERATED by gen/emit_microcode.py — do not edit.", "package minicpm.generated", "", "import minicpm.UOp", "",
             "object Microcode {", f"  val nLayers = {LAYERS}", f"  val kvLayers = {LAYERS}", f"  val maxCtx = {MAX_CTX}", f"  val chunkRows = {CHUNK}",
             f"  val bankWords: Seq[Int] = Seq({', '.join(str(w) for w in BANK_WORDS)})",
             f"  val eosIds: Seq[Int] = Seq({', '.join(str(t) for t in dec['eos_ids'])})"]

    def m(d):
        return "Map(" + ", ".join(f'"{k}" -> BigInt("{int(v)}")' for k, v in d.items()) + ")"

    parts = [prog[i:i + 40] for i in range(0, len(prog), 40)]
    for pi, part in enumerate(parts):
        lines.append(f"  def part{pi}: Seq[UOp] = Seq(")
        for u in part:
            lines.append(f'    UOp({u["opc"]}, {m(u["mm"])}, {m(u["vec"])}, {m(u["att"])}, {m(u["kv"])}, {u["rowsSel"]}, {u["rowOffSel"]}, '
                         f'{u["posSel"]}, {u["keyOffSel"]}, {u["nKeysSel"]}, {u["nQSel"]}, {u["qPos0Sel"]}, {str(u["flushK"]).lower()}, '
                         f'BigInt({u["imm"]}), BigInt({u["imm2"]}), "{u["name"]}"),')
        lines.append("  )")
    lines.append("  val program: Seq[UOp] = " + " ++ ".join(f"part{i}" for i in range(len(parts))))
    lines.append("}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(prog)} instructions")
    lst = REPO / "docs" / "microcode.txt"
    lst.write_text("\n".join(f"{i:4d}  {u['name']}" for i, u in enumerate(prog)) + "\n")


if __name__ == "__main__":
    emit()
