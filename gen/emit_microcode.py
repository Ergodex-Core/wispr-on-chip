"""Emit a sequencer program (generated/*.scala) from weights/MANIFEST.json params + the chip map.

The program mirrors golden/minicpm_int.py op for op: one row-chunk loop per layer that serves both the
prefill (chunk = up to `--chunk` prompt rows) and decoding (chunk = the single new row at `pos`).

  uv run python gen/emit_microcode.py                      # the full model -> object Microcode
  uv run python gen/emit_microcode.py --layers 2 --max-ctx 256 --chunk 64 \
      --vocab-tiles 256 --object MicrocodeLayer            # the layer-level Verilator test program

Every emitted program is checked statically against the bank / KV / weight-space sizes it will run on
(`check_program`), for the worst-case prefill chunk and the worst-case decode position.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gen.chipmap import (CHUNK, D16, D32, D8, D_FF, D_MODEL, FF16, FF8, KV16, KV_DIM, MAX_CTX, N_KV_HEAD,  # noqa: E402
                         N_LAYERS, REPO, SPACE_OF, bank_words, k16_base, kv_kbase, kv_vbase, kv_words,
                         per_head, tensor_bases)

GEN = REPO / "src/main/scala/minicpm/generated"
OPC = dict(MATMUL=0, VEC=1, ATTN=2, KVSET=3, CHUNK_BEGIN=4, CHUNK_NEXT=5, JUMP=6, SAMPLE=7, HALT=8)
ROWS = dict(Const=0, Chunk=1)
BASE = dict(Zero=0, Chunk=1, Last=2)
KEYS = dict(Const=0, ChunkEnd=1)
VOP = dict(RMSNORM=0, DYNQ=1, ADD=2, EMBED=3, ROPE=4, SILUMUL=5)
BITS = dict(int8=0, int16=1, int32=2)
MODE = dict(int8=0, int16=1, raw=2, wide=3, int32=4)
MODE_WORDS = {0: 1, 1: 2, 4: 4}          # 256-bit words per output n-tile, by mode
SINK = dict(bank=0, attn=1, sampler=2, kv=3)
X, A8, QK16, Q8, T32, G16, U16, A8X = range(8)

man = json.load(open(REPO / "weights" / "MANIFEST.json"))
P = man["params"]
B = tensor_bases(man)
T = man["tensors"]


class Prog:
    """A program under construction, with the parameters it is emitted for."""

    def __init__(self, layers: int, max_ctx: int, chunk: int, vocab_tiles: int):
        self.layers, self.max_ctx, self.chunk, self.vocab_tiles = layers, max_ctx, chunk, vocab_tiles
        self.ops: list[dict] = []
        self.labels: dict[str, int] = {}
        self.banks = bank_words(max_ctx, chunk)
        self.k16 = k16_base(chunk)

    def op(self, opc, **kw):
        d = dict(opc=OPC[opc], mm={}, vec={}, att={}, kv={}, rowsSel=0, rowOffSel=0, posSel=0, keyOffSel=0,
                 nKeysSel=0, nQSel=0, qPos0Sel=0, flushK=False, imm=0, imm2=0, name="")
        d.update(kw)
        self.ops.append(d)
        return d

    def label(self, name):
        self.labels[name] = len(self.ops)

    # ---------------------------------------------------------------- instruction builders
    def matmul(self, tensor, act, out, rows_sel="Chunk", rows=0, rowoff="Zero", dyn_bank=None, name="",
               w_name=None, mult_name=None, n_tiles=None):
        """act=(bank, base, stride); out=(mode, sink, bank, base, stride)"""
        L = P["linears"][tensor]
        K, N = L["K"], L["N"]
        KT, NT = K // 32, (n_tiles or N // 32)
        mm = dict(rows=rows, kTiles=KT, nTiles=NT, actSrc=0, actBank=act[0], actBase=act[1], actStride=act[2], actKOff=0,
                  wSrc=0, wBase=B[w_name or f"{tensor}.w"], wStrideN=KT * 4, wStrideK=4, wStrideG=1,
                  outMode=MODE[out[0]], outSink=SINK[out[1]], outBank=out[2], outBase=out[3], outStride=out[4], outTag=0,
                  rowBase=0, rowOff=0, s1=L["s1"], multBase=B[mult_name or f"{tensor}.mult"], biasBase=0, hasBias=0,
                  dynamic=int(L["dynamic"]), rowfacBank=dyn_bank if dyn_bank is not None else 0, outLocal=1)
        return self.op("MATMUL", mm=mm, rowsSel=ROWS[rows_sel], rowOffSel=BASE[rowoff], name=name or tensor)

    def vec(self, vop, rows_sel="Chunk", rows=0, rowoff="Zero", pos_sel="Zero", name="", flush_k=False, **fields):
        v = dict(op=VOP[vop], rows=rows, rowOff=0)
        v.update(fields)
        return self.op("VEC", vec=v, rowsSel=ROWS[rows_sel], rowOffSel=BASE[rowoff],
                       posSel=1 if pos_sel == "Chunk" else 0, flushK=flush_k, name=name)

    def rmsnorm(self, nname, rows_sel="Chunk", rowoff="Chunk"):
        eps = P["norms"][nname]["eps_q"]
        return self.vec("RMSNORM", rows_sel, 1 if rows_sel == "Const" else 0, rowoff, name=nname, cols=D_MODEL,
                        inBank=X, inBase=0, inStride=D32, inBits=BITS["int32"], inLocal=0,
                        outBank=A8, outBase=0, outStride=D8, outLocal=1, rowfacBank=A8, rowBase=0, rfLocal=1,
                        gBase=B[f"{nname}.g"], epsLo=eps & 0xFFFFFFFF, epsHi=eps >> 32)

    def dynq(self, inb, outb, cols, name):
        return self.vec("DYNQ", name=name, cols=cols, inBank=inb[0], inBase=inb[1], inStride=inb[2], inBits=BITS["int16"],
                        inLocal=1, outBank=outb[0], outBase=outb[1], outStride=outb[2], outLocal=1,
                        rowfacBank=outb[0], rowBase=0, rfLocal=1)

    def rope(self, rname, inb, cols, outb=None, to_kv=False, name=""):
        R = P["ropes"][rname]
        f = dict(cols=cols, inBank=inb[0], inBase=inb[1], inStride=inb[2], inBits=BITS["int16"], inLocal=1, outLocal=1,
                 reqMult=R["m_r"], reqShift=R["s_r"], ropeBase=B["rope"], rowBase=0)
        if to_kv:
            f.update(outSink=1, outBank=0, outBase=0, outStride=cols // 32)
        else:
            f.update(outSink=0, outBank=outb[0], outBase=outb[1], outStride=outb[2])
        return self.vec("ROPE", pos_sel="Chunk", name=name or rname, flush_k=to_kv, **f)

    def add(self, aname, b, name=""):
        A = P["adds"][aname]
        return self.vec("ADD", rowoff="Chunk", name=name or aname, cols=D_MODEL, inBank=X, inBase=0, inStride=D32,
                        inBits=BITS["int32"], inLocal=0, bBank=b[0], bBase=b[1], bStride=b[2], bBits=BITS["int32"],
                        bLocal=1, outBank=X, outBase=0, outStride=D32, outLocal=0, ma=A["ma"], mb=A["mb"])

    def silumul(self, sname):
        S = P["silus"][sname]
        return self.vec("SILUMUL", name=sname, cols=D_FF, inBank=G16, inBase=0, inStride=FF16, inBits=BITS["int16"],
                        inLocal=1, bBank=U16, bBase=0, bStride=FF16, bBits=BITS["int16"], bLocal=1,
                        outBank=A8X, outBase=0, outStride=FF8, outLocal=1, rowfacBank=A8X, rowBase=0, rfLocal=1,
                        mSig=S["m_sig"], sSig=S["s_sig"])

    def kvset(self, is_k, base):
        return self.op("KVSET", kv=dict(isK=int(is_k), base=base, keysMax=self.max_ctx, keyOff=0, flush=0),
                       keyOffSel=1, name=f"kvset {'K' if is_k else 'V'}")

    def attn(self, aname, kbase, vbase):
        A = P["attns"][aname]
        a = dict(nQueries=1, nKeys=0, causal=1, qPos0=0, qBank=Q8, qBase=0, qStride=D8, kBase=kbase, vBase=vbase,
                 keysMax=self.max_ctx, outBank=T32, outBase=0, outStride=D16)
        for h in range(16):
            a[f"mq.{h}"] = A["mq"][h]
            a[f"sq.{h}"] = A["sq"][h]
        return self.op("ATTN", att=a, nQSel=1, nKeysSel=KEYS["ChunkEnd"], qPos0Sel=1, name=aname)

    # ---------------------------------------------------------------- the program
    def build(self):
        self.label("loop")
        # ---- embedding of the chunk rows (prompt rows, or the one new row): X[pos] = emb[tok[pos]] * resmult
        self.op("CHUNK_BEGIN", imm2=self.chunk, name="embed chunks")
        self.label("embed")
        self.vec("EMBED", rowoff="Chunk", name="embed", cols=D_MODEL, inBits=BITS["int8"], outBank=X, outBase=0,
                 outStride=D32, outLocal=0, tokFromMem=1, maTableBase=B["embed.resmult"], embBase=B["embed.00"])
        self.op("CHUNK_NEXT", imm=self.labels["embed"], name="embed next")
        for l in range(self.layers):
            p = f"L{l}"
            self.op("CHUNK_BEGIN", imm2=self.chunk, name=f"{p} chunks")
            self.label(p)
            self.rmsnorm(f"{p}.norm1")
            self.matmul(f"{p}.q", (A8, 0, D8), ("int16", "bank", QK16, 0, D16), dyn_bank=A8)
            self.matmul(f"{p}.k", (A8, 0, D8), ("int16", "bank", QK16, self.k16, KV16), dyn_bank=A8)
            self.kvset(False, kv_vbase(l, self.max_ctx))
            self.matmul(f"{p}.v", (A8, 0, D8), ("int8", "kv", 0, 0, KV_DIM // 32), dyn_bank=A8, name=f"{p}.v -> KV")
            self.rope(f"{p}.rope_q", (QK16, 0, D16), D_MODEL, outb=(Q8, 0, D8))
            self.kvset(True, kv_kbase(l, self.max_ctx))
            self.rope(f"{p}.rope_k", (QK16, self.k16, KV16), KV_DIM, to_kv=True, name=f"{p}.rope_k -> KV")
            self.attn(f"{p}.attn", kv_kbase(l, self.max_ctx), kv_vbase(l, self.max_ctx))
            self.dynq((T32, 0, D16), (A8, 0, D8), D_MODEL, "dynq attn out")
            self.matmul(f"{p}.o", (A8, 0, D8), ("int32", "bank", T32, 0, D32), dyn_bank=A8)
            self.add(f"{p}.add1", (T32, 0, D32))
            self.rmsnorm(f"{p}.norm2")
            self.matmul(f"{p}.gate", (A8, 0, D8), ("int16", "bank", G16, 0, FF16), dyn_bank=A8)
            self.matmul(f"{p}.up", (A8, 0, D8), ("int16", "bank", U16, 0, FF16), dyn_bank=A8)
            self.silumul(f"{p}.silu")
            self.matmul(f"{p}.down", (A8X, 0, FF8), ("int32", "bank", T32, 0, D32), dyn_bank=A8X)
            self.add(f"{p}.add2", (T32, 0, D32))
            self.op("CHUNK_NEXT", imm=self.labels[p], name=f"{p} next")
        # ---- final norm on the last row, LM head over the vocabulary (contiguous slices), sample
        self.rmsnorm("norm_f", rows_sel="Const", rowoff="Last")
        self.matmul("lm", (A8, 0, D8), ("wide", "sampler", 0, 0, 0), rows_sel="Const", rows=1, dyn_bank=A8,
                    w_name="lm.00.w", mult_name="lm.00.mult", n_tiles=self.vocab_tiles)
        self.op("SAMPLE", name="tok = argmax (emit; halt on eos / limit)")
        self.op("JUMP", imm=self.labels["loop"], name="next position")
        self.op("HALT", name="halt")
        return self


# ------------------------------------------------------------------------------------------------ checks
def check_program(pr: Prog):
    """Static bounds check of every instruction against the memories it will address, for the worst-case
    prefill chunk (rowOff = max_ctx - chunk, rows = chunk) and decode position (rowOff = max_ctx - 1)."""
    banks, max_ctx, chunk = pr.banks, pr.max_ctx, pr.chunk
    wWords = max(B[n] + T[n]["depth"] for n in T if T[n]["kind"] == "w8")
    pWords = max(B[n] + T[n]["depth"] for n in T if T[n]["kind"] == "i32vec")
    tWords = max(B[n] + T[n]["depth"] for n in T if T[n]["kind"] in ("i8mat", "i16mat"))
    kvW = kv_words(pr.layers, max_ctx)
    bad = []

    def chk(cond, what):
        if not cond:
            bad.append(what)

    def rows_of(u, field):
        return chunk if u["rowsSel"] == ROWS["Chunk"] else max(field.get("rows", 1), 1)

    def last_row(u, rows, local):
        """largest row index this operand addresses"""
        if local or u["rowOffSel"] == BASE["Zero"]:
            return rows - 1
        return max_ctx - 1                       # Chunk (prefill/decode) or Last

    for i, u in enumerate(pr.ops):
        nm = f"{i} {u['name']}"
        if u["opc"] == OPC["MATMUL"]:
            m = u["mm"]
            rows = rows_of(u, m)
            chk(rows <= 1536, f"{nm}: rows {rows} > accRows")
            a = last_row(u, rows, False) * m["actStride"] + m["actBase"] + m["actKOff"] + m["kTiles"] - 1
            chk(a < banks[m["actBank"]], f"{nm}: act addr {a} >= bank {m['actBank']} ({banks[m['actBank']]})")
            w = m["wBase"] + (m["nTiles"] - 1) * m["wStrideN"] + (m["kTiles"] - 1) * m["wStrideK"] + 3 * m["wStrideG"]
            chk(w < (wWords if m["wSrc"] == 0 else kvW), f"{nm}: weight addr {w} out of range")
            chk(m["multBase"] + (m["nTiles"] - 1) // 32 < pWords, f"{nm}: mult addr out of range")
            if m["outSink"] == SINK["bank"]:
                o = m["outBase"] + last_row(u, rows, bool(m["outLocal"])) * m["outStride"] + (m["nTiles"] - 1) * MODE_WORDS[m["outMode"]]
                chk(o < banks[m["outBank"]], f"{nm}: out addr {o} >= bank {m['outBank']} ({banks[m['outBank']]})")
            if m["outSink"] == SINK["kv"]:
                chk(m["outMode"] == MODE["int8"], f"{nm}: KV sink needs int8 beats")
            if m["dynamic"]:
                chk(m["rowBase"] + last_row(u, rows, False) < max_ctx, f"{nm}: rowfac index out of range")
        elif u["opc"] == OPC["VEC"]:
            v = u["vec"]
            rows = rows_of(u, v)
            words = {BITS["int8"]: v["cols"] // 32, BITS["int16"]: v["cols"] // 16, BITS["int32"]: v["cols"] // 8}
            if v["op"] not in (VOP["EMBED"],):
                a = v["inBase"] + last_row(u, rows, bool(v.get("inLocal", 0))) * v["inStride"] + words[v.get("inBits", 1)] - 1
                chk(a < banks[v["inBank"]], f"{nm}: A addr {a} >= bank {v['inBank']} ({banks[v['inBank']]})")
            if v["op"] in (VOP["ADD"], VOP["SILUMUL"]):
                b = v["bBase"] + last_row(u, rows, bool(v.get("bLocal", 0))) * v["bStride"] + words[v.get("bBits", 1)] - 1
                chk(b < banks[v["bBank"]], f"{nm}: B addr {b} >= bank {v['bBank']} ({banks[v['bBank']]})")
            if v.get("outSink", 0) == 0:
                outw = {VOP["RMSNORM"]: v["cols"] // 32, VOP["DYNQ"]: v["cols"] // 32, VOP["SILUMUL"]: v["cols"] // 32,
                        VOP["ROPE"]: v["cols"] // 32, VOP["ADD"]: v["cols"] // 8, VOP["EMBED"]: v["cols"] // 8}[v["op"]]
                o = v["outBase"] + last_row(u, rows, bool(v.get("outLocal", 0))) * v["outStride"] + outw - 1
                chk(o < banks[v["outBank"]], f"{nm}: out addr {o} >= bank {v['outBank']} ({banks[v['outBank']]})")
            if v["op"] == VOP["ROPE"]:
                r = v["ropeBase"] + (max_ctx - 1) * 8 + 7
                chk(r < tWords, f"{nm}: rope table addr {r} >= t space ({tWords})")
                chk(v["cols"] in (D_MODEL, KV_DIM), f"{nm}: rope cols {v['cols']}")
            if v["op"] == VOP["EMBED"]:
                chk(v["embBase"] + (P["model"]["n_vocab"] - 1) * (v["cols"] // 32) + v["cols"] // 32 - 1 < tWords,
                    f"{nm}: embedding row out of range")
                chk(v["maTableBase"] + (P["model"]["n_vocab"] - 1) // 32 < pWords, f"{nm}: resmult out of range")
            if v["op"] in (VOP["RMSNORM"], VOP["DYNQ"], VOP["SILUMUL"]):
                chk(v["rowBase"] + last_row(u, rows, bool(v.get("rfLocal", 0))) < max_ctx, f"{nm}: rowfac index out of range")
            if v["op"] == VOP["RMSNORM"]:
                chk(v["gBase"] + v["cols"] // 32 - 1 < pWords, f"{nm}: gain out of range")
        elif u["opc"] == OPC["ATTN"]:
            a = u["att"]
            q = a["qBase"] + (chunk - 1) * a["qStride"] + D_MODEL // 32 - 1
            chk(q < banks[a["qBank"]], f"{nm}: q addr {q} >= bank {a['qBank']}")
            o = a["outBase"] + (chunk - 1) * a["outStride"] + D_MODEL // 16 - 1
            chk(o < banks[a["outBank"]], f"{nm}: out addr {o} >= bank {a['outBank']}")
            top = max(a["kBase"], a["vBase"]) + (N_KV_HEAD - 1) * per_head(max_ctx) + per_head(max_ctx) - 1
            chk(top < kvW, f"{nm}: KV addr {top} >= cache ({kvW})")
        elif u["opc"] == OPC["KVSET"]:
            k = u["kv"]
            chk(k["base"] + N_KV_HEAD * per_head(max_ctx) - 1 < kvW, f"{nm}: KV base out of range")
            chk(k["keysMax"] == max_ctx, f"{nm}: keysMax {k['keysMax']} != maxCtx {max_ctx}")
        elif u["opc"] in (OPC["JUMP"], OPC["CHUNK_NEXT"]):
            chk(0 <= u["imm"] < len(pr.ops), f"{nm}: jump target {u['imm']} out of range")
    chk(len(pr.ops) < 1024, f"program of {len(pr.ops)} instructions exceeds the 10-bit pc")
    if bad:
        raise SystemExit("microcode check failed:\n  " + "\n  ".join(bad))
    return dict(instructions=len(pr.ops), banks=banks, kv_words=kvW, w_words=wWords, p_words=pWords, t_words=tWords)


# ------------------------------------------------------------------------------------------------ emit
def emit(pr: Prog, obj: str, out: Path, listing: Path | None):
    info = check_program(pr)
    dec = P["decoding"]
    lines = [f"// GENERATED by gen/emit_microcode.py (--layers {pr.layers} --max-ctx {pr.max_ctx} --chunk {pr.chunk} "
             f"--vocab-tiles {pr.vocab_tiles}) — do not edit.", "package minicpm.generated", "",
             "import minicpm.{MicroProgram, UOp}", "",
             f"object {obj} extends MicroProgram {{",
             f"  val nLayers = {pr.layers}", f"  val kvLayers = {pr.layers}", f"  val maxCtx = {pr.max_ctx}",
             f"  val chunkRows = {pr.chunk}", f"  val vocabTiles = {pr.vocab_tiles}",
             f"  val bankWords: Seq[Int] = Seq({', '.join(str(w) for w in pr.banks)})",
             f"  val eosIds: Seq[Int] = Seq({', '.join(str(t) for t in dec['eos_ids'])})"]

    def m(d):
        return "Map(" + ", ".join(f'"{k}" -> BigInt("{int(v)}")' for k, v in d.items()) + ")"

    parts = [pr.ops[i:i + 40] for i in range(0, len(pr.ops), 40)]
    for pi, part in enumerate(parts):
        lines.append(f"  def part{pi}: Seq[UOp] = Seq(")
        for u in part:
            lines.append(f'    UOp({u["opc"]}, {m(u["mm"])}, {m(u["vec"])}, {m(u["att"])}, {m(u["kv"])}, {u["rowsSel"]}, {u["rowOffSel"]}, '
                         f'{u["posSel"]}, {u["keyOffSel"]}, {u["nKeysSel"]}, {u["nQSel"]}, {u["qPos0Sel"]}, {str(u["flushK"]).lower()}, '
                         f'BigInt({u["imm"]}), BigInt({u["imm2"]}), "{u["name"]}"),')
        lines.append("  )")
    lines.append("  val program: Seq[UOp] = " + " ++ ".join(f"part{i}" for i in range(len(parts))))
    lines.append("}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(pr.ops)} instructions, checks OK "
          f"(banks {info['banks']}, kv {info['kv_words']} words)")
    if listing:
        listing.write_text("\n".join(f"{i:4d}  {u['name']}" for i, u in enumerate(pr.ops)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--vocab-tiles", type=int, default=None, help="LM-head n-tiles (default: the whole vocabulary)")
    ap.add_argument("--object", default="Microcode")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    vt = a.vocab_tiles or P["model"]["n_vocab"] // 32
    pr = Prog(a.layers, a.max_ctx, a.chunk, vt).build()
    out = Path(a.out) if a.out else GEN / f"{a.object}.scala"
    emit(pr, a.object, out, REPO / "docs" / "microcode.txt" if a.object == "Microcode" else None)


if __name__ == "__main__":
    main()
