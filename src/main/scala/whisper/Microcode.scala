package whisper

import chisel3._
import chisel3.util._

/** Micro-op codes. One instruction issues one unit-level job (matmul / vector / attention / kv setup)
  * and waits for it, or performs sequencer control. Dynamic quantities (n_frames, n_ctx, pos, tok,
  * row-chunk state) are substituted by the Sequencer through the selector fields. */
object UOpc {
  val MATMUL = 0; val VEC = 1; val ATTN = 2; val KVSET = 3
  val CHUNK_BEGIN = 4; val CHUNK_NEXT = 5; val JUMP = 6; val BR_PROMPT = 7   // BR_PROMPT: jump if pos < promptLen-1
  val SAMPLE = 8; val NEXTPOS = 9; val HALT = 10; val SETPOS = 11; val TOK_PROMPT = 12
}
object RowsSel { val Const = 0; val NFrames = 1; val NCtx = 2; val Chunk = 3 }
object BaseSel { val Zero = 0; val Chunk = 1; val Pos = 2 }
object KeysSel { val Const = 0; val NCtx = 1; val PosPlus1 = 2 }

/** Software-side instruction (emitted by gen/emit_microcode.py into generated/Microcode.scala). */
case class UOp(
    opc: Int,
    // MATMUL / VEC / ATTN payloads as flat maps of field name -> value (missing = 0)
    mm: Map[String, BigInt] = Map.empty,
    vec: Map[String, BigInt] = Map.empty,
    att: Map[String, BigInt] = Map.empty,
    kv: Map[String, BigInt] = Map.empty,
    rowsSel: Int = 0,        // MATMUL/VEC rows
    framesSel: Int = 0,      // MATMUL im2col frames (0 const, 1 nFrames)
    rowOffSel: Int = 0,      // MATMUL/VEC rowOff: 0 zero, 1 chunkBase, 2 pos
    keyOffSel: Int = 0,      // KVSET keyOff: 0 zero, 2 pos
    nKeysSel: Int = 0,       // ATTN nKeys
    nQSel: Int = 0,          // ATTN nQueries: 0 const, 2 nCtx
    qPos0Sel: Int = 0,       // ATTN qPos0: 0 zero, 2 pos
    bBasePos: Boolean = false, // VEC ADD: bBase += pos*bStride
    tokSel: Boolean = false,   // VEC: tok from the current token register
    flushK: Boolean = false,   // MATMUL to KV-K: flush the transposer afterwards
    imm: BigInt = 0,           // control immediate: jump target / chunk total sel / chunk size
    imm2: BigInt = 0,
    name: String = "",
)

class MicroInstr extends Bundle {
  val opc = UInt(4.W)
  val mm = new MatmulCmd
  val vec = new VecCmd
  val att = new AttnCmd
  val kv = new KVCmd
  val rowsSel = UInt(2.W)
  val framesSel = UInt(1.W)
  val rowOffSel = UInt(2.W)
  val keyOffSel = UInt(2.W)
  val nKeysSel = UInt(2.W)
  val nQSel = UInt(2.W)
  val qPos0Sel = UInt(2.W)
  val bBasePos = Bool()
  val tokSel = Bool()
  val flushK = Bool()
  val imm = UInt(16.W)
  val imm2 = UInt(16.W)
}

object MicroInstr {
  /** Build a constant instruction from a UOp (elaboration time). Unknown field names are an error. */
  def fromUOp(u: UOp): MicroInstr = {
    val w = Wire(new MicroInstr)
    w := 0.U.asTypeOf(w)
    def fill(b: Record, m: Map[String, BigInt], what: String): Unit = {
      val els = b.elements
      for ((k, v) <- m) {
        require(els.contains(k), s"unknown $what field '$k' in ${u.name}")
        els(k) match {
          case vec: Vec[_] =>   // Vec fields are given as name -> packed value? not used
            throw new IllegalArgumentException(s"vector field $k must be given as $k.<index>")
          case d: Bits => d := v.U(d.getWidth.W)
          case d: Bool => d := (v != 0).B
        }
      }
    }
    // Vec fields (mq/sq) are passed as "mq.0".."mq.5"
    def fillWithVec(b: Record, m: Map[String, BigInt], what: String): Unit = {
      val (vecs, plain) = m.partition(_._1.contains('.'))
      fill(b, plain, what)
      for ((k, v) <- vecs) {
        val Array(f, i) = k.split('.')
        b.elements(f).asInstanceOf[Vec[UInt]](i.toInt) := v.U
      }
    }
    w.opc := u.opc.U
    fillWithVec(w.mm, u.mm, "MatmulCmd")
    fillWithVec(w.vec, u.vec, "VecCmd")
    fillWithVec(w.att, u.att, "AttnCmd")
    fillWithVec(w.kv, u.kv, "KVCmd")
    w.rowsSel := u.rowsSel.U; w.framesSel := u.framesSel.U; w.rowOffSel := u.rowOffSel.U; w.keyOffSel := u.keyOffSel.U
    w.nKeysSel := u.nKeysSel.U; w.nQSel := u.nQSel.U; w.qPos0Sel := u.qPos0Sel.U
    w.bBasePos := u.bBasePos.B; w.tokSel := u.tokSel.B; w.flushK := u.flushK.B
    w.imm := u.imm.U; w.imm2 := u.imm2.U
    w
  }
}
