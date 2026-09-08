package minicpm

import chisel3._
import chisel3.util._

/** Micro-op codes. One instruction issues one unit-level job (matmul / vector / attention / kv setup)
  * and waits for it, or performs sequencer control. Dynamic quantities (prompt length, decode position,
  * row-chunk state) are substituted by the Sequencer through the selector fields. */
object UOpc {
  val MATMUL = 0; val VEC = 1; val ATTN = 2; val KVSET = 3
  val CHUNK_BEGIN = 4; val CHUNK_NEXT = 5; val JUMP = 6; val SAMPLE = 7; val HALT = 8
}
object RowsSel { val Const = 0; val Chunk = 1 }

/** A generated micro-program together with the chip parameters it was emitted for (gen/emit_microcode.py
  * checks every instruction against these before emitting). `MiniCPMConfig.forProgram` builds the matching
  * configuration, and `MiniCPMTop` re-checks the agreement at elaboration. */
trait MicroProgram {
  def nLayers: Int
  def kvLayers: Int
  def maxCtx: Int
  def chunkRows: Int
  def vocabTiles: Int
  def bankWords: Seq[Int]
  def eosIds: Seq[Int]
  def program: Seq[UOp]
}
object BaseSel { val Zero = 0; val Chunk = 1; val Last = 2 }      // rowOff / posBase / keyOff / qPos0
object KeysSel { val Const = 0; val ChunkEnd = 1 }

/** Software-side instruction (emitted by gen/emit_microcode.py into generated/Microcode.scala). */
case class UOp(
    opc: Int,
    mm: Map[String, BigInt] = Map.empty,
    vec: Map[String, BigInt] = Map.empty,
    att: Map[String, BigInt] = Map.empty,
    kv: Map[String, BigInt] = Map.empty,
    rowsSel: Int = 0,        // MATMUL/VEC rows: 0 const, 1 chunkRows
    rowOffSel: Int = 0,      // MATMUL/VEC rowOff: 0 zero, 1 chunkBase, 2 lastRow
    posSel: Int = 0,         // VEC posBase: 0 zero, 1 chunkBase
    keyOffSel: Int = 0,      // KVSET keyOff: 0 zero, 1 chunkBase
    nKeysSel: Int = 0,       // ATTN nKeys: 0 const, 1 chunkBase + chunkRows
    nQSel: Int = 0,          // ATTN nQueries: 0 const, 1 chunkRows
    qPos0Sel: Int = 0,       // ATTN qPos0: 0 zero, 1 chunkBase
    flushK: Boolean = false, // MATMUL/VEC writing K into the cache: flush the transposer afterwards
    imm: BigInt = 0,         // control immediate: jump target
    imm2: BigInt = 0,        // CHUNK_BEGIN: chunk size
    name: String = "",
)

class MicroInstr extends Bundle {
  val opc = UInt(4.W)
  val mm = new MatmulCmd
  val vec = new VecCmd
  val att = new AttnCmd
  val kv = new KVCmd
  val rowsSel = UInt(1.W)
  val rowOffSel = UInt(2.W)
  val posSel = UInt(1.W)
  val keyOffSel = UInt(1.W)
  val nKeysSel = UInt(1.W)
  val nQSel = UInt(1.W)
  val qPos0Sel = UInt(1.W)
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
          case _: Vec[_] => throw new IllegalArgumentException(s"vector field $k must be given as $k.<index>")
          case d: Bool => d := (v != 0).B
          case d: UInt => require(v.bitLength <= d.getWidth, s"$what.$k = $v does not fit ${d.getWidth} bits in ${u.name}"); d := v.U(d.getWidth.W)
          case d => throw new IllegalArgumentException(s"unsupported field type for $k: $d")
        }
      }
    }
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
    w.rowsSel := u.rowsSel.U; w.rowOffSel := u.rowOffSel.U; w.posSel := u.posSel.U; w.keyOffSel := u.keyOffSel.U
    w.nKeysSel := u.nKeysSel.U; w.nQSel := u.nQSel.U; w.qPos0Sel := u.qPos0Sel.U
    w.flushK := u.flushK.B
    w.imm := u.imm.U; w.imm2 := u.imm2.U
    w
  }
}
