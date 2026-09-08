package minicpm

import chisel3._
import chisel3.util._

/** Streaming argmax over the LM-head logits (wide beats of 32 int32, n-tiles in ascending order).
  * Ties resolve to the lowest index. Token id = nTile*32 + lane (up to 130560 < 2^17). */
class Sampler(cfg: MiniCPMConfig, eosIds: Seq[Int] = Seq(1, 130073)) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new MatmulOut(cfg)))
    val start = Input(Bool())            // reset the running max before a new logits row
    val token = Output(UInt(18.W))
    val isEos = Output(Bool())
    val done = Output(Bool())            // pulse when the last n-tile has been consumed
  })
  val best = Reg(SInt(32.W)); val bestIdx = Reg(UInt(18.W)); val have = RegInit(false.B)
  io.in.ready := true.B
  val nt = io.in.bits.nTile
  val minV = (-(BigInt(1) << 31)).S(32.W)
  val vals = io.in.bits.data
  // lowest-index max within the beat
  def better(a: (SInt, UInt), b: (SInt, UInt)): (SInt, UInt) = { val take = b._1 > a._1; (Mux(take, b._1, a._1), Mux(take, b._2, a._2)) }
  val (bv, bi) = (0 until 32).map(i => (vals(i), i.U(5.W))).reduceLeft(better)
  val cand = Cat(nt, bi)
  val done = RegInit(false.B); done := false.B
  when(io.start) { have := false.B; best := minV; bestIdx := 0.U }
  when(io.in.fire) {
    when(!have || bv > best) { best := bv; bestIdx := cand; have := true.B }
    when(io.in.bits.last) { done := true.B }
  }
  io.token := bestIdx
  io.isEos := eosIds.map(e => bestIdx === e.U).reduce(_ || _)
  io.done := done
}
