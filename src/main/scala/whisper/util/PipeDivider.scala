package whisper.util

import chisel3._
import chisel3.util._

/** Unsigned restoring divider, fully pipelined (bitsPerStage quotient bits per pipeline stage).
  * q = n / d (d > 0). Throughput one division per cycle; latency = ceil(nBits/bitsPerStage) cycles. */
class PipeDivider(nBits: Int, dBits: Int, bitsPerStage: Int = 2) extends Module {
  val stages = (nBits + bitsPerStage - 1) / bitsPerStage
  val io = IO(new Bundle {
    val in = Flipped(Valid(new Bundle { val n = UInt(nBits.W); val d = UInt(dBits.W) }))
    val out = Valid(UInt(nBits.W))
  })
  class St extends Bundle { val rem = UInt((nBits + 1).W); val q = UInt(nBits.W); val d = UInt(dBits.W) }
  val v = RegInit(VecInit(Seq.fill(stages)(false.B)))
  val st = Reg(Vec(stages, new St))
  def step(s: St): St = {
    val o = Wire(new St)
    var rem = s.rem; var q = s.q
    for (_ <- 0 until bitsPerStage) {
      val shifted = Cat(rem(nBits - 1, 0), q(nBits - 1))
      val ge = shifted >= s.d
      rem = Mux(ge, (shifted - s.d)(nBits, 0), shifted)
      q = Cat(q(nBits - 2, 0), ge)
    }
    o.rem := rem; o.q := q; o.d := s.d
    o
  }
  val in0 = Wire(new St); in0.rem := 0.U; in0.q := io.in.bits.n; in0.d := io.in.bits.d
  for (i <- 0 until stages) {
    v(i) := (if (i == 0) io.in.valid else v(i - 1))
    st(i) := step(if (i == 0) in0 else st(i - 1))
  }
  val latency = stages
  io.out.valid := v(stages - 1)
  io.out.bits := st(stages - 1).q
}
