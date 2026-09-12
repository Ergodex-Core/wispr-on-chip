package whisper.util

import chisel3._
import chisel3.util._

/** Unsigned restoring divider, one quotient bit per cycle. q = n / d (d > 0). */
class SeqDivider(nBits: Int, dBits: Int) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val n = Input(UInt(nBits.W))
    val d = Input(UInt(dBits.W))
    val busy = Output(Bool())
    val done = Output(Bool())        // one-cycle pulse; q valid from then on until the next start
    val q = Output(UInt(nBits.W))
  })
  val rem = Reg(UInt((nBits + 1).W))
  val q = Reg(UInt(nBits.W))
  val d = Reg(UInt(dBits.W))
  val cnt = Reg(UInt(log2Ceil(nBits + 1).W))
  val busy = RegInit(false.B)
  val done = RegInit(false.B)
  done := false.B
  when(io.start) {
    busy := true.B; rem := 0.U; q := io.n; d := io.d; cnt := nBits.U
  }.elsewhen(busy) {
    val shifted = Cat(rem(nBits - 1, 0), q(nBits - 1))
    val sub = shifted -& d
    val ge = shifted >= d
    rem := Mux(ge, sub(nBits, 0), shifted)
    q := Cat(q(nBits - 2, 0), ge)
    cnt := cnt - 1.U
    when(cnt === 1.U) { busy := false.B; done := true.B }
  }
  io.busy := busy
  io.done := done
  io.q := q
}
