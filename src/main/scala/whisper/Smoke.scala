package whisper

import chisel3._

/** Trivial module used only to prove the toolchain (sbt + chisel + firtool + verilator/svsim) works. */
class Smoke extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(8.W))
    val b = Input(UInt(8.W))
    val y = Output(UInt(8.W))
  })
  io.y := io.a + io.b
}
