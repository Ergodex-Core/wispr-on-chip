package whisper

import chisel3._
import chisel3.simulator.stimulus.ResetProcedure
import org.scalatest.flatspec.AnyFlatSpec
import whisper.sim.WhisperSim

/** The chip must never fabricate mel rows: a start with a frame count that is zero, not a multiple of 128,
  * or larger than the number of frames streamed is refused and flagged (reg 0 bit 3); a valid count starts. */
class FrameCheckSpec extends AnyFlatSpec with WhisperSim {
  behavior of "WhisperTop frame-count check"

  it should "refuse bad frame counts and accept a valid one" in {
    simulate(new WhisperTop(WhisperConfig()), subdirectory = Some("framecheck")) { dut =>
      ResetProcedure.module()(dut)
      def reg(a: Int, v: BigInt): Unit = {
        dut.io.regWr.valid.poke(true.B); dut.io.regWr.bits.addr.poke(a.U); dut.io.regWr.bits.data.poke(v.U)
        dut.clock.step(); dut.io.regWr.valid.poke(false.B); dut.clock.step()
      }
      def rd(a: Int): BigInt = { dut.io.regRdAddr.poke(a.U); dut.io.regRdData.peek().litValue }
      def stream(n: Int): Unit = for (_ <- 0 until n) {
        while (!dut.io.mel.ready.peek().litToBoolean) dut.clock.step()
        dut.io.mel.valid.poke(true.B); dut.io.mel.bits.data.poke(0.U); dut.clock.step(); dut.io.mel.valid.poke(false.B); dut.clock.step(3)
      }
      def start(): Unit = { reg(0, 1); dut.clock.step(4) }
      dut.io.tokens.ready.poke(true.B)
      reg(3, 0); reg(1, 50259)
      stream(100)
      assert(rd(3) == 100)
      reg(2, 0); start()                                    // 100 frames, auto count: not a multiple of 128
      assert(!dut.io.busy.peek().litToBoolean && (rd(0) & 8) != 0, "start with 100 auto frames must be refused")
      reg(2, 128); start()                                  // asks for 128 but only 100 streamed
      assert(!dut.io.busy.peek().litToBoolean && (rd(0) & 8) != 0, "start with nFrames > framesIn must be refused")
      stream(28)
      reg(2, 100); start()                                  // explicit count not a multiple of 128
      assert(!dut.io.busy.peek().litToBoolean && (rd(0) & 8) != 0, "nFrames = 100 must be refused")
      reg(2, 0); start()                                    // 128 streamed, auto count -> runs
      assert(dut.io.busy.peek().litToBoolean && (rd(0) & 8) == 0, "128 auto frames must start")
      assert(rd(2) == 128)
    }
  }
}
