package whisper

import chisel3._
import chisel3.simulator.stimulus.ResetProcedure
import org.scalatest.flatspec.AnyFlatSpec
import whisper.sim.WhisperSim

/** The chip must never fabricate mel rows: a start with a frame count that is zero, odd, above maxFrames,
  * or larger than the number of frames streamed is refused and flagged (reg 0 bit 3); a valid count starts.
  * The count is deliberately not required to be a multiple of 128 -- the accuracy configuration is 3000
  * frames, which is not one, and requiring it silently stopped the chip from ever starting (decision #15). */
class FrameCheckSpec extends AnyFlatSpec with WhisperSim {
  behavior of "WhisperTop frame-count check"

  it should "refuse zero, odd, oversized and unstreamed frame counts, and accept a valid one" in {
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
      stream(101)
      assert(rd(3) == 101)
      reg(2, 0); start()                                    // auto count 101: odd, n_ctx would truncate
      assert(!dut.io.busy.peek().litToBoolean && (rd(0) & 8) != 0, "odd auto frame count must be refused")
      reg(2, 128); start()                                  // asks for 128 but only 101 streamed
      assert(!dut.io.busy.peek().litToBoolean && (rd(0) & 8) != 0, "nFrames > framesIn must be refused")
      reg(2, 4000); start()                                 // beyond maxFrames
      assert(!dut.io.busy.peek().litToBoolean && (rd(0) & 8) != 0, "nFrames > maxFrames must be refused")
      reg(2, 100); start()                                  // <= framesIn, even -> runs
      assert(dut.io.busy.peek().litToBoolean && (rd(0) & 8) == 0, "100 frames must start")
      assert(rd(2) == 100)
    }
  }
}
