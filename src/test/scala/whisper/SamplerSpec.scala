package whisper

import chisel3._
import chisel3.simulator.stimulus.ResetProcedure
import org.scalatest.flatspec.AnyFlatSpec
import whisper.generated.Microcode
import whisper.sim.WhisperSim

/** Sampler alone: feeds a full 51872-wide logits row as 1621 beats of 32 int32 and checks the pipelined
  * lowest-index argmax against a Scala reference, with the suppress / blank masks applied. */
class SamplerSpec extends AnyFlatSpec with WhisperSim {
  behavior of "Sampler"

  it should "pick the lowest-index maximum with suppression, and pulse done" in {
    val cfg = WhisperConfig()
    val NT = cfg.nVocabPad / 32
    val supp = (Microcode.suppressTokens ++ (51865 until cfg.nVocabPad)).toSet
    val blank = Microcode.suppressBlank.toSet
    val rnd = new scala.util.Random(7)
    simulate(new Sampler(cfg), subdirectory = Some("sampler")) { dut =>
      ResetProcedure.module()(dut)
      dut.io.in.valid.poke(false.B)
      def row(first: Boolean, hot: Seq[(Int, Int)]): Int = {
        val logits = Array.fill(cfg.nVocabPad)(rnd.nextInt(2000) - 1000)
        for ((i, v) <- hot) logits(i) = v
        val masked = logits.zipWithIndex.map { case (v, i) => if (supp(i) || (first && blank(i))) Int.MinValue else v }
        val expect = masked.indices.maxBy(i => (masked(i).toLong, -i.toLong))   // highest value, lowest index on ties
        dut.io.first.poke(first.B)
        dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
        for (t <- 0 until NT) {
          dut.io.in.valid.poke(true.B); dut.io.in.bits.nTile.poke(t.U); dut.io.in.bits.last.poke((t == NT - 1).B)
          for (i <- 0 until 32) dut.io.in.bits.data(i).poke(logits(t * 32 + i).S)
          dut.clock.step()
          if (t % 5 == 4) { dut.io.in.valid.poke(false.B); dut.clock.step() }   // bubbles in the stream
        }
        dut.io.in.valid.poke(false.B)
        var seen = false; var n = 0
        while (!seen && n < 8) { seen = dut.io.done.peek().litToBoolean; dut.clock.step(); n += 1 }
        assert(seen, "done must pulse within a few cycles of the last beat")
        val got = dut.io.token.peek().litValue.toInt
        assert(got == expect, s"token $got, expected $expect")
        got
      }
      row(first = false, Seq((1000, 5000), (1001, 5000)))                       // tie -> lowest index 1000
      assert(row(first = true, Seq((220, 9000), (50257, 9000), (12345, 8000))) == 12345)   // blank + eot suppressed on step 0
      assert(row(first = false, Seq((50257, 9000), (12345, 8000))) == 50257)          // eot allowed later
      row(first = false, Seq((51870, 9999), (7, 3000)))                            // padded column suppressed
      row(first = false, Seq((31, 4000), (32, 4000)))                              // tie across a beat boundary
      for (_ <- 0 until 3) row(first = false, Seq())
    }
  }
}
