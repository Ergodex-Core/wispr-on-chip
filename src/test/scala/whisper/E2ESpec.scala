package whisper

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.sim.WhisperSim

import java.io.File
import scala.io.Source

/** Runs every clip listed in $E2E_DIR/clips.txt through WhisperTop in one Verilator build/simulation.
  * Writes rtl_tokens.txt and rtl_stats.json per clip; comparison happens in tests/e2e/run_e2e.py. */
class E2ESpec extends AnyFlatSpec with WhisperSim {
  val dir = new File(sys.env.getOrElse("E2E_DIR", WhisperConfig.defaultRepoRoot + "/out/e2e/smoke_var"))
  val clips = Source.fromFile(new File(dir, "clips.txt")).getLines().filter(_.nonEmpty).toSeq
  val pollStep = sys.env.get("E2E_POLL").map(_.toInt).getOrElse(4096)  // FastSim: cycles between host polls

  it should s"transcribe ${clips.size} clips from $dir" in {
    val cfg = WhisperConfig()
    simulate(new WhisperTop(cfg), subdirectory = Some("e2e")) { dut =>
      dut.io.mel.valid.poke(false.B); dut.io.tokens.ready.poke(true.B); dut.io.regWr.valid.poke(false.B)
      dut.io.wsLoad.valid.poke(false.B); dut.io.regRdAddr.poke(0.U)
      dut.clock.step(4)
      def reg(addr: Int, v: BigInt): Unit = {
        dut.io.regWr.valid.poke(true.B); dut.io.regWr.bits.addr.poke(addr.U); dut.io.regWr.bits.data.poke(v.U); dut.clock.step(); dut.io.regWr.valid.poke(false.B)
      }
      for (clip <- clips) {
        val cd = new File(dir, clip)
        val meta = Vectors.meta(new File(cd, "meta.json"))
        val nFrames = meta("n_frames").toInt
        val mel = Vectors.hex(new File(cd, "mel.hex"))
        reg(3, 0)                                        // reset frame counter
        reg(1, meta("lang_token").toInt)
        reg(2, nFrames)
        // stream frames: 20 words = 80 bytes per frame
        for (f <- 0 until nFrames) {
          val v = (0 until 20).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(mel(20 * f + i)) << (32 * i)))
          dut.io.mel.valid.poke(true.B); dut.io.mel.bits.data.poke(v); dut.io.mel.bits.last.poke((f == nFrames - 1).B)
          while (!dut.io.mel.ready.peek().litToBoolean) dut.clock.step()
          dut.clock.step()
          dut.io.mel.valid.poke(false.B)
          dut.clock.step(3)
        }
        val t0 = System.nanoTime()
        dut.io.tokens.ready.poke(false.B)                // tokens accumulate in the 256-deep queue (FastSim: no per-cycle polling)
        reg(0, 1)                                        // start
        var cycles = 0L
        var running = true
        while (running) {
          dut.clock.step(pollStep); cycles += pollStep
          if (dut.io.done.peek().litToBoolean) running = false
          if (cycles > 400000000L) { info(s"$clip: timeout"); running = false }
        }
        val toks = scala.collection.mutable.ArrayBuffer[Int]()
        dut.io.tokens.ready.poke(true.B)
        for (_ <- 0 until 300) { if (dut.io.tokens.valid.peek().litToBoolean) toks += dut.io.tokens.bits.id.peek().litValue.toInt; dut.clock.step() }
        val secs = (System.nanoTime() - t0) / 1e9
        dut.io.regRdAddr.poke(7.U)
        info(f"$clip: ${toks.size} tokens, $cycles cycles, $secs%.1f s (${cycles / secs}%.0f cycles/s)")
        val pw = new java.io.PrintWriter(new File(cd, "rtl_tokens.txt")); pw.println(toks.mkString(" ")); pw.close()
        val ps = new java.io.PrintWriter(new File(cd, "rtl_stats.json")); ps.println(s"""{"cycles": $cycles, "seconds": $secs}"""); ps.close()
      }
    }
  }
}
