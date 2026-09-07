package whisper

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.sim.WhisperSim

import java.io.{File, PrintWriter}

/** Runs one clip (first of $E2E_DIR/clips.txt) and dumps activation banks after completion so the
  * Python side can compare intermediate tensors with the golden dumps (tests/e2e/compare_dumps.py). */
class E2EDebugSpec extends AnyFlatSpec with WhisperSim {
  val dir = new File(sys.env.getOrElse("E2E_DIR", WhisperConfig.defaultRepoRoot + "/out/e2e/smoke_var"))
  val clip = scala.io.Source.fromFile(new File(dir, "clips.txt")).getLines().filter(_.nonEmpty).next()
  val maxRows = sys.env.get("E2E_DBG_ROWS").map(_.toInt).getOrElse(64)
  val stopAfterEncoder = sys.env.get("E2E_STOP_PC").map(_.toInt)   // stop polling when pc reaches this (unused for now)

  it should s"dump banks after $clip" in {
    val cfg = WhisperConfig()
    simulate(new WhisperTop(cfg), subdirectory = Some("e2e_debug")) { dut =>
      dut.io.mel.valid.poke(false.B); dut.io.tokens.ready.poke(true.B); dut.io.regWr.valid.poke(false.B)
      dut.io.wsLoad.valid.poke(false.B); dut.io.regRdAddr.poke(0.U); dut.io.dbg.en.poke(false.B)
      dut.clock.step(4)
      def reg(addr: Int, v: BigInt): Unit = { dut.io.regWr.valid.poke(true.B); dut.io.regWr.bits.addr.poke(addr.U); dut.io.regWr.bits.data.poke(v.U); dut.clock.step(); dut.io.regWr.valid.poke(false.B) }
      val cd = new File(dir, clip)
      val meta = Vectors.meta(new File(cd, "meta.json"))
      val nFrames = meta("n_frames").toInt
      val mel = Vectors.hex(new File(cd, "mel.hex"))
      reg(3, 0); reg(1, meta("lang_token").toInt); reg(2, nFrames)
      for (f <- 0 until nFrames) {
        val v = (0 until 20).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(mel(20 * f + i)) << (32 * i)))
        dut.io.mel.valid.poke(true.B); dut.io.mel.bits.data.poke(v); dut.io.mel.bits.last.poke((f == nFrames - 1).B)
        while (!dut.io.mel.ready.peek().litToBoolean) dut.clock.step()
        dut.clock.step(); dut.io.mel.valid.poke(false.B); dut.clock.step(3)
      }
      reg(0, 1)
      val toks = scala.collection.mutable.ArrayBuffer[Int]()
      var cycles = 0L; var running = true
      val maxTokens = sys.env.get("E2E_DBG_TOKENS").map(_.toInt).getOrElse(400)
      while (running) {
        for (_ <- 0 until 64) { if (dut.io.tokens.valid.peek().litToBoolean) toks += dut.io.tokens.bits.id.peek().litValue.toInt; dut.clock.step() }
        cycles += 64
        if (dut.io.done.peek().litToBoolean) running = false
        if (toks.size >= maxTokens) running = false
        if (cycles > 300000000L) running = false
      }
      info(s"$clip: tokens ${toks.mkString(" ")} ($cycles cycles)")
      val pw = new PrintWriter(new File(cd, "rtl_tokens.txt")); pw.println(toks.mkString(" ")); pw.close()
      // dump banks: rows of 256-bit words
      val nCtx = nFrames / 2
      val rows = math.min(maxRows, nCtx)
      val banks = Seq((1, 12), (2, 24), (3, 12), (4, 24), (5, 96), (6, 48), (7, 12))
      val out = new PrintWriter(new File(cd, "rtl_banks.txt"))
      for ((b, stride) <- banks; r <- 0 until rows; w <- 0 until stride) {
        dut.io.dbg.en.poke(true.B); dut.io.dbg.bank.poke(b.U); dut.io.dbg.addr.poke((r * stride + w).U); dut.clock.step()
        out.println(s"$b $r $w ${dut.io.dbg.data.peek().litValue.toString(16)}")
      }
      for (b <- Seq(1, 6, 7); r <- 0 until rows) {
        dut.io.dbg.en.poke(true.B); dut.io.dbg.bank.poke(b.U); dut.io.dbg.addr.poke(r.U); dut.clock.step()
        out.println(s"rf $b $r ${dut.io.dbg.rowfac.peek().litValue}")
      }
      out.close()
    }
  }
}
