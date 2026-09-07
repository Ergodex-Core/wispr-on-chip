package whisper

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.sim.WhisperSim

import java.io.{File, PrintWriter}

/** Runs the first clip of $E2E_DIR/clips.txt, pausing at each pc in $E2E_BREAKS (comma list) to dump
  * activation banks (rtl_banks_pc<N>.txt); tests/e2e/compare_dumps.py compares them with golden dumps. */
class E2EDebugSpec extends AnyFlatSpec with WhisperSim {
  val dir = new File(sys.env.getOrElse("E2E_DIR", WhisperConfig.defaultRepoRoot + "/out/e2e/smoke_var"))
  val clip = scala.io.Source.fromFile(new File(dir, "clips.txt")).getLines().filter(_.nonEmpty).next()
  val maxRows = sys.env.get("E2E_DBG_ROWS").map(_.toInt).getOrElse(64)
  val breaks = sys.env.get("E2E_BREAKS").map(_.split(",").map(_.trim.toInt).toSeq).getOrElse(Seq(4, 5, 6, 7, 8, 13, 15, 16, 17, 22))
  val maxTokens = sys.env.get("E2E_DBG_TOKENS").map(_.toInt).getOrElse(12)

  it should s"dump banks after $clip" in {
    val cfg = WhisperConfig()
    simulate(new WhisperTop(cfg), subdirectory = Some("e2e_debug")) { dut =>
      dut.io.mel.valid.poke(false.B); dut.io.tokens.ready.poke(false.B); dut.io.regWr.valid.poke(false.B)
      dut.io.wsLoad.valid.poke(false.B); dut.io.regRdAddr.poke(0.U); dut.io.dbg.en.poke(false.B)
      dut.clock.step(4)
      def reg(addr: Int, v: BigInt): Unit = { dut.io.regWr.valid.poke(true.B); dut.io.regWr.bits.addr.poke(addr.U); dut.io.regWr.bits.data.poke(v.U); dut.clock.step(); dut.io.regWr.valid.poke(false.B) }
      def status(): Int = { dut.io.regRdAddr.poke(0.U); dut.io.regRdData.peek().litValue.toInt }
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
      def dump(tag: String): Unit = {
        val rows = math.min(maxRows, nFrames / 2)
        val banks = Seq((1, 12), (2, 24), (3, 12), (4, 24), (5, 96), (6, 48), (7, 12))
        val out = new PrintWriter(new File(cd, s"rtl_banks_$tag.txt"))
        for ((b, stride) <- banks; r <- 0 until rows; w <- 0 until stride) {
          dut.io.dbg.en.poke(true.B); dut.io.dbg.bank.poke(b.U); dut.io.dbg.addr.poke((r * stride + w).U); dut.clock.step()
          out.println(s"$b $r $w ${dut.io.dbg.data.peek().litValue.toString(16)}")
        }
        for (b <- Seq(1, 6, 7); r <- 0 until rows) {
          dut.io.dbg.en.poke(true.B); dut.io.dbg.bank.poke(b.U); dut.io.dbg.addr.poke(r.U); dut.clock.step()
          out.println(s"rf $b $r ${dut.io.dbg.rowfac.peek().litValue}")
        }
        dut.io.dbg.en.poke(false.B)
        out.close()
      }
      reg(8, breaks.headOption.getOrElse(0x3ff))
      reg(0, 1)
      var cycles = 0L
      var bi = 0
      var running = true
      val t0 = System.nanoTime()
      while (running) {
        dut.clock.step(256); cycles += 256
        val st = status()
        if ((st & 4) != 0) {              // paused at breakpoint
          val pc = breaks(bi)
          info(s"paused at pc $pc after $cycles cycles"); dump(s"pc$pc")
          bi += 1
          reg(8, if (bi < breaks.size) breaks(bi) else 0x3ff)
          reg(9, 1)
        }
        if ((st & 2) != 0) running = false
        if (cycles > 300000000L) running = false
        dut.io.regRdAddr.poke(4.U)
        if (dut.io.regRdData.peek().litValue.toInt >= maxTokens) running = false
      }
      val secs = (System.nanoTime() - t0) / 1e9
      // drain tokens
      val toks = scala.collection.mutable.ArrayBuffer[Int]()
      dut.io.tokens.ready.poke(true.B)
      for (_ <- 0 until 300) { if (dut.io.tokens.valid.peek().litToBoolean) toks += dut.io.tokens.bits.id.peek().litValue.toInt; dut.clock.step() }
      info(f"$clip: tokens ${toks.mkString(" ")} ($cycles cycles, $secs%.0f s, ${cycles / secs}%.0f cycles/s)")
      val pw = new PrintWriter(new File(cd, "rtl_tokens.txt")); pw.println(toks.mkString(" ")); pw.close()
      dump("end")
    }
  }
}
