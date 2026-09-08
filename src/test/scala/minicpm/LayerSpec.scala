package minicpm

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import minicpm.generated.{MicrocodeLayer, MicrocodeLayerWide}
import minicpm.sim.MiniCPMSim

import java.io.File

/** The whole chip on Verilator, running the generated micro-program end to end: prompt tokens in, two real
  * transformer layers of MiniCPM5-2B over a chunked prefill and two decode steps, tokens out. Everything the
  * unit specs cannot reach is under test here — the sequencer's control flow and chunk loop, the runtime
  * substitution of rows / row offsets / key counts / positions, the token buffer feedback, engine
  * arbitration between the sequencer and the attention unit, bank read/write muxing and the KV cache in
  * situ. Both the residual bank and the emitted token ids must equal the golden model's
  * (`tests/vectors/gen_layer.py`) bit for bit. */
class LayerSpec extends AnyFlatSpec with MiniCPMSim {
  val pollCycles = sys.env.get("LAYER_POLL").map(_.toInt).getOrElse(4096)
  val maxCycles = sys.env.get("LAYER_MAX_CYCLES").map(_.toLong).getOrElse(60000000L)
  /** (program, what it adds). Both run the same generated code; only the emission parameters differ. */
  val configs: Seq[(MicroProgram, String)] = Seq(
    MicrocodeLayer -> "two layers, chunked prefill and two decode steps (layer indexing, token feedback)",
    MicrocodeLayerWide -> "full-scale addressing (maxCtx 2048), 32-row chunks, attention over two key tiles",
  )

  def runProgram(mp: MicroProgram, what: String): Unit = {
    val dir = new File(Vectors.dir("layer"), s"L${mp.nLayers}_ctx${mp.maxCtx}")
    require(dir.exists(), s"run `uv run python tests/vectors/gen_layer.py` first ($dir)")
    val m = Vectors.meta(new File(dir, "meta.json"))
    val prompt = Vectors.ints(new File(dir, "prompt.txt"))
    val expTokens = Vectors.ints(new File(dir, "tokens_exp.txt"))
    val xExp = Vectors.hex(new File(dir, "x_exp.hex"))
    val maxNew = m("max_new").toInt
    val rows = m("rows").toInt
    val layers = (0 until mp.nLayers).map(l => s"L$l")
    val cfg = MiniCPMConfig.forProgram(mp, Some(layers ++ Seq("norm_f", "embed.00", "embed.resmult", "rope", "lm.00")))
    info(s"program ${mp.program.length} instructions, ${mp.nLayers} layers, maxCtx ${mp.maxCtx}, " +
      s"chunk ${mp.chunkRows}, ${mp.vocabTiles} vocabulary tiles; ${prompt.length} prompt tokens, maxNew $maxNew")
    val tElab = System.nanoTime()
    simulate(new MiniCPMTop(cfg), subdirectory = Some(s"layer_L${mp.nLayers}_ctx${mp.maxCtx}")) { dut =>
      info(f"elaboration + Verilator build ${(System.nanoTime() - tElab) / 1e9}%.0f s")
      dut.io.regWr.valid.poke(false.B)
      dut.io.wsLoad.valid.poke(false.B)
      dut.io.prompt.valid.poke(false.B)
      dut.io.tokens.ready.poke(false.B)
      dut.io.dbg.en.poke(false.B)
      dut.clock.step(4)
      def wr(addr: Int, data: Long): Unit = {
        dut.io.regWr.valid.poke(true.B); dut.io.regWr.bits.addr.poke(addr.U); dut.io.regWr.bits.data.poke(data.U)
        dut.clock.step(); dut.io.regWr.valid.poke(false.B); dut.clock.step()
      }
      def rd(addr: Int): BigInt = { dut.io.regRdAddr.poke(addr.U); dut.clock.step(); dut.io.regRdData.peek().litValue }
      wr(3, 0)                                  // reset the token buffer
      for (t <- prompt) {                       // stream the prompt
        dut.io.prompt.valid.poke(true.B); dut.io.prompt.bits.poke(t.U)
        while (!dut.io.prompt.ready.peek().litToBoolean) dut.clock.step()
        dut.clock.step()
      }
      dut.io.prompt.valid.poke(false.B)
      assert(rd(2) == prompt.length, s"token buffer holds ${rd(2)} of ${prompt.length} prompt tokens")
      wr(1, maxNew)                             // generation limit
      wr(0, 1)                                  // start
      var cycles = 0L
      val t0 = System.nanoTime()
      while (!dut.io.done.peek().litToBoolean && cycles < maxCycles) {
        dut.clock.step(pollCycles); cycles += pollCycles
      }
      val secs = (System.nanoTime() - t0) / 1e9
      assert(cycles < maxCycles, s"the chip did not finish in $maxCycles cycles (pc ${rd(5)}, pos ${rd(6)})")
      info(f"$cycles cycles in $secs%.0f s (${cycles / secs / 1000}%.1f k cycles/s); engine busy ${rd(7)} cycles " +
        f"(${100.0 * rd(7).toDouble / cycles}%.1f %%), requant saturations ${rd(10)}")
      // ---- tokens
      val got = scala.collection.mutable.ArrayBuffer[Long]()
      dut.io.tokens.ready.poke(true.B)
      var guard = 0
      while (dut.io.tokens.valid.peek().litToBoolean && guard < 1000) {
        got += dut.io.tokens.bits.id.peek().litValue.toLong; dut.clock.step(); guard += 1
      }
      dut.io.tokens.ready.poke(false.B)
      info(s"tokens: ${got.mkString(", ")} (expected ${expTokens.mkString(", ")})")
      assert(got.toSeq == expTokens.toSeq, s"token stream ${got.mkString(",")} != ${expTokens.mkString(",")}")
      assert(rd(4) == expTokens.length, s"token count ${rd(4)} != ${expTokens.length}")
      // ---- residual bank (int32 rows of 2048 = 256 words each), all processed positions
      val stride = cfg.dModel / 8
      var bad = 0
      dut.io.dbg.bank.poke(0.U)
      dut.io.dbg.en.poke(true.B)
      for (i <- 0 until rows * stride) {
        dut.io.dbg.addr.poke(i.U); dut.clock.step()
        val gotW = dut.io.dbg.data.peek().litValue
        val exp = Vectors.pack(xExp, 8 * i, 8)
        if (gotW != exp) {
          bad += 1
          if (bad <= 6) info(f"X word $i (row ${i / stride}, lane group ${i % stride}): got ${gotW.toString(16)} exp ${exp.toString(16)}")
        }
      }
      dut.io.dbg.en.poke(false.B)
      assert(bad == 0, s"$bad of ${rows * stride} residual words differ")
      info(s"residual bank matches the golden model on all $rows rows")
    }
  }

  for ((mp, what) <- configs)
    it should s"match the golden model with ${mp.nLayers} layer(s), maxCtx ${mp.maxCtx}, chunk ${mp.chunkRows} — $what" in runProgram(mp, what)
}
