package minicpm

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import minicpm.sim.MiniCPMSim

import java.io.File

/** Attention on golden Q/K/V of layers 0 / 20 / 41: a 128-row causal prefill, a chunk with a non-zero
  * query base, a ragged non-causal case, and single-query decode steps at positions 40 and 127.
  * All 16 query heads over 2 KV heads, key tiles of 64, bit-exact int16 outputs. */
class AttentionSpec extends AnyFlatSpec with MiniCPMSim {
  val allCases = Vectors.cases("attention")
  require(allCases.nonEmpty, "run `uv run python tests/vectors/gen_all.py` first")

  def runCase(name: String): Unit = {
    val dir = new File(Vectors.dir("attention"), name)
    val m = Vectors.meta(new File(dir, "meta.json"))
    val nq = m("n_queries").toInt; val nk = m("n_keys").toInt; val keysMax = m("keys_max").toInt
    val q = Vectors.hex(new File(dir, "q.hex")); val k = Vectors.hex(new File(dir, "k.hex")); val v = Vectors.hex(new File(dir, "v.hex"))
    val y = Vectors.hex(new File(dir, "y.hex"))
    val mq = Vectors.list(m("mq")); val sq = Vectors.list(m("sq"))
    val cfg = MiniCPMConfig(weightTensors = Some(Seq("L0.norm1.g")), maxCtx = keysMax, kvLayers = 1)
    val kb = KVRegion.kBase(cfg, 0); val vb = KVRegion.vBase(cfg, 0)
    info(s"$name: queries=$nq keys=$nk keysMax=$keysMax causal=${m("causal")} qPos0=${m("q_pos0")}")
    simulate(new AttentionTestbench(cfg), subdirectory = Some(name)) { dut =>
      dut.io.cmd.valid.poke(false.B); dut.io.load.valid.poke(false.B); dut.io.kvLoad.valid.poke(false.B); dut.io.read.en.poke(false.B)
      dut.io.kvLoadMask.poke(((BigInt(1) << 256) - 1).U)
      dut.clock.step(2)
      for (i <- 0 until q.length / 8) {
        dut.io.load.valid.poke(true.B); dut.io.load.bits.addr.poke(i.U); dut.io.load.bits.data.poke(Vectors.pack(q, 8 * i, 8)); dut.clock.step()
      }
      dut.io.load.valid.poke(false.B)
      def loadKV(base: Int, w: Array[Long]): Unit = {
        for (i <- 0 until w.length / 64) {
          dut.io.kvLoad.valid.poke(true.B); dut.io.kvLoad.bits.addr.poke((base + i).U); dut.io.kvLoad.bits.data.poke(Vectors.pack(w, 64 * i, 64)); dut.clock.step()
        }
        dut.io.kvLoad.valid.poke(false.B)
      }
      loadKV(kb, k); loadKV(vb, v)
      val c = dut.io.cmd.bits
      c.nQueries.poke(nq.U); c.nKeys.poke(nk.U); c.causal.poke((m("causal") == "true").B); c.qPos0.poke(m("q_pos0").toInt.U)
      c.qBank.poke(0.U); c.qBase.poke(0.U); c.qStride.poke((cfg.dModel / 32).U)
      c.kBase.poke(kb.U); c.vBase.poke(vb.U); c.keysMax.poke(keysMax.U)
      c.outBank.poke(1.U); c.outBase.poke(0.U); c.outStride.poke((cfg.dModel / 16).U)
      for (h <- 0 until 16) { c.mq(h).poke(mq(h).U); c.sq(h).poke(sq(h).U) }
      dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
      var n = 0
      while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 6000000, "timeout") }
      dut.clock.step(4)
      info(s"$name: ${dut.io.cycles.peek().litValue} cycles (engine busy ${dut.io.engCycles.peek().litValue})")
      var bad = 0
      val stride = cfg.dModel / 16
      for (i <- 0 until y.length / 8) {
        dut.io.read.addr.poke(i.U); dut.io.read.en.poke(true.B); dut.clock.step()
        val got = dut.io.read.data.peek().litValue; val exp = Vectors.pack(y, 8 * i, 8)
        if (got != exp) { bad += 1; if (bad <= 6) info(f"word $i (row ${i / stride} head ${(i % stride) / 8}): got ${got.toString(16)} exp ${exp.toString(16)}") }
      }
      assert(bad == 0, s"$name: $bad mismatching words of ${y.length / 8}")
    }
  }
  val selected = sys.env.get("ATT_CASES").map(_.split(",").toSeq).getOrElse(allCases)
  for (c <- selected) it should s"be bit-exact on $c" in runCase(c)
}
