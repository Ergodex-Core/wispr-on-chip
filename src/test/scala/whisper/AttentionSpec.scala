package whisper

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.sim.WhisperSim

import java.io.File

class AttentionSpec extends AnyFlatSpec with WhisperSim {
  val root = new File(WhisperConfig.defaultRepoRoot, "out/vectors/attention")
  val allCases = Option(root.listFiles()).getOrElse(Array.empty[File]).map(_.getName).sorted
  require(allCases.nonEmpty, "run `uv run python tests/vectors/gen_attention.py` first")
  def pack(words: Array[Long], from: Int, n: Int): BigInt =
    (0 until n).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(words(from + i)) << (32 * i)))
  def ints(s: String): Array[Long] = s.split("[\\[\\], ]+").filter(_.nonEmpty).map(_.toLong)

  def runCase(name: String): Unit = {
    val dir = new File(root, name)
    val m = Vectors.meta(new File(dir, "meta.json"))
    val nq = m("n_queries").toInt; val nk = m("n_keys").toInt; val keysMax = m("keys_max").toInt
    val q = Vectors.hex(new File(dir, "q.hex")); val k = Vectors.hex(new File(dir, "k.hex")); val v = Vectors.hex(new File(dir, "v.hex"))
    val y = Vectors.hex(new File(dir, "y.hex"))
    val mq = ints(m("mq")); val sq = ints(m("sq"))
    val cfg = WhisperConfig(weightTensors = Some(Seq("enc.0.ln1")))   // tiny weight store (unused)
    val perHead = 2 * (keysMax / 32) * 4
    info(s"$name: queries=$nq keys=$nk keysMax=$keysMax causal=${m("causal")}")
    simulate(new AttentionTestbench(cfg), subdirectory = Some(name)) { dut =>
      val encKeys = KVRegion.encKeys(cfg); val decKeys = KVRegion.decKeys(cfg); val H = cfg.nHead
      val phEnc = KVRegion.wordsPerHeadK(encKeys); val phDec = KVRegion.wordsPerHeadK(decKeys)
      val encSelfK = 0; val encSelfV = encSelfK + H * phEnc; val decSelfK = encSelfV + H * phEnc; val decSelfV = decSelfK + 4 * H * phDec
      val (kb, vb) = if (keysMax == decKeys) (decSelfK, decSelfV) else (encSelfK, encSelfV)
      dut.io.cmd.valid.poke(false.B); dut.io.load.valid.poke(false.B); dut.io.kvLoad.valid.poke(false.B); dut.io.read.en.poke(false.B)
      dut.clock.step(2)
      for (i <- 0 until q.length / 8) {
        dut.io.load.valid.poke(true.B); dut.io.load.bits.addr.poke(i.U); dut.io.load.bits.data.poke(pack(q, 8 * i, 8)); dut.clock.step()
      }
      dut.io.load.valid.poke(false.B)
      def loadKV(base: Int, w: Array[Long]): Unit = {
        for (i <- 0 until w.length / 64) {
          dut.io.kvLoad.valid.poke(true.B); dut.io.kvLoad.bits.addr.poke((base + i).U); dut.io.kvLoad.bits.data.poke(pack(w, 64 * i, 64)); dut.clock.step()
        }
        dut.io.kvLoad.valid.poke(false.B)
      }
      loadKV(kb, k); loadKV(vb, v)
      val c = dut.io.cmd.bits
      c.nQueries.poke(nq.U); c.nKeys.poke(nk.U); c.causal.poke((m("causal") == "true").B); c.qPos0.poke(m("q_pos0").toInt.U)
      c.qBank.poke(0.U); c.qBase.poke(0.U); c.qStride.poke(12.U)
      c.kBase.poke(kb.U); c.vBase.poke(vb.U); c.keysMax.poke(keysMax.U)
      c.outBank.poke(1.U); c.outBase.poke(0.U); c.outStride.poke(24.U)
      for (h <- 0 until 6) { c.mq(h).poke(mq(h).U); c.sq(h).poke(sq(h).U) }
      dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
      var n = 0
      while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 3000000, "timeout") }
      info(s"$name: ${dut.io.cycles.peek().litValue} cycles (engine busy ${dut.io.engCycles.peek().litValue})")
      var bad = 0
      for (i <- 0 until y.length / 8) {
        dut.io.read.addr.poke(i.U); dut.io.read.en.poke(true.B); dut.clock.step()
        val got = dut.io.read.data.peek().litValue; val exp = pack(y, 8 * i, 8)
        if (got != exp) { bad += 1; if (bad <= 6) info(f"word $i (row ${i / 24} head ${(i % 24) / 4}): got ${got.toString(16)} exp ${exp.toString(16)}") }
      }
      assert(bad == 0, s"$name: $bad mismatching words of ${y.length / 8}")
    }
  }
  val selected = sys.env.get("ATT_CASES").map(_.split(",").toSeq).getOrElse(allCases.toSeq)
  for (c <- selected) it should s"be bit-exact on $c" in runCase(c)
}
