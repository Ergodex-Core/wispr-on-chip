package minicpm

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import minicpm.generated.WeightTables
import minicpm.sim.MiniCPMSim

import java.io.File

/** Every vector-unit op on golden activations of layers 0 / 41 (RMSNorm on the int32 residual, dynamic
  * quantisation, the SiLU gate with 32-bit products, residual adds, embedding lookups across table slices,
  * RoPE at positions 0.. and 1000.., and RoPE-K as KV-cache beats). Bit-exact, including row factors. */
class VectorUnitSpec extends AnyFlatSpec with MiniCPMSim {
  val allCases = Vectors.cases("vector")
  require(allCases.nonEmpty, "run `uv run python tests/vectors/gen_all.py` first")
  val baseTensors = Seq("L0.norm1.g", "L41.norm2.g", "norm_f.g", "embed.resmult", "rope")
  val bitsCode = Map(8 -> VecBits.I8, 16 -> VecBits.I16, 32 -> VecBits.I32)
  def wordsPerRow(cols: Int, bits: Int): Int = cols / (256 / bits)

  def runCase(name: String): Unit = {
    val dir = new File(Vectors.dir("vector"), name)
    val m = Vectors.meta(new File(dir, "meta.json"))
    val rows = m("rows").toInt; val cols = m("cols").toInt
    val aBits = m("a_bits").toInt; val yBits = m("y_bits").toInt; val bBits = m("b_bits").toInt
    val aWords = Vectors.hex(new File(dir, "a.hex")); val yWords = Vectors.hex(new File(dir, "y.hex"))
    val bFile = new File(dir, "b.hex")
    val bWords = if (bFile.exists()) Vectors.hex(bFile) else Array.empty[Long]
    val rfExp = Vectors.ints(new File(dir, "rowfac_exp.txt"))
    val toks = Vectors.ints(new File(dir, "toks.txt"))
    val slices = m.get("slices").map(Vectors.list(_).map(i => f"embed.$i%02d").toSeq).getOrElse(Seq("embed.00"))
    val cfg = MiniCPMConfig(weightBackend = RomInit, weightTensors = Some(baseTensors ++ slices))
    val aStride = wordsPerRow(cols, aBits); val yStride = wordsPerRow(cols, yBits)
    val bStride = if (bBits > 0) wordsPerRow(cols, bBits) else 0
    val op = m("op")
    info(s"$name: op=$op rows=$rows cols=$cols a=$aBits b=$bBits y=$yBits")
    simulate(new VectorTestbench(cfg), subdirectory = Some(name)) { dut =>
      dut.io.cmd.valid.poke(false.B); dut.io.load.valid.poke(false.B); dut.io.tokLoad.valid.poke(false.B)
      dut.io.read.en.poke(false.B); dut.io.rowfacRead.en.poke(false.B)
      dut.clock.step(2)
      def loadBank(bank: Int, w: Array[Long]): Unit = {
        for (i <- 0 until w.length / 8) {
          dut.io.load.valid.poke(true.B); dut.io.load.bits.bank.poke(bank.U); dut.io.load.bits.addr.poke(i.U)
          dut.io.load.bits.data.poke(Vectors.pack(w, 8 * i, 8)); dut.clock.step()
        }
        dut.io.load.valid.poke(false.B)
      }
      if (op != "EMBED") loadBank(0, aWords)
      if (bWords.nonEmpty) loadBank(1, bWords)
      for (i <- toks.indices) {
        dut.io.tokLoad.valid.poke(true.B); dut.io.tokLoad.bits.addr.poke(i.U); dut.io.tokLoad.bits.data.poke(toks(i).U); dut.clock.step()
      }
      dut.io.tokLoad.valid.poke(false.B)
      val c = dut.io.cmd.bits
      c.elements.values.foreach { case u: UInt => u.poke(0.U); case b: Bool => b.poke(false.B) }
      val opId = op match { case "RMSNORM" => VecOp.RMSNORM; case "DYNQ" => VecOp.DYNQ; case "ADD" => VecOp.ADD; case "EMBED" => VecOp.EMBED; case "ROPE" => VecOp.ROPE; case "SILUMUL" => VecOp.SILUMUL }
      c.op.poke(opId.U); c.rows.poke(rows.U); c.cols.poke(cols.U)
      c.inBank.poke(0.U); c.inBase.poke(0.U); c.inStride.poke(aStride.U); c.inBits.poke(bitsCode(aBits).U); c.inLocal.poke(true.B)
      c.outBank.poke(2.U); c.outBase.poke(0.U); c.outStride.poke(yStride.U); c.outLocal.poke(true.B)
      c.rowfacBank.poke(0.U); c.rowBase.poke(0.U); c.rfLocal.poke(true.B); c.rowOff.poke(0.U)
      op match {
        case "RMSNORM" =>
          c.gBase.poke(WeightTables.byName(m("norm") + ".g").base.U)
          val eps = BigInt(m("eps_q")); c.epsLo.poke((eps & 0xffffffffL).U); c.epsHi.poke((eps >> 32).U)
        case "DYNQ" =>
        case "SILUMUL" =>
          c.bBank.poke(1.U); c.bBase.poke(0.U); c.bStride.poke(bStride.U); c.bBits.poke(bitsCode(bBits).U); c.bLocal.poke(true.B)
          c.mSig.poke(m("m_sig").toLong.U); c.sSig.poke(m("s_sig").toInt.U)
        case "ADD" =>
          c.bBank.poke(1.U); c.bBase.poke(0.U); c.bStride.poke(bStride.U); c.bBits.poke(bitsCode(bBits).U); c.bLocal.poke(true.B)
          c.ma.poke(m("ma").toLong.U); c.mb.poke(m("mb").toLong.U)
        case "EMBED" =>
          c.tokFromMem.poke(true.B); c.maTableBase.poke(WeightTables.byName("embed.resmult").base.U)
          c.embBase.poke(WeightTables.byName("embed.00").base.U); c.inBits.poke(VecBits.I8.U)
        case "ROPE" =>
          c.reqMult.poke(m("m_r").toLong.U); c.reqShift.poke(m("s_r").toInt.U); c.ropeBase.poke(WeightTables.byName("rope").base.U)
          c.posBase.poke(m("pos_base").toInt.U)
          if (m("to_kv") == "true") c.outSink.poke(1.U)
      }
      dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
      var n = 0
      while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 2000000, "timeout") }
      dut.clock.step(4)
      info(s"$name: ${dut.io.cycles.peek().litValue} cycles for $rows rows (${dut.io.cycles.peek().litValue / rows} per row)")
      if (op == "ROPE" && m("to_kv") == "true") info(s"$name: ${dut.io.kvBeats.peek().litValue} KV beats")
      var bad = 0
      for (i <- 0 until yWords.length / 8) {
        dut.io.read.addr.poke(i.U); dut.io.read.en.poke(true.B); dut.clock.step()
        val got = dut.io.read.data.peek().litValue; val exp = Vectors.pack(yWords, 8 * i, 8)
        if (got != exp) { bad += 1; if (bad <= 4) info(f"word $i (row ${i / yStride}): got ${got.toString(16)} exp ${exp.toString(16)}") }
      }
      for (r <- rfExp.indices) {
        dut.io.rowfacRead.addr.poke(r.U); dut.io.rowfacRead.en.poke(true.B); dut.clock.step()
        val got = dut.io.rowfacRead.data.peek().litValue.toLong
        if (got != rfExp(r)) { bad += 1; if (bad <= 8) info(s"rowfac $r: got $got exp ${rfExp(r)}") }
      }
      assert(bad == 0, s"$name: $bad mismatches")
    }
  }
  val selected = sys.env.get("VEC_CASES").map(_.split(",").toSeq).getOrElse(allCases)
  for (c <- selected) it should s"be bit-exact on $c" in runCase(c)
}
