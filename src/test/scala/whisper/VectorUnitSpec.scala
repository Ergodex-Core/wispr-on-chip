package whisper

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.generated.WeightTables
import whisper.sim.WhisperSim

import java.io.File
import scala.io.Source

class VectorUnitSpec extends AnyFlatSpec with WhisperSim {
  val root = new File(WhisperConfig.defaultRepoRoot, "out/vectors/vector")
  val allCases = Option(root.listFiles()).getOrElse(Array.empty[File]).map(_.getName).sorted
  require(allCases.nonEmpty, "run `uv run python tests/vectors/gen_vector.py` first")
  def pack(words: Array[Long], from: Int, n: Int): BigInt =
    (0 until n).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(words(from + i)) << (32 * i)))

  def runCase(name: String): Unit = {
    val dir = new File(root, name)
    val m = Vectors.meta(new File(dir, "meta.json"))
    val rows = m("rows").toInt; val cols = m("cols").toInt
    val a16 = m("a_bits16") == "true"; val y16 = m("y_bits16") == "true"; val b16 = m.getOrElse("b_bits16", "false") == "true"
    val aWords = Vectors.hex(new File(dir, "a.hex")); val yWords = Vectors.hex(new File(dir, "y.hex"))
    val bFile = new File(dir, "b.hex")
    val bWords = if (bFile.exists()) Vectors.hex(bFile) else Array.empty[Long]
    val rfFile = new File(dir, "rowfac_exp.txt")
    val rfExp = if (rfFile.exists()) Source.fromFile(rfFile).getLines().filter(_.nonEmpty).map(_.toInt).toArray else Array.empty[Int]
    val toks = m.get("toks").map(_.split("[\\[\\], ]+").filter(_.nonEmpty).map(_.toInt)).getOrElse(Array.empty[Int])
    val tensors = Seq("enc.0.ln1", "enc.0.gelu", "enc.pos", "dec.pos", "dec.emb.resmult", "dec.lm.w")
    val cfg = WhisperConfig(weightBackend = RomInit, weightTensors = Some(tensors))
    val aStride = if (a16) cols / 16 else cols / 32
    val bStride = if (b16) cols / 16 else cols / 32
    val yStride = if (y16) cols / 16 else cols / 32
    info(s"$name: op=${m("op")} rows=$rows cols=$cols")
    simulate(new VectorTestbench(cfg), subdirectory = Some(name)) { dut =>
      dut.io.cmd.valid.poke(false.B); dut.io.load.valid.poke(false.B); dut.io.read.en.poke(false.B); dut.io.rowfacRead.en.poke(false.B)
      dut.clock.step(2)
      def loadBank(bank: Int, w: Array[Long]): Unit = {
        for (i <- 0 until w.length / 8) {
          dut.io.load.valid.poke(true.B); dut.io.load.bits.bank.poke(bank.U); dut.io.load.bits.addr.poke(i.U)
          dut.io.load.bits.data.poke(pack(w, 8 * i, 8)); dut.clock.step()
        }
        dut.io.load.valid.poke(false.B)
      }
      loadBank(0, aWords)
      if (bWords.nonEmpty) loadBank(1, bWords)
      val c = dut.io.cmd.bits
      c.elements.values.foreach { case u: UInt => u.poke(0.U); case b: Bool => b.poke(false.B); case v: Vec[_] => v.foreach { case u: UInt => u.poke(0.U) } }
      val op = m("op") match { case "LN" => VecOp.LN; case "DYNQ" => VecOp.DYNQ; case "ADD" => VecOp.ADD; case "EMBED" => VecOp.EMBED }
      def issue(rowsN: Int, rowOff: Int): Unit = {
        c.op.poke(op.U); c.rows.poke(rowsN.U); c.cols.poke(cols.U)
        c.inBank.poke(0.U); c.inBase.poke((rowOff * aStride).U); c.inStride.poke(aStride.U); c.inBits16.poke(a16.B)
        c.outBank.poke(2.U); c.outBase.poke((rowOff * yStride).U); c.outStride.poke(yStride.U); c.rowfacBank.poke(0.U); c.rowBase.poke(rowOff.U)
        m("op") match {
          case "LN" =>
            c.gBase.poke(WeightTables.byName(m("ln") + ".g").base.U); c.bParamBase.poke(WeightTables.byName(m("ln") + ".b").base.U)
            val eps = BigInt(m("eps_q")); c.epsLo.poke((eps & 0xffffffffL).U); c.epsHi.poke((eps >> 32).U)
          case "DYNQ" =>
            c.gelu.poke((m("gelu") == "true").B); c.mPhi.poke(m("m_phi").toLong.U); c.sPhi.poke(m("s_phi").toInt.U)
            c.smooth.poke((m("smooth") == "true").B)
            if (m("smooth") == "true") c.smoothBase.poke(WeightTables.byName(m("smooth_tensor")).base.U)
            c.static8.poke((m.getOrElse("static8", "false") == "true").B)
            if (m.getOrElse("static8", "false") == "true") { c.reqMult.poke(m("req_mult").toLong.U); c.reqShift.poke(m("req_shift").toInt.U) }
          case "ADD" =>
            c.ma.poke(m("ma").toLong.U); c.mb.poke(m("mb").toLong.U)
            if (m("b_src") == "rom") { c.bSrc.poke(1.U); c.bBase.poke((WeightTables.byName(m("b_tensor")).base + rowOff * 12).U); c.bStride.poke(12.U); c.bBits16.poke(false.B) }
            else { c.bSrc.poke(0.U); c.bBank.poke(1.U); c.bBase.poke((rowOff * bStride).U); c.bStride.poke(bStride.U); c.bBits16.poke(b16.B) }
            if (m.getOrElse("ma_from_table", "false") == "true") { c.maFromTable.poke(true.B); c.maTableBase.poke(WeightTables.byName(m("ma_table")).base.U); c.tok.poke(toks(rowOff).U) }
          case "EMBED" =>
            c.lmBase.poke(WeightTables.byName("dec.lm.w").base.U); c.tok.poke(toks(rowOff).U)
        }
        dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
        var n = 0
        while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 400000, "timeout") }
      }
      val perRow = toks.nonEmpty   // EMBED / token-add: one command per row (different token)
      if (perRow) for (r <- 0 until rows) issue(1, r) else issue(rows, 0)
      info(s"$name: ${dut.io.cycles.peek().litValue} cycles for $rows rows (${dut.io.cycles.peek().litValue / rows} per row)")
      var bad = 0
      for (i <- 0 until yWords.length / 8) {
        dut.io.read.addr.poke(i.U); dut.io.read.en.poke(true.B); dut.clock.step()
        val got = dut.io.read.data.peek().litValue; val exp = pack(yWords, 8 * i, 8)
        if (got != exp) { bad += 1; if (bad <= 4) info(f"word $i: got ${got.toString(16)} exp ${exp.toString(16)}") }
      }
      for (r <- rfExp.indices) {
        dut.io.rowfacRead.addr.poke(r.U); dut.io.rowfacRead.en.poke(true.B); dut.clock.step()
        val got = dut.io.rowfacRead.data.peek().litValue.toInt
        if (got != rfExp(r)) { bad += 1; if (bad <= 8) info(s"rowfac $r: got $got exp ${rfExp(r)}") }
      }
      assert(bad == 0, s"$name: $bad mismatches")
    }
  }
  val selected = sys.env.get("VEC_CASES").map(_.split(",").toSeq).getOrElse(allCases.toSeq)
  for (c <- selected) it should s"be bit-exact on $c" in runCase(c)
}
