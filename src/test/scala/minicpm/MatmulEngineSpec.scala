package minicpm

import chisel3._
import chisel3.util._
import org.scalatest.flatspec.AnyFlatSpec
import minicpm.generated.WeightTables
import minicpm.sim.MiniCPMSim

import java.io.File
import scala.io.Source

object Vectors {
  val root = new File(MiniCPMConfig.defaultRepoRoot, "out/vectors")
  def dir(unit: String) = new File(root, unit)
  def cases(unit: String): Seq[String] = Option(dir(unit).listFiles()).getOrElse(Array.empty[File]).filter(_.isDirectory).map(_.getName).sorted.toSeq
  def hex(f: File): Array[Long] = {
    val src = Source.fromFile(f)
    try src.getLines().filter(_.trim.nonEmpty).map(l => java.lang.Long.parseUnsignedLong(l.trim, 16)).toArray
    finally src.close()
  }
  def ints(f: File): Array[Long] = if (f.exists()) Source.fromFile(f).getLines().filter(_.trim.nonEmpty).map(_.trim.toLong).toArray else Array.empty[Long]
  def meta(f: File): Map[String, String] = {
    // tiny JSON reader for flat {"k": v} objects
    val txt = Source.fromFile(f).mkString
    "\"([A-Za-z_0-9]+)\":\\s*(\"[^\"]*\"|\\[[^\\]]*\\]|[^,}\\n]+)".r.findAllMatchIn(txt).map { m =>
      m.group(1) -> m.group(2).trim.stripPrefix("\"").stripSuffix("\"")
    }.toMap
  }
  def list(s: String): Array[Long] = s.split("[\\[\\], ]+").filter(_.nonEmpty).map(_.toLong)
  def pack(words: Array[Long], from: Int, n: Int): BigInt =
    (0 until n).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(words(from + i)) << (32 * i)))
  def bank(m: Map[String, String], key: String): Int = m(key).toInt
}

/** Every matmul case: random shapes (all output modes, dynamic / static / unsigned rows, 32-bit row factors)
  * through the Sram weight store, and the real tensors of layers 0 / 20 / 41 and the LM head on the golden
  * model's activations through RomInit. Outputs must match the golden words bit for bit. */
class MatmulEngineSpec extends AnyFlatSpec with MiniCPMSim {
  val allCases = Vectors.cases("matmul")
  require(allCases.nonEmpty, "run `uv run python tests/vectors/gen_all.py` first")
  def realBackend: WeightBackend = RomInit
  def romLiteralMaxBits: Int = 4 * 1024 * 1024
  val modeId = Map("int8" -> MatmulMode.Int8, "int16" -> MatmulMode.Int16, "raw" -> MatmulMode.Raw, "wide" -> MatmulMode.Wide, "int32" -> MatmulMode.Int32)
  val wordsPerRow = Map("int8" -> 8, "int16" -> 8, "int32" -> 8, "raw" -> 32, "wide" -> 32)   // 32-bit words per 256/1024-bit compare unit

  def runCase(name: String): Unit = {
    val dir = new File(Vectors.dir("matmul"), name)
    val m = Vectors.meta(new File(dir, "meta.json"))
    val M = m("M").toInt; val K = m("K").toInt; val N = m("N").toInt
    val KT = K / 32; val NT = N / 32
    val mode = m("out_mode")
    val random = m("kind") == "random"
    val cfg = if (random) MiniCPMConfig(weightBackend = Sram, weightTensors = Some(Seq("L0.q", "L0.k")))
              else MiniCPMConfig(weightBackend = realBackend, weightTensors = Some(Seq(m("w_tensor"), m("mult_tensor"))), romLiteralMaxBits = romLiteralMaxBits)
    val wT = if (random) WeightTables.byName("L0.q.w") else WeightTables.byName(m("w_tensor"))
    val multT = if (random) WeightTables.byName("L0.q.mult") else WeightTables.byName(m("mult_tensor"))
    require(!random || (KT * NT * 4 <= wT.depth && NT <= multT.depth * 32), s"random case $name does not fit the container tensors")
    val aWords = Vectors.hex(new File(dir, "a.hex"))
    val yWords = Vectors.hex(new File(dir, "y.hex"))
    val rowfac = Vectors.ints(new File(dir, "rowfac.txt"))
    val mid = modeId(mode)
    val outStride = if (mid == MatmulMode.Raw || mid == MatmulMode.Wide) NT else m("y_stride").toInt
    val actWords = 1 << 16
    info(s"$name: M=$M K=$K N=$N mode=$mode weights=${wT.name} backend=${cfg.weightBackend}")
    val tElab = System.nanoTime()
    simulate(new MatmulTestbench(cfg, actWords), subdirectory = Some(name)) { dut =>
      info(f"$name: elaboration+build ${(System.nanoTime() - tElab) / 1e9}%.0f s")
      dut.io.cmd.valid.poke(false.B)
      dut.io.actLoad.valid.poke(false.B)
      dut.io.rowfacLoad.valid.poke(false.B)
      dut.io.wsLoad.valid.poke(false.B)
      dut.io.actRead.en.poke(false.B)
      dut.clock.step(2)
      if (random) {
        val w = Vectors.hex(new File(dir, "w.hex"))
        val mu = Vectors.hex(new File(dir, "mult.hex"))
        def load(space: Int, base: Int, lpw: Int, words: Array[Long]): Unit = {
          for (i <- words.indices) {
            dut.io.wsLoad.valid.poke(true.B)
            dut.io.wsLoad.bits.space.poke(space.U)
            dut.io.wsLoad.bits.addr.poke((base + i / lpw).U)
            dut.io.wsLoad.bits.slice.poke((i % lpw).U)
            dut.io.wsLoad.bits.data.poke(words(i).U)
            dut.clock.step()
          }
          dut.io.wsLoad.valid.poke(false.B)
        }
        load(0, wT.base, 64, w)
        load(1, multT.base, 32, mu)
      }
      for (i <- 0 until aWords.length / 8) {
        dut.io.actLoad.valid.poke(true.B)
        dut.io.actLoad.bits.addr.poke(i.U)
        dut.io.actLoad.bits.size.poke(ActSize.W256.U)
        dut.io.actLoad.bits.data.poke(Vectors.pack(aWords, 8 * i, 8))
        dut.clock.step()
      }
      dut.io.actLoad.valid.poke(false.B)
      for (i <- rowfac.indices) {
        dut.io.rowfacLoad.valid.poke(true.B)
        dut.io.rowfacLoad.bits.addr.poke(i.U)
        dut.io.rowfacLoad.bits.data.poke(rowfac(i).U)
        dut.clock.step()
      }
      dut.io.rowfacLoad.valid.poke(false.B)
      val c = dut.io.cmd.bits
      c.elements.values.foreach { case u: UInt => u.poke(0.U); case b: Bool => b.poke(false.B) }
      c.rows.poke(M.U); c.kTiles.poke(KT.U); c.nTiles.poke(NT.U)
      c.actSrc.poke(0.U); c.actBank.poke(0.U); c.actBase.poke(0.U); c.actStride.poke(m("a_stride").toInt.U); c.actKOff.poke(0.U)
      c.actUnsigned.poke((m("unsigned") == "true").B)
      c.wSrc.poke(0.U); c.wBase.poke(wT.base.U); c.wStrideN.poke((KT * 4).U); c.wStrideK.poke(4.U); c.wStrideG.poke(1.U)
      c.outMode.poke(mid.U); c.outSink.poke(0.U); c.outBank.poke(1.U); c.outBase.poke(0.U); c.outStride.poke(outStride.U)
      c.outTag.poke(0.U); c.rowBase.poke(0.U); c.outLocal.poke(true.B)
      c.s1.poke(m("s1").toInt.U); c.multBase.poke(multT.base.U); c.biasBase.poke(0.U)
      c.hasBias.poke(false.B); c.dynamic.poke((m("dynamic") == "true").B); c.rowfacBank.poke(0.U)
      dut.io.cmd.valid.poke(true.B)
      dut.clock.step()
      dut.io.cmd.valid.poke(false.B)
      var cycles = 0
      while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); cycles += 1; assert(cycles < 5000000, "timeout") }
      val macs = M.toLong * K * N
      val util = macs.toDouble / (cycles.toDouble * cfg.rows * cfg.cols)
      info(f"$name: $cycles cycles, ${dut.io.outCount.peek().litValue} outputs, utilisation ${util * 100}%.1f%%")
      assert(dut.io.outCount.peek().litValue == M * NT, s"expected ${M * NT} output beats")
      var bad = 0
      if (mid == MatmulMode.Raw || mid == MatmulMode.Wide) {
        val nWords = yWords.length / 32
        for (i <- 0 until nWords) {
          dut.io.rawRead.addr.poke(i.U); dut.clock.step()
          val got = dut.io.rawRead.data.peek().litValue
          val exp = Vectors.pack(yWords, 32 * i, 32)
          if (got != exp) { bad += 1; if (bad <= 5) info(f"raw word $i: got ${got.toString(16)} exp ${exp.toString(16)}") }
        }
      } else {
        val nWords = yWords.length / 8
        for (i <- 0 until nWords) {
          dut.io.actRead.addr.poke(i.U); dut.io.actRead.en.poke(true.B); dut.clock.step()
          val got = dut.io.actRead.data.peek().litValue
          val exp = Vectors.pack(yWords, 8 * i, 8)
          if (got != exp) { bad += 1; if (bad <= 5) info(f"word $i: got ${got.toString(16)} exp ${exp.toString(16)}") }
        }
      }
      assert(bad == 0, s"$name: $bad mismatching output words")
      if (name.startsWith("rand_9_")) assert(util >= 0.85, f"utilisation ${util}%.3f < 0.85")
    }
  }

  def selected: Seq[String] = sys.env.get("MATMUL_CASES").map(_.split(",").toSeq).getOrElse(allCases)
  for (c <- selected) it should s"be bit-exact on $c" in runCase(c)
}

/** The ASIC ROM path: real tensors as literal `VecInit` ROMs (the K and V projections of layer 0, 4 Mbit
  * and 4 Mbit), bit-exact on the same real-activation cases. */
class RomLiteralLayerSpec extends MatmulEngineSpec {
  override def realBackend: WeightBackend = RomLiteral
  override def romLiteralMaxBits: Int = 8 * 1024 * 1024
  override def selected: Seq[String] = sys.env.get("ROMLIT_CASES").map(_.split(",").toSeq).getOrElse(Seq("real_L0_k", "real_L0_v"))
}
