package whisper

import chisel3._
import chisel3.util._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.generated.WeightTables
import whisper.sim.WhisperSim

import java.io.File
import scala.io.Source

object Vectors {
  val root = new File(WhisperConfig.defaultRepoRoot, "out/vectors/matmul")
  def hex(f: File): Array[Long] = {
    val src = Source.fromFile(f)
    try src.getLines().filter(_.trim.nonEmpty).map(l => java.lang.Long.parseUnsignedLong(l.trim, 16)).toArray
    finally src.close()
  }
  def meta(f: File): Map[String, String] = {
    // tiny JSON reader for flat {"k": v} objects
    val txt = Source.fromFile(f).mkString
    "\"([A-Za-z_0-9]+)\":\\s*(\"[^\"]*\"|\\[[^\\]]*\\]|[^,}\\n]+)".r.findAllMatchIn(txt).map { m =>
      m.group(1) -> m.group(2).trim.stripPrefix("\"").stripSuffix("\"")
    }.toMap
  }
}

class MatmulEngineSpec extends AnyFlatSpec with WhisperSim {
  val allCases = Option(Vectors.root.listFiles()).getOrElse(Array.empty[File]).map(_.getName).sorted
  require(allCases.nonEmpty, "run `uv run python tests/vectors/gen_matmul.py` first")

  def pack(words: Array[Long], from: Int, n: Int): BigInt =
    (0 until n).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(words(from + i)) << (32 * i)))

  def runCase(name: String): Unit = {
    val dir = new File(Vectors.root, name)
    val m = Vectors.meta(new File(dir, "meta.json"))
    val M = m("M").toInt; val K = m("K").toInt; val N = m("N").toInt
    val KT = K / 32; val NT = N / 32
    val mode = m("out_mode")
    val random = m("kind") == "random"
    val cfg = if (random) WhisperConfig(weightBackend = Sram, weightTensors = Some(Seq("enc.0")))
              else WhisperConfig(weightBackend = RomInit, weightTensors = Some(Seq(m("tensor"))))
    val w8s = cfg.tensors.filter(_.kind == "w8").sortBy(_.depth)
    val wT = if (random) w8s.find(_.depth >= KT * NT * 4).getOrElse(fail(s"no container for $name"))
             else WeightTables.byName(m("tensor") + ".w")
    val multT = if (random) cfg.tensors.filter(_.kind == "i32vec").filter(_.depth >= NT).minBy(_.depth)
                else WeightTables.byName(m("tensor") + ".mult")
    val biasT = if (random) cfg.tensors.filter(_.kind == "i32vec").filter(t => t.depth >= NT && t.name != multT.name).minBy(_.depth)
                else WeightTables.byName(m("tensor") + (if (m("has_bias") == "true") ".bias" else ".mult"))
    val aWords = Vectors.hex(new File(dir, "a.hex"))
    val yWords = Vectors.hex(new File(dir, "y.hex"))
    val rowfac = Source.fromFile(new File(dir, "rowfac.txt")).getLines().filter(_.nonEmpty).map(_.toInt).toArray
    val modeId = mode match { case "int8" => 0; case "int16" => 1; case "raw" => 2; case "wide" => 3 }
    val outStride = if (modeId >= 2) N / 32 else m("y_stride").toInt
    val actWords = 1 << 16
    info(s"$name: M=$M K=$K N=$N mode=$mode weights=${wT.name}")
    simulate(new MatmulTestbench(cfg, actWords), subdirectory = Some(name)) { dut =>
      dut.io.cmd.valid.poke(false.B)
      dut.io.actLoad.valid.poke(false.B)
      dut.io.rowfacLoad.valid.poke(false.B)
      dut.io.wsLoad.valid.poke(false.B)
      dut.io.actRead.en.poke(false.B)
      dut.clock.step(2)
      if (random) {
        val w = Vectors.hex(new File(dir, "w.hex"))
        val mu = Vectors.hex(new File(dir, "mult.hex"))
        val bi = Vectors.hex(new File(dir, "bias.hex"))
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
        load(1, biasT.base, 32, bi)
      }
      // activations: 8 x 32-bit words per 256-bit bank word
      for (i <- 0 until aWords.length / 8) {
        dut.io.actLoad.valid.poke(true.B)
        dut.io.actLoad.bits.addr.poke(i.U)
        dut.io.actLoad.bits.wide.poke(false.B)
        dut.io.actLoad.bits.data.poke(pack(aWords, 8 * i, 8))
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
      // command
      val c = dut.io.cmd.bits
      c.rows.poke(M.U); c.kTiles.poke(KT.U); c.nTiles.poke(NT.U)
      c.actSrc.poke(0.U); c.actBank.poke(0.U); c.actBase.poke(0.U); c.actStride.poke(m("a_stride").toInt.U); c.actKOff.poke(0.U)
      c.actUnsigned.poke((m("unsigned") == "true").B)
      c.im2col.poke((m("im2col") == "true").B); c.tilesPerFrame.poke(m("tiles_per_frame").toInt.U)
      c.stride2.poke((m("stride2") == "true").B); c.frames.poke(m("frames").toInt.U)
      c.wSrc.poke(0.U); c.wBase.poke(wT.base.U); c.wStrideN.poke((KT * 4).U); c.wStrideK.poke(4.U); c.wStrideG.poke(1.U)
      c.outMode.poke(modeId.U); c.outSink.poke(0.U); c.outBank.poke(1.U); c.outBase.poke(0.U); c.outStride.poke(outStride.U)
      c.outTag.poke(0.U); c.rowBase.poke(0.U)
      c.s1.poke(m("s1").toInt.U); c.multBase.poke(multT.base.U); c.biasBase.poke(biasT.base.U)
      c.hasBias.poke((m("has_bias") == "true").B); c.dynamic.poke((m("dynamic") == "true").B); c.rowfacBank.poke(0.U)
      dut.io.cmd.valid.poke(true.B)
      dut.clock.step()
      dut.io.cmd.valid.poke(false.B)
      var cycles = 0
      while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); cycles += 1; assert(cycles < 5000000, "timeout") }
      val macs = M.toLong * K * N
      val util = macs.toDouble / (cycles.toDouble * cfg.rows * cfg.cols)
      info(f"$name: $cycles cycles, ${dut.io.outCount.peek().litValue} outputs, utilisation ${util * 100}%.1f%%")
      assert(dut.io.outCount.peek().litValue == M * NT, s"expected ${M * NT} output beats")
      // compare
      var bad = 0
      if (modeId < 2) {
        val nWords = yWords.length / 8
        for (i <- 0 until nWords) {
          dut.io.actRead.addr.poke(i.U); dut.io.actRead.en.poke(true.B); dut.clock.step()
          val got = dut.io.actRead.data.peek().litValue
          val exp = pack(yWords, 8 * i, 8)
          if (got != exp) { bad += 1; if (bad <= 5) info(f"word $i: got ${got.toString(16)} exp ${exp.toString(16)}") }
        }
      } else {
        val nWords = yWords.length / 32
        for (i <- 0 until nWords) {
          dut.io.rawRead.addr.poke(i.U); dut.clock.step()
          val got = dut.io.rawRead.data.peek().litValue
          val exp = pack(yWords, 32 * i, 32)
          if (got != exp) { bad += 1; if (bad <= 5) info(f"raw word $i: got ${got.toString(16)} exp ${exp.toString(16)}") }
        }
      }
      assert(bad == 0, s"$name: $bad mismatching output words")
      if (name.startsWith("rand_9")) assert(util >= 0.9, f"utilisation ${util}%.3f < 0.9")
    }
  }

  val selected = sys.env.get("MATMUL_CASES").map(_.split(",").toSeq).getOrElse(allCases.toSeq)
  for (c <- selected) it should s"be bit-exact on $c" in runCase(c)
}
