package minicpm

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import minicpm.generated.WeightTables
import minicpm.sim.MiniCPMSim

import java.io.File

/** The greedy sampler: random logit streams (ties resolve to the lowest index, eos ids are flagged) injected
  * as raw beats, and the real LM-head slice 0 streamed from the engine in wide mode on the golden model's
  * final-norm row. */
class SamplerSpec extends AnyFlatSpec with MiniCPMSim {
  val allCases = Vectors.cases("sampler")
  require(allCases.nonEmpty, "run `uv run python tests/vectors/gen_all.py` first")

  def runCase(name: String): Unit = {
    val dir = new File(Vectors.dir("sampler"), name)
    val m = Vectors.meta(new File(dir, "meta.json"))
    val nTiles = m("n_tiles").toInt; val expect = m("expect").toLong; val eos = m("eos") == "true"
    val y = Vectors.hex(new File(dir, "y.hex"))
    val real = name.startsWith("lm")
    val cfg = MiniCPMConfig(weightTensors = Some(if (real) Seq("lm.00") else Seq("L0.norm1.g")))
    info(s"$name: nTiles=$nTiles expect=$expect eos=$eos real=$real")
    simulate(new SamplerTestbench(cfg), subdirectory = Some(name)) { dut =>
      dut.io.cmd.valid.poke(false.B); dut.io.actLoad.valid.poke(false.B); dut.io.rowfacLoad.valid.poke(false.B); dut.io.inject.valid.poke(false.B)
      dut.io.start.poke(false.B)
      dut.clock.step(2)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      if (real) {
        val a = Vectors.hex(new File(dir, "a.hex"))
        for (i <- 0 until a.length / 8) {
          dut.io.actLoad.valid.poke(true.B); dut.io.actLoad.bits.addr.poke(i.U); dut.io.actLoad.bits.size.poke(ActSize.W256.U)
          dut.io.actLoad.bits.data.poke(Vectors.pack(a, 8 * i, 8)); dut.clock.step()
        }
        dut.io.actLoad.valid.poke(false.B)
        val wT = WeightTables.byName("lm.00.w"); val mT = WeightTables.byName("lm.00.mult")
        val c = dut.io.cmd.bits
        c.elements.values.foreach { case u: UInt => u.poke(0.U); case b: Bool => b.poke(false.B) }
        c.rows.poke(1.U); c.kTiles.poke((cfg.dModel / 32).U); c.nTiles.poke(nTiles.U); c.actStride.poke((cfg.dModel / 32).U)
        c.wBase.poke(wT.base.U); c.wStrideN.poke((cfg.dModel / 32 * 4).U); c.wStrideK.poke(4.U); c.wStrideG.poke(1.U)
        c.outMode.poke(MatmulMode.Wide.U); c.outSink.poke(2.U); c.s1.poke(m("s1").toInt.U); c.multBase.poke(mT.base.U); c.outLocal.poke(true.B)
        dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
        var n = 0
        while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 3000000, "timeout") }
        dut.clock.step(8)
      } else {
        for (t <- 0 until nTiles) {
          dut.io.inject.valid.poke(true.B)
          val b = dut.io.inject.bits
          b.sink.poke(2.U); b.mode.poke(MatmulMode.Wide.U); b.nTile.poke(t.U); b.last.poke((t == nTiles - 1).B)
          for (i <- 0 until 32) b.data(i).poke(BigInt(y(t * 32 + i).toInt).S)
          dut.clock.step()
        }
        dut.io.inject.valid.poke(false.B)
        dut.clock.step(3)
      }
      val tok = dut.io.token.peek().litValue.toLong
      info(s"$name: token $tok (expected $expect), eos ${dut.io.isEos.peek().litToBoolean}")
      assert(tok == expect, s"$name: token $tok != $expect")
      assert(dut.io.isEos.peek().litToBoolean == eos)
    }
  }
  for (c <- allCases) it should s"pick the golden argmax on $c" in runCase(c)
}
