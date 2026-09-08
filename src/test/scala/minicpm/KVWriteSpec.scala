package minicpm

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import minicpm.generated.WeightTables
import minicpm.sim.MiniCPMSim

import java.io.File

/** The two paths that fill the KV cache, on layer 0 of the real model: V through the engine (real v_proj on
  * real activations, int8 beats key-major) and K through the vector unit's RoPE op (real k rows rotated,
  * beats transposed by the cache). Also a single key appended at an arbitrary slot (decode). */
class KVWriteSpec extends AnyFlatSpec with MiniCPMSim {
  val dir = new File(Vectors.dir("kv"), "L0_rows40")
  require(dir.exists(), "run `uv run python tests/vectors/gen_all.py` first")

  it should "write V (engine) and K (RoPE beats) bit-exactly, whole rows and a single key" in {
    val m = Vectors.meta(new File(dir, "meta.json"))
    val rows = m("rows").toInt; val keysMax = m("keys_max").toInt
    val kExp = Vectors.hex(new File(dir, "k.hex")); val vExp = Vectors.hex(new File(dir, "v.hex"))
    val a = Vectors.hex(new File(dir, "a.hex")); val k16 = Vectors.hex(new File(dir, "k16.hex"))
    val rf = Vectors.ints(new File(dir, "rowfac.txt"))
    val cfg = MiniCPMConfig(weightTensors = Some(Seq("L0.v", "rope")), maxCtx = keysMax, kvLayers = 1)
    val perHead = KVRegion.wordsPerHead(cfg); val H = cfg.nKvHead
    val kBase = KVRegion.kBase(cfg, 0); val vBase = KVRegion.vBase(cfg, 0)
    val vT = WeightTables.byName("L0.v.w"); val vM = WeightTables.byName("L0.v.mult")
    val s1 = {
      val man = scala.io.Source.fromFile(new File(MiniCPMConfig.defaultRepoRoot, "weights/MANIFEST.json")).mkString
      "\"L0\\.v\": \\{[^}]*\"s1\": (\\d+)".r.findFirstMatchIn(man).map(_.group(1).toInt).getOrElse(fail("s1 for L0.v"))
    }
    simulate(new KVTestbench(cfg), subdirectory = Some("kvwrite")) { dut =>
      dut.io.mm.valid.poke(false.B); dut.io.vec.valid.poke(false.B); dut.io.actLoad.valid.poke(false.B); dut.io.rowfacLoad.valid.poke(false.B)
      dut.io.flush.poke(false.B); dut.io.kvRead.en.poke(false.B)
      dut.clock.step(2)
      def loadBank(bank: Int, w: Array[Long]): Unit = {
        for (i <- 0 until w.length / 8) { dut.io.actLoad.valid.poke(true.B); dut.io.actLoad.bits.bank.poke(bank.U); dut.io.actLoad.bits.addr.poke(i.U); dut.io.actLoad.bits.data.poke(Vectors.pack(w, 8 * i, 8)); dut.clock.step() }
        dut.io.actLoad.valid.poke(false.B)
      }
      loadBank(0, a); loadBank(1, k16)
      for (i <- rf.indices) { dut.io.rowfacLoad.valid.poke(true.B); dut.io.rowfacLoad.bits.addr.poke(i.U); dut.io.rowfacLoad.bits.data.poke(rf(i).U); dut.clock.step() }
      dut.io.rowfacLoad.valid.poke(false.B)
      def waitIdle(): Unit = { var n = 0; while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 2000000, "timeout") }; dut.clock.step(4) }
      def waitKV(): Unit = { var n = 0; while (dut.io.kvBusy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 100000, "kv busy timeout") }; dut.clock.step(4) }
      def setKV(isK: Boolean, base: Int, keyOff: Int): Unit = {
        dut.io.kvCmd.isK.poke(isK.B); dut.io.kvCmd.base.poke(base.U); dut.io.kvCmd.keysMax.poke(keysMax.U); dut.io.kvCmd.keyOff.poke(keyOff.U); dut.io.kvCmd.flush.poke(false.B)
      }
      def runV(base: Int, nRows: Int, rowOff: Int, keyOff: Int): Unit = {
        setKV(false, base, keyOff)
        val c = dut.io.mm.bits
        c.elements.values.foreach { case u: UInt => u.poke(0.U); case b: Bool => b.poke(false.B) }
        c.rows.poke(nRows.U); c.kTiles.poke((cfg.dModel / 32).U); c.nTiles.poke((cfg.kvDim / 32).U); c.actStride.poke((cfg.dModel / 32).U); c.rowOff.poke(rowOff.U)
        c.wBase.poke(vT.base.U); c.wStrideN.poke((cfg.dModel / 32 * 4).U); c.wStrideK.poke(4.U); c.wStrideG.poke(1.U)
        c.outMode.poke(MatmulMode.Int8.U); c.outSink.poke(3.U); c.outStride.poke((cfg.kvDim / 32).U); c.outLocal.poke(true.B)
        c.s1.poke(s1.U); c.multBase.poke(vM.base.U); c.dynamic.poke(true.B)
        dut.io.mm.valid.poke(true.B); dut.clock.step(); dut.io.mm.valid.poke(false.B)
        waitIdle(); waitKV()
      }
      def runK(base: Int, nRows: Int, rowOff: Int, keyOff: Int): Unit = {
        setKV(true, base, keyOff)
        val c = dut.io.vec.bits
        c.elements.values.foreach { case u: UInt => u.poke(0.U); case b: Bool => b.poke(false.B) }
        c.op.poke(VecOp.ROPE.U); c.rows.poke(nRows.U); c.cols.poke(cfg.kvDim.U)
        c.inBank.poke(1.U); c.inBase.poke(0.U); c.inStride.poke((cfg.kvDim / 16).U); c.inBits.poke(VecBits.I16.U); c.rowOff.poke(rowOff.U)
        c.outSink.poke(1.U); c.outStride.poke((cfg.kvDim / 32).U); c.posBase.poke(rowOff.U)
        c.reqMult.poke(m("m_r").toLong.U); c.reqShift.poke(m("s_r").toInt.U); c.ropeBase.poke(WeightTables.byName("rope").base.U)
        dut.io.vec.valid.poke(true.B); dut.clock.step(); dut.io.vec.valid.poke(false.B)
        waitIdle()
        dut.io.flush.poke(true.B); dut.clock.step(); dut.io.flush.poke(false.B)
        waitKV()
      }
      runV(vBase, rows, 0, 0)
      runK(kBase, rows, 0, 0)
      def check(name: String, base: Int, exp: Array[Long]): Unit = {
        var bad = 0
        for (i <- 0 until exp.length / 64) {
          val e = Vectors.pack(exp, 64 * i, 64)
          if (e != 0) {
            dut.io.kvRead.addr.poke((base + i).U); dut.io.kvRead.en.poke(true.B); dut.clock.step()
            val got = dut.io.kvRead.data.peek().litValue
            if (got != e) { bad += 1; if (bad <= 4) {
              val firstByte = (0 until 256).find(b => ((got >> (8 * b)) & 0xff) != ((e >> (8 * b)) & 0xff)).getOrElse(-1)
              info(f"$name word $i (head ${i / perHead}): first diff byte $firstByte (row ${firstByte / 32} col ${firstByte % 32})")
            } }
          }
        }
        assert(bad == 0, s"$name: $bad mismatching cache words")
      }
      check("K", kBase, kExp)
      check("V", vBase, vExp)
      // single key: activation row 5 (rowOff = 5, one row) written at key slot 5 + 200 of a second region
      val row = 5; val keyOffT = 200; val key = keyOffT + row
      val kBase2 = kBase; val vBase2 = vBase   // same regions, different key slots
      runV(vBase2, 1, row, keyOffT)
      // the RoPE op rotates with position rowOff + 0 = 5, so the expected bytes are those of key 5 in the packed K
      runK(kBase2, 1, row, keyOffT)
      def byteOf(words: Array[Long], word: Int, b: Int): Int = ((words(word * 64 + b / 4) >> (8 * (b % 4))) & 0xff).toInt
      var bad = 0
      val HT = cfg.headTiles
      for (h <- 0 until H; dTile <- 0 until HT; g <- 0 until 4; r <- 0 until 8) {
        val wExp = h * perHead + (((row / 32) * HT + dTile) * 4 + g)
        val wGot = h * perHead + (((key / 32) * HT + dTile) * 4 + g)
        val exp = byteOf(kExp, wExp, r * 32 + row % 32)
        dut.io.kvRead.addr.poke((kBase2 + wGot).U); dut.io.kvRead.en.poke(true.B); dut.clock.step()
        val got = ((dut.io.kvRead.data.peek().litValue >> (8 * (r * 32 + key % 32))) & 0xff).toInt
        if (got != exp) { bad += 1; if (bad <= 4) info(s"K single key h=$h dTile=$dTile g=$g r=$r: got $got exp $exp") }
      }
      for (h <- 0 until H; dTile <- 0 until HT; c <- 0 until 32) {
        val wExp = h * perHead + ((dTile * (keysMax / 32) + row / 32) * 4 + (row % 32) / 8)
        val wGot = h * perHead + ((dTile * (keysMax / 32) + key / 32) * 4 + (key % 32) / 8)
        val exp = byteOf(vExp, wExp, (row % 8) * 32 + c)
        dut.io.kvRead.addr.poke((vBase2 + wGot).U); dut.io.kvRead.en.poke(true.B); dut.clock.step()
        val got = ((dut.io.kvRead.data.peek().litValue >> (8 * ((key % 8) * 32 + c))) & 0xff).toInt
        if (got != exp) { bad += 1; if (bad <= 4) info(s"V single key h=$h dTile=$dTile c=$c: got $got exp $exp") }
      }
      assert(bad == 0, s"single-key write: $bad mismatching bytes")
    }
  }
}
