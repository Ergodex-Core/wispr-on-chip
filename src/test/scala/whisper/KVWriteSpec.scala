package whisper

import chisel3._
import chisel3.util._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.generated.WeightTables
import whisper.sim.WhisperSim

import java.io.File
import scala.io.Source

/** Engine -> KVCache write path: run the real K and V projections (enc.0) on real activations and
  * compare the cache contents with the packed K^T / V layouts produced by the golden (attention vectors). */
class KVTestbench(cfg: WhisperConfig, actWords: Int = 1 << 14) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new MatmulCmd))
    val busy = Output(Bool())
    val kvCmd = Input(new KVCmd)
    val flush = Input(Bool())
    val kvBusy = Output(Bool())
    val actLoad = Flipped(Valid(new ActWrite(log2Ceil(actWords))))
    val rowfacLoad = Flipped(Valid(new Bundle { val addr = UInt(13.W); val data = UInt(16.W) }))
    val kvRead = new Bundle { val addr = Input(UInt(20.W)); val en = Input(Bool()); val data = Output(UInt(2048.W)) }
  })
  val eng = Module(new MatmulEngine(cfg)); val ws = Module(new WeightStore(cfg)); val kv = Module(new KVCache(cfg))
  val bank = Module(new ActBank(actWords)); val rowfac = Module(new RowFacTable(8192))
  eng.io.cmd <> io.cmd; io.busy := eng.io.busy
  bank.io.rd.addr := eng.io.act.addr; bank.io.rd.en := eng.io.act.en; eng.io.act.data := bank.io.rd.data; bank.io.wr := io.actLoad
  ws.io.w.addr := eng.io.w.addr; ws.io.w.en := eng.io.w.en; eng.io.w.data := ws.io.w.data
  ws.io.p.addr := eng.io.mult.addr; ws.io.p.en := eng.io.mult.en; eng.io.mult.data := ws.io.p.data
  ws.io.p2.addr := eng.io.bias.addr; ws.io.p2.en := eng.io.bias.en; eng.io.bias.data := ws.io.p2.data
  ws.io.t.addr := 0.U; ws.io.t.en := false.B; ws.io.load.valid := false.B; ws.io.load.bits := DontCare
  rowfac.io.rd.addr := eng.io.rowfac.addr; rowfac.io.rd.en := eng.io.rowfac.en; eng.io.rowfac.data := rowfac.io.rd.data; rowfac.io.wr := io.rowfacLoad
  kv.io.in <> eng.io.out
  kv.io.cmd := io.kvCmd; kv.io.flush := io.flush; io.kvBusy := kv.io.busy
  kv.io.rd.addr := io.kvRead.addr; kv.io.rd.en := io.kvRead.en; io.kvRead.data := kv.io.rd.data
}

class KVWriteSpec extends AnyFlatSpec with WhisperSim {
  val root = new File(WhisperConfig.defaultRepoRoot, "out/vectors")
  def pack(words: Array[Long], from: Int, n: Int): BigInt = (0 until n).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(words(from + i)) << (32 * i)))

  it should "write K (transposed) and V through the engine bit-exactly" in {
    val cfg = WhisperConfig(weightTensors = Some(Seq("enc.0.attn.k", "enc.0.attn.v")))
    val aDir = new File(root, "attention/enc0_full")
    val kExp = Vectors.hex(new File(aDir, "k.hex")); val vExp = Vectors.hex(new File(aDir, "v.hex"))
    val mDir = new File(root, "matmul/real_enc0_k")
    val a = Vectors.hex(new File(mDir, "a.hex"))
    val rf = Source.fromFile(new File(mDir, "rowfac.txt")).getLines().filter(_.nonEmpty).map(_.toInt).toArray
    val M = 128; val keysMax = 1536; val perHead = 2 * (keysMax / 32) * 4
    val encKeys = KVRegion.encKeys(cfg); val H = cfg.nHead; val phEnc = KVRegion.wordsPerHeadK(encKeys)
    val kBase = 0; val vBase = H * phEnc
    simulate(new KVTestbench(cfg), subdirectory = Some("kvwrite")) { dut =>
      dut.io.cmd.valid.poke(false.B); dut.io.actLoad.valid.poke(false.B); dut.io.rowfacLoad.valid.poke(false.B); dut.io.flush.poke(false.B); dut.io.kvRead.en.poke(false.B)
      dut.clock.step(2)
      for (i <- 0 until a.length / 8) { dut.io.actLoad.valid.poke(true.B); dut.io.actLoad.bits.addr.poke(i.U); dut.io.actLoad.bits.wide.poke(false.B); dut.io.actLoad.bits.data.poke(pack(a, 8 * i, 8)); dut.clock.step() }
      dut.io.actLoad.valid.poke(false.B)
      for (i <- rf.indices) { dut.io.rowfacLoad.valid.poke(true.B); dut.io.rowfacLoad.bits.addr.poke(i.U); dut.io.rowfacLoad.bits.data.poke(rf(i).U); dut.clock.step() }
      dut.io.rowfacLoad.valid.poke(false.B)
      def run(tensor: String, isK: Boolean, base: Int, s1: Int, rows: Int = M, rowOff: Int = 0, keyOff: Int = 0): Unit = {
        val wT = WeightTables.byName(tensor + ".w"); val multT = WeightTables.byName(tensor + ".mult")
        val biasT = WeightTables.byName(tensor + (if (WeightTables.byName.contains(tensor + ".bias")) ".bias" else ".mult"))
        val c = dut.io.cmd.bits
        c.elements.values.foreach { case u: UInt => u.poke(0.U); case b: Bool => b.poke(false.B) }
        c.rows.poke(rows.U); c.kTiles.poke(12.U); c.nTiles.poke(12.U); c.actStride.poke(12.U); c.rowOff.poke(rowOff.U)
        c.wBase.poke(wT.base.U); c.wStrideN.poke(48.U); c.wStrideK.poke(4.U); c.wStrideG.poke(1.U)
        c.outMode.poke(0.U); c.outSink.poke(3.U); c.outStride.poke(12.U)
        c.s1.poke(s1.U)
        c.multBase.poke(multT.base.U); c.biasBase.poke(biasT.base.U); c.hasBias.poke(WeightTables.byName.contains(tensor + ".bias").B); c.dynamic.poke(true.B)
        dut.io.kvCmd.isK.poke(isK.B); dut.io.kvCmd.base.poke(base.U); dut.io.kvCmd.keysMax.poke(keysMax.U); dut.io.kvCmd.keyOff.poke(keyOff.U); dut.io.kvCmd.flush.poke(false.B)
        dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
        var n = 0
        while (dut.io.busy.peek().litToBoolean) { dut.clock.step(); n += 1 }
        dut.clock.step(4)
        if (isK) { dut.io.flush.poke(true.B); dut.clock.step(); dut.io.flush.poke(false.B) }
        n = 0
        while (dut.io.kvBusy.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 10000, "kv busy timeout") }
        dut.clock.step(4)
      }
      // V uses s1 of the v tensor: read from the manifest-derived vectors instead
      val man = Source.fromFile(new File(WhisperConfig.defaultRepoRoot, "weights/MANIFEST.json")).mkString
      def s1of(t: String): Int = ("\"" + t.replace(".", "\\.") + "\": \\{[^}]*\"s1\": (\\d+)").r.findFirstMatchIn(man).map(_.group(1).toInt).getOrElse(fail(s"s1 for $t"))
      run("enc.0.attn.k", true, kBase, s1of("enc.0.attn.k"))
      run("enc.0.attn.v", false, vBase, s1of("enc.0.attn.v"))
      def check(name: String, base: Int, exp: Array[Long]): Unit = {
        var bad = 0
        // only words that cover real keys (0..127) are checked: per head, per (nt<4), all 8 words of the tile column... simpler: check all words where exp != 0
        for (i <- 0 until exp.length / 64) {
          val e = pack(exp, 64 * i, 64)
          if (e != 0) {
            dut.io.kvRead.addr.poke((base + i).U); dut.io.kvRead.en.poke(true.B); dut.clock.step()
            val got = dut.io.kvRead.data.peek().litValue
            if (got != e) { bad += 1; if (bad <= 4) {
              val firstByte = (0 until 256).find(b => ((got >> (8 * b)) & 0xff) != ((e >> (8 * b)) & 0xff)).getOrElse(-1)
              val gb = (0 until 256).map(b => ((got >> (8 * b)) & 0xff).toInt); val eb = (0 until 256).map(b => ((e >> (8 * b)) & 0xff).toInt)
              info(f"$name word $i: first diff byte $firstByte (row ${firstByte / 32} col ${firstByte % 32}); rows got ${(0 until 8).map(r => gb(r * 32)).mkString(",")} exp ${(0 until 8).map(r => eb(r * 32)).mkString(",")}")
            } }
          }
        }
        assert(bad == 0, s"$name: $bad mismatching cache words")
      }
      check("K", kBase, kExp)
      check("V", vBase, vExp)
      // single key written at an offset (decoder-style): row 41 of the activations -> key slot 300 of a fresh region
      val row = 41; val keyOffT = 300; val key = keyOffT + row   // key = keyOff + (rowOff + m)
      val kBase2 = 2 * H * phEnc; val vBase2 = 3 * H * phEnc   // decSelfK / decSelfV regions (keysMax stays 1536 here)
      run("enc.0.attn.k", true, kBase2, s1of("enc.0.attn.k"), rows = 1, rowOff = row, keyOff = keyOffT)
      run("enc.0.attn.v", false, vBase2, s1of("enc.0.attn.v"), rows = 1, rowOff = row, keyOff = keyOffT)
      def byteOf(words: Array[Long], word: Int, b: Int): Int = ((words(word * 64 + b / 4) >> (8 * (b % 4))) & 0xff).toInt
      var bad = 0
      for (h <- 0 until H; dTile <- 0 until 2; g <- 0 until 4; r <- 0 until 8) {
        // K^T: word (nt=key/32, kt=dTile, g) ; byte r*32 + key%32 holds K[key][dTile*32 + g*8 + r]
        val wExp = h * perHead + (((row / 32) * 2 + dTile) * 4 + g)
        val wGot = h * perHead + (((key / 32) * 2 + dTile) * 4 + g)
        val exp = byteOf(kExp, wExp, r * 32 + row % 32)
        dut.io.kvRead.addr.poke((kBase2 + wGot).U); dut.io.kvRead.en.poke(true.B); dut.clock.step()
        val got = ((dut.io.kvRead.data.peek().litValue >> (8 * (r * 32 + key % 32))) & 0xff).toInt
        if (got != exp) { bad += 1; if (bad <= 4) info(s"K single key h=$h dTile=$dTile g=$g r=$r: got $got exp $exp") }
      }
      for (h <- 0 until H; dTile <- 0 until 2; c <- 0 until 32) {
        // V: word (nt=dTile, kt=key/32, g=(key%32)/8) ; byte (key%8)*32 + c holds V[key][dTile*32 + c]
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
