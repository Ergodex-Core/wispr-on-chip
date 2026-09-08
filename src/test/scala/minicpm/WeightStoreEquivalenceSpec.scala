package minicpm

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import minicpm.generated.WeightTables
import minicpm.sim.MiniCPMSim

/** RomLiteral, RomInit and Sram (after loading) must return identical data at every address, and that
  * data must equal the generated weight image parsed on the host. Covers every tensor kind (w8, i32vec,
  * i8mat, i16mat) on real tensors of MiniCPM5-2B. */
class WeightStoreEquivalenceSpec extends AnyFlatSpec with MiniCPMSim {
  val cfg = MiniCPMConfig()
  val root = cfg.repoRoot

  def checkBank(name: String, backend: WeightBackend, window: Option[Int], sample: Option[Int] = None): Unit = {
    val t = WeightTables.byName(name)
    val expect = t.memWords(root, window)
    val addrs: Seq[Int] = sample match {
      case None => expect.indices
      case Some(n) => (0 until n).map(i => (i.toLong * (expect.length - 1) / (n - 1)).toInt)
    }
    simulate(new WeightBank(t, backend, cfg, window), subdirectory = Some(s"ws_${name}_$backend")) { dut =>
      dut.io.wr.valid.poke(false.B)
      dut.io.rd2.en.poke(false.B)
      dut.io.rd2.addr.poke(0.U)
      if (backend == Sram) {
        for (a <- addrs; s <- 0 until t.linesPerWord) {
          dut.io.wr.valid.poke(true.B)
          dut.io.wr.bits.addr.poke(a.U)
          dut.io.wr.bits.slice.poke(s.U)
          dut.io.wr.bits.data.poke(((expect(a) >> (32 * s)) & 0xffffffffL).U)
          dut.clock.step()
        }
        dut.io.wr.valid.poke(false.B)
      }
      for (a <- addrs) {
        dut.io.rd.addr.poke(a.U)
        dut.io.rd.en.poke(true.B)
        dut.clock.step()
        dut.io.rd.en.poke(false.B)
        dut.io.rd.data.expect(expect(a).U(t.width.W))   // valid exactly one cycle after an enabled read
      }
    }
  }

  // small real tensors of every parameter kind, full depth, all three backends
  for (name <- Seq("L0.k.mult", "L0.norm1.g", "L20.gate.mult")) {
    for (b <- Seq(RomLiteral, RomInit, Sram))
      it should s"return the generated contents of $name under $b" in checkBank(name, b, None)
  }
  it should "match on a 64 Kbit window of L0.q.w under RomLiteral" in checkBank("L0.q.w", RomLiteral, Some(32))
  it should "match the whole of L0.k.w (4 Mbit) under RomInit" in checkBank("L0.k.w", RomInit, None)
  it should "match the whole of the RoPE table (i16mat) under RomInit" in checkBank("rope", RomInit, None)
  it should "match 256 sampled rows of embed.15 (i8mat, last slice) under RomInit" in checkBank("embed.15", RomInit, None, Some(256))
  it should "match 512 sampled words of L41.down.w (100 Mbit) under RomInit" in checkBank("L41.down.w", RomInit, None, Some(512))
  it should "match embed.resmult (4 Mbit i32vec) under RomLiteral and RomInit" in { checkBank("embed.resmult", RomLiteral, None, Some(256)); checkBank("embed.resmult", RomInit, None, Some(256)) }

  it should "decode the flat w8 address space correctly (RomInit, one layer)" in {
    val c = cfg.copy(weightTensors = Some(Seq("L0")))
    val ts = c.tensors.filter(_.kind == "w8")
    simulate(new WeightSpace(ts, c.weightWordBits, c.wAddrBits, RomInit, c), subdirectory = Some("ws_space_L0")) { dut =>
      dut.io.wr.valid.poke(false.B)
      dut.io.rd2.en.poke(false.B)
      dut.io.rd2.addr.poke(0.U)
      for (t <- ts; a <- Seq(0, 1, t.depth / 2, t.depth - 1)) {
        val expect = t.memWords(root)(a)
        dut.io.rd.addr.poke((t.base + a).U)
        dut.io.rd.en.poke(true.B)
        dut.clock.step()
        dut.io.rd.data.expect(expect.U(t.width.W))
      }
    }
  }
}
