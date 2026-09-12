package whisper

import chisel3._
import org.scalatest.flatspec.AnyFlatSpec
import whisper.generated.WeightTables
import whisper.sim.WhisperSim

/** RomLiteral, RomInit and Sram (after loading) must return identical data at every address, and that
  * data must equal the committed hex file parsed on the host. */
class WeightStoreEquivalenceSpec extends AnyFlatSpec with WhisperSim {
  val cfg = WhisperConfig()
  val root = cfg.repoRoot

  def checkBank(name: String, backend: WeightBackend, window: Option[Int]): Unit = {
    val t = WeightTables.byName(name)
    val expect = t.memWords(root, window)
    simulate(new WeightBank(t, backend, cfg, window)) { dut =>
      dut.io.wr.valid.poke(false.B)
      dut.io.rd2.en.poke(false.B)
      dut.io.rd2.addr.poke(0.U)
      if (backend == Sram) {
        for (a <- expect.indices; s <- 0 until t.linesPerWord) {
          dut.io.wr.valid.poke(true.B)
          dut.io.wr.bits.addr.poke(a.U)
          dut.io.wr.bits.slice.poke(s.U)
          dut.io.wr.bits.data.poke(((expect(a) >> (32 * s)) & 0xffffffffL).U)
          dut.clock.step()
        }
        dut.io.wr.valid.poke(false.B)
      }
      for (a <- expect.indices) {
        dut.io.rd.addr.poke(a.U)
        dut.io.rd.en.poke(true.B)
        dut.clock.step()
        dut.io.rd.en.poke(false.B)
        dut.io.rd.data.expect(expect(a).U(t.width.W))   // valid exactly one cycle after an enabled read
      }
    }
  }

  // two real tensors of different kinds, full depth
  for (name <- Seq("enc.0.attn.q.mult", "enc.0.ln1.g", "dec.emb.resmult")) {
    for (b <- Seq(RomLiteral, RomInit, Sram))
      it should s"return the committed contents of $name under $b" in checkBank(name, b, None)
  }
  // a 32-word window (64 Kbit) of a real weight matrix under RomLiteral; full tensor under RomInit/Sram
  it should "match on a 64 Kbit window of enc.0.attn.q.w under RomLiteral" in checkBank("enc.0.attn.q.w", RomLiteral, Some(32))
  it should "match the whole of enc.0.attn.q.w under RomInit" in checkBank("enc.0.attn.q.w", RomInit, None)
  it should "match the whole of enc.pos under RomInit" in checkBank("enc.pos", RomInit, None)

  it should "decode the flat w8 address space correctly (RomInit, one layer)" in {
    val c = cfg.copy(weightTensors = Some(Seq("enc.0")))
    val ts = c.tensors.filter(_.kind == "w8")
    simulate(new WeightSpace(ts, c.weightWordBits, RomInit, c)) { dut =>
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
