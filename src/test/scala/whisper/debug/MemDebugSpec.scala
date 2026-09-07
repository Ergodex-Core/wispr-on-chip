package whisper.debug

import chisel3._
import whisper.sim.WhisperSim
import org.scalatest.flatspec.AnyFlatSpec
import whisper._
import whisper.generated.WeightTables

class MemDebugSpec extends AnyFlatSpec with WhisperSim {
  val cfg = WhisperConfig()
  it should "emit verilog" in {
    val t = WeightTables.byName("enc.0.attn.q.mult")
    val sv = _root_.circt.stage.ChiselStage.emitSystemVerilog(new WeightBank(t, RomInit, cfg), firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"))
    java.nio.file.Files.write(java.nio.file.Paths.get("target/dbg_rominit.sv"), sv.getBytes)
    val sv2 = _root_.circt.stage.ChiselStage.emitSystemVerilog(new WeightBank(t, Sram, cfg), firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"))
    java.nio.file.Files.write(java.nio.file.Paths.get("target/dbg_sram.sv"), sv2.getBytes)
  }
  it should "trace reads" in {
    val t = WeightTables.byName("enc.0.attn.q.mult")
    val expect = t.memWords(cfg.repoRoot)
    simulate(new WeightBank(t, RomInit, cfg)) { dut =>
      dut.io.wr.valid.poke(false.B)
      dut.io.rd.en.poke(true.B)
      for (c <- 0 until 6) {
        dut.io.rd.addr.poke(c.U)
        val d = dut.io.rd.data.peek().litValue
        println(s"cycle $c addr=$c data_before_step=${d.toString(16).take(16)} expect(${c})=${expect(c).toString(16).take(16)} expect(${(c-1) max 0})=${expect((c-1) max 0).toString(16).take(16)}")
        dut.clock.step()
      }
    }
  }
}
