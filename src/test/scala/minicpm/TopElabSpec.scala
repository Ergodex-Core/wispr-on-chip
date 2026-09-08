package minicpm

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

/** Elaborates the whole chip (sequencer + 848-instruction micro-program + all units + banks + a 42-layer KV
  * cache) to SystemVerilog with layer 0's weights instantiated (the rest of the weight space decodes to zero).
  * This proves the generated program fits the units' command bundles and that the top is a consistent design;
  * the full-model simulation itself is out of scope (docs/status.md). */
class TopElabSpec extends AnyFlatSpec {
  it should "elaborate MiniCPMTop to SystemVerilog" in {
    val cfg = MiniCPMConfig(weightTensors = Some(Seq("L0", "norm_f", "embed.00", "embed.resmult", "rope", "lm.00")))
    val t0 = System.nanoTime()
    val out = new java.io.File(cfg.repoRoot, "target/top-sv")
    ChiselStage.emitSystemVerilogFile(new MiniCPMTop(cfg), Array("--target-dir", out.getAbsolutePath),
      Array("-disable-all-randomization", "-strip-debug-info", "--lowering-options=disallowLocalVariables,disallowPackedArrays"))
    val sv = new java.io.File(out, "MiniCPMTop.sv")
    assert(sv.exists(), "no SystemVerilog emitted")
    val files = out.listFiles().filter(_.getName.endsWith(".sv"))
    val seqSv = new java.io.File(out, "Sequencer.sv")
    info(f"elaborated in ${(System.nanoTime() - t0) / 1e9}%.0f s; ${files.length} modules, ${files.map(_.length()).sum / 1e6}%.1f MB of SystemVerilog " +
      f"(Sequencer with the ${minicpm.generated.Microcode.program.length}-instruction program: ${seqSv.length() / 1e6}%.1f MB)")
  }
}
