package whisper.synth

import chisel3._
import circt.stage.ChiselStage
import whisper._

/** Emits Verilog-2001-compatible SystemVerilog (no packed arrays / local variables) for synthesis
  * with Yosys: `sbt "Test/runMain whisper.synth.EmitSynth <outdir>"`. Memories become separate modules. */
object EmitSynth {
  def main(args: Array[String]): Unit = {
    val out = args.headOption.getOrElse("build/synth-rtl")
    val cfg = WhisperConfig()
    ChiselStage.emitSystemVerilogFile(
      new WhisperTop(cfg),
      firtoolOpts = Array(
        "--lowering-options=disallowPackedArrays,disallowLocalVariables,disallowExpressionInliningInPorts,locationInfoStyle=none",
        "-disable-all-randomization", "-strip-debug-info", "--split-verilog", "-o", out))
    println(s"emitted to $out")
  }
}
