package minicpm.sim

import chisel3.simulator.{FirtoolOptionsModifications, HasSimulator}
import org.scalatest.TestSuite
import svsim.CommonCompilationSettings
import svsim.CommonCompilationSettings.{AvailableParallelism, OptimizationStyle, VerilogPreprocessorDefine}
import svsim.verilator.Backend.CompilationSettings

/** Verilator settings shared by every spec: enable `$readmemh` initial blocks (RomInit), optimise for
  * simulation speed, use the machine's cores for the C++ compile, and multi-thread the model when asked. */
object VerilatorSettings {
  def threads: Int = sys.env.get("MINICPM_SIM_THREADS").map(_.toInt).getOrElse(1)
  def fast: Boolean = sys.env.get("MINICPM_SIM_FAST").forall(_ != "0")

  def common: CommonCompilationSettings = CommonCompilationSettings().copy(
    verilogPreprocessorDefines = Seq(VerilogPreprocessorDefine("ENABLE_INITIAL_MEM_")),
    optimizationStyle = if (fast) OptimizationStyle.OptimizeForSimulationSpeed else OptimizationStyle.Default,
    availableParallelism = AvailableParallelism.UpTo(Runtime.getRuntime.availableProcessors()),
  )
  def verilator: CompilationSettings = {
    val base = CompilationSettings().copy(
      disabledWarnings = Seq("WIDTH", "UNOPTFLAT"),
      disableFatalExitOnWarnings = true,
    )
    val par: Option[CompilationSettings.Parallelism.Type] =
      if (threads > 1) Some(CompilationSettings.Parallelism.Uniform.default.withNum(threads)) else None
    base.withParallelism(par)
  }
}

/** Mix into scalatest specs: `simulate(new Dut) { dut => ... }` with the settings above. */
trait MiniCPMSim extends chisel3.simulator.scalatest.ChiselSim { self: TestSuite =>
  implicit val minicpmHasSimulator: HasSimulator =
    HasSimulator.simulators.verilator(VerilatorSettings.common, VerilatorSettings.verilator)
  /** $readmemh-initialised ROMs must not be overwritten by firtool's memory-randomisation loop. */
  implicit val minicpmFirtoolOpts: FirtoolOptionsModifications =
    (opts: Array[String]) => opts ++ Array("-disable-mem-randomization")
}
