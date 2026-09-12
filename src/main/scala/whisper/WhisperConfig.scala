package whisper

import whisper.generated.WeightTables

sealed trait WeightBackend
case object RomLiteral extends WeightBackend   // VecInit constant ROM (ASIC flow, small banks)
case object RomInit extends WeightBackend      // SyncReadMem + $readmemh from the committed hex (Verilator default)
case object Sram extends WeightBackend         // writable SyncReadMem with a load port (FPGA / weight swap)

/** Generator parameters. Everything that sizes memories or selects a datapath option lives here. */
case class WhisperConfig(
    rows: Int = 32,                       // systolic array rows (k)
    cols: Int = 32,                       // systolic array cols (n)
    rowsPerStage: Int = 8,                // MAC rows chained combinationally per pipeline stage
    weightBackend: WeightBackend = RomInit,
    /** Subset of tensor names to instantiate (None = all). Unit tests elaborate one layer's tensors. */
    weightTensors: Option[Seq[String]] = None,
    /** Literal ROM banks are refused above this size (elaboration/Verilog blow-up guard). */
    romLiteralMaxBits: Int = 4 * 1024 * 1024,
    residualBits: Int = 16,
    maxFrames: Int = 3000,                // mel frames (30 s)
    maxCtx: Int = 1500,                   // encoder positions
    maxTextCtx: Int = 448,                // decoder positions
    dModel: Int = 384,
    nHead: Int = 6,
    dFF: Int = 1536,
    nVocabPad: Int = 51872,
    accRows: Int = 1536,                  // accumulator rows per matmul chunk
    fastSim: Boolean = false,
    /** repo root, used to locate weights/ at elaboration time */
    repoRoot: String = WhisperConfig.defaultRepoRoot,
) {
  require(rows == 32 && cols == 32, "v1 tiling assumes 32x32 tiles (see docs/tiling.md)")
  val headDim: Int = dModel / nHead
  val weightWordBits: Int = 8 * rows * 8                // 8 tile rows x 32 int8 = 2048
  val paramWordBits: Int = 32 * cols                    // 32 x int32 = 1024
  val actWordBits: Int = 8 * cols                       // 32 x int8 = 256
  val tileWords: Int = rows / 8                         // words per 32x32 tile = 4

  def tensors: Seq[WeightTensor] = weightTensors match {
    case None      => WeightTables.tensors
    case Some(sel) => WeightTables.tensors.filter(t => sel.exists(p => t.name == p || t.name.startsWith(p + ".")))
  }
}

object WhisperConfig {
  def defaultRepoRoot: String = sys.env.getOrElse("WHISPER_SI_ROOT", new java.io.File(".").getCanonicalPath)
}
