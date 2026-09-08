package minicpm

import minicpm.generated.WeightTables

sealed trait WeightBackend
case object RomLiteral extends WeightBackend   // VecInit constant ROM (ASIC flow, small banks)
case object RomInit extends WeightBackend      // SyncReadMem + $readmemh from the generated weight images (Verilator default)
case object Sram extends WeightBackend         // writable SyncReadMem with a load port (FPGA / off-chip weights)

/** Generator parameters. Everything that sizes memories or selects a datapath option lives here.
  * Defaults describe openbmb/MiniCPM5-2B (a 42-layer Llama). */
case class MiniCPMConfig(
    rows: Int = 32,                       // systolic array rows (k)
    cols: Int = 32,                       // systolic array cols (n)
    rowsPerStage: Int = 8,                // MAC rows chained combinationally per pipeline stage
    weightBackend: WeightBackend = RomInit,
    /** Subset of tensor names to instantiate (None = all 2.5 GB). Unit tests elaborate one layer's tensors. */
    weightTensors: Option[Seq[String]] = None,
    /** Literal ROM banks are refused above this size (elaboration/Verilog blow-up guard). */
    romLiteralMaxBits: Int = 4 * 1024 * 1024,
    residualBits: Int = 16,
    dModel: Int = 2048,
    nHead: Int = 16,
    nKvHead: Int = 2,
    headDim: Int = 128,
    dFF: Int = 6144,
    nVocab: Int = 130560,
    nLayers: Int = 42,
    maxCtx: Int = 2048,                   // positions: KV cache keys per layer, RoPE table rows, token buffer
    kvLayers: Int = 42,                   // layers with a KV region (tests instantiate 1)
    chunkRows: Int = 512,                 // prefill rows per chunk (sizes the chunk-local banks)
    accRows: Int = 1536,                  // accumulator rows per matmul job
    repoRoot: String = MiniCPMConfig.defaultRepoRoot,
) {
  require(rows == 32 && cols == 32, "v1 tiling assumes 32x32 tiles (see docs/tiling.md)")
  require(headDim % 32 == 0 && maxCtx % 64 == 0 && chunkRows <= accRows)
  val kvDim: Int = nKvHead * headDim                    // 256
  val headTiles: Int = headDim / 32                     // 4 (d-tiles per head)
  val group: Int = nHead / nKvHead                      // 8 query heads per kv head
  val weightWordBits: Int = 8 * rows * 8                // 8 tile rows x 32 int8 = 2048
  val paramWordBits: Int = 32 * cols                    // 32 x int32 = 1024
  val actWordBits: Int = 8 * cols                       // 32 x int8 = 256
  val tileWords: Int = rows / 8                         // words per 32x32 tile = 4
  val wAddrBits: Int = 24                               // w8 space (2048-bit words), up to 16 M words
  val tAddrBits: Int = 24                               // i8mat/i16mat space (256-bit words)
  val pAddrBits: Int = 16                               // i32vec space (1024-bit words)

  def tensors: Seq[WeightTensor] = weightTensors match {
    case None      => WeightTables.tensors
    case Some(sel) => WeightTables.tensors.filter(t => sel.exists(p => t.name == p || t.name.startsWith(p + ".")))
  }
}

object MiniCPMConfig {
  def defaultRepoRoot: String = sys.env.getOrElse("MINICPM_SI_ROOT", new java.io.File(".").getCanonicalPath)
}

/** Activation bank map (256-bit words). Mirrors gen/chipmap.py; the generated microcode carries the same
  * numbers and MiniCPMTop checks them at elaboration. */
object ChipMap {
  def bankWords(cfg: MiniCPMConfig): Seq[Int] = {
    val d32 = cfg.dModel / 8; val d16 = cfg.dModel / 16; val d8 = cfg.dModel / 32; val ff16 = cfg.dFF / 16; val ff8 = cfg.dFF / 32
    val kv16 = cfg.kvDim / 16
    Seq(
      d32 * cfg.maxCtx,                        // 0 X    int32 residual, all positions
      d8 * cfg.chunkRows,                      // 1 A8   int8 norm / attention-out rows (chunk-local)
      d16 * cfg.chunkRows + kv16 * cfg.chunkRows, // 2 QK16 int16 q rows, then k rows at K16_BASE
      d8 * cfg.chunkRows,                      // 3 Q8   int8 rotated q rows
      d32 * cfg.chunkRows,                     // 4 T32  scratch (int16 attention out / int32 o out, down out)
      ff16 * cfg.chunkRows,                    // 5 G16  int16 gate rows
      ff16 * cfg.chunkRows,                    // 6 U16  int16 up rows
      ff8 * cfg.chunkRows,                     // 7 A8X  int8 gated rows (down input)
    )
  }
  def k16Base(cfg: MiniCPMConfig): Int = (cfg.dModel / 16) * cfg.chunkRows
  def rowfacBanks: Seq[Int] = Seq(1, 7)
}
