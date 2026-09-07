package whisper

import chisel3._
import chisel3.util._

/** Key/value caches feeding the engine's weight port (docs/tiling.md).
  *  K is stored transposed per (layer, head): a tiled matrix W[k=d (64)][n=key (keysMax)]
  *  V is stored per (layer, head): W[k=key (keysMax)][n=d (64)]
  * Regions (word bases) are fixed by the config; the sequencer passes bases in commands.
  * Writes come from engine int8 output beats (row = key, nTile = d-tile) through a 32-key transposer (K)
  * or directly (V), with byte masks so partial key groups are exact. */
object KVRegion {
  // keysMax per region kind
  def encKeys(cfg: WhisperConfig): Int = cfg.maxCtx + 36        // 1536 (multiple of 64)
  def decKeys(cfg: WhisperConfig): Int = cfg.maxTextCtx          // 448 (multiple of 64)
  def wordsPerHeadK(keys: Int): Int = 2 * (keys / 32) * 4        // KT=2 (d), NT=keys/32, 4 words per tile
  def wordsPerHeadV(keys: Int): Int = 2 * (keys / 32) * 4        // KT=keys/32, NT=2 (d), 4 words per tile
}

class KVCmd extends Bundle {
  val isK      = Bool()
  val base     = UInt(20.W)     // word base of head 0 of the target (layer) region
  val keysMax  = UInt(13.W)     // allocation (1536 or 448)
  val keyOff   = UInt(13.W)     // key index of engine row 0 (decoder: pos)
  val flush    = Bool()         // K: flush the transposer (end of job)
}

class KVCache(cfg: WhisperConfig) extends Module {
  val encKeys = KVRegion.encKeys(cfg); val decKeys = KVRegion.decKeys(cfg)
  val H = cfg.nHead
  val perHeadEnc = KVRegion.wordsPerHeadK(encKeys)   // 384 words
  val perHeadDec = KVRegion.wordsPerHeadK(decKeys)   // 112 words
  // regions: [enc self K][enc self V][dec self K x4][dec self V x4][cross K x4][cross V x4]
  val encSelfK = 0
  val encSelfV = encSelfK + H * perHeadEnc
  val decSelfK = encSelfV + H * perHeadEnc
  val decSelfV = decSelfK + 4 * H * perHeadDec
  val crossK = decSelfV + 4 * H * perHeadDec
  val crossV = crossK + 4 * H * perHeadEnc
  val words = crossV + 4 * H * perHeadEnc
  val addrBits = log2Ceil(words)
  val io = IO(new Bundle {
    val rd = new Bundle { val addr = Input(UInt(20.W)); val en = Input(Bool()); val data = Output(UInt(cfg.weightWordBits.W)) }
    val in = Flipped(Decoupled(new MatmulOut(cfg)))         // int8 beats: row = key, nTile = d-tile (0..11)
    val cmd = Input(new KVCmd)                              // static during a job
    val flush = Input(Bool())                                // pulse after the last K beat of a job
    val busy = Output(Bool())
    val dbgWr = Flipped(Valid(new Bundle { val addr = UInt(20.W); val data = UInt(cfg.weightWordBits.W) }))  // tests only
  })
  val mem = SyncReadMem(words, Vec(256, UInt(8.W)))
  io.rd.data := mem.read(io.rd.addr(addrBits - 1, 0), io.rd.en).asUInt
  val keysMaxT = io.cmd.keysMax >> 5                          // key tiles
  val perHead = keysMaxT * 8.U                                // words per head (both K and V)
  io.in.ready := true.B
  io.busy := false.B

  // ---- V: beat (key j, nTile nd) -> head h = nd/2, dTile = nd%2 ; word = base + h*perHead + (dTile*KT + j/32)*4 + (j%32)/8 ; bytes (j%8)*32..+31
  val j = io.in.bits.row + io.cmd.keyOff
  val nd = io.in.bits.nTile
  val h = nd >> 1
  val dTile = nd(0)
  val vWord = io.cmd.base + h * perHead + ((dTile * keysMaxT + (j >> 5)) << 2) + j(4, 3)
  val vData = Wire(Vec(256, UInt(8.W)))
  val vMask = Wire(Vec(256, Bool()))
  for (b <- 0 until 256) {
    val r = b / 32; val c = b % 32
    vData(b) := io.in.bits.data(c)(7, 0)
    vMask(b) := (r.U === j(2, 0))
  }
  // ---- K: transposer collects up to 32 consecutive keys of one nTile, emits 4 words (g = 0..3)
  val tBuf = Reg(Vec(32, Vec(32, UInt(8.W))))     // [keyInGroup][d]
  val tValid = RegInit(VecInit(Seq.fill(32)(false.B)))
  val tNd = Reg(UInt(12.W)); val tKeyGroup = Reg(UInt(13.W))   // key/32 of the buffered group
  val tAny = tValid.asUInt.orR
  val kFlushing = RegInit(false.B); val kG = Reg(UInt(2.W))
  val doFlushNow = Wire(Bool())
  // K word for group g: base + h*perHead + (keyGroup*2 + dTile)*4 + g ; byte (r*32 + c) = K[key=keyGroup*32+c][d = dTile*32 + g*8 + r]
  val kh = tNd >> 1; val kdTile = tNd(0)
  val kWord = io.cmd.base + kh * perHead + (((tKeyGroup << 1) + kdTile) << 2) + kG
  val kData = Wire(Vec(256, UInt(8.W))); val kMask = Wire(Vec(256, Bool()))
  for (b <- 0 until 256) {
    val r = b / 32; val c = b % 32
    kData(b) := VecInit(Seq.tabulate(4)(g => tBuf(c)(g * 8 + r)))(kG)
    kMask(b) := tValid(c)
  }
  val newGroup = io.in.valid && io.cmd.isK && (!tAny || (nd =/= tNd) || ((j >> 5) =/= tKeyGroup))
  // a beat that starts a new group while the buffer holds data forces a flush first (stall the beat)
  io.in.ready := !(io.cmd.isK && kFlushing) && !(newGroup && tAny)
  doFlushNow := (io.cmd.isK && ((newGroup && tAny) || io.flush) && tAny && !kFlushing)
  when(doFlushNow) { kFlushing := true.B; kG := 0.U }
  when(kFlushing) {
    mem.write(kWord(addrBits - 1, 0), kData, kMask)
    kG := kG + 1.U
    when(kG === 3.U) { kFlushing := false.B; tValid.foreach(_ := false.B) }
  }
  io.busy := kFlushing || (io.cmd.isK && tAny)
  when(io.in.fire && io.cmd.isK) {
    when(!tAny) { tNd := nd; tKeyGroup := j >> 5 }
    for (d <- 0 until 32) tBuf(j(4, 0))(d) := io.in.bits.data(d)(7, 0)
    tValid(j(4, 0)) := true.B
  }
  when(io.in.fire && !io.cmd.isK) { mem.write(vWord(addrBits - 1, 0), vData, vMask) }
  when(io.dbgWr.valid) {
    mem.write(io.dbgWr.bits.addr(addrBits - 1, 0), VecInit(Seq.tabulate(256)(b => io.dbgWr.bits.data(8 * b + 7, 8 * b))))
  }
}
