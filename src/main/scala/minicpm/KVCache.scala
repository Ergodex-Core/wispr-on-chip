package minicpm

import chisel3._
import chisel3.util._

/** Key/value caches feeding the engine's weight port (docs/tiling.md). Per (layer, kv head):
  *  K is stored transposed: a tiled matrix W[k=d (128)][n=key (keysMax)]
  *  V is stored key-major:  W[k=key (keysMax)][n=d (128)]
  * Regions are fixed by the config: [L0 K][L0 V][L1 K][L1 V]...; the sequencer passes bases in commands.
  * Writes arrive as int8 output beats (row = key, nTile = kv-head * 4 + d-tile) from the engine (V) or the
  * vector unit's RoPE op (K) through a 32-key transposer (one slot per d-tile, so beats may be d-tile-major
  * or row-major), with byte masks so partial key groups are exact. */
object KVRegion {
  def keysMax(cfg: MiniCPMConfig): Int = cfg.maxCtx
  /** words per (layer, head) for K and for V: KT x NT x 4 with one of them = keysMax/32, the other headTiles */
  def wordsPerHead(cfg: MiniCPMConfig): Int = cfg.headTiles * (cfg.maxCtx / 32) * 4
  def kBase(cfg: MiniCPMConfig, layer: Int): Int = layer * 2 * cfg.nKvHead * wordsPerHead(cfg)
  def vBase(cfg: MiniCPMConfig, layer: Int): Int = kBase(cfg, layer) + cfg.nKvHead * wordsPerHead(cfg)
  def words(cfg: MiniCPMConfig): Int = kBase(cfg, cfg.kvLayers)
}

class KVCmd extends Bundle {
  val isK      = Bool()
  val base     = UInt(20.W)     // word base of kv head 0 of the target (layer) region
  val keysMax  = UInt(13.W)     // allocation (maxCtx)
  val keyOff   = UInt(13.W)     // key index of tensor row 0 (chunk base / decode position)
  val flush    = Bool()         // K: flush the transposer (end of job)
}

class KVCache(cfg: MiniCPMConfig, debugPort: Boolean = false) extends Module {
  val HT = cfg.headTiles
  require(HT == 4, "KV layout assumes head_dim 128 (4 d-tiles)")
  val words = KVRegion.words(cfg)
  val addrBits = log2Ceil(words)
  val io = IO(new Bundle {
    val rd = new Bundle { val addr = Input(UInt(24.W)); val en = Input(Bool()); val data = Output(UInt(cfg.weightWordBits.W)) }
    val in = Flipped(Decoupled(new MatmulOut(cfg)))         // int8 beats: row = key, nTile = kvh*4 + d-tile
    val cmd = Input(new KVCmd)                              // static during a job
    val flush = Input(Bool())                                // pulse after the last K beat of a job
    val busy = Output(Bool())
    // tests only (debugPort): a masked write port whose mask comes from IO (a constant all-true mask would make
    // firtool drop the masks of every port of this memory)
    val dbgWr = if (debugPort) Some(Flipped(Valid(new Bundle { val addr = UInt(24.W); val data = UInt(cfg.weightWordBits.W); val mask = UInt(256.W) }))) else None
  })
  val mem = SyncReadMem(words, Vec(256, UInt(8.W)))
  io.rd.data := mem.read(io.rd.addr(addrBits - 1, 0), io.rd.en).asUInt
  val keysMaxT = io.cmd.keysMax >> 5                          // key tiles
  val perHead = keysMaxT << 4                                 // words per head (both K and V) = keysMaxT * HT * 4
  io.in.ready := true.B
  io.busy := false.B

  // ---- V: beat (key j, nTile nd) -> head h = nd/4, dTile = nd%4 ; word = base + h*perHead + (dTile*keysMaxT + j/32)*4 + (j%32)/8 ; bytes (j%8)*32..+31
  val j = io.in.bits.trow + io.cmd.keyOff
  val nd = io.in.bits.nTile
  val h = nd >> 2
  val dTile = nd(1, 0)
  val vWord = io.cmd.base + h * perHead + ((dTile * keysMaxT + (j >> 5)) << 2) + j(4, 3)
  val vData = Wire(Vec(256, UInt(8.W)))
  val vMask = Wire(Vec(256, Bool()))
  for (b <- 0 until 256) {
    val r = b / 32; val c = b % 32
    vData(b) := io.in.bits.data(c)(7, 0)
    vMask(b) := (r.U === j(2, 0))
  }
  // ---- K: double-buffered transposer. A buffer holds one 32-key group for every d-tile slot (nTile 0..7):
  // beats may arrive d-tile-major (engine order) or row-major (vector-unit RoPE order). When the key group
  // changes, the full buffer is flushed slot by slot (4 words per used slot) while the other one collects.
  val NS = cfg.nKvHead * HT                                // 8 slots
  val tBuf = Reg(Vec(2, Vec(NS, Vec(32, Vec(32, UInt(8.W))))))   // [buf][slot][keyInGroup][d]
  val tValid = RegInit(VecInit(Seq.fill(2)(VecInit(Seq.fill(NS)(VecInit(Seq.fill(32)(false.B)))))))
  val tKeyGroup = Reg(Vec(2, UInt(13.W)))
  val cur = RegInit(0.U(1.W))
  val tSlotAny = VecInit(tValid.map(b => VecInit(b.map(_.asUInt.orR))))   // [buf][slot]
  val tAny = VecInit(tSlotAny.map(_.asUInt.orR))
  val kFlushing = RegInit(false.B); val kG = Reg(UInt(2.W)); val fBuf = Reg(UInt(1.W)); val fSlot = Reg(UInt(4.W))
  val kh = fSlot(3, 2); val kdTile = fSlot(1, 0)
  // K^T tile (kt = dTile, nt = keyGroup): word = (nt*HT + kt)*4 + g
  val kWord = io.cmd.base + kh * perHead + (((tKeyGroup(fBuf) << 2) + kdTile) << 2) + kG
  val kData = Wire(Vec(256, UInt(8.W))); val kMask = Wire(Vec(256, Bool()))
  val slotSel = fSlot(2, 0)
  for (b <- 0 until 256) {
    val r = b / 32; val c = b % 32
    kData(b) := VecInit(Seq.tabulate(4)(g => tBuf(fBuf)(slotSel)(c)(g * 8 + r)))(kG)
    kMask(b) := tValid(fBuf)(slotSel)(c)
  }
  val newGroup = io.in.valid && io.cmd.isK && tAny(cur) && ((j >> 5) =/= tKeyGroup(cur))
  val flushPend = RegInit(false.B)             // explicit end-of-job flush waits for a running flush
  when(io.flush && io.cmd.isK) { flushPend := true.B }
  val explicit = (flushPend || (io.flush && io.cmd.isK)) && !kFlushing && !newGroup
  val startFlush = io.cmd.isK && (newGroup || (explicit && tAny(cur)))
  when(explicit) { flushPend := false.B }
  assert(!(newGroup && kFlushing), "KV transposer: new key group while the other buffer is still flushing")
  assert(!kFlushing || !tSlotAny(fBuf)(slotSel) || kWord < words.U, "KV cache K write address out of range")
  assert(!(io.in.fire && !io.cmd.isK) || vWord < words.U, "KV cache V write address out of range")
  assert(!io.rd.en || io.rd.addr < words.U, "KV cache read address out of range")
  when(kFlushing) {
    when(tSlotAny(fBuf)(slotSel)) {
      mem.write(kWord(addrBits - 1, 0), kData, kMask.toSeq)
      kG := kG + 1.U
      when(kG === 3.U) { fSlot := fSlot + 1.U }
    }.otherwise { fSlot := fSlot + 1.U }
    when(fSlot === (NS - 1).U && (!tSlotAny(fBuf)(slotSel) || kG === 3.U)) {
      kFlushing := false.B
      for (sl <- 0 until NS) tValid(fBuf)(sl).foreach(_ := false.B)
    }
  }
  when(startFlush) { kFlushing := true.B; kG := 0.U; fSlot := 0.U; fBuf := cur; cur := ~cur }
  io.busy := kFlushing || flushPend || (io.cmd.isK && (tAny(0) || tAny(1)))
  when(io.in.fire && io.cmd.isK) {
    val b = Mux(newGroup, ~cur, cur)          // a new group goes to the other buffer (this cycle's flush takes `cur`)
    when(newGroup || !tAny(cur)) { tKeyGroup(b) := j >> 5 }
    for (d <- 0 until 32) tBuf(b)(nd(2, 0))(j(4, 0))(d) := io.in.bits.data(d)(7, 0)
    tValid(b)(nd(2, 0))(j(4, 0)) := true.B
  }
  when(io.in.fire && !io.cmd.isK) { mem.write(vWord(addrBits - 1, 0), vData, vMask.toSeq) }
  io.dbgWr.foreach { d =>
    when(d.valid) {
      mem.write(d.bits.addr(addrBits - 1, 0), VecInit(Seq.tabulate(256)(b => d.bits.data(8 * b + 7, 8 * b))), d.bits.mask.asBools)
    }
  }
}
