package minicpm

import chisel3._
import chisel3.util._

/** One matmul / attention-matmul job. Addresses: activation in 256-bit words, weights in 2048-bit words
  * (tile-row groups), params in 1024-bit words, output in 256-bit words. */
class MatmulCmd extends Bundle {
  val rows      = UInt(13.W)   // M >= 1 (rows of this chunk; the sequencer chunks M > accRows)
  val kTiles    = UInt(8.W)    // KT >= 1 (<= 192)
  val nTiles    = UInt(12.W)   // NT >= 1 (<= 4080)
  // activation source
  val actSrc    = UInt(2.W)    // 0 = activation banks, 1 = attention P buffer
  val actBank   = UInt(3.W)
  val actBase   = UInt(20.W)   // 256-bit word address of (row 0, k-tile 0)
  val actStride = UInt(9.W)    // 256-bit words per row (<= 192)
  val actKOff   = UInt(8.W)    // k-tile offset (head select for attention)
  val actUnsigned = Bool()     // activations are uint8 (P.V passes)
  // weight source
  val wSrc      = UInt(2.W)    // 0 = weight store, 1 = K cache, 2 = V cache
  val wBase     = UInt(24.W)
  val wStrideN  = UInt(24.W)
  val wStrideK  = UInt(12.W)
  val wStrideG  = UInt(12.W)
  // output
  val outMode   = UInt(3.W)    // 0 int8, 1 int16, 2 raw int32 (attention), 3 wide (LM head), 4 int32 (static scale)
  val outSink   = UInt(2.W)    // 0 activation bank, 1 attention unit, 2 sampler, 3 KV cache
  val outBank   = UInt(3.W)
  val outBase   = UInt(20.W)
  val outStride = UInt(9.W)    // 256-bit words per row
  val outTag    = UInt(2.W)    // attention: 0 scores, 1 PV-hi, 2 PV-lo
  val rowBase   = UInt(13.W)   // logical row index of row 0 (rowfac lookup / sink row tag)
  val rowOff    = UInt(13.W)   // row chunk offset: rows [rowOff, rowOff+rows) of the tensor (addresses, rowfac)
  val outLocal  = Bool()       // output rows are chunk-local (address not offset by rowOff)
  // requant
  val s1        = UInt(6.W)
  val multBase  = UInt(16.W)   // param word address of mult for n-tile 0 (word nt = multBase + nt)
  val biasBase  = UInt(16.W)
  val hasBias   = Bool()
  val dynamic   = Bool()       // per-row factor from the rowfac table (else 1)
  val rowfacBank = UInt(3.W)
}

object MatmulMode {
  val Int8 = 0; val Int16 = 1; val Raw = 2; val Wide = 3; val Int32 = 4
}

class MatmulOut(cfg: MiniCPMConfig) extends Bundle {
  val sink   = UInt(2.W)
  val mode   = UInt(3.W)
  val bank   = UInt(3.W)
  val addr   = UInt(20.W)      // 256-bit word address (int8: 1 word, int16: 2 words, int32: 4 words starting here)
  val row    = UInt(13.W)      // logical row (rowBase + rowOff + m): rowfac index
  val trow   = UInt(13.W)      // tensor row (rowOff + m): KV key index relative to keyOff
  val nTile  = UInt(12.W)
  val tag    = UInt(2.W)
  val data   = Vec(32, SInt(32.W))   // int8/int16 packed in data(i) low bits; raw/wide: full int32
  val last   = Bool()          // last output of the job
}

/** A beat travelling down the array: reaches stage s `s` cycles after entering stage 0. */
class ArrayBeat(dataBits: Int, tagBits: Int) extends Bundle {
  val valid    = Bool()
  val swap     = Bool()         // activation: first row of a new tile -> use/commit the shadow weights
  val unsigned = Bool()
  val group    = UInt(2.W)      // load: which 8-row group (= stage) these 2048 bits are for
  val data     = UInt(dataBits.W)
  val tag      = UInt(tagBits.W)
}

class OutTag extends Bundle {
  val m       = UInt(13.W)
  val nTile   = UInt(12.W)
  val firstKt = Bool()
  val lastKt  = Bool()
  val lastJob = Bool()
}

/** 32 x 32 weight-stationary array, organised as R/rowsPerStage pipeline stages. Within a stage the
  * rowsPerStage MACs of a column are chained combinationally; partial sums flow down stage by stage.
  * Activations (256 bits = 32 int8) and weight-load beats (2048 bits = 8 rows x 32 cols) travel down
  * parallel register chains, one stage per cycle, so a load for the next tile can follow the current
  * tile's first row immediately (docs/tiling.md, decisions.md). */
class SystolicArray(cfg: MiniCPMConfig) extends Module {
  val R = cfg.rows; val C = cfg.cols; val RPS = cfg.rowsPerStage; val S = R / RPS
  require(R % RPS == 0 && RPS <= 8 && 8 % RPS == 0)
  val tagBits = (new OutTag).getWidth
  val io = IO(new Bundle {
    val act  = Input(new ArrayBeat(256, tagBits))
    val load = Input(new ArrayBeat(cfg.weightWordBits, tagBits))
    val out  = Output(Vec(C, SInt(32.W)))
    val outTag = Output(new ArrayBeat(256, tagBits))   // the act beat that produced `out` (data unused)
    val peekValid = Output(Vec(S, Bool()))              // act chain stage valid/tag, for look-ahead reads
    val peekTag = Output(Vec(S, UInt(tagBits.W)))
  })
  val actChain  = Seq.fill(S)(Reg(new ArrayBeat(256, tagBits)))
  val loadChain = Seq.fill(S)(Reg(new ArrayBeat(cfg.weightWordBits, tagBits)))
  val chainValid = RegInit(VecInit(Seq.fill(2 * S)(false.B)))   // reset-initialised valid bits
  for (s <- 0 until S) {
    actChain(s)  := (if (s == 0) io.act else actChain(s - 1))
    loadChain(s) := (if (s == 0) io.load else loadChain(s - 1))
    chainValid(s) := (if (s == 0) io.act.valid else chainValid(s - 1))
    chainValid(S + s) := (if (s == 0) io.load.valid else chainValid(S + s - 1))
  }
  val actView = Seq.tabulate(S) { s => val w = Wire(new ArrayBeat(256, tagBits)); w := actChain(s); w.valid := chainValid(s); w }
  val loadView = Seq.tabulate(S) { s => val w = Wire(new ArrayBeat(cfg.weightWordBits, tagBits)); w := loadChain(s); w.valid := chainValid(S + s); w }
  val psum  = Seq.fill(S)(Reg(Vec(C, SInt(32.W))))
  val w     = Seq.fill(R)(Reg(Vec(C, SInt(8.W))))
  val wNext = Seq.fill(R)(Reg(Vec(C, SInt(8.W))))
  for (s <- 0 until S) {
    val beat = actView(s)
    val ld = loadView(s)
    val doSwap = beat.valid && beat.swap
    val acc = Wire(Vec(C, SInt(32.W)))
    val psIn: Vec[SInt] = if (s == 0) VecInit(Seq.fill(C)(0.S(32.W))) else psum(s - 1)
    for (n <- 0 until C) {
      var t: SInt = psIn(n)
      for (r <- 0 until RPS) {
        val k = s * RPS + r
        val byteK = beat.data(8 * k + 7, 8 * k)
        val a: SInt = Mux(beat.unsigned, Cat(0.U(1.W), byteK).asSInt, byteK.asSInt)   // 9-bit signed
        val wSel = Mux(doSwap, wNext(k)(n), w(k)(n))
        t = (t + a * wSel)(31, 0).asSInt
      }
      acc(n) := t
    }
    psum(s) := acc
    for (r <- 0 until RPS) {
      val k = s * RPS + r
      val g = k / 8; val rr = k % 8
      when(ld.valid && ld.group === g.U) {
        for (n <- 0 until C) wNext(k)(n) := ld.data(8 * (rr * 32 + n) + 7, 8 * (rr * 32 + n)).asSInt
      }
      when(doSwap) { w(k) := wNext(k) }
    }
  }
  io.out := psum(S - 1)
  io.outTag := RegNext(actView(S - 1))
  io.outTag.valid := RegNext(chainValid(S - 1), false.B)
  for (s <- 0 until S) { io.peekValid(s) := chainValid(s); io.peekTag(s) := actChain(s).tag }
}

class MatmulEngine(cfg: MiniCPMConfig) extends Module {
  val R = cfg.rows; val C = cfg.cols; val S = cfg.rows / cfg.rowsPerStage
  require(S >= 4, "pipeline peeks assume >= 4 stages")
  val accRows = cfg.accRows
  val io = IO(new Bundle {
    val cmd  = Flipped(Decoupled(new MatmulCmd))
    val busy = Output(Bool())
    // activation read (256-bit words), 1-cycle latency
    val act = new Bundle {
      val src  = Output(UInt(2.W)); val bank = Output(UInt(3.W)); val addr = Output(UInt(20.W)); val en = Output(Bool())
      val data = Input(UInt(256.W))
    }
    // weight read (2048-bit words), 1-cycle latency
    val w = new Bundle {
      val src = Output(UInt(2.W)); val addr = Output(UInt(24.W)); val en = Output(Bool())
      val data = Input(UInt(cfg.weightWordBits.W))
    }
    // requant params (two 1024-bit read ports), 1-cycle latency
    val mult = new Bundle { val addr = Output(UInt(16.W)); val en = Output(Bool()); val data = Input(UInt(1024.W)) }
    val bias = new Bundle { val addr = Output(UInt(16.W)); val en = Output(Bool()); val data = Input(UInt(1024.W)) }
    // rowfac read, 1-cycle latency
    val rowfac = new Bundle { val bank = Output(UInt(3.W)); val addr = Output(UInt(13.W)); val en = Output(Bool()); val data = Input(UInt(RowFac.bits.W)) }
    val out = Decoupled(new MatmulOut(cfg))
    val cycles = Output(UInt(32.W))          // cycles spent busy (for utilisation measurement)
    val satCount = Output(UInt(32.W))        // debug: output lanes that saturated (int8/int16 modes)
  })
  val cmd = Reg(new MatmulCmd)
  val busy = RegInit(false.B)
  io.cmd.ready := !busy
  io.busy := busy
  val cycles = RegInit(0.U(32.W))
  when(busy) { cycles := cycles + 1.U }
  io.cycles := cycles

  // ------------------------------------------------------------------ load generator
  val ldN = Reg(UInt(12.W)); val ldK = Reg(UInt(8.W)); val ldG = Reg(UInt(2.W))
  val ldAddrN = Reg(UInt(24.W)); val ldAddrK = Reg(UInt(24.W))
  val ldDone = Reg(Bool())                       // all tiles' loads issued
  val ldTilesBegun = Reg(UInt(20.W))             // tiles whose loading has begun
  val ldTilesDone = Reg(UInt(20.W))              // tiles whose 4 beats have all been issued
  // ------------------------------------------------------------------ activation generator
  val acN = Reg(UInt(12.W)); val acK = Reg(UInt(8.W)); val acM = Reg(UInt(13.W))
  val acTilesStarted = Reg(UInt(20.W))
  val acDone = Reg(Bool())
  val acRowAddr = Reg(UInt(20.W))                          // actBase + (rowOff + m)*stride

  val canLoad = busy && !ldDone && (ldTilesBegun - acTilesStarted) <= 1.U
  val ldFire = canLoad
  val canAct = busy && !acDone && (ldTilesDone > acTilesStarted)

  val loadBeat = Wire(new ArrayBeat(cfg.weightWordBits, (new OutTag).getWidth))
  val actBeat = Wire(new ArrayBeat(256, (new OutTag).getWidth))
  // weight read issued this cycle, data next cycle -> the beat enters the array one cycle later
  io.w.en := ldFire
  io.w.src := cmd.wSrc
  io.w.addr := ldAddrN + ldAddrK + ldG * cmd.wStrideG
  val ldFireQ = RegNext(ldFire, false.B)
  val ldGQ = RegNext(ldG)
  loadBeat := 0.U.asTypeOf(loadBeat)
  loadBeat.valid := ldFireQ
  loadBeat.group := ldGQ
  loadBeat.data := io.w.data                // one 2048-bit word = one 8-row group of the tile

  when(ldFire) {
    ldG := ldG + 1.U
    when(ldG === 3.U) {
      ldTilesDone := ldTilesDone + 1.U
      when(ldK === cmd.kTiles - 1.U) {
        ldK := 0.U; ldAddrK := 0.U
        ldAddrN := ldAddrN + cmd.wStrideN
        when(ldN === cmd.nTiles - 1.U) { ldDone := true.B } .otherwise { ldN := ldN + 1.U; ldTilesBegun := ldTilesBegun + 1.U }
      } .otherwise { ldK := ldK + 1.U; ldAddrK := ldAddrK + cmd.wStrideK; ldTilesBegun := ldTilesBegun + 1.U }
    }
  }

  // activation address
  val actAddr = acRowAddr + cmd.actKOff + acK
  val acFire = canAct
  io.act.en := acFire
  io.act.src := cmd.actSrc
  io.act.bank := cmd.actBank
  io.act.addr := actAddr(19, 0)
  val acFireQ = RegNext(acFire, false.B)
  val acTagQ = Reg(new OutTag)
  val acSwapQ = RegNext(acM === 0.U)
  acTagQ.m := acM
  acTagQ.nTile := acN
  acTagQ.firstKt := acK === 0.U
  acTagQ.lastKt := acK === cmd.kTiles - 1.U
  acTagQ.lastJob := (acK === cmd.kTiles - 1.U) && (acN === cmd.nTiles - 1.U) && (acM === cmd.rows - 1.U)
  actBeat := 0.U.asTypeOf(actBeat)
  actBeat.valid := acFireQ
  actBeat.swap := acFireQ && acSwapQ
  actBeat.unsigned := cmd.actUnsigned
  actBeat.data := io.act.data
  actBeat.tag := acTagQ.asUInt

  when(acFire) {
    when(acM === cmd.rows - 1.U) {
      acM := 0.U; acRowAddr := cmd.actBase + cmd.rowOff * cmd.actStride
      acTilesStarted := acTilesStarted + 1.U
      when(acK === cmd.kTiles - 1.U) {
        acK := 0.U
        when(acN === cmd.nTiles - 1.U) { acDone := true.B } .otherwise { acN := acN + 1.U }
      } .otherwise { acK := acK + 1.U }
    } .otherwise {
      acM := acM + 1.U
      acRowAddr := acRowAddr + cmd.actStride
    }
  }

  // ------------------------------------------------------------------ array
  val arr = Module(new SystolicArray(cfg))
  arr.io.act := actBeat
  arr.io.load := loadBeat

  // ------------------------------------------------------------------ tag pipeline & accumulator
  val outBeat = arr.io.outTag
  val outTagB = outBeat.tag.asTypeOf(new OutTag)
  // acc read: issue when the beat is at stage S-2 (data arrives when it is at the bottom + 1 = outBeat)
  val peekValid = arr.io.peekValid(S - 2)
  val peekTag = arr.io.peekTag(S - 2).asTypeOf(new OutTag)
  val accMem = SyncReadMem(accRows, Vec(C, SInt(32.W)))
  val accRd = accMem.read(peekTag.m(log2Ceil(accRows) - 1, 0), peekValid && !peekTag.firstKt)
  val accRdQ = RegNext(accRd)                                     // aligned with outBeat (2 cycles after peek)
  // rowfac & params: issue reads at stage S-4 so values are ready at the bottom with margin
  val peek2Valid = arr.io.peekValid(S - 4)
  val peek2Tag = arr.io.peekTag(S - 4).asTypeOf(new OutTag)
  io.rowfac.en := peek2Valid && peek2Tag.lastKt && cmd.dynamic
  io.rowfac.bank := cmd.rowfacBank
  io.rowfac.addr := cmd.rowBase + cmd.rowOff + peek2Tag.m
  io.mult.en := peek2Valid && peek2Tag.lastKt
  io.mult.addr := cmd.multBase + peek2Tag.nTile
  io.bias.en := peek2Valid && peek2Tag.lastKt && cmd.hasBias
  io.bias.addr := cmd.biasBase + peek2Tag.nTile
  // row factor (m16, b): u = t * m16, y = rsr(u, s2 - b)  (static inputs: m16 = 1, b = 0)
  val rowfacQ = ShiftRegister(Mux(cmd.dynamic, io.rowfac.data(15, 0), 1.U(16.W)), 3)
  val rowfacBQ = ShiftRegister(Mux(cmd.dynamic, io.rowfac.data(20, 16), 0.U(5.W)), 3)
  val multQ = ShiftRegister(io.mult.data, 3)
  val biasQ = ShiftRegister(Mux(cmd.hasBias, io.bias.data, 0.U), 3)

  val sum = Wire(Vec(C, SInt(32.W)))
  for (n <- 0 until C) sum(n) := Mux(outTagB.firstKt, arr.io.out(n), (arr.io.out(n) + accRdQ(n))(31, 0).asSInt)
  when(outBeat.valid && !outTagB.lastKt) { accMem.write(outTagB.m(log2Ceil(accRows) - 1, 0), sum) }

  // ------------------------------------------------------------------ requant pipeline (4 stages)
  val s2 = Mux(cmd.outMode === MatmulMode.Int16.U, 20.U, 24.U)
  val isInt32 = cmd.outMode === MatmulMode.Int32.U
  // stage A
  val vA = RegNext(outBeat.valid && outTagB.lastKt, false.B)
  val tagA = RegNext(outTagB)
  val prodA = Reg(Vec(C, SInt(64.W)))
  val rowfacA = RegNext(rowfacQ); val rowfacBA = RegNext(rowfacBQ)
  val biasA = RegNext(biasQ)
  for (n <- 0 until C) prodA(n) := sum(n) * multQ(32 * n + 31, 32 * n).asSInt
  // stage B: t = sat48(rsr(prod, s1))
  val vB = RegNext(vA, false.B); val tagB = RegNext(tagA); val rowfacB = RegNext(rowfacA); val rowfacBB = RegNext(rowfacBA); val biasB = RegNext(biasA)
  val tB = Reg(Vec(C, SInt(48.W)))
  for (n <- 0 until C) {
    val rnd = Mux(cmd.s1 === 0.U, prodA(n), ((prodA(n) + (1.S(64.W) << (cmd.s1 - 1.U))) >> cmd.s1))
    tB(n) := Sat.sint(rnd, 48)
  }
  // stage C: u = t*m16 + (bias << (s2-16))   (int32 outputs: no bias term)
  val vC = RegNext(vB, false.B); val tagC = RegNext(tagB); val rowfacBC = RegNext(rowfacBB)
  val uC = Reg(Vec(C, SInt(66.W)))
  val tC = RegNext(tB)
  for (n <- 0 until C) {
    val b = biasB(32 * n + 31, 32 * n).asSInt
    val bTerm = Mux(isInt32, 0.S(41.W), (b << (s2 - 16.U))(40, 0).asSInt)
    uC(n) := tB(n) * Cat(0.U(1.W), rowfacB).asSInt + bTerm
  }
  // stage D: y = sat(rsr(u, s2 - b)) ; wide: sat32(t)
  val vD = RegNext(vC, false.B); val tagD = RegNext(tagC)
  val yD = Reg(Vec(C, SInt(32.W)))
  val satLane = Wire(Vec(C, Bool()))
  val satCount = RegInit(0.U(32.W))
  io.satCount := satCount
  val shD = s2 - rowfacBC
  for (n <- 0 until C) {
    val r = (uC(n) + (1.S(66.W) << (shD - 1.U))) >> shD
    val lim8 = r > 127.S || r < -128.S
    val lim16 = r > 32767.S || r < -32768.S
    val lim32 = r > (BigInt(2147483647)).S || r < (-BigInt(2147483648L)).S
    satLane(n) := vC && MuxLookup(cmd.outMode, false.B)(Seq(MatmulMode.Int8.U -> lim8, MatmulMode.Int16.U -> lim16, MatmulMode.Int32.U -> lim32))
    yD(n) := MuxLookup(cmd.outMode, Sat.sint(r, 8))(Seq(
      MatmulMode.Int8.U -> Sat.sint(r, 8),
      MatmulMode.Int16.U -> Sat.sint(r, 16),
      MatmulMode.Raw.U -> 0.S,          // raw handled below (bypasses requant)
      MatmulMode.Wide.U -> Sat.sint(tC(n), 32),
      MatmulMode.Int32.U -> Sat.sint(r, 32),
    ))
  }
  when(vC) { satCount := satCount + PopCount(satLane) }
  // raw mode: the accumulator sum itself, delayed to stage D for a single output timing
  val rawD = ShiftRegister(sum, 4)

  io.out.valid := vD
  io.out.bits.sink := cmd.outSink
  io.out.bits.mode := cmd.outMode
  io.out.bits.bank := cmd.outBank
  io.out.bits.addr := cmd.outBase + (Mux(cmd.outLocal, 0.U, cmd.rowOff) + tagD.m) * cmd.outStride +
    MuxLookup(cmd.outMode, tagD.nTile)(Seq(MatmulMode.Int16.U -> (tagD.nTile << 1), MatmulMode.Int32.U -> (tagD.nTile << 2)))
  io.out.bits.row := cmd.rowBase + cmd.rowOff + tagD.m
  io.out.bits.trow := cmd.rowOff + tagD.m
  io.out.bits.nTile := tagD.nTile
  io.out.bits.tag := cmd.outTag
  io.out.bits.data := Mux(cmd.outMode === MatmulMode.Raw.U, rawD, yD)
  io.out.bits.last := tagD.lastJob
  assert(!io.out.valid || io.out.ready, "MatmulEngine output sink must never stall")
  // Decoupled irrevocability on the command port (valid may not drop / bits may not change without a fire)
  val cmdVQ = RegNext(io.cmd.valid && !io.cmd.ready, false.B); val cmdBQ = RegNext(io.cmd.bits.asUInt)
  assert(!cmdVQ || (io.cmd.valid && io.cmd.bits.asUInt === cmdBQ), "MatmulCmd handshake is not irrevocable")
  assert(!io.cmd.fire || (io.cmd.bits.rows =/= 0.U && io.cmd.bits.rows <= accRows.U && io.cmd.bits.kTiles =/= 0.U && io.cmd.bits.nTiles =/= 0.U),
    "MatmulCmd rows/kTiles/nTiles out of range")

  // ------------------------------------------------------------------ job control
  val finish = vD && tagD.lastJob
  when(io.cmd.fire) {
    cmd := io.cmd.bits
    busy := true.B
    ldN := 0.U; ldK := 0.U; ldG := 0.U; ldAddrN := io.cmd.bits.wBase; ldAddrK := 0.U; ldDone := false.B
    ldTilesBegun := 1.U; ldTilesDone := 0.U
    acN := 0.U; acK := 0.U; acM := 0.U; acTilesStarted := 0.U; acDone := false.B
    acRowAddr := io.cmd.bits.actBase + io.cmd.bits.rowOff * io.cmd.bits.actStride
    cycles := 0.U
  }
  when(finish) { busy := false.B }
}

object Sat {
  def sint(x: SInt, bits: Int): SInt = {
    val max = ((BigInt(1) << (bits - 1)) - 1).S
    val min = (-(BigInt(1) << (bits - 1))).S
    Mux(x > max, max, Mux(x < min, min, x))(bits - 1, 0).asSInt
  }
}
