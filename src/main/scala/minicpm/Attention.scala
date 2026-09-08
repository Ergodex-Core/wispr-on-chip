package minicpm

import chisel3._
import chisel3.util._
import minicpm.generated.Luts
import minicpm.util.PipeDivider

/** One attention op: all 16 query heads (kv head = h / 8), all query blocks of 64, key tiles of 64 with
  * online softmax (docs/numerics.md "Attention"). Drives the MatmulEngine for Q·Kᵀ (4 k-tiles of d) and
  * P·V (4 n-tiles of d). */
class AttnCmd extends Bundle {
  val nQueries = UInt(13.W)
  val nKeys    = UInt(13.W)
  val causal   = Bool()
  val qPos0    = UInt(13.W)    // causal: key j valid iff j <= qPos0 + m
  val qBank    = UInt(3.W); val qBase = UInt(20.W); val qStride = UInt(9.W)      // int8 rows (64 words)
  val kBase    = UInt(20.W); val vBase = UInt(20.W); val keysMax = UInt(13.W)    // KV cache regions (kv head 0)
  val outBank  = UInt(3.W); val outBase = UInt(20.W); val outStride = UInt(9.W)  // int16 rows (128 words)
  val mq       = Vec(16, UInt(32.W))
  val sq       = Vec(16, UInt(6.W))
}

class Attention(cfg: MiniCPMConfig) extends Module {
  val QB = 64; val KT = 64; val LANES = 16
  val HT = cfg.headTiles; val H = cfg.nHead
  require(HT == 4 && H == 16 && cfg.group == 8)
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new AttnCmd))
    val busy = Output(Bool())
    val eng = Decoupled(new MatmulCmd)                         // to the engine (arbitrated at top)
    val engBusy = Input(Bool())
    val in = Flipped(Decoupled(new MatmulOut(cfg)))            // raw beats tagged for us
    val pRd = new Bundle { val addr = Input(UInt(20.W)); val en = Input(Bool()); val data = Output(UInt(256.W)) } // P buffer (engine act src 1)
    val wr = Valid(new Bundle { val bank = UInt(3.W); val addr = UInt(20.W); val data = UInt(256.W) })
    val cycles = Output(UInt(32.W))
  })
  val cmd = Reg(new AttnCmd)
  val busy = RegInit(false.B)
  io.cmd.ready := !busy
  io.busy := busy
  val cycles = RegInit(0.U(32.W)); when(busy) { cycles := cycles + 1.U }; io.cycles := cycles

  // memories
  val scoreMem = SyncReadMem(2 * QB, Vec(32, SInt(32.W)))        // [m*2 + half]
  val pMem = SyncReadMem(4 * QB, UInt(256.W))                      // [hi: m*2+kt | lo: 128 + m*2+kt]
  val oMem = SyncReadMem(HT * QB, Vec(32, SInt(36.W)))             // [m*4 + dTile]
  val mx = Reg(Vec(QB, SInt(32.W)))
  val lSum = Reg(Vec(QB, UInt(28.W)))
  val started = Reg(Vec(QB, Bool()))
  val rescale = Reg(Vec(QB, Bool()))
  val expRom = VecInit((Luts.Exp :+ 0).map(_.U(16.W)))
  def expLut(d: UInt): UInt = {   // d in [0, 4095]
    val i = d(11, 4); val f = d(3, 0)
    val hi = expRom(i); val lo = expRom(i +& 1.U)
    hi - (((hi - lo) * f + 8.U) >> 4)
  }
  def rsrU(x: UInt, s: UInt): UInt = Mux(s === 0.U, x, (x + (1.U << (s - 1.U))) >> s)

  // ---- loop counters
  val h = Reg(UInt(4.W)); val qb = Reg(UInt(7.W)); val kt = Reg(UInt(7.W))
  val kvh = h >> 3
  val nQB = (cmd.nQueries + (QB - 1).U) >> 6
  val nKT = (cmd.nKeys + (KT - 1).U) >> 6
  val qRows = Mux(qb === nQB - 1.U, cmd.nQueries - (qb << 6), QB.U)     // rows in this block (1..64)
  val q0 = qb << 6
  val mqH = cmd.mq(h); val sqH = cmd.sq(h)
  val keysMaxT = cmd.keysMax >> 5
  val perHead = keysMaxT << 4                                             // keysMaxT * HT * 4 words

  // ---- states
  val sIdle :: sIssueS :: sWaitS :: sSoftmax :: sIssuePVhi :: sWaitPVhi :: sIssuePVlo :: sWaitPVlo :: sNextKt :: sFinal :: sNextQB :: Nil = Enum(11)
  val state = RegInit(sIdle)

  // ---- engine command construction
  val ec = Wire(new MatmulCmd); ec := 0.U.asTypeOf(ec)
  ec.rows := qRows
  val isPV = state === sIssuePVhi || state === sIssuePVlo
  when(!isPV) {
    // S = Q · Kᵀ : K^T tile (kt = dTile, nt = key tile): word (nt*HT + kt)*4 + g ; 64-key tile = 2 n-tiles = 32 words
    ec.kTiles := HT.U; ec.nTiles := 2.U
    ec.actSrc := 0.U; ec.actBank := cmd.qBank; ec.actBase := cmd.qBase + q0 * cmd.qStride; ec.actStride := cmd.qStride; ec.actKOff := h << 2
    ec.wSrc := 1.U; ec.wBase := cmd.kBase + kvh * perHead + (kt << 5)
    ec.wStrideN := (HT * 4).U; ec.wStrideK := 4.U; ec.wStrideG := 1.U
    ec.outMode := MatmulMode.Raw.U; ec.outSink := 1.U; ec.outTag := 0.U; ec.outStride := 2.U; ec.outBase := 0.U
  }.otherwise {
    // O += P · V : V tile (kt = key tile, nt = dTile): word (nt*keysMaxT + kt)*4 + g ; 64-key tile = 2 k-tiles = 8 words
    ec.kTiles := 2.U; ec.nTiles := HT.U
    ec.actSrc := 1.U; ec.actBase := Mux(state === sIssuePVhi, 0.U, 128.U); ec.actStride := 2.U; ec.actKOff := 0.U; ec.actUnsigned := true.B
    ec.wSrc := 2.U; ec.wBase := cmd.vBase + kvh * perHead + (kt << 3)
    ec.wStrideN := keysMaxT << 2; ec.wStrideK := 4.U; ec.wStrideG := 1.U
    ec.outMode := MatmulMode.Raw.U; ec.outSink := 1.U; ec.outTag := Mux(state === sIssuePVhi, 1.U, 2.U); ec.outStride := HT.U; ec.outBase := 0.U
  }
  io.eng.bits := ec
  io.eng.valid := state === sIssueS || isPV

  // ---- incoming raw beats
  io.in.ready := true.B
  val inB = io.in.bits
  val beatsExpected = Reg(UInt(9.W)); val beatsSeen = Reg(UInt(9.W))
  when(io.in.fire) { beatsSeen := beatsSeen + 1.U }
  when(io.in.fire && inB.tag === 0.U) {           // scores: [m*2 + nTile]
    scoreMem.write((inB.row(5, 0) << 1) + inB.nTile(0), VecInit(inB.data.map(_(31, 0).asSInt)))
  }
  // PV beats: O RMW (2-cycle read) at [m*4 + dTile]
  val pvV1 = RegNext(io.in.fire && inB.tag =/= 0.U, false.B)
  val pvAddr1 = RegNext((inB.row(5, 0) << 2) + inB.nTile(1, 0)); val pvData1 = RegNext(inB.data); val pvHi1 = RegNext(inB.tag === 1.U)
  val oRdPV = oMem.read((inB.row(5, 0) << 2) + inB.nTile(1, 0), io.in.fire && inB.tag =/= 0.U)
  val pvV2 = RegNext(pvV1, false.B); val pvAddr2 = RegNext(pvAddr1); val pvData2 = RegNext(pvData1); val pvHi2 = RegNext(pvHi1)
  val oRdPV2 = RegNext(oRdPV)
  val firstKt = RegNext(RegNext(kt === 0.U))     // O starts at zero on the first key tile (hi pass adds to 0)
  when(pvV2) {
    val upd = Wire(Vec(32, SInt(36.W)))
    for (i <- 0 until 32) {
      val add = Mux(pvHi2, (pvData2(i) << 8).asSInt, pvData2(i)).pad(36)
      val base = Mux(firstKt && pvHi2, 0.S(36.W), oRdPV2(i))
      upd(i) := (base + add)(35, 0).asSInt
    }
    oMem.write(pvAddr2, upd)
  }

  // ---- softmax pass state
  val sm = Reg(UInt(7.W))          // query index within block
  val smStep = Reg(UInt(4.W))
  val zeroSecond = RegInit(false.B)  // skipped query: second P word still to be zeroed
  val srow = Reg(Vec(64, SInt(32.W)))
  val j0 = kt << 6
  val keyValid = Wire(Vec(64, Bool()))
  for (jj <- 0 until 64) {
    val key = j0 + jj.U
    keyValid(jj) := key < cmd.nKeys && (!cmd.causal || key <= cmd.qPos0 + q0 + sm)
  }
  val anyValid = keyValid.asUInt.orR
  val smRd = scoreMem.read((sm << 1) + smStep(0), state === sSoftmax && (smStep === 0.U || smStep === 1.U))
  // max tree over valid entries
  val masked = VecInit((0 until 64).map(jj => Mux(keyValid(jj), srow(jj), (-(BigInt(1) << 31)).S(32.W))))
  val mt = masked.reduceTree((a, b) => Mux(a > b, a, b))
  val mnew = Reg(SInt(32.W)); val alpha = Reg(UInt(17.W)); val grew = Reg(Bool())
  val pSum = Reg(UInt(28.W))
  val hiWord = Reg(UInt(128.W)); val loWord = Reg(UInt(128.W))
  // exp lanes: 16 keys per cycle (smStep 4..7 -> lane group 0..3)
  val lg = smStep - 4.U
  val pLanes = Wire(Vec(LANES, UInt(16.W)))
  for (i <- 0 until LANES) {
    val jj = (lg << 4) + i.U
    val s = srow(jj)
    val diff = (mx(sm) - s).asUInt                       // >= 0 for valid keys
    val dFull = rsrU(diff * mqH, sqH)
    val d = Mux(dFull > 4095.U, 4095.U, dFull(11, 0))
    pLanes(i) := Mux(keyValid(jj), expLut(d), 0.U)
  }
  val pLaneSum = pLanes.map(_.pad(28)).reduce(_ +& _)
  val pHiBytes = Cat(pLanes.reverse.map(p => p(15, 8)))
  val pLoBytes = Cat(pLanes.reverse.map(p => p(7, 0)))
  val pWr = Wire(Bool()); pWr := false.B
  val pWrAddr = Wire(UInt(9.W)); pWrAddr := 0.U
  val pWrHi = Wire(UInt(256.W)); pWrHi := 0.U
  val pWrLo = Wire(UInt(256.W)); pWrLo := 0.U
  when(pWr) {
    pMem.write(pWrAddr, pWrHi)
    pMem.write(pWrAddr + 128.U, pWrLo)
  }
  io.pRd.data := pMem.read(io.pRd.addr(8, 0), io.pRd.en)
  // O rescale (smStep 8..11, one d-tile each): O[m][dt] := rsr(O * alpha, 15)
  val smRescaleRd = state === sSoftmax && smStep >= 8.U && smStep <= 11.U
  val oRdSM = oMem.read((sm << 2) + (smStep - 8.U)(1, 0), smRescaleRd)
  val oRdSMv = RegNext(smRescaleRd, false.B)
  val oRdSMaddr = RegNext((sm << 2) + (smStep - 8.U)(1, 0))
  when(oRdSMv && rescale(sm)) {
    val scaled = VecInit(oRdSM.map(o => ((o * Cat(0.U(1.W), alpha).asSInt + (1 << 14).S) >> 15)(35, 0).asSInt))
    oMem.write(oRdSMaddr, scaled)
  }

  // ---- final normalisation: rl = floor((2^40 + l/2)/l); out16 = sat16(rsr(O*rl, 33))
  val div = Module(new PipeDivider(41, 28, 2))
  div.io.in.valid := false.B; div.io.in.bits.n := 0.U; div.io.in.bits.d := 1.U
  val fm = Reg(UInt(7.W)); val fStep = Reg(UInt(4.W))
  val rl = Reg(UInt(41.W))
  val fRd = state === sFinal && fStep >= 2.U && fStep <= 5.U
  val oRdF = oMem.read((fm << 2) + (fStep - 2.U)(1, 0), fRd)
  val oRdFv = RegNext(fRd, false.B)
  val oRdFtile = RegNext((fStep - 2.U)(1, 0))
  val outLanes = VecInit(oRdF.map { o =>
    val prod = o * Cat(0.U(1.W), rl(40, 0)).asSInt              // 36 x 42 -> 78 bits
    Sat.sint((prod + (BigInt(1) << 32).S) >> 33, 16)
  })
  val outRegs = Reg(Vec(2 * HT, UInt(256.W)))                     // d 0..15, 16..31, ..., 112..127
  when(oRdFv) {
    val lo = Cat(outLanes.slice(0, 16).reverse.map(_.asUInt)); val hi = Cat(outLanes.slice(16, 32).reverse.map(_.asUInt))
    for (t <- 0 until HT) when(oRdFtile === t.U) { outRegs(2 * t) := lo; outRegs(2 * t + 1) := hi }
  }
  val wrV = RegInit(false.B); val wrAddr = Reg(UInt(20.W)); val wrData = Reg(UInt(256.W))
  wrV := false.B
  io.wr.valid := wrV; io.wr.bits.bank := cmd.outBank; io.wr.bits.addr := wrAddr; io.wr.bits.data := wrData

  // ---- FSM
  switch(state) {
    is(sIdle) {
      when(io.cmd.fire) { cmd := io.cmd.bits; busy := true.B; h := 0.U; qb := 0.U; kt := 0.U; state := sIssueS; cycles := 0.U
        for (i <- 0 until QB) { started(i) := false.B; lSum(i) := 0.U; mx(i) := 0.S } }
    }
    is(sIssueS) { when(io.eng.fire) { state := sWaitS; beatsSeen := 0.U; beatsExpected := qRows << 1 } }
    is(sWaitS) { when(beatsSeen === beatsExpected && !io.engBusy) { state := sSoftmax; sm := 0.U; smStep := 0.U } }
    is(sSoftmax) {
      smStep := smStep + 1.U
      switch(smStep) {
        is(1.U) { for (i <- 0 until 32) srow(i) := smRd(i) }
        is(2.U) { for (i <- 0 until 32) srow(32 + i) := smRd(i) }
        is(3.U) {
          val mn = Mux(started(sm), Mux(mx(sm) > mt, mx(sm), mt), mt)
          mnew := mn
          val g = started(sm) && anyValid && (mn > mx(sm))
          grew := g
          val da = rsrU((mn - mx(sm)).asUInt * mqH, sqH)
          val daC = Mux(da > 4095.U, 4095.U, da(11, 0))
          val al = Mux(g, expLut(daC), 32768.U)
          alpha := al
          rescale(sm) := g
          when(anyValid) { mx(sm) := mn; started(sm) := true.B
            lSum(sm) := Mux(g, ((lSum(sm) * al) + (1.U << 14)) >> 15, lSum(sm)) }
          pSum := 0.U
          // no valid key in this tile: P of this query must be zero for the P·V passes (both 64-key halves)
          when(!anyValid) { pWr := true.B; pWrAddr := sm << 1; pWrHi := 0.U; pWrLo := 0.U; zeroSecond := true.B; smStep := 12.U }
        }
        is(4.U) { hiWord := pHiBytes; loWord := pLoBytes; pSum := pSum + pLaneSum }
        is(5.U) { pWr := true.B; pWrAddr := sm << 1; pWrHi := Cat(pHiBytes, hiWord); pWrLo := Cat(pLoBytes, loWord); pSum := pSum + pLaneSum }
        is(6.U) { hiWord := pHiBytes; loWord := pLoBytes; pSum := pSum + pLaneSum }
        is(7.U) { pWr := true.B; pWrAddr := (sm << 1) + 1.U; pWrHi := Cat(pHiBytes, hiWord); pWrLo := Cat(pLoBytes, loWord); pSum := pSum + pLaneSum }
        is(8.U) { lSum(sm) := lSum(sm) + pSum }
        is(12.U) { when(zeroSecond) { pWr := true.B; pWrAddr := (sm << 1) + 1.U; pWrHi := 0.U; pWrLo := 0.U; zeroSecond := false.B } }
        is(13.U) {
          smStep := 0.U
          when(sm === qRows - 1.U) { state := sIssuePVhi } .otherwise { sm := sm + 1.U }
        }
      }
    }
    is(sIssuePVhi) { when(io.eng.fire) { state := sWaitPVhi; beatsSeen := 0.U; beatsExpected := qRows << 2 } }
    is(sWaitPVhi) { when(beatsSeen === beatsExpected && !io.engBusy && !pvV1 && !pvV2) { state := sIssuePVlo } }
    is(sIssuePVlo) { when(io.eng.fire) { state := sWaitPVlo; beatsSeen := 0.U; beatsExpected := qRows << 2 } }
    is(sWaitPVlo) { when(beatsSeen === beatsExpected && !io.engBusy && !pvV1 && !pvV2) { state := sNextKt } }
    is(sNextKt) {
      when(kt === nKT - 1.U) { state := sFinal; fm := 0.U; fStep := 0.U } .otherwise { kt := kt + 1.U; state := sIssueS }
    }
    is(sFinal) {
      // per query: 0: start divide ; 1: wait ; 2..5: read the 4 O tiles ; 7..14: write the 8 output words
      fStep := fStep + 1.U
      when(fStep === 0.U) {
        div.io.in.valid := true.B; div.io.in.bits.n := (BigInt(1) << 40).U + (lSum(fm) >> 1); div.io.in.bits.d := lSum(fm)
        fStep := 1.U
      }
      when(fStep === 1.U) { when(div.io.out.valid) { rl := div.io.out.bits; fStep := 2.U } .otherwise { fStep := 1.U } }
      for (w <- 0 until 2 * HT) {
        when(fStep === (7 + w).U) { wrV := true.B; wrAddr := cmd.outBase + (q0 + fm) * cmd.outStride + (h << 3) + w.U; wrData := outRegs(w) }
      }
      when(fStep === 15.U) {
        fStep := 0.U
        when(fm === qRows - 1.U) { state := sNextQB } .otherwise { fm := fm + 1.U }
      }
    }
    is(sNextQB) {
      for (i <- 0 until QB) { started(i) := false.B; lSum(i) := 0.U; mx(i) := 0.S }
      kt := 0.U
      when(qb === nQB - 1.U) {
        qb := 0.U
        when(h === (H - 1).U) { state := sIdle; busy := false.B } .otherwise { h := h + 1.U; state := sIssueS }
      }.otherwise { qb := qb + 1.U; state := sIssueS }
    }
  }
}
