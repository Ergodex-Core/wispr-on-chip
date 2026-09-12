package whisper

import chisel3._
import chisel3.util._
import whisper.generated.Luts
import whisper.util.SeqDivider

object VecOp {
  val LN = 0; val DYNQ = 1; val ADD = 2; val EMBED = 3
}

/** Row-wise vector op (see docs/numerics.md). Rows are processed sequentially, 16 int16 lanes/cycle. */
class VecCmd extends Bundle {
  val op        = UInt(2.W)
  val rows      = UInt(13.W)
  val cols      = UInt(11.W)    // elements per row (384 or 1536)
  // input A (bank rows)
  val inBank    = UInt(3.W)
  val inBase    = UInt(20.W)
  val inStride  = UInt(8.W)
  val inBits16  = Bool()        // int16 (true) or int8 rows
  // input B (ADD)
  val bSrc      = UInt(1.W)     // 0 bank, 1 i8mat ROM (positional embeddings)
  val bBank     = UInt(3.W)
  val bBase     = UInt(20.W)
  val bStride   = UInt(8.W)
  val bBits16   = Bool()
  // output
  val outBank   = UInt(3.W)
  val outBase   = UInt(20.W)
  val outStride = UInt(8.W)
  val rowfacBank = UInt(3.W)
  val rowBase   = UInt(13.W)
  val rowOff    = UInt(13.W)    // row chunk offset (rows [rowOff, rowOff+rows) of the tensor)
  val inLocal   = Bool()        // input A rows are chunk-local (not offset)
  val bLocal    = Bool()        // input B rows are chunk-local
  val outLocal  = Bool()        // output rows are chunk-local
  // LN
  val gBase     = UInt(16.W)    // param word base of G (i32vec)
  val bParamBase = UInt(16.W)   // param word base of B
  val epsLo     = UInt(32.W)
  val epsHi     = UInt(32.W)
  // DYNQ options
  val gelu      = Bool()
  val mPhi      = UInt(32.W)
  val sPhi      = UInt(6.W)
  val smooth    = Bool()
  val smoothBase = UInt(16.W)
  val static8   = Bool()        // static int8 requant instead of dynamic (conv1 GELU output)
  val reqMult   = UInt(32.W)
  val reqShift  = UInt(6.W)
  // ADD
  val ma        = UInt(32.W)
  val mb        = UInt(32.W)
  val geluA     = Bool()        // ADD: apply GELU16 (mPhi/sPhi) to operand A first
  val maFromTable = Bool()      // ma := i32vec[maTableBase + tok/32].entry(tok%32)   (decoder embedding)
  val maTableBase = UInt(16.W)
  val tok       = UInt(16.W)
  // EMBED
  val lmBase    = UInt(20.W)    // w8 base of dec.lm.w
}

class VectorUnit(cfg: WhisperConfig) extends Module {
  val L = 16                                   // lanes
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new VecCmd))
    val busy = Output(Bool())
    val rdA = new Bundle { val bank = Output(UInt(3.W)); val addr = Output(UInt(20.W)); val en = Output(Bool()); val data = Input(UInt(256.W)) }
    val rdB = new Bundle { val bank = Output(UInt(3.W)); val addr = Output(UInt(20.W)); val en = Output(Bool()); val data = Input(UInt(256.W)) }
    val rom = new Bundle { val addr = Output(UInt(20.W)); val en = Output(Bool()); val data = Input(UInt(256.W)) }        // i8mat space
    val w = new Bundle { val addr = Output(UInt(20.W)); val en = Output(Bool()); val data = Input(UInt(cfg.weightWordBits.W)) } // w8 space (EMBED)
    val p = new Bundle { val addr = Output(UInt(16.W)); val en = Output(Bool()); val data = Input(UInt(1024.W)) }
    val p2 = new Bundle { val addr = Output(UInt(16.W)); val en = Output(Bool()); val data = Input(UInt(1024.W)) }
    val wr = Valid(new Bundle { val bank = UInt(3.W); val addr = UInt(20.W); val data = UInt(256.W) })
    val rowfacWr = Valid(new Bundle { val bank = UInt(3.W); val addr = UInt(13.W); val data = UInt(16.W) })
  })
  val cmd = Reg(new VecCmd)
  val busy = RegInit(false.B)
  io.cmd.ready := !busy
  io.busy := busy

  // ---------------------------------------------------------------- LUT ROMs
  val phiRom = VecInit((Luts.Phi :+ 32768).map(_.U(16.W)))
  val rsqrtRom = VecInit(Luts.Rsqrt.map(_.U(17.W)))

  // ---------------------------------------------------------------- row buffer (int16 lanes)
  val rowBuf = SyncReadMem(cfg.dFF / L, Vec(L, SInt(16.W)))

  // ---------------------------------------------------------------- state
  val sIdle :: sP1 :: sS1 :: sP2 :: sS2 :: sP3 :: sNext :: Nil = Enum(7)
  val state = RegInit(sIdle)
  val row = Reg(UInt(13.W))
  val wordsIn = Mux(cmd.inBits16, cmd.cols >> 4, cmd.cols >> 5)   // words per input row
  val lanesWords = cmd.cols >> 4                                  // 16-lane groups per row
  val idx = Reg(UInt(8.W))                                        // issue counter within a pass
  val issuing = Reg(Bool())
  val rowAddrA = Reg(UInt(20.W)); val rowAddrB = Reg(UInt(20.W)); val rowAddrO = Reg(UInt(20.W))

  // scalar accumulators
  val sumX = Reg(SInt(32.W)); val sumX2 = Reg(UInt(64.W))
  val maxAbs = Reg(UInt(17.W))
  val meanF = Reg(SInt(32.W)); val y1 = Reg(UInt(18.W)); val shE = Reg(UInt(6.W))
  val recip = Reg(UInt(25.W))
  val maRow = Reg(UInt(32.W))
  val div = Module(new SeqDivider(25, 16))
  div.io.start := false.B; div.io.n := 0.U; div.io.d := 1.U

  // ---------------------------------------------------------------- pass-1 issue: reads
  // A read: for int16 rows one word per lane-group; for int8 rows one word per two lane-groups.
  val issueValid = busy && issuing && (state === sP1 || state === sP2 || state === sP3)
  val laneGroups = Mux(cmd.op === VecOp.EMBED.U && state === sP1, 48.U, lanesWords)   // EMBED pass 1: 48 ROM words
  val lastIssue = idx === laneGroups - 1.U
  io.rdA.bank := cmd.inBank
  io.rdA.addr := rowAddrA + Mux(cmd.inBits16, idx, idx >> 1)
  io.rdA.en := issueValid && (state =/= sP3) && cmd.op =/= VecOp.EMBED.U
  io.rdB.bank := cmd.bBank
  io.rdB.addr := rowAddrB + Mux(cmd.bBits16, idx, idx >> 1)
  io.rdB.en := issueValid && state === sP1 && cmd.op === VecOp.ADD.U && cmd.bSrc === 0.U
  io.rom.addr := rowAddrB + Mux(cmd.bBits16, idx, idx >> 1)
  io.rom.en := issueValid && state === sP1 && cmd.op === VecOp.ADD.U && cmd.bSrc === 1.U
  // EMBED: word = lmBase + ((tok/32)*12 + kt)*4 + g, idx = kt*4 + g
  io.w.addr := cmd.lmBase + ((cmd.tok >> 5) * 48.U) + idx
  io.w.en := issueValid && state === sP1 && cmd.op === VecOp.EMBED.U
  // params: G / smooth via p, B via p2 (half word per lane group)
  val pIdx = idx >> 1
  io.p.addr := Mux(cmd.op === VecOp.LN.U, cmd.gBase, cmd.smoothBase) + pIdx
  io.p.en := issueValid && ((state === sP2 && cmd.op === VecOp.LN.U) || (state === sP1 && cmd.op === VecOp.DYNQ.U && cmd.smooth))
  io.p2.addr := cmd.bParamBase + pIdx
  io.p2.en := issueValid && state === sP2 && cmd.op === VecOp.LN.U
  // rowbuf read in P3
  val rbRd = rowBuf.read(idx, issueValid && state === sP3)

  // ---------------------------------------------------------------- stage 1 (data arrives)
  val v1 = RegNext(issueValid, false.B)
  val idx1 = RegNext(idx); val st1 = RegNext(state); val last1 = RegNext(lastIssue)
  val half1 = idx1(0)
  def lanes16(word: UInt): Vec[SInt] = VecInit(Seq.tabulate(L)(i => word(16 * i + 15, 16 * i).asSInt))
  def lanes8(word: UInt, half: Bool): Vec[SInt] = {
    val w = Mux(half, word(255, 128), word(127, 0))
    VecInit(Seq.tabulate(L)(i => w(8 * i + 7, 8 * i).asSInt.pad(16)))
  }
  val aLanes = Mux(cmd.inBits16, lanes16(io.rdA.data), lanes8(io.rdA.data, half1))
  val bWord = Mux(cmd.bSrc === 0.U, io.rdB.data, io.rom.data)
  val bLanes = Mux(cmd.bBits16, lanes16(bWord), lanes8(bWord, half1))
  val gLanes = VecInit(Seq.tabulate(L)(i => Mux(half1, io.p.data(512 + 32 * i + 31, 512 + 32 * i), io.p.data(32 * i + 31, 32 * i)).asSInt))
  val bpLanes = VecInit(Seq.tabulate(L)(i => Mux(half1, io.p2.data(512 + 32 * i + 31, 512 + 32 * i), io.p2.data(32 * i + 31, 32 * i)).asSInt))
  // EMBED: 8 bytes (rows r=0..7 of column tok%32) from the 2048-bit word
  val col = cmd.tok(4, 0)
  val embBytes = VecInit(Seq.tabulate(8)(r => (io.w.data >> ((r * 32).U + col) * 8.U)(7, 0).asSInt))

  // ---------------------------------------------------------------- stage 2: compute (registered)
  val v2 = RegNext(v1, false.B); val idx2 = RegNext(idx1); val st2 = RegNext(st1); val last2 = RegNext(last1)
  val lanesOut2 = Reg(Vec(L, SInt(17.W)))     // int16 result lanes (17 bits so |−32768| fits for maxabs)
  val embOut2 = Reg(Vec(8, SInt(16.W)))
  // --- LN pass 1 stats
  val laneSq = aLanes.map(a => (a * a).asUInt)
  val sumLanes = VecInit(aLanes.map(_.pad(32))).reduceTree(_ +& _)
  val sumSq = VecInit(laneSq.map(_.pad(64))).reduceTree(_ +& _)
  when(v1 && st1 === sP1 && cmd.op === VecOp.LN.U) {
    sumX := sumX + sumLanes(31, 0).asSInt
    sumX2 := sumX2 + sumSq(63, 0)
  }
  // --- LN pass 2 affine: xc = (x<<8) - mean ; u = rsr(xc*y1, shE) ; v = rsr(u*G, 12) ; y = sat16(rsr(v+B, 4))
  def rsr(x: SInt, s: UInt): SInt = Mux(s === 0.U, x, (x + (1.S << (s - 1.U))) >> s)
  def rsrC(x: SInt, s: Int): SInt = if (s == 0) x else (x + (BigInt(1) << (s - 1)).S) >> s
  def sat(x: SInt, bits: Int): SInt = Sat.sint(x, bits)
  val lnLanes = Wire(Vec(L, SInt(17.W)))
  for (i <- 0 until L) {
    val xc = ((aLanes(i) << 8).asSInt - meanF).pad(26)
    val u = rsr(xc * Cat(0.U(1.W), y1).asSInt, shE)                  // <= 2^16 magnitude
    val v = rsrC(u(18, 0).asSInt * gLanes(i), 12)
    lnLanes(i) := sat(rsrC(v + bpLanes(i), 4), 16).pad(17)
  }
  // --- GELU16 (+smooth)
  val geluLanes = Wire(Vec(L, SInt(17.W)))
  for (i <- 0 until L) {
    val h = aLanes(i)
    val xfFull = rsr(h * Cat(0.U(1.W), cmd.mPhi).asSInt, cmd.sPhi)
    val xf = Mux(xfFull > 2047.S, 2047.S(13.W), Mux(xfFull < -2048.S, -2048.S(13.W), xfFull(12, 0).asSInt))
    val ii = ((xf >> 4).asSInt + 128.S)(8, 0).asUInt
    val f = xf(3, 0)
    val lo = phiRom(ii); val hi = phiRom(ii + 1.U)
    val phi = lo + (((hi - lo) * f + 8.U) >> 4)
    val g = sat(rsrC(h * Cat(0.U(1.W), phi).asSInt, 15), 16)
    val gs = Mux(cmd.smooth, sat(rsrC(g * gLanes(i), 15), 16), g)
    geluLanes(i) := Mux(cmd.gelu || cmd.geluA, gs, h).pad(17)
  }
  // --- ADD
  val addLanes = Wire(Vec(L, SInt(17.W)))
  for (i <- 0 until L) {
    val aIn = Mux(cmd.geluA, geluLanes(i)(15, 0).asSInt, aLanes(i))
    val s = aIn * Cat(0.U(1.W), maRow).asSInt + bLanes(i) * Cat(0.U(1.W), cmd.mb).asSInt
    addLanes(i) := sat(rsrC(s, 16), 16).pad(17)
  }
  // --- pass 3: a8 = sat8(rsr(y16 * mult, shift))
  val isEmbed = cmd.op === VecOp.EMBED.U
  val p3Mult = Mux(isEmbed, 1.U, Mux(cmd.static8, cmd.reqMult, recip))
  val p3Shift = Mux(isEmbed, 0.U, Mux(cmd.static8, cmd.reqShift, 16.U))
  val p3Lanes = Wire(Vec(L, SInt(8.W)))
  for (i <- 0 until L) p3Lanes(i) := sat(rsr(rbRd(i) * Cat(0.U(1.W), p3Mult).asSInt, p3Shift), 8)
  val p3Lanes2 = RegNext(p3Lanes)

  when(v1) {
    when(st1 === sP1) {
      lanesOut2 := Mux(cmd.op === VecOp.ADD.U, addLanes, geluLanes)
      embOut2 := embBytes
    }.elsewhen(st1 === sP2) { lanesOut2 := lnLanes }
  }
  // maxabs tracking + rowbuf write (stage 2)
  val absMax2 = VecInit(lanesOut2.map(x => Mux(x < 0.S, (-x).asUInt, x.asUInt)(16, 0))).reduceTree((a, b) => Mux(a > b, a, b))
  val rbWrite = v2 && ((st2 === sP1 && cmd.op === VecOp.DYNQ.U) || (st2 === sP2 && cmd.op === VecOp.LN.U))
  when(rbWrite) {
    rowBuf.write(idx2, VecInit(lanesOut2.map(_(15, 0).asSInt)))
    when(absMax2 > maxAbs) { maxAbs := absMax2 }
  }
  // EMBED: 8 bytes per word -> rowbuf lane group idx2/2, half idx2(0)
  val embHalf = Reg(Vec(8, SInt(16.W)))
  when(v2 && st2 === sP1 && cmd.op === VecOp.EMBED.U) {
    when(!idx2(0)) { embHalf := embOut2 }
    .otherwise { rowBuf.write(idx2 >> 1, VecInit((embHalf ++ embOut2).map(_.pad(16)))) }
  }
  // ADD output (int16 words)
  val addOut = Cat(lanesOut2.reverse.map(_(15, 0)))
  // pass-3 output assembly: two 16-lane groups -> one 256-bit int8 word (p3Lanes2 is stage-2 data)
  val lowHalf = Reg(UInt(128.W))
  val p3Half = Cat(p3Lanes2.reverse.map(_.asUInt))
  val p3Word = Cat(p3Half, lowHalf)
  when(v2 && st2 === sP3 && !idx2(0)) { lowHalf := p3Half }

  io.wr.valid := (v2 && st2 === sP1 && cmd.op === VecOp.ADD.U) || (v2 && st2 === sP3 && idx2(0))
  io.wr.bits.bank := cmd.outBank
  io.wr.bits.addr := rowAddrO + Mux(cmd.op === VecOp.ADD.U, idx2, idx2 >> 1)
  io.wr.bits.data := Mux(cmd.op === VecOp.ADD.U, addOut, p3Word)
  val rfWrite = RegInit(false.B)
  io.rowfacWr.valid := rfWrite
  io.rowfacWr.bits.bank := cmd.rowfacBank
  io.rowfacWr.bits.addr := cmd.rowBase + cmd.rowOff + row
  io.rowfacWr.bits.data := Mux(maxAbs === 0.U, 1.U, maxAbs(15, 0))
  rfWrite := false.B

  // ---------------------------------------------------------------- scalar S1 (LN): mean, V, rsqrt
  val s1Step = Reg(UInt(4.W))
  val V = Reg(UInt(64.W)); val mNorm = Reg(UInt(30.W)); val eSigned = Reg(SInt(6.W)); val y0 = Reg(UInt(17.W)); val tProd = Reg(UInt(64.W)); val dNewt = Reg(UInt(33.W))
  val tmp64 = Reg(SInt(64.W))
  val passDone = Reg(Bool())       // all writes of the pass drained

  // ---------------------------------------------------------------- FSM
  val drain = Reg(UInt(3.W))
  switch(state) {
    is(sIdle) {
      when(io.cmd.fire) {
        cmd := io.cmd.bits; busy := true.B; row := 0.U
        val ro = io.cmd.bits.rowOff
        rowAddrA := io.cmd.bits.inBase + Mux(io.cmd.bits.inLocal, 0.U, ro) * io.cmd.bits.inStride
        rowAddrB := io.cmd.bits.bBase + Mux(io.cmd.bits.bLocal, 0.U, ro) * io.cmd.bits.bStride
        rowAddrO := io.cmd.bits.outBase + Mux(io.cmd.bits.outLocal, 0.U, ro) * io.cmd.bits.outStride
        state := sNext; drain := 0.U; idx := 0.U; issuing := false.B
        maRow := io.cmd.bits.ma
      }
    }
    is(sNext) {
      // start a row
      sumX := 0.S; sumX2 := 0.U; maxAbs := 0.U; idx := 0.U; issuing := true.B; drain := 0.U
      when(cmd.maFromTable && cmd.op === VecOp.ADD.U) {
        // fetch ma from the i32vec table once (tok/32 word, tok%32 entry): use port p
        state := sS1; s1Step := 0.U
      }.otherwise { state := sP1 }
    }
    is(sP1) {
      when(issuing) {
        idx := idx + 1.U
        when(lastIssue) { issuing := false.B }
      }.otherwise {
        drain := drain + 1.U
        when(drain === 4.U) {
          when(cmd.op === VecOp.LN.U) { state := sS1; s1Step := 0.U }
          .elsewhen(cmd.op === VecOp.ADD.U) { state := sNext; row := row + 1.U; rowAddrA := rowAddrA + cmd.inStride; rowAddrB := rowAddrB + cmd.bStride; rowAddrO := rowAddrO + cmd.outStride
            when(row === cmd.rows - 1.U) { state := sIdle; busy := false.B } }
          .elsewhen(cmd.op === VecOp.EMBED.U || cmd.static8) { state := sP3; idx := 0.U; issuing := true.B; drain := 0.U }
          .otherwise { state := sS2; div.io.start := true.B; div.io.n := (127.U << 16) + (maxAbs >> 1); div.io.d := Mux(maxAbs === 0.U, 1.U, maxAbs(15, 0)) }
        }
      }
    }
    is(sS1) {
      s1Step := s1Step + 1.U
      when(cmd.op === VecOp.ADD.U) {
        // ma from table
        io.p.en := s1Step === 0.U
        io.p.addr := cmd.maTableBase + (cmd.tok >> 5)
        when(s1Step === 1.U) { maRow := VecInit(Seq.tabulate(32)(i => io.p.data(32 * i + 31, 32 * i)))(cmd.tok(4, 0)) }
        when(s1Step === 2.U) { state := sP1; idx := 0.U; issuing := true.B }
      }.otherwise {
        switch(s1Step) {
          is(0.U) { tmp64 := sumX * 5592405.S }
          is(1.U) { meanF := rsrC(tmp64, 23)(31, 0).asSInt }
          is(2.U) { // V = 65536*Sx2 - 512*mean*Sx + 384*mean^2 + eps
            val a = (sumX2 << 16)(63, 0).asSInt
            val b = (meanF * sumX) * 512.S
            val c = (meanF * meanF) * 384.S
            val eps = Cat(cmd.epsHi, cmd.epsLo).asSInt
            V := (a - b + c + eps).asUInt
          }
          is(3.U) { // normalise: e = (bl - 29) >> 1 ; m = V >> 2e
            val bl = (64.U - PriorityEncoder(Reverse(V)))     // bit length (V > 0)
            val e = (bl.asSInt - 29.S) >> 1                    // floor
            eSigned := e(5, 0).asSInt
            val e2 = (e << 1).asSInt
            mNorm := Mux(e2 >= 0.S, (V >> e2.asUInt)(29, 0), (V << (-e2).asUInt)(29, 0))
          }
          is(4.U) { y0 := rsqrtRom(mNorm(29, 22)); shE := (14.S + eSigned)(5, 0).asUInt }
          is(5.U) { tProd := mNorm * y0 * y0 }
          is(6.U) { dNewt := ((BigInt(3) << 60).U(64.W) - tProd + (1.U << 29))(63, 30) }
          is(7.U) { y1 := ((y0 * dNewt + (1.U << 30)) >> 31)(17, 0) }
          is(8.U) { state := sP2; idx := 0.U; issuing := true.B; drain := 0.U }
        }
      }
    }
    is(sP2) {
      when(issuing) {
        idx := idx + 1.U
        when(lastIssue) { issuing := false.B }
      }.otherwise {
        drain := drain + 1.U
        when(drain === 4.U) { state := sS2; div.io.start := true.B; div.io.n := (127.U << 16) + (maxAbs >> 1); div.io.d := Mux(maxAbs === 0.U, 1.U, maxAbs(15, 0)) }
      }
    }
    is(sS2) {
      when(div.io.done) { recip := div.io.q; rfWrite := true.B; state := sP3; idx := 0.U; issuing := true.B; drain := 0.U }
    }
    is(sP3) {
      when(issuing) {
        idx := idx + 1.U
        when(lastIssue) { issuing := false.B }
      }.otherwise {
        drain := drain + 1.U
        when(drain === 4.U) {
          row := row + 1.U; rowAddrA := rowAddrA + cmd.inStride; rowAddrO := rowAddrO + cmd.outStride; rowAddrB := rowAddrB + cmd.bStride
          state := sNext
          when(row === cmd.rows - 1.U) { state := sIdle; busy := false.B }
        }
      }
    }
  }
}
