package minicpm

import chisel3._
import chisel3.util._
import minicpm.generated.Luts
import minicpm.util.SeqDivider

object VecOp {
  val RMSNORM = 0; val DYNQ = 1; val ADD = 2; val EMBED = 3; val ROPE = 4; val SILUMUL = 5
}
object VecBits { val I8 = 0; val I16 = 1; val I32 = 2 }

/** Row-wise vector op (docs/numerics.md). Rows are processed sequentially, 16 lanes per cycle for int8 /
  * int16 rows and 16 lanes per two cycles for int32 rows.
  *  RMSNORM : int32 row -> rsqrt(mean sq) * gain -> int16 -> dynamic int8 + rowfac
  *  DYNQ    : int16 row -> dynamic int8 + rowfac
  *  ADD     : int32 a * ma + int32 b * mb -> int32 (residual add)
  *  EMBED   : int8 embedding row (i8mat ROM, token from the token buffer) * per-token mult -> int32
  *  ROPE    : int16 q/k row -> rotate pairs (d, d+64) with Q15 cos/sin of the row's position -> static int8
  *            (to a bank, or as key beats into the KV cache)
  *  SILUMUL : int16 gate row, int16 up row -> silu(g) * u (exact int32) -> dynamic int8 + rowfac */
class VecCmd extends Bundle {
  val op        = UInt(3.W)
  val rows      = UInt(13.W)
  val cols      = UInt(13.W)    // elements per row (2048, 256 or 6144)
  // input A (bank rows)
  val inBank    = UInt(3.W)
  val inBase    = UInt(20.W)
  val inStride  = UInt(9.W)
  val inBits    = UInt(2.W)     // VecBits: 0 int8, 1 int16, 2 int32
  // input B (ADD / SILUMUL, bank rows)
  val bBank     = UInt(3.W)
  val bBase     = UInt(20.W)
  val bStride   = UInt(9.W)
  val bBits     = UInt(2.W)
  // output
  val outBank   = UInt(3.W)
  val outBase   = UInt(20.W)
  val outStride = UInt(9.W)
  val outSink   = UInt(1.W)     // 0 bank, 1 KV cache beats (ROPE of K)
  val rowfacBank = UInt(3.W)
  val rowBase   = UInt(13.W)
  val rowOff    = UInt(13.W)    // row chunk offset (rows [rowOff, rowOff+rows) of the tensor)
  val posBase   = UInt(13.W)    // ROPE: position of row 0
  val inLocal   = Bool()        // input A rows are chunk-local (not offset by rowOff)
  val bLocal    = Bool()        // input B rows are chunk-local
  val outLocal  = Bool()        // output rows are chunk-local
  val rfLocal   = Bool()        // rowfac index is chunk-local
  // RMSNORM
  val gBase     = UInt(16.W)    // param word base of the gain G (i32vec)
  val epsLo     = UInt(32.W)
  val epsHi     = UInt(32.W)
  // ROPE: static int8 requant of the rotated int16, table base (i16mat rows of 8 words)
  val reqMult   = UInt(32.W)
  val reqShift  = UInt(6.W)
  val ropeBase  = UInt(24.W)
  // ADD
  val ma        = UInt(32.W)
  val mb        = UInt(32.W)
  // SILUMUL
  val mSig      = UInt(32.W)
  val sSig      = UInt(6.W)
  // EMBED
  val tokFromMem = Bool()       // token = tokMem[rowOff + row] (else cmd.tok)
  val tok       = UInt(18.W)
  val maTableBase = UInt(16.W)  // i32vec table of per-token multipliers (word tok/32, entry tok%32)
  val embBase   = UInt(24.W)    // i8mat base of the embedding table (row = tok, cols/32 words per row)
}

class VectorUnit(cfg: MiniCPMConfig) extends Module {
  val L = 16                                   // lanes
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new VecCmd))
    val busy = Output(Bool())
    val rdA = new Bundle { val bank = Output(UInt(3.W)); val addr = Output(UInt(20.W)); val en = Output(Bool()); val data = Input(UInt(256.W)) }
    val rdB = new Bundle { val bank = Output(UInt(3.W)); val addr = Output(UInt(20.W)); val en = Output(Bool()); val data = Input(UInt(256.W)) }
    val rom = new Bundle { val addr = Output(UInt(24.W)); val en = Output(Bool()); val data = Input(UInt(256.W)) }        // i8mat/i16mat space
    val p = new Bundle { val addr = Output(UInt(16.W)); val en = Output(Bool()); val data = Input(UInt(1024.W)) }         // i32vec space
    val tokRd = new Bundle { val addr = Output(UInt(13.W)); val en = Output(Bool()); val data = Input(UInt(18.W)) }        // token buffer
    val wr = Valid(new Bundle { val bank = UInt(3.W); val addr = UInt(20.W); val size = UInt(2.W); val data = UInt(512.W) })
    val kvOut = Valid(new MatmulOut(cfg))                                                                                 // ROPE-K beats
    val rowfacWr = Valid(new Bundle { val bank = UInt(3.W); val addr = UInt(13.W); val data = UInt(RowFac.bits.W) })
  })
  val cmd = Reg(new VecCmd)
  val busy = RegInit(false.B)
  io.cmd.ready := !busy
  io.busy := busy
  val op = cmd.op
  def isOp(o: Int): Bool = op === o.U
  val in32 = cmd.inBits === VecBits.I32.U && !isOp(VecOp.EMBED)   // A (and B) rows are int32: two words per lane group

  // ---------------------------------------------------------------- LUT ROMs
  val sigRom = VecInit((Luts.Sigmoid :+ Luts.SigmoidLast).map(_.U(16.W)))
  val rsqrtRom = VecInit(Luts.Rsqrt.map(_.U(17.W)))

  // ---------------------------------------------------------------- row buffer (int32 lanes)
  val RB = cfg.dFF / L                         // 384 lane groups
  val ROPE_OUT = RB / 2                        // rotated lanes live at [192, 192 + cols/16)
  val rowBuf = SyncReadMem(RB, Vec(L, SInt(32.W)))
  val ropeRow = Reg(Vec(8, UInt(256.W)))       // cos words 0..3 | sin words 4..7 of the current position

  // ---------------------------------------------------------------- state
  val sIdle :: sP1 :: sS1 :: sRopeLd :: sP2 :: sS2 :: sP3 :: sNext :: Nil = Enum(8)
  val state = RegInit(sIdle)
  val row = Reg(UInt(13.W))
  val lanesWords = cmd.cols >> 4                                  // 16-lane groups per row
  val cnt = Reg(UInt(10.W))                                       // issue counter within a pass
  val issuing = Reg(Bool())
  val rowAddrA = Reg(UInt(20.W)); val rowAddrB = Reg(UInt(24.W)); val rowAddrO = Reg(UInt(20.W))
  // int32 passes (P1/P2 reading A) issue two cycles per lane group
  val two = in32 && (state === sP1 || state === sP2)
  val idx = Mux(two, cnt >> 1, cnt)
  val sub = two && cnt(0)
  val issueTotal = Mux(two, lanesWords << 1, lanesWords)
  val lastIssue = cnt === issueTotal - 1.U

  // scalar registers
  val sumX2 = Reg(UInt(80.W))
  val maxAbs = Reg(UInt(32.W))
  val y1 = Reg(UInt(18.W)); val shE = Reg(UInt(6.W))
  val recip = Reg(UInt(25.W)); val rfB = Reg(UInt(5.W)); val rfM16 = Reg(UInt(16.W))
  val maRow = Reg(UInt(32.W))
  val tokR = Reg(UInt(18.W))
  val ropeIdx = Reg(UInt(4.W))
  val div = Module(new SeqDivider(25, 16))
  div.io.start := false.B; div.io.n := 0.U; div.io.d := 1.U

  // ---------------------------------------------------------------- issue: reads
  val issueValid = busy && issuing && (state === sP1 || state === sP2 || state === sP3)
  def wordOf(bits: UInt, i: UInt, s: Bool): UInt = MuxLookup(bits, i)(Seq(VecBits.I8.U -> (i >> 1), VecBits.I32.U -> ((i << 1) + s)))
  io.rdA.bank := cmd.inBank
  io.rdA.addr := rowAddrA + wordOf(cmd.inBits, idx, sub)
  io.rdA.en := issueValid && !isOp(VecOp.EMBED) && (state === sP1 || (state === sP2 && isOp(VecOp.RMSNORM)))
  io.rdB.bank := cmd.bBank
  io.rdB.addr := rowAddrB(19, 0) + wordOf(cmd.bBits, idx, sub)
  io.rdB.en := issueValid && state === sP1 && (isOp(VecOp.ADD) || isOp(VecOp.SILUMUL))
  val posRow = cmd.posBase + row
  io.rom.addr := Mux(state === sRopeLd, cmd.ropeBase + (posRow << 3) + ropeIdx, rowAddrB + (idx >> 1))
  io.rom.en := (state === sRopeLd && ropeIdx < 8.U) || (issueValid && state === sP1 && isOp(VecOp.EMBED))
  val s1Step = Reg(UInt(4.W))
  io.p.addr := Mux(state === sS1, cmd.maTableBase + (tokR >> 5), cmd.gBase + (idx >> 1))
  io.p.en := (issueValid && state === sP2 && isOp(VecOp.RMSNORM)) || (state === sS1 && isOp(VecOp.EMBED) && s1Step === 2.U)
  io.tokRd.addr := cmd.rowOff + row
  io.tokRd.en := state === sS1 && isOp(VecOp.EMBED) && s1Step === 0.U && cmd.tokFromMem
  // rowbuf reads: P3 (P3 of ROPE reads the rotated half), P2 of ROPE reads the pair (idx, idx^4)
  val rbAddr = Mux(state === sP3 && isOp(VecOp.ROPE), ROPE_OUT.U + idx, idx)
  val rbRd = rowBuf.read(rbAddr, issueValid && (state === sP3 || (state === sP2 && isOp(VecOp.ROPE))))
  val rbRd2 = rowBuf.read(idx ^ 4.U, issueValid && state === sP2 && isOp(VecOp.ROPE))

  // ---------------------------------------------------------------- stage 1 (data arrives)
  val v1raw = RegNext(issueValid, false.B)
  val idx1 = RegNext(idx); val st1 = RegNext(state); val sub1 = RegNext(sub); val two1 = RegNext(two)
  val half1 = idx1(0)
  def lanes16(word: UInt): Vec[SInt] = VecInit(Seq.tabulate(L)(i => word(16 * i + 15, 16 * i).asSInt.pad(32)))
  def lanes8(word: UInt, half: Bool): Vec[SInt] = {
    val w = Mux(half, word(255, 128), word(127, 0))
    VecInit(Seq.tabulate(L)(i => w(8 * i + 7, 8 * i).asSInt.pad(32)))
  }
  def lanes32lo(word: UInt): Seq[SInt] = Seq.tabulate(L / 2)(i => word(32 * i + 31, 32 * i).asSInt)
  // int32 rows: the first word of a group holds lanes 0..7 (latched), the second lanes 8..15 (group complete)
  val aLo = Reg(Vec(L / 2, SInt(32.W))); val bLo = Reg(Vec(L / 2, SInt(32.W)))
  when(v1raw && two1 && !sub1) { aLo := VecInit(lanes32lo(io.rdA.data)); bLo := VecInit(lanes32lo(io.rdB.data)) }
  val v1 = v1raw && !(two1 && !sub1)
  val aLanes = Wire(Vec(L, SInt(32.W))); val bLanes = Wire(Vec(L, SInt(32.W)))
  aLanes := Mux(isOp(VecOp.EMBED), lanes8(io.rom.data, half1),
    MuxLookup(cmd.inBits, lanes16(io.rdA.data))(Seq(VecBits.I8.U -> lanes8(io.rdA.data, half1), VecBits.I32.U -> VecInit(aLo ++ lanes32lo(io.rdA.data)))))
  bLanes := MuxLookup(cmd.bBits, lanes16(io.rdB.data))(Seq(VecBits.I8.U -> lanes8(io.rdB.data, half1), VecBits.I32.U -> VecInit(bLo ++ lanes32lo(io.rdB.data))))
  val gLanes = VecInit(Seq.tabulate(L)(i => Mux(half1, io.p.data(512 + 32 * i + 31, 512 + 32 * i), io.p.data(32 * i + 31, 32 * i)).asSInt))

  // ---------------------------------------------------------------- stage 2: compute (registered)
  val v2 = RegNext(v1, false.B); val idx2 = RegNext(idx1); val st2 = RegNext(st1)
  val lanesOut2 = Reg(Vec(L, SInt(32.W)))
  def rsr(x: SInt, s: UInt): SInt = Mux(s === 0.U, x, (x + (1.S << (s - 1.U))) >> s)
  def rsrC(x: SInt, s: Int): SInt = if (s == 0) x else (x + (BigInt(1) << (s - 1)).S) >> s
  def sat(x: SInt, bits: Int): SInt = Sat.sint(x, bits)
  // --- RMSNORM pass 1: sum of squares of the int32 row (each square < 2^62, sum < 2^74)
  val sumSq = aLanes.map(a => (a * a).asUInt.pad(80)).reduce(_ +& _)
  when(v1 && st1 === sP1 && isOp(VecOp.RMSNORM)) { sumX2 := sumX2 + sumSq(79, 0) }
  // --- RMSNORM pass 2 affine: u = rsr(x*y1, shE) ; v = rsr(u*G, 12) ; y = sat16(rsr(v, 4))
  val rmsLanes = Wire(Vec(L, SInt(32.W)))
  for (i <- 0 until L) {
    val u = rsr(aLanes(i) * Cat(0.U(1.W), y1).asSInt, shE)               // <= 2^16 magnitude
    val v = rsrC(u(18, 0).asSInt * gLanes(i), 12)
    rmsLanes(i) := sat(rsrC(v, 4), 16).pad(32)
  }
  // --- SILUMUL: xf = clamp(rsr(g*mSig, sSig)); sig = LUT(xf); s16 = sat16(rsr(g*sig, 15)); h32 = s16 * u
  val siluLanes = Wire(Vec(L, SInt(32.W)))
  for (i <- 0 until L) {
    val g = aLanes(i)(15, 0).asSInt
    val xfFull = rsr(g * Cat(0.U(1.W), cmd.mSig).asSInt, cmd.sSig)
    val xf = Mux(xfFull > 2047.S, 2047.S(13.W), Mux(xfFull < -2048.S, -2048.S(13.W), xfFull(12, 0).asSInt))
    val ii = ((xf >> 4).asSInt + 128.S)(8, 0).asUInt
    val f = xf(3, 0)
    val lo = sigRom(ii); val hi = sigRom(ii + 1.U)
    val sig = lo + (((hi - lo) * f + 8.U) >> 4)
    val s16 = sat(rsrC(g * Cat(0.U(1.W), sig).asSInt, 15), 16)
    siluLanes(i) := (s16 * bLanes(i)(15, 0).asSInt)(31, 0).asSInt        // |h| < 2^30
  }
  // --- ADD / EMBED (int32 results)
  val addLanes = Wire(Vec(L, SInt(32.W)))
  for (i <- 0 until L) {
    val s = aLanes(i) * Cat(0.U(1.W), maRow).asSInt + bLanes(i) * Cat(0.U(1.W), cmd.mb).asSInt   // 65 bits
    addLanes(i) := sat(rsrC(s, 16), 32)
  }
  val embLanes = VecInit(Seq.tabulate(L)(i => sat(aLanes(i)(7, 0).asSInt * Cat(0.U(1.W), maRow).asSInt, 32)))
  // --- ROPE pass 2: pair (idx, idx^4) of a head: j < 4: x1*c - x2*s ; j >= 4: x2*c + x1*s
  val ropeLanes = Wire(Vec(L, SInt(32.W)))
  val jr = idx1(2, 0)
  val cLanes = lanes16(ropeRow(jr(1, 0))); val sLanes = lanes16(ropeRow(4.U + jr(1, 0)))
  for (i <- 0 until L) {
    val a = rbRd(i)(15, 0).asSInt; val b = rbRd2(i)(15, 0).asSInt
    val c = cLanes(i)(15, 0).asSInt; val sn = sLanes(i)(15, 0).asSInt
    val r = Mux(jr(2), a * c + b * sn, a * c - b * sn)
    ropeLanes(i) := sat(rsrC(r, 15), 16).pad(32)
  }
  // --- pass 3: a8 = sat8(rsr(y * mult, shift))  (dynamic: mult = recip, shift = 16 + b ; ROPE: static reqMult/reqShift)
  val p3Mult = Mux(isOp(VecOp.ROPE), cmd.reqMult, recip)
  val p3Shift = Mux(isOp(VecOp.ROPE), cmd.reqShift, 16.U + rfB)
  val p3Lanes = Wire(Vec(L, SInt(8.W)))
  for (i <- 0 until L) p3Lanes(i) := sat(rsr(rbRd(i) * Cat(0.U(1.W), p3Mult).asSInt, p3Shift), 8)
  val p3Lanes2 = RegNext(p3Lanes)

  when(v1) {
    when(st1 === sP1) {
      lanesOut2 := MuxLookup(op, aLanes)(Seq(VecOp.ADD.U -> addLanes, VecOp.EMBED.U -> embLanes, VecOp.SILUMUL.U -> siluLanes))
    }.elsewhen(st1 === sP2) { lanesOut2 := Mux(isOp(VecOp.ROPE), ropeLanes, rmsLanes) }
  }
  // maxabs tracking + rowbuf write (stage 2)
  val absMax2 = lanesOut2.map(x => Mux(x < 0.S, (-x).asUInt, x.asUInt)(31, 0)).reduce((a, b) => Mux(a > b, a, b))
  val rbWriteDyn = v2 && ((st2 === sP1 && (isOp(VecOp.DYNQ) || isOp(VecOp.SILUMUL))) || (st2 === sP2 && isOp(VecOp.RMSNORM)))
  val rbWriteRope = v2 && isOp(VecOp.ROPE) && (st2 === sP1 || st2 === sP2)
  when(rbWriteDyn || rbWriteRope) {
    rowBuf.write(Mux(st2 === sP2 && isOp(VecOp.ROPE), ROPE_OUT.U + idx2, idx2), lanesOut2)
  }
  when(rbWriteDyn && absMax2 > maxAbs) { maxAbs := absMax2 }
  // int32 output words (ADD / EMBED): 16 lanes = 512 bits = two 256-bit words
  val out32 = Cat(lanesOut2.reverse.map(_.asUInt))
  // pass-3 output assembly: two 16-lane groups -> one 256-bit int8 word (p3Lanes2 is stage-2 data)
  val lowHalf = Reg(UInt(128.W))
  val p3Half = Cat(p3Lanes2.reverse.map(_.asUInt))
  val p3Word = Cat(p3Half, lowHalf)
  when(v2 && st2 === sP3 && !idx2(0)) { lowHalf := p3Half }
  val outWide = v2 && st2 === sP1 && (isOp(VecOp.ADD) || isOp(VecOp.EMBED))
  val out8 = v2 && st2 === sP3 && idx2(0)
  io.wr.valid := outWide || (out8 && cmd.outSink === 0.U)
  io.wr.bits.bank := cmd.outBank
  io.wr.bits.addr := rowAddrO + Mux(outWide, idx2 << 1, idx2 >> 1)
  io.wr.bits.size := Mux(outWide, ActSize.W512.U, ActSize.W256.U)
  io.wr.bits.data := Mux(outWide, out32, Cat(0.U(256.W), p3Word))
  io.kvOut.valid := out8 && cmd.outSink === 1.U
  io.kvOut.bits := 0.U.asTypeOf(new MatmulOut(cfg))
  io.kvOut.bits.sink := 3.U; io.kvOut.bits.mode := MatmulMode.Int8.U
  io.kvOut.bits.trow := cmd.rowOff + row; io.kvOut.bits.row := cmd.rowBase + cmd.rowOff + row; io.kvOut.bits.nTile := idx2 >> 1
  for (i <- 0 until 32) io.kvOut.bits.data(i) := p3Word(8 * i + 7, 8 * i).asSInt.pad(32)
  val rfWrite = RegInit(false.B)
  io.rowfacWr.valid := rfWrite
  io.rowfacWr.bits.bank := cmd.rowfacBank
  io.rowfacWr.bits.addr := cmd.rowBase + Mux(cmd.rfLocal, 0.U, cmd.rowOff) + row
  io.rowfacWr.bits.data := Cat(rfB, rfM16)
  rfWrite := false.B

  // ---------------------------------------------------------------- scalar S1 (RMSNORM): V, rsqrt
  val V = Reg(UInt(80.W)); val mNorm = Reg(UInt(30.W)); val eSigned = Reg(SInt(7.W)); val y0 = Reg(UInt(17.W)); val tProd = Reg(UInt(64.W)); val dNewt = Reg(UInt(33.W))
  // rope row preload data capture
  val ropeLdV = RegNext(state === sRopeLd && ropeIdx < 8.U, false.B)
  val ropeLdIdx = RegNext(ropeIdx)
  when(ropeLdV) { ropeRow(ropeLdIdx(2, 0)) := io.rom.data }
  // dynamic quant scalar: maxabs -> (m16, b), then recip = floor((127<<16 + m16/2) / m16)
  val dynStep = Reg(UInt(2.W))

  // ---------------------------------------------------------------- FSM
  val drain = Reg(UInt(3.W))
  def nextRow(): Unit = {
    row := row + 1.U; rowAddrA := rowAddrA + cmd.inStride; rowAddrO := rowAddrO + cmd.outStride; rowAddrB := rowAddrB + cmd.bStride
    state := sNext
    when(row === cmd.rows - 1.U) { state := sIdle; busy := false.B }
  }
  switch(state) {
    is(sIdle) {
      when(io.cmd.fire) {
        cmd := io.cmd.bits; busy := true.B; row := 0.U
        val ro = io.cmd.bits.rowOff
        rowAddrA := io.cmd.bits.inBase + Mux(io.cmd.bits.inLocal, 0.U, ro) * io.cmd.bits.inStride
        rowAddrB := io.cmd.bits.bBase + Mux(io.cmd.bits.bLocal, 0.U, ro) * io.cmd.bits.bStride
        rowAddrO := io.cmd.bits.outBase + Mux(io.cmd.bits.outLocal, 0.U, ro) * io.cmd.bits.outStride
        state := sNext; drain := 0.U; cnt := 0.U; issuing := false.B
        maRow := io.cmd.bits.ma
      }
    }
    is(sNext) {
      sumX2 := 0.U; maxAbs := 0.U; cnt := 0.U; issuing := true.B; drain := 0.U
      when(isOp(VecOp.EMBED)) { state := sS1; s1Step := 0.U; issuing := false.B } .otherwise { state := sP1 }
    }
    is(sP1) {
      when(issuing) {
        cnt := cnt + 1.U
        when(lastIssue) { issuing := false.B }
      }.otherwise {
        drain := drain + 1.U
        when(drain === 4.U) {
          when(isOp(VecOp.RMSNORM)) { state := sS1; s1Step := 0.U }
          .elsewhen(isOp(VecOp.ADD) || isOp(VecOp.EMBED)) { nextRow() }
          .elsewhen(isOp(VecOp.ROPE)) { state := sRopeLd; ropeIdx := 0.U }
          .otherwise { state := sS2; dynStep := 0.U }
        }
      }
    }
    is(sS1) {
      s1Step := s1Step + 1.U
      when(isOp(VecOp.EMBED)) {
        // 0: token read issued ; 1: token latched ; 2: mult-table word read ; 3: mult latched, row base ; 4: go
        when(s1Step === 1.U) { tokR := Mux(cmd.tokFromMem, io.tokRd.data, cmd.tok) }
        when(s1Step === 3.U) {
          maRow := VecInit(Seq.tabulate(32)(i => io.p.data(32 * i + 31, 32 * i)))(tokR(4, 0))
          rowAddrB := cmd.embBase + tokR * (cmd.cols >> 5)
        }
        when(s1Step === 4.U) { state := sP1; cnt := 0.U; issuing := true.B; drain := 0.U }
      }.otherwise {
        switch(s1Step) {
          is(0.U) { V := sumX2 + Cat(cmd.epsHi, cmd.epsLo) }
          is(1.U) { // normalise: e = (bl - 29) >> 1 ; m = V >> 2e (V > 0)
            val bl = (80.U - PriorityEncoder(Reverse(V)))     // bit length, 1..80 (7 bits)
            val e = ((bl.zext - 29.S(8.W)) >> 1).asSInt        // floor, in [-14, 25] (8 bits)
            eSigned := e(6, 0).asSInt
            val e2 = (e << 1).asSInt
            mNorm := Mux(e2 >= 0.S, (V >> e2.asUInt)(29, 0), (V << (-e2).asUInt)(29, 0))
          }
          is(2.U) { y0 := rsqrtRom(mNorm(29, 22)); shE := (14.S + eSigned)(5, 0).asUInt }
          is(3.U) { tProd := mNorm * y0 * y0 }
          is(4.U) { dNewt := ((BigInt(3) << 60).U(64.W) - tProd + (1.U << 29))(63, 30) }
          is(5.U) { y1 := ((y0 * dNewt + (1.U << 30)) >> 31)(17, 0) }
          is(6.U) { state := sP2; cnt := 0.U; issuing := true.B; drain := 0.U }
        }
      }
    }
    is(sRopeLd) {
      ropeIdx := ropeIdx + 1.U
      when(ropeIdx === 9.U) { state := sP2; cnt := 0.U; issuing := true.B; drain := 0.U }
    }
    is(sP2) {
      when(issuing) {
        cnt := cnt + 1.U
        when(lastIssue) { issuing := false.B }
      }.otherwise {
        drain := drain + 1.U
        when(drain === 4.U) {
          when(isOp(VecOp.ROPE)) { state := sP3; cnt := 0.U; issuing := true.B; drain := 0.U }
          .otherwise { state := sS2; dynStep := 0.U }
        }
      }
    }
    is(sS2) {
      dynStep := dynStep + 1.U
      when(dynStep === 0.U) {
        // b = max(bitlen(maxabs) - 16, 0) with maxabs = max(1, maxAbs); m16 = maxabs >> b
        val ma = Mux(maxAbs === 0.U, 1.U(32.W), maxAbs)
        val bl = 32.U - PriorityEncoder(Reverse(ma))
        val b = Mux(bl > 16.U, bl - 16.U, 0.U)
        rfB := b(4, 0)
        rfM16 := (ma >> b(4, 0))(15, 0)
      }
      when(dynStep === 1.U) { div.io.start := true.B; div.io.n := (127.U << 16) + (rfM16 >> 1); div.io.d := rfM16; dynStep := 2.U }
      when(dynStep === 2.U) {
        dynStep := 2.U
        when(div.io.done) { recip := div.io.q; rfWrite := true.B; state := sP3; cnt := 0.U; issuing := true.B; drain := 0.U }
      }
    }
    is(sP3) {
      when(issuing) {
        cnt := cnt + 1.U
        when(lastIssue) { issuing := false.B }
      }.otherwise {
        drain := drain + 1.U
        when(drain === 4.U) { nextRow() }
      }
    }
  }
}
