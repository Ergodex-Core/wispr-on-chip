package whisper

import chisel3._
import chisel3.util._

/** Host-facing control/status register file (AXI-Lite-like: a simple register bus). */
class CtrlRegs extends Bundle {
  val start = Bool()
  val langToken = UInt(16.W)
  val nFrames = UInt(13.W)
}

/** Whisper-tiny accelerator top. Streams: mel frames in (80 int8 per beat, `last` ends the utterance),
  * token ids out. Control: start / language token / n_frames (defaults to the number of frames streamed
  * in, rounded up to a multiple of 128 as the host does); status: busy, done, token count. */
class WhisperTop(cfg: WhisperConfig) extends Module {
  val io = IO(new Bundle {
    val mel = Flipped(Decoupled(new Bundle { val data = UInt(640.W); val last = Bool() }))
    val tokens = Decoupled(new Bundle { val id = UInt(16.W); val last = Bool() })
    // register bus (AXI-Lite subset: one write and one read port; word addresses)
    val regWr = Flipped(Valid(new Bundle { val addr = UInt(8.W); val data = UInt(32.W) }))
    val regRdAddr = Input(UInt(8.W))
    val regRdData = Output(UInt(32.W))
    val done = Output(Bool())
    val busy = Output(Bool())
    val wsLoad = Flipped(Valid(new Bundle { val space = UInt(2.W); val addr = UInt(24.W); val slice = UInt(6.W); val data = UInt(32.W) }))
    // debug read of activation banks / rowfac tables (tests only; idle-time use)
    val dbg = new Bundle {
      val bank = Input(UInt(3.W)); val addr = Input(UInt(20.W)); val en = Input(Bool())
      val data = Output(UInt(256.W)); val rowfac = Output(UInt(16.W))
    }
  })
  import gen.ChipMap._

  val seq = Module(new Sequencer(cfg))
  val eng = Module(new MatmulEngine(cfg))
  val vu = Module(new VectorUnit(cfg))
  val att = Module(new Attention(cfg))
  val kv = Module(new KVCache(cfg))
  val ws = Module(new WeightStore(cfg))
  val samp = Module(new Sampler(cfg))
  val banks = bankWords.map(w => Module(new ActBank(w)))
  val rowfac = Seq.fill(8)(Module(new RowFacTable(rowsMax)))

  // ---------------- registers
  val regs = RegInit(0.U.asTypeOf(new CtrlRegs))
  val framesIn = RegInit(0.U(13.W))          // frames streamed since the last start
  val melWr = RegInit(false.B)               // a frame is being written (3 words)
  val melWord = Reg(UInt(2.W)); val melData = Reg(UInt(640.W))
  val startPulse = RegInit(false.B); startPulse := false.B
  val breakPc = RegInit(0x3ff.U(10.W)); val resume = RegInit(false.B); resume := false.B
  val doneR = RegInit(false.B)
  val tokCount = RegInit(0.U(16.W))
  when(io.regWr.valid) {
    switch(io.regWr.bits.addr) {
      is(0.U) { startPulse := io.regWr.bits.data(0); doneR := false.B; tokCount := 0.U }
      is(1.U) { regs.langToken := io.regWr.bits.data(15, 0) }
      is(2.U) { regs.nFrames := io.regWr.bits.data(12, 0) }
      is(3.U) { framesIn := 0.U }              // reset the frame counter before streaming a new clip
      is(8.U) { breakPc := io.regWr.bits.data(9, 0) }
      is(9.U) { resume := true.B }
    }
  }
  seq.io.breakPc := breakPc
  seq.io.resume := resume
  val nFramesEff = Mux(regs.nFrames =/= 0.U, regs.nFrames, ((framesIn + 127.U) >> 7) << 7)
  io.regRdData := MuxLookup(io.regRdAddr, 0.U)(Seq(
    0.U -> Cat(seq.io.paused, doneR, seq.io.busy), 1.U -> regs.langToken, 2.U -> nFramesEff, 3.U -> framesIn,
    4.U -> tokCount, 5.U -> seq.io.pc, 6.U -> seq.io.pos, 7.U -> eng.io.cycles))
  io.done := doneR
  io.busy := seq.io.busy
  when(seq.io.busy) { doneR := false.B }
  val busyQ = RegNext(seq.io.busy, false.B)
  when(busyQ && !seq.io.busy) { doneR := true.B }

  // ---------------- mel stream -> bank 0 (3 words per frame, bytes 80..95 zero)
  io.mel.ready := !melWr && !seq.io.busy
  when(io.mel.fire) { melWr := true.B; melWord := 0.U; melData := io.mel.bits.data }
  val melWrite = Wire(Valid(new ActWrite(banks(0).addrBits)))
  melWrite.valid := melWr
  melWrite.bits.addr := framesIn * 3.U + melWord
  melWrite.bits.wide := false.B
  val melSlice = VecInit(Seq.tabulate(3)(i => if (i < 2) melData(256 * i + 255, 256 * i) else Cat(0.U(128.W), melData(639, 512))))(melWord)
  melWrite.bits.data := Cat(0.U(256.W), melSlice)
  when(melWr) { melWord := melWord + 1.U; when(melWord === 2.U) { melWr := false.B; framesIn := framesIn + 1.U } }

  // ---------------- sequencer wiring
  seq.io.start := startPulse
  seq.io.nFrames := nFramesEff
  seq.io.langToken := regs.langToken
  // engine command arbitration: attention owns the engine while busy
  val engCmdArb = Module(new Arbiter(new MatmulCmd, 2))
  engCmdArb.io.in(0) <> att.io.eng
  engCmdArb.io.in(1) <> seq.io.mm
  eng.io.cmd <> engCmdArb.io.out
  att.io.engBusy := eng.io.busy
  vu.io.cmd <> seq.io.vec
  att.io.cmd <> seq.io.att
  seq.io.mmBusy := eng.io.busy; seq.io.vecBusy := vu.io.busy; seq.io.attBusy := att.io.busy; seq.io.kvBusy := kv.io.busy
  kv.io.cmd := seq.io.kvCmd
  kv.io.flush := seq.io.kvFlush
  // sampler
  samp.io.first := seq.io.samplerFirst
  samp.io.start := seq.io.samplerStart
  seq.io.samplerDone := samp.io.done; seq.io.samplerToken := samp.io.token; seq.io.samplerIsEot := samp.io.isEot
  // tokens out
  val tokQ = Module(new Queue(UInt(16.W), 256))
  tokQ.io.enq <> seq.io.tokOut
  io.tokens.valid := tokQ.io.deq.valid
  io.tokens.bits.id := tokQ.io.deq.bits
  io.tokens.bits.last := false.B
  tokQ.io.deq.ready := io.tokens.ready
  when(seq.io.tokOut.fire) { tokCount := tokCount + 1.U }

  // ---------------- engine output demux
  val eo = eng.io.out
  eo.ready := true.B
  val packed8 = Cat(eo.bits.data.reverse.map(_(7, 0)))
  val packed16 = Cat(eo.bits.data.reverse.map(_(15, 0)))
  att.io.in.valid := eo.valid && eo.bits.sink === 1.U; att.io.in.bits := eo.bits
  samp.io.in.valid := eo.valid && eo.bits.sink === 2.U; samp.io.in.bits := eo.bits
  kv.io.in.valid := eo.valid && eo.bits.sink === 3.U; kv.io.in.bits := eo.bits
  assert(!(kv.io.in.valid && !kv.io.in.ready), "KV cache must accept every beat")

  // ---------------- activation bank read/write muxing
  // readers: engine act (src 0), vu A, vu B ; writers: engine out (sink 0), vu wr, attention wr, mel stream
  val actSrcQ = RegNext(eng.io.act.src)
  val engBankQ = RegNext(eng.io.act.bank); val vuABankQ = RegNext(vu.io.rdA.bank); val vuBBankQ = RegNext(vu.io.rdB.bank)
  for ((b, i) <- banks.zipWithIndex) {
    val engHit = eng.io.act.en && eng.io.act.src === 0.U && eng.io.act.bank === i.U
    val vuAHit = vu.io.rdA.en && vu.io.rdA.bank === i.U
    val vuBHit = vu.io.rdB.en && vu.io.rdB.bank === i.U
    assert(PopCount(Seq(engHit, vuAHit, vuBHit)) <= 1.U, "activation bank read conflict")
    val dbgHit = io.dbg.en && io.dbg.bank === i.U && (!seq.io.busy || seq.io.paused)
    b.io.rd.en := engHit || vuAHit || vuBHit || dbgHit
    b.io.rd.addr := Mux(engHit, eng.io.act.addr, Mux(vuAHit, vu.io.rdA.addr, Mux(vuBHit, vu.io.rdB.addr, io.dbg.addr)))(b.addrBits - 1, 0)
    val engW = eo.valid && eo.bits.sink === 0.U && eo.bits.bank === i.U
    val vuW = vu.io.wr.valid && vu.io.wr.bits.bank === i.U
    val attW = att.io.wr.valid && att.io.wr.bits.bank === i.U
    val melW = melWrite.valid && (i == 0).B
    assert(PopCount(Seq(engW, vuW, attW, melW)) <= 1.U, "activation bank write conflict")
    b.io.wr.valid := engW || vuW || attW || melW
    b.io.wr.bits.wide := engW && eo.bits.mode === MatmulMode.Int16.U
    b.io.wr.bits.addr := Mux(engW, eo.bits.addr, Mux(vuW, vu.io.wr.bits.addr, Mux(attW, att.io.wr.bits.addr, melWrite.bits.addr)))(b.addrBits - 1, 0)
    b.io.wr.bits.data := Mux(engW, Mux(eo.bits.mode === MatmulMode.Int16.U, packed16, Cat(0.U(256.W), packed8)),
      Mux(vuW, Cat(0.U(256.W), vu.io.wr.bits.data), Mux(attW, Cat(0.U(256.W), att.io.wr.bits.data), melWrite.bits.data)))
  }
  val bankRd = VecInit(banks.map(_.io.rd.data))
  val dbgBankQ = RegNext(io.dbg.bank)
  io.dbg.data := bankRd(dbgBankQ)
  eng.io.act.data := Mux(actSrcQ === 1.U, att.io.pRd.data, bankRd(engBankQ))
  att.io.pRd.addr := eng.io.act.addr; att.io.pRd.en := eng.io.act.en && eng.io.act.src === 1.U
  vu.io.rdA.data := bankRd(vuABankQ)
  vu.io.rdB.data := bankRd(vuBBankQ)
  // rowfac tables
  val rfBankQ = RegNext(eng.io.rowfac.bank)
  for ((t, i) <- rowfac.zipWithIndex) {
    val dbgHit = io.dbg.en && io.dbg.bank === i.U && (!seq.io.busy || seq.io.paused)
    t.io.rd.en := (eng.io.rowfac.en && eng.io.rowfac.bank === i.U) || dbgHit
    t.io.rd.addr := Mux(dbgHit, io.dbg.addr(t.addrBits - 1, 0), eng.io.rowfac.addr(t.addrBits - 1, 0))
    t.io.wr.valid := vu.io.rowfacWr.valid && vu.io.rowfacWr.bits.bank === i.U
    t.io.wr.bits.addr := vu.io.rowfacWr.bits.addr(t.addrBits - 1, 0)
    t.io.wr.bits.data := vu.io.rowfacWr.bits.data
  }
  eng.io.rowfac.data := VecInit(rowfac.map(_.io.rd.data))(rfBankQ)
  io.dbg.rowfac := VecInit(rowfac.map(_.io.rd.data))(dbgBankQ)

  // ---------------- weight / kv / param ports
  val wSrcQ = RegNext(eng.io.w.src)
  ws.io.w.en := (eng.io.w.en && eng.io.w.src === 0.U) || vu.io.w.en
  ws.io.w.addr := Mux(vu.io.w.en, vu.io.w.addr, eng.io.w.addr)
  kv.io.rd.en := eng.io.w.en && eng.io.w.src =/= 0.U
  kv.io.rd.addr := eng.io.w.addr
  eng.io.w.data := Mux(wSrcQ === 0.U, ws.io.w.data, kv.io.rd.data)
  vu.io.w.data := ws.io.w.data
  ws.io.p.en := eng.io.mult.en || vu.io.p.en
  ws.io.p.addr := Mux(vu.io.p.en, vu.io.p.addr, eng.io.mult.addr)
  eng.io.mult.data := ws.io.p.data; vu.io.p.data := ws.io.p.data
  ws.io.p2.en := eng.io.bias.en || vu.io.p2.en
  ws.io.p2.addr := Mux(vu.io.p2.en, vu.io.p2.addr, eng.io.bias.addr)
  eng.io.bias.data := ws.io.p2.data; vu.io.p2.data := ws.io.p2.data
  ws.io.t.en := vu.io.rom.en; ws.io.t.addr := vu.io.rom.addr; vu.io.rom.data := ws.io.t.data
  ws.io.load.valid := io.wsLoad.valid
  ws.io.load.bits.space := io.wsLoad.bits.space; ws.io.load.bits.addr := io.wsLoad.bits.addr
  ws.io.load.bits.slice := io.wsLoad.bits.slice; ws.io.load.bits.data := io.wsLoad.bits.data
  kv.io.dbgWr.valid := false.B; kv.io.dbgWr.bits := DontCare
}
