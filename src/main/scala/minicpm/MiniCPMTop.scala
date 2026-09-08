package minicpm

import chisel3._
import chisel3.util._

/** MiniCPM5-2B accelerator top. Streams: prompt token ids in (appended to the on-chip token buffer),
  * generated token ids out. Control: start / max_new; status: busy, done, token count.
  *
  * Register map (word addresses): 0 start (w) | {paused, done, busy} (r); 1 maxNew; 2 tokens in buffer (r);
  * 3 reset the token buffer (w); 4 generated count (r); 5 pc (r); 6 pos (r); 7 engine busy cycles (r);
  * 8 breakPc (w); 9 resume (w); 10 requant saturation count (r). */
class MiniCPMTop(cfg: MiniCPMConfig) extends Module {
  val io = IO(new Bundle {
    val prompt = Flipped(Decoupled(UInt(18.W)))
    val tokens = Decoupled(new Bundle { val id = UInt(18.W); val last = Bool() })
    val regWr = Flipped(Valid(new Bundle { val addr = UInt(8.W); val data = UInt(32.W) }))
    val regRdAddr = Input(UInt(8.W))
    val regRdData = Output(UInt(32.W))
    val done = Output(Bool())
    val busy = Output(Bool())
    val wsLoad = Flipped(Valid(new WsLoad))
    // debug read of activation banks / rowfac tables (tests only; idle-time use)
    val dbg = new Bundle {
      val bank = Input(UInt(3.W)); val addr = Input(UInt(20.W)); val en = Input(Bool())
      val data = Output(UInt(256.W)); val rowfac = Output(UInt(RowFac.bits.W))
    }
  })
  val mp = cfg.prog
  val bankWords = ChipMap.bankWords(cfg)
  require(mp.bankWords == bankWords, s"micro-program ${mp.getClass.getSimpleName} was emitted for banks ${mp.bankWords}, config has $bankWords")
  require(mp.maxCtx == cfg.maxCtx && mp.chunkRows == cfg.chunkRows && mp.nLayers == cfg.nLayers && mp.kvLayers == cfg.kvLayers,
    "the micro-program does not match the config (maxCtx / chunkRows / nLayers / kvLayers)")

  val seq = Module(new Sequencer(cfg))
  val eng = Module(new MatmulEngine(cfg))
  val vu = Module(new VectorUnit(cfg))
  val att = Module(new Attention(cfg))
  val kv = Module(new KVCache(cfg))
  val ws = Module(new WeightStore(cfg))
  val samp = Module(new Sampler(cfg, mp.eosIds))
  val banks = bankWords.map(w => Module(new ActBank(w)))
  val rowfac = Seq.fill(8)(Module(new RowFacTable(cfg.maxCtx)))
  val tokMem = SyncReadMem(cfg.maxCtx, UInt(18.W))

  // ---------------- registers
  val maxNew = RegInit(64.U(16.W))
  val nTokIn = RegInit(0.U(13.W))            // prompt tokens streamed since the last reset
  val startPulse = RegInit(false.B); startPulse := false.B
  val breakPc = RegInit(0x3ff.U(10.W)); val resume = RegInit(false.B); resume := false.B
  val doneR = RegInit(false.B)
  val tokCount = RegInit(0.U(16.W))
  when(io.regWr.valid) {
    switch(io.regWr.bits.addr) {
      is(0.U) { startPulse := io.regWr.bits.data(0); doneR := false.B; tokCount := 0.U }
      is(1.U) { maxNew := io.regWr.bits.data(15, 0) }
      is(3.U) { nTokIn := 0.U }
      is(8.U) { breakPc := io.regWr.bits.data(9, 0) }
      is(9.U) { resume := true.B }
    }
  }
  seq.io.breakPc := breakPc
  seq.io.resume := resume
  val engBusyCycles = RegInit(0.U(32.W))
  when(eng.io.busy) { engBusyCycles := engBusyCycles + 1.U }
  io.regRdData := MuxLookup(io.regRdAddr, 0.U)(Seq(
    0.U -> Cat(seq.io.paused, doneR, seq.io.busy), 1.U -> maxNew, 2.U -> nTokIn, 4.U -> tokCount, 5.U -> seq.io.pc,
    6.U -> seq.io.pos, 7.U -> engBusyCycles, 10.U -> eng.io.satCount))
  io.done := doneR
  io.busy := seq.io.busy
  when(seq.io.busy) { doneR := false.B }
  val busyQ = RegNext(seq.io.busy, false.B)
  when(busyQ && !seq.io.busy) { doneR := true.B }

  // ---------------- token buffer: host prompt stream + sampled tokens in, vector unit (EMBED) out
  io.prompt.ready := !seq.io.busy && nTokIn < (cfg.maxCtx - 1).U
  when(io.prompt.fire) { tokMem.write(nTokIn, io.prompt.bits); nTokIn := nTokIn + 1.U }
  when(seq.io.tokWr.valid) { tokMem.write(seq.io.tokWr.bits.addr, seq.io.tokWr.bits.data) }
  vu.io.tokRd.data := tokMem.read(vu.io.tokRd.addr(log2Ceil(cfg.maxCtx) - 1, 0), vu.io.tokRd.en)
  assert(!io.prompt.fire || io.prompt.bits < cfg.nVocab.U, "prompt token id out of vocabulary")

  // ---------------- sequencer wiring
  seq.io.start := startPulse
  seq.io.nTok := nTokIn
  seq.io.maxNew := maxNew
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
  samp.io.start := seq.io.samplerStart
  seq.io.samplerDone := samp.io.done; seq.io.samplerToken := samp.io.token; seq.io.samplerIsEos := samp.io.isEos
  val tokQ = Module(new Queue(UInt(18.W), 256))
  tokQ.io.enq <> seq.io.tokOut
  io.tokens.valid := tokQ.io.deq.valid
  io.tokens.bits.id := tokQ.io.deq.bits
  io.tokens.bits.last := false.B
  tokQ.io.deq.ready := io.tokens.ready
  when(seq.io.tokOut.fire) { tokCount := tokCount + 1.U }
  val tokVQ = RegNext(io.tokens.valid && !io.tokens.ready, false.B); val tokBQ = RegNext(io.tokens.bits.id)
  assert(!tokVQ || (io.tokens.valid && io.tokens.bits.id === tokBQ), "token stream is not irrevocable")
  assert(!io.tokens.valid || io.tokens.bits.id < cfg.nVocab.U, "token id out of vocabulary")
  for ((v, r, b, n) <- Seq((seq.io.vec.valid, seq.io.vec.ready, seq.io.vec.bits.asUInt, "vec"), (seq.io.att.valid, seq.io.att.ready, seq.io.att.bits.asUInt, "att"))) {
    val vq = RegNext(v && !r, false.B); val bq = RegNext(b)
    assert(!vq || (v && b === bq), s"$n command handshake is not irrevocable")
  }

  // ---------------- engine output demux (+ vector-unit K beats into the cache)
  val eo = eng.io.out
  eo.ready := true.B
  val packed8 = Cat(eo.bits.data.reverse.map(_(7, 0)))
  val packed16 = Cat(eo.bits.data.reverse.map(_(15, 0)))
  val packed32 = Cat(eo.bits.data.reverse.map(_.asUInt))
  val engSize = MuxLookup(eo.bits.mode, ActSize.W256.U)(Seq(MatmulMode.Int16.U -> ActSize.W512.U, MatmulMode.Int32.U -> ActSize.W1024.U))
  val engData = MuxLookup(eo.bits.mode, Cat(0.U(768.W), packed8))(Seq(MatmulMode.Int16.U -> Cat(0.U(512.W), packed16), MatmulMode.Int32.U -> packed32))
  att.io.in.valid := eo.valid && eo.bits.sink === 1.U; att.io.in.bits := eo.bits
  samp.io.in.valid := eo.valid && eo.bits.sink === 2.U; samp.io.in.bits := eo.bits
  val engKV = eo.valid && eo.bits.sink === 3.U
  kv.io.in.valid := engKV || vu.io.kvOut.valid
  kv.io.in.bits := Mux(engKV, eo.bits, vu.io.kvOut.bits)
  assert(!(engKV && vu.io.kvOut.valid), "engine and vector unit both writing the KV cache")
  assert(!(kv.io.in.valid && !kv.io.in.ready), "KV cache must accept every beat")

  // ---------------- activation bank read/write muxing
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
    assert(PopCount(Seq(engW, vuW, attW)) <= 1.U, "activation bank write conflict")
    assert(!b.io.wr.valid || b.io.wr.bits.addr < b.words256.U, "activation bank write address out of range")
    assert(!b.io.rd.en || b.io.rd.addr < b.words256.U, "activation bank read address out of range")
    b.io.wr.valid := engW || vuW || attW
    b.io.wr.bits.size := Mux(engW, engSize, Mux(vuW, vu.io.wr.bits.size, ActSize.W256.U))
    b.io.wr.bits.addr := Mux(engW, eo.bits.addr, Mux(vuW, vu.io.wr.bits.addr, att.io.wr.bits.addr))(b.addrBits - 1, 0)
    b.io.wr.bits.data := Mux(engW, engData, Mux(vuW, Cat(0.U(512.W), vu.io.wr.bits.data), Cat(0.U(768.W), att.io.wr.bits.data)))
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
  ws.io.w.en := eng.io.w.en && eng.io.w.src === 0.U
  ws.io.w.addr := eng.io.w.addr
  kv.io.rd.en := eng.io.w.en && eng.io.w.src =/= 0.U
  kv.io.rd.addr := eng.io.w.addr
  eng.io.w.data := Mux(wSrcQ === 0.U, ws.io.w.data, kv.io.rd.data)
  ws.io.p.en := eng.io.mult.en || vu.io.p.en
  ws.io.p.addr := Mux(vu.io.p.en, vu.io.p.addr, eng.io.mult.addr)
  assert(!(eng.io.mult.en && vu.io.p.en), "param port conflict")
  eng.io.mult.data := ws.io.p.data; vu.io.p.data := ws.io.p.data
  ws.io.p2.en := eng.io.bias.en
  ws.io.p2.addr := eng.io.bias.addr
  eng.io.bias.data := ws.io.p2.data
  ws.io.t.en := vu.io.rom.en; ws.io.t.addr := vu.io.rom.addr; vu.io.rom.data := ws.io.t.data
  ws.io.load := io.wsLoad
}
