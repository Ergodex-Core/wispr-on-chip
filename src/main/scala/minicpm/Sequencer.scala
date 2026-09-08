package minicpm

import chisel3._
import chisel3.util._
import minicpm.generated.Microcode

/** Executes the generated micro-program: one job at a time, substituting runtime quantities.
  * The same program serves prefill (rows = the whole prompt, in chunks) and decoding (one row at the
  * current position): CHUNK_BEGIN sizes the row loop from the phase. */
class Sequencer(cfg: MiniCPMConfig) extends Module {
  val prog = Microcode.program
  val io = IO(new Bundle {
    val start = Input(Bool())
    val nTok = Input(UInt(13.W))           // prompt tokens in the token buffer
    val maxNew = Input(UInt(16.W))         // generation limit
    val busy = Output(Bool())
    val pc = Output(UInt(10.W))
    val mm = Decoupled(new MatmulCmd)
    val vec = Decoupled(new VecCmd)
    val att = Decoupled(new AttnCmd)
    val kvCmd = Output(new KVCmd)
    val kvFlush = Output(Bool())
    val mmBusy = Input(Bool()); val vecBusy = Input(Bool()); val attBusy = Input(Bool()); val kvBusy = Input(Bool())
    val samplerStart = Output(Bool())
    val samplerDone = Input(Bool()); val samplerToken = Input(UInt(18.W)); val samplerIsEos = Input(Bool())
    val tokOut = Decoupled(UInt(18.W))     // emitted tokens (excluding eos)
    val tokWr = Valid(new Bundle { val addr = UInt(13.W); val data = UInt(18.W) })   // append to the token buffer
    val tokenCount = Output(UInt(16.W))
    val pos = Output(UInt(13.W))
    val decode = Output(Bool())
    val breakPc = Input(UInt(10.W))      // debug: pause before dispatching this pc (0x3ff = none)
    val resume = Input(Bool())
    val paused = Output(Bool())
  })
  val paused = RegInit(false.B)
  val brkDone = RegInit(false.B)
  io.paused := paused
  val rom = VecInit(prog.map(MicroInstr.fromUOp))
  val pc = RegInit(0.U(10.W))
  val busy = RegInit(false.B)
  val ins = rom(pc)
  val pcQ = RegNext(pc)
  val brkHit = pc === io.breakPc && !(brkDone && pc === pcQ)
  val decode = RegInit(false.B)
  val pos = Reg(UInt(13.W)); val nTokR = Reg(UInt(13.W)); val genCount = Reg(UInt(16.W))
  val chunkBase = Reg(UInt(13.W)); val chunkRows = Reg(UInt(13.W)); val chunkEnd = Reg(UInt(13.W)); val chunkSize = Reg(UInt(13.W))
  val lastRow = Mux(decode, pos, nTokR - 1.U)

  val sDispatch :: sWait :: sIdle :: Nil = Enum(3)
  val state = RegInit(sIdle)
  io.busy := busy
  io.pc := pc
  io.pos := pos
  io.decode := decode
  io.tokenCount := genCount

  def rows(sel: UInt, const: UInt): UInt = Mux(sel === RowsSel.Chunk.U, chunkRows, const)
  def base(sel: UInt): UInt = MuxLookup(sel, 0.U)(Seq(BaseSel.Chunk.U -> chunkBase, BaseSel.Last.U -> lastRow))

  // ---- matmul
  val mm = WireDefault(ins.mm)
  mm.rows := rows(ins.rowsSel, ins.mm.rows)
  mm.rowOff := base(ins.rowOffSel)
  io.mm.bits := mm
  io.mm.valid := busy && state === sDispatch && ins.opc === UOpc.MATMUL.U && !paused && !brkHit
  // ---- vector
  val vc = WireDefault(ins.vec)
  vc.rows := rows(ins.rowsSel, ins.vec.rows)
  vc.rowOff := base(ins.rowOffSel)
  vc.posBase := Mux(ins.posSel === 1.U, chunkBase, ins.vec.posBase)
  io.vec.bits := vc
  io.vec.valid := busy && state === sDispatch && ins.opc === UOpc.VEC.U && !paused && !brkHit
  // ---- attention
  val ac = WireDefault(ins.att)
  ac.nQueries := Mux(ins.nQSel === 1.U, chunkRows, ins.att.nQueries)
  ac.nKeys := Mux(ins.nKeysSel === KeysSel.ChunkEnd.U, chunkBase + chunkRows, ins.att.nKeys)
  ac.qPos0 := Mux(ins.qPos0Sel === 1.U, chunkBase, 0.U)
  io.att.bits := ac
  io.att.valid := busy && state === sDispatch && ins.opc === UOpc.ATTN.U && !paused && !brkHit
  // ---- kv context (registered on KVSET)
  val kvReg = Reg(new KVCmd)
  io.kvCmd := kvReg
  val kvFlush = RegInit(false.B); kvFlush := false.B
  io.kvFlush := kvFlush
  // ---- sampler / tokens
  val samplerStart = RegInit(false.B); samplerStart := false.B
  io.samplerStart := samplerStart
  val tokV = RegInit(false.B); val tokBits = Reg(UInt(18.W))
  io.tokOut.valid := tokV; io.tokOut.bits := tokBits
  when(io.tokOut.fire) { tokV := false.B }
  val tokWrV = RegInit(false.B); tokWrV := false.B
  io.tokWr.valid := tokWrV; io.tokWr.bits.addr := pos; io.tokWr.bits.data := tokBits
  val flushPending = RegInit(false.B)
  val waitCycles = Reg(UInt(4.W))

  switch(state) {
    is(sIdle) {
      when(io.start && io.nTok =/= 0.U) { busy := true.B; pc := 0.U; state := sDispatch; genCount := 0.U; pos := 0.U; nTokR := io.nTok; decode := false.B }
    }
    is(sDispatch) {
      when(brkHit && !paused) { paused := true.B }
      when(paused && io.resume) { paused := false.B; brkDone := true.B }
      when(!paused && !brkHit) {
        switch(ins.opc) {
          is(UOpc.MATMUL.U) { when(io.mm.fire) { state := sWait; flushPending := ins.flushK; waitCycles := 0.U } }
          is(UOpc.VEC.U) { when(io.vec.fire) { state := sWait; flushPending := ins.flushK; waitCycles := 0.U } }
          is(UOpc.ATTN.U) { when(io.att.fire) { state := sWait; waitCycles := 0.U } }
          is(UOpc.KVSET.U) {
            kvReg := ins.kv
            kvReg.keyOff := Mux(ins.keyOffSel === 1.U, chunkBase, 0.U)
            pc := pc + 1.U
          }
          is(UOpc.CHUNK_BEGIN.U) {
            val total = Mux(decode, 1.U, nTokR)
            val start = Mux(decode, pos, 0.U)
            chunkBase := start; chunkEnd := start + total; chunkSize := ins.imm2
            chunkRows := Mux(total < ins.imm2, total, ins.imm2)
            pc := pc + 1.U
          }
          is(UOpc.CHUNK_NEXT.U) {
            val nb = chunkBase + chunkSize
            when(nb < chunkEnd) {
              chunkBase := nb
              chunkRows := Mux(chunkEnd - nb < chunkSize, chunkEnd - nb, chunkSize)
              pc := ins.imm
            }.otherwise { pc := pc + 1.U }
          }
          is(UOpc.JUMP.U) { pc := ins.imm }
          is(UOpc.SAMPLE.U) {
            // the LM matmul just finished: sampler.done pulsed during the wait -> token ready now
            val t = io.samplerToken
            when(io.samplerIsEos || genCount === io.maxNew || nTokR === (cfg.maxCtx - 1).U) {
              busy := false.B; state := sIdle
            }.otherwise {
              tokBits := t; tokV := true.B; tokWrV := true.B
              pos := nTokR; nTokR := nTokR + 1.U; decode := true.B; genCount := genCount + 1.U
              pc := pc + 1.U
            }
          }
          is(UOpc.HALT.U) { busy := false.B; state := sIdle }
        }
      }
    }
    is(sWait) {
      waitCycles := Mux(waitCycles === 15.U, 15.U, waitCycles + 1.U)
      val unitBusy = io.mmBusy || io.vecBusy || io.attBusy || (io.kvBusy && !flushPending)
      when(waitCycles >= 2.U && !unitBusy) {
        when(flushPending) { kvFlush := true.B; flushPending := false.B; waitCycles := 0.U }
        .otherwise { state := sDispatch; pc := pc + 1.U }
      }
    }
  }
  when(pc =/= pcQ) { brkDone := false.B }
  // arm the sampler when the LM matmul is dispatched
  when(state === sDispatch && ins.opc === UOpc.MATMUL.U && ins.mm.outSink === 2.U && io.mm.fire) { samplerStart := true.B }
}
