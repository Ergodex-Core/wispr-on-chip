package whisper

import chisel3._
import chisel3.util._
import whisper.generated.Microcode

/** Executes the generated micro-program: one job at a time, substituting runtime quantities. */
class Sequencer(cfg: WhisperConfig) extends Module {
  val prog = Microcode.program
  val io = IO(new Bundle {
    val start = Input(Bool())
    val nFrames = Input(UInt(13.W))
    val langToken = Input(UInt(16.W))
    val busy = Output(Bool())
    val pc = Output(UInt(10.W))
    val mm = Decoupled(new MatmulCmd)
    val vec = Decoupled(new VecCmd)
    val att = Decoupled(new AttnCmd)
    val kvCmd = Output(new KVCmd)
    val kvFlush = Output(Bool())
    val mmBusy = Input(Bool()); val vecBusy = Input(Bool()); val attBusy = Input(Bool()); val kvBusy = Input(Bool())
    val samplerStart = Output(Bool()); val samplerFirst = Output(Bool())
    val samplerDone = Input(Bool()); val samplerToken = Input(UInt(16.W)); val samplerIsEot = Input(Bool())
    val tokOut = Decoupled(UInt(16.W))   // emitted tokens (excluding eot)
    val tokenCount = Output(UInt(16.W))
    val pos = Output(UInt(13.W))
  })
  val rom = VecInit(prog.map(MicroInstr.fromUOp))
  val pc = RegInit(0.U(10.W))
  val busy = RegInit(false.B)
  val ins = rom(pc)
  val nCtx = io.nFrames >> 1
  val pos = Reg(UInt(13.W)); val tok = Reg(UInt(16.W)); val genCount = Reg(UInt(16.W))
  val chunkBase = Reg(UInt(13.W)); val chunkRows = Reg(UInt(13.W)); val chunkTotal = Reg(UInt(13.W)); val chunkSize = Reg(UInt(13.W))
  val prompt = VecInit(Microcode.sotSequence.map(_.U(16.W)))
  val promptTok = Wire(Vec(Microcode.promptLen, UInt(16.W)))
  promptTok := prompt; promptTok(1) := io.langToken

  val sDispatch :: sWait :: sIdle :: Nil = Enum(3)
  val state = RegInit(sIdle)
  io.busy := busy
  io.pc := pc
  io.pos := pos
  io.tokenCount := genCount

  def rows(sel: UInt, const: UInt): UInt = MuxLookup(sel, const)(Seq(RowsSel.NFrames.U -> io.nFrames, RowsSel.NCtx.U -> nCtx, RowsSel.Chunk.U -> chunkRows))
  def rowOff(sel: UInt): UInt = MuxLookup(sel, 0.U)(Seq(BaseSel.Chunk.U -> chunkBase, BaseSel.Pos.U -> pos))

  // ---- matmul
  val mm = WireDefault(ins.mm)
  mm.rows := rows(ins.rowsSel, ins.mm.rows)
  mm.frames := Mux(ins.framesSel === 1.U, io.nFrames, ins.mm.frames)
  mm.rowOff := rowOff(ins.rowOffSel)
  io.mm.bits := mm
  io.mm.valid := busy && state === sDispatch && ins.opc === UOpc.MATMUL.U
  // ---- vector
  val vc = WireDefault(ins.vec)
  vc.rows := rows(ins.rowsSel, ins.vec.rows)
  vc.rowOff := rowOff(ins.rowOffSel)
  vc.bBase := ins.vec.bBase + Mux(ins.bBasePos, pos * ins.vec.bStride, 0.U)
  vc.tok := Mux(ins.tokSel, tok, ins.vec.tok)
  io.vec.bits := vc
  io.vec.valid := busy && state === sDispatch && ins.opc === UOpc.VEC.U
  // ---- attention
  val ac = WireDefault(ins.att)
  ac.nQueries := Mux(ins.nQSel === 2.U, nCtx, ins.att.nQueries)
  ac.nKeys := MuxLookup(ins.nKeysSel, ins.att.nKeys)(Seq(KeysSel.NCtx.U -> nCtx, KeysSel.PosPlus1.U -> (pos + 1.U)))
  ac.qPos0 := Mux(ins.qPos0Sel === BaseSel.Pos.U, pos, 0.U)
  io.att.bits := ac
  io.att.valid := busy && state === sDispatch && ins.opc === UOpc.ATTN.U
  // ---- kv context (registered on KVSET)
  val kvReg = Reg(new KVCmd)
  io.kvCmd := kvReg
  val kvFlush = RegInit(false.B); kvFlush := false.B
  io.kvFlush := kvFlush
  // ---- sampler / tokens
  val samplerStart = RegInit(false.B); samplerStart := false.B
  io.samplerStart := samplerStart
  io.samplerFirst := pos === (Microcode.promptLen - 1).U
  val tokV = RegInit(false.B); val tokBits = Reg(UInt(16.W))
  io.tokOut.valid := tokV; io.tokOut.bits := tokBits
  when(io.tokOut.fire) { tokV := false.B }
  val flushPending = RegInit(false.B)
  val waitCycles = Reg(UInt(4.W))

  switch(state) {
    is(sIdle) {
      when(io.start) { busy := true.B; pc := 0.U; state := sDispatch; genCount := 0.U; pos := 0.U; tok := promptTok(0) }
    }
    is(sDispatch) {
      switch(ins.opc) {
        is(UOpc.MATMUL.U) { when(io.mm.fire) { state := sWait; flushPending := ins.flushK; waitCycles := 0.U } }
        is(UOpc.VEC.U) { when(io.vec.fire) { state := sWait; waitCycles := 0.U } }
        is(UOpc.ATTN.U) { when(io.att.fire) { state := sWait; waitCycles := 0.U } }
        is(UOpc.KVSET.U) {
          kvReg := ins.kv
          kvReg.keyOff := Mux(ins.keyOffSel === BaseSel.Pos.U, pos, 0.U)
          pc := pc + 1.U
        }
        is(UOpc.CHUNK_BEGIN.U) {
          val total = rows(ins.imm(1, 0), 0.U)
          chunkTotal := total; chunkBase := 0.U; chunkSize := ins.imm2
          chunkRows := Mux(total < ins.imm2, total, ins.imm2)
          pc := pc + 1.U
        }
        is(UOpc.CHUNK_NEXT.U) {
          val nb = chunkBase + chunkSize
          when(nb < chunkTotal) {
            chunkBase := nb
            chunkRows := Mux(chunkTotal - nb < chunkSize, chunkTotal - nb, chunkSize)
            pc := ins.imm
          }.otherwise { pc := pc + 1.U }
        }
        is(UOpc.JUMP.U) { pc := ins.imm }
        is(UOpc.BR_PROMPT.U) { pc := Mux(pos < (Microcode.promptLen - 1).U, ins.imm, pc + 1.U) }
        is(UOpc.SETPOS.U) { pos := ins.imm; tok := promptTok(0); pc := pc + 1.U }
        is(UOpc.TOK_PROMPT.U) { tok := promptTok((pos + 1.U)(2, 0)); pc := pc + 1.U }
        is(UOpc.NEXTPOS.U) { pos := pos + 1.U; pc := ins.imm }
        is(UOpc.SAMPLE.U) {
          // the LM matmul just finished: sampler.done pulsed during the wait -> token ready now
          val t = io.samplerToken
          when(io.samplerIsEot || genCount === Microcode.sampleLen.U || pos === (cfg.maxTextCtx - 1).U) {
            busy := false.B; state := sIdle
          }.otherwise {
            tok := t; genCount := genCount + 1.U; tokV := true.B; tokBits := t
            pc := pc + 1.U
          }
        }
        is(UOpc.HALT.U) { busy := false.B; state := sIdle }
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
  // arm the sampler when the LM matmul is dispatched
  when(state === sDispatch && ins.opc === UOpc.MATMUL.U && ins.mm.outSink === 2.U && io.mm.fire) { samplerStart := true.B }
}
