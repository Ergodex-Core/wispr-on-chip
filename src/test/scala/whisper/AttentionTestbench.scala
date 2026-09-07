package whisper

import chisel3._
import chisel3.util._

/** Engine + Attention + KVCache + banks (0: Q int8 rows, 1: output int16 rows). */
class AttentionTestbench(cfg: WhisperConfig, actWords: Int = 1 << 14) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new AttnCmd))
    val busy = Output(Bool())
    val load = Flipped(Valid(new Bundle { val addr = UInt(log2Ceil(actWords).W); val data = UInt(256.W) }))   // bank 0
    val kvLoad = Flipped(Valid(new Bundle { val addr = UInt(20.W); val data = UInt(2048.W) }))
    val read = new Bundle { val addr = Input(UInt(log2Ceil(actWords).W)); val en = Input(Bool()); val data = Output(UInt(256.W)) }
    val cycles = Output(UInt(32.W))
    val engCycles = Output(UInt(32.W))
  })
  val eng = Module(new MatmulEngine(cfg))
  val att = Module(new Attention(cfg))
  val kv = Module(new KVCache(cfg))
  val bankQ = Module(new ActBank(actWords))
  val bankO = Module(new ActBank(actWords))
  att.io.cmd <> io.cmd
  io.busy := att.io.busy
  io.cycles := att.io.cycles
  io.engCycles := eng.io.cycles
  eng.io.cmd <> att.io.eng
  att.io.engBusy := eng.io.busy
  // activations: src 0 -> bank Q, src 1 -> P buffer
  bankQ.io.rd.addr := eng.io.act.addr; bankQ.io.rd.en := eng.io.act.en && eng.io.act.src === 0.U
  att.io.pRd.addr := eng.io.act.addr; att.io.pRd.en := eng.io.act.en && eng.io.act.src === 1.U
  val actSrcQ = RegNext(eng.io.act.src)
  eng.io.act.data := Mux(actSrcQ === 1.U, att.io.pRd.data, bankQ.io.rd.data)
  // weights: KV cache only
  kv.io.rd.addr := eng.io.w.addr; kv.io.rd.en := eng.io.w.en && eng.io.w.src =/= 0.U
  eng.io.w.data := kv.io.rd.data
  eng.io.mult.data := 0.U; eng.io.bias.data := 0.U; eng.io.rowfac.data := 1.U
  // outputs -> attention
  att.io.in <> eng.io.out
  // KV writes: none from the engine here; test loads
  kv.io.in.valid := false.B; kv.io.in.bits := DontCare
  kv.io.cmd := 0.U.asTypeOf(new KVCmd); kv.io.flush := false.B
  kv.io.dbgWr.valid := io.kvLoad.valid; kv.io.dbgWr.bits.addr := io.kvLoad.bits.addr; kv.io.dbgWr.bits.data := io.kvLoad.bits.data
  // bank Q load, bank O written by attention
  bankQ.io.wr.valid := io.load.valid; bankQ.io.wr.bits.addr := io.load.bits.addr; bankQ.io.wr.bits.wide := false.B
  bankQ.io.wr.bits.data := Cat(0.U(256.W), io.load.bits.data)
  bankO.io.wr.valid := att.io.wr.valid; bankO.io.wr.bits.addr := att.io.wr.bits.addr; bankO.io.wr.bits.wide := false.B
  bankO.io.wr.bits.data := Cat(0.U(256.W), att.io.wr.bits.data)
  bankO.io.rd.addr := io.read.addr; bankO.io.rd.en := io.read.en; io.read.data := bankO.io.rd.data
}
