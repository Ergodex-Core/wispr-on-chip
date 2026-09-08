package minicpm

import chisel3._
import chisel3.util._

/** Engine + weight store + two activation banks + rowfac table + a raw-output memory, with test-side
  * load/readback ports. Bank 0 = inputs, bank 1 = int8/int16/int32 outputs; raw/wide outputs go to rawMem. */
class MatmulTestbench(cfg: MiniCPMConfig, actWords: Int = 1 << 16, rawWords: Int = 1 << 13) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new MatmulCmd))
    val busy = Output(Bool())
    val cycles = Output(UInt(32.W))
    val actLoad = Flipped(Valid(new ActWrite(log2Ceil(actWords))))            // into bank 0
    val actRead = new ActReadPort(log2Ceil(actWords))                          // from bank 1
    val rowfacLoad = Flipped(Valid(new Bundle { val addr = UInt(13.W); val data = UInt(RowFac.bits.W) }))
    val wsLoad = Flipped(Valid(new WsLoad))
    val rawRead = new Bundle { val addr = Input(UInt(log2Ceil(rawWords).W)); val data = Output(UInt(1024.W)) }
    val outCount = Output(UInt(32.W))
  })
  val eng = Module(new MatmulEngine(cfg))
  val ws = Module(new WeightStore(cfg))
  val bankIn = Module(new ActBank(actWords))
  val bankOut = Module(new ActBank(actWords))
  val rowfac = Module(new RowFacTable(8192))
  val rawMem = SyncReadMem(rawWords, UInt(1024.W))

  eng.io.cmd <> io.cmd
  io.busy := eng.io.busy
  io.cycles := eng.io.cycles
  bankIn.io.rd.addr := eng.io.act.addr
  bankIn.io.rd.en := eng.io.act.en
  eng.io.act.data := bankIn.io.rd.data
  bankIn.io.wr := io.actLoad
  ws.io.w.addr := eng.io.w.addr
  ws.io.w.en := eng.io.w.en
  eng.io.w.data := ws.io.w.data
  ws.io.p.addr := eng.io.mult.addr
  ws.io.p.en := eng.io.mult.en
  eng.io.mult.data := ws.io.p.data
  ws.io.p2.addr := eng.io.bias.addr
  ws.io.p2.en := eng.io.bias.en
  eng.io.bias.data := ws.io.p2.data
  ws.io.t.addr := 0.U
  ws.io.t.en := false.B
  ws.io.load := io.wsLoad
  rowfac.io.rd.addr := eng.io.rowfac.addr
  rowfac.io.rd.en := eng.io.rowfac.en
  eng.io.rowfac.data := rowfac.io.rd.data
  rowfac.io.wr := io.rowfacLoad
  val o = eng.io.out
  o.ready := true.B
  val outCount = RegInit(0.U(32.W))
  when(o.fire) { outCount := outCount + 1.U }
  io.outCount := outCount
  val packed8 = Cat(o.bits.data.reverse.map(_(7, 0)))
  val packed16 = Cat(o.bits.data.reverse.map(_(15, 0)))
  val packed32 = Cat(o.bits.data.reverse.map(_.asUInt))
  val isBank = o.bits.mode === MatmulMode.Int8.U || o.bits.mode === MatmulMode.Int16.U || o.bits.mode === MatmulMode.Int32.U
  bankOut.io.wr.valid := o.fire && isBank
  bankOut.io.wr.bits.addr := o.bits.addr
  bankOut.io.wr.bits.size := MuxLookup(o.bits.mode, ActSize.W256.U)(Seq(MatmulMode.Int16.U -> ActSize.W512.U, MatmulMode.Int32.U -> ActSize.W1024.U))
  bankOut.io.wr.bits.data := MuxLookup(o.bits.mode, Cat(0.U(768.W), packed8))(Seq(MatmulMode.Int16.U -> Cat(0.U(512.W), packed16), MatmulMode.Int32.U -> packed32))
  when(o.fire && (o.bits.mode === MatmulMode.Raw.U || o.bits.mode === MatmulMode.Wide.U)) {
    rawMem.write(o.bits.addr(log2Ceil(rawWords) - 1, 0), packed32)
  }
  io.rawRead.data := rawMem.read(io.rawRead.addr, true.B)
  bankOut.io.rd <> io.actRead
}

/** VectorUnit + weight store (RomInit) + banks (0: A input, 1: B input, 2: output) + rowfac table + a token
  * memory. RoPE-K beats (kvOut) are captured into bank 2 at row*outStride + nTile so they compare like bank output. */
class VectorTestbench(cfg: MiniCPMConfig, actWords: Int = 1 << 15) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new VecCmd))
    val busy = Output(Bool())
    val load = Flipped(Valid(new Bundle { val bank = UInt(2.W); val addr = UInt(log2Ceil(actWords).W); val data = UInt(256.W) }))
    val tokLoad = Flipped(Valid(new Bundle { val addr = UInt(13.W); val data = UInt(18.W) }))
    val read = new Bundle { val addr = Input(UInt(log2Ceil(actWords).W)); val en = Input(Bool()); val data = Output(UInt(256.W)) }
    val rowfacRead = new Bundle { val addr = Input(UInt(13.W)); val en = Input(Bool()); val data = Output(UInt(RowFac.bits.W)) }
    val cycles = Output(UInt(32.W))
    val kvBeats = Output(UInt(32.W))
  })
  val vu = Module(new VectorUnit(cfg))
  val ws = Module(new WeightStore(cfg))
  val banks = Seq.fill(3)(Module(new ActBank(actWords)))
  val rowfac = Module(new RowFacTable(8192))
  val tokMem = SyncReadMem(8192, UInt(18.W))
  vu.io.cmd <> io.cmd
  io.busy := vu.io.busy
  val cycles = RegInit(0.U(32.W))
  when(vu.io.busy) { cycles := cycles + 1.U }
  io.cycles := cycles
  banks(0).io.rd.addr := vu.io.rdA.addr; banks(0).io.rd.en := vu.io.rdA.en; vu.io.rdA.data := banks(0).io.rd.data
  banks(1).io.rd.addr := vu.io.rdB.addr; banks(1).io.rd.en := vu.io.rdB.en; vu.io.rdB.data := banks(1).io.rd.data
  banks(2).io.rd.addr := io.read.addr; banks(2).io.rd.en := io.read.en; io.read.data := banks(2).io.rd.data
  val kvBeats = RegInit(0.U(32.W)); when(vu.io.kvOut.valid) { kvBeats := kvBeats + 1.U }; io.kvBeats := kvBeats
  val kvData = Cat(vu.io.kvOut.bits.data.reverse.map(_(7, 0)))
  // kv beats land at trow * kvStride + nTile (kvStride = int8 words per row, taken from the command's outStride)
  val kvStride = RegEnable(io.cmd.bits.outStride, io.cmd.fire)
  val kvAddr = (vu.io.kvOut.bits.trow * kvStride + vu.io.kvOut.bits.nTile)(log2Ceil(actWords) - 1, 0)
  for ((b, i) <- banks.zipWithIndex) {
    if (i == 2) {
      b.io.wr.valid := vu.io.wr.valid || vu.io.kvOut.valid
      b.io.wr.bits.addr := Mux(vu.io.wr.valid, vu.io.wr.bits.addr, kvAddr)
      b.io.wr.bits.size := Mux(vu.io.wr.valid, vu.io.wr.bits.size, ActSize.W256.U)
      b.io.wr.bits.data := Mux(vu.io.wr.valid, Cat(0.U(512.W), vu.io.wr.bits.data), Cat(0.U(768.W), kvData))
    } else {
      b.io.wr.valid := io.load.valid && io.load.bits.bank === i.U
      b.io.wr.bits.addr := io.load.bits.addr
      b.io.wr.bits.size := ActSize.W256.U
      b.io.wr.bits.data := Cat(0.U(768.W), io.load.bits.data)
    }
  }
  ws.io.t.addr := vu.io.rom.addr; ws.io.t.en := vu.io.rom.en; vu.io.rom.data := ws.io.t.data
  ws.io.w.addr := 0.U; ws.io.w.en := false.B
  ws.io.p.addr := vu.io.p.addr; ws.io.p.en := vu.io.p.en; vu.io.p.data := ws.io.p.data
  ws.io.p2.addr := 0.U; ws.io.p2.en := false.B
  ws.io.load.valid := false.B; ws.io.load.bits := DontCare
  rowfac.io.wr.valid := vu.io.rowfacWr.valid
  rowfac.io.wr.bits.addr := vu.io.rowfacWr.bits.addr
  rowfac.io.wr.bits.data := vu.io.rowfacWr.bits.data
  rowfac.io.rd.addr := io.rowfacRead.addr; rowfac.io.rd.en := io.rowfacRead.en; io.rowfacRead.data := rowfac.io.rd.data
  when(io.tokLoad.valid) { tokMem.write(io.tokLoad.bits.addr, io.tokLoad.bits.data) }
  vu.io.tokRd.data := tokMem.read(vu.io.tokRd.addr, vu.io.tokRd.en)
}

/** Engine + Attention + KVCache + banks (0: Q int8 rows, 1: output int16 rows). */
class AttentionTestbench(cfg: MiniCPMConfig, actWords: Int = 1 << 14) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new AttnCmd))
    val busy = Output(Bool())
    val load = Flipped(Valid(new Bundle { val addr = UInt(log2Ceil(actWords).W); val data = UInt(256.W) }))   // bank 0
    val kvLoad = Flipped(Valid(new Bundle { val addr = UInt(24.W); val data = UInt(2048.W) }))
    val kvLoadMask = Input(UInt(256.W))
    val read = new Bundle { val addr = Input(UInt(log2Ceil(actWords).W)); val en = Input(Bool()); val data = Output(UInt(256.W)) }
    val cycles = Output(UInt(32.W))
    val engCycles = Output(UInt(32.W))
  })
  val eng = Module(new MatmulEngine(cfg))
  val att = Module(new Attention(cfg))
  val kv = Module(new KVCache(cfg, debugPort = true))
  val bankQ = Module(new ActBank(actWords))
  val bankO = Module(new ActBank(actWords))
  att.io.cmd <> io.cmd
  io.busy := att.io.busy
  io.cycles := att.io.cycles
  io.engCycles := eng.io.cycles
  eng.io.cmd <> att.io.eng
  att.io.engBusy := eng.io.busy
  bankQ.io.rd.addr := eng.io.act.addr; bankQ.io.rd.en := eng.io.act.en && eng.io.act.src === 0.U
  att.io.pRd.addr := eng.io.act.addr; att.io.pRd.en := eng.io.act.en && eng.io.act.src === 1.U
  val actSrcQ = RegNext(eng.io.act.src)
  eng.io.act.data := Mux(actSrcQ === 1.U, att.io.pRd.data, bankQ.io.rd.data)
  kv.io.rd.addr := eng.io.w.addr; kv.io.rd.en := eng.io.w.en && eng.io.w.src =/= 0.U
  eng.io.w.data := kv.io.rd.data
  eng.io.mult.data := 0.U; eng.io.bias.data := 0.U; eng.io.rowfac.data := 1.U
  att.io.in <> eng.io.out
  kv.io.in.valid := false.B; kv.io.in.bits := DontCare
  kv.io.cmd := 0.U.asTypeOf(new KVCmd); kv.io.flush := false.B
  kv.io.dbgWr.get.valid := io.kvLoad.valid; kv.io.dbgWr.get.bits.addr := io.kvLoad.bits.addr; kv.io.dbgWr.get.bits.data := io.kvLoad.bits.data
  kv.io.dbgWr.get.bits.mask := io.kvLoadMask
  bankQ.io.wr.valid := io.load.valid; bankQ.io.wr.bits.addr := io.load.bits.addr; bankQ.io.wr.bits.size := ActSize.W256.U
  bankQ.io.wr.bits.data := Cat(0.U(768.W), io.load.bits.data)
  bankO.io.wr.valid := att.io.wr.valid; bankO.io.wr.bits.addr := att.io.wr.bits.addr; bankO.io.wr.bits.size := ActSize.W256.U
  bankO.io.wr.bits.data := Cat(0.U(768.W), att.io.wr.bits.data)
  bankO.io.rd.addr := io.read.addr; bankO.io.rd.en := io.read.en; io.read.data := bankO.io.rd.data
}

/** Engine + VectorUnit + KVCache + weight store: the two paths that fill the cache (V from the engine's int8
  * output, K from the RoPE op's beats through the transposer). */
class KVTestbench(cfg: MiniCPMConfig, actWords: Int = 1 << 14) extends Module {
  val io = IO(new Bundle {
    val mm = Flipped(Decoupled(new MatmulCmd))
    val vec = Flipped(Decoupled(new VecCmd))
    val busy = Output(Bool())
    val kvCmd = Input(new KVCmd)
    val flush = Input(Bool())
    val kvBusy = Output(Bool())
    val actLoad = Flipped(Valid(new Bundle { val bank = UInt(1.W); val addr = UInt(log2Ceil(actWords).W); val data = UInt(256.W) }))
    val rowfacLoad = Flipped(Valid(new Bundle { val addr = UInt(13.W); val data = UInt(RowFac.bits.W) }))
    val kvRead = new Bundle { val addr = Input(UInt(24.W)); val en = Input(Bool()); val data = Output(UInt(2048.W)) }
  })
  val eng = Module(new MatmulEngine(cfg)); val vu = Module(new VectorUnit(cfg)); val ws = Module(new WeightStore(cfg)); val kv = Module(new KVCache(cfg))
  val banks = Seq.fill(2)(Module(new ActBank(actWords))); val rowfac = Module(new RowFacTable(8192))
  eng.io.cmd <> io.mm; vu.io.cmd <> io.vec
  io.busy := eng.io.busy || vu.io.busy
  banks(0).io.rd.addr := eng.io.act.addr; banks(0).io.rd.en := eng.io.act.en; eng.io.act.data := banks(0).io.rd.data
  banks(1).io.rd.addr := vu.io.rdA.addr; banks(1).io.rd.en := vu.io.rdA.en; vu.io.rdA.data := banks(1).io.rd.data
  vu.io.rdB.data := 0.U
  for ((b, i) <- banks.zipWithIndex) {
    b.io.wr.valid := io.actLoad.valid && io.actLoad.bits.bank === i.U
    b.io.wr.bits.addr := io.actLoad.bits.addr; b.io.wr.bits.size := ActSize.W256.U; b.io.wr.bits.data := Cat(0.U(768.W), io.actLoad.bits.data)
  }
  ws.io.w.addr := eng.io.w.addr; ws.io.w.en := eng.io.w.en && eng.io.w.src === 0.U; eng.io.w.data := ws.io.w.data
  ws.io.p.addr := Mux(vu.io.p.en, vu.io.p.addr, eng.io.mult.addr); ws.io.p.en := eng.io.mult.en || vu.io.p.en
  eng.io.mult.data := ws.io.p.data; vu.io.p.data := ws.io.p.data
  ws.io.p2.addr := eng.io.bias.addr; ws.io.p2.en := eng.io.bias.en; eng.io.bias.data := ws.io.p2.data
  ws.io.t.addr := vu.io.rom.addr; ws.io.t.en := vu.io.rom.en; vu.io.rom.data := ws.io.t.data
  ws.io.load.valid := false.B; ws.io.load.bits := DontCare
  vu.io.tokRd.data := 0.U
  rowfac.io.rd.addr := eng.io.rowfac.addr; rowfac.io.rd.en := eng.io.rowfac.en; eng.io.rowfac.data := rowfac.io.rd.data; rowfac.io.wr := io.rowfacLoad
  val engKV = eng.io.out.valid && eng.io.out.bits.sink === 3.U
  eng.io.out.ready := true.B
  kv.io.in.valid := engKV || vu.io.kvOut.valid
  kv.io.in.bits := Mux(engKV, eng.io.out.bits, vu.io.kvOut.bits)
  kv.io.cmd := io.kvCmd; kv.io.flush := io.flush; io.kvBusy := kv.io.busy
  kv.io.rd.addr := io.kvRead.addr; kv.io.rd.en := io.kvRead.en; io.kvRead.data := kv.io.rd.data
}

/** Engine (wide mode) + weight store + bank + rowfac + Sampler; plus a direct raw-beat injection port. */
class SamplerTestbench(cfg: MiniCPMConfig, actWords: Int = 1 << 10) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new MatmulCmd))
    val busy = Output(Bool())
    val actLoad = Flipped(Valid(new ActWrite(log2Ceil(actWords))))
    val rowfacLoad = Flipped(Valid(new Bundle { val addr = UInt(13.W); val data = UInt(RowFac.bits.W) }))
    val inject = Flipped(Valid(new MatmulOut(cfg)))
    val start = Input(Bool())
    val token = Output(UInt(18.W)); val isEos = Output(Bool()); val done = Output(Bool())
  })
  val eng = Module(new MatmulEngine(cfg)); val ws = Module(new WeightStore(cfg)); val bank = Module(new ActBank(actWords))
  val rowfac = Module(new RowFacTable(8192)); val samp = Module(new Sampler(cfg))
  eng.io.cmd <> io.cmd; io.busy := eng.io.busy
  bank.io.rd.addr := eng.io.act.addr; bank.io.rd.en := eng.io.act.en; eng.io.act.data := bank.io.rd.data; bank.io.wr := io.actLoad
  ws.io.w.addr := eng.io.w.addr; ws.io.w.en := eng.io.w.en; eng.io.w.data := ws.io.w.data
  ws.io.p.addr := eng.io.mult.addr; ws.io.p.en := eng.io.mult.en; eng.io.mult.data := ws.io.p.data
  ws.io.p2.addr := eng.io.bias.addr; ws.io.p2.en := eng.io.bias.en; eng.io.bias.data := ws.io.p2.data
  ws.io.t.addr := 0.U; ws.io.t.en := false.B; ws.io.load.valid := false.B; ws.io.load.bits := DontCare
  rowfac.io.rd.addr := eng.io.rowfac.addr; rowfac.io.rd.en := eng.io.rowfac.en; eng.io.rowfac.data := rowfac.io.rd.data; rowfac.io.wr := io.rowfacLoad
  eng.io.out.ready := true.B
  samp.io.in.valid := (eng.io.out.valid && eng.io.out.bits.sink === 2.U) || io.inject.valid
  samp.io.in.bits := Mux(io.inject.valid, io.inject.bits, eng.io.out.bits)
  samp.io.start := io.start
  io.token := samp.io.token; io.isEos := samp.io.isEos; io.done := samp.io.done
}
