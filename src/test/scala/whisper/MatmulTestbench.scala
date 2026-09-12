package whisper

import chisel3._
import chisel3.util._

/** Engine + weight store + two activation banks + rowfac table + a raw-output memory, with test-side
  * load/readback ports. Bank 0 = inputs, bank 1 = int8/int16 outputs; raw/wide outputs go to rawMem. */
class MatmulTestbench(cfg: WhisperConfig, actWords: Int = 1 << 16, rawWords: Int = 1 << 12) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new MatmulCmd))
    val busy = Output(Bool())
    val cycles = Output(UInt(32.W))
    val actLoad = Flipped(Valid(new ActWrite(log2Ceil(actWords))))            // into bank 0
    val actRead = new ActReadPort(log2Ceil(actWords))                          // from bank 1
    val rowfacLoad = Flipped(Valid(new Bundle { val addr = UInt(13.W); val data = UInt(16.W) }))
    val wsLoad = Flipped(Valid(new Bundle { val space = UInt(2.W); val addr = UInt(24.W); val slice = UInt(6.W); val data = UInt(32.W) }))
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
  // activations
  bankIn.io.rd.addr := eng.io.act.addr
  bankIn.io.rd.en := eng.io.act.en
  eng.io.act.data := bankIn.io.rd.data
  bankIn.io.wr := io.actLoad
  // weights
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
  ws.io.load.valid := io.wsLoad.valid
  ws.io.load.bits.space := io.wsLoad.bits.space
  ws.io.load.bits.addr := io.wsLoad.bits.addr
  ws.io.load.bits.slice := io.wsLoad.bits.slice
  ws.io.load.bits.data := io.wsLoad.bits.data
  // rowfac
  rowfac.io.rd.addr := eng.io.rowfac.addr
  rowfac.io.rd.en := eng.io.rowfac.en
  eng.io.rowfac.data := rowfac.io.rd.data
  rowfac.io.wr := io.rowfacLoad
  // outputs
  val o = eng.io.out
  o.ready := true.B
  val outCount = RegInit(0.U(32.W))
  when(o.fire) { outCount := outCount + 1.U }
  io.outCount := outCount
  val packed8 = Cat(o.bits.data.reverse.map(_(7, 0)))
  val packed16 = Cat(o.bits.data.reverse.map(_(15, 0)))
  bankOut.io.wr.valid := o.fire && (o.bits.mode === MatmulMode.Int8.U || o.bits.mode === MatmulMode.Int16.U)
  bankOut.io.wr.bits.addr := o.bits.addr
  bankOut.io.wr.bits.wide := o.bits.mode === MatmulMode.Int16.U
  bankOut.io.wr.bits.data := Mux(o.bits.mode === MatmulMode.Int16.U, packed16, Cat(0.U(256.W), packed8))
  val rawAddr = o.bits.addr                       // raw/wide: outBase + m*stride + nTile in 1024-bit words
  when(o.fire && (o.bits.mode === MatmulMode.Raw.U || o.bits.mode === MatmulMode.Wide.U)) {
    rawMem.write(rawAddr(log2Ceil(rawWords) - 1, 0), Cat(o.bits.data.reverse.map(_.asUInt)))
  }
  io.rawRead.data := rawMem.read(io.rawRead.addr, true.B)
  bankOut.io.rd <> io.actRead
}
