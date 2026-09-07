package whisper

import chisel3._
import chisel3.util._

/** VectorUnit + weight store (RomInit) + banks (0: A input, 1: B input, 2: output) + rowfac table. */
class VectorTestbench(cfg: WhisperConfig, actWords: Int = 1 << 14) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new VecCmd))
    val busy = Output(Bool())
    val load = Flipped(Valid(new Bundle { val bank = UInt(2.W); val addr = UInt(log2Ceil(actWords).W); val data = UInt(256.W) }))
    val read = new Bundle { val addr = Input(UInt(log2Ceil(actWords).W)); val en = Input(Bool()); val data = Output(UInt(256.W)) }
    val rowfacRead = new Bundle { val addr = Input(UInt(13.W)); val en = Input(Bool()); val data = Output(UInt(16.W)) }
    val cycles = Output(UInt(32.W))
  })
  val vu = Module(new VectorUnit(cfg))
  val ws = Module(new WeightStore(cfg))
  val banks = Seq.fill(3)(Module(new ActBank(actWords)))
  val rowfac = Module(new RowFacTable(8192))
  vu.io.cmd <> io.cmd
  io.busy := vu.io.busy
  val cycles = RegInit(0.U(32.W))
  when(vu.io.busy) { cycles := cycles + 1.U }
  io.cycles := cycles
  // reads
  banks(0).io.rd.addr := vu.io.rdA.addr; banks(0).io.rd.en := vu.io.rdA.en; vu.io.rdA.data := banks(0).io.rd.data
  banks(1).io.rd.addr := vu.io.rdB.addr; banks(1).io.rd.en := vu.io.rdB.en; vu.io.rdB.data := banks(1).io.rd.data
  banks(2).io.rd.addr := io.read.addr; banks(2).io.rd.en := io.read.en; io.read.data := banks(2).io.rd.data
  // writes: VU -> bank 2 ; test load -> banks 0/1
  for ((b, i) <- banks.zipWithIndex) {
    b.io.wr.valid := (if (i == 2) vu.io.wr.valid else io.load.valid && io.load.bits.bank === i.U)
    b.io.wr.bits.addr := (if (i == 2) vu.io.wr.bits.addr else io.load.bits.addr)
    b.io.wr.bits.wide := false.B
    b.io.wr.bits.data := Cat(0.U(256.W), (if (i == 2) vu.io.wr.bits.data else io.load.bits.data))
  }
  // weight store
  ws.io.t.addr := vu.io.rom.addr; ws.io.t.en := vu.io.rom.en; vu.io.rom.data := ws.io.t.data
  ws.io.w.addr := vu.io.w.addr; ws.io.w.en := vu.io.w.en; vu.io.w.data := ws.io.w.data
  ws.io.p.addr := vu.io.p.addr; ws.io.p.en := vu.io.p.en; vu.io.p.data := ws.io.p.data
  ws.io.p2.addr := vu.io.p2.addr; ws.io.p2.en := vu.io.p2.en; vu.io.p2.data := ws.io.p2.data
  ws.io.load.valid := false.B; ws.io.load.bits := DontCare
  // rowfac
  rowfac.io.wr.valid := vu.io.rowfacWr.valid
  rowfac.io.wr.bits.addr := vu.io.rowfacWr.bits.addr
  rowfac.io.wr.bits.data := vu.io.rowfacWr.bits.data
  rowfac.io.rd.addr := io.rowfacRead.addr; rowfac.io.rd.en := io.rowfacRead.en; io.rowfacRead.data := rowfac.io.rd.data
}
