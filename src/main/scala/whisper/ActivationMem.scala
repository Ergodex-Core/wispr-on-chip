package whisper

import chisel3._
import chisel3.util._

/** Activation bank. Logical addressing is in 256-bit words (one k-tile of one row, docs/tiling.md);
  * the physical word is 512 bits so an int16 output tile (32 x 16b) is written in one cycle.
  * Reads return the addressed 256-bit half one cycle after `en`. One read + one write port. */
class ActReadPort(val addrBits: Int) extends Bundle {
  val addr = Input(UInt(addrBits.W))      // 256-bit word address
  val en   = Input(Bool())
  val data = Output(UInt(256.W))
}

class ActWrite(val addrBits: Int) extends Bundle {
  val addr = UInt(addrBits.W)             // 256-bit word address (even when wide)
  val wide = Bool()                       // true: write 512 bits at addr (must be even); false: 256 bits
  val data = UInt(512.W)                  // wide data, or low 256 bits for a narrow write
}

class ActBank(val words256: Int) extends Module {
  require(words256 % 2 == 0)
  val addrBits = log2Ceil(words256)
  val io = IO(new Bundle {
    val rd = new ActReadPort(addrBits)
    val wr = Flipped(Valid(new ActWrite(addrBits)))
  })
  val mem = SyncReadMem(words256 / 2, Vec(2, UInt(256.W)))
  val half = RegEnable(io.rd.addr(0), io.rd.en)
  val rdata = mem.read(io.rd.addr >> 1, io.rd.en)
  io.rd.data := rdata(half)
  when(io.wr.valid) {
    val w = io.wr.bits
    val v = Wire(Vec(2, UInt(256.W)))
    v(0) := Mux(w.wide, w.data(255, 0), w.data(255, 0))
    v(1) := Mux(w.wide, w.data(511, 256), w.data(255, 0))
    val mask = Mux(w.wide, "b11".U(2.W), Mux(w.addr(0), "b10".U(2.W), "b01".U(2.W)))
    mem.write(w.addr >> 1, v, mask.asBools)
  }
}

/** Per-row dynamic-quantisation factor (maxabs, uint16) side table. */
class RowFacTable(val rows: Int) extends Module {
  val addrBits = log2Ceil(rows)
  val io = IO(new Bundle {
    val rd = new Bundle {
      val addr = Input(UInt(addrBits.W))
      val en   = Input(Bool())
      val data = Output(UInt(16.W))
    }
    val wr = Flipped(Valid(new Bundle {
      val addr = UInt(addrBits.W)
      val data = UInt(16.W)
    }))
  })
  val mem = SyncReadMem(rows, UInt(16.W))
  io.rd.data := mem.read(io.rd.addr, io.rd.en)
  when(io.wr.valid) { mem.write(io.wr.bits.addr, io.wr.bits.data) }
}
