package minicpm

import chisel3._
import chisel3.util._

/** Activation bank. Logical addressing is in 256-bit words (one k-tile of one row, docs/tiling.md);
  * the physical word is 1024 bits so an int16 output tile (32 x 16b, 512 bits) or an int32 tile (1024 bits)
  * is written in one cycle. Reads return the addressed 256-bit quarter one cycle after `en`. One read +
  * one write port. */
class ActReadPort(val addrBits: Int) extends Bundle {
  val addr = Input(UInt(addrBits.W))      // 256-bit word address
  val en   = Input(Bool())
  val data = Output(UInt(256.W))
}

object ActSize { val W256 = 0; val W512 = 1; val W1024 = 2 }

class ActWrite(val addrBits: Int) extends Bundle {
  val addr = UInt(addrBits.W)             // 256-bit word address (aligned to the size)
  val size = UInt(2.W)                    // 0: 256 bits, 1: 512 bits (addr even), 2: 1024 bits (addr % 4 == 0)
  val data = UInt(1024.W)                 // data in the low bits
}

class ActBank(val words256: Int) extends Module {
  require(words256 % 4 == 0)
  val addrBits = log2Ceil(words256)
  val io = IO(new Bundle {
    val rd = new ActReadPort(addrBits)
    val wr = Flipped(Valid(new ActWrite(addrBits)))
  })
  val mem = SyncReadMem(words256 / 4, Vec(4, UInt(256.W)))
  val quarter = RegEnable(io.rd.addr(1, 0), io.rd.en)
  val rdata = mem.read(io.rd.addr >> 2, io.rd.en)
  io.rd.data := rdata(quarter)
  when(io.wr.valid) {
    val w = io.wr.bits
    val v = Wire(Vec(4, UInt(256.W)))
    val q = w.addr(1, 0)
    for (i <- 0 until 4) {
      // 256: the word lands in quarter q; 512: quarters {2k, 2k+1} get data(255,0)/(511,256); 1024: quarter i gets slice i
      v(i) := MuxLookup(w.size, w.data(255, 0))(Seq(
        ActSize.W512.U -> (if (i % 2 == 0) w.data(255, 0) else w.data(511, 256)),
        ActSize.W1024.U -> w.data(256 * i + 255, 256 * i)))
    }
    val mask = MuxLookup(w.size, UIntToOH(q, 4))(Seq(
      ActSize.W512.U -> Mux(q(1), "b1100".U(4.W), "b0011".U(4.W)),
      ActSize.W1024.U -> "b1111".U(4.W)))
    mem.write(w.addr >> 2, v, mask.asBools)
  }
}

/** Per-row dynamic-quantisation factor side table: m16 (bits 15:0) | b (bits 20:16); the effective row
  * factor is m16 << b (docs/numerics.md, dynamic quantisation). */
object RowFac { val bits = 24 }

class RowFacTable(val rows: Int) extends Module {
  val addrBits = log2Ceil(rows)
  val io = IO(new Bundle {
    val rd = new Bundle {
      val addr = Input(UInt(addrBits.W))
      val en   = Input(Bool())
      val data = Output(UInt(RowFac.bits.W))
    }
    val wr = Flipped(Valid(new Bundle {
      val addr = UInt(addrBits.W)
      val data = UInt(RowFac.bits.W)
    }))
  })
  val mem = SyncReadMem(rows, UInt(RowFac.bits.W))
  io.rd.data := mem.read(io.rd.addr, io.rd.en)
  when(io.wr.valid) { mem.write(io.wr.bits.addr, io.wr.bits.data) }
}
