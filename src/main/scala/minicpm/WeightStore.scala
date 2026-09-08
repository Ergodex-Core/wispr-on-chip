package minicpm

import chisel3._
import chisel3.util._
import chisel3.util.experimental.loadMemoryFromFileInline

/** Synchronous read port: data valid one cycle after `en`; holds while `en` is low. */
class RomReadPort(val addrBits: Int, val dataBits: Int) extends Bundle {
  val addr = Input(UInt(addrBits.W))
  val en   = Input(Bool())
  val data = Output(UInt(dataBits.W))
}

/** One bank = one tensor. Same contents from the same weight image under all three backends. */
class WeightBank(t: WeightTensor, backend: WeightBackend, cfg: MiniCPMConfig, window: Option[Int] = None) extends Module {
  val depth = window.getOrElse(t.depth)
  val io = IO(new Bundle {
    val rd = new RomReadPort(log2Ceil(depth max 2), t.width)
    val rd2 = new RomReadPort(log2Ceil(depth max 2), t.width)   // second read port (params)
    val wr = Flipped(Valid(new Bundle {        // Sram backend only (ignored otherwise)
      val addr  = UInt(log2Ceil(depth max 2).W)
      val slice = UInt(log2Ceil(t.linesPerWord max 2).W)   // which 32-bit slice of the word
      val data  = UInt(32.W)
    }))
  })
  override def desiredName = s"WeightBank_${t.name.replace('.', '_')}_${backend.toString}"
  t.verify(cfg.repoRoot)

  backend match {
    case RomLiteral =>
      require(t.width.toLong * depth <= cfg.romLiteralMaxBits,
        s"RomLiteral bank ${t.name} is ${t.width.toLong * depth} bits > romLiteralMaxBits=${cfg.romLiteralMaxBits}")
      val rom = VecInit(t.memWords(cfg.repoRoot, Some(depth)).map(_.U(t.width.W)))
      io.rd.data := RegEnable(rom(io.rd.addr), io.rd.en)
      io.rd2.data := RegEnable(rom(io.rd2.addr), io.rd2.en)
    case RomInit =>
      require(window.isEmpty, "RomInit banks always hold the whole tensor")
      val mem = SyncReadMem(depth, UInt(t.width.W))
      val f = t.writeReadmemh(cfg.repoRoot, new java.io.File(cfg.repoRoot, "target/readmemh"))
      loadMemoryFromFileInline(mem, f.getAbsolutePath)
      io.rd.data := mem.read(io.rd.addr, io.rd.en)
      io.rd2.data := mem.read(io.rd2.addr, io.rd2.en)
    case Sram =>
      val mem = SyncReadMem(depth, Vec(t.linesPerWord, UInt(32.W)))
      val rdata = mem.read(io.rd.addr, io.rd.en)
      io.rd.data := rdata.asUInt
      io.rd2.data := mem.read(io.rd2.addr, io.rd2.en).asUInt
      when(io.wr.valid) {
        val v = Wire(Vec(t.linesPerWord, UInt(32.W)))
        v.foreach(_ := io.wr.bits.data)
        mem.write(io.wr.bits.addr, v, UIntToOH(io.wr.bits.slice, t.linesPerWord).asBools)
      }
  }
}

/** Flat address spaces, one per tensor kind (w8: 2048-bit tile-row groups; i32vec: 1024-bit param words;
  * i8mat/i16mat: 256-bit rows). Each space is a mux over its banks, selected by address range. */
class WeightSpace(tensors: Seq[WeightTensor], width: Int, addrBits: Int, backend: WeightBackend, cfg: MiniCPMConfig)
    extends Module {
  val totalWords = tensors.map(t => t.base + t.depth).foldLeft(1)(_ max _)
  require(totalWords <= (1 << addrBits), s"space of $width-bit words needs ${log2Ceil(totalWords)} address bits > $addrBits")
  val io = IO(new Bundle {
    val rd = new RomReadPort(addrBits, width)
    val rd2 = new RomReadPort(addrBits, width)
    val wr = Flipped(Valid(new Bundle {
      val addr  = UInt(addrBits.W)
      val slice = UInt(log2Ceil((width / 32) max 2).W)
      val data  = UInt(32.W)
    }))
  })
  override def desiredName = s"WeightSpace${width}_${backend.toString}"
  val banks = tensors.map(t => Module(new WeightBank(t, backend, cfg)))
  val sel = tensors.zip(banks).map { case (t, b) =>
    val hit = io.rd.addr >= t.base.U && io.rd.addr < (t.base + t.depth).U
    b.io.rd.addr := (io.rd.addr - t.base.U)(b.io.rd.addrBits - 1, 0)
    b.io.rd.en := io.rd.en && hit
    b.io.wr.valid := io.wr.valid && io.wr.bits.addr >= t.base.U && io.wr.bits.addr < (t.base + t.depth).U
    b.io.wr.bits.addr := (io.wr.bits.addr - t.base.U)(b.io.rd.addrBits - 1, 0)
    b.io.wr.bits.slice := io.wr.bits.slice
    b.io.wr.bits.data := io.wr.bits.data
    hit
  }
  if (banks.isEmpty) {
    io.rd.data := 0.U
    io.rd2.data := 0.U
  } else {
    val selReg = RegEnable(VecInit(sel).asUInt, io.rd.en)
    io.rd.data := Mux1H(selReg.asBools, banks.map(_.io.rd.data))
    val sel2 = tensors.zip(banks).map { case (t, b) =>
      val hit = io.rd2.addr >= t.base.U && io.rd2.addr < (t.base + t.depth).U
      b.io.rd2.addr := (io.rd2.addr - t.base.U)(b.io.rd2.addrBits - 1, 0)
      b.io.rd2.en := io.rd2.en && hit
      hit
    }
    val sel2Reg = RegEnable(VecInit(sel2).asUInt, io.rd2.en)
    io.rd2.data := Mux1H(sel2Reg.asBools, banks.map(_.io.rd2.data))
  }
}

/** The weight store: three address spaces (w8 / i32vec / i8mat+i16mat). */
class WeightStore(cfg: MiniCPMConfig) extends Module {
  val ts = cfg.tensors
  val wT = ts.filter(_.kind == "w8")
  val pT = ts.filter(_.kind == "i32vec")
  val tT = ts.filter(t => t.kind == "i8mat" || t.kind == "i16mat")
  val wS = Module(new WeightSpace(wT, cfg.weightWordBits, cfg.wAddrBits, cfg.weightBackend, cfg))
  val pS = Module(new WeightSpace(pT, cfg.paramWordBits, cfg.pAddrBits, cfg.weightBackend, cfg))
  val tS = Module(new WeightSpace(tT, cfg.actWordBits, cfg.tAddrBits, cfg.weightBackend, cfg))
  val io = IO(new Bundle {
    val w = new RomReadPort(cfg.wAddrBits, cfg.weightWordBits)
    val p = new RomReadPort(cfg.pAddrBits, cfg.paramWordBits)
    val p2 = new RomReadPort(cfg.pAddrBits, cfg.paramWordBits)   // second param port
    val t = new RomReadPort(cfg.tAddrBits, cfg.actWordBits)
    /** 32-bit load port (Sram backend): space 0=w,1=p,2=t */
    val load = Flipped(Valid(new WsLoad))
  })
  io.w <> wS.io.rd
  io.p <> pS.io.rd
  io.p2 <> pS.io.rd2
  io.t <> tS.io.rd
  wS.io.rd2.addr := 0.U; wS.io.rd2.en := false.B
  tS.io.rd2.addr := 0.U; tS.io.rd2.en := false.B
  Seq((0, wS), (1, pS), (2, tS)).foreach { case (i, sp) =>
    sp.io.wr.valid := io.load.valid && io.load.bits.space === i.U
    sp.io.wr.bits.addr := io.load.bits.addr
    sp.io.wr.bits.slice := io.load.bits.slice
    sp.io.wr.bits.data := io.load.bits.data
  }
}

class WsLoad extends Bundle {
  val space = UInt(2.W)
  val addr  = UInt(24.W)
  val slice = UInt(6.W)
  val data  = UInt(32.W)
}
