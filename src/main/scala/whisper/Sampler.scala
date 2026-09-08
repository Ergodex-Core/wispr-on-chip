package whisper

import chisel3._
import chisel3.util._
import whisper.generated.Microcode

/** Streaming argmax over the LM-head logits (wide beats of 32 int32, n-tiles in ascending order).
  * Applies whisper's suppress list (+ blank suppression on the first sampled position) and pads.
  * Ties resolve to the lowest index. */
class Sampler(cfg: WhisperConfig) extends Module {
  val NT = cfg.nVocabPad / 32
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new MatmulOut(cfg)))
    val first = Input(Bool())            // first sampled position (suppress blank)
    val start = Input(Bool())            // reset the running max before a new logits row
    val token = Output(UInt(16.W))
    val isEot = Output(Bool())
    val done = Output(Bool())            // pulse when the last n-tile has been consumed
  })
  // suppress mask ROM: one 32-bit word per n-tile
  val supp = Array.fill(NT)(0L)
  for (t <- Microcode.suppressTokens) supp(t / 32) |= (1L << (t % 32))
  for (t <- 51865 until cfg.nVocabPad) supp(t / 32) |= (1L << (t % 32))
  val blank = Array.fill(NT)(0L)
  for (t <- Microcode.suppressBlank) blank(t / 32) |= (1L << (t % 32))
  val suppRom = VecInit(supp.map(_.U(32.W)))
  val blankRom = VecInit(blank.map(_.U(32.W)))
  val best = Reg(SInt(32.W)); val bestIdx = Reg(UInt(16.W)); val have = RegInit(false.B)
  io.in.ready := true.B
  val nt = io.in.bits.nTile
  val mask = suppRom(nt) | Mux(io.first, blankRom(nt), 0.U)
  val minV = (-(BigInt(1) << 31)).S(32.W)
  // Lowest-index max within the beat as a balanced tree ("take b only if strictly greater" keeps the
  // left = lower index on ties), pipelined in two register stages so the logic depth per cycle is
  // 2-3 compare levels (a 31-deep serial chain synthesised to ~9.5 ns on ASAP7).
  def better(a: (SInt, UInt), b: (SInt, UInt)): (SInt, UInt) = { val take = b._1 > a._1; (Mux(take, b._1, a._1), Mux(take, b._2, a._2)) }
  def tree(xs: Seq[(SInt, UInt)]): Seq[(SInt, UInt)] = xs.grouped(2).map { case Seq(a, b) => better(a, b); case Seq(a) => a }.toSeq
  // stage 0 -> 1: mask, first two tree levels (32 -> 8)
  val v1 = RegNext(io.in.fire, false.B); val nt1 = RegNext(nt); val last1 = RegNext(io.in.bits.last)
  val l0 = (0 until 32).map(i => (Mux(mask(i), minV, io.in.bits.data(i)), i.U(5.W)))
  val l2 = tree(tree(l0))
  val c1v = RegNext(VecInit(l2.map(_._1))); val c1i = RegNext(VecInit(l2.map(_._2)))
  // stage 1 -> 2: remaining three levels (8 -> 1)
  val v2 = RegNext(v1, false.B); val nt2 = RegNext(nt1); val last2 = RegNext(last1)
  val (bv2, bi2) = tree(tree(tree(c1v.zip(c1i)))).head
  val bv = RegNext(bv2); val bi = RegNext(bi2)
  val v3 = RegNext(v2, false.B); val nt3 = RegNext(nt2); val last3 = RegNext(last2)
  val cand = Cat(nt3(10, 0), bi)
  val done = RegInit(false.B); done := false.B
  when(io.start) { have := false.B; best := minV; bestIdx := 0.U }
  when(v3) {
    when(!have || bv > best) { best := bv; bestIdx := cand; have := true.B }
    when(last3) { done := true.B }
  }
  io.token := bestIdx
  io.isEot := bestIdx === Microcode.eot.U
  io.done := done
}
