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
  val vals = VecInit((0 until 32).map(i => Mux(mask(i), minV, io.in.bits.data(i))))
  // lowest-index max within the beat
  def better(a: (SInt, UInt), b: (SInt, UInt)): (SInt, UInt) = { val take = b._1 > a._1; (Mux(take, b._1, a._1), Mux(take, b._2, a._2)) }
  val (bv, bi) = (0 until 32).map(i => (vals(i), i.U(5.W))).reduceLeft(better)
  val cand = Cat(nt, bi)
  val done = RegInit(false.B); done := false.B
  when(io.start) { have := false.B; best := minV; bestIdx := 0.U }
  when(io.in.fire) {
    when(!have || bv > best) { best := bv; bestIdx := cand; have := true.B }
    when(io.in.bits.last) { done := true.B }
  }
  io.token := bestIdx
  io.isEot := bestIdx === Microcode.eot.U
  io.done := done
}
