package whisper.gen

/** Mirrors gen/chipmap.py (activation bank sizes in 256-bit words). */
object ChipMap {
  val bankWords: Seq[Int] = Seq(9216, 36864, 36864, 18432, 36864, 49152, 24576, 18432)
  val rowsMax: Int = 4096
}
