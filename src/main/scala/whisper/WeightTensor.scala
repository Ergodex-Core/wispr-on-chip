package whisper

import java.io.{BufferedReader, File, FileReader, PrintWriter}
import java.security.MessageDigest

/** One committed tensor (see weights/MANIFEST.json, docs/tiling.md). `base` is its first word address
  * in the address space of its kind. The hex file holds one 32-bit word per line; a memory word is
  * `width/32` consecutive lines, little-endian (first line = least-significant 32 bits). */
case class WeightTensor(name: String, kind: String, width: Int, depth: Int, base: Int, file: String,
                        sha256: String, shape: Seq[Int]) {
  def linesPerWord: Int = width / 32
  def bits: Long = width.toLong * depth

  def path(repoRoot: String): File = new File(new File(repoRoot, "weights"), file)

  /** Verify the committed file against the manifest digest (called once per elaboration). */
  def verify(repoRoot: String): Unit = {
    val f = path(repoRoot)
    require(f.exists(), s"weight file missing: $f")
    val md = MessageDigest.getInstance("SHA-256")
    val in = new java.io.FileInputStream(f)
    val buf = new Array[Byte](1 << 20)
    var n = in.read(buf)
    while (n > 0) { md.update(buf, 0, n); n = in.read(buf) }
    in.close()
    val got = md.digest().map(b => f"${b & 0xff}%02x").mkString
    require(got == sha256, s"sha256 mismatch for $name: manifest $sha256, file $got")
  }

  /** The 32-bit words of the hex file (unsigned, stored in Int). */
  def words32(repoRoot: String): Array[Int] = {
    val out = new Array[Int](depth * linesPerWord)
    val r = new BufferedReader(new FileReader(path(repoRoot)), 1 << 20)
    var i = 0
    var line = r.readLine()
    while (line != null) {
      val s = line.trim
      if (s.nonEmpty) { out(i) = java.lang.Integer.parseUnsignedInt(s, 16); i += 1 }
      line = r.readLine()
    }
    r.close()
    require(i == out.length, s"$name: expected ${out.length} words, got $i")
    out
  }

  /** Memory words (width bits each) as BigInts, for RomLiteral and for the equivalence checks. */
  def memWords(repoRoot: String, window: Option[Int] = None): Seq[BigInt] = {
    val w = words32(repoRoot)
    val lpw = linesPerWord
    val n = window.getOrElse(depth)
    (0 until n).map { a =>
      var v = BigInt(0)
      var j = lpw - 1
      while (j >= 0) { v = (v << 32) | (BigInt(w(a * lpw + j)) & 0xffffffffL); j -= 1 }
      v
    }
  }

  /** Write the $readmemh form (one memory word per line, width/4 hex digits) and return its path. */
  def writeReadmemh(repoRoot: String, outDir: File): File = {
    outDir.mkdirs()
    val f = new File(outDir, name + ".mem")
    val w = words32(repoRoot)
    val lpw = linesPerWord
    val pw = new PrintWriter(new java.io.BufferedWriter(new java.io.FileWriter(f), 1 << 20))
    var a = 0
    while (a < depth) {
      var j = lpw - 1
      while (j >= 0) { pw.print(f"${w(a * lpw + j)}%08x"); j -= 1 }
      pw.print('\n')
      a += 1
    }
    pw.close()
    f
  }
}
