package minicpm

import java.io.{File, PrintWriter}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.Files
import java.security.MessageDigest

/** One generated tensor image (weights/MANIFEST.json, docs/tiling.md). `base` is its first word address
  * in the address space of its kind. The .bin file holds little-endian 32-bit words; a memory word is
  * `width/32` consecutive 32-bit words (first = least-significant). */
case class WeightTensor(name: String, kind: String, width: Int, depth: Int, base: Int, file: String,
                        sha256: String, shape: Seq[Int]) {
  def linesPerWord: Int = width / 32
  def bits: Long = width.toLong * depth

  def path(repoRoot: String): File = new File(new File(repoRoot, "weights"), file)

  /** Verify the file against the manifest digest (called once per elaboration). */
  def verify(repoRoot: String): Unit = {
    val f = path(repoRoot)
    require(f.exists(), s"weight file missing: $f (run `make weights`)")
    val md = MessageDigest.getInstance("SHA-256")
    val in = new java.io.FileInputStream(f)
    val buf = new Array[Byte](1 << 20)
    var n = in.read(buf)
    while (n > 0) { md.update(buf, 0, n); n = in.read(buf) }
    in.close()
    val got = md.digest().map(b => f"${b & 0xff}%02x").mkString
    require(got == sha256, s"sha256 mismatch for $name: manifest $sha256, file $got")
  }

  /** The 32-bit words of the file (unsigned, stored in Int). */
  def words32(repoRoot: String): Array[Int] = {
    val bytes = Files.readAllBytes(path(repoRoot).toPath)
    require(bytes.length == depth * linesPerWord * 4, s"$name: expected ${depth * linesPerWord} words, got ${bytes.length / 4}")
    val out = new Array[Int](depth * linesPerWord)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out)
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

  /** Write the $readmemh form (one memory word per line, width/4 hex digits) and return its path.
    * The file name carries the source digest, so regenerated weights never reuse a stale image and an
    * unchanged tensor is not rewritten (the layer-level tests load ~100 MB of them per run). */
  def writeReadmemh(repoRoot: String, outDir: File): File = {
    outDir.mkdirs()
    val f = new File(outDir, s"$name.${sha256.take(8)}.mem")
    val expect = depth.toLong * (width / 4 + 1)
    if (f.exists() && f.length() == expect) return f
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
