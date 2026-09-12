# Tiling and memory layout

## Matmul engine (32 × 32, weight-stationary, output-tiled)
* Tile `(kt, nt)` of `W[K, N]` covers `k ∈ [32kt, 32kt+32)`, `n ∈ [32nt, 32nt+32)`.
* Loop order per op: `for nt: for kt: load tile(kt, nt); stream all M rows; accumulate` — the
  accumulator holds `M × 32` int32 partial sums; after the last `kt` the fused requant produces the
  output column block `y[:, 32nt : 32nt+32]`.
* `K` and `N` are multiples of 32 everywhere (mel padded 80→96, vocab 51865→51872).

## Weight ROM word
One memory word = 2048 bits = one *tile-row group*: 8 consecutive `k` rows × 32 `n` columns of int8.
Word address of group `g ∈ 0..3` of tile `(kt, nt)`:
```
addr = ((nt * KT) + kt) * 4 + g            KT = K / 32
byte[r*32 + c] = W[32*kt + 8*g + r][32*nt + c]      r ∈ 0..7, c ∈ 0..31
```
Byte `b` of a word occupies bits `[8b, 8b+8)`. A tile is 4 consecutive words (4 cycles to load).

## Hex file (`weights/<tensor>.hex`, committed): one 32-bit word per line
Line `8*addr + i` (i = 0..7 … 63) holds bytes `[4i, 4i+4)` of memory word `addr`, little-endian
(byte `4i` in bits [0,8)). For 2048-bit words that is 64 lines per word. Lowercase hex, 8 digits, no
prefix. `MANIFEST.json` records shape, kind, scales, word count, sha256 per file.

## Other tensor kinds
* `i32vec` (requant `mult`/`bias`, LN `g`/`b`, `resmult`, GELU smooth): one int32 per line (two's
  complement), padded to a multiple of 32 entries; memory word = 1024 bits = 32 entries
  (`entry i` in bits `[32i, 32i+32)`), word address = entry / 32.
* `i8mat` (positional embeddings `[rows, 384]`): row-major bytes, memory word = 256 bits = 32 bytes
  (one activation tile row), word address = `row * 12 + col/32`.

## Activation banks
Word = 256 bits = 32 int8 (one k-tile of one row). Row stride = `cols / 32` words (12 for 384,
48 for 1536). int16 tensors occupy 2 words per 32 columns (little-endian 16-bit lanes).
Per-row `maxabs` (uint16) lives in a side table indexed by row.

## KV cache (per layer, per head)
* K stored transposed: word = 8 head-dims × 32 keys (so a `K^T` tile-row group is one word);
  address = `((layer*H + h) * (64/8) + d/8) * (n_keys_max/32) + key/32`.
* V stored key-major: word = 8 keys × 32 head-dims; address = `((layer*H + h) * 2 + d/32) * (n_keys_max/8) + key/8`.
Both feed the engine's weight port directly (Q·Kᵀ uses K as the stationary tile, P·V uses V).
