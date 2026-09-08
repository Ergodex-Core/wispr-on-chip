# Tiling and memory layout

## Matmul engine (32 × 32, weight-stationary, output-tiled)
* Tile `(kt, nt)` of `W[K, N]` covers `k ∈ [32kt, 32kt+32)`, `n ∈ [32nt, 32nt+32)`.
* Loop order per op: `for nt: for kt: load tile(kt, nt); stream all M rows; accumulate` — the
  accumulator holds `M × 32` int32 partial sums; after the last `kt` the fused requant produces the
  output column block `y[:, 32nt : 32nt+32]`.
* `K` and `N` are multiples of 32 everywhere (2048, 256, 6144, 130560 = 4080 tiles).

## Weight ROM word
One memory word = 2048 bits = one *tile-row group*: 8 consecutive `k` rows × 32 `n` columns of int8.
Word address of group `g ∈ 0..3` of tile `(kt, nt)`:
```
addr = ((nt * KT) + kt) * 4 + g            KT = K / 32
byte[r*32 + c] = W[32*kt + 8*g + r][32*nt + c]      r ∈ 0..7, c ∈ 0..31
```
Byte `b` of a word occupies bits `[8b, 8b+8)`. A tile is 4 consecutive words (4 cycles to load).

## Weight images (`weights/<tensor>.bin`, generated, not committed): little-endian 32-bit words
Word `width/32 * addr + i` holds bits `[32i, 32i+32)` of memory word `addr`. `MANIFEST.json` (committed)
records kind, shape, width, depth, word count and sha256 per file plus every op parameter.
The LM head is stored as 16 column slices `lm.00..lm.15` (8192 columns each, the last 7680) that are
contiguous in the w8 address space, so one 4080-tile job spans them; the embedding table is stored as 16
row slices `embed.00..embed.15` contiguous in the i8mat space.

## Other tensor kinds
* `i32vec` (requant `mult`, RMSNorm gains `g`, `embed.resmult`): one int32 per entry, padded to a multiple
  of 32; memory word = 1024 bits = 32 entries (`entry i` in bits `[32i, 32i+32)`), word address = entry / 32.
* `i8mat` (embedding rows `[rows, 2048]`): row-major bytes, memory word = 256 bits = 32 bytes, word
  address = `row * 64 + col/32`.
* `i16mat` (RoPE table `[maxCtx, 128]` = cos[64] | sin[64]): little-endian int16, memory word = 256 bits =
  16 entries, word address = `pos * 8 + i/16`. Shares the 256-bit address space with `i8mat`.

## Activation banks
Logical word = 256 bits = 32 int8 (one k-tile of one row); the physical word is 1024 bits so an int16 output
tile (512 bits) or an int32 output tile (1024 bits) is written in one cycle. Row strides in 256-bit words:
int8 row of 2048 = 64, int16 row of 2048 = 128, int32 row of 2048 = 256, int8/int16/int32 rows of 6144 = 192/384/768,
int16 row of 256 (K projection) = 16. int16 tensors are little-endian 16-bit lanes, int32 tensors 32-bit lanes.
Per-row quantisation factors (`m16 | b<<16`, 24 bits) live in a side table indexed by row.

| bank | contents | words (maxCtx 2048, chunk 512) |
|---|---|---|
| 0 X | int32 residual, all positions | 524288 |
| 1 A8 | int8 norm / attention-out rows (chunk-local), LM input row | 32768 |
| 2 QK16 | int16 q rows @0, int16 k rows @65536 | 73728 |
| 3 Q8 | int8 rotated q rows | 32768 |
| 4 T32 | int16 attention out / int32 o out / int32 down out (chunk-local) | 131072 |
| 5 G16 | int16 gate rows | 196608 |
| 6 U16 | int16 up rows | 196608 |
| 7 A8X | int8 gated rows (down input) | 98304 |

Prefill runs in row chunks of 512 (`CHUNK`); decoding is the same program with a one-row chunk at `pos`.

## KV cache (per layer, per kv head; head_dim 128 = 4 d-tiles; keysMax = maxCtx)
* K stored transposed: tiled `W[k = d (128)][n = key (keysMax)]`; word = 8 head-dims × 32 keys;
  address = `base + (nt*4 + kt)*4 + g` with `nt = key/32`, `kt = d/32`, `g = (d%32)/8`.
* V stored key-major: tiled `W[k = key (keysMax)][n = d (128)]`; word = 8 keys × 32 head-dims;
  address = `base + (nt*keysMaxT + kt)*4 + g` with `nt = d/32`, `kt = key/32`, `g = (key%32)/8`.
* Words per (layer, head) = `4 · keysMax/32 · 4` (1024 for 2048 keys); layout `[L0 K][L0 V][L1 K][L1 V]…`
  (`KVRegion` in KVCache.scala, `gen/chipmap.py`). Both feed the engine's weight port directly
  (Q·Kᵀ uses K as the stationary tile with 4 k-tiles, P·V uses V with 4 n-tiles).
* Writes: V rows arrive as the engine's int8 output beats (row = key, nTile = kvhead·4 + d-tile);
  K rows arrive as the vector unit's RoPE beats and go through a double-buffered 32-key transposer.
