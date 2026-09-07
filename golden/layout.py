"""Memory-layout helpers shared by the vector generators and the e2e harness (docs/tiling.md).

Activation bank: 256-bit words = 32 bytes; row stride in words. int16 tensors are little-endian 16-bit
lanes, 2 words per 32 columns. All functions return/accept uint32 arrays (one 32-bit word per entry).
"""
from __future__ import annotations

import numpy as np


def act8_words(x8: np.ndarray, stride_words: int | None = None) -> np.ndarray:
    """int8 [M, K] -> uint32 words, K/32 words per row (row stride defaults to K/32)."""
    x8 = np.asarray(x8, dtype=np.int8)
    M, K = x8.shape
    assert K % 32 == 0
    sw = stride_words or K // 32
    assert sw >= K // 32
    buf = np.zeros((M, sw * 32), dtype=np.uint8)
    buf[:, :K] = x8.view(np.uint8)
    return buf.reshape(-1).view("<u4")


def act16_words(x16: np.ndarray, stride_words: int | None = None) -> np.ndarray:
    """int16 [M, N] -> uint32 words, 2 words per 32 columns."""
    x16 = np.asarray(x16, dtype="<i2")
    M, N = x16.shape
    assert N % 32 == 0
    sw = stride_words or N // 16
    buf = np.zeros((M, sw * 16), dtype="<i2")
    buf[:, :N] = x16
    return buf.view(np.uint8).reshape(-1).view("<u4")


def raw32_words(x32: np.ndarray) -> np.ndarray:
    """int32 [M, N] -> uint32 words (N words per row)."""
    return np.ascontiguousarray(np.asarray(x32, dtype="<i4")).reshape(-1).view("<u4")


def words_from_act8(words: np.ndarray, M: int, K: int, stride_words: int) -> np.ndarray:
    b = np.asarray(words, dtype="<u4").view(np.uint8).reshape(M, stride_words * 32)
    return b[:, :K].view(np.int8)


def words_from_act16(words: np.ndarray, M: int, N: int, stride_words: int) -> np.ndarray:
    b = np.asarray(words, dtype="<u4").view(np.uint8).reshape(M, stride_words * 32)
    return b[:, : 2 * N].view("<i2")


def write_hex(path, words: np.ndarray):
    with open(path, "w") as f:
        f.write("\n".join(f"{int(w):08x}" for w in np.asarray(words, dtype=np.uint32)) + "\n")


def read_hex(path) -> np.ndarray:
    return np.array([int(l, 16) for l in open(path) if l.strip()], dtype=np.uint32)
