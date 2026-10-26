# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Inline-scale NVFP4 sparse cache loads into FP8 MMA tiles.

The D512 record contains 256 packed E2M1 bytes followed by 32 E4M3 scales
in block order, including the rotary channels (DS v4.1). The D576 mixed
record inserts 64 unscaled E4M3 RoPE bytes before the scales and permutes
scale block s to 320 + 8*(s&3) + (s>>2). Neither has a global scale.
Conversion uses the existing mixed-precision decode
primitive; only the selected tile is expanded, directly into shared memory.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from ...fmha_decode.fmha_decode_resources.tmem_transformed_kv import (
    _has_packed_e4m3_e2m1_mul,
)
from ...fmha_decode.fmha_decode_resources.smem_transformed_kv import (
    _cvt_e2m1_word_to_e4m3_words,
)


NVFP4_DATA_BYTES = 256
NVFP4_SCALE_BYTES = 32


@cute.jit
def nvfp4_scale_byte_offset(block, rope_dim: cutlass.Constexpr[int] = 64):
    """Offset within the scale region; only the mixed format permutes it."""
    if cutlass.const_expr(rope_dim == 0):
        return block
    return 8 * (block % 4) + block // 4


def nvfp4_row_bytes(rope_dim: int) -> int:
    """Packed D512, optional unscaled FP8 tail, and block16 E4M3 scales."""
    return NVFP4_DATA_BYTES + rope_dim + NVFP4_SCALE_BYTES


@cute.jit
def convert_nvfp4x8(word: Int32, scale: Int32):
    """Scale dense E2M1 nibbles with the decode path's packed FP8 primitive."""
    if cutlass.const_expr(not _has_packed_e4m3_e2m1_mul()):
        # Older DSL builds lack packed multiply even with a newer system
        # toolkit. Convert dense nibbles directly instead of unpacking to
        # bytes only to repack them inside the multiply fallback.
        return _cvt_e2m1_word_to_e4m3_words(word, scale)
    scales = (scale & Int32(255)) * Int32(0x01010101)
    # Each 16-bit half holds four contiguous E2M1 nibbles. CUDA 13.4's
    # dense packed operand avoids padded-nibble masks and output permutations.
    values = cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi;\n"
        "  mov.b32 {lo, hi}, {$r0};\n"
        "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w0}, lo, {$r1};\n"
        "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w1}, hi, {$r1};\n"
        "}",
        write_only_types=[Int32, Int32],
        read_only_args=[word, scales],
    )
    return values[0], values[1]
