# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""UE8M0 block32 scales applied to E4M3 storage before ordinary FP8 MMA."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.experimental import primitives as prims
from .memory import store_shared_u32x4, swizzle_byte_offset


@cute.jit
def source_is_mxfp8(cfg, use_extra):
    if cutlass.const_expr(cfg.kv_cache_format == cfg.extra_kv_cache_format):
        return cutlass.Boolean(cfg.kv_cache_format == "mxfp8")
    # Coordinate caches populate only the issuing lanes. Conversion uses the
    # whole warp, so broadcast the source tag even when lane zero is a hole.
    extra = Int32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=Int32(use_extra),
            offset=0,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.IDX,
        )
    )
    if cutlass.const_expr(cfg.extra_kv_cache_format == "mxfp8"):
        return extra != Int32(0)
    return extra == Int32(0)


@cute.jit
def scale_fp8x4(word: Int32, scale: Int32):
    """Exact power-of-two scaling, then one saturating RNE conversion to E4M3.

    FP32 covers the UE8M0 range. Intermediate overflow saturates as E4M3 does;
    underflow is below E4M3's smallest rounding midpoint. Code 255 is NaN.
    """
    bits = scale << Int32(23)
    bits = Int32(0x00400000) if scale == Int32(0) else bits
    bits = Int32(0x7FC00000) if scale == Int32(255) else bits
    result = Int32(0)
    if scale >= Int32(103) and scale <= Int32(142):
        # Every power of two in [2^-24, 2^15] is exactly representable in
        # FP16. A product losing bits in FP16 is already below E4M3's
        # smallest rounding midpoint; overflow is beyond its saturation
        # limit. Thus the packed half path gives the same single-RNE result.
        result = cute.arch.inline_ptx(
            "{ .reg .b16 a,b,s,r0,r1; .reg .b32 h0,h1,s2; .reg .f32 sf;\n"
            "  mov.b32 {a,b}, {$r0}; mov.b32 sf, {$r1};\n"
            "  cvt.rn.f16.f32 s,sf; mov.b32 s2,{s,s};\n"
            "  cvt.rn.f16x2.e4m3x2 h0,a; cvt.rn.f16x2.e4m3x2 h1,b;\n"
            "  mul.rn.f16x2 h0,h0,s2; mul.rn.f16x2 h1,h1,s2;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r0,h0;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r1,h1;\n"
            "  mov.b32 {$w0},{r0,r1}; }",
            write_only_types=[Int32],
            read_only_args=[word, bits],
        )
    else:
        result = cute.arch.inline_ptx(
            "{ .reg .b16 a, b, h0, h1, h2, h3, r0, r1;\n"
            "  .reg .b32 h01, h23; .reg .f32 sf, f0, f1, f2, f3;\n"
            "  mov.b32 {a,b}, {$r0}; mov.b32 sf, {$r1};\n"
            "  cvt.rn.f16x2.e4m3x2 h01, a; cvt.rn.f16x2.e4m3x2 h23, b;\n"
            "  mov.b32 {h0,h1}, h01; mov.b32 {h2,h3}, h23;\n"
            "  cvt.f32.f16 f0,h0; cvt.f32.f16 f1,h1;\n"
            "  cvt.f32.f16 f2,h2; cvt.f32.f16 f3,h3;\n"
            "  mul.rn.f32 f0,f0,sf; mul.rn.f32 f1,f1,sf;\n"
            "  mul.rn.f32 f2,f2,sf; mul.rn.f32 f3,f3,sf;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 r0,f1,f0;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 r1,f3,f2;\n"
            "  mov.b32 {$w0}, {r0,r1}; }",
            write_only_types=[Int32],
            read_only_args=[word, bits],
        )

    return result


@cute.jit
def scale_fp8x16(w0: Int32, w1: Int32, w2: Int32, w3: Int32, scale: Int32):
    """Share one scale conversion and range decision across a 16-byte vector."""
    a, b, c, d = Int32(0), Int32(0), Int32(0), Int32(0)
    if scale >= Int32(103) and scale <= Int32(142):
        result = cute.arch.inline_ptx(
            "{ .reg .b16 a,b,s,r0,r1; .reg .b32 h0,h1,s2; .reg .f32 sf;\n"
            "  mov.b32 sf, {$r4}; cvt.rn.f16.f32 s,sf; mov.b32 s2,{s,s};\n"
            "  mov.b32 {a,b}, {$r0};\n"
            "  cvt.rn.f16x2.e4m3x2 h0,a; cvt.rn.f16x2.e4m3x2 h1,b;\n"
            "  mul.rn.f16x2 h0,h0,s2; mul.rn.f16x2 h1,h1,s2;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r0,h0;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r1,h1;\n"
            "  mov.b32 {$w0},{r0,r1};\n"
            "  mov.b32 {a,b}, {$r1};\n"
            "  cvt.rn.f16x2.e4m3x2 h0,a; cvt.rn.f16x2.e4m3x2 h1,b;\n"
            "  mul.rn.f16x2 h0,h0,s2; mul.rn.f16x2 h1,h1,s2;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r0,h0;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r1,h1;\n"
            "  mov.b32 {$w1},{r0,r1};\n"
            "  mov.b32 {a,b}, {$r2};\n"
            "  cvt.rn.f16x2.e4m3x2 h0,a; cvt.rn.f16x2.e4m3x2 h1,b;\n"
            "  mul.rn.f16x2 h0,h0,s2; mul.rn.f16x2 h1,h1,s2;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r0,h0;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r1,h1;\n"
            "  mov.b32 {$w2},{r0,r1};\n"
            "  mov.b32 {a,b}, {$r3};\n"
            "  cvt.rn.f16x2.e4m3x2 h0,a; cvt.rn.f16x2.e4m3x2 h1,b;\n"
            "  mul.rn.f16x2 h0,h0,s2; mul.rn.f16x2 h1,h1,s2;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r0,h0;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 r1,h1;\n"
            "  mov.b32 {$w3},{r0,r1};\n"
            "}",
            write_only_types=[Int32, Int32, Int32, Int32],
            read_only_args=[w0, w1, w2, w3, scale << Int32(23)],
        )
        a, b, c, d = result[0], result[1], result[2], result[3]
    else:
        a = scale_fp8x4(w0, scale)
        b = scale_fp8x4(w1, scale)
        c = scale_fp8x4(w2, scale)
        d = scale_fp8x4(w3, scale)
    return a, b, c, d


@cute.jit
def scale_fp8_tile(dst, scales, column, *, rows: cutlass.Constexpr[int]):
    """Scale one SW128 data panel with a SW32 scale box in place."""
    lane = cute.arch.thread_idx()[0] % 32
    for chunk in cutlass.range(lane, rows * 8, 32, unroll=4):
        row, col = chunk // 8, chunk % 8 * 16
        offset = swizzle_byte_offset(row * 128 + col, 128)
        words = cutlass.Pointer(dst + offset, dtype=Int32).load(count=4, alignment=16)
        sf = Int32(
            cutlass.Array(scales, dtype=cutlass.Uint8)[
                swizzle_byte_offset(row * 32 + (column + col) // 32, 32)
            ]
        )
        a, b, c, d = scale_fp8x16(words[0], words[1], words[2], words[3], sf)
        store_shared_u32x4(dst, offset, a, b, c, d)
