# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Gather token-interleaved NVFP4/per-tensor FP8 caches into shared FP8 MMA tiles."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.experimental import cuda, primitives as prims

from ...tensor_map import create_tensor_map_tiled_from_view
from ..helpers.gather import gather4, select_gather_map
from .layout import cache_record_size
from .nvfp4 import (
    convert_nvfp4x8,
    nvfp4_scale_byte_offset,
    nvfp4_row_bytes,
    NVFP4_DATA_BYTES,
)
from .memory import store_shared_u32x4, swizzle_byte_offset


@cute.jit
def make_whole_record_map(pool, record_bytes: cutlass.Constexpr[int]):
    """Native record extent with a gather4-aligned shared-memory row pitch."""
    word_bytes = 4 if record_bytes > 512 else 2
    raw_bytes = (record_bytes + 31) // 32 * 32
    packed = cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint32 if word_bytes == 4 else cutlass.Uint16,
            pool.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        cute.make_layout(
            (record_bytes // word_bytes, pool.shape[2]),
            stride=(1, pool.stride[2] // word_bytes),
        ),
    )
    return create_tensor_map_tiled_from_view(
        packed,
        box_dims=(raw_bytes // word_bytes, 1),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.none,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
    )


@cute.jit
def make_scaled_kv_maps(
    primary,
    extra,
    rope_dim: cutlass.Constexpr[int],
    whole_nope: cutlass.Constexpr[bool] = False,
    primary_format: cutlass.Constexpr[str] = "nvfp4",
    extra_format: cutlass.Constexpr[str] = "nvfp4",
):
    nope = 512
    maps = ()
    for pool, fmt in ((primary, primary_format), (extra, extra_format)):
        fp8 = fmt == "fp8"
        data = cute.make_tensor(
            pool.iterator,
            cute.make_layout(
                (((nope + 127) // 128) * 128, pool.shape[2]),
                stride=(1, pool.stride[2] * (1 if fp8 else 2)),
            ),
        )
        scales = cute.make_tensor(
            pool.iterator + (0 if fp8 else nope // 2 + rope_dim),
            cute.make_layout(
                (16 if fp8 else 32, pool.shape[2]), stride=(1, pool.stride[2])
            ),
        )
        rope = cute.make_tensor(
            pool.iterator + (nope // 2 if rope_dim else 0),
            cute.make_layout((64, pool.shape[2]), stride=(1, pool.stride[2])),
        )
        if cutlass.const_expr(whole_nope):
            data_map = make_whole_record_map(pool, cache_record_size(fmt, rope_dim))
        elif cutlass.const_expr(fp8):
            data_map = create_tensor_map_tiled_from_view(
                data,
                box_dims=(128, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        else:
            data_map = create_tensor_map_tiled_from_view(
                data,
                box_dims=(128, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                dtype=cutlass.Float4E2M1FNx2,
                tma_format=cuda.TensorMapDataFormat.B4X16_P64,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        maps += (
            data_map,
            create_tensor_map_tiled_from_view(
                scales,
                box_dims=(32, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s32b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            ),
            create_tensor_map_tiled_from_view(
                rope,
                box_dims=(64, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s64b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            ),
        )
    return maps[0], maps[3], maps[1], maps[4], maps[2], maps[5]


@cute.jit
def convert_nvfp4_panel(
    dst,
    scales,
    column,
    *,
    rows: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
):
    lane = cute.arch.thread_idx()[0] % 32
    for chunk in cutlass.range(lane, rows * 8, 32, unroll=4):
        row, col = chunk // 8, chunk % 8 * 16
        logical_col = column + col
        if logical_col < 512:
            byte = swizzle_byte_offset(row * 128 + col, 128)
            packed = cutlass.Pointer(dst + byte, dtype=Int32).load(count=2, alignment=8)
            block = logical_col // 16
            scale_index = nvfp4_scale_byte_offset(block, rope_dim)
            scale_byte = cutlass.Array(scales, dtype=cutlass.Uint8)[
                swizzle_byte_offset(row * 32 + scale_index, 32)
            ]
            a, b = convert_nvfp4x8(packed[0], Int32(scale_byte))
            c, d = convert_nvfp4x8(packed[1], Int32(scale_byte))
            store_shared_u32x4(dst, byte, a, b, c, d)


@cute.jit
def begin_raw_load(barrier, count):
    state = Int64(0)
    if cute.arch.thread_idx()[0] % 32 == 0:
        state = prims.mbarrier_arrive_expect_tx(barrier, Int32(count))
    cute.arch.sync_warp()
    return state


@cute.jit
def wait_raw_load(barrier, state):
    if cute.arch.thread_idx()[0] % 32 == 0:
        while not prims.mbarrier_try_wait(barrier, state):
            pass
    cute.arch.sync_warp()


@cute.jit
def gather_cached(
    dst,
    primary_map,
    extra_map,
    column,
    barrier,
    cached,
    base,
    *,
    rows: cutlass.Constexpr[int],
    width,
):
    lane = cute.arch.thread_idx()[0] % 32
    if lane < rows // 4:
        raw = cached[base]
        coords = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            row = cached[base + j] & Int32(0x7FFFFFFF)
            coords[j] = row if row != Int32(0x7FFFFFFF) else Int32(-1)
        descriptor = select_gather_map(primary_map, extra_map, raw < 0)
        gather4(
            dst + lane * 4 * width,
            descriptor,
            column,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            barrier,
        )


@cute.jit
def load_scaled_panel(
    dst,
    scratch,
    barrier,
    cached,
    cache_base,
    column,
    data_map,
    extra_data_map,
    scale_map,
    extra_scale_map,
    rope_map,
    extra_rope_map,
    *,
    rows: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    rope_only: cutlass.Constexpr[bool],
    load_scales: cutlass.Constexpr[bool],
    fp8=False,
):
    if cutlass.const_expr(rope_only):
        state = begin_raw_load(barrier, rows * 64)
        gather_cached(
            dst,
            rope_map,
            extra_rope_map,
            Int32(0),
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=64,
        )
        wait_raw_load(barrier, state)
    else:
        # B4X16_P64 writes 8 packed bytes followed by 8 pad bytes per 16
        # values. Its completion count is the packed bytes, not padded SMEM.
        state = begin_raw_load(
            barrier, rows * (128 if fp8 else 64 + (32 if load_scales else 0))
        )
        gather_cached(
            dst,
            data_map,
            extra_data_map,
            column,
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=128,
        )
        if load_scales and not fp8:
            gather_cached(
                scratch,
                scale_map,
                extra_scale_map,
                Int32(0),
                barrier,
                cached,
                cache_base,
                rows=rows,
                width=32,
            )
        wait_raw_load(barrier, state)
        if not fp8:
            convert_nvfp4_panel(dst, scratch, column, rows=rows, rope_dim=rope_dim)


@cute.jit
def load_scaled_panels(
    destinations,
    columns,
    scratch,
    barrier,
    cached,
    cache_base,
    data_map,
    extra_data_map,
    scale_map,
    extra_scale_map,
    rope_map,
    extra_rope_map,
    *,
    rows: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    separate_rope: cutlass.Constexpr[bool],
    fp8=False,
):
    """Overlap all feature-panel gathers before one raw-data wait."""
    panels = len(columns)
    state = begin_raw_load(
        barrier,
        rows
        * (
            panels * (128 if fp8 else 64)
            + (0 if fp8 else 32)
            + (64 if separate_rope else 0)
        ),
    )
    for part in cutlass.range_constexpr(panels):
        gather_cached(
            destinations[part],
            data_map,
            extra_data_map,
            columns[part],
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=128,
        )
    if not fp8:
        gather_cached(
            scratch,
            scale_map,
            extra_scale_map,
            Int32(0),
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=32,
        )
    if cutlass.const_expr(separate_rope):
        gather_cached(
            destinations[panels],
            rope_map,
            extra_rope_map,
            Int32(0),
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=64,
        )
    wait_raw_load(barrier, state)
    for part in cutlass.range_constexpr(panels):
        if not fp8:
            convert_nvfp4_panel(
                destinations[part], scratch, columns[part], rows=rows, rope_dim=rope_dim
            )


@cute.jit
def issue_whole_record(
    dst, barrier, cached, data_map, extra_data_map, *, issuer, row_bytes
):
    """One gather per quad for a complete NVFP4 or per-tensor FP8 storage record."""
    begin_raw_load(barrier, Int32(64) * row_bytes)
    gather_cached(
        dst + issuer * Int32(64) * row_bytes,
        data_map,
        extra_data_map,
        Int32(0),
        barrier,
        cached,
        0,
        rows=64,
        width=row_bytes,
    )


@cute.jit
def convert_nvfp4_record(
    dst,
    thread_idx,
    *,
    stage_bytes: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    raw=None,
    raw_empty=None,
):
    """Snapshot two16-value rows per lane, then write conflict-free panels."""
    row_bytes = nvfp4_row_bytes(rope_dim)
    if cutlass.const_expr(raw is None):
        raw = dst
    packed = cutlass.Array(Int32, 32, space=cutlass.AddressSpace.rmem)
    scales = cutlass.Array(Int32, 8, space=cutlass.AddressSpace.rmem)
    rope = cutlass.Array(Int32, 8, space=cutlass.AddressSpace.rmem)
    lane = thread_idx & Int32(31)
    first_row = thread_idx >> Int32(5)
    column = lane * Int32(16)
    sf_column = nvfp4_scale_byte_offset(lane, rope_dim)
    for chunk in cutlass.range_constexpr(8):
        row0 = first_row + Int32(chunk * 16)
        row1 = row0 + Int32(8)
        words0 = cutlass.Pointer(
            raw + row0 * Int32(row_bytes) + column // Int32(2), dtype=Int32
        ).load(count=2, alignment=8)
        words1 = cutlass.Pointer(
            raw + row1 * Int32(row_bytes) + column // Int32(2), dtype=Int32
        ).load(count=2, alignment=8)
        packed[chunk * 4], packed[chunk * 4 + 1] = words0[0], words0[1]
        packed[chunk * 4 + 2], packed[chunk * 4 + 3] = words1[0], words1[1]
        sf0 = cutlass.Array(raw, dtype=cutlass.Uint8)[
            row0 * Int32(row_bytes) + Int32(NVFP4_DATA_BYTES + rope_dim) + sf_column
        ]
        sf1 = cutlass.Array(raw, dtype=cutlass.Uint8)[
            row1 * Int32(row_bytes) + Int32(NVFP4_DATA_BYTES + rope_dim) + sf_column
        ]
        scales[chunk] = Int32(sf0) | (Int32(sf1) << Int32(8))
        if cutlass.const_expr(rope_dim):
            rope0 = cutlass.Pointer(
                raw
                + row0 * Int32(row_bytes)
                + Int32(NVFP4_DATA_BYTES)
                + lane * Int32(2),
                dtype=cutlass.Uint16,
            ).load()
            rope1 = cutlass.Pointer(
                raw
                + row1 * Int32(row_bytes)
                + Int32(NVFP4_DATA_BYTES)
                + lane * Int32(2),
                dtype=cutlass.Uint16,
            ).load()
            rope[chunk] = Int32(rope0) | (Int32(rope1) << Int32(16))
    if cutlass.const_expr(raw_empty is not None):
        # Raw storage is disjoint from every expanded destination. Each
        # warp snapshots all of its bytes before its release-arrival. The
        # raw-empty barrier has eight arrivals; the metadata TMA producer
        # acquires it and proxy-fences before any overwrite.
        cute.arch.sync_warp()
        if prims.elect_sync():
            prims.mbarrier_arrive(raw_empty, scope=prims.MemScope.CTA)
    else:
        # In-place expansion must join all readers before any output store.
        prims.barrier_cta_sync(barrier_id=12, thread_count=256)
    panel = lane >> Int32(3)
    local_column = (lane & Int32(7)) * Int32(16)
    for chunk in cutlass.range_constexpr(8):
        for pair in cutlass.range_constexpr(2):
            row = first_row + Int32(chunk * 16 + pair * 8)
            sf = (scales[chunk] >> Int32(pair * 8)) & Int32(255)
            a, b = convert_nvfp4x8(packed[chunk * 4 + pair * 2], sf)
            c, d = convert_nvfp4x8(packed[chunk * 4 + pair * 2 + 1], sf)
            store_shared_u32x4(
                dst + panel * Int32(stage_bytes),
                swizzle_byte_offset(row * Int32(128) + local_column, 128),
                a,
                b,
                c,
                d,
            )
            if cutlass.const_expr(rope_dim):
                packed_rope = cutlass.Uint16(
                    (rope[chunk] >> Int32(pair * 16)) & Int32(65535)
                )
                cutlass.Pointer(
                    dst
                    + Int32(4 * stage_bytes)
                    + swizzle_byte_offset(row * Int32(64) + lane * Int32(2), 64),
                    dtype=cutlass.Uint16,
                ).store(packed_rope)


@cute.jit
def copy_fp8_record(dst, raw, thread_idx, raw_empty):
    """Reorder unmodified FP8 bytes into four retained MMA panels."""
    lane = thread_idx % Int32(32)
    warp = thread_idx // Int32(32)
    for i in cutlass.range(16, unroll=2):
        row = warp + i * Int32(8)
        offset = (lane // Int32(8)) * Int32(16384) + swizzle_byte_offset(
            row * Int32(128) + (lane % Int32(8)) * Int32(16), 128
        )
        words = cutlass.Pointer(
            raw + row * Int32(512) + lane * Int32(16), dtype=Int32
        ).load(count=4, alignment=16)
        store_shared_u32x4(dst, offset, words[0], words[1], words[2], words[3])
    cute.arch.sync_warp()
    if prims.elect_sync():
        prims.mbarrier_arrive(raw_empty, scope=prims.MemScope.CTA)


@cute.jit
def prefetch_scaled_row(primary, extra, route, cfg):
    """Warm the complete selected record while metadata runs ahead of conversion."""
    row = route & Int32(0x7FFFFFFF)
    address = primary.iterator.toint()
    capacity = Int32(primary.shape[2])
    stride = Int32(primary.stride[2])
    width = Int32(cache_record_size(cfg.kv_cache_format, cfg.rope_dim))
    if route < Int32(0):
        address = extra.iterator.toint()
        capacity = Int32(extra.shape[2])
        stride = Int32(extra.stride[2])
        width = Int32(cache_record_size(cfg.extra_kv_cache_format, cfg.rope_dim))
    if row < capacity:
        address += Int64(row) * Int64(stride)
        for line in cutlass.range_constexpr(3):
            cute.arch.inline_ptx(
                "prefetch.global.L2 [{$r0}];",
                read_only_args=[address + Int64(line * 128)],
            )
        if cutlass.const_expr(cfg.rope_dim):
            cute.arch.inline_ptx(
                "prefetch.global.L2 [{$r0}];", read_only_args=[address + Int64(351)]
            )
        if cutlass.const_expr(cfg.has_tensor_fp8):
            if width == Int32(512):
                for line in cutlass.range_constexpr(3, 4):
                    cute.arch.inline_ptx(
                        "prefetch.global.L2 [{$r0}];",
                        read_only_args=[address + Int64(line * 128)],
                    )
