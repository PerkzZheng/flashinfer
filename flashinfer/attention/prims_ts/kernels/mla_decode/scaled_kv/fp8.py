# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Source selection for per-tensor E4M3 caches; payload bytes stay unchanged."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.experimental import primitives as prims


@cute.jit
def source_is_fp8(cfg, use_extra):
    if cutlass.const_expr(cfg.kv_cache_format == cfg.extra_kv_cache_format):
        return cutlass.Boolean(cfg.kv_cache_format == "fp8")
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
    if cutlass.const_expr(cfg.extra_kv_cache_format == "fp8"):
        return extra != Int32(0)
    return extra == Int32(0)
