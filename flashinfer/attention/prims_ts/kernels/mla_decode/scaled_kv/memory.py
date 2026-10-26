# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Shared-memory addressing, vector stores and publication for converted K/V."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.experimental import primitives as prims
from ..helpers.gather import _address


@cute.jit
def publish_scaled_kv(barrier, count, *, cta_group: cutlass.Constexpr[int] = 1):
    """Publish ordinary shared stores before signaling the MMA pipeline."""
    cute.arch.fence_view_async_shared()
    # The elected publisher must observe every lane's converted stores.
    # Proxy fencing alone does not make a relaxed manual complete_tx a
    # release operation. Join the warp, then publish through a release fence.
    cute.arch.sync_warp()
    if prims.elect_sync():
        if cutlass.const_expr(cta_group == 2):
            prims.fence_sync_restrict(prims.MemOrder.RELEASE)
            # Use the cluster address mapping API, not encoded CTA-rank bits.
            leader = prims.mapa(barrier, Int32(0))
            prims.mbarrier_complete_tx(
                leader, Int32(count), scope=prims.MemScope.CLUSTER
            )
        else:
            cute.arch.inline_ptx("fence.acq_rel.cta;")
            prims.mbarrier_complete_tx(barrier, Int32(count))


@cute.jit
def swizzle_byte_offset(offset, width: cutlass.Constexpr[int]):
    mask = {32: 0x10, 64: 0x30, 128: 0x70}[width]
    return offset ^ ((offset >> Int32(3)) & Int32(mask))


@cute.jit
def store_shared_u32x4(dst, offset, a, b, c, d):
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[_address(dst, Int32) + offset, a, b, c, d],
    )
