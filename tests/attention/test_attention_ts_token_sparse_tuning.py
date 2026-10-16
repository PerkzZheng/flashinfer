# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared-BF16 token-gather geometry and rescaling qualification."""

import math
from dataclasses import replace

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")

from cutlass import BFloat16, Float16

from flashinfer.attention.prims_ts import _q_token_kv_block_sparse_policy as policy
from flashinfer.attention.prims_ts import decode
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
    CAUSAL,
    FmhaDecodeConfig,
    _validate_profile_support,
)
from flashinfer.decode import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
)


def _shared_config():
    return FmhaDecodeConfig(
        headdim=256,
        heads_q_per_kv=128,
        tile_size_q=128,
        groups_tokens_heads_q=True,
        use_keeps_mma_ab=True,
        head_dim_per_stage_kv=128,
        num_insts_kv=1,
        o_stages=1,
        q_stages=1,
        tmem_o_cols=256,
        q_dtype=BFloat16,
        k_dtype=BFloat16,
        v_dtype=BFloat16,
        out_dtype=BFloat16,
        use_q_token_kv_block_sparse_route=True,
        use_paged_kv=True,
        num_tokens_per_page=1,
        storage_tokens_per_page=64,
        max_num_pages_per_seq_kv=512,
        max_seq_len_q=1,
        use_flat_native_kv_tma=True,
        reuse_shared_kv=True,
        use_persistent_scheduler=True,
        load_num_warps=8,
        correction_warp_idx=4,
        mask_type=CAUSAL,
    )


@pytest.mark.parametrize("topk", (1, 63, 64, 65, 512, 2048, 2049))
@pytest.mark.parametrize("cache_resident", (False, True))
@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
def test_shared_bf16_tile_policy(topk, cache_resident, capability):
    l2 = 128 << 20
    selected = policy.select_shared_bf16_kv_tile(
        block_topk=topk,
        cache_bytes=l2 if cache_resident else l2 + 1,
        l2_cache_bytes=l2,
        compute_capability=capability,
    )
    expected = (
        64 if capability == (10, 0) and not cache_resident and topk <= 2048 else 128
    )
    assert selected == expected


def test_shared_bf16_small_tile_layout_and_original_fallback():
    assert (
        policy.select_shared_bf16_kv_tile(
            block_topk=512,
            cache_bytes=1 << 30,
            l2_cache_bytes=0,
            compute_capability=(10, 0),
        )
        == 128
    )
    original = _shared_config()
    tuned = decode._specialize_shared_bf16_token_config(
        original,
        block_topk=512,
        cache_bytes=1 << 30,
        l2_cache_bytes=128 << 20,
        compute_capability=(10, 0),
    )
    assert tuned is not original
    assert original.tile_size_kv == 128
    assert tuned.uses_token_sparse_kv64 and tuned.unroll_staged_qk
    assert tuned.defers_softmax_anchor_updates
    assert (tuned.tmem_s_cols, tuned.tmem_p_cols, tuned.kv_stages) == (64, 32, 8)
    assert (tuned.mma_warp_idx, tuned.scheduler_warp_idx) == (8, 9)
    assert (tuned.page_offsets_warp_idx, tuned.load_warp_idx) == (10, 12)
    assert tuned.load_num_warps == 8 and tuned.threads_per_cta == 640
    tuned.validate_paged_kv_staging_config()
    tuned.validate_q_token_kv_block_sparse_grouped_keeps_profile()
    _validate_profile_support(
        cfg=tuned,
        seq_len_q=1,
        num_heads_q=256,
        num_heads_kv=2,
        split_kv_mode="disabled",
    )
    assert tuned.compile_signature() != original.compile_signature()
    assert FmhaDecodeConfig(**dict(tuned.compile_signature())).uses_token_sparse_kv64
    assert not original.unroll_staged_qk
    assert original.defers_softmax_anchor_updates
    assert not replace(original, reuse_shared_kv=False).defers_softmax_anchor_updates


@pytest.mark.parametrize(
    "changes",
    (
        {"reuse_shared_kv": False},
        {"use_flat_native_kv_tma": False},
        {"out_dtype": Float16},
        {"use_variable_seqlens_q": True},
        {"use_split_kv": True},
        {"use_persistent_scheduler": False},
        {"heads_q_per_kv": 64},
        {"tile_size_q": 64},
        {"use_sliding_window_causal": True},
        {"use_attention_sinks": True},
    ),
)
def test_shared_bf16_small_tile_keeps_unqualified_profiles(changes):
    original = replace(_shared_config(), **changes)
    assert (
        decode._specialize_shared_bf16_token_config(
            original,
            block_topk=512,
            cache_bytes=1 << 30,
            l2_cache_bytes=128 << 20,
            compute_capability=(10, 0),
        )
        is original
    )


_REQUIRES_SM100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="KV64 tuning is qualified on SM100",
)


@_REQUIRES_SM100
@pytest.mark.parametrize("topk", (65, 257, 512, 2048))
@torch.no_grad()
def test_shared_bf16_small_tile_rescale_and_live_graph(topk, monkeypatch):
    """Deferred/rebased maxima, masked tiles, tails and dynamic graph inputs."""
    torch.manual_seed(2831)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    batch, queries, context, page, dim, kv_heads = 64, 4, 2305, 64, 256, 2
    heads, rows = 128 * kv_heads, batch * queries
    pages = (context + page - 1) // page
    cache = torch.randn(
        batch * pages, kv_heads, page, dim, device="cuda", dtype=torch.bfloat16
    )
    table = torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(
        batch, pages
    )
    # These tile maxima defer a 7.5-log2 rise, cross the 8-log2 threshold,
    # then force a much larger rebase. Head-dependent scales exercise partial
    # warp votes in correction as well as uniformly unchanged anchors.
    maxima = (torch.tensor((0.0, 7.5, 9.0, 80.0), device="cuda") * math.log(2.0)).to(
        cache.dtype
    )
    cache[table.long(), :, :, 0] = maxima[torch.arange(pages, device="cuda") % 4].view(
        1, pages, 1, 1
    )
    query = torch.zeros(batch, queries, 1, heads, dim, device="cuda", dtype=cache.dtype)
    query[..., 0] = 16 * torch.linspace(0.5, 1.5, heads, device="cuda")
    indices = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .expand(rows, -1)
        .contiguous()
    )
    requests = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(
        queries
    )
    positions = torch.full((rows,), context - 1, device="cuda", dtype=torch.int32)
    # Exercise both 32-byte sectors and the legal 16-byte aligned fallback.
    output_offset = 8 if topk in (65, 257) else 0
    output_storage = torch.empty(
        query.numel() + output_offset, device="cuda", dtype=query.dtype
    )
    output = output_storage[output_offset:].view_as(query)
    workspace = torch.empty(
        get_q_token_kv_block_sparse_workspace_size(
            query,
            cache,
            table,
            block_topk=topk,
            max_seq_len_kv=context,
            kv_block_size=1,
            split_kv=False,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    compiled = []
    original_compile = decode._get_compiled_decode

    def record_compile(spec):
        compiled.append(dict(spec.config_items))
        return original_compile(spec)

    monkeypatch.setattr(decode, "_get_compiled_decode", record_compile)
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        rows,
        1,
        heads,
        kv_heads,
        dim,
        1,
        page,
        topk,
        context,
        device=query.device,
        workspace_buffer=workspace,
        split_kv=False,
        q_data_type=query.dtype,
        kv_data_type=cache.dtype,
        o_data_type=cache.dtype,
    )

    def run(value_cache=cache):
        wrapper.run(
            query, (cache, value_cache), table, indices, requests, positions, out=output
        )

    def check(value_cache=cache):
        q = query.view(rows, kv_heads, 128, dim).float()
        actual = output.view(rows, kv_heads, 128, dim).float()
        for row in range(rows):
            selected = indices[row].long()
            selected = selected[(selected >= 0) & (selected <= positions[row])]
            if selected.numel() == 0:
                assert torch.equal(actual[row], torch.zeros_like(actual[row]))
                continue
            physical = table[requests[row].long(), selected // page].long()
            key = cache[physical, :, selected % page].permute(1, 0, 2).float()
            value = value_cache[physical, :, selected % page].permute(1, 0, 2).float()
            ref = ((q[row] @ key.transpose(-1, -2)) * dim**-0.5).softmax(-1) @ value
            rms = (actual[row] - ref).square().mean(-1).sqrt() / ref.square().mean(
                -1
            ).sqrt().clamp_min(1e-6)
            assert torch.isfinite(actual[row]).all() and rms.max() < 0.01
            cosine = torch.nn.functional.cosine_similarity(actual[row], ref, dim=-1)
            assert (1 - cosine).max() < 1e-5

    run()
    check()
    assert compiled[-1]["tile_size_kv"] == 64
    assert compiled[-1]["reuse_shared_kv"] and compiled[-1]["kv_stages"] == 8
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for mode in ("reverse", "contract", "empty", "large-rebase", "restore"):
        positions.fill_(context - 1)
        cache[table[:, 0].long(), :, :, 0] = 0
        indices.copy_(
            torch.arange(topk, device="cuda", dtype=torch.int32).expand(rows, -1)
        )
        if mode == "reverse":
            indices.copy_(indices.flip(-1))
            requests.copy_(requests.flip(0))
        elif mode == "contract":
            positions.copy_(
                torch.arange(rows, device="cuda", dtype=torch.int32) % min(topk, 129)
            )
        elif mode == "empty":
            positions.fill_(-1)
        elif mode == "large-rebase":
            # Q1 requires a complete valid index prefix, not -1 holes. Make
            # the first tile's contribution underflow through a large finite
            # score gap instead, and separately test empty routes above.
            cache[table[:, 0].long(), :, :, 0] = -1e4
        output.fill_(torch.nan)
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        check()
    separate_v = torch.randn_like(cache)
    run(separate_v)
    check(separate_v)
    assert compiled[-1]["tile_size_kv"] == 128
    assert not compiled[-1]["reuse_shared_kv"]
