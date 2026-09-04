# Copyright (c) 2026, FlashInfer Project.
#
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

"""PrimTS QSA examples for packed prefill and fixed-shape MTP decode.

These are the two layouts used by serving-framework integrations. Prefill keeps
queries packed as ``[total_q, Hq, D]`` and supplies request-safe route offsets.
Uniform decode uses ``[B, Nq, G, Hq, D]``, where ``G = MTP + 1``, and needs no
route-offset tensor. Both paths prepare metadata and attention together so the
hot-path call accepts only current semantic inputs and a framework-owned output;
workspace-owned intermediate metadata remains hidden.

Run on SM100 or SM103 after installing FlashInfer with PrimTS support. To keep
the example compact, requests share the same physical cache pages; a serving
framework normally provides distinct physical mappings.
"""

from __future__ import annotations

import torch

from flashinfer.decode import (
    get_prims_ts_qsa_workspace_size,
    make_prims_ts_qsa_qo_indptr,
    prepare_prims_ts_qsa_attention,
    validate_prims_ts_qsa_group_size,
)


_NUM_QO_HEADS = 12
_NUM_KV_HEADS = 1
_HEAD_DIM = 256
_BLOCK_TOPK = 512
_STORAGE_PAGE_SIZE = 16
_CONTEXT_LENGTH = 8192


def _make_cache_and_block_table(
    num_requests: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_storage_pages = (_CONTEXT_LENGTH + _STORAGE_PAGE_SIZE - 1) // (
        _STORAGE_PAGE_SIZE
    )
    k_cache = torch.randn(
        num_storage_pages,
        _NUM_KV_HEADS,
        _STORAGE_PAGE_SIZE,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(
        num_storage_pages,
        dtype=torch.int32,
        device=device,
    ).repeat(num_requests, 1)
    return k_cache, v_cache, block_table


def _make_block_indices(
    num_query_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    base_blocks = torch.arange(_BLOCK_TOPK, dtype=torch.int32, device=device)
    return torch.stack(
        [torch.roll(base_blocks, row % 7) for row in range(num_query_tokens)]
    ).contiguous()


def run_packed_prefill(device: torch.device) -> None:
    """Run variable-length prefill with packed queries and explicit routes."""

    request_q_lengths = (5, 3)
    num_requests = len(request_q_lengths)
    num_query_tokens = sum(request_q_lengths)
    query = torch.randn(
        num_query_tokens,
        _NUM_QO_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache, v_cache, block_table = _make_cache_and_block_table(num_requests, device)
    block_indices = _make_block_indices(num_query_tokens, device)
    token_to_request = torch.tensor(
        [0] * request_q_lengths[0] + [1] * request_q_lengths[1],
        dtype=torch.int32,
        device=device,
    )
    query_positions = torch.tensor(
        [
            *range(_CONTEXT_LENGTH - request_q_lengths[0], _CONTEXT_LENGTH),
            *range(_CONTEXT_LENGTH - request_q_lengths[1], _CONTEXT_LENGTH),
        ],
        dtype=torch.int64,
        device=device,
    )
    query_start_loc_cpu = torch.tensor(
        (0, request_q_lengths[0], num_query_tokens),
        dtype=torch.int32,
        device="cpu",
    )

    group_size = validate_prims_ts_qsa_group_size(
        query_start_loc_cpu,
        num_query_tokens,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        group_size=4,
    )
    qo_indptr = make_prims_ts_qsa_qo_indptr(
        query_start_loc_cpu,
        num_query_tokens,
        group_size=group_size,
        device=device,
    )
    output = torch.empty_like(query)
    workspace_bytes = get_prims_ts_qsa_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=_BLOCK_TOPK,
        out_dtype=output.dtype,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        workspace,
        out=output,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
    )
    plan.run(
        query,
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        out=output,
    )
    torch.cuda.synchronize()
    print(
        "packed prefill: "
        f"query={tuple(query.shape)}, routes={qo_indptr.numel() - 1}, "
        f"group_size<={group_size}, workspace_bytes={workspace_bytes}"
    )


def run_fixed_mtp_decode(device: torch.device) -> None:
    """Run uniform MTP decode with the fixed five-dimensional layout."""

    batch_size = 8
    num_query_groups = 1
    mtp_num_speculative_tokens = 3
    group_size = mtp_num_speculative_tokens + 1
    num_query_tokens = batch_size * num_query_groups * group_size

    # vLLM owns flat token storage and exposes this zero-copy fixed decode view.
    flat_query = torch.randn(
        num_query_tokens,
        _NUM_QO_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    query = flat_query.view(
        batch_size,
        num_query_groups,
        group_size,
        _NUM_QO_HEADS,
        _HEAD_DIM,
    )
    flat_output = torch.empty_like(flat_query)
    output = flat_output.view_as(query)
    k_cache, v_cache, block_table = _make_cache_and_block_table(batch_size, device)
    block_indices = _make_block_indices(num_query_tokens, device)
    token_to_request = torch.arange(
        batch_size,
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(num_query_groups * group_size)
    query_positions = torch.arange(
        _CONTEXT_LENGTH - group_size,
        _CONTEXT_LENGTH,
        dtype=torch.int64,
        device=device,
    ).repeat(batch_size * num_query_groups)

    workspace_bytes = get_prims_ts_qsa_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=_BLOCK_TOPK,
        out_dtype=output.dtype,
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        workspace,
        out=output,
    )

    # Compile and initialize outside capture, then replay only the hot path.
    plan.run(
        query,
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        out=output,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(
            query,
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            out=output,
        )
    graph.replay()
    torch.cuda.synchronize()

    print(
        "fixed MTP decode: "
        f"query={tuple(query.shape)}, group_size={group_size}, "
        f"workspace_bytes={workspace_bytes}, "
        f"metadata=({tuple(plan.qsa_page_indptr.shape)}, "
        f"{tuple(plan.qsa_page_indices.shape)}, {tuple(plan.seq_lens.shape)})"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((10, 0), (10, 3)):
        raise RuntimeError("PrimTS QSA currently requires SM100 or SM103")

    torch.manual_seed(42)
    device = torch.device("cuda")
    run_packed_prefill(device)
    run_fixed_mtp_decode(device)


if __name__ == "__main__":
    main()
