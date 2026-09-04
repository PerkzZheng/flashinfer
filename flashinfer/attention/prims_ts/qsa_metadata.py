# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA-graph-safe compact QSA metadata construction.

The public attention kernel consumes native page-four CSR metadata.  Q1 rows
map their compact selected logical blocks directly to encoded physical
subpage locators.  Q2/Q4/Q5 rows additionally union the selected blocks of
every adjacent query and pack one low-byte membership bit per query into each
locator.

Grouped construction uses one bitmap CTA per query and one pack CTA per group.
The kernels execute in stream order; programmatic dependent launch is left for
a separately qualified integration because metadata correctness must not rely
on an unproven producer/consumer launch contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import triton
import triton.language as tl

_QSA_PAGE_SIZE = 4
_QSA_MEMBERSHIP_BITS = 8
_QSA_WORKSPACE_ALIGNMENT = 256
_QSA_WIDE_PACK_BLOCK_SIZE = 1024


def _qsa_pack_num_warps(pack_block_size: int) -> int:
    """Use more lanes when a static long-context bitmap widens the pack."""

    return 8 if pack_block_size >= _QSA_WIDE_PACK_BLOCK_SIZE else 4


@dataclass(frozen=True)
class _PrimsTSQSAWorkspaceViews:
    """Typed views bound to one caller-owned QSA attention workspace."""

    qsa_page_indptr: torch.Tensor
    qsa_page_indices: torch.Tensor
    seq_lens: torch.Tensor
    metadata_scratch_buffer: torch.Tensor
    attention_workspace_buffer: torch.Tensor


@dataclass(frozen=True)
class _QSATensorDescriptor:
    """Structural replacement-storage contract for a prepared QSA plan."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype


def _describe_qsa_tensor(tensor: torch.Tensor) -> _QSATensorDescriptor:
    return _QSATensorDescriptor(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _validate_qsa_plan_tensor(
    tensor: torch.Tensor,
    name: str,
    descriptor: _QSATensorDescriptor,
) -> None:
    """Check one replacement tensor without inspecting device values."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        tuple(tensor.shape) != descriptor.shape
        or tuple(tensor.stride()) != descriptor.stride
        or tensor.device != descriptor.device
        or tensor.dtype != descriptor.dtype
    ):
        raise ValueError(
            f"{name} must preserve the shape, strides, device, and dtype "
            "validated by the QSA plan"
        )


@dataclass(frozen=True)
class _PrimsTSQSAWorkspaceLayout:
    """Layout of QSA metadata outputs and disjoint kernel scratch.

    The CSR indptr, page indices, and compact sequence lengths remain live from
    metadata construction through the attention launch. Grouped-union bitmaps
    and attention scratch occupy disjoint regions: a decode split-KV plan must
    preserve its partials and self-resetting completion counters independently
    of metadata. A direct prefill plan has no split-KV storage; its small
    attention region contains only uniform-call-ABI placeholder tensors.
    Semantic inputs such as block tables and query mappings remain outside.
    """

    qsa_page_indptr_bytes: int
    qsa_page_indices_byte_offset: int
    qsa_page_indices_bytes: int
    seq_lens_byte_offset: int
    seq_lens_bytes: int
    metadata_scratch_byte_offset: int
    metadata_scratch_bytes: int
    attention_workspace_byte_offset: int
    attention_scratch_bytes: int
    uses_split_kv: bool
    max_seq_len: int
    total_bytes: int

    def bind(self, workspace_buffer: torch.Tensor) -> _PrimsTSQSAWorkspaceViews:
        """Return zero-copy typed views over a validated byte workspace."""

        _validate_qsa_attention_workspace(workspace_buffer, self.total_bytes)
        workspace_bytes = workspace_buffer.reshape(-1).view(torch.uint8)
        qsa_page_indptr = workspace_bytes[: self.qsa_page_indptr_bytes].view(
            torch.int32
        )
        qsa_page_indices = workspace_bytes[
            self.qsa_page_indices_byte_offset : self.qsa_page_indices_byte_offset
            + self.qsa_page_indices_bytes
        ].view(torch.int32)
        seq_lens = workspace_bytes[
            self.seq_lens_byte_offset : self.seq_lens_byte_offset + self.seq_lens_bytes
        ].view(torch.int32)
        metadata_scratch_buffer = workspace_bytes[
            self.metadata_scratch_byte_offset : self.metadata_scratch_byte_offset
            + self.metadata_scratch_bytes
        ]
        attention_workspace_buffer = workspace_bytes[
            self.attention_workspace_byte_offset : self.attention_workspace_byte_offset
            + self.attention_scratch_bytes
        ]
        return _PrimsTSQSAWorkspaceViews(
            qsa_page_indptr=qsa_page_indptr,
            qsa_page_indices=qsa_page_indices,
            seq_lens=seq_lens,
            metadata_scratch_buffer=metadata_scratch_buffer,
            attention_workspace_buffer=attention_workspace_buffer,
        )


@dataclass(frozen=True)
class _PrimsTSQSAMetadataPlan:
    """Unchecked metadata launch state with all geometry pre-resolved."""

    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    seq_lens: torch.Tensor
    bitsets: Optional[torch.Tensor]
    qo_indptr: Optional[torch.Tensor]
    rows: int
    groups: int
    num_requests: int
    group_size: int
    use_packed_q: bool
    block_topk: int
    page_capacity: int
    storage_page_size: int
    bitset_words: int
    builder_block_size: int
    pack_block_size: int
    pack_num_warps: int
    block_indices_row_stride: int
    block_indices_column_stride: int
    block_table_request_stride: int
    block_table_page_stride: int
    page_table_width: int

    def run(
        self,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        """Launch metadata kernels without validation or shape resolution."""

        if self.group_size == 1:
            _build_qsa_page4_q1_metadata_kernel[(self.rows,)](
                block_indices,
                block_table,
                token_to_request,
                query_positions,
                self.paged_kv_indptr,
                self.paged_kv_indices,
                self.seq_lens,
                self.block_indices_row_stride,
                self.block_indices_column_stride,
                self.block_table_request_stride,
                self.block_table_page_stride,
                self.rows,
                self.num_requests,
                BLOCK_TOPK=self.block_topk,
                BLOCK_SIZE=self.builder_block_size,
                SEMANTIC_PAGE_SIZE=_QSA_PAGE_SIZE,
                PAGE_TABLE_WIDTH=self.page_table_width,
                STORAGE_PAGE_SIZE=self.storage_page_size,
                PAGE_CAPACITY=self.page_capacity,
                num_warps=4,
            )
            return

        assert self.bitsets is not None
        qo_indptr = self.paged_kv_indptr if self.qo_indptr is None else self.qo_indptr
        _build_qsa_page4_grouped_bitsets_kernel[(self.groups, self.group_size)](
            block_indices,
            token_to_request,
            query_positions,
            self.bitsets,
            qo_indptr,
            self.block_indices_row_stride,
            self.block_indices_column_stride,
            self.rows,
            self.num_requests,
            PACKED_Q=self.use_packed_q,
            GROUP_SIZE=self.group_size,
            BLOCK_TOPK=self.block_topk,
            BITSET_WORDS=self.bitset_words,
            BLOCK_SIZE=self.builder_block_size,
            SEMANTIC_PAGE_SIZE=_QSA_PAGE_SIZE,
            num_warps=4,
        )
        _pack_qsa_page4_grouped_union_kernel[(self.groups,)](
            self.bitsets,
            block_table,
            token_to_request,
            query_positions,
            qo_indptr,
            self.paged_kv_indptr,
            self.paged_kv_indices,
            self.seq_lens,
            self.block_table_request_stride,
            self.block_table_page_stride,
            self.groups,
            self.num_requests,
            PACKED_Q=self.use_packed_q,
            GROUP_SIZE=self.group_size,
            BITSET_WORDS=self.bitset_words,
            BLOCK_SIZE=self.pack_block_size,
            PAGE_TABLE_WIDTH=self.page_table_width,
            SEMANTIC_PAGE_SIZE=_QSA_PAGE_SIZE,
            PAGE_MEMBERSHIP_BITS=_QSA_MEMBERSHIP_BITS,
            STORAGE_PAGE_SIZE=self.storage_page_size,
            PAGE_CAPACITY=self.page_capacity,
            num_warps=self.pack_num_warps,
        )


@dataclass(frozen=True)
class PrimsTSQSAPlan:
    """Prepared compact-metadata and PrimTS-attention launch state.

    The plan binds output/workspace storage and freezes input geometry once.
    Eager calls may pass new input storage with the same shapes, strides,
    devices, and dtypes; CUDA graph replay retains its usual stable-address
    requirement. The hot path performs synchronization-free replacement-storage
    checks followed by metadata and already-prepared attention launches. The
    original tensors are validated during preparation. Callers must keep all
    replacement tensors alive and disjoint, and must not mutate them
    concurrently with a launch or CUDA-graph replay that reads them. Call
    :meth:`run` once before capture so both metadata and attention kernels are
    compiled and their workspace state is initialized outside the graph.
    """

    _metadata_plan: _PrimsTSQSAMetadataPlan
    _bmm1_scale: float
    _bmm2_scale: float
    _attention_plan: Any
    _query: _QSATensorDescriptor
    _block_indices: _QSATensorDescriptor
    _block_table: _QSATensorDescriptor
    _token_to_request: _QSATensorDescriptor
    _query_positions: _QSATensorDescriptor
    _out: _QSATensorDescriptor
    _fixed_query_group_size: Optional[int] = None

    @property
    def qsa_page_indptr(self) -> torch.Tensor:
        """Graph-stable CSR indptr view owned by the byte workspace."""

        return self._metadata_plan.paged_kv_indptr

    @property
    def qsa_page_indices(self) -> torch.Tensor:
        """Graph-stable packed page-four locator view owned by the workspace."""

        return self._metadata_plan.paged_kv_indices

    @property
    def seq_lens(self) -> torch.Tensor:
        """Graph-stable compact-KV lengths owned by the byte workspace."""

        return self._metadata_plan.seq_lens

    def run(
        self,
        query: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Launch prepared QSA metadata and attention on the current stream."""

        _validate_qsa_plan_tensor(query, "query", self._query)
        _validate_qsa_plan_tensor(block_indices, "block_indices", self._block_indices)
        _validate_qsa_plan_tensor(block_table, "block_table", self._block_table)
        _validate_qsa_plan_tensor(
            token_to_request,
            "token_to_request",
            self._token_to_request,
        )
        _validate_qsa_plan_tensor(
            query_positions,
            "query_positions",
            self._query_positions,
        )
        _validate_qsa_plan_tensor(out, "out", self._out)

        from .decode import _validate_16byte_alignment

        _validate_16byte_alignment(query, "query")
        _validate_16byte_alignment(out, "out")

        self._metadata_plan.run(
            block_indices,
            block_table,
            token_to_request,
            query_positions,
        )
        attention_query = _flatten_fixed_qsa_groups(query, self._fixed_query_group_size)
        attention_out = _flatten_fixed_qsa_groups(out, self._fixed_query_group_size)
        self._attention_plan._run_unchecked(
            attention_query,
            attention_out,
            self._bmm1_scale,
            self._bmm2_scale,
        )
        return out


def _flatten_fixed_qsa_groups(
    tensor: torch.Tensor,
    group_size: Optional[int],
) -> torch.Tensor:
    """Return the lower-level decode view for a canonical fixed QSA tensor."""

    if group_size is None:
        return tensor
    flattened = tensor.flatten(0, 1)
    return flattened.squeeze(1) if group_size == 1 else flattened


@triton.jit
def _qsa_popcount_u32(value):
    return tl.inline_asm_elementwise(
        asm="popc.b32 $0, $1;",
        constraints="=r,r",
        args=[value.to(tl.uint32)],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _build_qsa_page4_q1_metadata_kernel(
    block_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    paged_kv_indptr_ptr,
    paged_kv_indices_ptr,
    seq_lens_ptr,
    stride_indices_row,
    stride_indices_column,
    stride_table_request,
    stride_table_page,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SEMANTIC_PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    STORAGE_PAGE_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
) -> None:
    """Map one compact QSA row directly to page-four CSR metadata."""

    row = tl.program_id(0)
    page_ranks = tl.arange(0, BLOCK_SIZE)
    request = tl.load(token_to_request_ptr + row)
    query_position = tl.load(query_positions_ptr + row)
    active = (
        (row < rows) & (query_position >= 0) & (request >= 0) & (request < num_requests)
    )
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    visible_tokens = tl.maximum(query_position + 1, 0)
    complete_pages = tl.minimum(visible_tokens // SEMANTIC_PAGE_SIZE, BLOCK_TOPK)
    tail_tokens = visible_tokens % SEMANTIC_PAGE_SIZE

    logical_block = tl.load(
        block_indices_ptr
        + row * stride_indices_row
        + page_ranks * stride_indices_column,
        mask=active & (page_ranks < complete_pages),
        other=-1,
    )
    logical_token = tl.maximum(logical_block, 0) * SEMANTIC_PAGE_SIZE
    logical_storage_page = logical_token // STORAGE_PAGE_SIZE
    table_live = (
        active
        & (page_ranks < complete_pages)
        & (logical_block >= 0)
        & (logical_storage_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_request
        + tl.minimum(logical_storage_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
        mask=table_live,
        other=-1,
    )
    subpages_per_storage_page: tl.constexpr = STORAGE_PAGE_SIZE // SEMANTIC_PAGE_SIZE
    subpage = (logical_token % STORAGE_PAGE_SIZE) // SEMANTIC_PAGE_SIZE
    locator = physical_page * subpages_per_storage_page + subpage
    # Q1 attention rounds page work to fixed tiles and may load a locator
    # before the corresponding token lanes are predicated. Initialize every
    # vector-covered slot so a shorter replay cannot expose stale locators.
    output_locator = tl.where(table_live & (physical_page >= 0), locator, -1)
    tl.store(
        paged_kv_indices_ptr + row * PAGE_CAPACITY + page_ranks,
        output_locator,
        mask=(row < rows) & (page_ranks < PAGE_CAPACITY),
    )

    # BLOCK_SIZE deliberately remains block_topk (512) rather than rounding
    # PAGE_CAPACITY (513) to 1,024 lanes. Clear the one scalar slot outside the
    # vector block before the optional causal tail overwrites it below.
    tl.store(
        paged_kv_indices_ptr + row * PAGE_CAPACITY + PAGE_CAPACITY - 1,
        -1,
        mask=row < rows,
    )

    # The causal tail is derived from the query position rather than emitted
    # by an expanded-index kernel.  Keeping it scalar avoids rounding the 513
    # page output to a 1024-lane vector program.
    tail_logical_block = visible_tokens // SEMANTIC_PAGE_SIZE
    tail_logical_token = tail_logical_block * SEMANTIC_PAGE_SIZE
    tail_storage_page = tail_logical_token // STORAGE_PAGE_SIZE
    tail_live = (
        active
        & (tail_tokens > 0)
        & (tail_storage_page < PAGE_TABLE_WIDTH)
        & (complete_pages < PAGE_CAPACITY)
    )
    tail_physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_request
        + tl.minimum(tail_storage_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
        mask=tail_live,
        other=-1,
    )
    tail_subpage = (tail_logical_token % STORAGE_PAGE_SIZE) // SEMANTIC_PAGE_SIZE
    tail_locator = tail_physical_page * subpages_per_storage_page + tail_subpage
    tl.store(
        paged_kv_indices_ptr + row * PAGE_CAPACITY + complete_pages,
        tail_locator,
        mask=tail_live & (tail_physical_page >= 0),
    )

    if row < rows:
        compact_length = complete_pages * SEMANTIC_PAGE_SIZE + tail_tokens
        tl.store(paged_kv_indptr_ptr + row, row * PAGE_CAPACITY)
        tl.store(seq_lens_ptr + row, tl.maximum(compact_length, 1))
        if row == rows - 1:
            tl.store(paged_kv_indptr_ptr + rows, rows * PAGE_CAPACITY)


@triton.jit
def _build_qsa_page4_grouped_bitsets_kernel(
    block_indices_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    bitsets_ptr,
    qo_indptr_ptr,
    stride_indices_row,
    stride_indices_column,
    rows,
    num_requests,
    PACKED_Q: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BITSET_WORDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SEMANTIC_PAGE_SIZE: tl.constexpr,
) -> None:
    """Build one compact logical-page bitmap per query."""

    group = tl.program_id(0)
    q_index = tl.program_id(1)
    if PACKED_Q:
        group_row_begin = tl.load(qo_indptr_ptr + group)
        group_row_end = tl.load(qo_indptr_ptr + group + 1)
    else:
        group_row_begin = group * GROUP_SIZE
        group_row_end = group_row_begin + GROUP_SIZE
    row = group_row_begin + q_index
    member_live = (row < group_row_end) & (q_index < GROUP_SIZE)
    bitset_base = group * GROUP_SIZE * BITSET_WORDS
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(
        bitsets_ptr + bitset_base + q_index * BITSET_WORDS + offsets,
        0,
        mask=offsets < BITSET_WORDS,
    )
    tl.debug_barrier()

    request = tl.load(
        token_to_request_ptr + row,
        mask=member_live & (row < rows),
        other=-1,
    )
    query_position = tl.load(
        query_positions_ptr + row,
        mask=member_live & (row < rows),
        other=-1,
    )
    active = (
        member_live
        & (row < rows)
        & (query_position >= 0)
        & (request >= 0)
        & (request < num_requests)
    )
    visible_tokens = tl.maximum(query_position + 1, 0)
    complete_pages = tl.minimum(visible_tokens // SEMANTIC_PAGE_SIZE, BLOCK_TOPK)
    selected_block = tl.load(
        block_indices_ptr + row * stride_indices_row + offsets * stride_indices_column,
        mask=active & (offsets < BLOCK_TOPK) & (offsets < complete_pages),
        other=-1,
    )
    word = selected_block // 32
    bit = selected_block % 32
    selected_live = (
        active
        & (offsets < BLOCK_TOPK)
        & (offsets < complete_pages)
        & (selected_block >= 0)
        & (word >= 0)
        & (word < BITSET_WORDS)
    )
    tl.atomic_or(
        bitsets_ptr + bitset_base + q_index * BITSET_WORDS + tl.maximum(word, 0),
        (1 << bit).to(tl.int32),
        mask=selected_live,
        sem="relaxed",
        scope="cta",
    )

    tail_tokens = visible_tokens % SEMANTIC_PAGE_SIZE
    tail_block = visible_tokens // SEMANTIC_PAGE_SIZE
    tail_word = tail_block // 32
    tail_bit = tail_block % 32
    tail_live = (
        active
        & (tail_tokens > 0)
        & (tail_block >= 0)
        & (tail_word >= 0)
        & (tail_word < BITSET_WORDS)
    )
    tl.atomic_or(
        bitsets_ptr + bitset_base + q_index * BITSET_WORDS + tl.maximum(tail_word, 0),
        (1 << tail_bit).to(tl.int32),
        mask=tail_live,
        sem="relaxed",
        scope="cta",
    )


@triton.jit
def _pack_qsa_page4_grouped_union_kernel(
    bitsets_ptr,
    block_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    qo_indptr_ptr,
    paged_kv_indptr_ptr,
    paged_kv_indices_ptr,
    seq_lens_ptr,
    stride_table_request,
    stride_table_page,
    groups,
    num_requests,
    PACKED_Q: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITSET_WORDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    SEMANTIC_PAGE_SIZE: tl.constexpr,
    PAGE_MEMBERSHIP_BITS: tl.constexpr,
    STORAGE_PAGE_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
) -> None:
    """Pack sorted locator/membership unions from per-query bitmaps."""

    group = tl.program_id(0)
    if PACKED_Q:
        first_row = tl.load(qo_indptr_ptr + group)
        group_row_end = tl.load(qo_indptr_ptr + group + 1)
    else:
        first_row = group * GROUP_SIZE
        group_row_end = first_row + GROUP_SIZE
    group_query_len = group_row_end - first_row
    last_row = group_row_end - 1
    request = tl.load(token_to_request_ptr + first_row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    first_position = tl.load(query_positions_ptr + first_row)
    last_position = tl.load(query_positions_ptr + last_row)
    group_valid = (
        (group < groups)
        & (group_query_len > 0)
        & (group_query_len <= GROUP_SIZE)
        & (request >= 0)
        & (request < num_requests)
        & (first_position >= 0)
        & (last_position == first_position + group_query_len - 1)
    )
    for q_index in tl.static_range(1, GROUP_SIZE):
        member_live = q_index < group_query_len
        q_request = tl.load(
            token_to_request_ptr + first_row + q_index,
            mask=member_live,
            other=request,
        )
        q_position = tl.load(
            query_positions_ptr + first_row + q_index,
            mask=member_live,
            other=first_position + q_index,
        )
        group_valid &= (~member_live) | (
            (q_request == request) & (q_position == first_position + q_index)
        )

    offsets = tl.arange(0, BLOCK_SIZE)
    word_live = offsets < BITSET_WORDS
    bitset_base = group * GROUP_SIZE * BITSET_WORDS
    q_word_0 = tl.load(
        bitsets_ptr + bitset_base + offsets,
        mask=word_live,
        other=0,
    ).to(tl.uint32)
    q_word_1 = tl.load(
        bitsets_ptr + bitset_base + BITSET_WORDS + offsets,
        mask=word_live,
        other=0,
    ).to(tl.uint32)
    q_word_2 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
    q_word_3 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
    q_word_4 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
    if GROUP_SIZE == 4 or GROUP_SIZE == 5:
        q_word_2 = tl.load(
            bitsets_ptr + bitset_base + 2 * BITSET_WORDS + offsets,
            mask=word_live,
            other=0,
        ).to(tl.uint32)
        q_word_3 = tl.load(
            bitsets_ptr + bitset_base + 3 * BITSET_WORDS + offsets,
            mask=word_live,
            other=0,
        ).to(tl.uint32)
    if GROUP_SIZE == 5:
        q_word_4 = tl.load(
            bitsets_ptr + bitset_base + 4 * BITSET_WORDS + offsets,
            mask=word_live,
            other=0,
        ).to(tl.uint32)
    q_word_0 = tl.where(word_live, q_word_0, 0)
    q_word_1 = tl.where(word_live, q_word_1, 0)
    q_word_2 = tl.where(word_live, q_word_2, 0)
    q_word_3 = tl.where(word_live, q_word_3, 0)
    q_word_4 = tl.where(word_live, q_word_4, 0)
    union_word = q_word_0 | q_word_1 | q_word_2 | q_word_3 | q_word_4
    word_counts = _qsa_popcount_u32(union_word)
    word_offsets = tl.cumsum(word_counts, axis=0) - word_counts
    union_pages = tl.sum(word_counts, axis=0)

    subpages_per_storage_page: tl.constexpr = STORAGE_PAGE_SIZE // SEMANTIC_PAGE_SIZE
    for bit_index in tl.static_range(0, 32):
        selected = word_live & ((union_word & (1 << bit_index)) != 0)
        rank = word_offsets + _qsa_popcount_u32(union_word & ((1 << bit_index) - 1))
        logical_block = offsets * 32 + bit_index
        logical_token = logical_block * SEMANTIC_PAGE_SIZE
        logical_storage_page = logical_token // STORAGE_PAGE_SIZE
        table_live = (
            group_valid
            & selected
            & (rank < PAGE_CAPACITY)
            & (logical_storage_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_storage_page, PAGE_TABLE_WIDTH - 1)
            * stride_table_page,
            mask=table_live,
            other=-1,
        )
        subpage = (logical_token % STORAGE_PAGE_SIZE) // SEMANTIC_PAGE_SIZE
        locator = physical_page * subpages_per_storage_page + subpage
        membership = (
            ((q_word_0 >> bit_index) & 1).to(tl.int32)
            | (((q_word_1 >> bit_index) & 1).to(tl.int32) << 1)
            | (((q_word_2 >> bit_index) & 1).to(tl.int32) << 2)
            | (((q_word_3 >> bit_index) & 1).to(tl.int32) << 3)
            | (((q_word_4 >> bit_index) & 1).to(tl.int32) << 4)
        )
        packed = (locator << PAGE_MEMBERSHIP_BITS) | membership
        tl.store(
            paged_kv_indices_ptr + group * PAGE_CAPACITY + rank,
            packed,
            mask=table_live & (physical_page >= 0),
        )

    tail_tokens = (last_position + 1) % SEMANTIC_PAGE_SIZE
    tail_padding = tl.where(
        tail_tokens == 0,
        0,
        SEMANTIC_PAGE_SIZE - tail_tokens,
    )
    seq_len = union_pages * SEMANTIC_PAGE_SIZE - tail_padding
    tl.store(paged_kv_indptr_ptr + group, group * PAGE_CAPACITY)
    tl.store(seq_lens_ptr + group, tl.where(group_valid, seq_len, 1))
    tl.store(
        paged_kv_indices_ptr + group * PAGE_CAPACITY,
        -1,
        mask=~group_valid,
    )
    if group == tl.num_programs(0) - 1:
        tl.store(
            paged_kv_indptr_ptr + group + 1,
            (group + 1) * PAGE_CAPACITY,
        )


def get_prims_ts_qsa_metadata_output_shapes(
    num_query_tokens: int,
    block_topk: int,
    group_size: int,
    *,
    num_query_groups: Optional[int] = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(indptr, indices, seq_lens)`` shapes for compact QSA input."""

    _validate_shape_parameters(num_query_tokens, block_topk, group_size)
    groups = _resolve_num_query_groups(
        num_query_tokens,
        group_size,
        num_query_groups,
    )
    page_capacity = group_size * (block_topk + 1)
    return (groups + 1,), (groups * page_capacity,), (groups,)


def get_prims_ts_qsa_metadata_workspace_size(
    num_query_tokens: int,
    max_num_storage_pages: int,
    storage_page_size: int,
    group_size: int,
    *,
    num_query_groups: Optional[int] = None,
) -> int:
    """Return caller-workspace bytes for compact QSA metadata construction.

    Q1 performs no union and requires no scratch.  Q2/Q4/Q5 reserve one logical
    page bitmap per input query.  The storage is stable across CUDA graph
    capture and is completely reinitialized by every launch.
    """

    if not isinstance(max_num_storage_pages, int) or isinstance(
        max_num_storage_pages, bool
    ):
        raise TypeError("max_num_storage_pages must be an integer")
    if max_num_storage_pages <= 0:
        raise ValueError("max_num_storage_pages must be positive")
    _validate_storage_page_size(storage_page_size)
    _validate_shape_parameters(num_query_tokens, 1, group_size)
    if group_size == 1:
        return 0
    groups = _resolve_num_query_groups(
        num_query_tokens,
        group_size,
        num_query_groups,
    )
    logical_block_capacity = max_num_storage_pages * storage_page_size // _QSA_PAGE_SIZE
    bitset_words = (logical_block_capacity + 31) // 32
    # Packed routes reserve G bitmap slots even when their final query group
    # has fewer than G live rows. This keeps each route's bitmap base static
    # and lets the producer clear inactive membership lanes cheaply.
    return groups * group_size * bitset_words * 4


def _get_prims_ts_qsa_workspace_layout(
    num_query_tokens: int,
    block_topk: int,
    max_num_storage_pages: int,
    storage_page_size: int,
    group_size: int,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype = torch.float16,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str | int] = None,
    num_query_groups: Optional[int] = None,
    use_packed_q: bool = False,
) -> _PrimsTSQSAWorkspaceLayout:
    """Return the unified allocation layout for QSA metadata and attention.

    The workspace owns the CSR metadata outputs, transient page indices, and
    kernel scratch. The caller continues to provide block tables, request
    mappings, and query positions as explicit semantic inputs.

    Metadata and attention use disjoint regions. Direct prefill policies have
    no split-KV storage; decode policies that select split-KV own dedicated
    partial-output, statistics, and completion-counter storage.
    """

    _validate_shape_parameters(num_query_tokens, block_topk, group_size)
    if num_query_tokens == 0:
        raise ValueError("num_query_tokens must be positive for QSA attention")
    if not isinstance(max_num_storage_pages, int) or isinstance(
        max_num_storage_pages, bool
    ):
        raise TypeError("max_num_storage_pages must be an integer")
    if max_num_storage_pages <= 0:
        raise ValueError("max_num_storage_pages must be positive")
    _validate_storage_page_size(storage_page_size)

    from .decode import (
        _resolve_decode_workspace_layout,
        _validate_prims_ts_qsa_group_capacity,
    )

    group_size = _validate_prims_ts_qsa_group_capacity(
        group_size,
        num_qo_heads,
        num_kv_heads,
    )

    groups = _resolve_num_query_groups(
        num_query_tokens,
        group_size,
        num_query_groups,
    )
    page_capacity = group_size * (block_topk + 1)
    qsa_page_indptr_numel = groups + 1
    qsa_page_indptr_bytes = qsa_page_indptr_numel * 4
    qsa_page_indices_byte_offset = _align_up_qsa_workspace(qsa_page_indptr_bytes)
    qsa_page_indices_numel = groups * page_capacity
    qsa_page_indices_bytes = qsa_page_indices_numel * 4
    seq_lens_byte_offset = _align_up_qsa_workspace(
        qsa_page_indices_byte_offset + qsa_page_indices_bytes
    )
    seq_lens_numel = groups
    seq_lens_bytes = seq_lens_numel * 4
    metadata_scratch_bytes = get_prims_ts_qsa_metadata_workspace_size(
        num_query_tokens,
        max_num_storage_pages,
        storage_page_size,
        group_size,
        num_query_groups=groups,
    )
    max_seq_len = (
        block_topk * _QSA_PAGE_SIZE + (_QSA_PAGE_SIZE - 1)
        if group_size == 1
        else page_capacity * _QSA_PAGE_SIZE
    )
    if kv_dtype is None:
        kv_dtype = q_dtype
    if out_dtype is None:
        out_dtype = q_dtype
    attention_layout = _resolve_decode_workspace_layout(
        groups,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        _QSA_PAGE_SIZE,
        max_seq_len,
        group_size,
        q_dtype,
        kv_dtype,
        out_dtype,
        "HND",
        "causal",
        use_packed_q,
        -1,
        storage_page_size,
        device,
        use_qsa_route=True,
    )
    attention_scratch_bytes = attention_layout.total_bytes
    metadata_scratch_byte_offset = _align_up_qsa_workspace(
        seq_lens_byte_offset + seq_lens_bytes
    )
    attention_workspace_byte_offset = _align_up_qsa_workspace(
        metadata_scratch_byte_offset + metadata_scratch_bytes
    )
    return _PrimsTSQSAWorkspaceLayout(
        qsa_page_indptr_bytes=qsa_page_indptr_bytes,
        qsa_page_indices_byte_offset=qsa_page_indices_byte_offset,
        qsa_page_indices_bytes=qsa_page_indices_bytes,
        seq_lens_byte_offset=seq_lens_byte_offset,
        seq_lens_bytes=seq_lens_bytes,
        metadata_scratch_byte_offset=metadata_scratch_byte_offset,
        metadata_scratch_bytes=metadata_scratch_bytes,
        attention_workspace_byte_offset=attention_workspace_byte_offset,
        attention_scratch_bytes=attention_scratch_bytes,
        uses_split_kv=attention_layout.uses_split_kv,
        max_seq_len=max_seq_len,
        total_bytes=attention_workspace_byte_offset + attention_scratch_bytes,
    )


def get_prims_ts_qsa_workspace_size(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_topk: int,
    out_dtype: Optional[torch.dtype] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
) -> int:
    """Return bytes for QSA metadata outputs and kernel scratch.

    Packed queries use ``[num_query_tokens, num_qo_heads, head_dim]`` and
    Int32 ``qo_indptr`` partitions their rows into request-safe routes of
    length at most ``max_seq_len_q``. Fixed queries use
    ``[batch, num_query_groups, group_size, num_qo_heads, head_dim]`` and do
    not require query-offset metadata. The first two fixed axes are flattened
    into the attention route axis without copying tensor storage.

    CSR indptr, packed page indices, and compact sequence lengths are owned by
    the workspace. Metadata and attention scratch are disjoint. The attention
    region contains split-KV partials, statistics, and counters only when the
    resolved policy uses split-KV; direct prefill retains only small
    uniform-ABI placeholders.
    """

    return _get_prims_ts_qsa_workspace_layout_from_tensors(
        query,
        k_cache,
        block_table,
        block_topk,
        out_dtype=out_dtype,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    ).total_bytes


def _get_prims_ts_qsa_workspace_layout_from_tensors(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_topk: int,
    *,
    out_dtype: Optional[torch.dtype],
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
) -> _PrimsTSQSAWorkspaceLayout:
    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise ValueError("query must be a CUDA tensor")
    use_packed_q = qo_indptr is not None
    if not use_packed_q and max_seq_len_q is not None:
        raise ValueError("max_seq_len_q is only valid with packed QSA qo_indptr")
    if use_packed_q:
        if query.ndim != 3:
            raise ValueError("packed QSA query must have shape [total_q,Hq,D]")
        if max_seq_len_q is None:
            raise ValueError("max_seq_len_q is required with QSA qo_indptr")
        _validate_qsa_qo_indptr_layout_tensor(qo_indptr)
        group_size = int(max_seq_len_q)
        groups = int(qo_indptr.numel()) - 1
        num_query_tokens, num_qo_heads, head_dim = query.shape
    elif query.ndim == 5 and query.shape[2] in (1, 2, 4, 5):
        batch_size, groups_per_request, group_size, num_qo_heads, head_dim = query.shape
        if batch_size <= 0 or groups_per_request <= 0:
            raise ValueError("fixed QSA batch and query-group counts must be positive")
        groups = int(batch_size) * int(groups_per_request)
    else:
        raise ValueError(
            "query must be packed [total_q,Hq,D] with qo_indptr or fixed "
            "[B,Nq,1|2|4|5,Hq,D] without qo_indptr"
        )
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have shape [pages, Hkv, storage_page_size, D]")
    if k_cache.device != query.device or k_cache.shape[3] != head_dim:
        raise ValueError("query and k_cache must share device and head dimension")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be a rank-two int32 tensor")
    if block_table.device != query.device:
        raise ValueError("block_table must be on the query device")
    if out_dtype is None:
        out_dtype = query.dtype
    return _get_prims_ts_qsa_workspace_layout(
        int(query.shape[0]) if use_packed_q else groups * group_size,
        block_topk,
        block_table.shape[1],
        k_cache.shape[2],
        group_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=k_cache.shape[1],
        head_dim=head_dim,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=out_dtype,
        device=query.device,
        num_query_groups=groups,
        use_packed_q=use_packed_q,
    )


def _build_prims_ts_qsa_page4_metadata(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    workspace_buffer: Optional[torch.Tensor] = None,
    *,
    group_size: int,
    storage_page_size: int,
    qo_indptr: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build native page-four QSA CSR metadata from compact block IDs.

    ``block_indices`` contains compact logical four-token block IDs with shape
    ``[num_query_tokens, block_topk]``.  Adjacent Q2/Q4/Q5 rows must belong to
    the same request and have consecutive positions.  Q1 output locators are
    ordinary encoded subpage locators.  Grouped output words use
    ``(locator << 8) | membership``, where membership bit ``i`` marks visibility
    for query ``i`` in the group.

    Pass preallocated ``out`` and ``workspace_buffer`` tensors for CUDA graph
    capture.  Output row strides are fixed at
    ``group_size * (block_topk + 1)``; ``seq_lens`` selects the live prefix.
    The workspace may be int8, uint8, or int32 and must not be shared by
    concurrent launches. Q1 uses one kernel. Q2/Q4/Q5 use a stream-ordered
    per-query bitmap kernel followed by a union-pack kernel.
    Packed ``qo_indptr`` must be an Int32 device copy of CPU-validated route
    offsets; this builder checks only its structural tensor contract and does
    not read route values back to the host.
    """

    _validate_inputs(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size,
        storage_page_size,
    )
    rows, block_topk = block_indices.shape
    use_packed_q = qo_indptr is not None
    if use_packed_q:
        _validate_qsa_qo_indptr_tensor(
            qo_indptr,
            expected_device=block_indices.device,
        )
    groups = int(qo_indptr.numel()) - 1 if use_packed_q else None
    expected_shapes = get_prims_ts_qsa_metadata_output_shapes(
        rows,
        block_topk,
        group_size,
        num_query_groups=groups,
    )
    if out is None:
        outputs = tuple(
            torch.empty(shape, dtype=torch.int32, device=block_indices.device)
            for shape in expected_shapes
        )
    else:
        if not isinstance(out, tuple) or len(out) != 3:
            raise TypeError("out must be an (indptr, indices, seq_lens) tuple")
        outputs = out
        for tensor, shape in zip(outputs, expected_shapes, strict=True):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.int32
                or tensor.device != block_indices.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "QSA metadata outputs must be contiguous int32 tensors with "
                    f"shapes {expected_shapes} on {block_indices.device}"
                )
    paged_kv_indptr, paged_kv_indices, seq_lens = outputs
    groups = _resolve_num_query_groups(rows, group_size, groups)
    if groups == 0:
        paged_kv_indptr.zero_()
        return outputs

    page_capacity = group_size * (block_topk + 1)
    if group_size == 1:
        _build_qsa_page4_q1_metadata_kernel[(rows,)](
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens,
            block_indices.stride(0),
            block_indices.stride(1),
            block_table.stride(0),
            block_table.stride(1),
            rows,
            block_table.shape[0],
            BLOCK_TOPK=block_topk,
            BLOCK_SIZE=triton.next_power_of_2(block_topk),
            SEMANTIC_PAGE_SIZE=_QSA_PAGE_SIZE,
            PAGE_TABLE_WIDTH=block_table.shape[1],
            STORAGE_PAGE_SIZE=storage_page_size,
            PAGE_CAPACITY=page_capacity,
            num_warps=4,
        )
        return outputs

    workspace_bytes = get_prims_ts_qsa_metadata_workspace_size(
        rows,
        block_table.shape[1],
        storage_page_size,
        group_size,
        num_query_groups=groups,
    )
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            workspace_bytes,
            dtype=torch.uint8,
            device=block_indices.device,
        )
    _validate_workspace(workspace_buffer, block_indices.device, workspace_bytes)
    if workspace_buffer.dtype == torch.int32:
        bitsets = workspace_buffer.flatten()
    else:
        bitsets = workspace_buffer.flatten()[:workspace_bytes].view(torch.int32)

    logical_block_capacity = block_table.shape[1] * storage_page_size // _QSA_PAGE_SIZE
    bitset_words = (logical_block_capacity + 31) // 32
    builder_block_size = triton.next_power_of_2(max(block_topk, bitset_words))
    q_offsets = paged_kv_indptr if qo_indptr is None else qo_indptr
    _build_qsa_page4_grouped_bitsets_kernel[(groups, group_size)](
        block_indices,
        token_to_request,
        query_positions,
        bitsets,
        q_offsets,
        block_indices.stride(0),
        block_indices.stride(1),
        rows,
        block_table.shape[0],
        PACKED_Q=use_packed_q,
        GROUP_SIZE=group_size,
        BLOCK_TOPK=block_topk,
        BITSET_WORDS=bitset_words,
        BLOCK_SIZE=builder_block_size,
        SEMANTIC_PAGE_SIZE=_QSA_PAGE_SIZE,
        num_warps=4,
    )
    pack_block_size = triton.next_power_of_2(bitset_words)
    pack_num_warps = _qsa_pack_num_warps(pack_block_size)
    _pack_qsa_page4_grouped_union_kernel[(groups,)](
        bitsets,
        block_table,
        token_to_request,
        query_positions,
        q_offsets,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        block_table.stride(0),
        block_table.stride(1),
        groups,
        block_table.shape[0],
        PACKED_Q=use_packed_q,
        GROUP_SIZE=group_size,
        BITSET_WORDS=bitset_words,
        BLOCK_SIZE=pack_block_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        SEMANTIC_PAGE_SIZE=_QSA_PAGE_SIZE,
        PAGE_MEMBERSHIP_BITS=_QSA_MEMBERSHIP_BITS,
        STORAGE_PAGE_SIZE=storage_page_size,
        PAGE_CAPACITY=page_capacity,
        num_warps=pack_num_warps,
    )
    return outputs


def build_prims_ts_qsa_page4_metadata(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    workspace_buffer: Optional[torch.Tensor] = None,
    *,
    group_size: int,
    storage_page_size: int,
    qo_indptr: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build native page-four QSA CSR metadata from compact block IDs.

    Q1 uses one kernel. Q2/Q4/Q5 use a stream-ordered per-query bitmap kernel
    followed by a union-pack kernel. Advanced callers that capture this raw
    path must preallocate ``out`` and ``workspace_buffer``, warm the same
    metadata geometry once before capture, and retain every tensor at a stable
    address through replay.
    Packed ``qo_indptr`` must be an Int32 device copy of CPU-validated route
    offsets and is checked structurally without a host-side value read.
    """

    return _build_prims_ts_qsa_page4_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        workspace_buffer,
        group_size=group_size,
        storage_page_size=storage_page_size,
        qo_indptr=qo_indptr,
        out=out,
    )


def _prepare_prims_ts_qsa_metadata_plan(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    metadata_workspace: torch.Tensor,
    qsa_page_indptr: torch.Tensor,
    qsa_page_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    group_size: int,
    storage_page_size: int,
    qo_indptr: Optional[torch.Tensor] = None,
) -> _PrimsTSQSAMetadataPlan:
    """Freeze validated metadata tensors and launch constants."""

    rows, block_topk = block_indices.shape
    use_packed_q = qo_indptr is not None
    groups = _resolve_num_query_groups(
        rows,
        group_size,
        int(qo_indptr.numel()) - 1 if use_packed_q else None,
    )
    page_capacity = group_size * (block_topk + 1)
    logical_block_capacity = block_table.shape[1] * storage_page_size // _QSA_PAGE_SIZE
    bitset_words = (logical_block_capacity + 31) // 32 if group_size > 1 else 0
    builder_block_size = triton.next_power_of_2(max(block_topk, bitset_words))
    pack_block_size = triton.next_power_of_2(bitset_words) if group_size > 1 else 1
    pack_num_warps = _qsa_pack_num_warps(pack_block_size)
    bitsets = None
    if group_size > 1:
        metadata_bytes = get_prims_ts_qsa_metadata_workspace_size(
            rows,
            block_table.shape[1],
            storage_page_size,
            group_size,
            num_query_groups=groups,
        )
        bitsets = metadata_workspace[:metadata_bytes].view(torch.int32)
    return _PrimsTSQSAMetadataPlan(
        paged_kv_indptr=qsa_page_indptr,
        paged_kv_indices=qsa_page_indices,
        seq_lens=seq_lens,
        bitsets=bitsets,
        qo_indptr=qo_indptr,
        rows=rows,
        groups=groups,
        num_requests=block_table.shape[0],
        group_size=group_size,
        use_packed_q=use_packed_q,
        block_topk=block_topk,
        page_capacity=page_capacity,
        storage_page_size=storage_page_size,
        bitset_words=bitset_words,
        builder_block_size=builder_block_size,
        pack_block_size=pack_block_size,
        pack_num_warps=pack_num_warps,
        block_indices_row_stride=block_indices.stride(0),
        block_indices_column_stride=block_indices.stride(1),
        block_table_request_stride=block_table.stride(0),
        block_table_page_stride=block_table.stride(1),
        page_table_width=block_table.shape[1],
    )


def prepare_prims_ts_qsa_attention(
    query: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    out: torch.Tensor,
    bmm1_scale: Optional[float] = None,
    bmm2_scale: float = 1.0,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
) -> PrimsTSQSAPlan:
    """Prepare one graph-stable metadata-plus-attention QSA launch.

    Packed Q/output use ``[num_query_tokens, Hq, D]`` and describe Q1/Q2/Q4/Q5
    routes with ``qo_indptr`` and ``max_seq_len_q``. A request's final route
    may be shorter than the maximum group size. Fixed Q/output use
    ``[B, Nq, G, Hq, D]`` and omit ``qo_indptr``; ``B * Nq`` becomes the
    internal attention route count through a zero-copy view.

    Frameworks call :meth:`PrimsTSQSAPlan.run` with current inputs that preserve
    the prepared geometry. The plan owns all CSR metadata views inside
    ``workspace_buffer`` and resolves metadata geometry and the PrimTS attention
    launch only once. Packed ``qo_indptr`` must be an Int32 device copy of
    CPU-validated route offsets, such as those from
    :func:`make_prims_ts_qsa_qo_indptr`. Preparation checks only its structural
    tensor contract and does not materialize its values on the host.

    Call ``run`` once outside CUDA graph capture to compile and initialize the
    plan, then capture with all semantic inputs, output, and workspace storage
    kept at stable addresses.
    """

    if (
        not isinstance(paged_kv_cache, tuple)
        or len(paged_kv_cache) != 2
        or not all(isinstance(cache, torch.Tensor) for cache in paged_kv_cache)
    ):
        raise TypeError("paged_kv_cache must be a (k_cache, v_cache) tuple")
    k_cache, v_cache = paged_kv_cache
    if (
        v_cache.shape != k_cache.shape
        or v_cache.device != k_cache.device
        or v_cache.dtype != k_cache.dtype
    ):
        raise ValueError(
            "K and V cache tensors must have matching shapes, devices, and dtypes"
        )
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a caller-owned torch.Tensor")

    use_packed_q = qo_indptr is not None
    fixed_query_group_size = None
    if use_packed_q:
        if query.ndim != 3:
            raise ValueError("packed QSA query must have shape [total_q,Hq,D]")
        if max_seq_len_q is None:
            raise ValueError("max_seq_len_q is required with QSA qo_indptr")
        group_size = int(max_seq_len_q)
        num_query_tokens = int(query.shape[0])
        _validate_qsa_qo_indptr_tensor(
            qo_indptr,
            expected_device=query.device,
        )
        attention_query = query
        attention_out = out
    elif query.ndim == 5:
        if query.shape[2] not in (1, 2, 4, 5):
            raise ValueError("fixed QSA group size must be one, two, four, or five")
        if out.shape != query.shape:
            raise ValueError("fixed QSA output must have the same shape as query")
        if not query.is_contiguous() or not out.is_contiguous():
            raise ValueError("fixed QSA query and output must be contiguous")
        group_size = int(query.shape[2])
        num_query_tokens = int(query.shape[0] * query.shape[1]) * group_size
        attention_query = _flatten_fixed_qsa_groups(query, group_size)
        attention_out = _flatten_fixed_qsa_groups(out, group_size)
        fixed_query_group_size = group_size
    else:
        raise ValueError(
            "query must be packed [total_q,Hq,D] with qo_indptr or fixed "
            "[B,Nq,1|2|4|5,Hq,D] without qo_indptr"
        )

    _validate_inputs(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size,
        int(k_cache.shape[2]),
    )
    if block_indices.shape[0] != num_query_tokens:
        raise ValueError("block_indices must have one row per flattened query token")

    layout = _get_prims_ts_qsa_workspace_layout_from_tensors(
        query,
        k_cache,
        block_table,
        int(block_indices.shape[1]),
        out_dtype=out.dtype,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    views = layout.bind(workspace_buffer)
    metadata_plan = _prepare_prims_ts_qsa_metadata_plan(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        views.metadata_scratch_buffer,
        views.qsa_page_indptr,
        views.qsa_page_indices,
        views.seq_lens,
        group_size=group_size,
        storage_page_size=int(k_cache.shape[2]),
        qo_indptr=qo_indptr,
    )
    from .decode import _prepare_prims_ts_batch_decode_plan, _validate_scale

    scale_qk = _validate_scale(
        query.shape[-1] ** -0.5 if bmm1_scale is None else bmm1_scale,
        "bmm1_scale",
    )
    scale_v = _validate_scale(bmm2_scale, "bmm2_scale")
    attention_plan, prepared_attention_out = _prepare_prims_ts_batch_decode_plan(
        attention_query,
        paged_kv_cache,
        views.attention_workspace_buffer,
        views.qsa_page_indptr,
        views.qsa_page_indices,
        views.seq_lens,
        layout.max_seq_len,
        out=attention_out,
        seq_len_q=group_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size if use_packed_q else None,
        out_dtype=out.dtype,
        mask_type="causal",
        window_left=-1,
        kv_layout="HND",
        page_size=_QSA_PAGE_SIZE,
        use_qsa_route=True,
    )
    if prepared_attention_out is not attention_out:
        raise RuntimeError("prepared PrimTS output storage changed unexpectedly")
    return PrimsTSQSAPlan(
        _metadata_plan=metadata_plan,
        _bmm1_scale=scale_qk,
        _bmm2_scale=scale_v,
        _attention_plan=attention_plan,
        _query=_describe_qsa_tensor(query),
        _block_indices=_describe_qsa_tensor(block_indices),
        _block_table=_describe_qsa_tensor(block_table),
        _token_to_request=_describe_qsa_tensor(token_to_request),
        _query_positions=_describe_qsa_tensor(query_positions),
        _out=_describe_qsa_tensor(out),
        _fixed_query_group_size=fixed_query_group_size,
    )


def prims_ts_qsa_attention(
    query: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    bmm1_scale: Optional[float] = None,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
) -> torch.Tensor:
    """Build compact page metadata and launch QSA from one byte workspace.

    Packed Q/O use ``[num_query_tokens, Hq, D]``; ``qo_indptr`` supplies
    request-safe route boundaries and ``max_seq_len_q`` supplies their maximum
    length. Fixed Q/O use ``[B, Nq, G, Hq, D]`` without ``qo_indptr``.

    All semantic input tensors remain explicit. The workspace internally owns
    CSR metadata outputs and disjoint metadata/attention scratch regions. This
    is an eager convenience that performs preparation on every call. Repeated
    launches and CUDA graphs should use :func:`prepare_prims_ts_qsa_attention`,
    run the prepared plan once before capture, and retain stable input, output,
    and workspace addresses through replay.

    Packed ``qo_indptr`` must be an Int32 device copy of CPU-validated route
    offsets, such as those from :func:`make_prims_ts_qsa_qo_indptr`. This path
    checks only the tensor's structural contract so it can remain
    synchronization-free; callers retain responsibility for valid route values
    and stable storage throughout a launch or CUDA-graph replay.
    """

    if (
        not isinstance(paged_kv_cache, tuple)
        or len(paged_kv_cache) != 2
        or not all(isinstance(cache, torch.Tensor) for cache in paged_kv_cache)
    ):
        raise TypeError("paged_kv_cache must be a (k_cache, v_cache) tuple")
    k_cache, v_cache = paged_kv_cache
    if (
        v_cache.shape != k_cache.shape
        or v_cache.device != k_cache.device
        or v_cache.dtype != k_cache.dtype
    ):
        raise ValueError(
            "K and V cache tensors must have matching shapes, devices, and dtypes"
        )
    if not isinstance(block_indices, torch.Tensor) or block_indices.ndim != 2:
        raise ValueError("block_indices must be a rank-two tensor")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out_dtype is None:
        out_dtype = out.dtype if out is not None else query.dtype
    layout = _get_prims_ts_qsa_workspace_layout_from_tensors(
        query,
        k_cache,
        block_table,
        block_indices.shape[1],
        out_dtype=out_dtype,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    use_packed_q = qo_indptr is not None
    fixed_query_group_size = None
    if use_packed_q:
        if max_seq_len_q is None:
            raise ValueError("max_seq_len_q is required with QSA qo_indptr")
        group_size = int(max_seq_len_q)
        num_query_tokens = query.shape[0]
        attention_query = query
        attention_out = out
    elif query.ndim == 5:
        if query.shape[2] not in (1, 2, 4, 5):
            raise ValueError("fixed QSA group size must be one, two, four, or five")
        if out is not None and out.shape != query.shape:
            raise ValueError("fixed QSA output must have the same shape as query")
        if not query.is_contiguous() or (out is not None and not out.is_contiguous()):
            raise ValueError("fixed QSA query and output must be contiguous")
        group_size = int(query.shape[2])
        num_query_tokens = int(query.shape[0] * query.shape[1]) * group_size
        attention_query = _flatten_fixed_qsa_groups(query, group_size)
        attention_out = (
            None if out is None else _flatten_fixed_qsa_groups(out, group_size)
        )
        fixed_query_group_size = group_size
    else:
        raise ValueError(
            "query must be packed [total_q,Hq,D] with qo_indptr or fixed "
            "[B,Nq,1|2|4|5,Hq,D] without qo_indptr"
        )
    if block_indices.shape[0] != num_query_tokens:
        raise ValueError("block_indices must have one row per flattened query token")
    views = layout.bind(workspace_buffer)

    # Keep metadata -> attention stream ordered until the complete dependency
    # chain is qualified independently.
    _build_prims_ts_qsa_page4_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        views.metadata_scratch_buffer,
        group_size=group_size,
        storage_page_size=k_cache.shape[2],
        qo_indptr=qo_indptr,
        out=(views.qsa_page_indptr, views.qsa_page_indices, views.seq_lens),
    )

    from .decode import _prepare_prims_ts_batch_decode_plan

    attention_plan, prepared_attention_out = _prepare_prims_ts_batch_decode_plan(
        attention_query,
        paged_kv_cache,
        views.attention_workspace_buffer,
        views.qsa_page_indptr,
        views.qsa_page_indices,
        views.seq_lens,
        layout.max_seq_len,
        seq_len_q=group_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size if use_packed_q else None,
        out=attention_out,
        out_dtype=out_dtype,
        mask_type="causal",
        window_left=-1,
        kv_layout="HND",
        page_size=_QSA_PAGE_SIZE,
        use_qsa_route=True,
    )
    result = attention_plan.run(
        attention_query,
        out=prepared_attention_out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    if fixed_query_group_size is not None:
        if out is not None:
            return out
        fixed_result = result.unflatten(0, (query.shape[0], query.shape[1]))
        return fixed_result.unsqueeze(2) if group_size == 1 else fixed_result
    return result


def _validate_shape_parameters(
    num_query_tokens: int,
    block_topk: int,
    group_size: int,
) -> None:
    for value, name in (
        (num_query_tokens, "num_query_tokens"),
        (block_topk, "block_topk"),
        (group_size, "group_size"),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if num_query_tokens < 0:
        raise ValueError("num_query_tokens must be nonnegative")
    if block_topk <= 0:
        raise ValueError("block_topk must be positive")
    if group_size not in (1, 2, 4, 5):
        raise ValueError("group_size must be one, two, four, or five")


def _resolve_num_query_groups(
    num_query_tokens: int,
    group_size: int,
    num_query_groups: Optional[int],
) -> int:
    """Resolve fixed or request-partitioned route count."""

    if num_query_groups is None:
        if num_query_tokens % group_size:
            raise ValueError(
                "num_query_tokens must be divisible by group_size without "
                "packed QSA qo_indptr"
            )
        return num_query_tokens // group_size
    if not isinstance(num_query_groups, int) or isinstance(num_query_groups, bool):
        raise TypeError("num_query_groups must be an integer")
    if num_query_groups <= 0:
        raise ValueError("num_query_groups must be positive")
    min_groups = (num_query_tokens + group_size - 1) // group_size
    if num_query_groups < min_groups or num_query_groups > num_query_tokens:
        raise ValueError(
            "num_query_groups cannot partition num_query_tokens into nonempty "
            f"routes of size at most {group_size}"
        )
    return num_query_groups


def _validate_qsa_qo_indptr_layout_tensor(qo_indptr: torch.Tensor) -> None:
    """Validate the packed-Q offsets required to size QSA workspace."""

    if (
        not isinstance(qo_indptr, torch.Tensor)
        or qo_indptr.ndim != 1
        or qo_indptr.numel() < 2
        or qo_indptr.dtype != torch.int32
    ):
        raise ValueError("QSA qo_indptr must be a rank-one int32 tensor")


def _validate_qsa_qo_indptr_tensor(
    qo_indptr: torch.Tensor,
    *,
    expected_device: torch.device,
) -> None:
    """Validate the synchronization-free packed-Q offset tensor contract."""

    _validate_qsa_qo_indptr_layout_tensor(qo_indptr)
    if qo_indptr.device != expected_device or not qo_indptr.is_contiguous():
        raise ValueError(
            "QSA qo_indptr must be a contiguous CUDA int32 tensor on the query device"
        )


def _validate_storage_page_size(storage_page_size: int) -> None:
    if not isinstance(storage_page_size, int) or isinstance(storage_page_size, bool):
        raise TypeError("storage_page_size must be an integer")
    if storage_page_size < _QSA_PAGE_SIZE or storage_page_size % _QSA_PAGE_SIZE:
        raise ValueError("storage_page_size must be a positive multiple of four")


def _align_up_qsa_workspace(value: int) -> int:
    return (
        (value + _QSA_WORKSPACE_ALIGNMENT - 1)
        // _QSA_WORKSPACE_ALIGNMENT
        * _QSA_WORKSPACE_ALIGNMENT
    )


def _validate_inputs(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    group_size: int,
    storage_page_size: int,
) -> None:
    if not block_indices.is_cuda:
        raise ValueError("QSA metadata inputs must be CUDA tensors")
    if block_indices.ndim != 2 or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be a rank-two int32 tensor")
    rows, block_topk = block_indices.shape
    _validate_shape_parameters(rows, block_topk, group_size)
    _validate_storage_page_size(storage_page_size)
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be a nonempty rank-two int32 tensor")
    if not all(block_table.shape):
        raise ValueError("block_table must be nonempty")
    if token_to_request.shape != (rows,) or token_to_request.dtype != torch.int32:
        raise ValueError("token_to_request must be int32 with one value per row")
    if query_positions.shape != (rows,) or query_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("query_positions must be int32/int64 with one value per row")
    tensors = (block_table, token_to_request, query_positions)
    if any(tensor.device != block_indices.device for tensor in tensors):
        raise ValueError("QSA metadata inputs must share one CUDA device")
    if block_indices.stride(1) != 1 or block_table.stride(1) != 1:
        raise ValueError("block_indices and block_table rows must be contiguous")
    if token_to_request.stride(0) != 1 or query_positions.stride(0) != 1:
        raise ValueError("per-row QSA metadata must be contiguous")


def _validate_workspace(
    workspace_buffer: torch.Tensor,
    device: torch.device,
    required_bytes: int,
) -> None:
    if (
        workspace_buffer.device != device
        or workspace_buffer.dtype not in (torch.int8, torch.uint8, torch.int32)
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.numel() * workspace_buffer.element_size() < required_bytes
    ):
        raise ValueError(
            "workspace_buffer must be a contiguous CUDA int8, uint8, or int32 "
            f"tensor with at least {required_bytes} bytes on {device}"
        )
    if workspace_buffer.data_ptr() % 4:
        raise ValueError("workspace_buffer must be at least four-byte aligned")


def _validate_qsa_attention_workspace(
    workspace_buffer: torch.Tensor,
    required_bytes: int,
) -> None:
    if not isinstance(workspace_buffer, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if workspace_buffer.dtype not in (torch.int8, torch.uint8):
        raise TypeError("workspace_buffer must have dtype torch.int8 or torch.uint8")
    if not workspace_buffer.is_cuda:
        raise ValueError("workspace_buffer must be a CUDA tensor")
    if not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    available_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if available_bytes < required_bytes:
        raise ValueError(
            "workspace_buffer is too small: requires at least "
            f"{required_bytes} bytes, got {available_bytes}"
        )
    if workspace_buffer.data_ptr() % 32:
        raise ValueError("workspace_buffer data pointer must be 32-byte aligned")


__all__ = [
    "PrimsTSQSAPlan",
    "build_prims_ts_qsa_page4_metadata",
    "get_prims_ts_qsa_metadata_output_shapes",
    "get_prims_ts_qsa_metadata_workspace_size",
    "get_prims_ts_qsa_workspace_size",
    "prepare_prims_ts_qsa_attention",
    "prims_ts_qsa_attention",
]
