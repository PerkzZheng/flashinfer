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

from __future__ import annotations

import pytest
import torch

from flashinfer.attention.prims_ts.qsa_metadata import (
    _get_prims_ts_qsa_workspace_layout,
    build_prims_ts_qsa_page4_metadata,
    get_prims_ts_qsa_metadata_output_shapes,
    get_prims_ts_qsa_metadata_workspace_size,
    get_prims_ts_qsa_workspace_size,
    prepare_prims_ts_qsa_attention,
    prims_ts_qsa_attention,
)
from flashinfer.decode import make_prims_ts_qsa_qo_indptr


def test_qsa_apis_are_available_from_flashinfer_decode() -> None:
    import flashinfer.decode as public_decode

    public_names = (
        "PrimsTSQSAPlan",
        "build_prims_ts_qsa_page4_metadata",
        "get_prims_ts_qsa_metadata_output_shapes",
        "get_prims_ts_qsa_metadata_workspace_size",
        "get_prims_ts_qsa_workspace_size",
        "make_prims_ts_qsa_qo_indptr",
        "prepare_prims_ts_qsa_attention",
        "prims_ts_qsa_attention",
        "validate_prims_ts_qsa_group_size",
    )

    for name in public_names:
        assert getattr(public_decode, name) is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unified_qsa_workspace_8k_q_128k_kv_separates_scratch() -> None:
    layout = _get_prims_ts_qsa_workspace_layout(
        8192,
        512,
        2048,
        64,
        4,
        num_qo_heads=24,
        num_kv_heads=2,
        head_dim=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device="cuda",
        num_query_groups=2048,
        use_packed_q=True,
    )

    assert layout.qsa_page_indices_bytes == 16_809_984
    assert layout.metadata_scratch_bytes == 32 * 1024 * 1024
    assert not layout.uses_split_kv
    assert layout.total_bytes == (
        layout.attention_workspace_byte_offset + layout.attention_scratch_bytes
    )

    workspace = torch.empty(layout.total_bytes, dtype=torch.uint8, device="cuda")
    views = layout.bind(workspace)
    assert views.qsa_page_indptr.numel() == 2049
    assert views.qsa_page_indptr.data_ptr() == workspace.data_ptr()
    assert views.qsa_page_indices.data_ptr() == (
        workspace.data_ptr() + layout.qsa_page_indices_byte_offset
    )
    assert views.qsa_page_indices.numel() == 8192 * 513
    assert views.seq_lens.numel() == 2048
    assert views.seq_lens.data_ptr() == (
        workspace.data_ptr() + layout.seq_lens_byte_offset
    )
    assert views.metadata_scratch_buffer.data_ptr() == (
        workspace.data_ptr() + layout.metadata_scratch_byte_offset
    )
    assert views.metadata_scratch_buffer.numel() == layout.metadata_scratch_bytes
    assert views.attention_workspace_buffer.data_ptr() == (
        workspace.data_ptr() + layout.attention_workspace_byte_offset
    )
    assert views.attention_workspace_buffer.numel() == layout.attention_scratch_bytes
    assert (
        views.metadata_scratch_buffer.data_ptr() + views.metadata_scratch_buffer.numel()
        <= views.attention_workspace_buffer.data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_qsa_workspace_size_accepts_cpu_route_offsets() -> None:
    query = torch.empty((8, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")
    cpu_qo_indptr = torch.tensor((0, 4, 5, 8), dtype=torch.int32)

    workspace_bytes = get_prims_ts_qsa_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=8,
        qo_indptr=cpu_qo_indptr,
        max_seq_len_q=4,
    )

    assert workspace_bytes > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_qsa_workspace_size_rejects_int64_route_offsets() -> None:
    query = torch.empty((8, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="int32"):
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
            qo_indptr=torch.tensor((0, 4, 5, 8), dtype=torch.int64),
            max_seq_len_q=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("query_shape", ((2, 12, 256), (2, 4, 12, 256)))
def test_qsa_workspace_size_rejects_legacy_fixed_layouts(
    query_shape: tuple[int, ...],
) -> None:
    query = torch.empty(query_shape, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="fixed.*B,Nq"):
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_qsa_workspace_size_rejects_packed_route_bound() -> None:
    query = torch.empty((2, 1, 4, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="only valid with packed"):
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
            max_seq_len_q=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_packed_q", (False, True))
def test_qsa_workspace_size_enforces_q64_group_capacity(
    use_packed_q: bool,
) -> None:
    group_size = 5
    num_qo_heads = 16
    if use_packed_q:
        query = torch.empty(
            (group_size, num_qo_heads, 256),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_kwargs = {
            "qo_indptr": torch.tensor((0, group_size), dtype=torch.int32),
            "max_seq_len_q": group_size,
        }
    else:
        query = torch.empty(
            (1, 1, group_size, num_qo_heads, 256),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_kwargs = {}
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((1, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="TileQ64/head capacity"):
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
            **q_kwargs,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_qsa_metadata_rejects_int64_route_offsets() -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(4, 8)

    with pytest.raises(ValueError, match="int32"):
        build_prims_ts_qsa_page4_metadata(
            blocks,
            table,
            requests,
            positions,
            torch.empty(1, dtype=torch.uint8, device="cuda"),
            group_size=4,
            storage_page_size=storage_page_size,
            qo_indptr=torch.tensor((0, 4, 8), dtype=torch.int64, device="cuda"),
        )


def _reference(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    storage_page_size: int,
    group_size: int,
    qo_indptr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = block_indices.device
    blocks = block_indices.cpu()
    table = block_table.cpu()
    requests = token_to_request.cpu()
    positions = query_positions.cpu()
    rows, block_topk = blocks.shape
    if qo_indptr is None:
        assert rows % group_size == 0
        route_offsets = list(range(0, rows + 1, group_size))
    else:
        route_offsets = [int(value) for value in qo_indptr.cpu().tolist()]
    groups = len(route_offsets) - 1
    page_capacity = group_size * (block_topk + 1)
    indptr = torch.arange(groups + 1, dtype=torch.int32) * page_capacity
    indices = torch.full((groups, page_capacity), -1, dtype=torch.int32)
    seq_lens = torch.ones(groups, dtype=torch.int32)
    subpages_per_storage_page = storage_page_size // 4

    for group in range(groups):
        first_row = route_offsets[group]
        group_end = route_offsets[group + 1]
        group_query_len = group_end - first_row
        request = int(requests[first_row].item())
        first_position = int(positions[first_row].item())
        valid = 0 <= request < table.shape[0]
        for q_index in range(group_query_len):
            row = first_row + q_index
            valid &= int(requests[row].item()) == request
            valid &= int(positions[row].item()) == first_position + q_index
        if not valid:
            continue

        if group_size == 1:
            visible_tokens = first_position + 1
            complete_pages = min(visible_tokens // 4, block_topk)
            tail_tokens = visible_tokens % 4
            live_blocks = [int(value) for value in blocks[first_row, :complete_pages]]
            if tail_tokens:
                live_blocks.append(visible_tokens // 4)
            seq_lens[group] = complete_pages * 4 + tail_tokens
            for rank, logical_block in enumerate(live_blocks):
                logical_token = logical_block * 4
                storage_page, token_offset = divmod(logical_token, storage_page_size)
                physical_page = int(table[request, storage_page].item())
                indices[group, rank] = (
                    physical_page * subpages_per_storage_page + token_offset // 4
                )
            continue

        membership_by_block: dict[int, int] = {}
        for q_index in range(group_query_len):
            row = first_row + q_index
            visible_tokens = int(positions[row].item()) + 1
            complete_pages = min(visible_tokens // 4, block_topk)
            for logical_block_tensor in blocks[row, :complete_pages]:
                logical_block = int(logical_block_tensor.item())
                membership_by_block[logical_block] = membership_by_block.get(
                    logical_block, 0
                ) | (1 << q_index)
            if visible_tokens % 4:
                tail_block = visible_tokens // 4
                membership_by_block[tail_block] = membership_by_block.get(
                    tail_block, 0
                ) | (1 << q_index)

        for rank, (logical_block, membership) in enumerate(
            sorted(membership_by_block.items())
        ):
            logical_token = logical_block * 4
            storage_page, token_offset = divmod(logical_token, storage_page_size)
            physical_page = int(table[request, storage_page].item())
            locator = physical_page * subpages_per_storage_page + token_offset // 4
            indices[group, rank] = (locator << 8) | membership
        last_tail = (int(positions[group_end - 1].item()) + 1) % 4
        tail_padding = 0 if last_tail == 0 else 4 - last_tail
        seq_lens[group] = len(membership_by_block) * 4 - tail_padding

    return indptr.to(device), indices.flatten().to(device), seq_lens.to(device)


def _make_case(group_size: int, block_topk: int, storage_page_size: int = 16):
    groups = 2
    rows = groups * group_size
    base_position = 4 * block_topk - 1
    positions = torch.tensor(
        [
            base_position + q_index
            for _group in range(groups)
            for q_index in range(group_size)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    requests = torch.arange(groups, dtype=torch.int32, device="cuda").repeat_interleave(
        group_size
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [torch.roll(base_blocks, row % 7) for row in range(rows)]
    ).contiguous()
    table_width = max(64, (4 * block_topk + 64) // storage_page_size)
    block_table = torch.arange(
        groups * table_width, dtype=torch.int32, device="cuda"
    ).reshape(groups, table_width)
    return blocks, block_table, requests, positions, storage_page_size


def _fixed_qsa_shape(
    num_routes: int,
    group_size: int,
    num_qo_heads: int = 12,
    head_dim: int = 256,
) -> tuple[int, int, int, int, int]:
    """Return the canonical fixed decode shape [B, Nq, G, Hq, D]."""

    return (num_routes, 1, group_size, num_qo_heads, head_dim)


def _as_lower_level_decode_view(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten canonical fixed QSA route axes without copying storage."""

    flattened = tensor.flatten(0, 1)
    return flattened.squeeze(1) if tensor.shape[2] == 1 else flattened


def _run_private_qsa_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    group_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Exercise the lower-level QSA route without exposing it publicly."""

    from flashinfer.attention.prims_ts.decode import (
        _prepare_prims_ts_batch_decode_plan,
        _resolve_decode_workspace_layout,
    )

    layout = _resolve_decode_workspace_layout(
        int(query.shape[0]),
        int(query.shape[-2]),
        int(k_cache.shape[1]),
        int(query.shape[-1]),
        4,
        max_seq_len,
        group_size,
        query.dtype,
        k_cache.dtype,
        out.dtype,
        "HND",
        "causal",
        False,
        -1,
        int(k_cache.shape[2]),
        query.device,
        use_qsa_route=True,
    )
    workspace = torch.zeros(layout.total_bytes, dtype=torch.uint8, device=query.device)
    plan, prepared_out = _prepare_prims_ts_batch_decode_plan(
        query,
        (k_cache, v_cache),
        workspace,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        max_seq_len,
        seq_len_q=group_size,
        qo_indptr=None,
        max_seq_len_q=None,
        out=out,
        out_dtype=out.dtype,
        mask_type="causal",
        window_left=-1,
        kv_layout="HND",
        page_size=4,
        use_qsa_route=True,
    )
    return plan.run(query, out=prepared_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
def test_qsa_attention_hides_workspace_page_indices(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    num_routes = table.shape[0]
    num_qo_heads = 12
    head_dim = 256
    query_shape = _fixed_qsa_shape(num_routes, group_size, num_qo_heads, head_dim)
    query = torch.zeros(query_shape, dtype=torch.bfloat16, device="cuda")
    num_physical_pages = int(table.max().item()) + 1
    k_cache = torch.zeros(
        num_physical_pages,
        1,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    workspace_bytes = get_prims_ts_qsa_workspace_size(
        query,
        k_cache,
        table,
        block_topk=blocks.shape[1],
        out_dtype=torch.bfloat16,
    )
    workspace_layout = _get_prims_ts_qsa_workspace_layout(
        blocks.shape[0],
        blocks.shape[1],
        table.shape[1],
        storage_page_size,
        group_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=1,
        head_dim=head_dim,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device="cuda",
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    views = workspace_layout.bind(workspace)
    output = torch.empty_like(query)
    calls: dict[str, torch.Tensor] = {}

    class FakeAttentionPlan:
        def run(
            self,
            actual_query: torch.Tensor,
            *,
            out: torch.Tensor,
            bmm1_scale: float | None,
            bmm2_scale: float,
        ) -> torch.Tensor:
            expected_query = _as_lower_level_decode_view(query)
            expected_output = _as_lower_level_decode_view(output)
            assert actual_query.shape == expected_query.shape
            assert actual_query.data_ptr() == expected_query.data_ptr()
            assert out.shape == expected_output.shape
            assert out.data_ptr() == expected_output.data_ptr()
            assert bmm1_scale is None
            assert bmm2_scale == 1.0
            return out

    def fake_prepare(
        actual_query: torch.Tensor,
        _paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        scratch_buffer: torch.Tensor,
        actual_indptr: torch.Tensor,
        qsa_page_indices: torch.Tensor,
        actual_seq_lens: torch.Tensor,
        _max_seq_len: int,
        **kwargs,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        calls["scratch"] = scratch_buffer
        calls["indices"] = qsa_page_indices
        expected_query = _as_lower_level_decode_view(query)
        expected_output = _as_lower_level_decode_view(output)
        assert actual_query.shape == expected_query.shape
        assert actual_query.data_ptr() == expected_query.data_ptr()
        assert actual_indptr.data_ptr() == views.qsa_page_indptr.data_ptr()
        assert actual_seq_lens.data_ptr() == views.seq_lens.data_ptr()
        assert kwargs["out"].shape == expected_output.shape
        assert kwargs["out"].data_ptr() == expected_output.data_ptr()
        assert scratch_buffer.data_ptr() == (
            workspace.data_ptr() + workspace_layout.attention_workspace_byte_offset
        )
        return FakeAttentionPlan(), expected_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    assert (
        prims_ts_qsa_attention(
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            workspace,
            out=output,
        )
        is output
    )

    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
    )
    torch.testing.assert_close(views.qsa_page_indptr, expected[0])
    torch.testing.assert_close(views.seq_lens, expected[2])
    qsa_page_indices = calls["indices"]
    for group in range(expected[2].numel()):
        begin = int(expected[0][group].item())
        live_pages = (int(expected[2][group].item()) + 3) // 4
        torch.testing.assert_close(
            qsa_page_indices[begin : begin + live_pages],
            expected[1][begin : begin + live_pages],
        )
    assert calls["scratch"].data_ptr() > qsa_page_indices.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 4))
def test_qsa_fixed_5d_layout_flattens_route_axes_without_copy(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    batch_size = 2
    groups_per_request = 1
    query = torch.zeros(
        batch_size,
        groups_per_request,
        group_size,
        12,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(query)
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    workspace = torch.empty(
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            table,
            block_topk=blocks.shape[1],
            out_dtype=output.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    calls: dict[str, torch.Tensor] = {}

    class FakeAttentionPlan:
        def run(
            self,
            actual_query: torch.Tensor,
            *,
            out: torch.Tensor,
            bmm1_scale: float | None,
            bmm2_scale: float,
        ) -> torch.Tensor:
            expected_shape = (
                (batch_size, 12, 256)
                if group_size == 1
                else (batch_size, group_size, 12, 256)
            )
            assert actual_query.shape == expected_shape
            assert out.shape == actual_query.shape
            assert actual_query.data_ptr() == query.data_ptr()
            assert out.data_ptr() == output.data_ptr()
            assert bmm1_scale is None
            assert bmm2_scale == 1.0
            out.fill_(3)
            calls["query"] = actual_query
            return out

    def fake_prepare(
        actual_query: torch.Tensor,
        _paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        _scratch_buffer: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        actual_output = kwargs["out"]
        assert isinstance(actual_output, torch.Tensor)
        expected_shape = (
            (batch_size, 12, 256)
            if group_size == 1
            else (batch_size, group_size, 12, 256)
        )
        assert actual_query.shape == expected_shape
        assert actual_output.shape == actual_query.shape
        assert actual_query.data_ptr() == query.data_ptr()
        assert actual_output.data_ptr() == output.data_ptr()
        return FakeAttentionPlan(), actual_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    result = prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
    )
    assert result is output
    assert "query" in calls
    assert torch.all(output == 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 4))
def test_prepared_qsa_fixed_5d_plan_flattens_runtime_views(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    query = torch.zeros(2, 1, group_size, 12, 256, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    workspace = torch.empty(
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            table,
            block_topk=blocks.shape[1],
            out_dtype=output.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    calls: dict[str, bool] = {}

    class FakeAttentionPlan:
        def _run_unchecked(
            self,
            actual_query: torch.Tensor,
            actual_output: torch.Tensor,
            _scale_qk: float,
            _scale_v: float,
        ) -> torch.Tensor:
            expected_shape = (
                (2, 12, 256) if group_size == 1 else (2, group_size, 12, 256)
            )
            assert actual_query.shape == expected_shape
            assert actual_output.shape == actual_query.shape
            assert actual_query.data_ptr() == query.data_ptr()
            assert actual_output.data_ptr() == output.data_ptr()
            actual_output.fill_(4)
            calls["ran"] = True
            return actual_output

    def fake_prepare(
        actual_query: torch.Tensor,
        _cache: tuple[torch.Tensor, torch.Tensor],
        _attention_workspace: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        actual_output = kwargs["out"]
        assert isinstance(actual_output, torch.Tensor)
        expected_shape = (2, 12, 256) if group_size == 1 else (2, group_size, 12, 256)
        assert actual_query.shape == expected_shape
        assert actual_output.shape == actual_query.shape
        return FakeAttentionPlan(), actual_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
    )
    assert (
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )
        is output
    )
    assert calls["ran"] is True
    assert torch.all(output == 4)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 4))
def test_qsa_fixed_decode_matches_packed_prefill_layout(group_size: int) -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    routes = table.shape[0]
    torch.manual_seed(42)
    fixed_query = torch.randn(
        _fixed_qsa_shape(routes, group_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    packed_query = fixed_query.reshape(blocks.shape[0], 12, 256)
    qo_indptr = torch.arange(
        0,
        blocks.shape[0] + 1,
        group_size,
        dtype=torch.int32,
        device="cuda",
    )
    k_cache = torch.randn(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    def run(
        query: torch.Tensor,
        *,
        packed_qo_indptr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        workspace = torch.empty(
            get_prims_ts_qsa_workspace_size(
                query,
                k_cache,
                table,
                block_topk=blocks.shape[1],
                out_dtype=output.dtype,
                qo_indptr=packed_qo_indptr,
                max_seq_len_q=group_size if packed_qo_indptr is not None else None,
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        return prims_ts_qsa_attention(
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            workspace,
            out=output,
            qo_indptr=packed_qo_indptr,
            max_seq_len_q=group_size if packed_qo_indptr is not None else None,
        )

    fixed_output = run(fixed_query)
    packed_output = run(packed_query, packed_qo_indptr=qo_indptr)
    torch.testing.assert_close(fixed_output.reshape_as(packed_output), packed_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
def test_prepared_qsa_plan_keeps_metadata_and_attention_scratch_disjoint(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    num_routes = table.shape[0]
    query_shape = _fixed_qsa_shape(num_routes, group_size)
    query = torch.zeros(query_shape, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    output = torch.empty_like(query)
    layout = _get_prims_ts_qsa_workspace_layout(
        blocks.shape[0],
        blocks.shape[1],
        table.shape[1],
        storage_page_size,
        group_size,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=output.dtype,
        device="cuda",
    )
    workspace = torch.empty(layout.total_bytes, dtype=torch.uint8, device="cuda")
    views = layout.bind(workspace)
    calls: dict[str, object] = {}

    class FakeAttentionPlan:
        def _run_unchecked(
            self,
            actual_query: torch.Tensor,
            actual_output: torch.Tensor,
            scale_qk: float,
            scale_v: float,
        ) -> torch.Tensor:
            expected_query = _as_lower_level_decode_view(query)
            assert actual_query.shape == expected_query.shape
            assert actual_query.stride() == expected_query.stride()
            assert actual_output.shape == expected_query.shape
            assert actual_output.stride() == expected_query.stride()
            assert scale_qk == pytest.approx(256**-0.5)
            assert scale_v == 1.0
            actual_output.zero_()
            calls["ran"] = True
            calls["query_ptr"] = actual_query.data_ptr()
            calls["output_ptr"] = actual_output.data_ptr()
            return actual_output

    def fake_prepare(
        actual_query: torch.Tensor,
        _cache: tuple[torch.Tensor, torch.Tensor],
        attention_workspace: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        expected_query = _as_lower_level_decode_view(query)
        assert actual_query.shape == expected_query.shape
        assert actual_query.data_ptr() == expected_query.data_ptr()
        assert attention_workspace.data_ptr() == (
            workspace.data_ptr() + layout.attention_workspace_byte_offset
        )
        actual_output = kwargs["out"]
        assert isinstance(actual_output, torch.Tensor)
        return FakeAttentionPlan(), actual_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
    )
    views.attention_workspace_buffer.fill_(0x5A)
    assert (
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )
        is output
    )
    torch.cuda.synchronize()
    assert calls["ran"] is True
    assert torch.all(views.attention_workspace_buffer == 0x5A)
    assert plan.qsa_page_indptr.data_ptr() == views.qsa_page_indptr.data_ptr()
    assert plan.qsa_page_indices.data_ptr() == views.qsa_page_indices.data_ptr()
    assert plan.seq_lens.data_ptr() == views.seq_lens.data_ptr()
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
    )
    torch.testing.assert_close(plan.qsa_page_indptr, expected[0])
    torch.testing.assert_close(plan.seq_lens, expected[2])

    replacement_query = query.clone()
    replacement_blocks = blocks.clone()
    replacement_table = table.clone()
    replacement_requests = requests.clone()
    replacement_positions = positions.clone()
    replacement_output = torch.empty_like(output)
    assert (
        plan.run(
            replacement_query,
            replacement_blocks,
            replacement_table,
            replacement_requests,
            replacement_positions,
            out=replacement_output,
        )
        is replacement_output
    )
    assert calls["query_ptr"] == replacement_query.data_ptr()
    assert calls["output_ptr"] == replacement_output.data_ptr()

    valid_args = (
        replacement_query,
        replacement_blocks,
        replacement_table,
        replacement_requests,
        replacement_positions,
    )
    invalid_inputs = (
        ("query", (replacement_query[:-1], *valid_args[1:])),
        ("block_indices", (valid_args[0], replacement_blocks[:-1], *valid_args[2:])),
        (
            "block_table",
            (
                valid_args[0],
                valid_args[1],
                replacement_table[:-1],
                *valid_args[3:],
            ),
        ),
        (
            "token_to_request",
            (
                valid_args[0],
                valid_args[1],
                valid_args[2],
                replacement_requests[:-1],
                valid_args[4],
            ),
        ),
        (
            "query_positions",
            (
                valid_args[0],
                valid_args[1],
                valid_args[2],
                valid_args[3],
                replacement_positions[:-1],
            ),
        ),
        ("out", valid_args),
    )
    for name, args in invalid_inputs:
        invalid_out = replacement_output[:-1] if name == "out" else replacement_output
        with pytest.raises(ValueError, match=name):
            plan.run(*args, out=invalid_out)

    wrong_stride_query = torch.empty(
        (
            *replacement_query.shape[:-2],
            replacement_query.shape[-1],
            replacement_query.shape[-2],
        ),
        dtype=replacement_query.dtype,
        device=replacement_query.device,
    ).transpose(-1, -2)
    with pytest.raises(ValueError, match="strides"):
        plan.run(wrong_stride_query, *valid_args[1:], out=replacement_output)

    misaligned_query = torch.empty(
        replacement_query.numel() + 1,
        dtype=replacement_query.dtype,
        device=replacement_query.device,
    )[1:].view(replacement_query.shape)
    assert misaligned_query.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        plan.run(misaligned_query, *valid_args[1:], out=replacement_output)

    misaligned_output = torch.empty(
        replacement_output.numel() + 1,
        dtype=replacement_output.dtype,
        device=replacement_output.device,
    )[1:].view(replacement_output.shape)
    assert misaligned_output.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        plan.run(*valid_args, out=misaligned_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_packed_qsa_does_not_materialize_route_offsets_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    group_size = 4
    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    query = torch.zeros((8, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(query)
    route_offsets = make_prims_ts_qsa_qo_indptr(
        torch.tensor((0, 5, 8), dtype=torch.int32),
        8,
        group_size=group_size,
    )
    qo_indptr = route_offsets.to("cuda")
    workspace = torch.empty(
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            table,
            block_topk=blocks.shape[1],
            qo_indptr=qo_indptr,
            max_seq_len_q=group_size,
        ),
        dtype=torch.uint8,
        device="cuda",
    )

    def fake_get_compiled_decode(
        *args: object,
    ) -> tuple[object, object, object, object]:
        spec = decode_module._resolve_decode_launch_spec(*args[:17])
        return (lambda *_args: None, None, (), spec.scratch_shapes)

    monkeypatch.setattr(
        decode_module,
        "_get_compiled_decode",
        fake_get_compiled_decode,
    )
    original_to = torch.Tensor.to
    original_cpu = torch.Tensor.cpu
    original_tolist = torch.Tensor.tolist

    def reject_qo_indptr_host_copy(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if tensor is qo_indptr and (args == ("cpu",) or kwargs.get("device") == "cpu"):
            raise AssertionError("prepared QSA must not copy qo_indptr to the host")
        return original_to(tensor, *args, **kwargs)

    def reject_qo_indptr_cpu(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is qo_indptr:
            raise AssertionError("prepared QSA must not copy qo_indptr to the host")
        return original_cpu(tensor)

    def reject_qo_indptr_tolist(tensor: torch.Tensor) -> list[object]:
        if tensor is qo_indptr:
            raise AssertionError("prepared QSA must not materialize qo_indptr values")
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "to", reject_qo_indptr_host_copy)
    monkeypatch.setattr(torch.Tensor, "cpu", reject_qo_indptr_cpu)
    monkeypatch.setattr(torch.Tensor, "tolist", reject_qo_indptr_tolist)
    prepare_prims_ts_qsa_attention(
        query,
        (k_cache, k_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
    )


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 4, 5))
def test_prepared_qsa_plan_replays_without_counter_reset(group_size: int) -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(
        group_size,
        512,
    )
    query_shape = _fixed_qsa_shape(table.shape[0], group_size)
    torch.manual_seed(4117 + group_size)
    query = torch.randn(query_shape, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(query)
    layout = _get_prims_ts_qsa_workspace_layout(
        blocks.shape[0],
        blocks.shape[1],
        table.shape[1],
        storage_page_size,
        group_size,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=output.dtype,
        device="cuda",
    )
    workspace = torch.full(
        (layout.total_bytes,),
        0x55,
        dtype=torch.uint8,
        device="cuda",
    )
    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
    )

    def run() -> None:
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )

    run()
    torch.cuda.synchronize()
    reference = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    if layout.uses_split_kv:
        assert not torch.count_nonzero(
            plan._attention_plan._workspace.split_kv_counter
        ).item()


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", (1, 2))
def test_prepared_qsa_fp8_q1_small_cuda_graph(rows: int) -> None:
    """Cover tiny fixed Q1 decode graph buckets."""

    block_topk = 512
    num_qo_heads = 12
    head_dim = 256
    storage_page_size = 1600
    blocks, table, requests, positions, _ = _make_case(1, block_topk, storage_page_size)
    blocks = blocks[:rows].contiguous()
    requests = requests[:rows].contiguous()
    positions = positions[:rows].contiguous()
    table = table[:rows].contiguous()

    torch.manual_seed(4189 + rows)
    query = (
        torch.randn(
            _fixed_qsa_shape(rows, 1, num_qo_heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    k_cache = (
        torch.randn(
            int(table.max().item()) + 1,
            1,
            storage_page_size,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    v_cache = (
        torch.randn_like(k_cache, dtype=torch.bfloat16)
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    output = torch.empty(query.shape, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            table,
            block_topk=block_topk,
            out_dtype=output.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
    )

    def run() -> None:
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )

    run()
    torch.cuda.synchronize()
    reference = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_qsa_fp8_q4_tp4_bs256_cuda_graph() -> None:
    """Match the TP4 vLLM MTP3 graph geometry with a prepared QSA plan."""

    batch_size = 256
    group_size = 4
    block_topk = 512
    num_qo_heads = 6
    head_dim = 256
    storage_page_size = 1600
    max_storage_pages = 88
    context_length = 8192
    rows = batch_size * group_size
    max_seq_len = group_size * (block_topk + 1) * 4

    torch.manual_seed(9425)
    query = (
        torch.randn(
            batch_size,
            1,
            group_size,
            num_qo_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    raw_query = _as_lower_level_decode_view(query)
    k_cache = (
        torch.randn(
            max_storage_pages,
            1,
            storage_page_size,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    v_cache = (
        torch.randn_like(k_cache, dtype=torch.bfloat16)
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    block_table = torch.arange(
        max_storage_pages, dtype=torch.int32, device="cuda"
    ).repeat(batch_size, 1)
    token_to_request = torch.arange(
        batch_size, dtype=torch.int32, device="cuda"
    ).repeat_interleave(group_size)
    query_positions = (
        torch.arange(
            context_length - group_size,
            context_length,
            dtype=torch.int64,
            device="cuda",
        )
        .repeat(batch_size)
        .contiguous()
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    block_indices = torch.stack(
        [torch.roll(base_blocks, row % 7) for row in range(rows)]
    ).contiguous()

    raw_metadata_shapes = get_prims_ts_qsa_metadata_output_shapes(
        rows, block_topk, group_size
    )
    raw_indptr, raw_indices, raw_seq_lens = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in raw_metadata_shapes
    )
    metadata_bytes = get_prims_ts_qsa_metadata_workspace_size(
        rows, max_storage_pages, storage_page_size, group_size
    )
    build_prims_ts_qsa_page4_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        torch.empty(metadata_bytes, dtype=torch.uint8, device="cuda"),
        group_size=group_size,
        storage_page_size=storage_page_size,
        out=(raw_indptr, raw_indices, raw_seq_lens),
    )
    raw_output = torch.empty(raw_query.shape, dtype=torch.bfloat16, device="cuda")
    _run_private_qsa_decode(
        raw_query,
        k_cache,
        v_cache,
        raw_indptr,
        raw_indices,
        raw_seq_lens,
        max_seq_len,
        group_size,
        raw_output,
    )

    combined_bytes = get_prims_ts_qsa_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=block_topk,
        out_dtype=torch.bfloat16,
    )
    combined_workspace = torch.empty(combined_bytes, dtype=torch.uint8, device="cuda")
    prepared_output = torch.empty(query.shape, dtype=torch.bfloat16, device="cuda")
    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        combined_workspace,
        out=prepared_output,
    )
    assert plan._metadata_plan.pack_block_size == 2048
    assert plan._metadata_plan.pack_num_warps == 8

    def run() -> None:
        plan.run(
            query,
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            out=prepared_output,
        )

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        _as_lower_level_decode_view(prepared_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        _as_lower_level_decode_view(prepared_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )

    # vLLM captures a fixed 1,024-token graph bucket even when only a short
    # live prefix belongs to requests. Exercise the same static graph with
    # almost every Q4 group marked inert; metadata must publish the invalid-row
    # sentinel without letting attention dereference it.
    active_groups = 3
    token_to_request[active_groups * group_size :].zero_()
    query_positions[active_groups * group_size :].fill_(-1)
    raw_output.fill_(torch.nan)
    prepared_output.fill_(torch.nan)
    build_prims_ts_qsa_page4_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        torch.empty(metadata_bytes, dtype=torch.uint8, device="cuda"),
        group_size=group_size,
        storage_page_size=storage_page_size,
        out=(raw_indptr, raw_indices, raw_seq_lens),
    )
    _run_private_qsa_decode(
        raw_query,
        k_cache,
        v_cache,
        raw_indptr,
        raw_indices,
        raw_seq_lens,
        max_seq_len,
        group_size,
        raw_output,
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        plan.seq_lens[active_groups:],
        torch.ones_like(plan.seq_lens[active_groups:]),
    )
    assert torch.isfinite(prepared_output).all()
    torch.testing.assert_close(
        _as_lower_level_decode_view(prepared_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_attention_unified_workspace_matches_two_step_cuda_graph() -> None:
    group_size = 4
    block_topk = 512
    blocks, table, requests, positions, storage_page_size = _make_case(
        group_size,
        block_topk,
    )
    num_routes = table.shape[0]
    num_qo_heads = 12
    head_dim = 256
    torch.manual_seed(9231)
    query = torch.randn(
        _fixed_qsa_shape(num_routes, group_size, num_qo_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_query = _as_lower_level_decode_view(query)
    num_physical_pages = int(table.max().item()) + 1
    k_cache = torch.randn(
        num_physical_pages,
        1,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    unified_bytes = get_prims_ts_qsa_workspace_size(
        query,
        k_cache,
        table,
        block_topk=block_topk,
        out_dtype=torch.bfloat16,
    )
    unified_workspace = torch.full(
        (unified_bytes,),
        0x55,
        dtype=torch.uint8,
        device="cuda",
    )
    unified_layout = _get_prims_ts_qsa_workspace_layout(
        blocks.shape[0],
        block_topk,
        table.shape[1],
        storage_page_size,
        group_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=1,
        head_dim=head_dim,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=torch.bfloat16,
        device="cuda",
    )
    unified_views = unified_layout.bind(unified_workspace)
    unified_output = torch.empty_like(query)
    prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        unified_workspace,
        out=unified_output,
    )

    metadata_bytes = get_prims_ts_qsa_metadata_workspace_size(
        blocks.shape[0],
        table.shape[1],
        storage_page_size,
        group_size,
    )
    metadata_shapes = get_prims_ts_qsa_metadata_output_shapes(
        blocks.shape[0],
        block_topk,
        group_size,
    )
    raw_metadata = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in metadata_shapes
    )
    build_prims_ts_qsa_page4_metadata(
        blocks,
        table,
        requests,
        positions,
        torch.empty(metadata_bytes, dtype=torch.uint8, device="cuda"),
        group_size=group_size,
        storage_page_size=storage_page_size,
        out=raw_metadata,
    )
    max_seq_len = group_size * (block_topk + 1) * 4
    raw_output = torch.empty_like(raw_query)
    _run_private_qsa_decode(
        raw_query,
        k_cache,
        v_cache,
        raw_metadata[0],
        raw_metadata[1],
        raw_metadata[2],
        max_seq_len,
        group_size,
        raw_output,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(unified_views.qsa_page_indptr, raw_metadata[0])
    torch.testing.assert_close(unified_views.seq_lens, raw_metadata[2])
    torch.testing.assert_close(
        _as_lower_level_decode_view(unified_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )

    graph_output = torch.empty_like(query)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prims_ts_qsa_attention(
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            unified_workspace,
            out=graph_output,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        _as_lower_level_decode_view(graph_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
@pytest.mark.parametrize("block_topk", (8, 512))
@pytest.mark.parametrize("storage_page_size", (4, 16, 256))
def test_fused_qsa_metadata_matches_reference(
    group_size: int,
    block_topk: int,
    storage_page_size: int,
) -> None:
    blocks, table, requests, positions, _ = _make_case(
        group_size, block_topk, storage_page_size
    )
    workspace_bytes = get_prims_ts_qsa_metadata_workspace_size(
        blocks.shape[0], table.shape[1], storage_page_size, group_size
    )
    workspace = torch.full((workspace_bytes,), 0x55, dtype=torch.uint8, device="cuda")
    output_shapes = get_prims_ts_qsa_metadata_output_shapes(
        blocks.shape[0], block_topk, group_size
    )
    outputs = tuple(
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in output_shapes
    )
    actual = build_prims_ts_qsa_page4_metadata(
        blocks,
        table,
        requests,
        positions,
        workspace,
        group_size=group_size,
        storage_page_size=storage_page_size,
        out=outputs,
    )
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[2], expected[2])
    for group in range(expected[2].numel()):
        begin = int(expected[0][group].item())
        live_pages = (int(expected[2][group].item()) + 3) // 4
        if group_size == 1:
            page_capacity = block_topk + 1
            torch.testing.assert_close(
                actual[1][begin : begin + page_capacity],
                expected[1][begin : begin + page_capacity],
            )
        else:
            torch.testing.assert_close(
                actual[1][begin : begin + live_pages],
                expected[1][begin : begin + live_pages],
            )


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_qsa_physical_page4_matches_torch_reference() -> None:
    """Decode membership-packed locators when storage and sparse pages match."""

    group_size = 4
    block_topk = 8
    num_qo_heads = 12
    head_dim = 256
    positions = torch.arange(31, 35, dtype=torch.int64, device="cuda")
    requests = torch.zeros(group_size, dtype=torch.int32, device="cuda")
    blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda").repeat(
        group_size, 1
    )
    table = torch.arange(64, dtype=torch.int32, device="cuda").unsqueeze(0)
    torch.manual_seed(9527)
    query = torch.randn(
        1,
        1,
        group_size,
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.randn(64, 1, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(query)
    workspace = torch.empty(
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            table,
            block_topk=block_topk,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
    )

    reference = torch.empty_like(output, dtype=torch.float32)
    for query_idx, position in enumerate(positions.tolist()):
        visible_tokens = position + 1
        full_pages = min(visible_tokens // 4, block_topk)
        token_chunks = [k_cache[page, 0].float() for page in range(full_pages)]
        value_chunks = [v_cache[page, 0].float() for page in range(full_pages)]
        tail_tokens = visible_tokens % 4
        if tail_tokens:
            tail_page = visible_tokens // 4
            token_chunks.append(k_cache[tail_page, 0, :tail_tokens].float())
            value_chunks.append(v_cache[tail_page, 0, :tail_tokens].float())
        keys = torch.cat(token_chunks).unsqueeze(1).expand(-1, num_qo_heads, -1)
        values = torch.cat(value_chunks).unsqueeze(1).expand(-1, num_qo_heads, -1)
        scores = (
            torch.einsum("hd,thd->ht", query[0, 0, query_idx].float(), keys)
            / head_dim**0.5
        )
        probability = torch.softmax(scores, dim=-1)
        reference[0, 0, query_idx] = torch.einsum("ht,thd->hd", probability, values)

    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_qsa_metadata_handles_partial_request_groups() -> None:
    group_size = 4
    block_topk = 8
    storage_page_size = 16
    request_lengths = (5, 3)
    rows = sum(request_lengths)
    qo_indptr = torch.tensor([0, 4, 5, 8], dtype=torch.int32, device="cuda")
    requests = torch.tensor([0] * 5 + [1] * 3, dtype=torch.int32, device="cuda")
    positions = torch.tensor(
        [31 + index for index in range(5)] + [47 + index for index in range(3)],
        dtype=torch.int64,
        device="cuda",
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [torch.roll(base_blocks, row % 3) for row in range(rows)]
    ).contiguous()
    table = torch.arange(2 * 64, dtype=torch.int32, device="cuda").reshape(2, 64)
    groups = qo_indptr.numel() - 1
    workspace_bytes = get_prims_ts_qsa_metadata_workspace_size(
        rows,
        table.shape[1],
        storage_page_size,
        group_size,
        num_query_groups=groups,
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    output_shapes = get_prims_ts_qsa_metadata_output_shapes(
        rows,
        block_topk,
        group_size,
        num_query_groups=groups,
    )
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda") for shape in output_shapes
    )
    actual = build_prims_ts_qsa_page4_metadata(
        blocks,
        table,
        requests,
        positions,
        workspace,
        group_size=group_size,
        storage_page_size=storage_page_size,
        qo_indptr=qo_indptr,
        out=outputs,
    )
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
        qo_indptr,
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[2], expected[2])
    for group in range(groups):
        begin = int(expected[0][group].item())
        live_pages = (int(expected[2][group].item()) + 3) // 4
        torch.testing.assert_close(
            actual[1][begin : begin + live_pages],
            expected[1][begin : begin + live_pages],
        )


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", [1, 4])
def test_packed_qsa_groups_match_packed_q1_attention(group_size: int) -> None:
    block_topk = 512
    storage_page_size = 16
    rows = 8
    num_query_heads = 12
    head_dim = 256
    qo_indptr = (
        torch.arange(rows + 1, dtype=torch.int32, device="cuda")
        if group_size == 1
        else torch.tensor([0, 4, 5, 8], dtype=torch.int32, device="cuda")
    )
    q1_qo_indptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    requests = torch.tensor([0] * 5 + [1] * 3, dtype=torch.int32, device="cuda")
    positions = torch.tensor(
        [2047 + index for index in range(5)] + [2047 + index for index in range(3)],
        dtype=torch.int64,
        device="cuda",
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [torch.roll(base_blocks, row % 3) for row in range(rows)]
    ).contiguous()
    table = torch.arange(2 * 129, dtype=torch.int32, device="cuda").reshape(2, 129)
    # Zero Q/K makes softmax exactly uniform. Comparing the grouped result to
    # Q1 then checks route membership and packed output placement without
    # conflating them with different accumulation recurrences.
    query = torch.zeros(
        rows,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = (
        torch.arange(k_cache.shape[0], dtype=torch.float32, device="cuda")
        .div(k_cache.shape[0])
        .to(query.dtype)
        .view(-1, 1, 1, 1)
        .expand_as(k_cache)
        .contiguous()
    )
    packed_output = torch.empty_like(query)
    q1_output = torch.empty_like(query)
    packed_workspace = torch.empty(
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            table,
            block_topk=block_topk,
            out_dtype=query.dtype,
            qo_indptr=qo_indptr,
            max_seq_len_q=group_size,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    q1_workspace = torch.empty(
        get_prims_ts_qsa_workspace_size(
            query,
            k_cache,
            table,
            block_topk=block_topk,
            out_dtype=query.dtype,
            qo_indptr=q1_qo_indptr,
            max_seq_len_q=1,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        packed_workspace,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
        out=packed_output,
    )
    prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        q1_workspace,
        qo_indptr=q1_qo_indptr,
        max_seq_len_q=1,
        out=q1_output,
    )
    prepared_output = torch.empty_like(query)
    prepared_workspace = torch.empty_like(packed_workspace)
    plan = prepare_prims_ts_qsa_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        prepared_workspace,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
        out=prepared_output,
    )

    def run_prepared() -> None:
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=prepared_output,
        )

    run_prepared()
    torch.cuda.synchronize()
    torch.testing.assert_close(prepared_output, packed_output, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_prepared()
    prepared_output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(prepared_output, packed_output, rtol=0, atol=0)

    for row, request in enumerate(requests.tolist()):
        selected_v = v_cache[table[request, :128].long()].reshape(-1, head_dim)
        tail_tokens = (int(positions[row].item()) + 1) % 4
        if tail_tokens:
            tail_v = v_cache[table[request, 128].long(), 0, :tail_tokens]
            selected_v = torch.cat((selected_v, tail_v), dim=0)
        expected = (
            selected_v.float()
            .mean(dim=0)
            .to(query.dtype)
            .expand(num_query_heads, head_dim)
        )
        torch.testing.assert_close(
            packed_output[row],
            expected,
            rtol=2e-2,
            atol=2e-2,
            msg=lambda message, row=row: f"packed QSA row {row}: {message}",
        )
        torch.testing.assert_close(
            q1_output[row],
            expected,
            rtol=2e-2,
            atol=2e-2,
            msg=lambda message, row=row: f"Q1 QSA row {row}: {message}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q1_graph_replay_clears_stale_locator_suffix() -> None:
    group_size = 1
    block_topk = 512
    blocks, table, requests, positions, storage_page_size = _make_case(
        group_size, block_topk
    )
    output_shapes = get_prims_ts_qsa_metadata_output_shapes(
        blocks.shape[0], block_topk, group_size
    )
    outputs = tuple(
        torch.full(shape, 0x12345, dtype=torch.int32, device="cuda")
        for shape in output_shapes
    )

    def run() -> None:
        build_prims_ts_qsa_page4_metadata(
            blocks,
            table,
            requests,
            positions,
            group_size=group_size,
            storage_page_size=storage_page_size,
            out=outputs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Replay a much shorter sequence over rows previously containing all 513
    # locators. Every slot outside the new live prefix must become -1.
    positions.fill_(31)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    torch.testing.assert_close(outputs[0], expected[0])
    torch.testing.assert_close(outputs[1], expected[1])
    torch.testing.assert_close(outputs[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (4, 5))
def test_grouped_metadata_cuda_graph_reloads_inputs(group_size: int) -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    workspace_bytes = get_prims_ts_qsa_metadata_workspace_size(
        blocks.shape[0], table.shape[1], storage_page_size, group_size
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    output_shapes = get_prims_ts_qsa_metadata_output_shapes(
        blocks.shape[0], blocks.shape[1], group_size
    )
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda") for shape in output_shapes
    )

    def run() -> None:
        build_prims_ts_qsa_page4_metadata(
            blocks,
            table,
            requests,
            positions,
            workspace,
            group_size=group_size,
            storage_page_size=storage_page_size,
            out=outputs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    blocks.add_(16)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    torch.testing.assert_close(outputs[0], expected[0])
    torch.testing.assert_close(outputs[2], expected[2])
    for group in range(outputs[2].numel()):
        begin = int(expected[0][group].item())
        live_pages = (int(expected[2][group].item()) + 3) // 4
        torch.testing.assert_close(
            outputs[1][begin : begin + live_pages],
            expected[1][begin : begin + live_pages],
        )
