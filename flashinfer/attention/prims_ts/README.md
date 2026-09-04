# Experimental Task-Scheduled Attention

`flashinfer.attention.prims_ts` exposes experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but is not yet signoff-qualified.

## Guides and public APIs

Import all entries below from `flashinfer.attention.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md) | `BatchPrefillTSWrapper`, `batch_prefill`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prepare_prims_ts_batch_decode_with_kv_cache`, `prims_ts_batch_decode_with_kv_cache` |
| QSA page-4 | [Packed-prefill and fixed-decode example](../../../examples/prims_ts/qsa_page4_attention.py) | `validate_prims_ts_qsa_group_size`, `make_prims_ts_qsa_qo_indptr`, `get_prims_ts_qsa_workspace_size`, `prepare_prims_ts_qsa_attention`, `prims_ts_qsa_attention` |
| Block-sparse FMHA | — | `BlockSparseTSWrapper`, `block_sparse_attention`; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_decode_mla_with_paged_kv_cache`, `get_prims_ts_batch_decode_mla_workspace_size`, `prims_ts_batch_decode_with_kv_cache_mla` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

## QSA page-4 interface

QSA consumes selected logical four-token block IDs without expanding them to
token indices. `block_indices` has one row per flattened query token, and
`block_table` maps each request's logical storage pages to physical cache
pages. K and V are separate tensors shaped
`[num_pages, Hkv, storage_page_size, D]`; the physical storage page size must
be a multiple of four. The metadata builder adds the zero-to-three-token
causal tail and converts selected blocks into the encoded page-4 CSR consumed
by attention.

The framework chooses `group_size` explicitly from 1, 2, 4, or 5.
`validate_prims_ts_qsa_group_size` verifies that
`group_size * (Hq / Hkv) <= 64`; the combined workspace and launch APIs enforce
the same invariant even when the helper is not called. Packed prefill uses
`[total_q, Hq, D]` with request-safe `qo_indptr` routes whose maximum length is
the selected group size. Uniform MTP decode uses
`[B, num_query_groups, group_size, Hq, D]` without query offsets.

Prefer `prepare_prims_ts_qsa_attention` for serving and CUDA graphs. Allocate
one byte workspace with `get_prims_ts_qsa_workspace_size`, prepare once, run
once outside capture to compile and initialize the plan, then capture `run`
with stable input, output, and workspace addresses. `prims_ts_qsa_attention`
is an eager one-shot convenience. The lower-level metadata shape, workspace,
and build functions are an advanced two-step interface for frameworks that
manage the resulting CSR tensors themselves.

For `BlockSparsePagedTSWrapper`, `plan` freezes the logical fixed-Q geometry
and copies the paged-KV row offsets and optional per-request K/V lengths into
plan-owned storage. When lengths are provided, the scalar K/V length is the
static maximum; each page-table row may contain spare entries beyond its live
length. `run` consumes live physical page IDs and per-KV-head sparse routes,
and every selected BSR block must start before that request's frozen live K
length. A violating sorted row fails closed. Eager
launches retain all launch tensors on the run stream; CUDA Graph users must
keep the wrapper and Q/cache/output/runtime-metadata tensors alive and
unmodified until replay completes.

Qualified Q64/coarse-KV profiles retain KV256 routes for page sizes 64 and
128. Optional `kv_valid_bits` is a `torch.uint32` per-request bitset with shape
`[B, ceil(max_seq_len_kv / 32)]` over logical KV tokens; it is shared by all KV
heads and independent of the physical page mapping.

## Validation

Run the numerical, graph, scheduler/resource, alias-safety, and public-surface
contracts:

```bash
pytest -q \
  tests/attention/test_attention_ts_context.py \
  tests/attention/test_attention_ts_decode.py \
  tests/attention/test_attention_ts_qsa_metadata.py \
  tests/attention/test_attention_ts_block_sparse.py \
  tests/attention/test_attention_ts_mask.py \
  tests/attention/test_attention_ts_mla_decode.py
```
