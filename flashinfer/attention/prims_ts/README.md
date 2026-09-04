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
| QSA sparse-block | [Packed-prefill and fixed-decode example](../../../examples/prims_ts/qsa_page4_attention.py) | `PrimsTSQSAPlan`, `validate_prims_ts_qsa_group_size`, `make_prims_ts_qsa_qo_indptr`, `get_prims_ts_qsa_workspace_size`, `prepare_prims_ts_qsa_attention`, `prims_ts_qsa_attention`; advanced metadata: `get_prims_ts_qsa_metadata_output_shapes`, `get_prims_ts_qsa_metadata_workspace_size`, `build_prims_ts_qsa_metadata` |
| Block-sparse FMHA | — | `BlockSparseTSWrapper`, `block_sparse_attention`; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_decode_mla_with_paged_kv_cache`, `get_prims_ts_batch_decode_mla_workspace_size`, `prims_ts_batch_decode_with_kv_cache_mla` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

## QSA sparse-block interface

QSA consumes selected logical sparse-block IDs without expanding them to token
indices. `block_indices` has one row per flattened query token, and
`block_table` maps each request's logical storage pages to physical cache
pages. K and V are separate tensors shaped
`[num_pages, Hkv, storage_page_size, D]`; the physical storage page size must
be a multiple of `sparse_block_size`. Every QSA sizing, metadata, prepare, and
eager API accepts `sparse_block_size=4`. The argument is a positive power of
two; only four is implemented today, and other power-of-two values raise
`NotImplementedError`. The metadata builder adds the final causal tail of up
to `sparse_block_size - 1` tokens and converts selected blocks into the dense
encoded route table consumed by attention.

The production QSA specialization supports bottom-right causal, non-windowed
attention only. It uses a 128-token K/V tile for every supported query group.
For one route, the scheduler computes
`group_rows = group_size * (Hq / Hkv)` and chooses the smallest qualified
TileQ in 8, 16, 32, or 64 that contains those rows. TileQ8 supports both direct
and split routes because the standalone reducer consumes actual logical rows.
Thus the caller fixes the semantic query group while the kernel caps padding
deterministically; it does not rewrite `group_size` from workload thresholds.

The framework chooses `group_size` explicitly from 1, 2, 4, or 5.
`validate_prims_ts_qsa_group_size` verifies that
`group_size * (Hq / Hkv) <= 64`; the combined workspace and launch APIs enforce
the same invariant even when the helper is not called. Packed prefill uses
`[total_q, Hq, D]` with request-safe `qo_indptr` routes whose maximum length is
the selected group size and always runs without split-KV. Uniform MTP decode
uses `[B, num_query_groups, group_size, Hq, D]` without query offsets. Fixed
decode may split K/V to fill otherwise idle capacity, but its fanout never
crosses the first active-CTA service wave, is bounded by available K/V work,
and is capped at 8.

Prefer `prepare_prims_ts_qsa_attention` for serving and CUDA graphs. Allocate
a byte-addressed workspace of the size returned by
`get_prims_ts_qsa_workspace_size`, prepare once, run once outside capture to
compile and initialize the plan, then capture `run` with stable input, output,
and workspace addresses. `prims_ts_qsa_attention` is an eager one-shot
convenience. The lower-level metadata shape, workspace, and
`build_prims_ts_qsa_metadata` functions are an advanced two-step interface for
frameworks that manage the resulting route table themselves.

Metadata and the attention producer launch in normal stream order. On a
fixed-decode split-KV path, the producer remains an ordinary non-PDL launch and
signals dependents only after completion and TMEM teardown. Only the separate
reducer uses programmatic dependent launch (PDL). It initializes its register
state and any required shared-memory storage before acquiring, then acquires
before reading any producer-written partial output or statistics. Independent
sequence-length and query-offset metadata may be read before that acquire.
Packed prefill has no split-KV reducer.

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

For the separate block-sparse FMHA API, qualified Q64/coarse-KV profiles retain
KV256 routes for page sizes 64 and 128. Optional `kv_valid_bits` is a
`torch.uint32` per-request bitset with shape
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
