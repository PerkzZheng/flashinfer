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
| QSA sparse-block | [Packed-prefill and fixed-decode example](../../../examples/prims_ts/qsa_page4_attention.py) | `PrimsTSQSAPlan`, `suggest_prims_ts_qsa_group_size`, `validate_prims_ts_qsa_group_size`, `make_prims_ts_qsa_qo_indptr`, `get_prims_ts_qsa_workspace_size`, `prepare_prims_ts_qsa_attention`, `prims_ts_qsa_attention`; advanced metadata: `get_prims_ts_qsa_metadata_output_shapes`, `build_prims_ts_qsa_metadata` |
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
to `sparse_block_size - 1` tokens and converts selected blocks into the compact
metadata triple consumed by attention:
`(qsa_page_indices, qsa_page_memberships, seq_lens)`.

QSA sizing, metadata, prepare, and eager attention calls take an explicit
`max_seq_len_kv`: the model's static per-request logical context bound and an
upper bound on every live `query_position + 1`, including current/MTP tokens.
It is not the number of tokens or physical pages allocated in the global KV
cache across all requests. Each dense block-table row must merely have enough
columns to address this logical bound. Q1 directly maps its selected and tail
blocks. Grouped metadata sorts at most `group_size * (block_topk + 1)`
candidates inside one CTA, unique-reduces equal logical block IDs while ORing
their query-membership bits, then translates only the compact union through
the dense block table. Its work and temporary storage are independent of the
configured model length and reserved KV-cache capacity.
The bound must remain static across prepared-plan and CUDA-graph reuse. A
framework should therefore pass its configured model length for eager and
captured execution alike rather than derive the value from a live batch or
from total KV-cache capacity.

For `groups` query groups and
`page_capacity = group_size * (block_topk + 1)`, `qsa_page_indices` is a
contiguous Int32 table shaped `[groups, page_capacity]`. Its live entries are
plain cache locators; query-membership bits are not fused into them.
`qsa_page_memberships` is a contiguous Int32 table shaped
`[groups, ceil(page_capacity / 4)]`. Each word packs four consecutive 8-bit
membership masks, and bit `i` in a byte marks visibility for query `i` in the
group. Q1 does not consume membership metadata, so its shape is `[groups, 0]`.
`seq_lens[g]` is the live compact K/V length in tokens; only the first
`ceil(seq_lens[g] / sparse_block_size)` locator slots and corresponding
membership bytes are valid. The remaining locator suffix and membership
padding are unspecified.

The production QSA specialization supports bottom-right causal, non-windowed
attention only. It uses a 128-token K/V tile for every supported query group.
For one route, the scheduler computes
`group_rows = group_size * (Hq / Hkv)` and chooses the smallest qualified
TileQ in 8, 16, 32, or 64 that contains those rows. TileQ8 supports both direct
and split routes because the standalone reducer consumes actual logical rows.
Thus the caller fixes the semantic query group while the kernel caps padding
deterministically; attention does not rewrite `group_size` at launch time.

The framework chooses `group_size` explicitly from 1, 2, 4, or 5. The optional
pure-host `suggest_prims_ts_qsa_group_size` policy prefers the largest legal
group whose request routes and useful split-KV fanout can fill one SM wave,
then falls back toward Q1 to expose more independent routes. The caller passes
a cached `multi_processor_count`; the helper performs no device query or tensor
read and is safe to use while building a CUDA-graph plan. Its
`selected_seq_len_kv` argument is the per-query candidate-token bound,
including the causal tail, rather than the original context length or global
cache capacity.
`validate_prims_ts_qsa_group_size` verifies that
`group_size * (Hq / Hkv) <= 64`; the combined workspace and launch APIs enforce
the same invariant even when the helper is not called. Packed prefill uses
`[total_q, Hq, D]` with request-safe `qo_indptr` routes whose maximum length is
the selected group size and always runs without split-KV. Query lengths need
not be divisible by the selected group: packed routes may be short, while a
fixed route may use consecutive semantic dummy rows for its suffix and discard
their outputs. Uniform MTP decode uses
`[B, num_query_groups, group_size, Hq, D]` without query offsets. Fixed decode
may split K/V to fill otherwise idle capacity, but its fanout never crosses the
first active-CTA service wave, is bounded by available K/V work, and uses the
qualified Q1 or grouped reducer fanout cap.

Prefer `prepare_prims_ts_qsa_attention` for serving and CUDA graphs. Allocate
a byte-addressed workspace of the size returned by
`get_prims_ts_qsa_workspace_size`, prepare once, run once outside capture to
compile and initialize the plan, then capture `run` with stable input, output,
and workspace addresses. `prims_ts_qsa_attention` is an eager one-shot
convenience. The lower-level metadata shape and
`build_prims_ts_qsa_metadata` functions are an advanced two-step interface for
frameworks that manage the resulting metadata triple themselves. The combined
attention interface hides all three tensors in its caller-owned byte workspace;
its required ``max_seq_len_kv`` argument validates the per-request logical
model bound and dense block-table coverage.

On SM90 and newer, the combined API uses a PDL handoff from the single metadata
grid to attention. The attention consumer initializes its independent state
before waiting immediately ahead of the first metadata-dependent read. Older
devices retain ordinary stream ordering. A
fixed-decode split-KV path uses the same rule for attention-to-reducer PDL:
attention signals only after completion and TMEM teardown, and the reducer
waits before reading partial outputs, statistics, or metadata-produced QSA
sequence lengths. Packed prefill has no split-KV reducer. Standalone attention
over already-built QSA metadata remains stream ordered.

For `BlockSparsePagedTSWrapper`, `plan` freezes only the compact fixed-Q
geometry, dtypes, sparse-route capacity, and `max_seq_len_kv`; it retains no
request metadata. Every `run` reads live paged-KV row offsets, physical page
IDs, per-request K/V lengths, per-KV-head sparse routes, and optional token
bits from device tensors. The physical-page ID tensor is capacity: its live
prefix ends at `paged_kv_indptr[-1]`, which may be smaller than its `numel()`.
The caller owns every live value contract: dense K/V lengths must be in
`[1, max_seq_len_kv]`, and causal lengths must be in `[Sq, max_seq_len_kv]`.
`paged_kv_indptr` must start at zero and contain bounded, monotone rows with at
least `ceil(seq_lens_kv[b] / page_size)` entries; every physical page ID in
the live prefix ending at `paged_kv_indptr[-1]` must lie in `[0, P)`. Every BSR
row must have bounded offsets, strictly increasing unique block IDs, and at
most the planned `max_blocks_per_row` entries. Contiguous IDs must lie below
`ceil(seq_len_kv / kv_block_size)`; paged IDs must start below the owning
request's live K/V length.

Reusable wrappers validate tensor structure but read values directly without
host synchronization. Invalid values therefore have undefined behavior and
may access out of bounds. Set `CUTE_DSL_ENABLE_ASSERTIONS=1` before the process
first compiles these kernels to diagnose violations encountered while preparing
selected routes; such assertions report asynchronously and leave the CUDA
context unusable. The one-shot APIs instead synchronize once to validate all
live values, including the complete physical-page-ID prefix, before creating
their temporary plans and cannot run during CUDA Graph capture.

The one-shot `block_sparse_attention_with_paged_kv_cache` API takes
`max_seq_len_kv` as the static capacity and requires `seq_lens_kv` with the
live per-request logical lengths. Paged PrimTS does not support packed or
mixed/variable Q lengths.
Eager launches retain all launch tensors on the run stream; CUDA Graph users
must keep the wrapper and Q/cache/output/runtime-metadata tensors alive and
unmodified until replay completes. Values may change between completed replays
while tensor addresses, shapes, dtypes, and strides remain stable.

For the separate block-sparse FMHA API, qualified Q64/coarse-KV profiles retain
KV256 routes for page sizes 64 and 128. Optional `kv_valid_bits` is a
`torch.uint32` per-request bitset with shape
`[B, ceil(max_seq_len_kv / 32)]` over logical KV tokens; it is shared by all KV
heads and independent of the physical page mapping.

For contiguous block-sparse attention, both `BlockSparseTSWrapper.plan` and
the `block_sparse_attention` one-shot API can opt into
`sparse_format="bitmask"` and/or `use_proxy_routes=True`. BSR and packed
exact-block bitmaps are alternative frontends; both are prepared into the same
route stream before attention. The bitmask one-shot uses the full structural
KV-block count as its temporary plan capacity, while reusable plans accept a
tighter caller-provided bound. Proxy routes are supported across the existing
contiguous block-sparse profiles and preserve the profile's Q tile, KV route,
and KeepsAB/SWAPAB geometry. Proxy routes currently require
`mask_type="dense"`; paged K/V proxy execution remains unsupported. Route rows
are owned by `(batch, KV head, Q block)`, so all Q heads in one GQA/MQA group
share sparsity. A proxy run supplies one K arithmetic mean and one V sum per
semantic KV block. The final partial block uses only its structural tokens.
Optional `kv_valid_bits` filters exact K/V tokens only and does not change
proxy summaries or their represented mass.

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
