# Task-Scheduled MLA Decode

Native D512/D576 two-source sparse MLA is available through
`BatchSparseMLADecodePagedTSWrapper` (see its `plan()` / `run()` docstrings).
Every sparse schedule consumes separate storage-row index lists and live lengths
for the primary and optional extra pool. Pass `indices`, `lengths`, optional
`extra_indices`/`extra_lengths`, and attention sinks directly to `run()`.
The caller supplies two effective scales: `softmax_scale = attention_scale *
q_descale * kv_descale` and `output_scale = output_multiplier * kv_descale`.
Both pools share the KV descale. Sink logits retain their original scale;
`output_scale` applies before output quantization, while `out_scale` holds its
generated per-block scales. Dynamic scale composition belongs in upstream work.
Index lists use `-1` for holes and need no 128-entry padding.
The kernel keeps source boundaries aligned to its KV tiles internally.
The dense API contract below retains its 512-latent plus 64-RoPE layout.

## Block-scaled sparse caches

The cache-format selectors support token-interleaved block-scaled storage with
E4M3 Q and ordinary FP8 MMA. `kv_cache_format` selects the primary pool;
`extra_kv_cache_format` defaults to that format. D512 pools may independently
use `"mxfp8"` or `"nvfp4"`. Native and mixed-RoPE pools require matching formats.
Caches use uint8 records; MXFP8 also accepts an E4M3 byte view of the complete
528-byte records, including the scale bytes.

| `kv_cache_format` | QK / V dimensions | `qk_rope_head_dim` | Bytes per token | Scale byte for block `b` |
| --- | --- | --- | --- | --- |
| `"mxfp8"` (DS v4.1 SWA) | 512 / 512 | 0 | 512 E4M3 + 16 UE8M0 scales | `512 + b` (block32) |
| `"nvfp4"` (DS v4.1) | 512 / 512 | 0 | 256 packed E2M1 + 32 E4M3 scales | `256 + b` |
| `"nvfp4.fp8rope"` | 576 / 512 | 64 | 256 packed E2M1 + 64 E4M3 RoPE + 32 E4M3 scales | `320 + 8*(b % 4) + b//4` |

NVFP4 scales cover 16 consecutive values; even-indexed values occupy the low
nibble. MXFP8 scales cover 32 values and encode powers of two. Select
`kv_cache_format="mxfp8", extra_kv_cache_format="nvfp4"` for FP8 SWA plus
compressed NVFP4 KV. There are no global storage scales. In the DS v4.1 layout all 512
channels, including the rotated channels, are quantized uniformly. A zero
`qk_rope_head_dim` means there is no additional QK-only tail; the caller still
applies any required query/key rotation before attention. The mixed format
copies its FP8 tail unchanged. NVFP4 conversion applies one saturating RNE
rounding to E4M3. Both QK and PV use E4M3 operands and FP32 accumulation,
with BF16 O by default.

Physical storage-row indices still include the external page stride. The
internal page-size-one gather view does not repack or change the external cache.
The default softmax scale is `1/sqrt(576)` for D576 and `1/sqrt(512)` for D512.

The H64 throughput policy uses WS M64 QK, native Layout E softmax/correction,
Q NoPE in TMEM, and two expanded KV tiles retained through both N256 PV passes.
Eight loader warps convert each tile gathered by metadata. NVFP4 readers snapshot their
packed data before releasing raw scratch. MXFP8 readers release it after their
last raw read. Q stages through an empty KV slot before the attention loop.

Small decode grids with long selected lists also use the M64 whole-tile path.
KV splits fill roughly one resident wave, using power-of-two split counts to
limit merge traffic. Small H128 grids with an MXFP8 primary pool can use two
M64 head tiles: repeated KV conversion costs less than group-2 startup in this
region. The crossover depends on planned query rows, head tiles, selected-list
capacity and SM count; it is a measured scheduling choice, not a cache-layout
restriction. No input preparation or extra cache representation is introduced.

The standalone reducer for unquantized M64 whole-tile output processes two
complete heads per CTA with shared row statistics and vec8 partial-O loads.
This halves the block count while retaining the same per-output accumulation
order. Quantized output retains its existing one-head reducer. Both forms
acquire the producer dependency before reading partial results.

The 1CTA whole-tile path gathers complete records with a shared tensor-map
helper. MXFP8 uses a Uint32 view of the native 528-byte extent and a 544-byte
shared-memory pitch: TMA zero-fills the extra 16 bytes so every four-row
destination is 128-byte aligned. This padding is internal; the caller's cache
layout and physical row indices are unchanged.

The H128 throughput path uses M128 group-2 MMA with eight loader warps per CTA. Each CTA converts
64 distinct rows of each 128-row KV tile once, writes its local V features,
and exchanges the peer's V features through DSMEM (16 KiB in each direction).
There is no second global V gather or duplicate value conversion. NVFP4 K uses
whole-record gathers; MXFP8 K uses panel gathers with separate scale scratch.

D512 uses two K stages, three V stages, and direct asynchronous DSMEM stores
from the conversion registers. D576 uses two stages for each operand and
warp-private bulk copies from converted K. Coupled K/V acquisition retains the
bulk-copy source until both V consumers finish. One publishing warp per CTA
waits for incoming DSMEM completion before allowing the V stage to become full.
K publication already fences ordinary local V stores, so V publication only
needs to order the completion acquire before its release.

Both schedules copy FP8 Q into TMEM and use one score stage to reserve its
columns. The Q copy geometry is shared with the 1CTA WS path; the 2CTA copy
and dependent MMAs issue in the same leader warp. The Q SMEM lifetime remains
protected by its existing pipeline, including the separate D576 RoPE tail.

Loaders reuse the next metadata stage under its consumer token for raw K
prefetch. Source boundaries are cached from live lengths once per query. With
`assume_valid_prefix=True`, each CTA stages only its own 64 row indices; the
general hole-masking path retains the full metadata tile. Persistent grids
balance complete work waves using the existing planned query/head geometry.
The 2CTA loader selects the source format before calling a format-specialized
producer, keeping each format's gather and conversion branches together.

The `scaled_kv/` package owns storage layouts, conversions, shared-memory
publication and TMA preparation for both kernel families. The 2CTA producer
keeps its task/resource ownership in `throughput_2cta/scaled_kv_load.py`.
Scaled 2CTA execution always shares converted K/V through DSMEM; the independent
reload experiment is retained only in the local validation archives.

Both families share record-map construction, source and scale layout helpers,
packed conversion, vector stores, prefetch, and publication. Cluster consumers
acquire at cluster scope after producers release converted shared data.
All conversion lanes receive the source tag, including invalid first entries.
The MXFP8 scaler shares scale setup across 16 values, uses packed FP16 arithmetic
for exactly representable powers of two, and falls back to FP32 for the remaining
UE8M0 range. Both paths implement the same saturating RNE E4M3 result.

H8/H16 use narrow Swap-AB Q tiles with the same raw gather and retained
converted tiles. H32 can also use Q=32; the measured throughput policy prefers
WS Q=64 because its N256 PV requires two feature passes instead of four.
Narrow whole-tile schedules compact softmax statistics to four column-group
leaders per warp. Tail sums are reduced before publication with the same
shuffle order, and correction consumes those warp sums without reducing twice.
This removes redundant statistics storage and makes Q=32 fit the SMEM budget.
KV release follows the final PV feature slice for every Q tile width.

Both families select primary/extra descriptors from source-tagged indices,
including the prefetched next tile. Live lengths may move the source boundary
between graph replays. Different physical page sizes and page padding are
allowed; each pool retains its own planned record width and scale layout.

For BF16 queries, allocate an E4M3 scratch tensor before capture and call
`q_fp8.copy_(q_bf16)` immediately before `wrapper.run(...)` **inside** the graph.
Full-call measurements include this conversion, attention and any reduction.
Prepared-Q attention-only measurements omit the conversion and have a different
scope. The local reproduction uses GB300, CUDA 13.4, CUTLASS DSL 4.8.1, the native
page-64 cache, B32/Q8, top-k 2048 and context 65536 per request. Older toolchains
retain the conversion fallback; the packed multiply is capability-checked.

**FlashMLA reference scope:** in reference revision
`e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b`, the 352-byte D576/H128 cache path
launches the legacy H64 attention kernel twice, for heads 0–63 and 64–127,
then combines split-KV partials. Both passes use all 576 QK dimensions and
repeat KV gathering/conversion. The D512 V4.1 path instead has a native H128
kernel. Report this launch strategy alongside absolute times and speedup
ratios: a D576/H128 speedup against that revision is against the two-launch
implementation, and an optimized native D576/H128 reference could narrow it.
This is implementation coverage, not a hardware restriction. CUDA graphs
retain both launches; attention-only timing must include both and the combine.

This directory contains the CuTe DSL task-scheduled (TS) Multi-head Latent
Attention (MLA) decode kernels used by FlashInfer's experimental Blackwell
paged-cache APIs. The implementation accepts the post-matrix-absorption MLA
layout in the dense API: every query/cache row contains a 512-element latent component followed
by a 64-element RoPE component, while output contains the 512 latent values.

FlashInfer selects the 1-CTA throughput/latency family or the 2-CTA throughput
family automatically. Flat-row tiling, persistent scheduling, split-KV, and
local versus separate reduction are implementation decisions. Unsupported
shape, dtype, or mask combinations raise an error rather than falling back to
another backend.

Persistent dispatch may use either a static schedule or cluster launch control
(CLC). When the selected policy enables CLC, it assigns work to resident CTAs.

## Public APIs

Import these entry points from `flashinfer.attention.prims_ts`:

| API | Use |
| --- | --- |
| `BatchMLADecodePagedTSWrapper` | Reusable static `plan()` plus per-run request-metadata `run()` interface. |
| `batch_mla_decode_with_paged_kv_cache` | One-shot interface with optional caller scratch, explicit bounds, and trusted capture-safe execution. |
| `get_prims_ts_batch_mla_decode_workspace_size` | Size caller-owned standalone scratch. |

Trace a planned stateful wrapper with `flashinfer.fi_trace(wrapper.run, ...)`.
The unbound `wrapper.run.fi_trace(...)` form is rejected because it cannot
carry the wrapper's plan-owned fixed-versus-packed query mode.

`plan()` receives the device, exact batch and head geometry, page size, static
Q and K/V bounds, dtypes, mask, and fixed-versus-packed query mode. It compiles
the specialization and either binds an optional caller-owned workspace or
allocates private scratch; it does not retain request metadata. Every `run()`
supplies the current query, cache, dense block table, and K/V lengths, plus Q
offsets for packed queries. Validation is enabled by default. `validate=False`
skips explicit wrapper checks and host metadata reads for a previously
validated steady state or CUDA Graph launch; the caller then owns every value,
bounds, aliasing, and lifetime precondition.

The one-shot helper reads request metadata to derive plan bounds and constructs
a temporary wrapper, so it is not CUDA-graph-capturable. Graph-sensitive callers
must plan `BatchMLADecodePagedTSWrapper` before capture and bind the live
metadata directly in `run()`.

## Dense API contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Query/cache dimensions | `kv_lora_rank=512`, `qk_rope_head_dim=64` (576 input elements) |
| Output dimension | 512 |
| Q/cache dtype | Matching `torch.bfloat16` or `torch.float8_e4m3fn` |
| Output dtype | `torch.bfloat16` only |
| Q length | Fixed or packed variable length; packed requests may be empty, while the static maximum remains positive |
| K/V length | Positive runtime lengths are at most `2**31 - 32768`; nonpositive lengths mark inactive requests. The reserve keeps the largest padded split-KV coordinate span in signed `int32`. |
| Q heads | Validated at 6, 8, 12, 16, 24, 32, 48, 64, 96, and 128; other positive counts are accepted only when automatic selection reports an implementation |
| K/V cache | Paged, one logical KV head; `[num_pages, page_size, 576]` or `[num_pages, 1, page_size, 576]` compact storage |
| Metadata/cache index extents | Flattened query-head capacity, block-table elements, and physical page count must fit signed `int32` |
| Page size | 16, 32, 64, or 128 tokens |
| Mask | Dense or bottom-right causal |
| Scheduling | Automatic nonpersistent, static-persistent, or CLC-persistent selection by kernel family and work shape |
| Accumulation | FP32 QK/PV and softmax state |

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but remains to be qualified.

Some head and Q-length combinations outside the validated matrix do not have a
generated implementation and are rejected.

Query, cache, and output tensors must be compact, 16-byte-aligned CUDA tensors
on the metadata device. `block_tables` is a 4-byte-aligned CUDA
`torch.int32[B, max_num_pages]` view with contiguous, non-overlapping rows:
`stride(1) == 1` and `stride(0) >= max_num_pages`. This includes both compact
tables and a plane view sliced from `[B, 2, max_num_pages]` metadata.
`seq_lens` is compact, 4-byte-aligned CUDA `torch.int32`; packed `qo_indptr`
is contiguous CUDA `torch.int32`. A caller-provided `out` must not overlap
query, cache, `block_tables`, `seq_lens`, packed `qo_indptr`, or caller-owned
workspace.
Storage overlap is an unchecked caller precondition in both validation modes. The API returns O
only. FP32 LSE is internal workspace and is not exposed as an output.

## Tensor and metadata layouts

- Fixed query: `[B, SQ, H, 576]`.
- Packed query: `[total_q, H, 576]`, with contiguous CUDA `int32[B + 1]`
  `qo_indptr`.
- Fixed output: `[B, SQ, H, 512]`.
- Packed output: `[total_q, H, 512]`.
- Cache storage has one logical KV head and may omit or retain that singleton
  axis: `[num_pages, page_size, 576]` or
  `[num_pages, 1, page_size, 576]`. "Compact" means adjacent elements follow
  those row-major shapes without padding in the inner page, token, or feature
  dimensions.
- Dense page table: inner-contiguous, non-overlapping CUDA
  `int32[B, max_num_pages]` `block_tables`. Padding between rows is accepted.
- Runtime lengths: contiguous CUDA `int32[B]` `seq_lens`. Positive lengths must not exceed the static K/V bound; nonpositive lengths mark inactive requests.

Internally, the query-token and query-head axes form one affine row space,
`flat_row = query_idx * H + head_idx`. The 1-CTA profiles partition this space
with physical M8/M16/M32/M64 tiles, while the 2-CTA family uses M128. Therefore
the number of scheduled query tiles is `ceil(H * SQ / M)` and only the final
physical tile can contain inactive rows. Public tensor shapes remain unchanged;
valid physical rows map back to their logical `(query_idx, head_idx)` output
coordinates.

Split-KV workspaces intentionally retain rectangular physical geometry with
`M * ceil(H * SQ / M)` rows. Producer stores and every reducer use the same
mapping, and final-tile rows beyond `H * SQ` are predicated from public output.
For packed Q, each request applies the same mapping to its runtime query length;
rectangular tiles beyond a shorter request's active prefix retire scheduler
bookkeeping without issuing Q/K/V or output accesses.

For causal request `b`, query row `i` can attend through
`seq_lens[b] - query_length[b] + i`. `bmm1_scale` and `bmm2_scale` default to
1 and must be finite, positive Python scalars representable as positive
`float32` values.

Each `block_tables` row must contain at least
`ceil(max_kv_len / page_size)` columns, and every page ID used by a runtime
length must index the physical cache. Packed offsets start at zero, are
nondecreasing, end at `total_q`, and have every nonnegative per-request delta
no greater than `max_seq_len_q`.

The wrapper receives `block_tables`, `seq_lens`, and packed `qo_indptr` on
every run. Their storage and values may change between completed launches
without replanning while active page IDs remain valid, positive K/V lengths stay within the static bound, packed-Q deltas stay nonnegative and within their bound,
and the final packed offset equals the current query/output extent. Causal
metadata must also preserve `q_len[b] <= seq_lens[b]` for every active request.
Individual packed requests may be empty. An all-empty packed launch requires a
positive `max_seq_len_q` and returns an empty output without GPU dispatch.

With default `validate=True`, a wrapper run checks those metadata values and
the tensor and output contracts. Once the caller has established
the conditions, `validate=False` avoids the explicit checks and host metadata
reads. The caller-workspace standalone launch likewise trusts device-side
metadata values, but still validates tensor structure.
Invalid per-run page IDs, lengths, or offsets in either unchecked-value path
may cause incorrect results or out-of-bounds access. The caller owns the
aliasing contract in both validation modes. Do not mutate
metadata concurrently with a launch or graph replay that reads it. CUDA Graph
replay also requires stable captured addresses, shapes, and strides.

A wrapper owns its plan-bound mutable scratch and supports only one in-flight
run or captured-graph replay. If `workspace_buffer` is omitted from `plan()`,
the wrapper allocates private scratch. Use separate wrappers and workspaces for
concurrent execution. Caller-owned scratch must remain alive and must not
overlap query, K/V cache, metadata, or output storage.

## Dataflow and source map

```text
Q(latent | RoPE) + paged cache(latent | RoPE)
    -> staged Q/K/V + page IDs
    -> QK MMA -> masked online softmax -> P
    -> PV MMA -> corrected latent O + internal log-normalizer state
    -> direct O, local split reduction, or partials -> reducer -> O
```

The 1-CTA throughput/latency and 2-CTA throughput families select their launch,
reduction topology, and nonpersistent, static-persistent, or CLC-persistent
schedule automatically. Callers do not select a scheduler or kernel family
through the public wrappers.

For a static split-KV 2-CTA launch, every reducer follows the same physical
flat-row geometry. Padded M128 tails retain the compact logical-row reducer;
fully populated layouts may use row-parallel or clustered reduction when their
work and capacity bounds are satisfied.

The BF16 2-CTA path enables CLC only when logical work exceeds one resident
wave. Inactive physical Q tiles, pruned split slots, and zero-visible-K tiles skip
their Q/K/V and TMEM data work and skip the active-tile throttle edge
symmetrically. Every participating task still advances the work queue, so all
tasks retire the same work-tile sequence. This matched progression is the
synchronization contract for CLC replay and runtime pruning.

For BF16 2-CTA paged K/V, the TMA warp holds one 32-page-ID register window.
It refreshes the window after `32 / pages_per_k_tile` logical K tiles, when the
next 32 page-table entries are needed. This period follows directly from page
and tile geometry and is internal to the task schedule.

| Source | Responsibility |
| --- | --- |
| [`../../mla_decode.py`](../../mla_decode.py) | Public validation, automatic family selection, workspace binding, JIT caching, and launch adaptation |
| [`kernel_policy.py`](kernel_policy.py) | Eligibility and automatic kernel-family selection |
| [`throughput_latency_1cta/`](throughput_latency_1cta/) | Latency-oriented 1-CTA configs, resources, tasks, kernels, and reducers |
| [`throughput_2cta/`](throughput_2cta/) | M128 2-CTA configs, resources, tasks, kernels, and reducers |
| [`parallel_reduction_topology.py`](parallel_reduction_topology.py) | Shared split-reduction topology decisions |
| [`helpers/`](helpers/) | Layout, mask, schedule, math, and tile helpers |

## Example

```python
import torch
from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper

device = "cuda"
B, H = 2, 16
latent_dim, rope_dim = 512, 64
page_size, pages_per_request = 32, 4
num_pages = B * pages_per_request

query = torch.randn(
    B, 1, H, latent_dim + rope_dim,
    device=device,
    dtype=torch.bfloat16,
)
kv_cache = torch.randn(
    num_pages, page_size, latent_dim + rope_dim,
    device=device,
    dtype=torch.bfloat16,
)
block_tables = torch.arange(
    num_pages, device=device, dtype=torch.int32
).view(B, pages_per_request)
seq_lens = torch.full(
    (B,), pages_per_request * page_size, device=device, dtype=torch.int32
)

wrapper = BatchMLADecodePagedTSWrapper()
wrapper.plan(
    query.device,
    B,
    H,
    latent_dim,
    rope_dim,
    page_size,
    pages_per_request * page_size,
    max_seq_len_q=1,
    packed_query=False,
    q_data_type=query.dtype,
    kv_data_type=kv_cache.dtype,
    o_data_type=torch.bfloat16,
    mask_type="causal",
)
out = wrapper.run(query, kv_cache, block_tables, seq_lens)
assert out.shape == (B, 1, H, latent_dim)

# Packed Q uses cumulative per-request offsets and compact token-major rows.
q_lens = (1, 3)
qo_indptr = torch.tensor((0, 1, 4), device=device, dtype=torch.int32)
packed_query = torch.randn(
    sum(q_lens), H, latent_dim + rope_dim,
    device=device,
    dtype=torch.bfloat16,
)
packed_wrapper = BatchMLADecodePagedTSWrapper()
packed_wrapper.plan(
    packed_query.device,
    B,
    H,
    latent_dim,
    rope_dim,
    page_size,
    pages_per_request * page_size,
    max_seq_len_q=max(q_lens),
    packed_query=True,
    q_data_type=packed_query.dtype,
    kv_data_type=kv_cache.dtype,
    o_data_type=torch.bfloat16,
    mask_type="causal",
)
packed_out = packed_wrapper.run(
    packed_query,
    kv_cache,
    block_tables,
    seq_lens,
    qo_indptr=qo_indptr,
)
assert packed_out.shape == (sum(q_lens), H, latent_dim)
```

For the standalone API, call
`get_prims_ts_batch_mla_decode_workspace_size()` with the same shape, dtype,
mask, and Q-bound arguments as the launch. Allocate at least that many bytes as
a contiguous, 32-byte-aligned CUDA `torch.int8` or `torch.uint8` tensor. The
buffer includes internal FP32 LSE storage, does not require initialization,
and is exclusive to one in-flight launch or captured graph. It must not overlap
query, K/V cache, metadata, or output storage. The standalone hot path does not
copy metadata to the host; callers must maintain all page, length, and
packed-offset preconditions for every launch. These live values are not fully
value-checked by the launch.

For CUDA graph capture, call `plan()` and perform one default-validating
`run()` first. Capture subsequent calls with `validate=False`, retain all
run-time metadata and workspace storage at stable addresses, and provide a
compact, 16-byte-aligned `out` tensor to avoid allocation.

## Limitations

- The latent/RoPE dimensions are fixed at 512/64.
- Output is `torch.bfloat16` even when query/cache input is
  `torch.float8_e4m3fn`.
- Only dense paged MLA cache addressing is exposed; sparse/indexed MLA,
  sliding windows, attention sinks, and custom masks are not public features.
- Query and cache dtypes must match.
- Automatic kernel selection is intentionally not user-selectable through the
  public wrapper.

## Validation

The public accuracy suite covers fixed and packed Q, `torch.bfloat16` and
`torch.float8_e4m3fn` input, dense and causal masks, all four page sizes, both
automatic kernel families, H6/H12/H24/H48/H96 flat-row cases, M128 tails,
runtime K pruning, split-KV reduction, output/workspace contracts, and CUDA
graphs:

```bash
pytest -q tests/attention/test_attention_ts_mla_decode.py
pytest -q tests/trace/test_fi_trace_template_consistency.py
```

Both dense 1CTA and 2CTA families treat nonpositive runtime K/V lengths as inactive requests. They skip Q/K/V and partial-output work, preserve caller-owned O/LSE, and retain queue, cluster, TMEM-lifecycle and PDL progress. Separate reducers do not consume stale scratch for these requests. Newly allocated inactive output is unspecified. Positive plan capacity is still required.

The two-source sparse API instead accepts nonnegative source lengths. An empty primary source may still attend to the extra source; two empty sources publish zero output and negative-infinity LSE.
