# QToken-KvBlock-Sparse Decode: Metadata and Split-Reducer Optimizations

The QToken-KvBlock-Sparse-Attention decode step of the task-scheduled (PrimTS)
kernels runs three kernels: the route metadata builder (CUDA C++, one CTA per
route), the task-scheduled attention kernel, and, for split-KV plans, the split
reducer. At the batch sizes used for speculative decoding (MTP-4, batch 16 to
64) the attention kernel is fixed-cost bound and the metadata and reducer
kernels are a large share of the chain. This document records the optimizations
made to those two kernels, why each one helps, the variants that were rejected,
and the numbers measured on GB300 (SM103). None of the changes depend on the
K/V cache data type; BF16 and FP8 K/V were measured and gain alike.

Commits (in order):

1. `perf(prims-ts): cut QToken metadata and split-reducer latency`
2. `perf(prims-ts): size the QToken union map from the model bound (>1M tokens)`
3. `perf(prims-ts): make the QToken bit-map union prefix-independent`
4. `perf(prims-ts): occupancy-aware QToken union dispatch (map vs sort)`
5. `perf(prims-ts): analytic occupancy model for the QToken union dispatch`

Files: `include/flashinfer/attention/prims_ts/q_token_kv_block_sparse_metadata.cuh`,
`csrc/prims_ts_q_token_kv_block_sparse_metadata.cu`,
`flashinfer/attention/prims_ts/kernels/fmha_decode/reduction.py`,
`fmha_decode_config.py`, `fmha_decode_constants.py`, `fmha_decode_kernel.py`,
`flashinfer/attention/prims_ts/decode.py`, the two PrimTS READMEs, and
`tests/attention/test_attention_ts_q_token_kv_block_sparse_metadata.py`.

## Summary of results

End-to-end decode step (metadata + attention + reducer, CUDA graph replay,
cold L2, mean of 200 iterations, GB300), MTP-4 routes (`group_size = 4`),
KV 8K context:

| K/V dtype | Batch | Before (us) | After (us) | Change |
|---|---:|---:|---:|---:|
| BF16 | 16 | 37.6 | 30.8 | -18.2% |
| BF16 | 32 | 46.4 | 40.0 | -13.9% |
| BF16 | 64 | 63.8 | 58.8 | -7.8% |
| FP8 | 16 | 39.5 | 32.7 | -17.1% |
| FP8 | 32 | 46.7 | 40.2 | -13.8% |
| FP8 | 64 | 62.3 | 56.8 | -8.9% |

KV 16K and 32K contexts gain 12-20% at batch 16 and 12-15% at batch 32 (full
matrix below). Single-query routes (`group_size = 1`) gain 5-10% up to batch
16 and 1-2% at batch 256, where the attention kernel dominates.

Kernel-level, batch 16, `group_size = 4`, KV 8K:

| Kernel | Before | After |
|---|---:|---:|
| Metadata builder | ~6 us (ncu, isolated) | 5.1 us (nsys, in the graph) |
| Split reducer | 8.2 us (ncu, isolated) | 4.5 us (nsys, in the graph), 6.8 us isolated |

In the graph timeline after the pass the attention grid starts 0.9 us after the
metadata kernel (its programmatic-dependent-launch wait releases when metadata
completes), and the reducer starts 0.35 us before the attention grid ends, so
about 4.2 us of the reducer remain exposed. The GPU kernel chain is 23.1 us;
the per-iteration benchmark figure of 28.7-30.8 us includes about 5.6 us of
graph-launch and measurement overhead that is constant across variants.

## Metadata kernel

The grouped-route path (`group_size` 2, 4, or 5) builds, per route, the union
of the `group_size * (block_topk + 1)` selected and causal-tail sparse blocks
of its queries, in ascending logical order, with a packed per-block
query-membership word, and maps the compact union through the dense page
table.

### Shared-memory block map instead of a radix sort

The union was a 544-thread CUB radix sort over the tagged candidates. It is now
a shared-memory block map: each candidate `atomicOr`s its query bit into its
logical block's entry, each thread counts its non-empty entries
(`__popc(__vsetne4)` over four blocks per word), one block-wide scan ranks
them, and a compaction writes the union in ascending order followed by a
balanced emit through the page table. The sort remains as the fallback and can
be forced with `FLASHINFER_QSA_METADATA_UNION=sort`.

Isolated kernel, 2 routes, `group_size = 4`, block top-k 512, route placed at
the end of the model so the causal prefix equals the model length (ncu, warm
L2):

| Causal prefix | Map | Map dynamic SMEM | Sort |
|---:|---:|---:|---:|
| 8K tokens | 6.6 us (byte) | 14.6 KiB | 9.6 us |
| 32K | 7.4 (byte) | 22.8 KiB | 10.5 |
| 128K | 9.1 (byte) | 47 KiB | 10.7 |
| 256K | 10.4 (bit) | 56 KiB | 11.5 |
| 512K | 11.0 (bit) | 56 KiB | 11.4 |
| 1M | 11.2 (bit) | 56 KiB | 11.2 |
| 4M | 13.8 (bit) | 154 KiB | 12.6 |
| 8M | sort fallback | n/a | ~12 |

### Model-sized hybrid map (models beyond 1M tokens)

The first version kept a static 32 KiB byte map and fell back to the sort for
models over 32768 sparse blocks (131072 tokens). The map now lives in dynamic
shared memory sized per launch from `max_seq_len_kv`, one bit per logical
block (1M tokens = 32 KiB, 4M = 128 KiB). The launcher opts in above 48 KiB,
so models up to roughly 6M tokens fit the SM100 limit, and pins the kernel to
the maximum shared-memory carveout so SMs that hosted a metadata CTA do not
reconfigure before hosting the 227 KiB attention CTAs.

Each route picks its granularity from its own visible causal prefix:

- byte map (membership accumulated in place, four blocks per word) up to
  32768 blocks: identical work to the static byte map for today's prefixes;
- bit map beyond that, where a second candidate pass recovers each block's
  compact rank from the prefix popcounts and ORs its membership bit into the
  packed compact entry (block index in bits 0-23, membership in bits 24-31).

Per-thread word ranges stay contiguous (ascending output) but are stored with
an odd padded stride so the clear, count, and compaction loops are free of
bank conflicts at large maps: a plain contiguous layout made the 4M-token case
69 us; the padded layout brought it to 24.8 us before the summary bits below.

### Prefix-independent bit path

The bit path's count, compaction, and rank passes initially walked every map
word of the prefix, so a 1M-token prefix cost 14.1 us and 4M 24.8 us against
the prefix-independent 11-12 us sort. Each thread now keeps one summary bit per
map word of its range, set alongside the block bit during the scatter and
stored after the map (allocated only for models past the byte-map limit), so
those passes visit only the at most `group_size * (block_topk + 1)` non-empty
words. The active map span is cleared with coalesced 16-byte stores. Result:
1M 11.2 us and 4M 13.8 us (table above). The byte path is unchanged in work.

### Loads issued before the route is resolved

Dense routes issue their selected-block (`block_indices`) loads at kernel
entry, before the route is resolved, since the rows are known from `blockIdx`.
They overlap the route's own loads instead of forming a second dependent
global round trip. Under the cold-L2 protocol this alone was worth 2.0 us at
batch 16 and 1.6 us at batch 32. Route validation likewise loads every row's
request and position at once and validates afterwards, instead of one
dependent round trip per query.

### PDL release at kernel entry

`griddepcontrol.launch_dependents` now runs at kernel entry. It only lets the
attention grid schedule; the attention grid's `griddepcontrol.wait` still
blocks until the whole metadata grid has completed and published its page
indices, membership words, and sequence lengths. The attention launch and
prologue (barrier init, TMEM allocation, descriptor prefetch) therefore overlap
the metadata work. `FLASHINFER_QSA_METADATA_PDL_RELEASE=tail` restores the
release after the final store.

### Map-versus-sort dispatch

The map is used whenever its model-sized footprint fits the device's opt-in
shared-memory limit and it needs no more waves than the sort for the launched
grid. Occupancy comes from an analytic model rather than
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` calls: the sort is assumed to
place three CTAs per SM, the map `min(4, smem_per_sm / (footprint +
reserved))`. Device attributes are cached per device, so the decision costs a
few integer operations per launch. On GB300 (152 SMs, 228 KiB per SM) the map
gets 4 CTAs per SM up to about 1.5M-token models, 2 at 2M, and 1 at 4M, so for
example 256 routes on a 4M-token model take the one-wave sort instead of a
two-wave map. `FLASHINFER_QSA_METADATA_DEBUG=1` prints the decision once per
grid.

End to end at KV 8K the hybrid map is within run-to-run noise of the static
byte map (about +0.2-0.3 us at batch 16 and +0.5-0.8 us at batch 32, against
noise of 0.3-0.7 us); about half of that is the summary bookkeeping the byte
path cannot compile out.

### Rejected metadata variants (same-binary A/B)

- Pure bit map for every prefix: +1.7 us at KV 8K (a 64-word map concentrates
  the scatter atomics and the compaction on a few threads).
- Separate membership byte array next to the bit map: +0.4 us.
- `__forceinline__` template split or lambda-shared candidate loop for the two
  granularities: +1.8 us (the prefetched candidate array spills to local
  memory).
- Staging the block-table row in shared memory with a synchronous loop:
  +1.1 us (stalls every thread one round trip before the scatter barrier);
  with `cp.async`: neutral.
- Custom warp-shuffle scan plus a full model-bound map clear: +0.4 us at
  batch 32.

## Split reducer

### Speculative slot loads

After its programmatic-dependent-launch wait, the reducer issues the stats and
partial-O loads of every configured split slot of its row at once, and only
then reads the runtime split prefix and the `cu_seqlens_q` row validity. The
workspace always holds every configured slot, so the reads are in bounds;
folding and storing keep the original predicates, and slots beyond the active
prefix are discarded at fold time. This collapses two dependent global round
trips into one.

### Single CTA for 5 to 16 splits

Plans with 5-16 split slots previously used a 2-CTA cluster with a
distributed-shared-memory merge. They now use one 512-thread CTA per 2 KiB
output slice, with `PARALLEL_REDUCTION_LOAD_BATCH = 16` so a thread pays one
round trip for all its slots. 2-4 splits keep one CTA per 8 KiB slice; 17 and
more splits keep the clustered schedule (4, 8, or 16 CTAs of 8 slots each).

### Four-lane slot split

ncu showed the reducer to be instruction-latency bound at 7.6% warp occupancy
(about 1000 dependent FP32 instructions per warp, 166 registers), not memory
bound. Four adjacent lanes now share one 16-byte output fragment, each folds an
interleaved quarter of the slots, and a two-level warp butterfly (`shfl.bfly`)
merges them before the first lane stores. This quadruples the resident warps
and shortens the serial chain: 10.3 us to 6.8 us isolated at batch 16. Eight
lanes were worse (10.2 us): the shuffle overhead doubles the instruction count.
The topology validator accepts 16 slots per CTA for this layout.

### Rejected reducer variants

Releasing the reducer grid at the attention producer's acquire instead of at
its true tail measured slower (+0.5 us at batch 16, +2.2 us at batch 32):
resident waiting reducer CTAs slow the attention body. It stays available as
`FLASHINFER_TS_REDUCER_RELEASE=acquire`.

## Step-by-step contribution

Measured on the development tree by swapping one binary at a time (KV 8K,
`group_size = 4`, same protocol as below, mean us). The deltas are those of
one K/V variant; the metadata and reducer kernels are the same for every K/V
dtype, but how much of their time is exposed depends on the attention kernel's
own duration, so the cumulative BF16 and FP8 numbers in the matrix differ
slightly from this sum.

| Step | Batch 16 | Batch 32 |
|---|---:|---:|
| Byte-map union, reducer speculative loads, single reducer CTA | -2.0 | -4.8 |
| Metadata PDL release at kernel entry | -0.6 | -0.9 |
| Reducer four-lane slot split | -3.6 | -0.7 |
| Selected-block loads before route resolution | -2.0 | -1.9 |
| Single-round-trip route validation | 0.0 | -0.2 |
| Total | -8.2 | -8.5 |

## Benchmark protocol

- GPU: GB300 (SM103), one device.
- Workload: recorded TP2-local decode routes of a sparse-attention model (12
  query heads, 1 K/V head, head dim 256, sparse block 4 tokens, token top-k
  2048 so block top-k 512, model length 131072). A route is replicated
  `batch` times; `group_size` is the number of queries per route (MTP-4 uses
  4); the KV column is the context length of the recorded route (8K, 16K,
  32K).
- Timing: `QTokenKvBlockSparsePagedTSWrapper.run` captured in a CUDA graph and
  replayed; a 256 MiB buffer is written before each replay to evict L2; 20
  warm-up and 200 timed replays; mean microseconds per replay reported. The
  figure includes the three kernels plus graph-launch and measurement overhead
  (about 5.6 us), constant across variants.
- Correctness: every timed configuration is checked against an FP32 reference
  after replay (BF16 tolerance 0.02, FP8 0.05).

## Full matrix (mean us, before and after this branch's changes)

| KV | Group | Batch | BF16 before | BF16 after | BF16 change | FP8 before | FP8 after | FP8 change |
|---:|--:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1 | 1 | 20.5 | 18.5 | -9.6% | 20.5 | 19.3 | -5.9% |
| 8192 | 1 | 8 | 22.4 | 20.4 | -9.0% | 22.5 | 20.4 | -9.5% |
| 8192 | 1 | 16 | 24.8 | 22.8 | -8.2% | 24.3 | 21.9 | -9.7% |
| 8192 | 1 | 32 | 31.1 | 29.7 | -4.6% | 28.5 | 26.7 | -6.4% |
| 8192 | 1 | 64 | 44.0 | 43.0 | -2.4% | 36.7 | 35.0 | -4.8% |
| 8192 | 1 | 256 | 106.3 | 105.0 | -1.2% | 79.5 | 78.6 | -1.1% |
| 8192 | 4 | 1 | 29.3 | 26.5 | -9.8% | 28.6 | 26.0 | -9.2% |
| 8192 | 4 | 8 | 33.0 | 29.0 | -12.1% | 34.6 | 30.3 | -12.5% |
| 8192 | 4 | 16 | 37.6 | 30.8 | -18.2% | 39.5 | 32.7 | -17.1% |
| 8192 | 4 | 32 | 46.4 | 40.0 | -13.9% | 46.7 | 40.2 | -13.8% |
| 8192 | 4 | 64 | 63.8 | 58.8 | -7.8% | 62.3 | 56.8 | -8.9% |
| 8192 | 4 | 256 | 160.9 | 150.3 | -6.6% | 129.9 | 116.9 | -10.0% |
| 16384 | 1 | 1 | 20.4 | 18.7 | -8.5% | 20.9 | 19.8 | -5.1% |
| 16384 | 1 | 8 | 22.5 | 20.3 | -9.5% | 22.4 | 20.4 | -8.8% |
| 16384 | 1 | 16 | 24.8 | 22.6 | -8.7% | 24.6 | 22.2 | -9.7% |
| 16384 | 1 | 32 | 30.8 | 29.5 | -4.2% | 28.3 | 26.5 | -6.6% |
| 16384 | 1 | 64 | 43.7 | 42.8 | -2.0% | 36.5 | 34.9 | -4.3% |
| 16384 | 1 | 256 | 106.3 | 106.2 | 0.0% | 79.4 | 78.8 | -0.8% |
| 16384 | 4 | 1 | 30.5 | 26.5 | -13.1% | 29.4 | 26.3 | -10.7% |
| 16384 | 4 | 8 | 34.6 | 28.9 | -16.6% | 35.0 | 30.6 | -12.5% |
| 16384 | 4 | 16 | 38.9 | 30.9 | -20.5% | 41.1 | 34.1 | -17.2% |
| 16384 | 4 | 32 | 48.1 | 41.6 | -13.4% | 50.4 | 42.5 | -15.5% |
| 16384 | 4 | 64 | 66.7 | 60.3 | -9.7% | 65.8 | 59.5 | -9.6% |
| 16384 | 4 | 256 | 166.5 | 153.6 | -7.7% | 134.5 | 121.0 | -10.0% |
| 32768 | 1 | 1 | 20.5 | 18.5 | -9.7% | 20.6 | 20.3 | -1.2% |
| 32768 | 1 | 8 | 22.4 | 20.5 | -8.7% | 22.5 | 20.4 | -9.2% |
| 32768 | 1 | 16 | 25.0 | 23.0 | -7.9% | 24.5 | 21.9 | -10.5% |
| 32768 | 1 | 32 | 31.3 | 29.4 | -6.2% | 28.2 | 26.6 | -5.8% |
| 32768 | 1 | 64 | 43.5 | 42.7 | -1.8% | 36.7 | 34.9 | -5.0% |
| 32768 | 1 | 256 | 106.7 | 104.9 | -1.7% | 79.4 | 78.1 | -1.6% |
| 32768 | 4 | 1 | 30.7 | 26.4 | -14.1% | 30.6 | 26.3 | -13.9% |
| 32768 | 4 | 8 | 34.9 | 30.6 | -12.4% | 36.5 | 31.0 | -15.1% |
| 32768 | 4 | 16 | 41.1 | 34.0 | -17.3% | 43.0 | 36.3 | -15.7% |
| 32768 | 4 | 32 | 51.8 | 45.3 | -12.4% | 53.1 | 46.4 | -12.6% |
| 32768 | 4 | 64 | 72.9 | 66.6 | -8.6% | 71.6 | 65.4 | -8.7% |
| 32768 | 4 | 256 | 188.8 | 174.3 | -7.7% | 148.6 | 134.4 | -9.5% |

The "before" column was measured on the development tree at the state
preceding the first commit's changes, the "after" column with them applied
(byte-map union, early loads, PDL release at entry, reducer speculative loads,
single CTA, four-lane split); the metadata and reducer code paths are the ones
on this branch. The later four commits (model-sized hybrid map, summary bits,
dispatch model) were measured separately against that state and are within
run-to-run noise at these context lengths (see "Map-versus-sort dispatch").

## Knobs

All knobs are opt-in escape hatches; the defaults are the measured-best paths.

| Variable | Default | Effect |
|---|---|---|
| `FLASHINFER_QSA_METADATA_UNION` | map when it fits and needs no extra wave | `sort` forces the shared-memory radix sort union. |
| `FLASHINFER_QSA_METADATA_PDL_RELEASE` | `entry` | `tail` releases the dependent attention grid after the final store instead of at kernel entry. |
| `FLASHINFER_QSA_METADATA_DEBUG` | unset | `1` prints the map-versus-sort decision (footprint, CTAs per SM, waves) once per grid. |
| `FLASHINFER_TS_REDUCER_RELEASE` | `tail` | `acquire` releases the reducer grid at the attention producer's acquire (measured slower). |

## Validation

On the rebased branch, GB300:

- `tests/attention/test_attention_ts_q_token_kv_block_sparse_metadata.py`:
  131 passed. The wide-context parametrizations cover 1M+1-token models
  (dynamic shared memory opt-in), 4M tokens, and 8M tokens (sort fallback),
  and compare the union, membership words, and locators against a host
  reference.
- `tests/attention/test_attention_ts_decode.py`: 226 passed, 95 skipped
  (capability and shape filters), 0 failed. The QToken decode tests compare
  the full metadata + attention + reducer chain against a dense reference.

The benchmark checks every timed configuration against an FP32 reference
after graph replay; all configurations in the matrix passed at the stated
tolerances.

## Remaining fixed cost and possible next steps

At batch 16 the GPU chain after this work is metadata 5.1 us, attention body
about 13 us (8 KV tiles at about 1.0 us each plus a 4-5 us post-acquire chain
from the sequence-length read through the first MMA, and the split-partial
epilogue), and about 4.2 us of exposed reducer. Options not pursued here:

1. Publish per-split first-tile locators in a fixed slot the attention reads
   together with the sequence lengths, or emit the first tile's TMA
   coordinates from the metadata kernel, to shorten the attention's
   post-acquire chain.
2. Fuse the union build into the attention CTAs (each split CTA maps its own
   route) to remove the metadata kernel from the chain; this requires the page
   table task to source locators from shared memory.
3. Fuse the reducer into the attention grid's tail for the last-arriving split
   CTA of each row.
