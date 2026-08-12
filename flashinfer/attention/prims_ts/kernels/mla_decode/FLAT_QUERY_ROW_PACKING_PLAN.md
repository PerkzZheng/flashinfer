# PrimTS MLA Decode Flat Query-Row Packing Port Plan

- Status: functional checkpoint implemented; exact-tree revalidation and performance signoff in progress
- Date: 2026-08-12
- Working branch: `port/pr4178-groups-tokens-heads-prims-ts`
- Upstream baseline: `065971254bca6ad0509d775e5806de53b64ac7b9`
- Reference change: FlashInfer [PR #4178](https://github.com/flashinfer-ai/flashinfer/pull/4178), PR head `78dfa5c1186aa80a66d0f1375a93c089c2775145`, squash commit `4890932d`

## 1. Objective and scope

Port PR #4178's padding-removal idea from the monolithic CuTe DSL MLA decode
implementation into both PrimTS MLA decode families:

- `throughput_latency_1cta`, with profile tile sizes M8/M16/M32/M64;
- `throughput_2cta`, with the cooperative M128 tile.

Both BF16 and FP8 task schedules, the 1CTA Swaps-MMA-AB and Keeps-MMA-AB
resource paths, fixed and packed variable-Q inputs, dense and causal masks,
direct output, and every applicable split-KV reduction path are in scope.
H6 is included with H12/H24/H48/H96 because it is a Kimi K3 TP-local head
count covered by the reference PR. Packed requests with zero query tokens,
including scheduler gaps between active requests, are also in scope.

The public tensor contract remains unchanged:

- fixed Q/O: `[B, SQ, H, D]`;
- packed Q/O: `[total_q, H, D]` with `qo_indptr`;
- query/cache input dimensions 512 latent + 64 RoPE;
- output dimension 512;
- dense and bottom-right causal masks;
- automatic public policy selection.

The port generalizes the reference M128 formula to a profile-specific physical
tile size M. Every family schedules consecutive rows in the flattened
`(query_token, query_head)` space and pays padding only in the final physical
tile. The existing whole-token helpers remain available temporarily for a
frozen old-PrimTS benchmark checkout and for unrelated kernels, but neither
updated MLA decode family may derive its launch or workspace geometry from
them.

No public kernel-family, profile, or split-KV tuning argument will be added.
Product tests and end-to-end benchmarks must assert the automatically selected
family. A private benchmark/test harness may force a 1CTA profile or split
count to isolate geometry, but those controls must not enter the public API.

### 1.1 Execution and implementation constraints

All source work, builds, tests, and benchmarks run inside the allocated B200
Docker container. Host-side commands are limited to allocation and container
launch. Follow
[`prepare-flashinfer-b200-container`](../../../../../../skills/prepare-flashinfer-b200-container/SKILL.md):

```bash
srun -p b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb \
  --gres=gpu:1 -t 240 --pty bash
../flashinfer_launch_docker.sh
```

Before each run, verify the mounted checkout/branch, B200 visibility and power
limit, recursive submodules, editable package import path, PyTorch/CUDA/CuTe
DSL versions, and writable result location. If the allocation or container
expires, preserve work on mounted storage, reacquire a GPU, relaunch the
container, repeat the checks, and resume the incomplete shard.

Touched kernel code must continue to use `cutlass.experimental`/PrimTS
interfaces and the repository's existing primitives. Do not introduce new
`cutlass.cute` (`ctm`) dependencies. Keep task/resource ownership and explicit
synchronization semantics intact while changing coordinates.

## 2. Source analysis and design delta

### 2.1 Reference behavior in PR #4178

The monolithic CuTe DSL implementation treats query token and query head as
one affine row space:

```text
flat_row  = query_token * H + query_head
total_rows = SQ * H
num_q_tiles = ceil(total_rows / 128)
tail_rows = total_rows - (num_q_tiles - 1) * 128
```

An M128 tile owns consecutive `flat_row` values and may cross one or more token
boundaries. Only the final tile is partial. Fixed Q uses an explicit batch mode;
packed Q applies the same request-local layout after reading the request's
start and length from `qo_indptr`. PR #4178 uses the same `num_q_tiles` for
launch geometry, split selection, workspace shape, producer storage, and
reducer addressing.

The reference also establishes four correctness rules that must be preserved:

1. A rectangular packed-Q slot with no rows is a no-op, not scheduler
   termination.
2. The producer partitions K using the latest causal query row in the active
   M128 tile, while softmax applies the exact endpoint for every row.
3. The reducer maps each logical output row back to
   `(q_tile, row_in_q_tile)` and ignores split slots outside that row's visible
   K prefix.
4. Public O/LSE stores are predicated on logical-row validity; padded tail rows
   may synchronize but may not form public or partial-workspace accesses.

### 2.2 Current PrimTS behavior

The 2CTA family currently computes a whole-token capacity

```text
ratio = max(1, floor(128 / H))
effective_heads = H * ratio
query_groups = ceil(SQ / ratio)
```

and schedules one M128 cluster for each query group. When `H` does not divide
128, every group asks the MMA to process rows beyond `effective_heads`.
Because the TMA descriptor is already flattened, those rows can be rows from
the next logical token and are then reloaded by the following group. They are
computed but discarded. The problem recurs at every group boundary rather
than only once at the final request tail.

Current Q loading, CLC/static work-queue activity, CTA-visible causal K bounds,
row masks, direct/partial output stores, reference reduction, and parallel
reduction all derive coordinates from `groups_tokens_heads_q_ratio`. A safe
port must change all of those consumers together.

The 1CTA family applies the same whole-token assumption through a second set of
profile-specific paths. Its launch shape uses
`ratio = max(1, floor(M / H))`, an effective head extent `H * ratio`, and a
grouped SQ extent `ceil(SQ / ratio)`. When H exceeds M, it instead tiles heads
and repeats that head-tail padding for every token. The persistent scheduler,
nonpersistent grid, Q TMA coordinates, task-visible K bounds, TMEM-S row mask,
Swaps/Keeps epilogues, partial workspace, reference reducer, parallel reducer,
and cluster-local reducer all encode that geometry. The host configuration also
currently requires an effective power-of-two head count, which prevents some
requested H12/H24/H48/H96 cases from reaching otherwise valid 1CTA physical
profiles.

The 1CTA port therefore keeps logical H and SQ separate from physical launch
geometry. For a selected tile M it schedules:

```text
total_rows  = H * SQ
num_q_tiles = ceil(total_rows / M)
valid_rows(tile) = clamp(total_rows - tile * M, 0, M)
```

Internally, the scheduler may retain its current coordinate rank by using one
physical head tile and placing all flat tiles on the Q-tile axis. It must not
pretend the public head count is M: logical H/SQ remain explicit configuration
traits, while physical M and `num_q_tiles` are separately named traits.

### 2.3 Requested geometry and expected reduction

Use the following matched-384-row shapes as the primary 2CTA acceptance set.
They make the M128 scheduling gain directly comparable across the requested
head counts.

| H | SQ | Current ratio | Current groups | Packed M128 tiles | Tile reduction | Current workspace rows per B/split | Packed workspace rows per B/split |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 4 | 1 | 4 | 3 | 25% | 384 | 384 |
| 48 | 8 | 2 | 4 | 3 | 25% | 384 | 384 |
| 24 | 16 | 5 | 4 | 3 | 25% | 480 | 384 |
| 12 | 32 | 10 | 4 | 3 | 25% | 480 | 384 |

The physical split workspace will follow PR #4178 and use
`128 * num_q_tiles` rows. This can be larger than the current compact workspace
for a short tail such as H96/SQ3, but it bounds padding to one final tile and
keeps producer/reducer geometry unambiguous. The benchmark report must include
workspace bytes so that this tradeoff is visible.

Use the following forced-profile shapes to isolate 1CTA packing. Each case
crosses an old per-token head-tile boundary and removes one of four physical
tiles:

| M | H | SQ | Old physical tiles | Flat tiles | Tile reduction | Flat tail rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12 | 2 | 4 | 3 | 25% | 8 |
| 16 | 24 | 2 | 4 | 3 | 25% | 16 |
| 32 | 48 | 2 | 4 | 3 | 25% | 32 |
| 64 | 96 | 2 | 4 | 3 | 25% | 64 |

Also isolate the H-below-M grouping case with `(M,H,SQ)` equal to
`(16,12,4)`, `(32,24,4)`, and `(64,48,4)`: each changes four old grouped
tiles to three flat tiles. Tail-specific cases use SQ1 for the four rows above,
yielding final valid-row counts 4, 8, 16, and 32 respectively. These are
internal profile tests, not promises that public auto-policy selects the named
profile for every shape.

## 3. Proposed kernel design

### 3.1 Introduce explicit flat-tile geometry

Add a host helper such as `FlatQueryTileLayout` in `helpers/query.py` with
`total_rows`, `num_tiles`, `tail_rows`, and `tile_size`. Validate positive H,
SQ, and tile size. H may be smaller than, equal to, or larger than M; crossing
a token or a former head-tile boundary is the purpose of the layout. Instantiate
the helper with the selected 1CTA profile M or with M128 for 2CTA. Keep legacy
whole-token helpers only until old/new source checkpoints and unrelated users
no longer need them.

Add device helpers with one shared definition of:

- request start and runtime Q length;
- `valid_rows = clamp(q_len * H - q_tile * M, 0, M)`;
- physical row `row_in_tile` and request-local `flat_row`;
- logical `q_idx = flat_row // H` and `head_idx = flat_row % H`;
- fixed or packed storage row;
- first and last logical Q token in an active tile.

All hot-path divisions are by compile-time H. Hoist them to one calculation per
row or tile; do not introduce division into every score element.

### 3.2 Change host policy, profile, split, and workspace geometry

In `flashinfer/attention/prims_ts/mla_decode.py`:

- form a flat layout after each 1CTA profile or 2CTA family decision;
- use `num_q_tiles`, not grouped SQ or per-token head-tile count, in work and
  split heuristics;
- retain logical H/SQ and physical M as distinct values in validation, compiled
  traits, JIT cache keys, and provenance;
- compute 1CTA work as `B * num_q_tiles * split_kv` and 2CTA work as the same
  number of two-CTA clusters;
- base nonpersistent, static-persistent, and CLC selection on physical work;
- allocate split workspace from `M * num_q_tiles` physical rows.

For `throughput_latency_1cta/config.py`:

- allow requested non-power-of-two logical H values; keep the selected physical
  M restricted to the supported power-of-two profiles;
- replace effective-head/grouped-SQ fields with explicit logical H/SQ,
  physical M, `num_q_tiles`, and `tail_rows` traits;
- choose the existing performance profile from logical workload properties,
  then calculate its flat layout; do not silently round logical H;
- calculate `num_ctas_for_all_heads = 1` and
  `num_ctas_per_seq_q = num_q_tiles` for the normalized scheduler shape;
- revisit the current `num_heads > tile_size` divisibility guard: flat tiles
  make H greater than M valid, so only real MMA/profile constraints may reject
  a candidate;
- re-evaluate cluster-reduction eligibility using tail validity and reducer
  topology, not partial per-token head tiles.

Rename the 2CTA `compute_split_kv` argument from `seq_len_q` to
`num_q_tiles`. Its occupancy denominator remains two CTAs per cluster. Change
the common family workspace calculation to:

```text
0,                           split_kv == 1 or 1CTA cluster-local reduction
B * M * num_q_tiles * split_kv *
    (512 * sizeof(partial_o) + sizeof(partial_lse)),     standalone reduction
```

Here M is the selected 1CTA tile or 128 for 2CTA. Retain BF16 partial O and
FP32 partial LSE. Add overflow validation before byte allocation and keep the
standalone workspace sizer identical to every compiled producer/reducer
layout.

### 3.3 Change the 2CTA producer scheduling and Q loading

In `throughput_2cta/kernel.py` and `resources.py`:

- store `total_q_rows`, `num_q_tiles`, and `tail_q_rows` on `MlaDecodeTs`;
- make static and CLC scheduler Q extents equal `num_q_tiles`;
- make CLC grid X `2 * num_q_tiles` and grid Z `B * split_kv`;
- replace group activity with `valid_rows > 0`;
- preserve the rule that every task advances over an inactive work item;
- for CTA rank `r`, load Q at
  `q_tile * 128 + r * 64`;
- for packed Q, add `q_start * H` to the storage coordinate and use the
  remaining request-local row count as the ragged extent;
- rely on TMA OOB fill only for the one final tail, and never issue a negative
  ragged extent for an inactive tile.

The work-throttle acquire/release path, Q/K/V pipelines, TMEM lifecycle, and
CLC response protocol must take the same active predicate. An inactive packed
tile must skip data access while all participating tasks retire the same work
sequence.

### 3.4 Change the 1CTA profile resources and scheduling

In `throughput_latency_1cta/kernel.py`, `tasks.py`, and `resources/`:

- normalize persistent work coordinates to one physical head-tile slot and
  `num_q_tiles` Q-tile slots;
- make the nonpersistent X dimension enumerate `num_q_tiles * split_kv`, and
  make persistent work queues enumerate exactly the same logical tasks;
- decode batch directly rather than folding it with a logical head-tile index;
- load the tile's Q rows from `q_tile * M`, adding `q_start * H` for packed Q;
- derive ragged TMA extents from `valid_rows`, with no access for an inactive
  rectangular packed-Q slot;
- replace grouped-Q calculations in `tasks.py` and cached task-K state with the
  shared first/last logical-row helpers;
- update both Swaps-MMA-AB and Keeps-MMA-AB resources, including SmemQ,
  TMEM-S, TMEM-correction, sliced/full epilogues, and their barrier predicates;
- preserve the current profile's MMA shape, pipeline stages, warp ownership,
  TMEM allocation, and cluster topology unless measurement demonstrates a
  separate reason to tune them.

The first implementation checkpoint should preserve the existing scheduler
coordinate rank to limit structural churn. Once correctness is established,
dead logical-head-tile fields can be removed in a cleanup change. There must be
no runtime fallback to the old grouping when H is not a power of two.

### 3.5 Rebuild causal geometry around flat rows

For each active tile, derive:

```text
first_flat = q_tile * M
last_flat  = first_flat + valid_rows - 1
first_q    = first_flat // H
last_q     = last_flat // H
```

For causal mode, the work item and split partition use the K endpoint visible
to `last_q`. The coarse unmasked/masked-tile decision uses the endpoint for
`first_q`. Within a boundary K tile, each row recovers its exact `q_idx` from
the affine flat row and applies the bottom-right causal endpoint.

This replaces every decision based on
`groups_tokens_heads_q_ratio > 1`. That predicate is incorrect for H96 in the
2CTA path because `ratio == 1` even though an M128 tile crosses token
boundaries, and it cannot represent 1CTA tiles that cross a former head-tile
boundary. Dense mode must remain free of row-causal work. Fully masked split
rows must still publish neutral partials: zero O and negative-infinity LSE.

### 3.6 Normalize direct output and split workspace stores

The epilogue uses physical row coordinates until the final destination is
known:

- split-KV partials: `[row_in_tile, split, D, q_tile, B]`, with extent M;
- fixed public output: map local flat row to `[B, q_idx, head_idx, D]`;
- packed public output: map to `(q_start * H + local_flat_row, D)`;
- LSE follows the same row mapping without D.

Apply the same mapping to 2CTA full/sliced output and to every 1CTA
Swaps/Keeps output path. Predicate every public and partial store with
`row_in_tile < valid_rows`. Tail rows may participate in barriers and TMEM
loads but may not form a GMEM address.

### 3.7 Remap all split-KV reducers

Every standalone reducer should schedule consecutive logical output rows
rather than padded per-token head groups. For the 2CTA reference reducer, with
eight logical rows per 512-thread CTA:

```text
logical_flat = reducer_cta * 8 + row_in_cta
q_tile       = logical_flat // M
row_in_tile  = logical_flat % M
```

Use a fixed-Q grid over `ceil(H * SQ / 8)` row groups per batch. Packed-Q uses
the same maximum rectangular grid and predicates `logical_flat < q_len * H`.
Read partial O/LSE from the physical tile layout and write public output by the
logical/storage row mapping.

Apply the same quotient/remainder mapping, with each reducer's existing output
rows per CTA, to the 1CTA reference reducer. The 1CTA and 2CTA parallel
reducers similarly launch one logical row per CTA cluster and use real logical
rows `B * H * SQ` in their topology models. Update 1CTA cluster-local
reduction and TMEM-correction stores to predicate the final M-row tail while
preserving required cluster synchronization.

All reducers must compute the producer tile's latest causal Q row from
`valid_rows`, then retain the existing row-specific active-split-prefix logic.
This is required when an earlier query row sees fewer K tiles than the last row
that determined the producer split partition. Reducer launch counts use
logical rows; workspace addresses use physical tiles. Those two coordinate
systems must meet only through the shared mapping helper.

## 4. File-level change map

| File | Planned change |
| --- | --- |
| `helpers/query.py` | Add the M-parameterized host layout and shared logical/physical row helpers; retain legacy grouping helpers only for unaffected users and baseline comparison. |
| `helpers/tile.py` | Derive tile activity, CTA-visible K, row-visible K, and cached task K from flat geometry. |
| `throughput_latency_1cta/config.py` | Separate logical H/SQ from physical M/tile count, relax artificial power-of-two-H constraints, and update profile work, split, and cluster-reduction eligibility. |
| `throughput_latency_1cta/kernel.py` | Normalize grids/work queues to flat Q tiles, construct physical workspace, and wire the new traits through both 1CTA resource variants and reducer launches. |
| `throughput_latency_1cta/tasks.py` | Remove grouped head/Q decoding from batch ownership and task-visible K bounds. |
| `throughput_latency_1cta/resources/smem_resources.py` | Load consecutive flat Q rows and apply packed-Q ragged extents. |
| `throughput_latency_1cta/resources/tmem_s.py` | Apply exact per-flat-row causal endpoints for Swaps and Keeps score paths. |
| `throughput_latency_1cta/resources/tmem_corr.py` | Map full/sliced/cluster-local output from physical tile rows to public logical rows. |
| `throughput_latency_1cta/reduction.py` | Read M-row physical workspace while scheduling only logical output rows. |
| `throughput_latency_1cta/parallel_reduction.py` | Apply the same mapping and recompute logical-row cluster topology. |
| `throughput_2cta/config.py` | Make split selection consume query-tile count and size workspace as M128 tiles. |
| `throughput_2cta/kernel.py` | Replace effective H/grouped SQ state, grids, workspace construction, resource wiring, and reducer launches. |
| `throughput_2cta/resources.py` | Update work queue, Q TMA coordinates, causal mask, full/sliced epilogues, and inactive-tile behavior. |
| `throughput_2cta/reduction.py` | Map logical rows to physical tile workspace and preserve row-prefix split reduction. |
| `throughput_2cta/parallel_reduction.py` | Apply the same mapping and update logical-row topology. |
| `throughput_2cta/tasks.py` | Adapt resource interfaces only where renamed flat-tile traits propagate; do not alter task ownership. |
| `kernel_policy.py` and `flashinfer/attention/prims_ts/mla_decode.py` | Use physical flat work for family/profile policy resolution, compilation, workspace sizing, and provenance without changing the public API. |
| `tests/attention/test_attention_ts_mla_decode.py` | Add geometry, workspace, accuracy, scheduler, reducer, packed-Q, and CUDA-graph coverage. |
| `benchmarks/routines/attention.py` | Allow low-head monolithic CuTe DSL MLA, assert/record the PrimTS family/profile, and record tile/workspace provenance. |
| `README.md` | Document validated H12/H24/H48/H96 shapes and flat-row behavior after qualification. |

Do not change the monolithic CuTe DSL implementation as part of the port. The
latest upstream baseline already contains PR #4178 and is the comparison
oracle.

## 5. Correctness and regression test plan

### 5.1 Host geometry and workspace tests

Add exact unit cases for:

- generic M8/M16/M32/M64/M128 full and partial tiles;
- 2CTA full tiles: H128/SQ3 and the four matched-384-row shapes;
- 2CTA tails: H96/SQ3 -> `(288, 3, 32)`, H48/SQ3 -> `(144, 2, 16)`,
  H24/SQ6 -> `(144, 2, 16)`, and H12/SQ11 -> `(132, 2, 4)`;
- the forced 1CTA table from section 2.3, including the four H-greater-than-M
  cases, three H-below-M grouping cases, and SQ1 partial tails;
- cross-token mapping, especially H96 row 127 -> `(q=1, h=31)` and row
  128 -> `(q=1, h=32)` for M128, plus M64 tile 1 rows 31/32 around the
  H96 token boundary;
- invalid nonpositive H/SQ/M and unsupported 1CTA physical tile sizes; H greater
  than M must be accepted;
- split-KV 1 returning zero workspace;
- 1CTA cluster-local reduction returning zero GMEM workspace even when split-KV
  is greater than one;
- standalone split-KV greater than 1 matching
  `B * M * num_q_tiles * split_kv` for all five physical tile sizes;
- integer-overflow rejection and equality between public sizing and compiled
  workspace bytes.

Add regression assertions that the matched 2CTA and forced-profile 1CTA shapes
schedule exactly three tiles versus four under the corresponding old formula.
Add policy/config tests proving logical H stays 12/24/48/96 while the physical
profile remains M8/M16/M32/M64/M128 and while JIT keys distinguish those
traits.

### 5.2 Requested head-count matrix on B200

Add a public auto-policy GPU accuracy product over H in `{12, 24, 48, 96}`, SQ
in `{1, 2, 4, 8, 16, 32}`, and input dtype in `{BF16, FP8}` at a representative
causal K length. Use the existing PyTorch MLA oracle and existing
dtype-specific tolerances. Check the expected family from a checked-in case
table; never accept whichever family happened to run.

The stable short-row auto-policy cases `(H,SQ)` equal to `(12,1)`, `(12,2)`,
`(12,4)`, `(24,1)`, `(24,2)`, and `(48,1)` must exercise 1CTA. Add a private
profile-level product over `(M,H,SQ)` equal to `(8,12,2)`, `(16,24,2)`,
`(32,48,2)`, and `(64,96,2)` so every 1CTA M profile and every requested H is
tested even where public auto-policy correctly prefers 2CTA. Cover both
Swaps-MMA-AB and Keeps-MMA-AB through profiles/K lengths that select them, and
assert the selected variant rather than inferring it from output.

Add focused 2CTA rows for the exact tile-saving shapes:

- H96/SQ4;
- H48/SQ8;
- H24/SQ16;
- H12/SQ32.

Run these at a batch/K point that deterministically selects 2CTA. They must
cover BF16 and FP8 and verify that provenance reports tile size 128, three
tiles, the expected split count, and exact kernel workspace bytes. Add one
direct internal 2CTA case if auto-policy tuning later changes an end-to-end
family choice.

### 5.3 Boundary, mask, and output tests

Add targeted tests for:

- H96/SQ2 causal masking, where `ratio == 1` in the old code but tile 0 crosses
  a token boundary;
- forced M64/H96/SQ2, where the second 1CTA tile crosses from token 0 to token
  1;
- dense versus causal tail markers at cross-token M16/M32/M64/M128 boundaries;
- split-KV 1 direct output with H48/SQ3;
- final tails of 4/8/16/32 rows in the 1CTA profiles and four rows with
  M128/H12/SQ11;
- a full final tile for all four matched-384-row shapes;
- output and workspace canaries proving invalid tail rows do not store;
- compact rank-3 and rank-4 paged cache layouts through existing coverage.

### 5.4 Packed variable-Q and scheduler tests

Use ragged lengths such as `[8, 0, 1, 3, 5]` with H6/H12/H24/H48/H96 in both
families. Cover a zero-length request and a request whose later rectangular
q-tile slots are inactive, followed by an active request, so the 1CTA and 2CTA
persistent work queues prove that a no-op slot does not terminate progress.
Also qualify an all-empty packed batch if the public empty-tensor contract can
be represented without introducing a fallback backend.

Exercise:

- 1CTA and 2CTA nonpersistent, static-persistent, and applicable CLC-persistent
  paths;
- BF16 and FP8 task schedules;
- 1CTA Swaps-MMA-AB and Keeps-MMA-AB resources;
- fixed K and variable runtime K lengths;
- page sizes already supported by the public API;
- eager, standalone caller-workspace, and captured CUDA-graph launches.

For CUDA-graph replay, mutate `qo_indptr`, `seq_lens`, and page tables in place
while preserving planned bounds and total packed-Q storage. Compare every
replay with the reference and eager result.

### 5.5 Reducer matrix

Cover:

- direct output (`split_kv == 1`);
- the common two-split reference reducer in both families;
- 1CTA cluster-local reduction, including an M-row tail;
- an odd non-power split prefix such as 17;
- a long-K case that selects each family's parallel reducer;
- row-specific causal prefixes within one flat tile;
- fully masked partial rows and finite final output/LSE behavior.

Run the existing entire PrimTS MLA suite, trace-template suite, and monolithic
CuTe DSL MLA suite after the focused tests. Existing H8/H16/H32/H64/H128,
page-size, dtype, mask, alias, overflow, and runtime-pruning coverage is the
regression set, not a substitute for the new low-head cases.

## 6. Throughput comparison plan

### 6.1 Harness changes and fairness rules

Use `BatchMLAPagedAttentionWrapper` in `benchmarks/flashinfer_benchmark.py` for
the public end-to-end comparison. Remove the stale benchmark-only rule that
skips CuTe DSL below 128 heads; force `--mla_cute_dsl_impl monolithic`, which
already contains PR #4178. Add a private profile-forcing microbenchmark around
the same compiled 1CTA kernel only for M8/M16/M32/M64 geometry isolation.

Maintain an immutable old-PrimTS checkout at upstream baseline
`065971254bca6ad0509d775e5806de53b64ac7b9` and the candidate checkout at the
working branch. Run both inside the same allocated B200 container with explicit
checkout identity and isolated JIT/cache directories. The old checkout is a
measurement source, never a source of copied generated artifacts. Use a fixed
random seed so separate processes regenerate identical tensors and metadata.

For every retained row:

- allocate Q, KV, page tables, lengths, output, and workspace once;
- feed identical tensors, scales, mask, page size, dtype, output dtype, and
  random seed to old PrimTS, candidate PrimTS, and CuTe DSL;
- compile and warm each source/backend before timing and exclude compilation;
- use CUDA-graph replay for the primary short-kernel measurement;
- use the same cache policy and iteration count;
- run `--refcheck` before accepting performance data;
- assert and record `throughput_latency_1cta` or `throughput_2cta` from a
  checked-in expectation for every candidate PrimTS row;
- record H, SQ, total rows, q-tile count, tail rows, split-KV, producer CTA
  count, family, 1CTA profile/variant or 2CTA cluster shape, reducer
  kind/topology, workspace bytes, persistent/CLC state, and CuTe DSL
  implementation;
- report latency, query tokens/s, query-head rows/s, TFLOP/s, and TB/s.

The primary command template is:

```bash
python3 benchmarks/flashinfer_benchmark.py \
  --routine BatchMLAPagedAttentionWrapper \
  --backends prims-ts cute-dsl \
  --batch_size B --s_qo SQ --s_kv K --num_qo_heads H \
  --head_dim_ckv 512 --head_dim_kpe 64 --page_size 64 \
  --q_dtype DTYPE --kv_dtype DTYPE \
  --mla_cute_dsl_impl monolithic --mla_is_var_seq false \
  --refcheck --dry_run_iters 20 --num_iters 100 -vv \
  --output_path RESULTS.csv
```

Alternate backend order across repeated campaigns to expose order/thermal
bias. Run an eager/non-graph diagnostic only after the primary graph result;
do not mix graph and eager numbers in one speedup. Randomize or alternate the
old/candidate source order as well. Monitor clocks, power, and thermal state;
discard a pair if throttling or unrelated GPU activity invalidates it.

### 6.2 Benchmark matrix

Use three shape cohorts.

1. **1CTA public-auto cohort:** `(H,SQ)` = `(6,8)`, `(12,4)`, `(24,2)`, and
   `(48,1)` (48 logical rows), plus `(6,1)`, `(12,1)`, and `(24,1)` tail cases. These are new
   requested-head functionality because the old 1CTA config rejects their
   non-power-of-two logical H. Report old PrimTS as unsupported where that is
   the actual result; do not convert a failure into a speedup.
2. **1CTA forced-profile diagnostic:** `(M,H,SQ)` = `(8,12,2)`, `(16,24,2)`,
   `(32,48,2)`, and `(64,96,2)`, plus the H-below-M cases in section 2.3.
   This validates each physical profile and separates flat packing from public
   family-policy changes. It is a diagnostic series, not a public API result.
3. **2CTA primary cohort:** `(H,SQ)` = `(96,4)`, `(48,8)`, `(24,16)`, and
   `(12,32)`, each with 384 logical rows and a four-to-three M128 tile change.

Add legacy controls `(H,SQ)` = `(8,8)`, `(16,4)`, `(32,2)`, `(64,1)` for
1CTA and `(128,1)` for 2CTA. The flat and old layouts have equal tile counts
on these rows, so they isolate mapping overhead and ordinary build drift.

For every public-auto and 2CTA-primary shape, use an orthogonal expanded sweep:

- batch anchor grid: B in `{1, 4, 16, 64, 256}` crossed with K in
  `{512, 2048, 8192, 32768}`;
- batch-resolution sweep: B in `{2, 8, 32, 128, 512}` at K4096;
- KV-resolution sweep: K in `{128, 256, 1024, 4096, 16384, 65536}` at B16;
- dtype: BF16 and FP8;
- page size: 64, mask: causal, output: BF16.

This is 31 unique B/K points per shape after removing overlaps. Shard it by
family, dtype, and shape so it is resumable across four-hour allocations. Run
the forced-profile diagnostics and legacy controls on the smaller diagnostic
grid B in `{1,16,128}` and K in `{512,4096,16384}`. Add dense-mask spot checks
at B16/K4096 rather than doubling the full causal campaign.

Boundary supplement at B4, K1024 and K8192, BF16:

- H96/SQ2 in both forced M64 1CTA and M128 2CTA diagnostics;
- H48/SQ3;
- H24/SQ6;
- H12/SQ11;
- H128/SQ1 as the no-layout-change control.

If automatic split-KV differs between the pre-port and post-port builds,
retain the end-to-end result because split selection is part of the adopted
geometry. Also run a focused internal fixed-split diagnostic for the affected
shape to separate packing gain from changed K parallelism. That diagnostic
must remain benchmark-only.

### 6.3 Baseline, repetition, and provenance

Capture three versioned datasets, but evaluate them in two sequential gates:

1. unmodified PrimTS at the upstream baseline;
2. packed PrimTS candidate;
3. monolithic CuTe DSL at the same source baseline/candidate (expected to be
   unchanged).

Run at least five campaigns per primary case and report the median of campaign
medians plus dispersion. Flag any reproducible regression greater than 3%.
Define speedup as `comparison_latency / candidate_latency`; values above one
favor the candidate. The report must show:

- Gate A: packed PrimTS versus old PrimTS, completed and signed off before
  CuTe DSL is treated as the optimization target;
- Gate B: packed PrimTS versus monolithic CuTe DSL after Gate A passes;
- H128/SQ1 control drift;
- family/dtype and overall geometric means plus every per-shape ratio, without
  hiding unsupported rows or regressions.

Gate A passes when the 2CTA primary cohort has speedup at least 1.00 by
family/dtype geometric mean, no primary row is reproducibly below 0.97, and
the legacy 1CTA/2CTA controls have no reproducible regression beyond 3%.
Requested non-power-of-two 1CTA rows that old PrimTS cannot launch are labeled
"unsupported in baseline" and must pass correctness; they are not included in
the old/new latency geometric mean. Where old PrimTS can launch a different
family for the same public shape, report that best valid old public result as
an additional end-to-end comparison, clearly labeled as a family change.

Gate B is the final performance target. Candidate public-auto PrimTS must meet
or beat monolithic CuTe DSL (speedup at least 1.00) on the geometric mean of
the complete public-auto primary matrix for each dtype, with no reproducible
primary row below 0.97. Report 1CTA/2CTA and profile breakdowns to localize
misses, but do not make a forced family/profile diagnostic an acceptance gate:
the automatically selected PrimTS implementation is what must match CuTe DSL.
A miss leaves the performance project open: retain a correctness-qualified
checkpoint, identify the selected family/profile/reducer gap, and continue
tuning rather than weakening correctness or silently changing the matrix.

Keep pre/post pairs on the same GPU allocation. If the four-hour Slurm job
expires, reallocate using the environment skill and rerun both members of any
incomplete pair on the new allocation. Do not compare an orphaned baseline
from one node with a candidate from another as the sole result.

Record Slurm job ID, hostname, GPU UUID and power limit, Docker image ID, Git
SHA and diff state, PyTorch/CUDA versions, exact command, warmup/repetition
counts, CUDA-graph state, and cache policy with every CSV/report.

## 7. Implementation sequence and gates

1. **Freeze baseline:** create/verify the immutable old-PrimTS checkout, run
   focused correctness, enumerate current family/profile decisions, and capture
   all primary/control old-PrimTS rows before kernel edits. Save results and
   provenance on mounted storage outside either Git worktree.
2. **Shared geometry:** add the M-parameterized host/device flat layout and
   unit tests. Gate on exact tile, tail, row mapping, overflow, and workspace
   calculations for M8/M16/M32/M64/M128.
3. **2CTA host and direct path:** change M128 policy work, split selection,
   grids, Q load, activity, causal masks, and direct output. Gate first on BF16
   split1 fixed-Q boundary cases, then FP8 and dense mode.
4. **2CTA split path:** change partial stores, reference reducer, and parallel
   reducer. Gate on split2, odd split, long-K parallel reduction, row-specific
   prefixes, and fully masked partials.
5. **1CTA host/profile path:** separate logical H/SQ from physical profile M,
   remove artificial logical-H restrictions, normalize scheduler coordinates,
   and update split/workspace/cluster-reduction selection. Gate on config and
   workspace tests without launching kernels.
6. **1CTA direct resources:** update SmemQ, tasks, TMEM-S, and output mapping
   for Swaps-MMA-AB and Keeps-MMA-AB. Qualify M8, then M16/M32/M64, first in
   BF16 split1 fixed-Q and then FP8; do not enable public auto-policy until all
   requested H values pass.
7. **1CTA reduction paths:** update partial stores, reference reduction,
   cluster-local reduction, parallel reduction, and TMEM correction. Gate on
   the same split/tail/causal matrix used for 2CTA.
8. **Packed and scheduler paths:** qualify ragged Q, inactive rectangular
   tiles, variable K, nonpersistent/static/CLC scheduling as applicable, caller
   workspace, and CUDA-graph replay in both families.
9. **Full regression:** run all PrimTS MLA, trace-template, and monolithic CuTe
   DSL MLA tests in the B200 container. Run static formatting/lint checks and
   inspect generated compile keys/provenance for logical/physical ambiguity.
10. **Performance Gate A:** rebuild isolated old/candidate environments, run
    the expanded paired campaign, investigate every >3% flag, and meet the
    old-PrimTS criteria in section 6.3.
11. **Performance Gate B:** only after Gate A passes, run candidate PrimTS
    against monolithic CuTe DSL, tune remaining family/profile/reducer gaps,
    and meet the final criteria in section 6.3.
12. **Documentation/cleanup:** publish raw CSV and a concise comparison report,
    update the MLA README, and remove superseded MLA grouping terminology/code
    only after correctness, both performance gates, and source review agree.

Do not combine producer geometry and reducer geometry into one unvalidated
step. A physical workspace mismatch can produce plausible but incorrect output
and is easier to localize when direct output passes before split reduction is
enabled.

## 8. Acceptance criteria

The port is complete when all of the following hold:

- Every 1CTA/2CTA launch, split heuristic, producer workspace, and applicable
  reducer shares `ceil(H * SQ / M)` geometry with M equal to the selected
  physical tile size.
- H96/H48/H24/H12/H6 fixed and packed cases pass BF16 and FP8 reference checks
  in the intended public family and forced-profile coverage.
- Mixed active/zero-length packed requests, scheduler gaps, and an all-empty
  packed batch (when representable by the public tensor contract) are explicit
  correctness cases rather than implicit consequences of tail predication.
- Cross-token causal boundaries, tail rows, inactive packed slots, direct
  output, reference reduction, 1CTA cluster-local reduction, both parallel
  reducers, persistent scheduling, Swaps/Keeps resources, and CUDA graphs are
  explicitly covered.
- The four matched-384-row 2CTA shapes and four forced 1CTA profile shapes each
  schedule three physical tiles, not four.
- H greater than a 1CTA tile is supported when the physical profile otherwise
  permits it; logical non-power-of-two H is never rounded in public indexing.
- H128/SQ1 and legacy 1CTA/2CTA controls remain correct and meet the Gate A
  no-regression thresholds.
- Public signatures and automatic policy selection remain unchanged.
- Workspace sizing exactly matches kernel indexing and all overflow/alias
  guards continue to pass.
- A provenance-complete report compares old PrimTS, packed PrimTS, and
  monolithic CuTe DSL across the expanded batch/K matrix.
- Gate A meets the old-PrimTS targets before Gate B is evaluated, and Gate B
  meets the final CuTe DSL parity-or-better targets.
- Any performance loss is reported by shape and investigated; correctness is
  never relaxed to recover throughput.

## 9. Risks and fallback points

- **Causal under-masking:** H96 is the critical case because old `ratio == 1`
  hides a cross-token M128 tile, while forced M64/H96 crosses a token in its
  second 1CTA tile. Keep first/last-row helpers shared by work queue, softmax,
  and reducers.
- **Logical/physical trait confusion:** keep H/SQ/M/tile-count names distinct
  in configs and JIT keys; assert all four in tests and benchmark provenance.
- **1CTA profile selection loop:** profile choice determines M and M determines
  physical work. Select from logical workload inputs first, then compute flat
  work and make any occupancy comparison explicit and deterministic.
- **H-greater-than-M assumptions:** audit every old head-tile division and
  divisibility check. The normalized one-head-tile scheduler must not leave a
  hidden per-token head offset in tasks, Q TMA, or epilogues.
- **Swaps/Keeps divergence:** implement and gate the shared mapping in both
  resource graphs; passing one variant is not evidence for the other.
- **Packed-Q inactive access:** calculate `valid_rows` before TMA or workspace
  addressing and preserve balanced task progression.
- **Reducer/producer drift:** derive both layouts from the same host helper and
  assert public workspace bytes against the compiled spec.
- **FP8 task divergence:** qualify BF16 first, then FP8's dual-softmax schedule
  with identical coordinates and separate synchronization checks.
- **Split heuristic confounding:** record split counts and use a controlled
  fixed-split diagnostic only when needed.
- **Tail-workspace growth:** report workspace bytes; do not replace the physical
  M-row workspace with an unreviewed compact ABI during this port.
- **Old 1CTA unsupported rows:** label them as functional enablement, not
  infinite speedup. Use legacy supported-head controls and valid old public
  family results for quantitative Gate A comparisons.
- **Baseline/cache contamination:** isolate old/candidate compiled caches and
  record checkout SHA plus loaded module provenance before accepting timings.
- **Allocation expiry:** preserve source/results on mounted storage and follow
  [`prepare-flashinfer-b200-container`](../../../../../../skills/prepare-flashinfer-b200-container/SKILL.md)
  to reallocate and rerun paired measurements.

The safe fallback is family-local: keep source checkpoints for old 2CTA and
1CTA implementations until each new direct-output path passes, then checkpoint
again before changing that family's reducers. Do not silently fall back at
runtime, combine old producers with new reducers, or expose a public switch;
use source-level checkpoints during development and remove superseded MLA code
only after qualification.

## 10. Execution checkpoint (2026-08-12)

The functional port described above is implemented on
`port/pr4178-groups-tokens-heads-prims-ts`. Shared M-parameterized geometry now
drives both 1CTA and 2CTA scheduling, Q loads, causal limits, direct epilogues,
physical split workspaces, and every standalone or parallel reducer. Public
policy provenance exposes logical H/SQ alongside physical M, tile count, tail
rows, producer CTAs, reducer choice, and workspace bytes. The benchmark driver
accepts fail-closed expected family/profile assertions and reports both query
tokens/s and query-head rows/s.

The requested H12/H24/H48/H96 tests are checked in for auto and forced 1CTA,
matched-row 2CTA, split reduction, BF16/FP8, fixed Q, and packed variable Q.
The clean B200 results were 144/144 PrimTS MLA tests, 274/274 monolithic CuTe
DSL MLA tests, and 981 passed plus 168 skipped trace-template checks. Ruff,
Python compilation, and `git diff --check` also passed.

A representative five-point B/K Gate A campaign over all four matched 384-row
2CTA shapes produced an old/candidate geometric-mean speedup of 1.00071. The
minimum observed row was 0.97683 and the five-repetition B64/K8192 rows were all
at least 0.99931, so no greater-than-3% regression was reproduced. The FP8
B16/K1024 anchor geometric mean was 0.99986. This qualifies the functional
checkpoint against old PrimTS for the measured cohort, but it is not the full
31-point, five-campaign matrix specified in section 6.2.

Gate B is measured but remains open. BF16 candidate versus monolithic CuTe DSL
has a 0.98407 geometric mean over the measured 16-row matrix. Throughput-heavy
B16/K4096 and B64/K8192 cohort geometric means are 1.01340 and 1.00645, while
B1/K2048 and B4/K512 are 0.97047 and 0.94745. The worst row is H48/SQ8,
B4/K512 at 0.92517. The FP8 B16/K1024 anchor geometric mean is 1.06331. Per the
original acceptance rule, the small-batch BF16 misses must be tuned and the
full repeated matrix run before the performance project is declared complete.
For H48/SQ8, a controlled split sweep showed that the current split 4 at
B4/K512 and split 16 at B1/K2048 were faster than the tested smaller 2CTA
splits; odd split 3/12 choices regressed. This rejects a simple split-count
reduction and isolates the follow-up to launch/reducer-policy tuning rather
than the flat-row mapping itself.
Raw CSVs and a formal checkpoint report are retained under
`artifacts/groups_tokens_heads_20260812` outside the source checkout.

## 11. Recovery and reducer follow-up checkpoint (2026-08-12)

The branch still points at upstream baseline `065971254bca`; the implementation
and this plan are an uncommitted working-tree patch. Before exact-tree
revalidation, the tracked diff contains 19 modified files with 1,155 insertions
and 256 deletions. The patch SHA-256 is
`a1fa816e4a7397bfa6b46bce2acd1fd94bf2329b0a88cc5686367a4c794a4717`;
this plan was previously untracked.

The post-report reducer follow-up adds a topology-derived one-row G1 reducer
for static Q128 split counts through S16 only when the eight-row reference
grid underfills one physical-SM wave, the producer supplies at least half a
wave, and the expanded row grid remains within four waves. This selects G1 for
H48/SQ8, B1/K2048, S16, but retains the reference reducer for B4/K512, S4,
where forcing G1 measured about 10.8% slower.

On the second B200 allocation, isolated H48/SQ8 B1/K2048 timing improved from
13.6544 us with the reference reducer to a five-run median of 13.4080 us with
G1 (1.0184x). The same-allocation old-PrimTS/current comparison over all four
matched-row H/SQ shapes had geometric mean 1.00150 and minimum 0.99905. A
16-row paired PrimTS/CuTe DSL campaign reported geometric mean 1.00619 and
minimum 0.97742, but isolated processes measured CuTe DSL at 12.8352 us versus
PrimTS at 13.4080 us for H48/SQ8 B1/K2048. That reproducible isolated result is
only 0.95728x by the Gate-B convention, so Gate B remains open and the paired
versus isolated discrepancy must be resolved.

After adding the G1 policy and tests, clean B200 XML results recorded 159
PrimTS MLA tests, 274 monolithic CuTe DSL MLA tests, 629 focused trace-template
tests, and the full trace registry with 981 passed plus 168 skipped. The 2CTA
parallel-reducer source was subsequently touched while probing and reverting
a vector output store. Targeted refchecked runs and Nsight capture succeeded
after that touch, but the complete suites must be rerun on the exact current
tree before committing the correctness checkpoint.

User-confirmed resume decisions:

- qualify H6 and zero-length packed-Q requests from PR #4178;
- keep DCP, TRTLLM-GEN, and FP16 outside this PrimTS port;
- gate CuTe DSL parity on public-auto PrimTS dispatch overall, while retaining
  forced 1CTA/2CTA and profile runs as diagnostics;
- update this plan and the external checkpoint report after every meaningful
  correctness or performance checkpoint so work can be resumed after context
  or allocation loss.

Immediate resume order is exact-tree validation, a committed correctness
checkpoint, H6/zero-length implementation and coverage, another documented
correctness commit, then public-auto performance tuning and the full repeated
Gate A/Gate B campaigns.

## 12. Exact-tree correctness checkpoint before H6/zero-Q (2026-08-12)

The uncommitted implementation described in sections 10 and 11 was revalidated
without changing any MLA source after the reverted 2CTA vector-store probe.
The run used NVIDIA B200 UUID
`GPU-ae9bb2c7-bbba-d0e7-ae58-972b3ebd587e` in container hostname
`2af4c59c8b38`, PyTorch `2.10.0+cu128`, CUDA runtime 12.8, compute capability
10.0, and the editable FlashInfer `0.6.18` package from this checkout.

Exact-current-tree results were:

- Python compilation, Ruff lint, Ruff format check, `git diff --check`, and the
  pure geometry/topology subset all passed (41 tests in the subset).
- `tests/attention/test_attention_ts_mla_decode.py`: 159 passed.
- `tests/attention/test_cute_dsl_mla_decode.py -k monolithic`: all 274 selected
  cases passed. The first run passed 266 CuTe DSL cases but exposed eight
  CuTe-vs-TRTLLM-GEN JIT failures caused by stale editable-install data links;
  after repairing the links, those same eight cases passed from a fresh JIT
  workspace.
- `tests/trace/test_fi_trace_template_consistency.py`: 629 passed.
- The focused trace file plus `test_template_init.py` and
  `test_template_registry.py`: 981 passed and 168 skipped.

The environment initially mounted the checkout at
`/workspace/prims_ts/flashinfer`, while the existing editable install's
generated `flashinfer/data/{csrc,include,cutlass,cccl,spdlog}` links still
targeted `/workspace`. This caused four extra trace-init skips and the eight
TRTLLM-GEN build failures above. Reinstalling the checkout editable with
`BUILD_NVEP=0` regenerated those links against the live checkout; the repaired
reruns restored the prior 274/274 and 981/168 results. This was an environment
repair only: generated `flashinfer/data` remains ignored and no tracked source
was changed by the reinstall.

JUnit artifacts and the failed-before-repair diagnostics are retained under
`artifacts/groups_tokens_heads_20260812/exact_tree_revalidation_gpu_ae9bb2c7`
outside the Git checkout. This section identifies the correctness checkpoint
that is committed before expanding the public contract to H6 and zero-length
packed requests. Gate B is still open for the isolated small-batch BF16 gap
recorded in section 11.

## 13. H6 and zero-length packed-Q implementation checkpoint (2026-08-12)

Work after committed checkpoint `4c13b424` widens the packed-query public
contract from strictly increasing to nondecreasing `qo_indptr`. Individual
requests may now have Q length zero. Packed query tensors may have
`total_q == 0`, flattened query-head extent validation accepts that empty
extent, and every public launch path returns the validated empty output before
calling a compiled GPU kernel. Planning an all-empty packed batch still
requires an explicit positive `max_seq_len_q`, because its offsets cannot
derive a positive static policy/JIT/workspace bound.

H6 uses the existing flat-row machinery without a kernel-layout fork: public
auto dispatch maps its logical rows to a physical M8 1CTA tile for the 48-row
H6/SQ8 case, and the existing M128 2CTA family covers H6/SQ64. The expanded
tests cover H6 fixed Q in BF16 and FP8 for both families; mixed packed lengths
`[8, 1, 0, 3]` for H6/H12/H24/H48/H96 in both dtypes; zero-request CLC
progression; CUDA-graph replay whose live offsets contain alternating empty
requests; and all-empty wrapper, caller-workspace, and one-shot APIs for
public-auto 1CTA and 2CTA plans.

The first B200 smoke set passed 6/6. The first 34-case expanded run passed 33
cases and exposed one over-specific test expectation: an underfilled
`B=2, H=96, SQmax=8, FP8` plan legitimately selected 1CTA through the existing
public-auto occupancy probe. Moving the all-empty family anchor to B4 retained
public-auto selection and produced the intended 2CTA policy for both dtypes;
the corrected all-empty matrix then passed 4/4.

Strengthening the packed public-path test to retain and replay a CUDA graph for
each H6/H96 dtype anchor exposed a test-lifetime hazard. Destroying the owner
objects for two otherwise valid captured graphs, then allowing their storage
to be recycled by a later parametrized case, could produce a delayed illegal
address. Each isolated capture and replay passed, mixed captured/noncaptured
sequences passed, and retaining each graph together with its wrapper, output,
and input allocations made the full 11-case public-path sequence pass. The
harness now preserves that complete owner set for the module lifetime, matching
the public CUDA-graph contract that captured pointer storage remains stable for
the graph lifetime. No failure was observed while captured graph owners were
valid.

Final exact-tree results on the same B200 were:

- Python compilation, Ruff lint/format, and `git diff --check`: passed.
- `tests/attention/test_attention_ts_mla_decode.py`: 178 passed.
- The strengthened mixed-zero public-path slice: 11 passed.
- `tests/trace/test_fi_trace_template_consistency.py`: 629 passed.
- The focused trace file plus `test_template_init.py` and
  `test_template_registry.py`: 981 passed and 168 skipped.

The first full trace invocation collected zero tests because Python resolved a
different top-level `tests` package. Rerunning with this checkout first on
`PYTHONPATH` restored the expected registry and passed; this was an invocation
repair, not a source change. JUnit files and the graph-lifetime diagnostics are
retained under
`artifacts/groups_tokens_heads_20260812/h6_zero_q_gpu_ae9bb2c7`. These results
qualify the H6/zero-length functionality checkpoint for commit on top of
`4c13b424`; public-auto performance Gate B remains open.

## 14. Gate-B isolated reproduction before further tuning (2026-08-12)

Functionality checkpoint `91cc73df` was measured in fresh Python processes on
B200 UUID `GPU-ae9bb2c7-bbba-d0e7-ae58-972b3ebd587e`. Five alternating-order,
CUDA-graph, CUDA-event pairs reproduced the public-auto H48/SQ8, B1/K2048 BF16
gap. PrimTS selected `throughput_2cta`, split 16, 96 producer CTAs, and the G1
parallel standalone reducer. Its five campaign medians were 13.6240, 13.5536,
13.6480, 13.5552, and 13.6272 us. Monolithic CuTe DSL measured 13.0400 us in
all five processes. The median-of-medians result is therefore 13.6240 versus
13.0400 us, a 0.5840 us delta and 0.95713447x Gate-B speedup.

This resolves the earlier paired-versus-isolated discrepancy in favor of the
isolated evidence: the reproducible row is below both the 1.00 parity target
and the 0.97 per-row guard. No topology change was made for this measurement.
Raw CSVs, separate JIT caches, the exact runner, source hashes, and environment
provenance are under
`artifacts/groups_tokens_heads_20260812/gate_b_resume_gpu_ae9bb2c7_91cc73df`.
The next diagnostic separates reference/parallel reducer cost from producer
and launch cost before any policy or kernel edit.

## 15. One-wave 2CTA launch/reducer tuning checkpoint (2026-08-12)

The isolated miss in section 14 was traced to redundant live-metadata work in
the nonpersistent producer prologue and to serialized producer/reducer launch
ordering, rather than to the G1 reducer topology. For the same H48/SQ8,
B1/K2048 row, forcing the reference reducer measured 13.8592 us and forcing
public auto to 1CTA measured 14.8864 us. A compile-time static-K diagnostic
that bypassed the live-Q/K producer precheck measured 13.2448 us and identified
the actionable producer overhead.

The current uncommitted tuning patch therefore:

- lets nonpersistent work items derive their live Q/K/split domain once in the
  task schedule and skip a zero-domain item there, instead of repeating that
  metadata calculation in a CTA-wide producer prologue;
- caches the raw graph-live request K length in the 2CTA work tile while using
  the causally adjusted endpoint only for split geometry, removing repeated
  `cache_seqs` loads in TMEM-S;
- uses PR #4178's division-free flat-row causal predicates for the tile
  boundary and per-score mask; and
- enables PDL only for a one-wave, nonpersistent 2CTA producer with a separate
  reducer. The producer signals dependents only after its CTA-pair TMEM
  deallocation, and the reducer waits before entering its body. Persistent
  multi-wave producers and direct-output launches retain ordinary stream
  ordering.

Nsight Compute showed the producer's global-load count fall from 2,544 at the
section-14 baseline to 1,392 after the task-owned zero-domain change and to
1,008 after caching raw K (monolithic CuTe DSL: 1,056). Producer duration in
the same capture sequence fell from 21,088 ns to 20,640 ns and then 20,512 ns.
End-to-end latency before PDL stabilized at 13.2448 us. A CTA-wide shared-memory
metadata cache (13.6448--13.6512 us) and a reducer warp-to-shared row-state
handoff (13.4528 us) were rejected and fully removed.

Signaling PDL after TMEM deallocation reduced the target row to 12.6304 us.
Five fresh alternating isolated pairs all measured PrimTS at 12.6304 us and
CuTe DSL at 13.0400 us, a 1.03242969x speedup. Restricting PDL to the intended
nonpersistent one-wave path preserved exactly the same 12.6304/13.0400 us
result. A focused packed-Q/non-power-head/live-metadata set passed all 12 cases,
including H6/H12/H24/H48/H96, BF16/FP8, mixed zero-length requests, odd split
17, and persistent CUDA-graph metadata replay.

The first representative public-auto spot matrix is also favorable:

| Dtype | H/SQ | B/K | PrimTS us | CuTe DSL us | Speedup | Auto policy |
| --- | --- | --- | ---: | ---: | ---: | --- |
| BF16 | 48/8 | 1/2048 | 12.6304 | 13.0400 | 1.03243 | 2CTA, S16, G1, one wave |
| BF16 | 12/32 | 4/512 | 12.6336 | 12.8352 | 1.01596 | 2CTA, S4, reference, one wave |
| BF16 | 24/16 | 4/512 | 12.6304 | 12.8352 | 1.01621 | 2CTA, S4, reference, one wave |
| BF16 | 48/8 | 4/512 | 12.6416 | 12.8352 | 1.01531 | 2CTA, S4, reference, one wave |
| BF16 | 96/4 | 4/512 | 12.6336 | 12.8352 | 1.01596 | 2CTA, S4, reference, one wave |
| BF16 | 48/8 | 16/4096 | 58.2944 | 58.9056 | 1.01048 | 2CTA, direct, one wave |
| FP8 | 48/8 | 16/1024 | 14.4768 | 15.0880 | 1.04222 | 2CTA, direct, one wave |
| BF16 | 48/8 | 64/8192 | 391.2832 | 391.1616 | 0.99969 | 2CTA, direct, CLC persistent |

Every spot row used CUDA-graph replay, CUDA events, 20 warmups, 100 timed
iterations, seed 42, and `--refcheck`. The single persistent result is within
timing noise. Five additional 20/100 campaigns gave PrimTS medians of
392.8592, 392.6960, 392.3536, 391.4752, and 394.1344 us and CuTe DSL medians
of 389.8768, 392.7504, 392.4960, 389.9824, and 387.5200 us. The independent
median-of-medians comparison is 392.6960 versus 389.9824 us, or 0.99308982x:
above the 0.97 row guard but slightly below strict row parity. A sustained
100/1000 diagnostic was discarded as a primary gate because both backends
heated substantially and per-run standard deviation rose to 11--20 us.

A same-GPU, source-isolated A/B against clean functionality commit `91cc73df`
shows that the tuning patch does not cause the persistent result. Across five
alternating 20/100 campaigns, clean `91cc73df` had a 399.2608 us
median-of-medians and the candidate had 396.9008 us, a 1.00594506x candidate
speedup. The exact tuning tree also passed all 178 PrimTS MLA tests in 618.50 s,
in addition to the 12-case focused run; Python compilation, Ruff lint/format,
and `git diff --check` passed.

Gate B remains open until the complete public-auto matrix and campaign
repetitions meet section 6.3 and Gate A is refreshed against old PrimTS for the
final patch. Raw CSV, XML, NCU, and Nsight Systems artifacts are in the
section-14 artifact directory.
