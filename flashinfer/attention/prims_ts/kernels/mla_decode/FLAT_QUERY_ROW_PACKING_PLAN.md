# PrimTS MLA Decode Flat Query-Row Packing Port Plan

- Status: exact-tree validation complete; formal public-auto performance signoff in progress
- Date: 2026-08-13
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

Gate B is the final performance target. Candidate public-auto PrimTS must stay
within 6% of monolithic CuTe DSL (speedup at least 0.94) on the geometric mean
of the complete public-auto primary matrix for each dtype, with no reproducible
primary row below 0.94. Exact ratios remain visible, and parity-or-better is
still the optimization target rather than an acceptance requirement. Report
1CTA/2CTA and profile breakdowns to localize misses, but do not make a forced
family/profile diagnostic an acceptance gate: the automatically selected
PrimTS implementation is what is evaluated against CuTe DSL. A miss leaves the
performance project open: retain a correctness-qualified checkpoint, identify
the selected family/profile/reducer gap, and continue tuning rather than
weakening correctness or silently changing the matrix. The requester expanded
the tolerance from 5% to 6% on 2026-08-14; this `0.94` floor supersedes the
earlier `0.95` floor, while Gate A's old-PrimTS regression threshold remains
unchanged.

Keep pre/post pairs on the same GPU allocation. If the four-hour Slurm job
expires, reallocate using the environment skill and rerun both members of any
incomplete pair on the new allocation. Do not compare an orphaned baseline
from one node with a candidate from another as the sole result.

Record Slurm job ID, hostname, GPU UUID and power limit, Docker image ID, Git
SHA and diff state, PyTorch/CUDA versions, exact command, warmup/repetition
counts, CUDA-graph state, and cache policy with every CSV/report.

### 6.4 Final issue-#4390 three-backend comparison

After Gate B is correctness-qualified and its primary campaign is complete,
run a final public-path comparison modeled on
[issue #4390 comment 5239925531](https://github.com/flashinfer-ai/flashinfer/issues/4390#issuecomment-5239925531).
The comparison adds public-auto PrimTS to the comment's TRTLLM-GEN and
monolithic CuTe DSL columns; it must not substitute a forced PrimTS family or
profile.

Match the comment's B200, batch-1, TP8/H12 workload:

- query length in `{1, 8}`;
- context length in `{131072, 500000, 1000000}`;
- KV/query dtype in `{BF16, FP8 E4M3}` with BF16 output;
- one MQA KV head, latent rank 512, RoPE dimension 64, QK dimension 576,
  QK-NOPE dimension 128, and page size 64;
- `bmm1_scale = 1 / sqrt(192)` and `bmm2_scale = 1.0`;
- backends `trtllm-gen`, monolithic `cute-dsl`, and public-auto `prims-ts`;
- TRTLLM-GEN PDL enabled with the required multi-CTA counter buffer; and
- separate eager and CUDA-graph timings for every row.

Use identical seeded tensors and metadata for all three backends, run the
same correctness/reference check before retaining a timing, compile and warm
all runners before measurement, alternate backend order across repetitions,
and keep all three members of a row on one GPU allocation. Mirror the
comment's eight warmups and 30 measured calls for a direct reproduction, then
run the normal campaign repetition policy from section 6.3 so the final table
reports median-of-campaign-medians and dispersion rather than one sample.

Report microseconds per call for all six backend/mode columns, PrimTS family,
profile, split-KV, reducer, and CUDA-graph compatibility, plus ratios of
PrimTS to CuTe DSL, TRTLLM-GEN, and the faster non-PrimTS backend. This final
comparison is a required deliverable and crossover diagnostic. Gate B's
public-auto PrimTS-versus-CuTe-DSL threshold remains the performance
acceptance rule; TRTLLM-GEN is reported as an additional production baseline
unless acceptance scope is explicitly expanded.

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
12. **Final three-backend comparison:** reproduce the issue-#4390 B200
    eager/CUDA-graph matrix with TRTLLM-GEN, monolithic CuTe DSL, and
    public-auto PrimTS as specified in section 6.4. Publish matched raw rows,
    ratios, dispatch metadata, and provenance.
13. **Documentation/cleanup:** publish raw CSV and a concise comparison report,
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
  keeps public-auto PrimTS within 6% of CuTe DSL for every reproducible row and
  each dtype-wide geometric mean.
- The final issue-#4390-shaped H12 comparison reports TRTLLM-GEN, monolithic
  CuTe DSL, and public-auto PrimTS in both eager and CUDA-graph modes for
  Q1/Q8, BF16/FP8, and all three requested context lengths.
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

## 16. Public-auto 1CTA full-row promotion checkpoint (2026-08-13)

The first complete public-auto 1CTA spot matrix on `0cfd4866` exposed a
schedule-level performance problem rather than a flat-row correctness issue.
For the 48-row H6/SQ8, H12/SQ4, and H24/SQ2 products, the latency-oriented
M8/M16 profiles scanned long K six or three times. At B16/K4096 their BF16
speedups versus monolithic CuTe DSL were 0.3307x, 0.6482x, and 0.6510x; their
FP8 speedups were 0.4712x, 0.7801x, and 0.7804x. H24/SQ1 similarly used two
M16 scans and measured 0.8207x BF16 at B16 and 0.8024x at B64. One-query-tile
profiles and the B1/B4 latency points were otherwise broadly healthy.

Forced-profile diagnostics showed that long-K work should collapse the flat
row product to one M64 keeps-MMA-AB tile. For H6/SQ8 at B16/K4096, M64 reduced
BF16 latency from 80.9344 us to about 27.37 us, and selecting split 8 instead
of the ordinary split 9 reduced it again to about 26.3 us. FP8 followed the
same crossover. At the 24-row H24/SQ1 boundary, M64 split 8 measured
24.7168 us BF16 and 16.3200 us FP8, while M32 split 8 measured 25.9424 us and
16.9312 us. M64 is therefore the measured throughput tile for 17--64 flat
rows even when M32 would be the smallest covering tile.

The automatic 1CTA policy now promotes a non-power-of-two logical-head launch
with at most 64 flat rows only when the projected split-KV/V decomposition
fills at least five sixths of a resident 1CTA wave. This retains M8/M16 for
short underfilled latency work and preserves every legacy power-of-two-head
tile choice. Promoted 17--64-row work uses one M64 tile. If its ordinary
high-M split count is not a power of two, the public planner rounds down only
when the rounded producer grid still meets the same five-sixths occupancy
guard. The adjustment is applied as an explicit split only to the promoted
non-power-of-two launch; a host contract confirms that legacy H64 continues
to choose its established automatic split 9.

On replacement B200 UUID `GPU-3a152337-616f-84c8-e9f2-5f7ed45a6c56`, the
five shape-resolution cases and six split-policy cases passed. GPU correctness
then passed all ten focused cases: the two BF16/FP8 H6/SQ8 B16/K4097 promoted
launches plus the existing eight-case short-K H6/H12/H24/H48 product. The
promoted cases selected one M64 tile, split 8, 128 producer CTAs, and the
parallel standalone reducer; all short cases retained their previous tiles.

A fresh 24-pair target gate used public-auto PrimTS versus monolithic CuTe DSL,
CUDA graphs/events, 20 warmups, 100 iterations, seed 42, alternating process
order, and `--refcheck` for every row. It covered all promoted 48-row shapes
and H24/SQ1 at B16/K4096 and B64/K8192, plus H6/SQ8 B1/K2048 and B4/K512
latency controls, in BF16 and FP8. Results were:

- BF16 geometric-mean speedup 1.053815x. The initial minimum was 0.967192x at
  H24/SQ1 B64/K8192, while the next minimum was 0.9707x at H48/SQ1.
- FP8 geometric-mean speedup 1.064841x and minimum 0.991069x.
- The short BF16 controls retained 1.2577x and 1.2769x speedups; FP8 retained
  1.2210x and 1.2646x.

Because long B64 timings varied materially between fresh CuTe DSL processes,
the two borderline BF16 rows were repeated five times with alternating order.
H24/SQ1 had a 130.2816 us PrimTS median and 130.9808 us CuTe DSL median, with
a 1.005754x median paired speedup. H48/SQ1 had 134.8384 us versus 133.7952 us,
or 0.992263x. The repeat medians clear the 0.97 guard and identify the single
126.7904 us CuTe DSL sample in the target matrix as a non-representative
outlier.

Raw target and repeat CSVs, logs, scripts, summaries, and environment
provenance are retained under
`artifacts/groups_tokens_heads_20260812/gate_b_public_auto_1cta_policy_20260813`.
This is the documented 1CTA policy checkpoint on top of `0cfd4866`. The full
public-auto matrix, complete exact-tree correctness suite, refreshed Gate A,
and formal multi-campaign section-6 signoff remain outstanding.

## 17. Exact `738ea7d3` validation and complete 1CTA spot matrix (2026-08-13)

The full-row promotion in section 16 is committed as `738ea7d3`
(`perf(mla): promote filled flat query rows`). The source worktree was clean
for the complete validation and expanded performance runs. They used NVIDIA
B200 UUID `GPU-3a152337-616f-84c8-e9f2-5f7ed45a6c56`, container hostname
`c27f46ee8010`, PyTorch `2.10.0+cu128`, CUDA runtime 12.8, and compute
capability 10.0.

Exact-source correctness and regression results were:

- `tests/attention/test_attention_ts_mla_decode.py`: 191/191 passed in
  622.32 seconds. This is the prior 178-case suite plus 13 host/GPU contracts
  for full-row promotion, split rounding, and legacy split preservation.
- `tests/attention/test_cute_dsl_mla_decode.py -k monolithic`: all 274
  selected cases passed; 404 non-monolithic cases were deselected.
- `tests/trace/test_fi_trace_template_consistency.py`: 629/629 passed.
- Trace consistency plus template init and registry: 981 passed and 168
  skipped, matching the established registry result.
- Ruff format/lint, Python compilation, and `git diff --check` passed before
  the `738ea7d3` checkpoint commit.

JUnit XML is retained under
`artifacts/groups_tokens_heads_20260812/full_validation_738ea7d3_gpu_3a152`.
No numerical, packed-Q, H6, zero-length request, reducer, public-path,
CUDA-graph, runtime-pruning, 1CTA, or 2CTA regression was observed.

The exact-source public-auto 1CTA spot campaign was expanded to all seven
section-6.2 shapes `(6,8)`, `(12,4)`, `(24,2)`, `(48,1)`, `(6,1)`, `(12,1)`,
and `(24,1)` at B/K points `1/2048`, `4/512`, `16/4096`, and `64/8192`, in
both BF16 and FP8. All 56 PrimTS/CuTe DSL pairs passed `--refcheck`, and every
PrimTS row satisfied the fail-closed `throughput_latency_1cta` dispatch
assertion. The aggregate results were:

| Dtype/cohort | Rows | Geomean speedup | Raw minimum |
| --- | ---: | ---: | ---: |
| BF16 overall | 28 | 1.133396 | 0.967192 |
| BF16 B1/K2048 | 7 | 1.241395 | 1.020199 |
| BF16 B4/K512 | 7 | 1.280066 | 1.238608 |
| BF16 B16/K4096 | 7 | 1.000202 | 0.975326 |
| BF16 B64/K8192 | 7 | 1.038235 | 0.967192 |
| FP8 overall | 28 | 1.130757 | 0.991069 |
| FP8 B1/K2048 | 7 | 1.174970 | 1.123177 |
| FP8 B4/K512 | 7 | 1.238905 | 1.162979 |
| FP8 B16/K4096 | 7 | 1.045285 | 1.011534 |
| FP8 B64/K8192 | 7 | 1.074427 | 0.991069 |

The raw BF16 minimum is the H24/SQ1 B64/K8192 outlier already repeated in
section 16. Its five-pair median speedup is 1.005754x. The other long-B
borderline row, H48/SQ1 B64/K8192, has a five-pair median speedup of 0.992263x.
Those repeat medians, rather than the isolated fast CuTe DSL samples, clear the
0.97 row guard. The only unrepeated BF16 row below parity by more than noise is
H12/SQ1 B16/K4096 at 0.975326x, still above the guard; the seven-row B16
geometric mean is 1.000202x.

The complete per-row table and resumable exact-source runner are in
`gate_b_public_auto_1cta_policy_20260813/FULL_SUMMARY.txt` and
`run_full_spot_matrix.sh`. This closes the four-point public-auto 1CTA spot
gate with positive dtype-wide geometric means and no repeat-confirmed row
below 0.97. It does not replace the larger 31-point/five-campaign formal matrix
in section 6.2. Refreshed old-PrimTS Gate A for the final source and the broader
multi-campaign signoff remain outstanding.

## 18. Refreshed old-PrimTS Gate A (2026-08-13)

The final kernel source (`738ea7d3`, with documentation commit `234889ed` on
top) was compared against a freshly created detached worktree at the immutable
upstream baseline `065971254bca6ad0509d775e5806de53b64ac7b9`. The campaign
covered all four 384-row shapes H12/SQ32, H24/SQ16, H48/SQ8, and H96/SQ4 at
B/K points `1/2048`, `4/512`, `16/1024`, `16/4096`, and `64/8192`, in BF16
and FP8. Every one of the 40 source pairs ran in fresh processes with separate
JIT caches, alternating source order, seed 42, CUDA graph/event timing, 20
warmups, 100 iterations, and `--refcheck`.

An initial campaign was rejected before analysis because installing the old
checkout editable changed the global package mapping. Invoking a benchmark
script by path placed its `benchmarks/` directory, but not its repository root,
ahead of that mapping; both nominal source sides consequently imported old
PrimTS. The invalid directory is retained with a prominent warning. The valid
`_v2` campaign explicitly set `PYTHONPATH` to the intended worktree in every
subprocess, and independent probes verified both `flashinfer.__file__` and
PrimTS `mla_decode.__file__` under their exact source roots. The main checkout
was restored as the editable package after the comparison.

The corrected public-auto results were:

| Dtype/cohort | Rows | Geomean candidate speedup | Minimum |
| --- | ---: | ---: | ---: |
| BF16 overall | 20 | 1.107425 | 0.999380 |
| BF16 B1/K2048 | 4 | 1.101244 | 1.097215 |
| BF16 B4/K512 | 4 | 1.081139 | 1.081013 |
| BF16 B16/K1024 | 4 | 0.999845 | 0.999380 |
| BF16 B16/K4096 | 4 | 1.104019 | 1.046083 |
| BF16 B64/K8192 | 4 | 1.267353 | 1.231896 |
| FP8 overall | 20 | 1.162874 | 1.000224 |
| FP8 B1/K2048 | 4 | 1.204395 | 1.190745 |
| FP8 B4/K512 | 4 | 1.267862 | 1.267158 |
| FP8 B16/K1024 | 4 | 1.010905 | 1.000224 |
| FP8 B16/K4096 | 4 | 1.082120 | 1.075971 |
| FP8 B64/K8192 | 4 | 1.273022 | 1.260036 |

BF16 public auto selected 2CTA throughout. FP8 public auto selected 1CTA at
B1/B4 and 2CTA at B16/B64; this family switch is part of the accepted public
policy and improved the small-batch rows by roughly 19--27% over old PrimTS.
The 2CTA flat geometry reduced the old four padded M128 query tiles to three
physical tiles. The largest benefits appeared at B64, where eliminating that
fourth tile improved BF16 and FP8 cohort geometric means by 26.7% and 27.3%.
The no-layout-change B16/K1024 controls remained at parity, and no row was more
than 0.062% below old PrimTS.

Complete corrected CSVs, logs, the exact table, import-isolated runner, and
provenance are under
`artifacts/groups_tokens_heads_20260812/gate_a_refresh_234889ed_v2`. The
sibling directory without `_v2` is invalid by construction and must not be
used. This closes the refreshed five-point Gate-A checkpoint with no measured
old-PrimTS regression. The full 31-point and five-campaign formal expansion in
section 6.2 remains resumable future qualification rather than a completed
claim.

## 19. Formal Gate-B multi-wave scheduler recovery (2026-08-13)

The first section-6.2 formal 1CTA shard caught a pre-existing persistent-work
completion bug at H6/SQ8, B256/K512 BF16. Public auto selected the promoted
M64 Keeps-MMA-AB profile with one physical query tile per batch. The launch
reported CUDA-graph/eager mismatch, and an exact comparison with monolithic
CuTe DSL localized wrong output to batches 148--255. B200 admitted 148 of
these 1CTA clusters in the resident wave, so the failure boundary showed that
only the first wave completed. H64/SQ1 and forced static-persistent M64
diagnostics reproduced the same boundary, proving that this was not caused by
H6 or the new flat-row mapping.

Two independent scheduler issues were separated:

- the general 1CTA CLC scheduler task initialized and fetched only once
  because its body was not enclosed in `work_tile_loop`; and
- the M64 Keeps-MMA-AB task graph still does not reliably retire a second
  persistent work tile even after the scheduler loop is repaired.

Commit `319dfd05` (`fix(mla): complete multiwave 1CTA scheduling`) wraps the
CLC scheduler in the standard persistent work-tile loop. Runtime-empty packed
Q tiles skip only the throttle-token body; queue fetch, wait, advance, and
release remain unconditionally outside the skip guard so all tasks progress in
lockstep. Public automatic M64 Keeps-MMA-AB enumeration now omits the
unqualified persistent candidate whenever work exceeds the resident capacity
and uses the complete direct grid instead. Smaller M8/M16/M32 CLC profiles
retain dynamic persistence and use the repaired loop.

Focused B200 qualification passed four scheduler-sensitive tests, including
the existing runtime-skipped packed-Q CLC case and two new public-path
regressions. B160 H6/SQ8/K129 exercised M64 work beyond the resident wave and
passed eager, standalone-workspace, CUDA-graph, and sampled FP32 reference
checks with a 160-CTA direct grid. B320 H8/SQ1/K129 exercised M8 CLC beyond two
resident waves and passed the same paths. A direct exact comparison for the
latter produced zero mismatches after the loop repair.

The original failing public benchmark then passed `--refcheck` and CUDA-graph
replay at B256/H6/SQ8/K512. It selected profile `h64_keeps_mma_ab` with 256
producer CTAs and persistence disabled, measuring 46.0224 us versus
45.1024 us for monolithic CuTe DSL, or 0.98000972x. This clears the 0.97
reproducible-row guard but remains a row to repeat. The repaired persistent
path also passed H6/SQ1, B512/K4096 and measured 449.2192 us versus
535.3840 us, or 1.19181011x.

The interrupted formal directory
`gate_b_formal_1cta_d8129703` is not an acceptance dataset: it mixes the
pre-fix source with failure diagnostics. Its runner now requires an explicit
`.ok` marker in addition to a two-line CSV, preventing a refcheck failure's
header-only output from being mistaken for a completed row. Formal Gate B
must restart from a clean documentation checkpoint containing `319dfd05`,
rerun all rows on that exact source, and retain the five-campaign repetition
rule from section 6.3.

## 20. Public-auto short-K direct-2CTA crossover (2026-08-13)

The clean `4188a8c6` restart completed its first 31-point H6/SQ8 BF16 shard
with all 62 backend subprocesses refchecked and marked successful. Its
geometric-mean speedup was 1.091554x, but two reproducible rows violated the
0.97 guard. At B64/K512, the M64 split-2 1CTA path measured 17.9648 us versus
16.9408 us, or 0.94299962x across five alternating-order repeats. At
B64/K2048 it measured 42.5520 us versus 38.6192 us, or 0.90757663x. Both used
128 producer CTAs plus the standalone parallel reducer.

Profile isolation ruled out split-1 and smaller 1CTA tiles: they removed the
reducer but rescanned K and were slower. A direct-output 2CTA launch used the
same 128 producer CTAs as one full resident cluster wave and measured
16.3296 us at K512 and 37.9648 us at K2048. Monolithic CuTe DSL speedups became
1.03742893x and 1.01723704x. FP8 improved from 14.4832 and 25.1264 us on the
split 1CTA path to 11.6128 and 22.0672 us on direct 2CTA, producing
1.08845409x and 0.98600639x speedups.

Commit `66024fbf` (`perf(mla): route short flat rows to direct 2CTA`) adds a
bounded automatic family probe. It selects 2CTA only when all of the following
are true:

- the logical head count is non-power-of-two and the launch has 48--64 flat
  query rows in one M64 tile;
- the projected M64 1CTA profile needs split-KV while M128 2CTA has a
  direct-output split-1 wave; and
- the planned K extent per 1CTA split is at most 1024 tokens.

The rule preserves smaller partial tails, direct 1CTA cases, longer local K,
2CTA split cases, and legacy power-of-two heads. Six host boundary contracts
cover those decisions. Public B200 accuracy then passed all eight combinations
of H6/SQ8, H12/SQ4, H24/SQ2, and H48/SQ1 with BF16/FP8 at the short-K
crossover. Every case selected 2CTA split 1 with zero kernel workspace and
passed the existing reference path.

A matched H12/H24/H48 B64, K512/K2048 matrix gave a 1.017958x BF16 geometric
mean and a 1.036133x FP8 geometric mean. Eleven of twelve raw rows cleared the
guard. The isolated H48/SQ1 BF16 K2048 sample was 0.968150x, so it was repeated
five times with alternating order. Every repeat was at least 0.970680x and the
median comparison was 37.7152 versus 36.7456 us, or 0.97429155x. A diagnostic
H24/SQ1 tail did not meet the guard under 2CTA, confirming why the 48-row lower
bound is required.

The `gate_b_formal_1cta_4188a8c6` shard is retained as the exact evidence that
found this policy miss, but it is superseded for acceptance by `66024fbf`.
Formal public-auto qualification must restart from a clean documentation
checkpoint on top of that commit. Its dispatch assertion must accept the
checked 2CTA crossover at B64/K512 and B64/K2048 for the four 48-row shapes,
while continuing to require 1CTA for the other points in this cohort.

## 21. Formal public-auto Gate-B restart and final-comparison addition (2026-08-13)

Formal qualification restarted from clean commit `6e2338db`, which contains
the short-K family crossover and its documentation. The resumable campaign is
under `gate_b_formal_public_auto_6e2338db`. Each backend row runs in a fresh
process with a backend-specific JIT cache, CUDA-graph/event timing, 20 warmups,
100 iterations, seed 42, `--refcheck`, and an explicit `.ok` marker. Backend
order alternates within the 31-point B/K matrix. Dispatch assertions require
direct 2CTA only for 48 total rows at B64/K512 and B64/K2048; all other rows
in the seven-shape cohort must select throughput-latency 1CTA.

The first H6/SQ8 coverage pass is complete in both dtypes:

| Dtype | Matched pairs | Geomean speedup vs CuTe DSL | Raw minimum | Worst row |
| --- | ---: | ---: | ---: | --- |
| BF16 | 31 | 1.096186 | 0.973400 | B16/K8192 |
| FP8 E4M3 | 31 | 1.137953 | 0.975843 | B1/K32768 |

All 124 backend rows passed reference checking and wrote completion markers.
BF16 direct-2CTA crossover speedups were 1.0376x at B64/K512 and 1.0122x at
B64/K2048. FP8 crossover speedups were 1.0882x and 0.9823x. Public auto chose
the asserted family on every row, each dtype geometric mean is above parity,
and neither raw minimum falls below the 0.97 guard. These are first-pass
coverage results, not the final five-campaign signoff required by section 6.3.

Cases 0--11 of the BF16 shard ran on B200 UUID
`GPU-3a152337-616f-84c8-e9f2-5f7ed45a6c56`. That allocation expired before
case 12 initialized CUDA, so the header-only failed attempt is excluded.
Case 12 onward and the complete FP8 shard ran as matched pairs on replacement
B200 UUID `GPU-3f241fbe-8ae5-204d-35a7-8f613c2a22f0`. The runner now resolves
the single currently visible GPU with `nvidia-smi`, optionally validates an
explicit UUID, and never resumes a row without its `.ok` marker.

The requested terminal deliverable now includes the exact issue-#4390-shaped
three-backend comparison in section 6.4. It will report TRTLLM-GEN,
monolithic CuTe DSL, and public-auto PrimTS for H12, Q1/Q8, BF16/FP8, and
131072/500000/1000000-token contexts in both eager and CUDA-graph modes.

## 22. Equal-work BF16 crossover and logical-row 2CTA reducer (2026-08-13)

The next formal BF16 shard, H12/SQ4, completed all 31 matched rows with a
1.105557x geometric mean. Its raw minimum was 0.966298x at B16/K8192: public
auto selected M64 1CTA split 8 with the parallel reducer and measured 42.5376
us versus 41.1040 us for monolithic CuTe DSL. Five independent repeats were
0.966150x, 0.966446x, 0.966664x, 0.966516x, and 0.966260x, proving that the row
was stable rather than an isolated timing sample. It passes the subsequently
clarified 0.95 Gate-B threshold, but it also exposed a useful family crossover.

At this geometry, M128 2CTA split 4 has the same 128 producer CTAs as M64 1CTA
split 8. Forced-family diagnostics across H6/SQ8, H12/SQ4, H24/SQ2, and
H48/SQ1 showed that 2CTA closes the BF16 reducer gap, while FP8 remains faster
on the 1CTA parallel reducer. The B32/K4096 companion point has the analogous
split4/split2 relationship. The automatic-policy candidate therefore selects
2CTA only when all existing 48--64-row M64 conditions hold, the dtype is BF16,
the 1CTA split is exactly twice the 2CTA split, the 2CTA reference split is at
most four, and the local 1CTA K span is 513--1024 tokens. The lower bound comes
from a measured tile boundary: at B16/K4096, 1CTA's 512-token local span was
26.2528 us versus 26.5600 us for 2CTA, while at B16/K4097 the extra local tile
reversed the five-run medians to 28.2048 us versus 27.5904 us. Existing direct
split-1 crossover behavior remains enabled for both BF16 and FP8.

Nsight Systems then identified padding in the 2CTA reference reducer itself.
For H48/SQ1, its old launch reduced all 128 physical M128 rows even though only
48 logical rows were public. The candidate now launches over
`ceil(H * SQ / rows_per_cta)` and decomposes each logical row back into its
physical producer workspace coordinate. Fixed Q uses its direct flat storage
coordinate; packed Q retains `qo_indptr` mapping and predicates zero-length
requests. The last CTA clamps inactive row groups before forming any workspace
address. Reducer CTAs use four rows when the producer supplies at least half a
B200 wave and the resulting grid stays within four waves; otherwise they keep
the established eight-row form. The launch bound preserves 1,024 resident
threads as CTA size changes. Six BF16/FP8 R4-versus-R8 controls were neutral or
faster with R4, including H48/SQ1 B16/K8192 and H96/SQ4 B4/K2048.

Correctness completed for kernel checkpoint `97a40ae9`:

- 30 host selector/topology contracts passed, plus Ruff, Python compilation,
  and `git diff --check`;
- five forced-reference GPU cases passed across BF16/FP8, H96/SQ2 and
  H12/SQ11 tails, packed per-request Q lengths including zero, and graph replay;
- four BF16 public-auto equal-work factorizations all selected 2CTA split 4,
  the logical-row reference reducer, and four rows per CTA; eager accuracy
  passed for all four and the H48/SQ1 anchor also passed standalone and graph
  parity; and
- the FP8 H12/SQ4 counterpart retained M64 1CTA split 8 and passed accuracy.

The first matched candidate matrix measured the following before final source
checkpointing:

| B/K | H/SQ | PrimTS us | CuTe DSL us | Speedup | Public policy |
| --- | --- | ---: | ---: | ---: | --- |
| 16/8192 | 6/8 | 40.8848 | 41.1040 | 1.005361 | 2CTA, S4, R4 reference |
| 16/8192 | 12/4 | 40.8160 | 41.1040 | 1.007056 | 2CTA, S4, R4 reference |
| 16/8192 | 24/2 | 40.8896 | 41.4064 | 1.012639 | 2CTA, S4, R4 reference |
| 16/8192 | 48/1 | 40.6896 | 39.2752 | 0.965239 | 2CTA, S4, R4 reference |
| 32/4096 | 6/8 | 41.1040 | 41.7088 | 1.014714 | 2CTA, S2, R4 reference |
| 32/4096 | 12/4 | 41.1696 | 41.7120 | 1.013175 | 2CTA, S2, R4 reference |
| 32/4096 | 24/2 | 41.1136 | 41.7136 | 1.014594 | 2CTA, S2, R4 reference |
| 32/4096 | 48/1 | 40.9024 | 39.7984 | 0.973009 | 2CTA, S2, R4 reference |

The H48/SQ1 B16/K8192 outlier is still within the requester-confirmed 5%
tolerance. Five exact-candidate PrimTS repeats had a 40.5920 us median; five
matched CuTe DSL repeats had a 39.2640 us median, or 0.967284x. Experimental PDL
signal movement improved the ratio slightly but was removed because it changed
producer/dependent ordering solely to chase the superseded 0.97 threshold.
The safe post-TMEM signal ordering remains intact.

All diagnostics and profiles are retained below
`gate_b_formal_public_auto_6e2338db`, including the equal-wave matrices,
R4/R8 controls, split sweep, Nsight Systems reports, and five-run H48 series.
The next recovery point is this documentation commit on top of `97a40ae9`,
followed by a new exact-SHA formal directory with updated dispatch assertions.
After the 31-point/five-campaign Gate-A/Gate-B qualification, step 12 remains
the required TRTLLM-GEN/CuTe-DSL/public-auto-PrimTS eager and CUDA-graph
comparison.

The first complete test-file run after `97a40ae9` passed 230/231 cases. Its
only failure was the pre-crossover BF16 family assertion at H6/SQ8,
B16/K4097; the planned 2CTA output was not executed because the assertion ran
first. All other correctness cases passed. The 513-token lower boundary and
dtype-specific assertion are committed in `12dc0e4f`. The focused selector and
two-dtype GPU gate passed 15/15. The complete exact-checkpoint rerun then passed
233/233 cases in 744.92 seconds on B200 UUID
`GPU-3f241fbe-8ae5-204d-35a7-8f613c2a22f0`. Its JUnit report is under
`full_validation_12dc0e4f_gpu_3f241`. This closes exact-tree correctness for
the logical-row reducer and bounded family crossover; create the formal
directory pinned to `12dc0e4f` next.

## 23. Exact-source formal Gate-B campaign (2026-08-13)

The acceptance campaign is now running under
`gate_b_formal_public_auto_12dc0e4f`, pinned to kernel checkpoint `12dc0e4f`.
Every backend/row is a fresh process with explicit candidate `PYTHONPATH`, a
backend-specific JIT cache, refcheck, CUDA-event graph timing, 20 warmups, 100
measured iterations, seed 42, alternating backend order, and an explicit
completion marker. The runner rejects source changes after the kernel
checkpoint except commits to this recovery plan. Its analyzer independently
checks public-auto family dispatch, requires all 31 pairs and their `.ok`
markers, and fails if the shard geometric mean or any measured row is below
the requester-confirmed 0.95 Gate-B floor.

The first BF16 H6/SQ8 shard completed all 31 pairs. Public-auto selected 2CTA
at B64/K512, B64/K2048, B16/K8192, and B32/K4096 and throughput-latency 1CTA
on the other 27 points, exactly matching the checked-in campaign expectation.
The CuTeDSL/PrimTS geometric-mean speedup was 1.100814x. The minimum was
0.961794x at B16/K128 (8.3376 us CuTe DSL versus 8.6688 us PrimTS), so this
first-coverage shard passes the 0.95 pointwise and geometric-mean requirements.
BF16 H12/SQ4 is the next active shard on the same B200 allocation. These are
first-coverage results; the required multi-campaign aggregation and dispersion
report remain open.

The remaining three 48-row BF16 factorizations then completed. H12/SQ4 had a
1.107340x geometric mean and 0.980895x minimum; H24/SQ2 had a 1.104245x
geometric mean and 0.979954x minimum. H48/SQ1 had a 1.088510x geometric mean,
but its first pass exposed two pointwise misses. Five alternating-order pairs
confirmed B256/K512 at 45.8144 us PrimTS versus 43.1552 us CuTe DSL
(0.941957x ratio, 0.942696x pair geomean) and B4/K32768 at 42.9696 us versus
40.6976 us (0.947125x ratio, 0.947111x pair geomean). The `12dc0e4f`
checkpoint therefore remains correctness-qualified but does not pass formal
Gate B.

Forced M8/M16/M32 1CTA profiles were materially slower on both rows. Forced
2CTA was also slower at B256/K512 and only barely sufficient at B4/K32768.
Multi-wave CLC and static-persistent M64 variants failed refcheck, preserving
their existing safety exclusion. Dense-equivalent SQ1 planning was neutral.
The M64 pipeline sweep instead isolated an unadopted existing dimension:
four KV stages with two instructions per loop passed refcheck and reduced the
two PrimTS rows to 31.2832 us and 27.1872 us. Two and three stages were slower,
while five and six exceed B200 SMEM capacity. Before adopting the two-
instruction form, qualify BF16 and FP8 K tails, split reduction, fixed and
packed Q (including zero-length requests), graph replay, page sizes, and the
complete PrimTS MLA test file. Any source change invalidates the current
acceptance directory; restart formal performance under a new exact kernel SHA.

## 24. Replacement-GPU tuning and current candidate (2026-08-14)

The two-instruction result in section 23 was a false performance signal from
an insufficient reference check. Keeps-MMA-AB has one QK/PV instruction
stream; setting `num_insts_kv=2` changed the loop-domain partition without
adding the second stream used by the swaps schedule and therefore skipped
alternating KV tiles. New product tests with K tails, split KV, graph replay,
packed Q, and zero-length requests exposed the error. The production candidate
retains one KV instruction, and the config now documents why that value is a
correctness invariant. Do not revive the two-instruction experiment without a
real second Keeps QK/PV stream.

The replacement B200 is UUID
`GPU-c574acab-9bdc-aadc-b45c-57d9489db33f`, container hostname
`a95ef9b435cc`, driver 595.58.03, PyTorch 2.10.0+cu128, runtime CUDA 12.8,
compute capability 10.0, and a 1000 W power limit. Results from the expired
GPU are not combined with this allocation. A fresh same-GPU one-instruction
baseline measured H48/SQ1 B256/K512 at 46.0192 us PrimTS versus 43.5616 us
CuTe DSL (`0.946596x`) and B4/K32768 at 42.9568 us versus 40.7072 us
(`0.947631x`).

The following diagnostics were rejected:

- Keeps M64 KV depths 2 and 3 were slower; depth 5 exceeds shared-memory
  capacity. Register redistributions did not beat the accepted configuration.
- Persistent CLC/static scheduling was correct in eager execution but about
  55 us and produced stale graph-replay output. The direct multi-wave grid
  remains required.
- Removing the nominally idle twelfth warp produced an invalid task/register
  grouping; launching the retained task with 352 threads deadlocked. Keep the
  qualified 384-thread CTA.
- Three active softmax/correction row warps passed direct, split, graph,
  packed-zero, and FP8 controls, but measured 45.926 us. A normal-order repeat
  restored CuTe DSL to 43.464 us, only `0.94634x`, so the prototype was removed.
- A fixed-SQ1 flat-index fast path was correct but performance-neutral at
  45.8256 us and was removed.
- Forced 1CTA split 2 was correct but 65.5696 us. Throughput-2CTA combined K/V
  depths 5, 6, and 7 measured 48.0736, 47.0560, and 46.2304 us. Depth 8 needs
  238,960 bytes versus the 232,448-byte SM100a limit; exchanging one P stage
  for the eighth K/V stage was legal but remained about 46.24 us.

Two bounded changes remain in the uncommitted source candidate on top of the
exact `12dc0e4f` kernel checkpoint:

1. BF16 Keeps uses four page-offset stages instead of six; FP8 retains six.
   The same-GPU sweep found depths 1--4 near the minimum, while 5--8 were
   slower. Four matches the BF16 KV pipeline depth and keeps more latency
   tolerance than the equivalent one/two-stage points.
2. Public auto selects throughput 2CTA for the BF16 48-row, one-wave,
   1024-token-local-span topology when 1CTA has at least 32 splits, 2CTA has
   more than four splits, and both producer grids fit one resident wave. This
   is the measured B4/K32768 family; FP8 and all other topology boundaries
   retain their previous choice.

Five alternating-order same-GPU repeats under graph/event timing, 20 warmups,
100 iterations, seed 42, and refcheck produced:

| H48/SQ1 point | PrimTS median (us) | CuTe DSL median (us) | median ratio | pair geomean | minimum pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4/K32768 | 42.7552 | 40.7040 | 0.952025 | 0.951932 | 0.951668 |
| B256/K512 | 45.8240 | 43.3696 | 0.946439 | 0.947110 | 0.946372 |

The bounded long-K 2CTA rule therefore robustly closes one of the two original
Gate-B misses. B256/K512 remains the only known failure and needs roughly
0.4% additional improvement for a robust `>=0.95` result. Focused validation
for the retained source passed 17/17 host selector/config cases and 11/11 GPU
cases across all four 48-row factorizations, direct/split K tails, fixed and
packed Q, zero-length requests, graph replay, and an FP8 control. A later
9/9 stress subset independently repeated the key GPU coverage while testing
the now-rejected active-row prototype.

Artifacts are under `current_gpu_baseline_n1_c574`,
`candidate_public_auto_repeats_gpu_c574`, `diagnostic_keep_page_stages_gpu_c574`,
`diagnostic_keep_registers_gpu_c574`, `diagnostic_keep_combined_stages_gpu_c574`,
`diagnostic_long_2cta_gpu_c574`, `diagnostic_active_row_warps_gpu_c574`,
`diagnostic_2cta_stages_gpu_c574`, and `diagnostic_1cta_split_gpu_c574`.

The next-action paragraph originally recorded here used the then-current 0.95
floor and the uncommitted long-K candidate. Section 25 supersedes both that
acceptance floor and that candidate.

## 25. Six-percent Gate B and general-policy candidate (2026-08-14)

The requester expanded the allowed pointwise CuTe DSL gap to 6%, so the
normative Gate-B floor in section 6.3 is now `CuTeDSL / PrimTS >= 0.94`. Gate A
remains unchanged: no reproducible old-PrimTS regression beyond 3%. Under the
new floor, the exact `12dc0e4f` runtime passes every measured first-campaign
BF16 48-row point. Its previously flagged H48/SQ1 rows were `0.941957x` at
B256/K512 and `0.947125x` at B4/K32768. The H6/SQ8, H12/SQ4, H24/SQ2, and
H48/SQ1 shard geometric means were `1.100814x`, `1.107340x`, `1.104245x`, and
`1.088510x`, respectively. Required repeated-campaign aggregation is still
open; this threshold change reclassifies the known rows but does not replace
the remaining qualification runs.

The requester also required general, defensible policy changes rather than
shape-only heuristics. The uncommitted BF16 B4/K32768 long-one-wave 2CTA branch
from section 24 and its dedicated assertions were therefore removed. Although
it raised that row to roughly `0.952x`, its split-count and local-span bounds
were fitted to one topology and are not an acceptable public dispatch rule.
The checked-in topology-derived direct-output and equal-producer-work
crossovers from `12dc0e4f` remain unchanged.

The remaining BF16 page-offset-depth candidate was tested before retention.
One complete, alternating-order, refchecked CUDA-event pass compared four
stages against the established six stages across direct/split, short/long,
full-48-row, and low-row shapes:

| Shape | Four stages (us) | Six stages (us) | Six/four |
| --- | ---: | ---: | ---: |
| H48/Q1 B256/K512 | 45.8240 | 45.9264 | 1.002235 |
| H48/Q1 B4/K32768 | 42.9568 | 42.9536 | 0.999926 |
| H6/Q8 B16/K128 | 8.5536 | 8.5504 | 0.999626 |
| H6/Q8 B16/K65536 | 260.6176 | 259.4496 | 0.995518 |
| H12/Q4 B16/K4096 | 26.1632 | 26.1632 | 1.000000 |
| H24/Q2 B64/K8192 | 137.1904 | 136.1216 | 0.992209 |
| H6/Q1 B16/K4096 | 23.2992 | 23.2992 | 1.000000 |
| H24/Q1 B128/K4096 | 131.3056 | 130.3840 | 0.992981 |

Four stages had no broad advantage and regressed several long/throughput rows,
so the additional repeats were stopped and the production setting was restored
to six. Artifacts are under `page_stage4_vs6_general_gpu_c574`. The retained
source is therefore runtime-equivalent to `12dc0e4f`; its only config edit is a
comment that records the single-instruction Keeps correctness invariant.
Stronger tests retain direct and split K tails, graph replay, packed zero-length
Q, BF16/FP8 config contracts, and the BF16 M64 product. Static checks passed,
the config contract passed 2/2, and the focused GPU set passed 5/5 on B200 UUID
`GPU-c574acab-9bdc-aadc-b45c-57d9489db33f`.

Nsight profiling of the remaining H48/Q1 B256/K512 gap points to the
MMA-to-softmax chain rather than flat-row padding. The public 1CTA kernel had
14.07% eligible warps, 21.79 cycles per issued instruction, and 12.88 cycles of
long-scoreboard stall, while monolithic CuTe DSL had 17.38%, 15.42, and 9.15,
respectively. Source sampling placed the largest stalls at TMEM-score,
softmax-local-stat, and output-correction barriers. A tuned 2CTA register layout
remained slower at about 46.02 us; static persistent, static-K, and direct
nonpersistent 2CTA variants measured about 46.33, 46.23, and 56 us and were
rejected. No profiling-only schedule or register heuristic remains in source.

Next, run the complete PrimTS MLA test file on this general-policy tree, commit
an exact source/test checkpoint, and run formal Gate A plus repeated Gate B
under that identity. The terminal deliverable remains the issue-#4390-shaped
TRTLLM-GEN/CuTeDSL/public-auto-PrimTS comparison in eager and CUDA-graph modes.

## 26. General-policy exact-tree correctness (2026-08-14)

The general-policy source/test checkpoint is `69161c6c`. It is runtime-equivalent
to the qualified `12dc0e4f` kernel: the only production-source delta is the
comment documenting why Keeps-MMA-AB must retain one KV instruction. The
shape-specific long-K dispatch and the BF16 page-depth experiment are absent.

The complete PrimTS MLA test file passed 239/239 cases in 751.05 seconds on
B200 UUID `GPU-c574acab-9bdc-aadc-b45c-57d9489db33f` at 1000 W, with PyTorch
2.10.0+cu128, CUDA runtime 12.8, and compute capability 10.0. This includes all
233 cases from the prior exact checkpoint plus two BF16/FP8 Keeps configuration
contracts, three BF16 M64 direct/split/tail cases, and one packed zero-length-Q
case with eager, standalone, or CUDA-graph coverage as applicable. The JUnit
report and dedicated JIT cache are under
`full_validation_69161c6c_gpu_c574`. Ruff, Python compilation, formatting, and
`git diff --check` also passed before the checkpoint.

The next acceptance work is formal Gate A against the immutable old-PrimTS
baseline and repeated Gate B at the new 0.94 floor. Performance evidence must
identify `69161c6c` even though its generated kernel is identical to
`12dc0e4f`; do not mix results from the removed long-K/page-depth candidate.
