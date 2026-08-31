# PrimTS MLA Decode Flat Query-Row Packing

- Status: latest-main rebase and diagnostic B200 qualification complete
- Updated: 2026-08-31
- Working branch: `port/pr4178-groups-tokens-heads-prims-ts`
- Upstream baseline: `012cfdb97f217e0d48bc9352c17a74068c9e495b`
- Reference: FlashInfer PR #4178, head `78dfa5c1186aa80a66d0f1375a93c089c2775145`
- Performance floor: public-auto PrimTS must retain at least `0.94x` CuTe DSL

## Objective

Port PR #4178's no-padding query-row geometry to both PrimTS MLA decode
families while preserving public automatic dispatch:

- throughput-latency 1CTA with M8, M16, M32, and M64 profiles;
- throughput 2CTA with cooperative M128 tiles;
- BF16 and FP8 inputs with BF16 output;
- fixed and packed variable-Q inputs, including empty packed requests;
- dense and bottom-right causal masks;
- direct output, cluster reduction, reference GMEM reduction, and parallel
  GMEM reduction; and
- non-power-of-two query-head counts, especially H6, H12, H24, H48, and H96.

No public family, profile, split-KV, reducer, or scheduler knob is added. The
public dispatcher owns those choices and reports the selected topology through
private policy provenance used by tests and benchmarks.

## Geometry contract

Query tokens and query heads form one affine row space:

```text
flat_row  = query_token * H + query_head
total_rows = SQ * H
num_q_tiles = ceil(total_rows / M)
tail_rows = total_rows - (num_q_tiles - 1) * M
```

Every physical tile owns consecutive rows. It may cross token boundaries, and
only the final tile may be partial. There is no whole-token grouping ratio and
no structural Q padding between logical tokens.

The public storage contract remains:

- fixed Q/O: `[B, SQ, H, D]`;
- packed Q/O: `[total_q, H, D]` with cumulative `qo_indptr`;
- Q/K input dimension: 512 latent + 64 RoPE;
- output dimension: 512; and
- paged K/V with supported page sizes 16, 32, 64, and 128.

For each physical row, shared helpers derive the logical query index, logical
head index, public storage row, and validity predicate. Invalid tail rows use a
safe control-flow coordinate so every participant can synchronize, but they
must not issue public or split-workspace transactions.

## Kernel-family implementation

### Throughput-latency 1CTA

The selected M profile determines the physical head extent and number of flat
Q tiles. Logical H and SQ remain separate config fields for causal coordinates
and public storage. Swaps-MMA-AB and Keeps-MMA-AB consume the same flat-row
mapping.

Split-KV policy is based on projected producer work, resident capacity, local
K-tile depth, dtype, and task-graph variant. Power-of-two split rounding is
allowed only when the rounded grid retains at least five sixths of the target
wave and stays within the bounded local-mainloop window. Explicit profile,
split, and persistence choices remain authoritative inside internal selectors.

### Throughput 2CTA

M128 scheduling uses `ceil(H * SQ / 128)` physical tiles. Q descriptors keep
the logical row extent, so TMA out-of-bounds fill supplies zeros for the final
tail. Static and CLC work queues use the same tile count and map valid rows back
to fixed or packed storage.

The producer and every reducer share the normalized partial layout. Parallel
G1 reduction is selected only when the logical row grid is complete and its
wave/work criteria are met; padded M128 tails retain the compact reference
reducer. High split counts may use the established clustered reducer.

### Dynamic batch compilation

Batch size is a runtime tensor extent, not a raw compile specialization, for
context, standard decode, and MLA decode. A compiled callable may be reused
across batch sizes when all policy and reducer topology fields match.

Planning still uses batch size to select policy and size workspace. Standard
decode workspace must therefore be re-zeroed whenever any workspace-layout
input, including batch size, changes. MLA workspace does not require initial
zeroing. Both 1CTA and 2CTA derive the live batch from runtime tensors and use
64-bit-safe partial-output offsets where batch participates in products.

## Public automatic policy

Automatic family selection is intentionally structural. It uses:

- logical flat-Q row count;
- candidate producer work and resident capacity;
- selected 1CTA and 2CTA split counts;
- local K128 tiles per producer;
- dtype and 1CTA task-graph variant; and
- direct-output versus bounded reducer cost.

The policy contains no GPU-name whitelist, named query-head branch, or
single-shape override. Small flat workloads normally retain 1CTA, but a compact
2CTA topology may win when it removes or shrinks reduction without losing wave
fill. Larger workloads normally retain 2CTA; 1CTA is considered when measured
work/capacity rules show a materially underfilled 2CTA launch. Thresholds are
defined beside the policy and apply to shape families rather than benchmark
row identities.

## Correctness gates

Run host checks first, then GPU tests in fresh process shards. A long
single-process MLA run has previously poisoned the CUDA context after many JIT
topologies, so process-isolated families are the qualification result.

Required coverage:

- context: the complete PrimTS context suite, including fixed, packed,
  paged-KV, masks, windows, graph replay, and cross-batch callable reuse;
- standard decode: the complete PrimTS decode suite, including grouped Q,
  non-power row counts, page-size 16, Keeps sliding window, packed Q, graph
  replay, split-KV, and cross-batch reuse;
- MLA decode: the complete PrimTS MLA suite, including 1CTA and 2CTA, all M
  profiles, H6/H12/H24/H48/H96, power-of-two controls, BF16/FP8, fixed and
  packed Q, empty requests, dense/causal masks, all page sizes, direct and
  split paths, every reducer, graph replay, runtime-K pruning, and cross-batch
  reuse; and
- reference checking for every timed benchmark row before performance timing.

All numerical checks use the existing independent references. Test matrices
exercise public-auto behavior. Monkeypatched family/profile fallbacks are not
part of the durable GPU suite.

## Performance gates

Use one idle B200, CUDA graphs for short kernels, cold-L2 timing where the
runner supports it, and fresh same-session backend results.

### Gate A: previous PrimTS

Compare the candidate with a frozen pre-port PrimTS checkout on matched shapes.
Record family, M, split-KV, producer CTA count, reducer, workspace bytes, and
latency. No reproducible regression may be hidden by changing the compared
topology or backend.

### Gate B: CuTe DSL

Compare public-auto PrimTS with monolithic CuTe DSL across the no-padding and
non-power-head matrix. Every reproducible row and each dtype-wide geometric
mean must satisfy:

```text
CuTe DSL latency / PrimTS latency >= 0.94
```

If the PrimTS heuristic selects 2CTA, the comparison still uses that public
choice. A matched-split diagnostic may explain a gap but does not replace the
public-auto gate.

### Final three-backend matrix

Reproduce the issue-#4390 comparison with TRTLLM-GEN, CuTe DSL, and public-auto
PrimTS in both eager and CUDA-graph modes. Cover BF16/FP8, Q1/Q8, and
K131072/K500000/K1M. Run all 72 backend rows fresh in separate processes and
retain refcheck, selected policy, split-KV, and latency.

## Last verified checkpoint before cleanup

The exact pre-cleanup source passed:

- 219 host-only context/decode/MLA tests;
- context: 74/74 process-isolated functional and numerical cases;
- standard decode: 78/78 process-isolated cases;
- MLA decode: 162/162 process-isolated cases; and
- nine cross-batch callable-reuse and numerical cases.

Diagnostic same-GPU performance results were:

- BF16 H48/Q1/B256/K512, 1CTA: CuTe/PrimTS `0.942334`;
- BF16 H48/Q1/B64/K2048, public-auto 2CTA: CuTe/PrimTS `0.972863`; and
- issue-#4390 matrix: 72/72 refchecks, PrimTS won 24/24 comparison cells,
  CuTe/PrimTS geometric mean `1.422547`, minimum `1.155084`.

Those runs used GPU UUID
`GPU-49b85be8-916a-35e4-257d-a2ed1798e814` in an eight-GPU container without a
Slurm-isolated allocation. They establish a regression checkpoint but are not
formal isolated timing signoff.

## Current cleanup checkpoint

The 2026-08-30 cleanup preserves policy equations and removes representation
debt around them:

- removed the retired whole-token grouping ratio, compatibility helpers,
  ignored constructor arguments, and unused resource fields;
- renamed the internal launch record to describe flat query rows directly;
- represented the automatic 1CTA family probe as one candidate record;
- represented context compile identities as named immutable records;
- moved standard-decode config freezing into `FmhaDecodeConfig`;
- removed unused MLA launch-spec dtype fields;
- added a fixed-Q 2CTA runtime assertion that output batch matches
  `cache_seqs`; and
- removed three monkeypatched forced-policy GPU tests whose row/reducer
  behavior is covered by public-auto and structural matrices.

The exact cleaned source passed all post-cleanup checks on 2026-08-30:

- `compileall`, Ruff check/format, and `git diff --check`;
- host-only complete files: 218 passed and 305 skipped;
- context on B200: 137/137 passed;
- standard decode on B200: 127/127 passed; and
- MLA decode in four exhaustive fresh-process shards: 216 + 24 + 10 + 9 =
  259/259 passed.

The 28-cell structural Gate A/B matrix used 84 fresh, refchecked processes.
The candidate/prior-PrimTS latency ratio had geometric mean `1.032582` and
minimum `0.971828`. CuTe DSL/candidate had geometric mean `1.102841` and
minimum `0.940324`; the BF16 and FP8 geometric means were `1.093364` and
`1.112401`. Every point passed the `0.97` prior-PrimTS and `0.94` CuTe floors.
The matrix covers equal 48-row factorizations, single-token H12/H24/H48/H96,
general reducer-depth controls, and the resident-wave boundary.

The final issue-#4390 matrix used 72 additional fresh, refchecked processes.
PrimTS won all 24 eager/graph cells against TRTLLM-GEN and monolithic CuTe
DSL. CuTe DSL/PrimTS geometric mean was `1.436939` and the minimum was
`1.154125`.

All GPU runs used NVIDIA B200 UUID
`GPU-49b85be8-916a-35e4-257d-a2ed1798e814` with `CUDA_VISIBLE_DEVICES=0`.
The current container exposes eight GPUs and could not reach an Umbriel Slurm
login, so timing remains diagnostic rather than formal isolated signoff. Raw
logs, JUnit XML, CSVs, analyzers, resumable runners, and exact provenance are
under `/workspace/prims_ts/artifacts/groups_tokens_heads_20260830`.

## Latest-main rebase checkpoint

On 2026-08-31 the complete branch and 17-file worktree were backed up before
rebasing. The verified bundle, binary patch, full-file archive, backup branch,
and preserved stash are documented under
`/workspace/prims_ts/backups/flashinfer_rebase_20260831_6542ed6d`.

The 38 branch commits were rebased from main `065971254bca` onto fetched main
`93151678bcd0`, producing branch HEAD `f2c68fe17f52`. Range comparison matched
37 commits exactly; the remaining commit only lost a deletion already present
in new main. Restoring the worktree required three composed resolutions:
variable-window context plus batch-dynamic compilation, paged-KV/block-sparse
decode validation plus compile signatures, and Q64/KV256 decode coverage plus
cross-batch reuse. No commit was dropped and no remote branch was pushed.

The first post-rebase source passed:

- compile/Ruff/diff checks passed;
- host-only complete files: 260 passed and 319 skipped;
- context on B200: 144/144 passed; and
- standard decode on B200: 176/176 passed; and
- MLA on B200: 216 + 24 + 10 + 9 = 259/259 passed.

That qualification exposed one same-topology regression at BF16 H64/Q1/B16/
K12288. Making batch dynamic had widened every partial-output index operation
to 64 bits. The general correction keeps the compile-time-bounded within-batch
offset in 32-bit arithmetic, widens one runtime batch-stride product, and
retains a full-64-bit fallback when one batch can exceed Int32. It contains no
shape or device exception. Three alternating candidate/baseline probes passed
with ratio geometric mean `0.998701` and minimum `0.991610`. The final stable
runtime/test/benchmark fingerprint is
`bad7646a3b4590a4df0f07603e37732211e9e7040fb173fedcd4d08901a0b6a2`.

The final source again passed compile/Ruff/diff checks, 260 host tests, and all
259 MLA GPU cases. The prior context/decode files and kernel sources were
unchanged by the isolated MLA helper correction, so their 144/144 and 176/176
results remain exact for those paths.

The 28-cell structural matrix used 84 fresh, refchecked processes. CuTe DSL/
candidate geometric mean was `1.104270`, minimum `0.942195`, with all points
above the `0.94` floor. Prior-PrimTS/candidate geometric mean was `1.029460`.
One high-variance BF16 H64/Q1/B64/K4096 primary sample was `0.964721`; five
alternating paired repeats all passed the `0.97` floor, with median `1.003826`,
geometric mean `1.021193`, and minimum `0.980226`. No reproducible prior-PrimTS
regression remains.

The issue-#4390 matrix used 72 additional fresh, refchecked processes across
TRTLLM-GEN, CuTe DSL, and public-auto PrimTS. PrimTS won 24/24 eager/graph
cells. CuTe DSL/PrimTS geometric mean was `1.423550`, minimum `1.093133`, with
zero failures.

Before handoff, remote main advanced by three unrelated commits to
`012cfdb97f217e0d48bc9352c17a74068c9e495b`. A second verified backup of the
qualified source was created under
`/workspace/prims_ts/backups/flashinfer_second_rebase_20260831_f2c68fe1_bad7646a`.
All 38 commits then rebased cleanly to HEAD `8ae15e8d8a1b`; range comparison
against the first rebased sequence matched all 38 commits exactly. The three
upstream commits do not change PrimTS, MLA benchmark, CuTe MLA, or TRTLLM MLA
paths, and the 16-file fingerprint remains `bad7646a...`. Latest-main static
checks, the complete 579-item host collection (260 passed, 319 skipped), and a
27-case B200 context/decode/MLA integration smoke all passed.

Both named stashes remain preserved: `66c705f1ceb4` for the fully qualified
source and `8e19c02b98b7` for the original pre-rebase worktree. No commit or
remote branch was pushed. Complete logs, XML, CSVs, analyzers, runners, and
provenance are under `/workspace/prims_ts/artifacts/groups_tokens_heads_20260831`.

## Recovery procedure

1. Verify branch, tracked diff, loaded package path, CUDA/PyTorch versions, and
   the visible B200 UUID.
2. Prefer a one-GPU Umbriel Slurm allocation and the FlashInfer development
   container. If the current environment cannot reach Slurm, record that fact,
   pin `CUDA_VISIBLE_DEVICES` to one idle B200, and label timing diagnostic.
3. Run `py_compile`, Ruff check/format, and `git diff --check` on every touched
   file.
4. Run host-only tests, then the three complete GPU suites in fresh-process
   shards. Preserve resumable results outside the source tree.
5. Run Gate A and Gate B with fresh same-session baselines and CUDA graphs.
6. Run the 72-row issue-#4390 three-backend matrix with refcheck.
7. Record exact results here before any further optimization.

Durable prior artifacts are outside the source checkout under:

- `groups_tokens_heads_20260828/dynamic_batch_diagnostic_gpu_49b8`; and
- `issue4390_dynamic_batch_eb8079d4_gpu_49b8_r0`.

## Pitfalls, limitations, and fallbacks

- Pitfalls: final physical tiles may cross token boundaries; causal K bounds
  and output validity are row-specific. Empty packed tiles must advance the
  scheduler rather than terminate it. Producer and reducer geometry must stay
  derived from the same flat-row layout.
- Regressions: none accepted. The post-cleanup rerun must name any correctness,
  compilation, resource, or performance regression explicitly.
- Limitations: SM100/SM103 only; input dtypes BF16/E4M3; current public output
  BF16; supported MLA dimensions and page sizes remain unchanged. Formal
  performance signoff requires an isolated B200 allocation.
- Fallbacks: automatic policy may choose the other implemented PrimTS MLA
  family when the general work/capacity rule selects it. Unsupported shapes
  fail closed; there is no silent CuTe DSL or reference-backend fallback.
