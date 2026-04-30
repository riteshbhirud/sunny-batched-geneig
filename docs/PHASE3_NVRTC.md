# Phase 3: NVRTC runtime compilation

## Goal

Remove the N=54 hardcode from the kernel so Sunny.jl can use any
matrix size determined at runtime by the user's magnetic unit cell.

## Sub-step plan

- **3a (done, commit a64a862):** NVRTC plumbing for the reduction
  stage; hybrid with static heev finishing stage to work around
  CUSOLVERDX_NVRTC_HEEV_BUG. N=54 still hardcoded via `-DM_SIZE=54`.
- **3b (done, commit d9a835d):** cuSolverDx header overlay applied
  (`src/nvrtc/cusolverdx_overlay/`) — heev now compiles under NVRTC.
  Hybrid split removed; the NVRTC kernel runs all five stages
  (`potrf → trsm × 2 → heev → trsm`) in one launch. Constructor
  signature is `NvrtcGeneigSolver(int n, int device_id)`. Fixtures
  generated at N ∈ {16, 32, 48, 54, 64, 96, 128}; per-N validation in
  `tests/test_nvrtc_multiN.cpp`.
- **3.5a (done, commit c8a5adf):** BatchesPerBlock parameterization.
  Constructor now takes a third optional argument
  (`batches_per_block_request`). When 0 (default), the constructor
  auto-selects: tries BPB ∈ {16, 8, 4, 2, 1} from largest to smallest,
  accepts the first that compiles AND whose `solver_shared_memory_size`
  fits the device's per-block dynamic shared-memory ceiling. cuSolverDx's
  own `static_assert` rejects BPB combinations whose shared memory
  exceeds the SOLVER_SM tag's ceiling (164 KB at SM<800>) before
  reaching the device-level check. New test
  `tests/test_bpb_sweep_n54.cpp` validates explicit-BPB construction
  at all candidates and reports throughput on the n54_b32 fixture.
- **3.5c (done, this commit, no-op kernel change):** Negative-result
  experiment on cuSolverDx primitive synchronization. Reading
  `cusolverdx/database/cholesky.cuh` and `htev.cuh` shows their `dispatch`
  functions end with `__syncthreads()` in every code path, suggesting
  the external `__syncthreads()` we issue after each `Solver().execute()`
  is redundant for the Cholesky and Heev cases. Removing those two syncs
  *was source-level safe* but **regressed throughput by 14% on N=54
  BPB=1**: 5-sample median dropped from **6,494 mat/s** (5 syncs) to
  **5,791 mat/s** (3 syncs). Hypothesis: cuSolverDx's internal sync uses
  the partial-warp/warp-per-batch heuristic and may only sync the warps
  that participated in that operator's dispatch, leaving other warps
  free to race ahead and conflict with the next operator's full-CTA
  load. The 5 explicit syncs in our kernel are therefore **load-bearing
  for throughput on this hardware**, not just a defensive default. We
  keep them, ship the experiment as documented evidence, and add the
  finding to the project's "things that look like obvious wins but are
  not" register. The 5-sample medians (each sample is a fresh NVRTC
  cold compile + a 1024-matrix bench at N=54 BPB=1):
    - 5 syncs (3.5b kernel):  6,581 6,494 6,601 6,406 6,335 → median 6,494
    - 3 syncs (3.5c attempt): 5,740 5,784 5,800 5,808 5,791 → median 5,791
- **3.5b (done, commit cbf80bb):** Per-(N, BPB, arch) block dimension via
  cuSolverDx's `suggested_block_dim`. Constructor now does a two-phase
  compile per BPB candidate: a cheap probe (~3-5 s on this hardware)
  instantiates the solver types and reads each operator's
  `__constant__ suggested_block_dim`; we take the max .x across the
  five operators (potrf, three trsms, heev) and recompile the
  production kernel with `BlockDim<max>`. The probe source is held
  alongside the production source in
  [src/nvrtc/kernel_source.hpp](../src/nvrtc/kernel_source.hpp). The
  production source's hardcoded `BlockDim<128>` is replaced with
  `BlockDim<BLOCK_DIM_X>` driven by a new `-DBLOCK_DIM_X=` macro. The
  probe is fast because the source has no `execute()` calls and
  therefore no LTO IR cross-references into the cuSolverDx fatbin
  during nvJitLink. Per-N picked block dims on the development
  hardware: see Phase 3.5b table below.
- **3c:** Per-size CUmodule cache inside `NvrtcGeneigSolver`.
  Construct once per N, reuse across all matrices.
- **3d:** Disk-backed cubin cache, keyed on (M_SIZE, arch, library
  version hash). Solves the cold-start problem across Julia sessions.
- **3e:** Performance characterization vs the static N=54 path,
  plus end-to-end timing on the existing batched test.

## 3.5b multi-N validation outcome (sm_120 RTX 5070 Ti Laptop, 101,376 bytes max dynamic shared memory per block)

| N   | BPB | BlockDim.x | shared_mem | cold compile (ms) | rand max_eig_rel | rand max_phase | near max_phase | status |
|-----|----:|-----------:|-----------:|------------------:|------------------:|----------------:|---------------:|--------|
|  16 |   8 |         96 |     70,400 |            52,058 |          0.00e+00 |        1.78e-15 |       1.89e-15 | PASS   |
|  32 |   2 |         64 |     68,032 |            55,329 |          0.00e+00 |        1.89e-15 |       1.78e-15 | PASS   |
|  48 |   1 |        128 |     78,704 |            53,489 |          1.50e-15 |        2.55e-15 |       2.33e-15 | PASS   |
|  54 |   1 |        128 |     99,712 |           221,632 |          2.16e-15 |        3.77e-15 |       2.44e-15 | PASS   |
|  64 |   — |          — |          — |                 — |                 — |               — |              — | SKIP   |
|  96 |   — |          — |          — |                 — |                 — |               — |              — | SKIP   |
| 128 |   — |          — |          — |                 — |                 — |               — |              — | SKIP   |

**Per-operator suggested block_dim.x at successful (N, BPB):**

| N  | BPB | Cholesky | TrsmLeft | TrsmRight | TrsmLeftConj | Heev | max → BDx |
|---:|----:|---------:|---------:|----------:|-------------:|-----:|----------:|
| 16 |   8 |       96 |       32 |        32 |           32 |   32 |        96 |
| 32 |   2 |       64 |       64 |        64 |           64 |   32 |        64 |
| 48 |   1 |       64 |       64 |        64 |           64 |  128 |       128 |
| 54 |   1 |       64 |       64 |        64 |           64 |  128 |       128 |

cuSolverDx's recommendations vary per primitive: at small N with high BPB
(N=16, BPB=8) Cholesky wants **96** threads per block while heev only wants
32. At larger N (N=48, 54) heev's recommendation of **128** dominates. The
auto-select takes the max so every operator has at least its required
threads.

For N=32, BPB=2 the new code uses **BlockDim<64>** vs the previous
hardcoded BlockDim<128>; this also slightly reduces heev's workspace_size
(shared_mem dropped from 69,600 to 68,032 bytes — 1.5 KB saved per block).

### Phase 3.5b BlockDim micro-bench at N=32, BPB=2 (test_blockdim_n32)

Direct A/B benchmark on 1024 random (H, S) pairs, same workload, same
solver, only difference is BlockDim.x:

| BlockDim.x | source       | mat/sec | wall (s) | shared_mem | correctness |
|-----------:|--------------|--------:|---------:|-----------:|-------------|
|         64 | suggested    |  46,692 |   0.0219 |     68,032 | PASS        |
|        128 | 3.5a hardcode |  40,480 |   0.0253 |     69,600 | PASS        |

**Ratio: 1.153 — suggested BlockDim<64> is 15.3% faster.**

This is a real architectural win, not just future-proofing for H100. At
N=32, every cuSolverDx primitive in our pipeline (potrf, three trsms,
heev) reports a suggested block_dim.x ≤ 64. Running with BlockDim<128>
launched twice as many threads as any operator could use, cutting
occupancy in half. cuSolverDx's per-arch tuning database knows this;
the probe surfaces it. Both variants produce eigvals/eigvecs at fp64
epsilon, so this is a pure throughput improvement.

At N=54 (Sunny's actual SW08 size on this hardware) the suggested max
matches 128, so the production kernel is byte-identical to 3.5a — no
delta either way. The gains are at smaller N. For the SW08 workload
specifically, 3.5b is correctness-equivalent to 3.5a; the win materializes
when Sunny extends to other matrix sizes.

**Summary:** 8/8 (N, fixture) pairs passed at fp64 epsilon for
N ∈ {16, 32, 48, 54}. N ∈ {64, 96, 128} skipped — even BPB=1's
shared-memory requirement exceeds the device ceiling on this
consumer hardware.

The auto-select picked **BPB=8 at N=16** and **BPB=2 at N=32** —
multiple matrices per block fit at the smaller sizes, packing more
work into each launch. At N=48 and above, only BPB=1 fits.

Note the eigenvalue relative diff at N=16 and N=32 is **0.00e+00**
(LAPACK's outputs are bit-identical to ours, given the same input
fixture). At larger N the residual is single-digit ulps, well within
fp64 epsilon.

### BPB sweep at N=54 (test_bpb_sweep_n54, Phase 3.5b)

| BPB | BlockDim.x | constructable | matrices/sec | max_eig_rel | max_phase | status |
|----:|-----------:|---------------|-------------:|------------:|----------:|--------|
|  1  |        128 | yes           |        4,645 |   2.22e-15  | 3.44e-15  | PASS   |
|  2  |          — | no            |            — |          —  |        —  | SKIP (probe rejected: cuSolverDx static_assert) |
|  4  |          — | no            |            — |          —  |        —  | SKIP   |
|  8  |          — | no            |            — |          —  |        —  | SKIP   |

On H100 NVL (228 KB per-block opt-in shared memory), BPB=2 at N=54
is expected to fit (~199 KB requirement). That measurement is in
Phase 6.

### Phase 3.5b throughput regression check (N=54, B=1024, BPB=1)

Same workload as the Phase 2.2e baseline benchmark (1024 synthetic
matrices, BPB=1, NVRTC path):

| Path | matrices/sec | wall (s) |
|---|---:|---:|
| Phase 2.2e static-link  | 6,740 | 0.152 |
| Phase 3.5a NVRTC (hardcoded BDx=128) | 6,699 | 0.153 |
| Phase 3.5b NVRTC (suggested BDx=128) | 6,740 | 0.152 |

At N=54, BPB=1 the cuSolverDx-suggested `max(64,64,64,64,128)=128`
matches the previous hardcode, so the production kernel is identical.
The 0.6% delta vs 3.5a is run-to-run noise. **No regression.**

The single-block kernel layout requires
`Cholesky::shared_memory_size + Heev::shared_memory_size` = roughly
`2·N²·sizeof(cdouble) + workspace`. At N=64 that exceeds the
sm_120 laptop's 101 KB per-block ceiling. cuSolverDx 25.12's
`static_assert "Provided combination of data type and sizes makes this
problem not fit into shared memory available on the specified architecture"`
catches the larger N values before they reach the GPU.

H100 SXM5 / NVL has 228 KB max dynamic shared memory per block (opt-in),
which would push the single-block ceiling to roughly N≈80 for the
unified five-stage layout. Beyond that, a multi-block strategy (e.g.,
splitting the matrix tile across cooperative blocks, or reusing one
block's shared memory across stages instead of holding As + Heev region
live simultaneously) is needed for very large N. Deferred to Phase 6.

For Sunny's current SW08 usage, N=54 is the operative size and is
fully validated. The runtime-N pitch holds for N up to 54 on consumer
hardware; up to ~80 on H100; further extension is a future-work item.

## Notes

- The cold NVRTC + nvJitLink pipeline takes 5–215 s per N (scales
  superlinearly with N as more cuSolverDx LTO objects are pulled in).
  Acceptable for Sunny's usage pattern (one-time cost per (session, N),
  amortized over hundreds of thousands of matrix solves) but motivates
  the disk cache in 3d. See timing breakdown in
  [CUSOLVERDX_NVRTC_HEEV_BUG.md](CUSOLVERDX_NVRTC_HEEV_BUG.md).
- Removing the temporary `[NvrtcGeneigSolver] phase timings` printf
  instrumentation is deferred until 3c, where it'll be replaced by
  a structured introspection method on the solver class.
