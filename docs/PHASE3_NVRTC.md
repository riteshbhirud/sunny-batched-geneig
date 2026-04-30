# Phase 3: NVRTC runtime compilation

## Goal

Remove the N=54 hardcode from the kernel so Sunny.jl can use any
matrix size determined at runtime by the user's magnetic unit cell.

## Sub-step plan

- **3a (done, commit a64a862):** NVRTC plumbing for the reduction
  stage; hybrid with static heev finishing stage to work around
  CUSOLVERDX_NVRTC_HEEV_BUG. N=54 still hardcoded via `-DM_SIZE=54`.
- **3b (done, this commit):** cuSolverDx header overlay applied
  (`src/nvrtc/cusolverdx_overlay/`) — heev now compiles under NVRTC.
  Hybrid split removed; the NVRTC kernel runs all five stages
  (`potrf → trsm × 2 → heev → trsm`) in one launch. Constructor
  signature is now `NvrtcGeneigSolver(int n, int device_id)`. Fixtures
  generated at N ∈ {16, 32, 48, 54, 64, 96, 128}; per-N validation in
  `tests/test_nvrtc_multiN.cpp`.
- **3c:** Per-size CUmodule cache inside `NvrtcGeneigSolver`.
  Construct once per N, reuse across all matrices.
- **3d:** Disk-backed cubin cache, keyed on (M_SIZE, arch, library
  version hash). Solves the cold-start problem across Julia sessions.
- **3e:** Performance characterization vs the static N=54 path,
  plus end-to-end timing on the existing batched test.

## 3b multi-N validation outcome (sm_120 RTX 5070 Ti Laptop, 101,376 bytes max dynamic shared memory per block)

| N   | shared_mem | cold compile (ms) | rand max_eig_rel | rand max_phase | near max_phase | status |
|-----|-----------:|------------------:|------------------:|----------------:|---------------:|--------|
|  16 |      9,840 |             5,146 |          1.61e-15 |        2.11e-15 |       1.44e-15 | PASS   |
|  32 |     36,080 |             8,575 |          1.78e-15 |        2.33e-15 |       2.66e-15 | PASS   |
|  48 |     78,704 |            19,302 |          1.50e-15 |        2.55e-15 |       2.33e-15 | PASS   |
|  54 |     99,712 |           213,297 |          2.16e-15 |        3.77e-15 |       2.44e-15 | PASS   |
|  64 |          — |                 — |                 — |               — |              — | SKIP   |
|  96 |          — |                 — |                 — |               — |              — | SKIP   |
| 128 |          — |                 — |                 — |               — |              — | SKIP   |

**Summary:** 8/8 (N, fixture) pairs passed at fp64 epsilon for
N ∈ {16, 32, 48, 54}. N ∈ {64, 96, 128} skipped due to per-block
shared-memory ceiling on the development hardware.

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
