# Phase 3: NVRTC runtime compilation

## Goal

Remove the N=54 hardcode from the kernel so Sunny.jl can use any
matrix size determined at runtime by the user's magnetic unit cell.

## Sub-step plan

- **3a (done, commit a64a862):** NVRTC plumbing for the reduction
  stage; hybrid with static heev finishing stage to work around
  CUSOLVERDX_NVRTC_HEEV_BUG. N=54 still hardcoded via `-DM_SIZE=54`.
- **3b (next):** Template the kernel source on M_SIZE. Generate
  LAPACK reference fixtures at multiple sizes (32, 48, 54, 64, 96,
  128). Validate at each.
- **3c:** Per-size CUmodule cache inside `NvrtcGeneigSolver`.
  Construct once per N, reuse across all matrices.
- **3d:** Disk-backed cubin cache, keyed on (M_SIZE, arch, library
  version hash). Solves the cold-start problem across Julia sessions.
- **3e:** Performance characterization vs the static N=54 path,
  plus end-to-end timing on the existing batched test.

## Notes

- The cold NVRTC + nvJitLink pipeline takes ~75s for one (N, arch)
  pair. This is acceptable for Sunny's usage pattern (one-time
  cost per session, amortized over hundreds of thousands of
  matrix solves) but motivates the disk cache in 3d. See the
  timing breakdown in CUSOLVERDX_NVRTC_HEEV_BUG.md.
- Removing the temporary `[NvrtcGeneigSolver] phase timings` printf
  instrumentation is deferred until 3c, where it'll be replaced by
  a structured introspection method on the solver class.
