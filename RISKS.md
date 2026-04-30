# Project risks

This document tracks risks that could substantively undermine the
demonstration. It does not track polish items, performance cleanups
that have known fix paths, or other Type-B issues — those go in the
relevant phase docs.

## Open

### R-1: Performance vs Sunny.jl PR #8 on H100 (CRITICAL)

If the hybrid kernel is more than ~2× slower than Steven Hahn's PR #8
batched cuSOLVER path on H100 NVL at SW08's 360,000-q-point workload,
the speedup story is undermined regardless of how well-engineered
the rest is.

Mitigation plan: Phase 6 H100 measurement against PR #8 baseline.
If gap is significant, Phase 6 includes design revision (occupancy
tuning, alternative fusion strategy, batched dispatch tuning).

Status: NOT MEASURED YET. Phase 6 deliverable.

### R-2: Sunny end-to-end integration

If the Julia FFI (Phase 7) doesn't produce a complete `SW08_sqrt3_kagome_AFM`
run matching Sunny's reference scattering intensity output, the
demonstration is incomplete. The kernel correctness alone is not
sufficient — it must drop into Sunny's actual workflow.

Mitigation plan: Phase 7 implementation includes scattering-intensity
voxel-by-voxel comparison against Sunny's CPU reference path. Pass
criterion: relative error < 1e-10 per voxel.

Status: NOT IMPLEMENTED YET. Phase 7 deliverable.

### R-3: Multi-N validation

The kernel currently validates only at N=54 (Sunny's SW08 magnetic cell).
If it fails at sizes above ~128 (shared memory ceiling) or if the
NVRTC compile fails for some N values, the runtime-N pitch is
incomplete.

Mitigation plan: Phase 3b validates at N ∈ {32, 48, 54, 64, 96, 128}.
Phase 6 sweep extends to higher N. We document the maximum N at which
the single-block strategy works and the rationale for the ceiling.

Status: **PARTIAL** (Phase 3b run). On the development hardware (RTX
5070 Ti Laptop, sm_120, **101 KB max dynamic shared memory per block**),
the single-block unified kernel works at fp64 epsilon for N ∈
{16, 32, 48, 54} (4 sizes × 2 fixtures = 8/8 pairs PASS). N ∈
{64, 96, 128} fail because
`Cholesky::shared_memory_size + Heev::shared_memory_size` exceeds the
device's 101 KB per-block cap. cuSolverDx's
`static_assert "doesn't fit in shared memory"` catches the failure at
NVRTC compile time on this hardware. H100 (228 KB per-block opt-in)
should extend the single-block ceiling to roughly N ≈ 80; beyond
that, multi-block strategies are needed and are out of Phase 3b
scope. See `docs/PHASE3_NVRTC.md` for the full table of validated
sizes and per-N timings. Sunny's SW08 (N=54) is the operative size
and is fully validated.

### R-4: Numerical correctness on physical Sunny matrices

Random Hermitian-definite and near-identity-S fixtures cover the
generic numerical regime. They do not cover Goldstone-shifted matrices,
which is what Sunny actually generates. If our kernel handles random
matrices but fails on the specific structure Sunny produces, the
correctness claim is incomplete.

Mitigation plan: Phase 5 includes dumping (H, S) pairs from a real
SW08 run, plus the Goldstone-shifted construction explicitly tested
in our fixture, plus comparison against Sunny's CPU reference path
for the same matrices.

Status: PARTIAL. Near-identity fixture covers part of the regime.
Phase 5 closes the rest.

## Closed

(none yet)

## Notes

This document evolves with the project. Each phase closure should
review whether the phase mitigated any open risks and update the
status accordingly. New risks discovered during implementation should
be added here, not buried in commit messages.
