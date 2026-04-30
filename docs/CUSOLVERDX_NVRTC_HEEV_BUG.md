# cuSolverDx 25.12 + NVRTC: `function::heev` instantiation fails

## Summary

When the `block_execution<...>` specialization for `Function<function::heev>` is
instantiated under NVRTC (CUDA 12.8), the static-constexpr member
`workspace_size`'s initializer references an unqualified identifier
`max_threads_per_block` that is declared *later* in the same class body.
NVRTC reports `max_threads_per_block` as undefined; downstream
`heev::suggested_batches` and `heev::suggested_block_dim` template
substitutions then fail with `<expression>` markers because the class is in
an error state.

The same code compiles cleanly under nvcc, where complete-class-context
name lookup for member function bodies makes order within the class body
irrelevant. NVRTC's stricter lookup is exposed by this declaration order.

`gesvd` is affected by the same code path; `potrf`, `trsm`, all the QR/LU
variants, etc. return 0 from `get_workspace_size()` and therefore never
trigger the lookup.

## Verified raw header text

`include/cusolverdx/detail/solver_execution.hpp` lines 197-205 — the
`get_workspace_size()` implementation:

```cpp
__host__ __device__ __forceinline__ static constexpr int get_workspace_size() {
    if constexpr (function_of_v<this_type> == function::heev) {
        return heev::workspace_size<a_cuda_data_type, base_type::m_size, job, max_threads_per_block, batches_per_block, base_type::this_sm_v>();
    } else if constexpr (function_of_v<this_type> == function::gesvd) {
        return gesvd::workspace_size<a_cuda_data_type, base_type::m_size, base_type::n_size, max_threads_per_block, batches_per_block, base_type::this_sm_v>();
    } else {
        return 0;
    }
}
```

Lines 685-694 — the static-constexpr declaration block of
`block_execution<Operators...>`:

```cpp
static constexpr unsigned int suggested_batches_per_block = get_suggested_batches_per_block();

static constexpr dim3 suggested_block_dim = get_suggested_block_dim();
static constexpr dim3 block_dim           = get_block_dim();

static constexpr unsigned int workspace_size = get_workspace_size();
static constexpr unsigned int shared_memory_size = get_shared_memory_size();

static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
static constexpr unsigned int min_blocks_per_multiprocessor = 1;
```

`workspace_size` (line 690) is initialized via `get_workspace_size()`, which
references `max_threads_per_block` (declared three lines below at line 693).

## Minimal repros (in this repo)

[`tests/repro_heev_nvrtc.cpp`](../tests/repro_heev_nvrtc.cpp) and
[`tests/repro_potrf_nvrtc.cpp`](../tests/repro_potrf_nvrtc.cpp) are
deliberately identical except for the `Function<>` operator. Both run
NVRTC with the standard cuSolverDx option set
(`--std=c++17 --device-as-default-execution-space -dlto
--relocatable-device-code=true --gpu-architecture=sm_NN` plus `--include-path`
flags for cuSolverDx, CUTLASS, and the CUDA toolkit). The kernel sources
do nothing but instantiate the `Solver` type and read
`Solver::shared_memory_size` to force class instantiation; no
`execute()` is called.

| Repro | Function operator | NVRTC result | Exit |
|---|---|---|---|
| `repro_heev_nvrtc`  | `Function<function::heev>()`  | **FAILED** with 6 errors (3 distinct, each emitted twice) | 1 |
| `repro_potrf_nvrtc` | `Function<function::potrf>()` | **SUCCEEDED**                                              | 0 |

The NVRTC log from the heev repro reports the three error sites with the
identical pattern observed in the project's main NVRTC kernel:
`solver_execution.hpp:143` (`heev::suggested_batches` no instance match),
`:177` (`heev::suggested_block_dim` no instance match), and `:199`
(`identifier "max_threads_per_block" is undefined`). The `:199` error
is the proximate cause; the other two are downstream cascades.

## Suggested upstream fix

Reorder the static-constexpr declarations of `block_execution` so
`max_threads_per_block` precedes `workspace_size`:

```diff
             static constexpr dim3 block_dim           = get_block_dim();

-            static constexpr unsigned int workspace_size = get_workspace_size();
-            static constexpr unsigned int shared_memory_size = get_shared_memory_size();
-
             static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
+
+            static constexpr unsigned int workspace_size = get_workspace_size();
+            static constexpr unsigned int shared_memory_size = get_shared_memory_size();
+
             static constexpr unsigned int min_blocks_per_multiprocessor = 1;
```

This is a mechanical reorder, no semantic change. Under nvcc behaviour is
unchanged; under NVRTC the heev/gesvd `get_workspace_size()` initializer
can resolve `max_threads_per_block` because it is now declared lexically
earlier.

## Project workaround

Phase 3a uses a hybrid two-kernel pipeline instead of patching the cuSolverDx
headers:

- **NVRTC kernel** (`geneig_reduce_kernel`, runtime-compiled, will be
  templated on N in Phase 3b): runs `potrf(S) → trsm(L⁻¹·H) → trsm(·L⁻ᴴ)`,
  produces L and M = L⁻¹ H L⁻ᴴ. Uses only `Function<function::potrf>` and
  `Function<function::trsm>` operators, none of which trigger the bug.
- **Static N=54 kernel** (`geneig_finish_n54_kernel`, statically compiled
  by nvcc): consumes L and M from the NVRTC kernel, runs
  `heev(M) → trsm(L⁻ᴴ·V)`, produces eigenvalues Λ and generalised
  eigenvectors U. Uses `Function<function::heev>` and
  `Function<function::trsm>` — heev compiles fine here because nvcc tolerates
  the declaration order.

Data is GPU-resident between the two kernels. The split adds two extra
device-memory tiles (L and M, 54×54 cdouble each = ~46 KB each) but no
extra host↔device traffic.

When cuSolverDx ships a fix, the hybrid can collapse back to a single
NVRTC kernel — the kernel-source surgery is local to
[`src/nvrtc/kernel_source.hpp`](../src/nvrtc/kernel_source.hpp) and
[`src/nvrtc/nvrtc_solver.cpp`](../src/nvrtc/nvrtc_solver.cpp).
