// Phase 3.5a — runtime-compiled generalized eigenvalue solver.
//
// Constructor compiles the unified five-stage NVRTC kernel for the supplied
// matrix size N and a chosen BatchesPerBlock (BPB). BPB defaults to auto-
// select: try BPB ∈ {16, 8, 4, 2, 1} from largest to smallest, accept the
// first whose required shared memory fits the device's per-block dynamic
// shared-memory ceiling (with a 2 KB safety margin). Pass an explicit
// `batches_per_block > 0` to override.
//
// Caller contract: when launching, `batch_size` should be a multiple of
// `batches_per_block()` — the kernel does not bounds-check the partial last
// block.

#pragma once

#include <cuda.h>
#include <map>
#include <tuple>
#include <vector>
#include <vector_types.h>  // dim3

// Phase 3.5d helpers — create/destroy a vector of CUstreams with default
// flags. These do not depend on solver state (free functions); the test
// harness uses them so it doesn't have to repeat the cuStreamCreate/Destroy
// loop.
std::vector<CUstream> make_streams(int n);
void                  destroy_streams(std::vector<CUstream>& streams);

// Phase 3.5e — BlockDim selection strategy.
//   Suggested: use cuSolverDx's `suggested_block_dim` (probe-driven).
//   Autotune:  sweep candidate BlockDim values around `suggested`, measure
//              each on a 256-matrix tuning workload (5-sample median),
//              validate output against the suggested-candidate reference,
//              pick the empirically fastest correctness-validated value.
//              Process-static cache memoizes the result per (N, BPB, arch).
enum class TuningMode { Suggested, Autotune };

class NvrtcGeneigSolver {
public:
    // Construct solver for matrix size `n` on `device_id`. If
    // `batches_per_block_request <= 0`, auto-select the largest BPB that
    // fits the device's shared-memory ceiling. Otherwise honor the explicit
    // value (and throw if it doesn't fit).
    //
    // `force_block_dim_x > 0` is a testing-only escape hatch that skips both
    // the probe and any autotune sweep, compiling the production kernel
    // directly with that BlockDim.x. Use 0 (default) to let the probe (or
    // autotune) decide.
    //
    // `mode` selects how BlockDim.x is chosen when `force_block_dim_x == 0`:
    // Suggested (default, 3.5b path) reads cuSolverDx's recommendation;
    // Autotune (3.5e path) sweeps candidates around the recommendation and
    // benchmarks them.
    explicit NvrtcGeneigSolver(int n,
                               int device_id,
                               int batches_per_block_request = 0,
                               int force_block_dim_x = 0,
                               TuningMode mode = TuningMode::Suggested);
    ~NvrtcGeneigSolver();

    NvrtcGeneigSolver(const NvrtcGeneigSolver&)            = delete;
    NvrtcGeneigSolver& operator=(const NvrtcGeneigSolver&) = delete;
    NvrtcGeneigSolver(NvrtcGeneigSolver&&)                 = delete;
    NvrtcGeneigSolver& operator=(NvrtcGeneigSolver&&)      = delete;

    // Full generalised eigenvalue solve. `batch_size` is the total number
    // of matrices; the launcher computes gridDim = (batch_size + BPB - 1)/BPB.
    void launch(CUdeviceptr d_H, CUdeviceptr d_S,
                CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
                int batch_size, CUstream stream);

    // Phase 3.5d — multi-stream chunked launcher. Splits `total_matrices`
    // into chunks of `chunk_size` and round-robins them across `streams`.
    // No synchronization between chunks; the CUDA scheduler is free to
    // overlap them. Caller is responsible for synchronizing all streams
    // before reading results.
    //
    // Pointer offsets into d_H, d_S, d_U: chunk_idx * chunk_size * N² elts.
    // Pointer offsets into d_W: chunk_idx * chunk_size * N elts.
    // Pointer offsets into d_info: chunk_idx * chunk_size elts.
    //
    // Each chunk's batch_size is `chunk_size`, except the final chunk which
    // may be smaller (`total_matrices % chunk_size`). All chunk sizes must be
    // multiples of `batches_per_block()`; this is enforced via the same
    // contract as `launch`.
    void launch_chunked(CUdeviceptr d_H, CUdeviceptr d_S,
                        CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
                        int total_matrices, int chunk_size, int num_streams,
                        const std::vector<CUstream>& streams);

    int          matrix_size()           const { return n_; }
    int          batches_per_block()     const { return batches_per_block_; }
    dim3         block_dim()             const { return block_dim_; }
    unsigned int shared_mem_bytes()      const { return shared_mem_bytes_; }

    // Phase 3.5e introspection. `was_autotuned()` is true iff the solver was
    // built with TuningMode::Autotune (whether it ran the sweep or hit the
    // process-static cache). `autotuned_block_dim_x()` returns the chosen
    // BlockDim.x (same as `block_dim().x` after construction; provided for
    // semantic clarity in benchmarks). `autotune_sweep_ran()` is true iff
    // the sweep actually ran (cache miss); false on cache hit.
    int          autotuned_block_dim_x() const { return autotuned_block_dim_x_; }
    bool         was_autotuned()         const { return was_autotuned_; }
    bool         autotune_sweep_ran()    const { return autotune_sweep_ran_; }

private:
    // Probe-compile a small kernel that exposes per-operator
    // `suggested_block_dim` via `__constant__`. Returns the maximum .x value
    // across the five operators. Throws on NVRTC compile failure (used by
    // the auto-select loop to skip BPB candidates that don't satisfy
    // cuSolverDx's static_assert).
    unsigned int probe_suggested_block_dim_x_(int bpb);

    // Build the production CUmodule for (n, bpb, block_dim_x). Throws on
    // failure. Loads `kernel_`, `block_dim_`, `shared_mem_bytes_` from the
    // resulting module.
    void compile_production_(int bpb, unsigned int block_dim_x);

    // Phase 3.5e — empirical BlockDim sweep. Compiles each candidate around
    // `suggested`, runs a 5-sample median throughput benchmark on a 256-
    // matrix synthetic workload, validates output against the suggested-
    // candidate reference (eigenvalues + S-inner-product phase), and picks
    // the fastest correctness-validated candidate. Returns the winner's
    // BlockDim.x. After this call, `module_` may point to any candidate;
    // caller is responsible for recompiling with the winner.
    unsigned int autotune_block_dim_x_(int bpb, unsigned int suggested);

    int          n_                      = 0;
    int          device_id_              = 0;
    int          batches_per_block_      = 0;
    int          device_max_smem_        = 0;
    CUcontext    context_                = nullptr;
    CUmodule     module_                 = nullptr;
    CUfunction   kernel_                 = nullptr;
    dim3         block_dim_              {0, 0, 0};
    unsigned int shared_mem_bytes_       = 0;

    // 3.5e introspection.
    int          autotuned_block_dim_x_  = 0;
    bool         was_autotuned_          = false;
    bool         autotune_sweep_ran_     = false;

    // Process-static autotune cache. Keyed on (N, BPB, arch). On cache hit
    // we skip the multi-compile sweep and reuse the previously-found winner.
    static std::map<std::tuple<int, int, int>, int> autotune_cache_;
};
