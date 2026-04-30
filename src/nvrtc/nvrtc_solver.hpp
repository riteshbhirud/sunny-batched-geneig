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
#include <vector_types.h>  // dim3

class NvrtcGeneigSolver {
public:
    // Construct solver for matrix size `n` on `device_id`. If
    // `batches_per_block_request <= 0`, auto-select the largest BPB that
    // fits the device's shared-memory ceiling. Otherwise honor the explicit
    // value (and throw if it doesn't fit).
    explicit NvrtcGeneigSolver(int n,
                               int device_id,
                               int batches_per_block_request = 0);
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

    int          matrix_size()       const { return n_; }
    int          batches_per_block() const { return batches_per_block_; }
    dim3         block_dim()         const { return block_dim_; }
    unsigned int shared_mem_bytes()  const { return shared_mem_bytes_; }

private:
    // Build a CUmodule for (n, bpb). Throws on failure. Loads `kernel_`,
    // `block_dim_`, `shared_mem_bytes_` from the resulting module.
    void compile_for_(int bpb);

    int          n_                  = 0;
    int          device_id_          = 0;
    int          batches_per_block_  = 0;
    int          device_max_smem_    = 0;
    CUcontext    context_            = nullptr;
    CUmodule     module_             = nullptr;
    CUfunction   kernel_             = nullptr;
    dim3         block_dim_          {0, 0, 0};
    unsigned int shared_mem_bytes_   = 0;
};
