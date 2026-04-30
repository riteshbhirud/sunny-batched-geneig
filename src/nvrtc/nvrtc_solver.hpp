// Phase 3b — runtime-compiled generalized eigenvalue solver, templated on N.
//
// Compiles the unified five-stage kernel in src/nvrtc/kernel_source.hpp via
// NVRTC + nvJitLink at construction time and exposes `launch()` which
// produces eigenvalues W and generalized eigenvectors U directly.
//
// The cuSolverDx 25.12 NVRTC + heev declaration-order bug is worked around
// via a header overlay (see src/nvrtc/cusolverdx_overlay/ and
// docs/CUSOLVERDX_NVRTC_HEEV_BUG.md). With the overlay in place, all five
// solver primitives (potrf, trsm×3, heev) instantiate cleanly under NVRTC.
//
// On any compilation/link failure the constructor throws std::runtime_error
// with the NVRTC or nvJitLink error log included.

#pragma once

#include <cuda.h>
#include <vector_types.h>  // dim3

class NvrtcGeneigSolver {
public:
    // Compile the unified kernel for matrix size `n` on `device_id`.
    explicit NvrtcGeneigSolver(int n, int device_id);
    ~NvrtcGeneigSolver();

    NvrtcGeneigSolver(const NvrtcGeneigSolver&)            = delete;
    NvrtcGeneigSolver& operator=(const NvrtcGeneigSolver&) = delete;
    NvrtcGeneigSolver(NvrtcGeneigSolver&&)                 = delete;
    NvrtcGeneigSolver& operator=(NvrtcGeneigSolver&&)      = delete;

    // Full generalised eigenvalue solve. batch_size == 1 launches one block;
    // > 1 launches a grid (one block per matrix). batch_size <= 0 is a no-op.
    void launch(CUdeviceptr d_H, CUdeviceptr d_S,
                CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
                int batch_size, CUstream stream);

    int          matrix_size()      const { return n_; }
    dim3         block_dim()        const { return block_dim_; }
    unsigned int shared_mem_bytes() const { return shared_mem_bytes_; }

private:
    int          n_                = 0;
    int          device_id_        = 0;
    CUcontext    context_          = nullptr;
    CUmodule     module_           = nullptr;
    CUfunction   kernel_           = nullptr;
    dim3         block_dim_        {0, 0, 0};
    unsigned int shared_mem_bytes_ = 0;
};
