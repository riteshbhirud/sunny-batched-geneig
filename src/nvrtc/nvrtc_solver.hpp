// Phase 3a — runtime-compiled REDUCE half of the generalized eigenvalue
// solver. NVRTC compiles the kernel source in src/nvrtc/kernel_source.hpp at
// construction time and exposes `launch_reduce()` which produces L (Cholesky
// factor of S) and M (= L^{-1} H L^{-H}). The eigendecomposition + back-
// transform half is provided by a statically-compiled finishing kernel
// (see geneig_finish_n54_launch in src/geneig_fixed.cu).
//
// Why the split: cuSolverDx 25.12's heev fails to compile under NVRTC due to
// a header declaration-order issue. See docs/CUSOLVERDX_NVRTC_HEEV_BUG.md.
//
// On any compilation/link failure the constructor throws std::runtime_error
// with the NVRTC or nvJitLink error log included.

#pragma once

#include <cuda.h>
#include <vector_types.h>  // dim3

class NvrtcGeneigSolver {
public:
    explicit NvrtcGeneigSolver(int device_id);
    ~NvrtcGeneigSolver();

    NvrtcGeneigSolver(const NvrtcGeneigSolver&)            = delete;
    NvrtcGeneigSolver& operator=(const NvrtcGeneigSolver&) = delete;
    NvrtcGeneigSolver(NvrtcGeneigSolver&&)                 = delete;
    NvrtcGeneigSolver& operator=(NvrtcGeneigSolver&&)      = delete;

    // Reduce stage: writes L (Cholesky factor of S, lower triangular with
    // strict upper zeroed) and M (= L^{-1} H L^{-H}) to the supplied device
    // buffers. batch_size == 1 launches one block; > 1 launches a grid (one
    // block per matrix). batch_size <= 0 is a no-op.
    void launch_reduce(CUdeviceptr d_H, CUdeviceptr d_S,
                       CUdeviceptr d_L, CUdeviceptr d_M, CUdeviceptr d_info,
                       int batch_size, CUstream stream);

    dim3         block_dim()        const { return block_dim_; }
    unsigned int shared_mem_bytes() const { return shared_mem_bytes_; }

private:
    int          device_id_        = 0;
    CUcontext    context_          = nullptr;
    CUmodule     module_           = nullptr;
    CUfunction   kernel_           = nullptr;
    dim3         block_dim_        {0, 0, 0};
    unsigned int shared_mem_bytes_ = 0;
};
