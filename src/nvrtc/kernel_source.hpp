// Phase 3a — NVRTC kernel source for the REDUCE half of the generalized
// eigenvalue solve: S = L L^H, M = L^{-1} H L^{-H}. Outputs both L (lower
// triangle, strict upper zeroed) and M to global memory.
//
// The eigendecomposition + back-transform half lives in a separate, statically
// compiled kernel (geneig_finish_n54_kernel) because cuSolverDx 25.12's heev
// path fails to compile under NVRTC. See docs/CUSOLVERDX_NVRTC_HEEV_BUG.md.
//
// Macros filled in at NVRTC compile time:
//   M_SIZE       — matrix dimension (54 in 3a; runtime in 3b)
//   SOLVER_LDA   — leading dimension (54)
//   SOLVER_SM    — cuSolverDx tuning-database tag (e.g. 800). NOT the
//                  runtime architecture; --gpu-architecture controls that.
//
// Kernel exposes:
//   solver_block_dim          (dim3)            — `__constant__`
//   solver_shared_memory_size (unsigned int)    — `__constant__`
//
// Kernel entry: `geneig_reduce_kernel`.

#pragma once

namespace nvrtc_geneig {

inline constexpr const char* kKernelSource = R"kernel(
#include <cusolverdx.hpp>

using namespace cusolverdx;

// Operator-ordering note: under NVRTC, SM<> must come BEFORE Block() in the
// description. See cusolverdx/detail/solver_description_arithmetic.hpp.

using Cholesky = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>()
                        + Type<type::complex>()
                        + Function<function::potrf>()
                        + FillMode<lower>()
                        + Arrangement<col_major>()
                        + LeadingDimension<SOLVER_LDA>()
                        + SM<SOLVER_SM>()
                        + Block()
                        + BlockDim<128>());

using TrsmLeft = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>()
                        + Type<type::complex>()
                        + Function<function::trsm>()
                        + Side<side::left>()
                        + FillMode<lower>()
                        + TransposeMode<non_trans>()
                        + Diag<diag::non_unit>()
                        + Arrangement<col_major, col_major>()
                        + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                        + SM<SOLVER_SM>()
                        + Block()
                        + BlockDim<128>());

using TrsmRight = decltype(Size<M_SIZE, M_SIZE>()
                         + Precision<double>()
                         + Type<type::complex>()
                         + Function<function::trsm>()
                         + Side<side::right>()
                         + FillMode<lower>()
                         + TransposeMode<conj_trans>()
                         + Diag<diag::non_unit>()
                         + Arrangement<col_major, col_major>()
                         + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                         + SM<SOLVER_SM>()
                         + Block()
                         + BlockDim<128>());

using DType      = typename Cholesky::a_data_type;
using StatusType = typename Cholesky::status_type;

constexpr int kN   = M_SIZE;
constexpr int kLDA = SOLVER_LDA;
// Just two matrix tiles in shared memory — no eigendecomposition workspace.
constexpr unsigned int kReduceSmem =
    Cholesky::shared_memory_size + Cholesky::shared_memory_size;

__constant__ dim3         solver_block_dim          = Cholesky::block_dim;
__constant__ unsigned int solver_shared_memory_size = kReduceSmem;

extern "C" __global__ __launch_bounds__(Cholesky::max_threads_per_block)
void geneig_reduce_kernel(const cuDoubleComplex* __restrict__ H_in,
                          const cuDoubleComplex* __restrict__ S_in,
                          cuDoubleComplex*       __restrict__ L_out,
                          cuDoubleComplex*       __restrict__ M_out,
                          int*                   __restrict__ info_out,
                          int                                 batch_size) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const unsigned long long mat_stride = (unsigned long long)kN * kLDA;

    const cuDoubleComplex* H_b   = H_in    + mat_stride * b;
    const cuDoubleComplex* S_b   = S_in    + mat_stride * b;
    cuDoubleComplex*       L_b   = L_out   + mat_stride * b;
    cuDoubleComplex*       M_b   = M_out   + mat_stride * b;
    int*                   info_b = info_out + b;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    DType* As = reinterpret_cast<DType*>(smem_raw);                                  // S → L
    DType* Bs = reinterpret_cast<DType*>(smem_raw + Cholesky::shared_memory_size);   // H → M

    const DType* S_typed = reinterpret_cast<const DType*>(S_b);
    const DType* H_typed = reinterpret_cast<const DType*>(H_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky().execute(As, reinterpret_cast<StatusType*>(info_b));
    __syncthreads();
    TrsmLeft().execute(As, kLDA, Bs, kLDA);            // Bs ← L^{-1} H
    __syncthreads();
    TrsmRight().execute(As, kLDA, Bs, kLDA);           // Bs ← Bs L^{-H} = M
    __syncthreads();

    // Writeback. L: lower triangle of As, strict upper zeroed.
    DType* L_typed = reinterpret_cast<DType*>(L_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        int col = idx / kLDA;
        int row = idx % kLDA;
        if (row >= col) L_typed[idx] = As[idx];
        else            L_typed[idx] = DType{};
    }
    // M: full Hermitian matrix (the finishing kernel's heev only reads the
    // lower triangle per FillMode<lower>, but we write all of Bs for clarity).
    DType* M_typed = reinterpret_cast<DType*>(M_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        M_typed[idx] = Bs[idx];
    }
}
)kernel";

}  // namespace nvrtc_geneig
