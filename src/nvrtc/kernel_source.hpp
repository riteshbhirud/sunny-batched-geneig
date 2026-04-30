// Phase 3b — unified five-stage NVRTC kernel for the generalized eigenvalue
// solve at runtime-templated N.
//
//   S = L L^H   (potrf)
//   M = L^{-1} H L^{-H}   (two trsms)
//   M V = V diag(W)   (heev)
//   U = L^{-H} V   (back-trsm)
//
// The unified pipeline is enabled by the cuSolverDx header overlay at
// src/nvrtc/cusolverdx_overlay/, which patches the heev/gesvd
// `get_workspace_size()` declaration order so NVRTC can instantiate the
// `Function<function::heev>` specialization. See
// docs/CUSOLVERDX_NVRTC_HEEV_BUG.md.
//
// Macros filled in at NVRTC compile time:
//   M_SIZE       — matrix dimension (driven by NvrtcGeneigSolver::n)
//   SOLVER_LDA   — leading dimension (= M_SIZE)
//   SOLVER_SM    — cuSolverDx tuning-database tag (e.g. 800). NOT the
//                  runtime architecture; --gpu-architecture controls that.
//
// Kernel exposes:
//   solver_block_dim          (dim3)            — `__constant__`
//   solver_shared_memory_size (unsigned int)    — `__constant__`
//
// Kernel entry: `geneig_full_kernel`.

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

using TrsmLeftConj = decltype(Size<M_SIZE, M_SIZE>()
                            + Precision<double>()
                            + Type<type::complex>()
                            + Function<function::trsm>()
                            + Side<side::left>()
                            + FillMode<lower>()
                            + TransposeMode<conj_trans>()
                            + Diag<diag::non_unit>()
                            + Arrangement<col_major, col_major>()
                            + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                            + SM<SOLVER_SM>()
                            + Block()
                            + BlockDim<128>());

using Heev = decltype(Size<M_SIZE>()
                    + Precision<double>()
                    + Type<type::complex>()
                    + Function<function::heev>()
                    + FillMode<lower>()
                    + Arrangement<col_major>()
                    + LeadingDimension<SOLVER_LDA>()
                    + Job<job::overwrite_vectors>()
                    + SM<SOLVER_SM>()
                    + Block()
                    + BlockDim<128>());

using DType      = typename Cholesky::a_data_type;
using StatusType = typename Cholesky::status_type;
using PType      = typename Heev::a_precision;

constexpr int kN   = M_SIZE;
constexpr int kLDA = SOLVER_LDA;
constexpr unsigned int kFullSmem =
    Cholesky::shared_memory_size + Heev::shared_memory_size;

__constant__ dim3         solver_block_dim          = Cholesky::block_dim;
__constant__ unsigned int solver_shared_memory_size = kFullSmem;

extern "C" __global__ __launch_bounds__(Cholesky::max_threads_per_block)
void geneig_full_kernel(const cuDoubleComplex* __restrict__ H_in,
                        const cuDoubleComplex* __restrict__ S_in,
                        double*                __restrict__ W_out,
                        cuDoubleComplex*       __restrict__ U_out,
                        int*                   __restrict__ info_out,
                        int                                 batch_size) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const unsigned long long mat_stride = (unsigned long long)kN * kLDA;
    const unsigned long long w_stride   = (unsigned long long)kN;

    const cuDoubleComplex* H_b   = H_in   + mat_stride * b;
    const cuDoubleComplex* S_b   = S_in   + mat_stride * b;
    cuDoubleComplex*       U_b   = U_out  + mat_stride * b;
    double*                W_b   = W_out  + w_stride   * b;
    int*                   info_b = info_out + b;

    extern __shared__ __align__(16) unsigned char smem_raw[];

    constexpr unsigned int kBsOffset       = Cholesky::shared_memory_size;
    constexpr unsigned int kLambdaOffset   = kBsOffset + sizeof(DType) * kN * kLDA;
    constexpr unsigned int kLambdaBytes    = sizeof(PType) * kN;
    constexpr unsigned int kWorkspaceOffset =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    DType* Bs           = reinterpret_cast<DType*>(smem_raw + kBsOffset);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOffset);

    const DType* S_typed = reinterpret_cast<const DType*>(S_b);
    const DType* H_typed = reinterpret_cast<const DType*>(H_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky().execute(As, reinterpret_cast<StatusType*>(info_b));
    __syncthreads();
    TrsmLeft().execute(As, kLDA, Bs, kLDA);                   // Bs ← L^{-1} H
    __syncthreads();
    TrsmRight().execute(As, kLDA, Bs, kLDA);                  // Bs ← Bs L^{-H} = M
    __syncthreads();
    Heev().execute(Bs, kLDA, lambda_s, workspace_s,
                   reinterpret_cast<StatusType*>(info_b));    // Bs = V (eigvecs of M)
    __syncthreads();
    TrsmLeftConj().execute(As, kLDA, Bs, kLDA);              // Bs ← L^{-H} V = U
    __syncthreads();

    for (int i = threadIdx.x; i < kN; i += blockDim.x)
        W_b[i] = lambda_s[i];
    DType* U_typed = reinterpret_cast<DType*>(U_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x)
        U_typed[idx] = Bs[idx];
}
)kernel";

}  // namespace nvrtc_geneig
