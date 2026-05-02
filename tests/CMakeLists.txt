add_executable(lapack_reference lapack_reference.cpp)
target_link_libraries(lapack_reference PRIVATE LAPACK::lapacke)

add_executable(dump_pair dump_pair.cpp)
target_link_libraries(dump_pair PRIVATE LAPACK::lapacke)

add_executable(test_chol_n54 test_chol_n54.cpp)
target_link_libraries(test_chol_n54
    PRIVATE
        geneig_fixed
        LAPACK::lapacke
        CUDA::cudart
)
set_target_properties(test_chol_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_reduce_n54 test_reduce_n54.cpp)
target_link_libraries(test_reduce_n54
    PRIVATE
        geneig_fixed
        LAPACK::lapacke
        CUDA::cudart
)
set_target_properties(test_reduce_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_reduce_eig_n54 test_reduce_eig_n54.cpp)
target_link_libraries(test_reduce_eig_n54
    PRIVATE
        geneig_fixed
        LAPACK::lapacke
        CUDA::cudart
)
set_target_properties(test_reduce_eig_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_full_n54 test_full_n54.cpp)
target_link_libraries(test_full_n54
    PRIVATE
        geneig_fixed
        CUDA::cudart
)
set_target_properties(test_full_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_full_batched_n54 test_full_batched_n54.cpp)
target_link_libraries(test_full_batched_n54
    PRIVATE
        geneig_fixed
        LAPACK::lapacke
        CUDA::cudart
)
set_target_properties(test_full_batched_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

# Phase 3b regression — single-N=54 unified-kernel correctness test.
add_executable(test_nvrtc_n54 test_nvrtc_n54.cpp)
target_link_libraries(test_nvrtc_n54
    PRIVATE
        nvrtc_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# Phase 3b — multi-N validation of the unified NVRTC kernel.
add_executable(test_nvrtc_multiN test_nvrtc_multiN.cpp)
target_link_libraries(test_nvrtc_multiN
    PRIVATE
        nvrtc_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# Phase 3.5a — BatchesPerBlock sweep at N=54.
add_executable(test_bpb_sweep_n54 test_bpb_sweep_n54.cpp)
target_link_libraries(test_bpb_sweep_n54
    PRIVATE
        nvrtc_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# Phase 3.5f — module cache benchmark and corruption recovery test.
add_executable(test_cache_n54 test_cache_n54.cpp)
target_link_libraries(test_cache_n54
    PRIVATE
        nvrtc_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

add_executable(test_cache_invalidation test_cache_invalidation.cpp)
target_link_libraries(test_cache_invalidation
    PRIVATE
        nvrtc_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# Phase 3.5e — autotune benchmarks at N=54 BPB=1 and N=32 BPB=2.
add_executable(test_autotune_n54 test_autotune_n54.cpp)
target_link_libraries(test_autotune_n54
    PRIVATE
        nvrtc_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

add_executable(test_autotune_n32 test_autotune_n32.cpp)
target_link_libraries(test_autotune_n32
    PRIVATE
        nvrtc_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# Phase 3.5d — multi-stream chunked launch benchmark at N=54.
add_executable(test_streams_n54 test_streams_n54.cpp)
target_link_libraries(test_streams_n54
    PRIVATE
        nvrtc_geneig
        LAPACK::lapacke
        CUDA::cudart
        CUDA::cuda_driver
)

# Phase 3.5b BlockDim micro-bench at N=32, BPB=2.
add_executable(test_blockdim_n32 test_blockdim_n32.cpp)
target_link_libraries(test_blockdim_n32
    PRIVATE
        nvrtc_geneig
        LAPACK::lapacke
        CUDA::cudart
        CUDA::cuda_driver
)

# Phase 3.5b probe-compile timing measurement (one-off).
add_executable(probe_compile_timing probe_compile_timing.cpp)
target_link_libraries(probe_compile_timing
    PRIVATE
        CUDA::nvrtc
        CUDA::cuda_driver
        CUDA::nvJitLink
)
target_compile_definitions(probe_compile_timing
    PRIVATE
        CUSOLVERDX_FATBIN_PATH="${MathDx_FATBIN}"
        CUSOLVERDX_INCLUDE_DIR="${MathDx_INCLUDE_DIR}"
        CUSOLVERDX_CUTLASS_INCLUDE_DIR="${MathDx_CUTLASS_INCLUDE_DIR}"
        CUSOLVERDX_OVERLAY_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/src/nvrtc/cusolverdx_overlay"
        CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
)

# Phase 3a — minimal NVRTC bug repros (heev fails, potrf succeeds).
add_executable(repro_heev_nvrtc  repro_heev_nvrtc.cpp)
add_executable(repro_potrf_nvrtc repro_potrf_nvrtc.cpp)
foreach(_t repro_heev_nvrtc repro_potrf_nvrtc)
    target_link_libraries(${_t} PRIVATE CUDA::nvrtc CUDA::cuda_driver)
    target_compile_definitions(${_t}
        PRIVATE
            CUSOLVERDX_INCLUDE_DIR="${MathDx_INCLUDE_DIR}"
            CUSOLVERDX_CUTLASS_INCLUDE_DIR="${MathDx_CUTLASS_INCLUDE_DIR}"
            CUSOLVERDX_OVERLAY_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/src/nvrtc/cusolverdx_overlay"
            CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
    )
endforeach()
