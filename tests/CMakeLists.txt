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

# Phase 3a — hybrid NVRTC + static-link correctness test. Links both libraries:
#   nvrtc_geneig : reduce stage (NVRTC-compiled potrf + 2 trsms)
#   geneig_fixed : finishing stage (static heev + back-transform trsm)
add_executable(test_nvrtc_n54 test_nvrtc_n54.cpp)
target_link_libraries(test_nvrtc_n54
    PRIVATE
        nvrtc_geneig
        geneig_fixed
        LAPACK::lapacke
        CUDA::cudart
        CUDA::cuda_driver
)
set_target_properties(test_nvrtc_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
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
            CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
    )
endforeach()
