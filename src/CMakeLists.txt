add_executable(smoke smoke.cu)

target_link_libraries(smoke
    PRIVATE
        MathDx::cusolverdx
        CUDA::cudart
)

set_target_properties(smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_library(geneig_fixed STATIC geneig_fixed.cu)

target_link_libraries(geneig_fixed
    PUBLIC
        MathDx::cusolverdx
        CUDA::cudart
)

set_target_properties(geneig_fixed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Phase 3a — NVRTC + nvJitLink runtime-compile path. Pure host C++; no .cu
# sources, no separable compilation. The cuSolverDx headers and fatbin are
# referenced only at runtime, via NVRTC `--include-path` flags and an
# `nvJitLinkAddFile(NVJITLINK_INPUT_FATBIN, ...)` call respectively.
add_library(nvrtc_geneig STATIC nvrtc/nvrtc_solver.cpp)

target_include_directories(nvrtc_geneig
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(nvrtc_geneig
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::nvJitLink
)

target_compile_definitions(nvrtc_geneig
    PRIVATE
        CUSOLVERDX_FATBIN_PATH="${MathDx_FATBIN}"
        CUSOLVERDX_INCLUDE_DIR="${MathDx_INCLUDE_DIR}"
        CUSOLVERDX_CUTLASS_INCLUDE_DIR="${MathDx_CUTLASS_INCLUDE_DIR}"
        CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
)

set_target_properties(nvrtc_geneig PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
