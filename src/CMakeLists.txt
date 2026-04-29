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
