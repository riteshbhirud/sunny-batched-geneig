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
