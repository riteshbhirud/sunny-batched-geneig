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
