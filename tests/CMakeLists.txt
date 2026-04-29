add_executable(lapack_reference lapack_reference.cpp)
target_link_libraries(lapack_reference PRIVATE LAPACK::lapacke)

add_executable(dump_pair dump_pair.cpp)
target_link_libraries(dump_pair PRIVATE LAPACK::lapacke)
