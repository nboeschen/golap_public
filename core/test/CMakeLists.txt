
# Build TEST_BASIC
add_executable(test_benchmark
               benchmark_tests.cu
               )

target_link_libraries(test_benchmark golap_core gflags)
set_target_properties(test_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_benchmark PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_benchmark PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(test_benchmark PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(test_benchmark PROPERTIES CUDA_ARCHITECTURES 61 70 80)


add_subdirectory("${THIRD_PARTY_DIR}/Catch2" Catch2)
enable_testing()

# Build TEST_BASIC
add_executable(test_basic
               mem_tests.cu
               table_tests.cu
               storage_tests.cu
               join_tests.cu
               group_by_tests.cu
               sort_tests.cu
               metadata_tests.cu
               cluster_tests.cu
               agg_tests.cu
               apps_tests.cu
               comp_tests.cu
               access_tests.cu
               op_tests.cu
               bloom_tests.cu
               util_tests.cu
               sample_tests.cu
               net_tests.cu
               )

target_link_libraries(test_basic golap_core Catch2::Catch2WithMain)
set_target_properties(test_basic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_basic PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_basic PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(test_basic PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(test_basic PROPERTIES CUDA_ARCHITECTURES 61 70 80)

add_test(NAME BasicTest COMMAND test_basic)
