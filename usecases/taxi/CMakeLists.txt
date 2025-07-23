
# Build executable taxi
add_executable(taxi
               taxi.cu
               )

target_link_libraries(taxi golap_core gflags)
# file(GLOB QUERY_HEADERS ${PROJECT_SOURCE_DIR}/queries/*.hpp ${PROJECT_SOURCE_DIR}/queries/*.cuh)
# target_precompile_headers(taxi PUBLIC ${QUERY_HEADERS})
set_target_properties(taxi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(taxi PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(taxi PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(taxi PROPERTIES CUDA_ARCHITECTURES 61 70 80)


# Build executable sample_taxi
add_executable(sample_taxi
               taxi.hpp
               sample_taxi.cu
               )

target_link_libraries(sample_taxi stdc++fs gflags numa lz4 cufile ${SNAPPY_LIBRARY} ${NVCOMP_LIBRARY})
set_target_properties(sample_taxi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sample_taxi PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sample_taxi PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(sample_taxi PROPERTIES CUDA_ARCHITECTURES 61 70 80)

# # Build executable taxi_disk_db
add_executable(taxi_disk_db
               taxi_disk_db.cu
               )
target_link_libraries(taxi_disk_db stdc++fs gflags numa lz4 cufile ${SNAPPY_LIBRARY} ${NVCOMP_LIBRARY})
set_target_properties(taxi_disk_db PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(taxi_disk_db PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(taxi_disk_db PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(taxi_disk_db PROPERTIES CUDA_ARCHITECTURES 61 70 80)

# # Build executable duckdb_taxi
add_executable(duckdb_taxi
               duckdb_taxi.cpp
               )
target_include_directories(duckdb_taxi PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
target_link_directories(duckdb_taxi PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
target_link_libraries(duckdb_taxi gflags duckdb)

