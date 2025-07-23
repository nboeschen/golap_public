
# Build executable tpch
add_executable(tpch
               tpch.cu
               )

target_link_libraries(tpch golap_core gflags)
# file(GLOB QUERY_HEADERS ${PROJECT_SOURCE_DIR}/queries/*.hpp ${PROJECT_SOURCE_DIR}/queries/*.cuh)
# target_precompile_headers(tpch PUBLIC ${QUERY_HEADERS})
set_target_properties(tpch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tpch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(tpch PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(tpch PROPERTIES CUDA_ARCHITECTURES 61 70 80)


# # Build executable tpch_disk_db
add_executable(tpch_disk_db
               tpch_disk_db.cu
               )
target_link_libraries(tpch_disk_db stdc++fs gflags numa lz4 cufile ${SNAPPY_LIBRARY} ${NVCOMP_LIBRARY})
set_target_properties(tpch_disk_db PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tpch_disk_db PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(tpch_disk_db PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(tpch_disk_db PROPERTIES CUDA_ARCHITECTURES 61 70 80)

# # Build executable duckdb_tpch
# add_executable(duckdb_tpch
#                duckdb_tpch.cpp
#                )
# target_include_directories(duckdb_tpch PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
# target_link_directories(duckdb_tpch PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
# target_link_libraries(duckdb_tpch gflags duckdb)

# # Build executable duckdb_tpch
add_executable(duckdb_tpch
               duckdb_tpch.cpp
               )
target_include_directories(duckdb_tpch PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
target_link_directories(duckdb_tpch PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
target_link_libraries(duckdb_tpch gflags duckdb)
