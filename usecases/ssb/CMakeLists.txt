
# Build executable ssb
add_executable(ssb
               ssb.cu
               )

target_link_libraries(ssb golap_core gflags)
# file(GLOB QUERY_HEADERS ${PROJECT_SOURCE_DIR}/queries/*.hpp ${PROJECT_SOURCE_DIR}/queries/*.cuh)
# target_precompile_headers(ssb PUBLIC ${QUERY_HEADERS})
set_target_properties(ssb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ssb PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(ssb PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(ssb PROPERTIES CUDA_ARCHITECTURES 70 75 80)


# Build executable sample_ssb
add_executable(sample_ssb
               ssb.hpp
               sample_ssb.cu
               )

target_link_libraries(sample_ssb golap_core gflags)
set_target_properties(sample_ssb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sample_ssb PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sample_ssb PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(sample_ssb PROPERTIES CUDA_ARCHITECTURES 70 75 80)


# # Build executable ssb_disk_db
add_executable(ssb_disk_db
               ssb_disk_db.cu
               )
target_link_libraries(ssb_disk_db golap_core gflags)
set_target_properties(ssb_disk_db PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ssb_disk_db PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(ssb_disk_db PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(ssb PROPERTIES CUDA_ARCHITECTURES 70 75 80)


# Build executable duckdb_ssb, with either libduckdb, or from source
add_executable(duckdb_ssb
               duckdb_ssb.cpp
               # "${THIRD_PARTY_DIR}/duckdb_src/duckdb.cpp"
               )
target_include_directories(duckdb_ssb PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
target_link_directories(duckdb_ssb PUBLIC "${THIRD_PARTY_DIR}/duckdb/")
target_link_libraries(duckdb_ssb gflags duckdb)

add_executable(duckdb_ssb1.0
               duckdb_ssb.cpp
               # "${THIRD_PARTY_DIR}/duckdb_src/duckdb.cpp"
               )
target_include_directories(duckdb_ssb1.0 PUBLIC "${THIRD_PARTY_DIR}/duckdb1.0/")
target_link_directories(duckdb_ssb1.0 PUBLIC "${THIRD_PARTY_DIR}/duckdb1.0/")
set_target_properties(duckdb_ssb1.0 PROPERTIES COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(duckdb_ssb1.0 gflags_cxx_abi_fix duckdb)

# alternative for duckdb src:
# target_include_directories(duckdb_ssb PUBLIC "${THIRD_PARTY_DIR}/duckdb_src/")
# target_link_directories(duckdb_ssb PUBLIC "${THIRD_PARTY_DIR}/duckdb_src/")
# target_link_libraries(duckdb_ssb dl gflags )

# target_include_directories(duckdb_ssb1.0 PUBLIC "${THIRD_PARTY_DIR}/duckdb_src1.0/")
# target_link_directories(duckdb_ssb1.0 PUBLIC "${THIRD_PARTY_DIR}/duckdb_src1.0/")
# target_link_libraries(duckdb_ssb1.0 gflags )

