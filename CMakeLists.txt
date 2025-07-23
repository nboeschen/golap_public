cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(golap LANGUAGES CXX CUDA)

##############################################
option(WITH_CPU_COMP_CMAKE "Adds CPU compression integration" ON)
option(WITH_AWS_CMAKE "Adds AWS integration" OFF)
option(WITH_GDSASYNC_CMAKE "Enable GDS Async" OFF)
##############################################


file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if (NOT CMAKE_BUILD_TYPE)
  message("CMAKE_BUILD_TYPE not defined")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
message("[${CMAKE_BUILD_TYPE}] config will be created...")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wno-unused-function")

##############################################
# third party libraries
##############################################
set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party/")


if (WITH_AWS_CMAKE)
    message(">>>>>>>>>>>>> Compiling with AWS")
    add_compile_definitions(WITH_AWS)
    find_package(ZLIB REQUIRED)
    set(AWSSDK_DIR "${THIRD_PARTY_DIR}/vcpkg/vcpkg_installed/x64-linux/share/AWSSDK/")
    set(CMAKE_PREFIX_PATH "${THIRD_PARTY_DIR}/vcpkg/vcpkg_installed/x64-linux/share/:${CMAKE_PREFIX_PATH}")
    find_package(AWSSDK REQUIRED COMPONENTS s3 s3-crt)
    include_directories(${AWSSDK_INCLUDE_DIRS})
endif()
if (WITH_GDSASYNC_CMAKE)
    message(">>>>>>>>>>>>> Compiling with GDS Async")
    add_compile_definitions(WITH_GDSASYNC)
endif()
if (WITH_CPU_COMP_CMAKE)
    message(">>>>>>>>>>>>> Compiling with CPU Compression")
    add_compile_definitions(WITH_CPU_COMP)
    find_library(SNAPPY_LIBRARY NAMES libsnappy.a HINTS "${THIRD_PARTY_DIR}/snappy/build/")
    find_library(LZ4_LIBRARY NAMES liblz4.so HINTS "${THIRD_PARTY_DIR}/lz4/install/usr/local/lib/")
    include_directories("${THIRD_PARTY_DIR}/lz4/install/usr/local/include/")
    include_directories("${THIRD_PARTY_DIR}/snappy/")
    include_directories("${THIRD_PARTY_DIR}/snappy/build/")
endif()

set(gflags_DIR "${THIRD_PARTY_DIR}/gflags/build/")
find_package(gflags REQUIRED)
# set(gflags_cxx_abi_fix_DIR "${THIRD_PARTY_DIR}/gflags_cxx_abi_fix/build/")
# find_package(gflags_cxx_abi_fix REQUIRED NAMES gflags-config.cmake)

find_library(NVCOMP_LIBRARY NAMES libnvcomp.so HINTS "${THIRD_PARTY_DIR}/nvcomp/install/lib/")
include_directories("${THIRD_PARTY_DIR}/nvcomp/install/include/")

include_directories(SYSTEM "${THIRD_PARTY_DIR}/concurrentqueue/")
include_directories("${THIRD_PARTY_DIR}/perfevent/")

find_library(duckdb NAMES libduckdb.so libduckdb_static.a HINTS "${THIRD_PARTY_DIR}/duckdb1.0/")
##############################################
# end of third party libraries
##############################################

# for helper_cuda.h, nvToolsExt.h
foreach(sample_dir
        "/opt/cuda/targets/x86_64-linux/include/nvtx3/"
        "/usr/local/cuda-11.4/targets/x86_64-linux/include/nvtx3/"
        "/usr/local/cuda-11.8/targets/x86_64-linux/include/nvtx3/"
        "/usr/local/cuda-12/targets/x86_64-linux/include/nvtx3/")
    if(EXISTS ${sample_dir})
        message("Including ${sample_dir}")
        include_directories(${sample_dir})
    endif()
endforeach(sample_dir)


include_directories(core)
include_directories(core/hl)
add_subdirectory(core)

add_subdirectory(usecases/ssb)
add_subdirectory(usecases/ssb_orig)
add_subdirectory(usecases/taxi)
add_subdirectory(usecases/tpch)
