#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "mem.hpp"
#include "util.hpp"
#include "sort.cuh"



TEST_CASE("Sort", "[sort]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t ROW_NUM = (1<<22); // 4 Mio

    golap::MirrorMem unsorted_table(golap::Tag<uint32_t>(), ROW_NUM);
    golap::MirrorMem sorted_table(golap::Tag<uint32_t>(), ROW_NUM);
    golap::MirrorMem unsorted_values(golap::Tag<float>(), ROW_NUM);
    golap::MirrorMem sorted_values(golap::Tag<float>(), ROW_NUM);

    for(int i = 0; i < unsorted_table.size<uint32_t>(); ++i){
        unsorted_table.hst.ptr<uint32_t>()[i] = ROW_NUM-i;
    }
    unsorted_table.sync_to_device();

    golap::Sort sort(unsorted_table.dev.ptr<uint32_t>(), sorted_table.dev.ptr<uint32_t>(), 
                     unsorted_values.dev.ptr<float>(), sorted_values.dev.ptr<float>(), 
                     unsorted_table.dev.size<uint32_t>());

    sort.execute();

    sorted_table.sync_to_host();


    for(int i = 1; i<sorted_table.size<uint32_t>(); ++i){
        REQUIRE(sorted_table.hst.ptr<uint32_t>()[i-1] < sorted_table.hst.ptr<uint32_t>()[i]);
    }



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}



