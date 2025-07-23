#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "test_common.hpp"
#include "mem.hpp"


struct struct16{
    int a,b,c,d;
};

TEST_CASE("Basic Memory", "[mem]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));

    golap::DeviceMem dev_mem(golap::Tag<int>{}, 512);
    golap::HostMem hst_mem(golap::Tag<int>{}, 512);

    (hst_mem.ptr<int>())[16] = 15;

    REQUIRE(dev_mem.size_bytes() == sizeof(int)*512);
    REQUIRE(dev_mem.size<int>() == 512);

    REQUIRE(hst_mem.size_bytes() == sizeof(int)*512);
    REQUIRE(hst_mem.size<int>() == 512);
    REQUIRE(hst_mem.ptr<int>()[16] == 15);

}


TEST_CASE("MirrorMem", "[mem]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));

    golap::MirrorMem mem(golap::Tag<int>{}, 512);

    REQUIRE(mem.size_bytes() == sizeof(int)*512);
    REQUIRE(mem.size<int>() == 512);

    for (int i = 0; i<512; ++i) mem.hst.ptr<int>()[i] = 512-i;
    mem.sync_to_device();
    for (int i = 0; i<512; ++i) mem.hst.ptr<int>()[i] = 0;
    mem.sync_to_host();

    for (int i = 0; i<512; ++i){
        REQUIRE(mem.hst.ptr<int>()[i] == 512-i);
    }

}

TEST_CASE("Alloc unit", "[mem]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));

    golap::DeviceMem dev_mem;

    dev_mem.resize_num<char[11]>(2);
    REQUIRE(dev_mem.size_bytes() == 22);
    REQUIRE(dev_mem.size<char[11]>() == 2);


    dev_mem.resize_num<struct16>(4);
    REQUIRE(dev_mem.size_bytes() == 64);
    REQUIRE(dev_mem.size<struct16>() == 4);

}


TEST_CASE("Allocation Helper", "[mem]"){

    // 2000 bytes in units of 4kb aligned to 8192 
    golap::AllocHelper helper{2000, 4096, 8192};

    int* dummy = new int;
    REQUIRE(helper.alloc_size() == 4096+8192);


    REQUIRE(((uint64_t)helper.align(dummy))%8192 == 0);
    util::Log::get().info_fmt("Dummy was %p, aligned is %p",dummy,helper.align(dummy));

    delete dummy;
}

TEST_CASE("Alignment", "[mem]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));

    uint64_t alloc_unit = 16;
    uint64_t alignment = 4096;

    golap::DeviceMem dev_mem{golap::Tag<char[11]>{}, 50, alloc_unit, alignment};

    REQUIRE(dev_mem.size_bytes() % sizeof(alloc_unit) == 0);
    REQUIRE(((uint64_t)dev_mem.data) % alignment == 0);

    dev_mem.resize_num<char[11]>(100,alloc_unit,alignment,false);
    REQUIRE(dev_mem.size_bytes() % sizeof(alloc_unit) == 0);
    REQUIRE(((uint64_t)dev_mem.data) % alignment == 0);
    

    dev_mem.resize_num<char[11]>(100,alloc_unit,alignment,true);
    REQUIRE(dev_mem.size_bytes() % sizeof(alloc_unit) == 0);
    REQUIRE(((uint64_t)dev_mem.data) % alignment == 0);


}