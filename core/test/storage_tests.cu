#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "core.hpp"
#include "mem.hpp"
#include "storage.hpp"


struct SomeOtherType{
    int id;
    char name[11];
    float discount;
    friend std::ostream& operator<<(std::ostream &out, SomeOtherType const& obj){
        out << "SomeOtherType(id="<<obj.id<<", name="<<obj.name<<", discount="<<obj.discount<<")";
        return out;
    }
};

TEST_CASE("Host WR", "[storage]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    int N = 1<<20;
    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);
    golap::HostMem mem_w(golap::Tag<uint32_t>{}, N);
    golap::HostMem mem_r(golap::Tag<uint32_t>{}, N);
    uint32_t* buf_w = mem_w.ptr<uint32_t>();
    uint32_t* buf_r = mem_r.ptr<uint32_t>();
    // char* buf_w = new (std::align_val_t(4096)) char[N];
    // char* buf_r = new (std::align_val_t(4096)) char[N];
    for (int i = 0; i<N; ++i){
        buf_w[i] = i%128;
    }

    REQUIRE(sm.host_write_bytes(buf_w, N*sizeof(uint32_t), 0));
    REQUIRE(sm.host_read_bytes(buf_r, N*sizeof(uint32_t), 0));

    for (int i = 0; i<N; ++i){
        REQUIRE(buf_r[i] == i%128);
    }

    // delete[] buf_w;
    // delete[] buf_r;

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("Host W Dev R", "[storage]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    int N = 1<<20;
    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);

    golap::HostMem mem_w(golap::Tag<char>{}, N);
    golap::HostMem res(golap::Tag<char>{}, N);
    golap::DeviceMem mem_r(golap::Tag<char>{}, N);
    char* buf_w = mem_w.ptr<char>();
    char* buf_r = mem_r.ptr<char>();
    char* buf_res = res.ptr<char>();

    for (int i = 0; i<N; ++i){
        buf_w[i] = i%128;
    }

    REQUIRE(sm.host_write_bytes(buf_w, N, 0));
    REQUIRE(sm.dev_read_bytes(buf_r, N, 0));

    checkCudaErrors(cudaMemcpy(buf_res, buf_r, N, cudaMemcpyDefault));

    for (int i = 0; i<N; ++i){
        REQUIRE(buf_res[i] == i%128);
    }

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("Dev W Host R", "[storage]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    int N = 1<<20;
    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);
    golap::HostMem mem_h(golap::Tag<char>{}, N);
    golap::HostMem res(golap::Tag<char>{}, N);
    golap::DeviceMem mem_d(golap::Tag<char>{}, N);
    char* buf_h = mem_h.ptr<char>();
    char* buf_d = mem_d.ptr<char>();
    char* buf_res = res.ptr<char>();

    for (int i = 0; i<N; ++i){
        buf_h[i] = i%128;
    }
    checkCudaErrors(cudaMemcpy(buf_d, buf_h, N, cudaMemcpyDefault));


    REQUIRE(sm.dev_write_bytes(buf_d, N, 0));
    REQUIRE(sm.host_read_bytes(buf_res, N, 0));


    for (int i = 0; i<N; ++i){
        REQUIRE(buf_res[i] == i%128);
    }

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}
