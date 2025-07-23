#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "mem.hpp"
#include "util.hpp"
#include "dev_structs.cuh"


__global__ void bloommapkernel(golap::BloomFilter filter, uint64_t* table, uint64_t num){
    uint32_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_idx>=num) return;

    filter.map(table[thread_idx]);
}

__global__ void bloomtestkernel(golap::BloomFilter filter, uint64_t* matches, uint64_t num){
    uint32_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_idx>=num) return;

    if (filter.query(thread_idx)){
        atomicAdd((unsigned long long*) matches,1);
    }
}


TEST_CASE("Bloom", "[bloom]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    for (double p = 0.125; p<1.0; p+=0.125){
        uint64_t ROW_NUM = (1<<20); // 4 Mio

        golap::MirrorMem table(golap::Tag<uint64_t>(), ROW_NUM);
        golap::MirrorMem matches(golap::Tag<uint64_t>(), 1);
        checkCudaErrors(cudaMemset(matches.dev.ptr<uint8_t>(),0,sizeof(uint64_t)));

        for(int i = 0; i < table.size<uint64_t>(); ++i){
            table.hst.ptr<uint64_t>()[i] = i;
        }
        table.sync_to_device();

        golap::BloomFilter filter(ROW_NUM, p);


        bloommapkernel<<<util::div_ceil(table.size<uint64_t>(),512),512>>>(filter, table.dev.ptr<uint64_t>(), ROW_NUM);
        bloomtestkernel<<<util::div_ceil(ROW_NUM<<2,512),512>>>(filter, matches.dev.ptr<uint64_t>(), ROW_NUM<<2);
        getLastCudaError("Something wrong in bloom kernels");

        matches.sync_to_host();

        // std::cout << matches.hst.ptr<uint64_t>()[0] << " found\n";
        
        // fpr = FP / (FP + TN)
        // ground truth negatives N = (FP + TN)
        double N = (ROW_NUM<<2) - ROW_NUM;
        double FP = matches.hst.ptr<uint64_t>()[0] - (ROW_NUM);

        std::cout << "Actual FPR is "<<(FP / N) << ", should be "<<p<<" \n";
        REQUIRE((FP / N) <= Catch::Approx(p).epsilon(0.05));
    }
    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}



