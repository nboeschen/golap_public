#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "util.hpp"
#include "dev_util.cuh"


TEST_CASE("Pipeline Work Distribution", "[util]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t PIPELINES = 8;
    uint64_t BLOCKS = 6;

    util::Log::get().info_fmt("Pipelines=%lu,Blocks=%lu",PIPELINES,BLOCKS);
    util::SliceSeq workslice(BLOCKS,PIPELINES);
    uint64_t slicestart,sliceend,total=0;
    for (uint32_t pipeline_idx=0; pipeline_idx<PIPELINES; ++pipeline_idx){

        workslice.get(slicestart,sliceend);
        total += sliceend-slicestart;
        util::Log::get().info_fmt("Pipeline[%lu] got assigned %lu: [%lu - %lu)",pipeline_idx,sliceend-slicestart,slicestart,sliceend);
    }
    
    REQUIRE(total == BLOCKS);
    } // stack for cuda-memcheck
}

TEST_CASE("Sample Tests", "[util]") {

    uint64_t num = 64;
    uint64_t min = 500;
    uint64_t max = 1000;

    auto samples = util::sample_range(min,max,num);
    std::unordered_set<uint32_t> seen;

    REQUIRE(samples.size() == num);

    for(auto& sample: samples){

        REQUIRE(sample < max);
        REQUIRE(sample >= min);
        REQUIRE(seen.find(sample) == seen.end());
        seen.insert(sample);

    }

}

TEST_CASE("Sample Invalid Range", "[util]") {

    uint64_t num = 64;
    uint64_t min = 500;
    uint64_t max = 550;

    REQUIRE_THROWS_AS(util::sample_range(min,max,num), std::runtime_error);
}


TEST_CASE("Pin thread", "[util]") {
    // cat /sys/fs/cgroup/cpuset/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/step_${SLURM_STEP_ID}/cpuset.cpus
    // 71-86,199-214
    int32_t cpuid;

    std::thread t([&](){
        if (util::pin_thread(71) != 0){
            std::cout << strerror(errno) << "\n";
        }
        cpuid = sched_getcpu();
        if (cpuid == -1){
            std::cout << "getcpu failed\n";
        }

        return;
    });

    t.join();

    REQUIRE(cpuid == 71);
}

TEST_CASE("Wait kernel", "[util]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    util::Timer t;

    util::waiting_kernel<<<1,1,0,stream>>>(5);
    checkCudaErrors(cudaStreamSynchronize(stream));
    auto x = t.elapsed();

    std::cout << "Elapsed: " << x << "\n";
    REQUIRE(true);
    cudaStreamDestroy(stream);
    } // stack for cuda-memcheck
}
