#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>
#include <vector>
#include <thread>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "mem.hpp"
#include "util.hpp"
#include "group.cuh"
#include "dev_structs.cuh"
#include "host_structs.hpp"


struct alignas(8) GROUP_ID_DISC{
    int id;
    float discount;

    __host__ __device__ inline
    bool operator==(const GROUP_ID_DISC& other){
        return id == other.id && discount == other.discount;
    }

    __host__ __device__ inline
    uint64_t hash(){
        return *((uint64_t*) this);
    }
};

struct alignas(8) LARGER_GROUP{
    int id;
    float discount;
    int a;
    int b;

    __host__ __device__ inline
    bool operator==(const LARGER_GROUP& other){
        return id == other.id && discount == other.discount && a == other.a && b == other.b;
    }

    __host__ __device__ inline
    uint64_t hash(){
        return (*((uint64_t*) this)) ^ (((uint64_t)a)<<32) ^ b;
    }
};

struct SumFloat{
    __device__ inline
    void operator()(float *agg, float val){
        atomicAdd(agg, val);
    }
};

struct HostSumInt{
    __host__
    void operator()(std::atomic<uint64_t> *agg, uint64_t val){
        agg->fetch_add(val, std::memory_order_relaxed);
    }
};


__global__ void group_test(golap::HashAggregate<GROUP_ID_DISC, float> hashagg, int *col0, float *col1,
                                  float *agg_col,
                                  uint64_t num, SumFloat agg_func){
    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= num) return;

    hashagg.add(GROUP_ID_DISC{col0[r_id],col1[r_id]}, agg_col[r_id], agg_func);
}

TEST_CASE("GROUP BY AGG COLUMNS", "[gb-agg]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t ROW_NUM = ((uint64_t)1<<30);
    uint64_t GROUP_NUM = 64;

    golap::MirrorMem col_id(golap::Tag<int>(), ROW_NUM);
    golap::MirrorMem col_disc(golap::Tag<float>(), ROW_NUM);
    golap::MirrorMem col_agg(golap::Tag<float>(), ROW_NUM);


    golap::MirrorMem groups_table(golap::Tag<GROUP_ID_DISC>{}, GROUP_NUM);
    golap::MirrorMem result_table(golap::Tag<float>{}, GROUP_NUM);
    checkCudaErrors(cudaMemset(groups_table.dev.ptr<char>(), 0xff, groups_table.size_bytes()));


    for(uint64_t i = 0; i < ROW_NUM; ++i){
        col_id.hst.ptr<int>()[i] = i%GROUP_NUM;
        col_disc.hst.ptr<float>()[i] = 10.0*(i%GROUP_NUM);
        col_agg.hst.ptr<float>()[i] = 1.0;
    }
    col_id.sync_to_device();
    col_disc.sync_to_device();
    col_agg.sync_to_device();

    golap::HashAggregate hash_agg(GROUP_NUM,groups_table.dev.ptr<GROUP_ID_DISC>(),
                                  result_table.dev.ptr<float>());

    
    util::Timer timer;
    group_test<<<util::div_ceil(ROW_NUM,512),512>>>(hash_agg,
                                                    col_id.dev.ptr<int>(),
                                                    col_disc.dev.ptr<float>(),
                                                    col_agg.dev.ptr<float>(),
                                                    ROW_NUM,
                                                    SumFloat());

    groups_table.sync_to_host();
    result_table.sync_to_host();
    
    std::cout << "GroupBy Aggregate took " << timer.elapsed() << "ms.\n"; 

    golap::HostMem pop_group_slot{golap::Tag<uint32_t>{}, GROUP_NUM};
    checkCudaErrors(cudaMemcpy(pop_group_slot.ptr<uint32_t>(), hash_agg.wrote_group, GROUP_NUM*sizeof(uint32_t), cudaMemcpyDefault));

    for(int i = 0; i<result_table.size<float>(); ++i){
        if (pop_group_slot.ptr<uint32_t>()[i] == 0) continue;
        GROUP_ID_DISC group = groups_table.hst.ptr<GROUP_ID_DISC>()[i];
        // printf("Group[%d,%f] -> %f\n",group.id,group.discount, result_table.hst.ptr<float>()[i]);
        REQUIRE(result_table.hst.ptr<float>()[i] == ROW_NUM / GROUP_NUM);
    }



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}


TEST_CASE("LARGE GROUP", "[gb-agg]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t ROW_NUM = (1<<22); // 4 Mio
    // uint64_t ROW_NUM = 512;
    uint64_t GROUP_NUM = 32;
    // uint64_t GROUP_NUM = 4;

    golap::MirrorMem col_id(golap::Tag<int>(), ROW_NUM);
    golap::MirrorMem col_disc(golap::Tag<float>(), ROW_NUM);
    golap::MirrorMem col_a(golap::Tag<int>(), ROW_NUM);
    golap::MirrorMem col_b(golap::Tag<int>(), ROW_NUM);
    golap::MirrorMem col_agg(golap::Tag<float>(), ROW_NUM);


    golap::MirrorMem groups_table(golap::Tag<LARGER_GROUP>{}, GROUP_NUM);
    golap::MirrorMem result_table(golap::Tag<float>{}, GROUP_NUM);
    checkCudaErrors(cudaMemset(groups_table.dev.ptr<char>(), 0xff, groups_table.size_bytes()));


    for(uint64_t i = 0; i < ROW_NUM; ++i){
        col_id.hst.ptr<int>()[i] = i%GROUP_NUM;
        col_disc.hst.ptr<float>()[i] = 10.0*(i%GROUP_NUM);
        col_a.hst.ptr<int>()[i] = i%GROUP_NUM;
        col_b.hst.ptr<int>()[i] = i%GROUP_NUM;
        col_agg.hst.ptr<float>()[i] = 1.0;
    }
    col_id.sync_to_device();
    col_disc.sync_to_device();
    col_a.sync_to_device();
    col_b.sync_to_device();
    col_agg.sync_to_device();

    golap::HashAggregate hash_agg(GROUP_NUM,groups_table.dev.ptr<LARGER_GROUP>(),
                                  result_table.dev.ptr<float>());


    golap::group_by_agg_4col<<<util::div_ceil(ROW_NUM,512),512>>>(hash_agg,
                                                                            col_id.dev.ptr<int>(),
                                                                            col_disc.dev.ptr<float>(),
                                                                            col_a.dev.ptr<int>(),
                                                                            col_b.dev.ptr<int>(),
                                                                            col_agg.dev.ptr<float>(),
                                                                            ROW_NUM,
                                                                            SumFloat());

    groups_table.sync_to_host();
    result_table.sync_to_host();


    for(int i = 0; i<result_table.size<float>(); ++i){
        LARGER_GROUP group = groups_table.hst.ptr<LARGER_GROUP>()[i];
        printf("Group[%d,%f,%d,%d] -> %f\n",group.id,group.discount,group.a,group.b, result_table.hst.ptr<float>()[i]);
        REQUIRE(result_table.hst.ptr<float>()[i] == ROW_NUM / GROUP_NUM);
    }



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}


TEST_CASE("HOST GROUP BY AGG COLUMNS", "[gb-agg]") {

    uint64_t ROW_NUM = ((uint64_t)1<<30);
    uint64_t GROUP_NUM = 128;

    golap::HostMem col_id(golap::Tag<int>(), ROW_NUM);
    golap::HostMem col_disc(golap::Tag<float>(), ROW_NUM);
    golap::HostMem col_agg(golap::Tag<uint64_t>(), ROW_NUM);


    golap::HostMem groups_table(golap::Tag<GROUP_ID_DISC>{}, GROUP_NUM);
    // golap::HostMem result_table(golap::Tag<float>{}, GROUP_NUM);

    auto result_table = new std::atomic<uint64_t>[GROUP_NUM]{};


    for(uint64_t i = 0; i < ROW_NUM; ++i){
        col_id.ptr<int>()[i] = i%GROUP_NUM;
        col_disc.ptr<float>()[i] = 10.0*(i%GROUP_NUM);
        col_agg.ptr<uint64_t>()[i] = 1;
    }

    golap::HostHashAggregate hash_agg(GROUP_NUM,groups_table.ptr<GROUP_ID_DISC>(),
                                  result_table);
    
    uint32_t thread_num = 32;
    std::vector<std::thread> threads;
    threads.reserve(thread_num);
    uint64_t min_rows_per_thread = ROW_NUM / thread_num; 

    util::Timer timer;
    for (uint32_t thread_idx = 0; thread_idx< thread_num; ++thread_idx){
        threads.emplace_back([thread_idx,&ROW_NUM,&thread_num,&min_rows_per_thread,
                              &hash_agg,
                              &col_id,&col_disc,&col_agg]{
            uint64_t start = thread_idx * min_rows_per_thread; 
            uint64_t end = thread_idx == thread_num-1 ? ROW_NUM : start + min_rows_per_thread;
            for(uint64_t r_id = start; r_id < end; ++r_id){
                hash_agg.add(GROUP_ID_DISC{col_id.ptr<int>()[r_id],col_disc.ptr<float>()[r_id]},
                             col_agg.ptr<uint64_t>()[r_id], HostSumInt());
            }
        });
    }

    for (auto& thread : threads) thread.join();
    
    std::cout << "GroupBy Aggregate took " << timer.elapsed() << "ms.\n"; 

    golap::HostMem pop_group_slot{golap::Tag<uint32_t>{}, GROUP_NUM};
    checkCudaErrors(cudaMemcpy(pop_group_slot.ptr<uint32_t>(), hash_agg.wrote_group, GROUP_NUM*sizeof(uint32_t), cudaMemcpyDefault));

    for(int i = 0; i<GROUP_NUM; ++i){
        if (hash_agg.wrote_group[i] == 0) continue;
        GROUP_ID_DISC group = groups_table.ptr<GROUP_ID_DISC>()[i];
        // printf("Group[%d,%f] -> %f\n",group.id,group.discount, result_table.ptr<float>()[i]);
        REQUIRE(result_table[i] == ROW_NUM / GROUP_NUM);
    }



}