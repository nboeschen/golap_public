#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "mem.hpp"
#include "util.hpp"
#include "join.cuh"
#include "table.hpp"


struct ATuple{
    int id;
    char name[11];
    float discount;
    __host__ __device__
    uint64_t key(){return id;}
    friend std::ostream& operator<<(std::ostream &out, ATuple const& obj){
        out << "ATuple(id="<<obj.id<<", name="<<obj.name<<", discount="<<obj.discount<<")";
        return out;
    }
};
struct ATupleJoinKey{
    __device__ inline
    uint64_t operator()(ATuple *at){
        return at->id;
    }
};

struct MatTuple{
    int key;
    float discount_build;
    float discount_probe;
    __host__ __device__
    MatTuple(ATuple *build_in, ATuple *probe_in){
        key = build_in->key();
        discount_build = build_in->discount;
        discount_probe = probe_in->discount;
    }
    friend std::ostream& operator<<(std::ostream &out, MatTuple const& obj){
        out << "MatTuple(key="<<obj.key<<", discount_build="<<obj.discount_build<<", discount_probe="<<obj.discount_probe<<")";
        return out;
    }
};

struct MaxFunctor{
    __device__ inline
    void operator()(uint64_t *agg, ATuple *build_in, ATuple *probe_in) {
        atomicMax((unsigned long long*) agg, build_in->key());
    }
};
template <uint64_t N>
struct PredFunctor{
    __device__ inline
    bool operator()(ATuple *tup) {
        return tup->key() >= N;
    }
};
struct AJoinKey{
    __device__ inline
    uint64_t operator()(ATuple *tup) {
        return tup->id;
    }
};



TEST_CASE("Join Count 1to1", "[join]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    golap::MirrorMem build_table(golap::Tag<ATuple>(), 500000);
    golap::MirrorMem probe_table(golap::Tag<ATuple>(), 10000000);
    golap::MirrorMem hits(golap::Tag<uint64_t>(), 1);
    hits.hst.ptr<uint64_t>()[0] = 0;
    hits.sync_to_device();

    for(int i = 0; i < build_table.size<ATuple>(); ++i){
        build_table.hst.ptr<ATuple>()[i].id = i;
    }
    for(int i = 0; i < probe_table.size<ATuple>(); ++i){
        probe_table.hst.ptr<ATuple>()[i].id = i%build_table.size<ATuple>();
    }
    build_table.sync_to_device();
    probe_table.sync_to_device();

    golap::HashMap hash_map(build_table.size<ATuple>(),build_table.dev.ptr<ATuple>());
    golap::hash_map_build<<<util::div_ceil(build_table.size<ATuple>(),512),512>>>(hash_map,build_table.size<ATuple>(),ATupleJoinKey());

    golap::hash_join_count<<<util::div_ceil(probe_table.size<ATuple>(),512),512>>>(hash_map, probe_table.dev.ptr<ATuple>(),
                                             ATupleJoinKey(),ATupleJoinKey(),
                                             probe_table.size<ATuple>(), hits.dev.ptr<uint64_t>());


    hits.sync_to_host();

    REQUIRE(hits.hst.ptr<uint64_t>()[0] == probe_table.size<ATuple>());



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}


TEST_CASE("Join Materialize 1to1", "[join]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    golap::MirrorMem build_table(golap::Tag<ATuple>(), 5000);
    golap::MirrorMem probe_table(golap::Tag<ATuple>(), 10000);
    golap::MirrorMem result_table(golap::Tag<MatTuple>(), 10000);
    golap::MirrorMem matches(golap::Tag<uint64_t>(), 1);
    matches.hst.ptr<uint64_t>()[0] = 0;
    matches.sync_to_device();

    for(int i = 0; i < build_table.size<ATuple>(); ++i){
        build_table.hst.ptr<ATuple>()[i].id = i;
    }
    for(int i = 0; i < probe_table.size<ATuple>(); ++i){
        probe_table.hst.ptr<ATuple>()[i].id = i%build_table.size<ATuple>();
    }
    build_table.sync_to_device();
    probe_table.sync_to_device();

    golap::HashMap hash_map(build_table.size<ATuple>(),build_table.dev.ptr<ATuple>());
    golap::hash_map_build<<<util::div_ceil(build_table.size<ATuple>(),512),512>>>(hash_map,build_table.size<ATuple>(),ATupleJoinKey());

    golap::hash_join_mat<<<util::div_ceil(probe_table.size<ATuple>(),512),512>>>(hash_map, probe_table.dev.ptr<ATuple>(),
                                             ATupleJoinKey(), ATupleJoinKey(),
                                             probe_table.size<ATuple>(), result_table.dev.ptr<MatTuple>(), matches.dev.ptr<uint64_t>());


    matches.sync_to_host();

    REQUIRE(matches.hst.ptr<uint64_t>()[0] == probe_table.size<ATuple>());



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("Column Join", "[join]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    struct BUILD{
        enum {ID=0,AFLOAT=1};
        uint64_t id;
        float afloat;
    };
    struct PROBE{
        enum {ID=0,SOMECHARS=1};
        uint64_t id;
        char somechars[12] ;
    };
    struct RES{
        enum {ID=0,SOMECHARS=1};
        uint64_t id; // not used yet
        decltype(PROBE::somechars) somechars;
    };
    // uint32_t BUILD_TABLE_NUM = 20;
    // uint32_t PROBE_TABLE_NUM = 50;
    uint32_t BUILD_TABLE_NUM = 2000000;
    uint32_t PROBE_TABLE_NUM = 5000000;
    golap::ColumnTable<golap::HostMem,decltype(BUILD::id),decltype(BUILD::afloat)> host_build_table{"id,afloat", BUILD_TABLE_NUM,1};
    golap::ColumnTable<golap::DeviceMem,decltype(BUILD::id),decltype(BUILD::afloat)> dev_build_table{"id,afloat", BUILD_TABLE_NUM,1};

    golap::ColumnTable<golap::HostMem,decltype(PROBE::id),decltype(PROBE::somechars)> host_probe_table{"id,somechars", PROBE_TABLE_NUM,1};
    golap::ColumnTable<golap::DeviceMem,decltype(PROBE::id),decltype(PROBE::somechars)> dev_probe_table{"id,somechars", PROBE_TABLE_NUM,1};

    golap::MirrorMem res_rids{golap::Tag<uint64_t>(), PROBE_TABLE_NUM};    

    golap::MirrorMem counter{golap::Tag<uint64_t>(), 1};
    counter.hst.ptr<uint64_t>()[0] = 0;
    counter.sync_to_device();

    for(int i = 0; i<BUILD_TABLE_NUM; ++i){
        host_build_table.col<BUILD::ID>().data()[i] = i;
        host_build_table.col<BUILD::AFLOAT>().data()[i] = i*3.0;
    }
    for(int i = 0; i<PROBE_TABLE_NUM; ++i){
        host_probe_table.col<PROBE::ID>().data()[i] = i%BUILD_TABLE_NUM;
        snprintf(host_probe_table.col<PROBE::SOMECHARS>().data()[i], 12, "%04d-%06d", i%BUILD_TABLE_NUM, i);
    }
    checkCudaErrors(cudaMemcpy(dev_build_table.col<BUILD::ID>().data(),
                               host_build_table.col<BUILD::ID>().data(),
                               sizeof(decltype(BUILD::id))*BUILD_TABLE_NUM, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(dev_build_table.col<BUILD::AFLOAT>().data(),
                               host_build_table.col<BUILD::AFLOAT>().data(),
                               sizeof(decltype(BUILD::afloat))*BUILD_TABLE_NUM, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(dev_probe_table.col<PROBE::ID>().data(),
                               host_probe_table.col<PROBE::ID>().data(),
                               sizeof(decltype(PROBE::id))*PROBE_TABLE_NUM, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(dev_probe_table.col<PROBE::SOMECHARS>().data(),
                               host_probe_table.col<PROBE::SOMECHARS>().data(),
                               sizeof(decltype(PROBE::somechars))*PROBE_TABLE_NUM, cudaMemcpyDefault));


    golap::HashMap hash_map(dev_build_table.col<BUILD::ID>().size(),dev_build_table.col<BUILD::ID>().data());
    util::Timer timer;
    golap::hash_map_build<<<util::div_ceil(dev_build_table.col<BUILD::ID>().size(),512),512>>>(hash_map,BUILD_TABLE_NUM,golap::DirectKey<uint64_t>());

    golap::hash_join_rids<<<util::div_ceil(dev_probe_table.col<PROBE::ID>().size(),512),512>>>(hash_map,dev_probe_table.col<PROBE::ID>().data(),
                                                                                                golap::DirectKey<uint64_t>(), golap::DirectKey<uint64_t>(),
                                                                                                dev_probe_table.col<PROBE::ID>().size(),
                                                                                                res_rids.dev.ptr<uint64_t>(),
                                                                                                counter.dev.ptr<uint64_t>());

    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Join took "<<timer.elapsed() << "ms.\n";

    counter.sync_to_host();
    res_rids.sync_to_host();

    REQUIRE(counter.hst.ptr<uint64_t>()[0] == dev_probe_table.col<PROBE::ID>().size());



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("UM Column Join", "[join]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    struct BUILD{
        enum {ID=0,AFLOAT=1};
        uint64_t id;
        float afloat;
    };
    struct PROBE{
        enum {ID=0,SOMECHARS=1};
        uint64_t id;
        char somechars[12] ;
    };
    struct RES{
        enum {ID=0,SOMECHARS=1};
        uint64_t id; // not used yet
        decltype(PROBE::somechars) somechars;
    };
    uint32_t BUILD_TABLE_NUM = 10;
    uint32_t PROBE_TABLE_NUM = 50;
    // uint32_t BUILD_TABLE_NUM = 2000000;
    // uint32_t PROBE_TABLE_NUM = 5000000;
    golap::ColumnTable<golap::HostMem,decltype(BUILD::id),decltype(BUILD::afloat)> host_build_table{"id,afloat", BUILD_TABLE_NUM,1};
    golap::ColumnTable<golap::DeviceMem,decltype(BUILD::id),decltype(BUILD::afloat)> dev_build_table{"id,afloat", BUILD_TABLE_NUM,1};

    golap::ColumnTable<golap::HostMem,decltype(PROBE::id),decltype(PROBE::somechars)> host_probe_table{"id,somechars", PROBE_TABLE_NUM,1};
    golap::ColumnTable<golap::DeviceMem,decltype(PROBE::id),decltype(PROBE::somechars)> dev_probe_table{"id,somechars", PROBE_TABLE_NUM,1};

    golap::MirrorMem res_rids{golap::Tag<uint64_t>(), PROBE_TABLE_NUM};    

    golap::MirrorMem counter{golap::Tag<uint64_t>(), 1};
    counter.hst.ptr<uint64_t>()[0] = 0;
    counter.sync_to_device();

    for(int i = 0; i<BUILD_TABLE_NUM; ++i){
        host_build_table.col<BUILD::ID>().data()[i] = i;
        host_build_table.col<BUILD::AFLOAT>().data()[i] = i*3.0;
    }
    for(int i = 0; i<PROBE_TABLE_NUM; ++i){
        host_probe_table.col<PROBE::ID>().data()[i] = i%BUILD_TABLE_NUM;
        snprintf(host_probe_table.col<PROBE::SOMECHARS>().data()[i], 12, "%04d-%06d", i%BUILD_TABLE_NUM, i);
    }
    checkCudaErrors(cudaMemcpy(dev_build_table.col<BUILD::ID>().data(),
                               host_build_table.col<BUILD::ID>().data(),
                               sizeof(decltype(BUILD::id))*BUILD_TABLE_NUM, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(dev_build_table.col<BUILD::AFLOAT>().data(),
                               host_build_table.col<BUILD::AFLOAT>().data(),
                               sizeof(decltype(BUILD::afloat))*BUILD_TABLE_NUM, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(dev_probe_table.col<PROBE::ID>().data(),
                               host_probe_table.col<PROBE::ID>().data(),
                               sizeof(decltype(PROBE::id))*PROBE_TABLE_NUM, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(dev_probe_table.col<PROBE::SOMECHARS>().data(),
                               host_probe_table.col<PROBE::SOMECHARS>().data(),
                               sizeof(decltype(PROBE::somechars))*PROBE_TABLE_NUM, cudaMemcpyDefault));

    golap::UnifiedMem RIDChains{golap::Tag<uint64_t>{}, dev_build_table.col<BUILD::ID>().size()};
    golap::UnifiedMem lastRIDs{golap::Tag<uint64_t>{}, dev_build_table.col<BUILD::ID>().size()};
    golap::DeviceMem filled{golap::Tag<uint64_t>{}, 1};

    golap::HashMap hash_map(dev_build_table.col<BUILD::ID>().size(),dev_build_table.col<BUILD::ID>().data(),
                            RIDChains.ptr<uint64_t>(),lastRIDs.ptr<uint64_t>(), filled.ptr<uint64_t>());
    util::Timer timer;
    golap::hash_map_build<<<util::div_ceil(dev_build_table.col<BUILD::ID>().size(),512),512>>>(hash_map,BUILD_TABLE_NUM,golap::DirectKey<uint64_t>());

    golap::hash_join_rids<<<util::div_ceil(dev_probe_table.col<PROBE::ID>().size(),512),512>>>(hash_map,dev_probe_table.col<PROBE::ID>().data(),
                                                                                                golap::DirectKey<uint64_t>(), golap::DirectKey<uint64_t>(),
                                                                                                dev_probe_table.col<PROBE::ID>().size(),
                                                                                                res_rids.dev.ptr<uint64_t>(),
                                                                                                counter.dev.ptr<uint64_t>());

    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Join took "<<timer.elapsed() << "ms.\n";

    counter.sync_to_host();
    res_rids.sync_to_host();

    REQUIRE(counter.hst.ptr<uint64_t>()[0] == dev_probe_table.col<PROBE::ID>().size());



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

