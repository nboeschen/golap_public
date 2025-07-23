#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "mem.hpp"
#include "util.hpp"
#include "agg.cuh"


struct AggTuple{
    int id;
    char name[11];
    float discount;
    __host__ __device__
    uint64_t key(){return id;}
    friend std::ostream& operator<<(std::ostream &out, AggTuple const& obj){
        out << "AggTuple(id="<<obj.id<<", name="<<obj.name<<", discount="<<obj.discount<<")";
        return out;
    }
};

struct CustomAgg{
    __device__ __forceinline__
    AggTuple operator()(const AggTuple &a, const AggTuple &b) const {
        return AggTuple{0,"",a.discount + b.discount};
    }
};



TEST_CASE("Agg", "[agg-simple]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t ROW_NUM = (1<<22); // 4 Mio

    golap::MirrorMem table(golap::Tag<AggTuple>(), ROW_NUM);
    golap::MirrorMem agg(golap::Tag<AggTuple>(), 1);


    for(int i = 0; i < table.size<AggTuple>(); ++i){
        table.hst.ptr<AggTuple>()[i].discount = 0.1;
    }
    table.sync_to_device();

    golap::Aggregate aggregate(table.dev.ptr<AggTuple>(), agg.dev.ptr<AggTuple>(), table.dev.size<AggTuple>(),
                          CustomAgg(), AggTuple{0,"",0.0});

    aggregate.execute();

    agg.sync_to_host();


    REQUIRE(agg.hst.ptr<AggTuple>()[0].discount == Catch::Approx(ROW_NUM*0.1));



    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}



