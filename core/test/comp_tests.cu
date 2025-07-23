#ifdef WITH_CPU_COMP
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "access.hpp"
#include "mem.hpp"
#include "table.hpp"
#include "util.hpp"
#include "comp.cuh"
#include "comp_cpu.hpp"


TEST_CASE("CPU", "[comp-cpu]") {

    auto names = {"LZ4","Snappy"};
    auto size_bytess = {(1<<12),(1<<15),(1<<20),(1<<25)};

    for (auto size_bytes: size_bytess){  
        for (auto name: names){

            golap::Column<golap::HostMem,char> start_col(size_bytes,size_bytes,"start");
            golap::Column<golap::HostMem,char> comp_col(size_bytes<<1,size_bytes<<1,"compressed");
            golap::Column<golap::HostMem,char> end_col(size_bytes,size_bytes,"end");
            golap::Column<golap::HostMem,char> reference_col(size_bytes,size_bytes,"reference");

            for (uint64_t i = 0; i< start_col.size(); ++i){
                auto val = (uint8_t) ( size_bytes-i );
                start_col.data()[i] = val;
                reference_col.data()[i] = val;
            }

            auto manager = golap::build_cpucomp_manager(name);

            auto comp_return = manager->compress(start_col.data(),comp_col.data(),start_col.size_bytes());
            auto decomp_return = manager->decompress(end_col.data(),comp_col.data(),comp_return,start_col.size_bytes());
            std::cout << name << " manager   compress: "<< comp_return << "\n";
            std::cout << name << " manager decompress: "<< decomp_return << "\n";


            for (uint64_t i = 0; i< start_col.size(); ++i){
                REQUIRE(start_col.data()[i] == end_col.data()[i]);
            }
        }
    }

}

#endif
