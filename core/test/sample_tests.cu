#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>

#include "helper_cuda.h"

#include "test_common.hpp"

#include "hl/sample.cuh"
#include "storage.hpp"
#include "util.hpp"

TEST_CASE("Test sample", "[sample]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t ROW_NUM = 50000000;

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);

    struct COLS{
        enum {ENTE=0,GANS=1};
    };

    golap::ColumnTable<golap::HostMem,uint32_t,double> table{"ente,gans", ROW_NUM};
    table.num_tuples = ROW_NUM;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
    }

    std::vector<uint64_t> chunk_bytess{(1<<20),(1<<21),(1<<22)};
    std::vector<std::string> comp_algos{"LZ4","Cascaded","Snappy","Gdeflate","Bitcomp","ANS"};

    table.apply([&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
        auto ress = sample_comps(a_col,
                                num_tuples, 0.8, 1,
                                    chunk_bytess,
                                   comp_algos);
        printf("---------------------------\nResults for Column:%s\n",a_col.attr_name.c_str());
        for (auto& res :ress){
            std::cout << res << "\n";
        }
        auto best_bw = *std::max_element(ress.begin(), ress.end(), [](golap::SampleRes &a, golap::SampleRes &b) {
            return (a.uncomp_bytes/a.decomp_time) < (b.uncomp_bytes/b.decomp_time);
        });
        auto best_ratio = *std::min_element(ress.begin(), ress.end(), [](golap::SampleRes &a, golap::SampleRes &b) {
            return a.comp_bytes < b.comp_bytes;
        });
        std::cout << "Best BW is:\n\t"<<best_bw<< "\n";
        std::cout << "Best Ratio is:\n\t"<<best_ratio<< "\n";

    });

    // sample with verification
    table.apply([&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
        auto ress = sample_comps(a_col,
                                num_tuples, 0.5, 1,
                                    chunk_bytess,
                                   comp_algos, true);

    });


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

