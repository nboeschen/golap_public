#include <iostream>
#include <helper_cuda.h>
#include <gflags/gflags.h>
#include <tuple>
#include <unordered_map>

#include "ssb.hpp"
#include "hl/sample.cuh"
#include "hl/serialization.hpp"


DEFINE_uint32(cuda_device, 0, "Index of CUDA device to use.");
DEFINE_uint32(block_limit, 16, "Limit the number of scheduled blocks per kernel call.");
DEFINE_uint32(scale_factor, 10, "SSB scale factor.");
DEFINE_uint32(workers, 1, "SSB scale factor.");
DEFINE_string(chunk_bytes, "0", "Chunk sizes in bytes, delimited by comma.");
DEFINE_uint64(nvchunk, (1<<16), "NVChunk size.");
DEFINE_int32(num_RLEs, 2, "Cascaded scheme only: Number of Run-Length-Encoding passes.");
DEFINE_int32(num_deltas, 1, "Cascaded scheme only: Number of delta passes.");
DEFINE_int32(use_bp, 1, "Cascaded scheme only: Whether to bitpack the final results.");
DEFINE_uint64(store_offset, (1<<30), "Offset on block device.");
DEFINE_uint32(repeat, 3, "Experiment repetition num.");
DEFINE_string(customer_factor, "1", "Factor by which to increase customer table.");
DEFINE_string(comp_algo, "LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS", "Compression algorithms to use, delimited by comma. [LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS]");
DEFINE_string(sample_ratio, "0.25", "Fraction of lineorder table to sample");
DEFINE_string(sort_by, "none", "Column of lineorder to sort on"); // order of parameters here matters
DEFINE_string(columns, "", "Column of lineorder to sample"); // order of parameters here matters
// serialization options
DEFINE_string(ssdpath, "/raid/gds/300G.file", "Path to block device or file.");
DEFINE_string(dbpath, "", "Path to saved disk db.");
DEFINE_string(csv_delimiter, ";", "Delimiter of csv diskdb.");
DEFINE_string(init_variant, "", "Variant of tables: [init_populate,init_only]");


int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    checkCudaErrors(cudaSetDevice(FLAGS_cuda_device));

    golap::StorageManager::get().init(FLAGS_ssdpath, FLAGS_store_offset);
    SSBVar var{FLAGS_cuda_device, FLAGS_block_limit, "", "", FLAGS_scale_factor, "", FLAGS_workers, 0,
                FLAGS_nvchunk, FLAGS_num_RLEs, FLAGS_num_deltas, FLAGS_use_bp, FLAGS_store_offset, "",
                false, ""};
    uint64_t dummy;
    checkCudaErrors(cudaMemGetInfo(&dummy,&var.device_mem_total));

    { // new stack for cuda-memcheck

    SSBColLayout ssb(var, FLAGS_init_variant);
    if (FLAGS_dbpath != ""){
        if (FLAGS_init_variant != "init_only"){
            std::cout << "dbpath parameter is only allowed for init_variant==\"init_only\"! Exiting.\n";
            return 1;
        }
        util::Log::get().info_fmt("Loading table data from disk ...");
        if (util::ends_with(FLAGS_dbpath,".dat")) read_col_db_bin(ssb.tables,FLAGS_dbpath);
        else if (util::ends_with(FLAGS_dbpath,".csv")) read_col_db_csv(ssb.tables,FLAGS_dbpath,FLAGS_csv_delimiter);
        else {
            std::cout << "Unknown file extension for diskdb, exiting!\n";
            return 1;
        }
    }

    auto comp_algos = util::str_split(FLAGS_comp_algo,",");
    auto chunk_bytess = util::str_split(FLAGS_chunk_bytes,",");
    auto sort_bys = util::str_split(FLAGS_sort_by,",");
    auto customer_factors = util::str_split(FLAGS_customer_factor,",");
    auto columns = util::str_split(FLAGS_columns,",");
    auto sample_ratios = util::str_split(FLAGS_sample_ratio,",");

    std::vector<uint64_t> chunk_bytess_uint;
    for (auto & chunk_bytes:chunk_bytess){
        chunk_bytess_uint.push_back(std::stoul(chunk_bytes));
    }

    std::cout << golap::SampleRes::csv_header() << "\n";


    for (int i = 0;i<FLAGS_repeat; ++i){
        for (auto &customer_factor: customer_factors){
            if (var.customer_factor != std::stoul(customer_factor)){
                var.customer_factor = std::stoul(customer_factor);
                ssb.tables.customer_num = ssb.tables.scale_factor * 30000 * var.customer_factor;
                ssb.tables.init_customer();
                ssb.tables.populate_customer();
                ssb.tables.repopulate_custkey();
            }

            for (auto &sort_by: sort_bys){
                var.sort_by = sort_by;
                if (sort_by == "discount") ssb.tables.lineorder.sort_by<Lineorder::DISCOUNT>();
                else if (sort_by == "quantity") ssb.tables.lineorder.sort_by<Lineorder::QUANTITY>();
                
                for (auto &sample_ratio: sample_ratios){

                    ssb.tables.lineorder.apply([&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
                        if (std::find(columns.begin(), columns.end(), a_col.attr_name) == columns.end()) return;
                        auto ress = sample_comps(a_col,
                                                num_tuples, std::stod(sample_ratio), FLAGS_workers,
                                                   chunk_bytess_uint,
                                                   comp_algos);
                        // printf("---------------------------\nResults for Column:%s\n",a_col.attr_name.c_str());        
                        // for (auto& res :ress){
                        //     std::cout << res << "\n";
                        // }
                        auto best_bw = *std::max_element(ress.begin(), ress.end(), [](golap::SampleRes &a, golap::SampleRes &b) {
                            return (a.uncomp_bytes/a.decomp_time) < (b.uncomp_bytes/b.decomp_time);
                        });
                        auto best_ratio = *std::min_element(ress.begin(), ress.end(), [](golap::SampleRes &a, golap::SampleRes &b) {
                            return a.comp_bytes < b.comp_bytes;
                        });
                        best_bw.col_name += "_BW";
                        best_ratio.col_name += "_RATIO";
                        std::cout << best_bw.to_csv() << "\n";
                        std::cout << best_ratio.to_csv() << "\n";

                    });
                }            
                
            }
            
        }
    }


    } // end new stack for cuda-memcheck


    // for cuda-memcheck --leak-check 
    checkCudaErrors(cudaDeviceReset());
    return 0;
}

