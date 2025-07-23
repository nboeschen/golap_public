#include <gflags/gflags.h>
#include "helper_cuda.h"

#include "core.hpp"
#include "apps.cuh"
#include "util.hpp"
#include "storage.hpp"
#include "hl/select.cuh"

DEFINE_uint32(cuda_device, 0, "Index of CUDA device to use.");
DEFINE_uint32(block_limit, 40, "Limit the number of scheduled blocks per kernel call.");
DEFINE_uint32(scale_factor, 10, "Scale factor == size in GB.");
DEFINE_string(core_pin, "", "Empty or list of dash delimited cpu ids. If set, the worker threads will be pinned to the specific cores (if possible). Currently only used for select.");
DEFINE_string(workers, "1", "Number of independent execution workers.");
DEFINE_string(chunk_bytes, "0", "Chunk sizes in bytes, delimited by comma.");
DEFINE_uint64(nvchunk, (1<<16), "NVChunk size.");
DEFINE_int32(num_RLEs, 2, "Cascaded scheme only: Number of Run-Length-Encoding passes.");
DEFINE_int32(num_deltas, 1, "Cascaded scheme only: Number of delta passes.");
DEFINE_int32(use_bp, 1, "Cascaded scheme only: Whether to bitpack the final results.");
DEFINE_uint64(store_offset, (1<<30), "Offset on block device.");
DEFINE_uint32(repeat, 3, "Experiment repetition num.");
DEFINE_string(print, "pretty", "Print output. Options are [pretty,csv,csv_header]");
DEFINE_string(query, "", "Querys to execute, delimited by comma.");
DEFINE_string(dataflow, "SSD2GPU2CPU", "[SSD2GPU2CPU, SSD2CPU], delimited by comma.");
DEFINE_string(comp_algo, "LZ4", "Compression algorithms to use, delimited by comma. [LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS,NONE]");
DEFINE_string(pruning, "DONTPRUNE", "Chunk pruning method to use. [DONTPRUNE,MINMAX,HIST]");
DEFINE_uint64(pruning_param, 16, "Pruning parameter (e.g. num buckets)");
DEFINE_double(pruning_p, 0.01, "Bloom Pruning parameter p");
DEFINE_uint64(pruning_m, 8192, "Bloom Pruning parameter m");
DEFINE_string(col_filter_lo, "", "Range start for column prune.");
DEFINE_string(col_filter_hi, "", "Range stop for column prune. For bloom, has to match col_filter_lo");
DEFINE_int64(sim_compute_us, -1, "For select queries: Add this many us of simulated computation.");
DEFINE_bool(event_sync, true, "Sync inside a pipeline. true: sync event (allow overlap), false: sync stream (no overlap)");
DEFINE_string(aws_endpoint, "", "AWS Endpoint");
DEFINE_string(max_gpu_um_memory, "0", "Limiting UM memory, by reserving remaining GPU memory except max_gpu_um_memory. 0 for no limit.");

DEFINE_bool(verify,false,"Verify Result.");
DEFINE_string(sort_by, "natural", "Column to sort on"); // NOT used here
DEFINE_string(cluster_algo, "kmeans", "Cluster algo [kmeans,dbscan]");
DEFINE_uint64(cluster_param_max_tuples, (1<<22), "Split down large cluster to contain this number of tuples");
DEFINE_uint64(cluster_param_k, 64, "Number of clusters");
DEFINE_uint32(cluster_param_rounds, 16, "Pruning parameter (rounds)");
DEFINE_double(shuffle_ratio, 0.0, "Ratio of number of swaps to table size.");

// serialization options
DEFINE_string(ssdpath, "", "Path to block device or file.");

using DATA_TYPE = uint32_t;

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    checkCudaErrors(cudaSetDevice(FLAGS_cuda_device));

    if(FLAGS_print == "csv_header"){
        std::cout << golap::Parameter::csv_header() << "\n";
        return 0;
    }
    bool pretty_print = FLAGS_print == "pretty";

    if (FLAGS_ssdpath != "") golap::StorageManager::get().init(FLAGS_ssdpath,FLAGS_store_offset);

    golap::VarLoader<char>::EVENT_SYNC = FLAGS_event_sync;
    golap::VarLoaderCPU<char>::EVENT_SYNC = FLAGS_event_sync;

    golap::Parameter var{FLAGS_cuda_device, FLAGS_block_limit, "", "SSD2GPU", FLAGS_scale_factor, FLAGS_core_pin, 8, 0,
                FLAGS_nvchunk, FLAGS_num_RLEs, FLAGS_num_deltas, FLAGS_use_bp, FLAGS_store_offset, "",
                FLAGS_verify, "natural", 0, "", FLAGS_pruning_param, FLAGS_pruning_p, FLAGS_pruning_m,
                "", "", FLAGS_sim_compute_us, FLAGS_event_sync};
    uint64_t dummy;
    checkCudaErrors(cudaMemGetInfo(&dummy,&var.device_mem_total));
    { // stack for cuda-memcheck

    uint64_t num_tuples = (uint64_t)FLAGS_scale_factor * (uint64_t)(1<<30) /sizeof(DATA_TYPE);
    golap::ColumnTable<golap::HostMem,DATA_TYPE> table{"bench_column", num_tuples};


    auto comp_algos = util::str_split(FLAGS_comp_algo,",");
    auto workerss = util::str_split(FLAGS_workers,",");
    auto querys = util::str_split(FLAGS_query,",");
    auto chunk_bytess = util::str_split(FLAGS_chunk_bytes,",");
    auto dataflows = util::str_split(FLAGS_dataflow,",");
    auto sort_bys = util::str_split(FLAGS_sort_by,",");
    auto max_gpu_um_memorys = util::str_split(FLAGS_max_gpu_um_memory,",");
    auto prunings = util::str_split(FLAGS_pruning,",");
    auto col_filter_los = util::str_split(FLAGS_col_filter_lo,",");
    auto col_filter_his = util::str_split(FLAGS_col_filter_hi,",");
    if (col_filter_los.size() != col_filter_his.size()){
        util::Log::get().error_fmt("Col filter sizes are different: %llu vs %llu, exiting!", col_filter_los.size(), col_filter_his.size());
        return 1;
    }
    // todo
    if (col_filter_los.size() == 0 && var.query.find("filter") == std::string::npos){
        // unused column filters
        col_filter_los.emplace_back("-");
        col_filter_his.emplace_back("-");
    }
    uint64_t min_chunk_bytes = (uint64_t) -1;
    for (auto &chunk_bytes:chunk_bytess){
        min_chunk_bytes = std::min(min_chunk_bytes,std::stoul(chunk_bytes));
    }

    auto init_col = [&](std::string& query){
        util::ThreadPool pool;
        pool.parallel_n(8, [&](int tid) {
            auto [start, stop] = util::RangeHelper::nth_chunk(0, num_tuples, 8, tid);
            if (query == "iota"){
                for (uint64_t i = start; i<stop; ++i){
                    table.col<0>().data()[i] = i;
                }
            }else if(util::starts_with(query, "uniform")){
                uint64_t range_end = num_tuples;
                // e.g. uniform5000 sets range_end to be 5000
                if (query != "uniform") range_end = std::stoul(query.substr(7));

                uint64_t seed = 0xC0FFEE ^ tid;
                for (uint64_t i = start; i<stop; ++i){
                    table.col<0>().data()[i] = (DATA_TYPE) util::uniform_int(seed,0,range_end);
                }
            }else if(util::starts_with(query, "ratio")){
                // e.g. ratio5 tries to generate data that can be compressed close to compression ratio 5 (for the smallest chunk size),
                // by populating chunks with 1/5th of the chunks with random data, and the rest with constants

                uint64_t ratio = std::stoul(query.substr(5));
                uint64_t range_end = num_tuples;
                uint64_t seed = 0xC0FFEE ^ tid;
                uint64_t chunk_bytes_in_tuples = min_chunk_bytes / sizeof(DATA_TYPE);
                uint64_t random_tuples = chunk_bytes_in_tuples * (1.0 / ratio);
                uint64_t i = start;
                // if (tid == 0){
                //     util::Log::get().info_fmt("min_chunk_bytes=%llu, chunk_bytes_in_tuples=%llu, random_tuples=%llu, start=%llu,stop=%llu",
                //                               min_chunk_bytes, chunk_bytes_in_tuples, random_tuples, start,stop);
                // }

                while (i < stop){
                    // first part random
                    for (; (i % chunk_bytes_in_tuples)< random_tuples; ++i){
                        table.col<0>().data()[i] = (DATA_TYPE) util::uniform_int(seed,0,range_end);
                    }
                    // second part constant
                    for (; (i % chunk_bytes_in_tuples) != 0; ++i){
                        table.col<0>().data()[i] = 0xC0FFEE;
                    }
                }
            }else {
                util::Log::get().warn_fmt("Unknown query, data unchanged: %s", query);
            }
        });
        pool.join();
        util::Log::get().info_fmt("Benchmark column (re-)initialized!");
    };


    for (int i = 0;i<FLAGS_repeat; ++i){
        for (auto &dataflow: dataflows){
            var.dataflow = dataflow;
            for (auto &query: querys){

                do {
                    if (query == "interactive"){
                        if (!util::ask("Query to run ('break' to break): ",var.query)) break;
                    }else{
                        var.query = query;
                    }

                    init_col(var.query);
                    if (FLAGS_shuffle_ratio > 0.0){
                        table.shuffle_ratio(FLAGS_shuffle_ratio);
                    }

                    for (auto &workers : workerss){
                        var.workers = std::stoul(workers);
                        for (auto &comp_algo : comp_algos){
                            var.comp_algo = comp_algo;
                            for (auto &pruning: prunings){
                                var.pruning = pruning;

                                for (int i = 0; i < col_filter_los.size(); ++i){
                                    var.col_filter_lo = col_filter_los[i];
                                    var.col_filter_hi = col_filter_his[i];

                                    for (auto &max_gpu_um_memory: max_gpu_um_memorys){
                                        var.max_gpu_um_memory = std::stoul(max_gpu_um_memory);

                                        for (auto &chunk_bytes:chunk_bytess){
                                            var.chunk_bytes = std::stoul(chunk_bytes);

                                            auto res =  golap::select(var,table.col<0>(),num_tuples);

                                            if (res){
                                                std::cout << var.repr(pretty_print) << "\n";
                                            }else {
                                                std::cout << "##############\nThis configuration failed:\n##############"<<var.repr(pretty_print) << "\n";
                                            }
                                            golap::StorageManager::get().set_offset(FLAGS_store_offset);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } while(query == "interactive");
            }
        }
    }


    } // stack for cuda-memcheck

    checkCudaErrors(cudaDeviceReset());
}
