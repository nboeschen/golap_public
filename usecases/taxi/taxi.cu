#include <iostream>
#include <helper_cuda.h>
#include <gflags/gflags.h>
#include <tuple>
#include <unordered_map>
#include <type_traits>
#include <memory>

#include "hl/serialization.hpp"
#include "taxi.hpp"
#include "helper.hpp"
#include "hl/cluster.cuh"

DEFINE_uint32(cuda_device, 0, "Index of CUDA device to use.");
DEFINE_uint32(block_limit, 16, "Limit the number of scheduled blocks per kernel call.");
DEFINE_uint32(scale_factor, 10, "Scale factor.");
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

DEFINE_bool(verify,false,"Verify Result.");
DEFINE_string(sort_by, "natural", "Column to sort on");
DEFINE_string(cluster_algo, "kmeans", "Cluster algo [kmeans,dbscan]");
DEFINE_uint64(cluster_param_max_tuples, (1<<22), "Split down large cluster to contain this number of tuples");
DEFINE_uint64(cluster_param_k, 64, "Number of clusters");
DEFINE_uint32(cluster_param_rounds, 16, "Pruning parameter (rounds)");
DEFINE_double(shuffle_ratio, 0.0, "Ratio of number of swaps to table size.");
DEFINE_string(max_gpu_um_memory, "0", "Limiting UM memory, by reserving remaining GPU memory except max_gpu_um_memory. 0 for no limit.");
// serialization options
DEFINE_string(ssdpath, "/dev/md127", "Path to block device or file.");
DEFINE_string(dbpath, "", "Path to saved disk db.");
DEFINE_string(csv_delimiter, ";", "Delimiter of csv diskdb.");
DEFINE_string(init_variant, "", "Variant of tables: [init_populate,init_only]");


int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    checkCudaErrors(cudaSetDevice(FLAGS_cuda_device));

    if(FLAGS_print == "csv_header"){
        std::cout << TaxiVar::csv_header() << "\n";
        return 0;
    }
    bool pretty_print = FLAGS_print == "pretty";

    golap::StorageManager::get().init(FLAGS_ssdpath,FLAGS_store_offset);
    golap::VarLoader<char>::EVENT_SYNC = FLAGS_event_sync;
    golap::VarLoaderCPU<char>::EVENT_SYNC = FLAGS_event_sync;
    TaxiVar var{FLAGS_cuda_device, FLAGS_block_limit, "", "SSD2GPU", FLAGS_scale_factor, FLAGS_core_pin, 8, 0,
                FLAGS_nvchunk, FLAGS_num_RLEs, FLAGS_num_deltas, FLAGS_use_bp, FLAGS_store_offset, "",
                FLAGS_verify, "natural", 0, "", FLAGS_pruning_param, FLAGS_pruning_p, FLAGS_pruning_m,
                "", "", FLAGS_sim_compute_us, FLAGS_event_sync};
    uint64_t dummy;
    checkCudaErrors(cudaMemGetInfo(&dummy,&var.device_mem_total));

    { // new stack for cuda-memcheck

    TaxiColLayout db(var, FLAGS_dbpath == "" ? "init_populate" : "init_only");
    if (FLAGS_dbpath != ""){
        util::Log::get().info_fmt("Loading table data from disk ...");
        if (util::ends_with(FLAGS_dbpath,".dat")) golap::read_col_db_bin(db.tables,FLAGS_dbpath);
        else if (util::ends_with(FLAGS_dbpath,".csv")) golap::read_col_db_csv(db.tables,FLAGS_dbpath,FLAGS_csv_delimiter);
        else {
            std::cout << "Unknown file extension for diskdb, exiting!\n";
            return 1;
        }
    }

    auto comp_algos = util::str_split(FLAGS_comp_algo,",");
    auto workerss = util::str_split(FLAGS_workers,",");
    auto query_funcs = util::str_split(FLAGS_query,",");
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

    // if (FLAGS_query.find("query") != std::string::npos){
    //     util::Log::get().info_fmt("Prejoining tables for metadata ...");
    //     SortHelper::get().prejoin_tables(db.tables, var.chunk_size_vec);
    //     SortHelper::get().prejoined->print_col_histogram();
    // }

    for (int i = 0;i<FLAGS_repeat; ++i){
        for (auto &sort_by: sort_bys){
            if (var.sort_by == sort_by) { /*pass*/
            }else if (sort_by.substr(0,8) == std::string("cluster|")){
                golap::ClusterHelper helper{sort_by, FLAGS_cluster_algo, FLAGS_cluster_param_max_tuples, FLAGS_cluster_param_rounds, FLAGS_cluster_param_k};
                helper.apply(db.tables.trips, var.chunk_size_vec);
            }else{
                auto& helper = SortHelper::get();
                util::Log::get().info_fmt("Sorting tables ...");
                helper.apply(sort_by, db.tables, var.chunk_size_vec);
            }
            var.sort_by = sort_by;

            // if (FLAGS_shuffle_ratio > 0.0){
            //     db.tables.trips.shuffle_ratio(FLAGS_shuffle_ratio);
            // }
            for (auto &dataflow: dataflows){
                var.dataflow = dataflow;
                for (auto &query_func_name: query_funcs){
                    auto query_func = &TaxiColLayout::select_VendorID;
                    var.query = query_func_name;

                    try {
                        if (var.dataflow == "INMEM") query_func = QUERY_FUNC_PTRS.at(query_func_name+"inmem");
                        else query_func = QUERY_FUNC_PTRS.at(query_func_name);
                    } catch (std::out_of_range const& exc) {
                        std::cout << "Skipping unknown query: "<<query_func_name << '\n';
                        continue;
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

                                        if(var.chunk_size_vec.size() != 0) chunk_bytess = {"-1"};
                                        for (auto &chunk_bytes:chunk_bytess){
                                            var.chunk_bytes = std::stoul(chunk_bytes);
                                            if ((db.*query_func)()){
                                                std::cout << var.repr(pretty_print) << "\n";
                                            }
                                            golap::StorageManager::get().set_offset(FLAGS_store_offset);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    } // end new stack for cuda-memcheck


    // for cuda-memcheck --leak-check 
    checkCudaErrors(cudaDeviceReset());
    return 0;
}

