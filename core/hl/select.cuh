#pragma once

#include "core.hpp"

namespace golap {


/**
 * Generic Select helper.
 */
template <typename T, typename NVCOMPTYPE=T>
bool select(golap::Parameter &var, golap::Column<golap::HostMem,T> &col, uint64_t num_tuples,
            std::unordered_map<std::string,std::string> const *BEST_BW_COMP = nullptr,
            std::unordered_map<std::string,std::string> const *BEST_RATIO_COMP = nullptr){
    auto comp_algo = var.comp_algo;

    if(BEST_BW_COMP != nullptr && var.comp_algo == "BEST_BW_COMP") comp_algo = BEST_BW_COMP->at(col.attr_name);
    if(BEST_RATIO_COMP != nullptr && var.comp_algo == "BEST_RATIO_COMP") comp_algo = BEST_RATIO_COMP->at(col.attr_name);

    golap::CompInfo compinfo{var.chunk_bytes,num_tuples*sizeof(T),comp_algo,var.nvchunk};
    auto host_column_alloc_unit = var.chunk_bytes;
    if (var.chunk_size_vec.size() != 0){
        // todo: fix this, this should be translated from tuples# to bytes
        compinfo.chunk_size_vec = var.chunk_size_vec;
        for (auto &tup_count : var.chunk_size_vec) compinfo.chunk_size_vec.push_back(tup_count*sizeof(T));
        compinfo.chunk_bytes = (uint64_t)-1;
        host_column_alloc_unit = (*std::max_element(compinfo.chunk_size_vec.begin(),compinfo.chunk_size_vec.end()));
    }
    golap::HostMem host_column{golap::Tag<char>(), compinfo.uncomp_bytes, host_column_alloc_unit};

    if (var.dataflow == "SSD2CPU2GPU" || var.dataflow == "SSD2GPU2CPU" || var.dataflow == "SSD2GPU"){
        if (var.comp_algo == "UNCOMPRESSED"){
            var.comp_ms = golap::prepare_uncompressed(col, num_tuples, compinfo);
        }else{
            var.comp_ms = golap::prepare_compressed_device<golap::HostMem,T,NVCOMPTYPE>(col, num_tuples, compinfo);
        }
        if (var.comp_algo == "UNCOMPRESSED"){
            var.time_ms = golap::load<golap::LoadEnv, T, NVCOMPTYPE>(var.workers, var.core_pin, var.dataflow, var.cuda_device,
                                                                     host_column.ptr<T>(), compinfo, var.host_mem_used, var.device_mem_used,
                                                                     var.simulate_compute_us);
        }else{
            var.time_ms = golap::load<golap::DecompressEnv, T, NVCOMPTYPE>(var.workers, var.core_pin, var.dataflow, var.cuda_device,
                                                                           host_column.ptr<T>(), compinfo, var.host_mem_used, var.device_mem_used,
                                                                           var.simulate_compute_us);
        }
        var.comp_bytes = compinfo.get_comp_bytes();
        var.uncomp_bytes = compinfo.uncomp_bytes;
    }else if(var.dataflow == "SSD2GPU2CPU_ASYNC" || var.dataflow == "SSD2GPU_ASYNC"){
    }else if(var.dataflow == "S32CPU2GPU"){
    }else if(var.dataflow == "S32CPU2GPU_ASYNC"){
    }else if(var.dataflow == "SSD2CPU"){
#ifdef WITH_CPU_COMP
        if (var.comp_algo == "UNCOMPRESSED"){
            var.comp_ms = golap::prepare_uncompressed(col, num_tuples, compinfo);
            var.time_ms = golap::load<golap::LoadEnvCPU, T, NVCOMPTYPE>(var.workers, var.core_pin, var.dataflow, var.cuda_device,
                                                                        host_column.ptr<T>(), compinfo, var.host_mem_used, var.device_mem_used,
                                                                        var.simulate_compute_us);
        }else{
            var.comp_ms = golap::prepare_compressed_host<golap::HostMem,T>(col, num_tuples, compinfo);
            var.time_ms = golap::load<golap::DecompressEnvCPU, T, NVCOMPTYPE>(var.workers, var.core_pin, var.dataflow, var.cuda_device,
                                                                              host_column.ptr<T>(), compinfo, var.host_mem_used, var.device_mem_used,
                                                                              var.simulate_compute_us);
        }
        var.comp_bytes = compinfo.get_comp_bytes();
        var.uncomp_bytes = compinfo.uncomp_bytes;
#else
        util::Log::get().error_fmt("Select with SSD2CPU: Not compiled WITH_CPU_COMP!");
        return false;
#endif //WITH_CPU_COMP
    }else{
        util::Log::get().error_fmt("Select: Probably unknown dataflow: %s", var.dataflow.c_str());
        return false;
    }

    if (var.verify){
        if (var.dataflow == "SSD2GPU" || var.dataflow == "SSD2CPU2GPU"){
            util::Log::get().error_fmt("Verify not implemented for dataflow %s!",var.dataflow.c_str());
            return false;
        }
        golap::verify<T,NVCOMPTYPE>(col, host_column.ptr<T>(), num_tuples);
    }
    return true;
}

/**
 * Generic filter helper.
 */
template <typename T, typename NVCOMPTYPE=T>
bool filter(golap::Parameter &var, golap::Column<golap::HostMem,T> &col, uint64_t num_tuples,
            std::unordered_map<std::string,std::string> const *BEST_BW_COMP = nullptr,
            std::unordered_map<std::string,std::string> const *BEST_RATIO_COMP = nullptr){
    auto comp_algo = var.comp_algo;

    T lo,hi;
    if constexpr(std::is_same_v<T,uint8_t>){
        uint16_t x,y;
        std::stringstream{var.col_filter_lo} >> x;
        std::stringstream{var.col_filter_hi} >> y;
        lo = static_cast<uint8_t>(x);
        hi = static_cast<uint8_t>(y);
    }else{
        std::stringstream{var.col_filter_lo} >> lo;
        std::stringstream{var.col_filter_hi} >> hi;
    }

    if(BEST_BW_COMP != nullptr && var.comp_algo == "BEST_BW_COMP") comp_algo = BEST_BW_COMP->at(col.attr_name);
    if(BEST_RATIO_COMP != nullptr && var.comp_algo == "BEST_RATIO_COMP") comp_algo = BEST_RATIO_COMP->at(col.attr_name);

    golap::CompInfo compinfo{var.chunk_bytes,num_tuples*sizeof(T),comp_algo,var.nvchunk};
    if (var.chunk_size_vec.size() != 0){
        compinfo.chunk_size_vec = var.chunk_size_vec;
        compinfo.chunk_bytes = (uint64_t)-1;
    }
    golap::HostMem host_column{golap::Tag<char>(), compinfo.uncomp_bytes, var.chunk_bytes};

    if (var.dataflow != "SSD2CPU2GPU" && var.dataflow != "SSD2GPU"){
        util::Log::get().error_fmt("Dataflow %s not implemented for filter!",var.dataflow.c_str());
        return false;
    }

    // 0) prepare metadata store
    golap::MinMaxMeta<T> minmax;
    golap::EqHistogram<T> hist(var.pruning_param);
    golap::BloomMeta<T> bloom(var.pruning_p, var.pruning_m);

    // 1) prepare column in storage, TODO gather metadata
    if (var.comp_algo == "UNCOMPRESSED"){
        var.comp_ms = golap::prepare_uncompressed(col, num_tuples, compinfo, &minmax, &hist, &bloom);
    }else{
        var.comp_ms = golap::prepare_compressed_device<golap::HostMem,T,NVCOMPTYPE>(col, num_tuples, compinfo, &minmax, &hist, &bloom);
    }
    golap::MirrorMem prune_check{golap::Tag<uint16_t>{}, compinfo.blocks.size()};
    prune_check.dev.set(0);
    uint64_t block_num = std::min((long long)var.block_limit, util::div_ceil(compinfo.blocks.size(),512));


    // 2) (optional) pruning
    golap::CStream meta_stream("metadatapruning");
    util::Timer prune_timer;
    if (var.pruning == "MINMAX"){
        golap::check_mmmeta<<<block_num,512,0,meta_stream.stream>>>(minmax, prune_check.dev.ptr<uint16_t>(), lo, hi);
    }else if (var.pruning == "HIST"){
        golap::check_hist<<<block_num,512,0,meta_stream.stream>>>(hist, prune_check.dev.ptr<uint16_t>(), lo, hi);
    }else if (var.pruning == "BLOOM"){
        if (lo != hi){
            util::Log::get().warn_fmt("Pruning is set to Bloom, but lo and hi filter are not equal!");
        }
        golap::check_bloom<<<block_num,512,0,meta_stream.stream>>>(bloom, prune_check.dev.ptr<uint16_t>(), lo);
    }else {
        util::Log::get().warn_fmt("Filter column query, but pruning not set. Probably not what you want!");
    }
    if (var.pruning != "DONTPRUNE") prune_check.sync_to_host(meta_stream.stream);

    checkCudaErrors(cudaStreamSynchronize(meta_stream.stream));
    var.prune_ms = prune_timer.elapsed();

    // 3) load and filter
    if (var.comp_algo == "UNCOMPRESSED"){
        var.time_ms = golap::load<golap::LoadEnv, T, NVCOMPTYPE>(var.workers, var.core_pin, var.dataflow, var.cuda_device,
                                                                 host_column.ptr<T>(), compinfo, var.host_mem_used, var.device_mem_used,
                                                                 var.simulate_compute_us, &prune_check);
    }else{
        var.time_ms = golap::load<golap::DecompressEnv, T, NVCOMPTYPE>(var.workers, var.core_pin, var.dataflow, var.cuda_device,
                                                                       host_column.ptr<T>(), compinfo, var.host_mem_used, var.device_mem_used,
                                                                       var.simulate_compute_us, &prune_check);
    }
    var.comp_bytes = compinfo.get_comp_bytes();
    var.uncomp_bytes = compinfo.uncomp_bytes;

    return true;
}

} // end of namespace
