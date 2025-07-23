#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include "../apps.cuh"
#include "../util.hpp"

namespace golap {

struct SampleRes{
    // in
    std::string col_name;
    std::string comp_algo;
    uint64_t uncomp_bytes;
    uint64_t chunk_bytes;
    double sample_ratio;
    uint32_t workers;

    // out
    uint64_t comp_bytes;
    double comp_time;
    double decomp_time;

    std::string to_csv(){
        std::stringstream ss;
        ss << col_name<<','<<comp_algo<<','<<uncomp_bytes<<','<<chunk_bytes<<','<<sample_ratio<<',';
        ss << workers<<','<<comp_bytes<<','<<comp_time<<','<<decomp_time;
        return ss.str();
    }

    std::string to_pretty(){
        std::stringstream ss;
        double bw = (1000.0 / (1<<30)) * ((double)uncomp_bytes/decomp_time);
        ss <<"SampleRes(col="<<col_name<<", comp_algo="<<comp_algo<<", uncomp_bytes="<<uncomp_bytes;
        ss <<", chunk_bytes="<<chunk_bytes<<", sample_ratio="<<sample_ratio<<", workers="<<workers<<", comp_bytes="<<comp_bytes<<", comp_time="<<comp_time;
        ss <<", decomp_time="<<decomp_time<<", BW="<<bw<<"GB/s)";
        return ss.str();
    }

    static std::string csv_header(){
        return "col_name,comp_algo,uncomp_bytes,chunk_bytes,sample_ratio,workers,comp_bytes,comp_time,decomp_time";
    }


    friend std::ostream& operator<<(std::ostream &out, SampleRes &obj){
        out << obj.to_pretty();
        return out;
    }
};

/**
 * For a given column, sample different chunkbytes and compalgos on a specific percentage of the columns tuples.
 * Populates the sample columns using full chunks of the original.
 */
template <typename MEM_TYPE, typename T, typename NVCOMPTYPE=T>
std::vector<SampleRes> sample_comps(golap::Column<MEM_TYPE,T> &col,
                                    uint64_t num_tuples, double sample_ratio, uint32_t workers,
                                    std::vector<uint64_t> &chunk_bytess,
                                    std::vector<std::string> &comp_algos,
                                    bool verify = false){
    std::vector<SampleRes> ress;
    uint64_t sample_tuples = (uint64_t) (sample_ratio * num_tuples);
    golap::CompInfo compinfo{0,sample_tuples*sizeof(T),"",(1<<16)};

    util::Log::get().debug_fmt("Sample comps: Sampling %lu tuples (of %lu total), %lu bytes.",sample_tuples,num_tuples,sample_tuples*sizeof(T));

    uint64_t max_chunk_size = *std::max_element(chunk_bytess.begin(),chunk_bytess.end());
    // we'll want to sample some chunks of the given column for compression metrics.
    // at the moment, its easiest to just copy portions of the original column:
    golap::Column<MEM_TYPE,T> copy_col{sample_tuples,
                                        // max chunk_size alloc_unit, so we dont have to reallocate
                                        max_chunk_size,
                                        col.attr_name+"_sample"};
    golap::Column<HostMem,T> host_cmp{sample_tuples,max_chunk_size,
                                        col.attr_name+"_host"};
    golap::Column<HostMem,T> host_col{sample_tuples,max_chunk_size,
                                        col.attr_name+"_host"};

    uint64_t original_full_chunks = col.size_bytes() / max_chunk_size;

    uint64_t tuples_per_max_chunk = max_chunk_size / sizeof(T);
    uint64_t tuples_remaining = sample_tuples;
    uint64_t cur_offset = 0;
    util::Log::get().debug_fmt("num_tuples=%lu, original_full_chunks=%lu, sample_tuples=%lu, sample_bytes=%lu, sample num=%lu",
                              num_tuples, original_full_chunks, sample_tuples, sample_tuples*sizeof(T), sample_tuples / tuples_per_max_chunk + 1);
    auto sample_chunks = util::sample_range(0, original_full_chunks, sample_tuples / tuples_per_max_chunk + 1,
                                            util::Timer::time_seed());
    for (auto& sample_chunk : sample_chunks){
        uint64_t tuples_this_chunk = std::min(tuples_remaining, tuples_per_max_chunk);

        checkCudaErrors(cudaMemcpy(copy_col.data()+cur_offset,
                                   col.data()+tuples_per_max_chunk*sample_chunk,
                                   tuples_this_chunk * sizeof(T),
                                   cudaMemcpyDefault));
        tuples_remaining -= tuples_this_chunk;
        cur_offset += tuples_this_chunk;
    }
    copy_col.transfer(host_cmp.data(), sample_tuples);

    auto& sm = StorageManager::get();
    uint64_t store_offset = sm.get_offset();

    SampleRes res;
    for (auto &chunk_bytes:chunk_bytess){
        compinfo.chunk_bytes = chunk_bytes;
        for (auto &comp_algo:comp_algos){
            compinfo.comp_algo = comp_algo;

            if (verify){
                res = sample_comp(copy_col,sample_tuples,compinfo,host_col.data(), workers);
                golap::verify<T,uint8_t>(host_cmp, host_col.data(), sample_tuples);
                host_col.set('#');
            }else{
                res = sample_comp(copy_col,sample_tuples,compinfo,(T*)nullptr, workers);
            }

            res.sample_ratio = sample_ratio;
            res.workers = workers;

            ress.push_back(res);
            compinfo.blocks.clear();
            // reset the store offset
            sm.set_offset(store_offset);
        }
    }


    return ress;

}

/**
 * Sample device (de)compression parameters of a column by compressing to disk, then loading again.
 */
template <typename MEM_TYPE, typename T, typename NVCOMPTYPE=T>
SampleRes sample_comp(golap::Column<MEM_TYPE,T> &col,
                 uint64_t num_tuples, golap::CompInfo &compinfo, T* host_ptr, uint32_t workers){

    auto res = SampleRes{col.attr_name,compinfo.comp_algo,compinfo.uncomp_bytes,compinfo.chunk_bytes};

    auto comp_time = prepare_compressed_device<MEM_TYPE,T,NVCOMPTYPE>(col, num_tuples, compinfo);
    uint64_t host_allocated=0,device_allocated=0;

    std::string dataflow = host_ptr == nullptr ? "SSD2GPU" : "SSD2GPU2CPU";
    std::string core_pin = "";

    auto decomp_time = load<DecompressEnv,T,T>(workers, core_pin, dataflow, 0, host_ptr, compinfo, host_allocated, device_allocated);


    res.comp_bytes = compinfo.get_comp_bytes();
    res.comp_time = comp_time;
    res.decomp_time = decomp_time;

    return res;
}

} // end of namespace
