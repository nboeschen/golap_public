#pragma once
#include <cstdint>
#include <iostream>
#include <cmath>
#include <memory>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#include "core.hpp"
#include "net.hpp"
#include "dev_structs.cuh"
#include "util.hpp"
#include "dev_util.cuh"
#include "access.hpp"
#include "comp.cuh"
#include "comp_cpu.hpp"
#include "metadata.cuh"

namespace golap {

/**
 * Simple verify helper
 */
template <typename T,typename COMPTYPE>
static void verify(golap::Column<golap::HostMem,T> &col, T* rhs_ptr, uint64_t num_tuples){
    COMPTYPE* lhs_ptr_comp = (COMPTYPE*) col.data();
    COMPTYPE* rhs_ptr_comp = (COMPTYPE*) rhs_ptr;

    util::Log::get().debug_fmt("Verification starting from ptr=%p",rhs_ptr);

    uint64_t i = 0,max_index=num_tuples*sizeof(T)/sizeof(COMPTYPE);
    for (;i<max_index; ++i){
        if(lhs_ptr_comp[i] != rhs_ptr_comp[i]){
            if constexpr (std::is_same_v<uint8_t,T>){
                std::cout << "#[WARN ]: Verification failed at idx "<<i<<" of "<<max_index<<", should be "<<+lhs_ptr_comp[i]<<", is "<< +rhs_ptr_comp[i]<<"\n";
            }else {
                std::cout << "#[WARN ]: Verification failed at idx "<<i<<" of "<<max_index<<", should be "<<lhs_ptr_comp[i]<<", is "<< rhs_ptr_comp[i]<<"\n";
            }
            return;
        }
    }

    util::Log::get().info_fmt("Verification succeeded for %lu items (%lu bytes)! :)",num_tuples,num_tuples*sizeof(T));
}

template <typename MEM_TYPE, typename T>
double prepare_uncompressed(golap::Column<MEM_TYPE,T> &col, uint64_t num_tuples, golap::CompInfo &compinfo,
                                        MinMaxMeta<T> *mmmeta = nullptr, EqHistogram<T> *histogram = nullptr, BloomMeta<T> *bloommeta = nullptr){
    auto& sm = StorageManager::get();
    uint64_t chunk_num,max_tuples_per_chunk;

    if (compinfo.chunk_bytes != (uint64_t)-1){
        chunk_num = util::div_ceil(compinfo.uncomp_bytes,compinfo.chunk_bytes);
        max_tuples_per_chunk = compinfo.chunk_bytes / sizeof(T);
    }else{
        chunk_num = compinfo.chunk_size_vec.size();
        max_tuples_per_chunk = (*std::max_element(compinfo.chunk_size_vec.begin(),compinfo.chunk_size_vec.end()))/sizeof(T);
    }

    golap::DeviceMem uncomp_buffer{golap::Tag<T>{}, max_tuples_per_chunk};
    // golap::DeviceMem temp{golap::Tag<char>{}, ((uint64_t)1<<32)};

    if (mmmeta != nullptr){
        new (mmmeta) MinMaxMeta<T>{chunk_num, max_tuples_per_chunk};
    }
    if (bloommeta != nullptr){
        new (bloommeta) BloomMeta<T>{chunk_num, bloommeta->p, bloommeta->m, max_tuples_per_chunk};
    }
    if (histogram != nullptr){
        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T,util::Date> || std::is_same_v<T,util::Datetime>){
            new (histogram) EqHistogram<T>{chunk_num, histogram->bucket_num, max_tuples_per_chunk};
        }else{
            util::Log::get().warn_fmt("Cant create histogram for column %s of non-arithmetic type!",col.attr_name.c_str());
        }
    }
    compinfo.blocks.reserve(chunk_num);

    // used to zero out the uncomp buffer before data is copied there
    golap::Memset memsetter{uncomp_buffer.template ptr<char>(), uncomp_buffer.size_bytes(), 0};

    // use datacopy with total_bytes limit
    golap::DataCopy copier{col.data(), uncomp_buffer.template ptr<T>(), true, false,
                            compinfo.chunk_bytes, compinfo.chunk_size_vec, num_tuples*sizeof(T)};
    copier.set_child(&memsetter);

    golap::DenseWriter writer{sm.get_offset(),compinfo.blocks,uncomp_buffer.template ptr<T>()};
    writer.set_child(&copier);

    compinfo.comp_bytes=0;
    golap::CEvent event;
    double comp_ms;
    golap::CStream streamA{"Prepare Uncompressed"};
    uint64_t tuples_this_chunk,tuples_handled=0;
    uint64_t chunk_idx = 0;
    util::Timer timer;

    while(writer.step(streamA.stream,event.event)){
        checkCudaErrors(cudaStreamSynchronize(streamA.stream));

        if (compinfo.chunk_bytes != (uint64_t)-1){
            tuples_this_chunk = std::min(num_tuples-tuples_handled,max_tuples_per_chunk);
        }else tuples_this_chunk = compinfo.chunk_size_vec[chunk_idx] / sizeof(T);

        util::Log::get().debug_fmt("Last writing step: %llu tuples, %llu bytes",tuples_this_chunk,writer.last_produced);

        if (mmmeta != nullptr){
            mmmeta->init_chunk(chunk_idx, uncomp_buffer.template ptr<T>(), tuples_this_chunk);
        }
        if (bloommeta != nullptr){
            golap::fill_bloom<<<80,512>>>(*bloommeta, uncomp_buffer.template ptr<T>(), chunk_idx, tuples_this_chunk);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T,util::Date> || std::is_same_v<T,util::Datetime>){
            if (histogram != nullptr){
                histogram->init_chunk(chunk_idx, uncomp_buffer.template ptr<T>(), tuples_this_chunk);
                golap::fill_hist<<<80,512>>>(*histogram, uncomp_buffer.template ptr<T>(), chunk_idx, tuples_this_chunk);
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }

        compinfo.comp_bytes += writer.last_produced;
        compinfo.blocks.back().tuples = tuples_this_chunk;

        tuples_handled += tuples_this_chunk;
        chunk_idx += 1;
    }
    checkCudaErrors(cudaStreamSynchronize(streamA.stream));
    comp_ms = timer.elapsed();

    if (mmmeta != nullptr) mmmeta->free_temp();
    if constexpr (std::is_arithmetic_v<T>){
        if (histogram != nullptr) histogram->free_temp();
    }

    sm.set_offset(util::next(compinfo.end_offset(), 4096));

    util::Log::get().debug_fmt("Wrote column %s, %llu bytes, %llu chunks of size %llu",
                           col.attr_name.c_str(),
                           compinfo.uncomp_bytes,
                           chunk_num, compinfo.chunk_bytes);
    return comp_ms;
}

/**
 * Copy a column, compress in chunks and write from there to disk in dense layout.
 */
template <typename TARGET_MEM_TYPE, typename COMP_OP, typename COMP_MANAGER,typename MEM_TYPE, typename T>
inline double prepare_compressed(golap::Column<MEM_TYPE,T> &col,
        uint64_t num_tuples, golap::CompInfo &compinfo, std::shared_ptr<COMP_MANAGER> comp_manager,
        MinMaxMeta<T> *mmmeta, EqHistogram<T> *histogram, BloomMeta<T> *bloommeta, cudaStream_t stream){

    auto& sm = StorageManager::get();

    uint64_t chunk_num,max_tuples_per_chunk;
    if (compinfo.chunk_bytes != (uint64_t)-1){
        chunk_num = util::div_ceil(compinfo.uncomp_bytes,compinfo.chunk_bytes);
        max_tuples_per_chunk = compinfo.chunk_bytes / sizeof(T);
    }else{
        chunk_num = compinfo.chunk_size_vec.size();
        max_tuples_per_chunk = (*std::max_element(compinfo.chunk_size_vec.begin(),compinfo.chunk_size_vec.end()))/sizeof(T);
    }

    // prepare an area to temporarily hold a compressed chunk (give it a little bit more space than usually needed)
    TARGET_MEM_TYPE uncomp_buffer{golap::Tag<char>{}, max_tuples_per_chunk*sizeof(T)};
    TARGET_MEM_TYPE comp_buffer{golap::Tag<char>{}, max_tuples_per_chunk*sizeof(T) + ((max_tuples_per_chunk*sizeof(T))>>2)};
    // TARGET_MEM_TYPE temp{golap::Tag<char>{}, ((uint64_t)1<<32)};

    if (mmmeta != nullptr){
        new (mmmeta) MinMaxMeta<T>{chunk_num, max_tuples_per_chunk};
    }
    if (bloommeta != nullptr){
        new (bloommeta) BloomMeta<T>{chunk_num, bloommeta->p, bloommeta->m, max_tuples_per_chunk};
    }
    if (histogram != nullptr){
        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T,util::Date> || std::is_same_v<T,util::Datetime>){
            new (histogram) EqHistogram<T>{chunk_num, histogram->bucket_num, max_tuples_per_chunk};
        }else{
            util::Log::get().warn_fmt("Cant create histogram for column %s of non-arithmetic type!",col.attr_name.c_str());
        }
    }

    compinfo.blocks.reserve(chunk_num);

    // used to zero out the uncomp buffer before data is copied there, since we usually compress the whole chunk_size bytes
    // (except when variable chunk sizes are used, i.e. chunk_bytes = -1 and chunk_size_vec is set)
    golap::Memset memsetter{uncomp_buffer.template ptr<char>(), uncomp_buffer.size_bytes(), 0};

    // use datacopy with total_bytes limit
    golap::DataCopy copier{col.data(), uncomp_buffer.template ptr<T>(), true, false,
                            compinfo.chunk_bytes, compinfo.chunk_size_vec, num_tuples*sizeof(T)};
    copier.set_child(&memsetter);

    COMP_OP compressor{compinfo.chunk_bytes, uncomp_buffer.template ptr<char>(), comp_buffer.template ptr<char>(),
                                 false, false, *comp_manager, compinfo.chunk_size_vec};
    compressor.set_child(&copier);

    golap::DenseWriter writer{sm.get_offset(),compinfo.blocks,comp_buffer.template ptr<T>()};
    writer.set_child(&compressor);

    util::Log::get().debug_fmt("Starting compressing of %llu chunks, %llu tuples",chunk_num,num_tuples);
    compinfo.comp_bytes=0;
    golap::CEvent event;
    double comp_ms;
    util::Timer timer;
    uint64_t tuples_this_chunk,tuples_handled=0;
    uint64_t chunk_idx = 0;
    while(writer.step(stream,event.event)){
        checkCudaErrors(cudaStreamSynchronize(stream));

        if (compinfo.chunk_bytes != (uint64_t)-1){
            tuples_this_chunk = std::min(num_tuples-tuples_handled,max_tuples_per_chunk);
        }else tuples_this_chunk = compinfo.chunk_size_vec[chunk_idx] / sizeof(T);
        util::Log::get().debug_fmt("Chunk idx %llu of %llu, %llu tuples, %llu bytes",chunk_idx,chunk_num,tuples_this_chunk,writer.last_produced);

        if (mmmeta != nullptr){
            mmmeta->init_chunk(chunk_idx, uncomp_buffer.template ptr<T>(), tuples_this_chunk);
        }
        if (bloommeta != nullptr){
            golap::fill_bloom<<<80,512>>>(*bloommeta, uncomp_buffer.template ptr<T>(), chunk_idx, tuples_this_chunk);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        if (histogram != nullptr){
            if constexpr ((std::is_arithmetic_v<T> || std::is_same_v<T,util::Date> || std::is_same_v<T,util::Datetime>) && std::is_same_v<TARGET_MEM_TYPE,golap::DeviceMem>){
                histogram->init_chunk(chunk_idx, uncomp_buffer.template ptr<T>(), tuples_this_chunk);
                golap::fill_hist<<<80,512>>>(*histogram, uncomp_buffer.template ptr<T>(), chunk_idx, tuples_this_chunk);
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }

        compinfo.comp_bytes += writer.last_produced;
        compinfo.blocks.back().tuples = tuples_this_chunk;

        tuples_handled += tuples_this_chunk;
        chunk_idx += 1;
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
    comp_ms = timer.elapsed();

    if (mmmeta != nullptr) mmmeta->free_temp();
    if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T,util::Date> || std::is_same_v<T,util::Datetime>){
        if (histogram != nullptr) histogram->free_temp();
    }

    sm.set_offset(util::next(compinfo.end_offset(), 4096));

    util::Log::get().debug_fmt("Compressed column %s (%llu) -> %llu bytes (ratio of %.2f), using %s and %llu chunks of size %llu",
                           col.attr_name.c_str(),
                           compinfo.uncomp_bytes,
                           compinfo.comp_bytes,
                           (compinfo.comp_bytes/(double)compinfo.uncomp_bytes),
                           compinfo.comp_algo.c_str(),
                           chunk_num, compinfo.chunk_bytes);
    return comp_ms;
}

template <typename MEM_TYPE, typename T, typename NVCOMPTYPE=T>
double prepare_compressed_device(golap::Column<MEM_TYPE,T> &col,
        uint64_t num_tuples, golap::CompInfo &compinfo, MinMaxMeta<T> *mmmeta = nullptr,
        EqHistogram<T> *histogram = nullptr, BloomMeta<T> *bloommeta = nullptr){
    golap::CStream streamA{"Prepare Compressed Device"};
    auto nvtype = [](){
        try{
            return nvcomp::TypeOf<NVCOMPTYPE>();
        }catch(nvcomp::NVCompException exc) {
            return nvcomp::TypeOf<uint8_t>();
        }
    }();

    auto comp_manager = golap::build_gpucomp_manager(compinfo.comp_algo, compinfo.nvchunk, nvtype, streamA.stream);
    double res = prepare_compressed<golap::DeviceMem,golap::ChunkCompressor,nvcomp::nvcompManagerBase,MEM_TYPE,T>(col,
                                                num_tuples,compinfo,comp_manager,
                                                mmmeta,histogram,bloommeta,streamA.stream);
    return res;
}

#ifdef WITH_CPU_COMP
template <typename MEM_TYPE, typename T>
double prepare_compressed_host(golap::Column<MEM_TYPE,T> &col,
        uint64_t num_tuples, golap::CompInfo &compinfo, MinMaxMeta<T> *mmmeta = nullptr,
        EqHistogram<T> *histogram = nullptr, BloomMeta<T> *bloommeta = nullptr){
    golap::CStream streamA{"Prepare Compressed Host"};
    auto comp_manager = golap::build_cpucomp_manager(compinfo.comp_algo);
    double res = prepare_compressed<golap::HostMem,golap::CPUChunkCompressor,golap::CPUCompManagerBase,MEM_TYPE,T>(col,
                                                num_tuples,compinfo,comp_manager,
                                                mmmeta,histogram,bloommeta,streamA.stream);
    return res;
}
#endif //WITH_CPU_COMP


struct LoadEnv{
    LoadEnv(const LoadEnv &obj) = delete;
    LoadEnv(LoadEnv&&) = default;
    LoadEnv(golap::CompInfo &compinfo, std::string dataflow, std::string stream_name, nvcompType_t nvtype,
              std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block, char* host_ptr = nullptr):
            gstream(stream_name),
            decomp_buf(golap::Tag<char>(), compinfo.max_uncomp_chunk(), CUFILE_MAGIC_NUMBER, CUFILE_MAGIC_NUMBER)
            {

                for(; start_block < past_end_block; ++start_block){
                    myblock_idxs.push_back(all_block_idxs[start_block]);
                    myblocks.push_back(compinfo.blocks[all_block_idxs[start_block]]);
                    if (compinfo.chunk_bytes != (uint64_t)-1) continue;
                    mychunksizes.push_back(compinfo.chunk_size_vec[all_block_idxs[start_block]]);
                    // todo: update the host_ptr here
                }
                loader = std::make_shared<golap::VarLoader<char>>(myblocks, decomp_buf.ptr<char>());
                copier = std::make_shared<golap::DataCopy<char>>(decomp_buf.ptr<char>(), host_ptr, false, true, compinfo.chunk_bytes, mychunksizes);

                copier->set_child(&(*loader));
                if(dataflow == "SSD2GPU2CPU"){
                    if (host_ptr == nullptr) util::Log::get().error_fmt("LoadEnv SSD2GPU2CPU, but host_ptr not set!");
                    root = &(*copier);
                }else if(dataflow == "SSD2CPU2GPU"){
                    // could be nicer ...
                    loader = std::make_shared<golap::VarLoaderCPU<char>>(myblocks, host_ptr);
                    copier = std::make_shared<golap::DataCopy<char>>(host_ptr, decomp_buf.ptr<char>(), false, false, compinfo.chunk_bytes, mychunksizes);
                    copier->set_child(&(*loader));
                    root =  &(*copier);
                }else if(dataflow == "SSD2GPU"){
                    root = &(*loader);
                }else {
                    util::Log::get().error_fmt("Unknown dataflow in LoadEnv, fix this!");
                }
            }
    std::vector<uint64_t> myblock_idxs;
    std::vector<golap::BlockInfo> myblocks;
    std::vector<uint64_t> mychunksizes;
    golap::CStream gstream;
    golap::CEvent gevent;
    golap::DeviceMem decomp_buf;

    std::shared_ptr<golap::Op> loader;
    std::shared_ptr<golap::DataCopy<char>> copier;
    golap::Op *root;
};



struct DecompressEnv{
    DecompressEnv(const DecompressEnv &obj) = delete;
    DecompressEnv(DecompressEnv&&) = default;
    DecompressEnv(golap::CompInfo &compinfo, std::string dataflow, std::string stream_name,
                  nvcompType_t nvtype, std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block,
                  char* host_ptr = nullptr):
            gstream(stream_name),
            comp_manager(golap::build_gpucomp_manager(compinfo.comp_algo, compinfo.nvchunk, nvtype, gstream.stream,
                                                      compinfo.num_RLEs, compinfo.num_deltas, compinfo.use_bp)),
            comp_buf(golap::Tag<char>(), compinfo.max_uncomp_chunk()+(compinfo.max_uncomp_chunk()>>2), CUFILE_MAGIC_NUMBER, CUFILE_MAGIC_NUMBER),
            decomp_buf(golap::Tag<char>(), compinfo.max_uncomp_chunk(), 4096, 4096)
            {

                for(; start_block < past_end_block; ++start_block){
                    myblock_idxs.push_back(all_block_idxs[start_block]);
                    myblocks.push_back(compinfo.blocks[all_block_idxs[start_block]]);
                    if (compinfo.chunk_bytes != (uint64_t)-1) continue;
                    mychunksizes.push_back(compinfo.chunk_size_vec[all_block_idxs[start_block]]);
                }
                loader = std::make_shared<golap::VarLoader<char>>(myblocks, comp_buf.ptr<char>());
                decompressor = std::make_shared<golap::ChunkDecompressor>(compinfo.chunk_bytes, comp_buf.ptr<char>(), decomp_buf.ptr<char>(), false, false,*comp_manager, mychunksizes);
                copier = std::make_shared<golap::DataCopy<char>>(decomp_buf.ptr<char>(), host_ptr, false, true, compinfo.chunk_bytes, mychunksizes);

                decompressor->set_child(&(*loader));

                copier->set_child(&(*decompressor));
                if(dataflow == "SSD2GPU2CPU"){
                    if (host_ptr == nullptr) util::Log::get().error_fmt("DecompressEnv SSD2GPU2CPU, but host_ptr not set!");
                    root = &(*copier);
                }else if(dataflow == "SSD2CPU2GPU"){
                    loader = std::make_shared<golap::VarLoaderCPU<char>>(myblocks, host_ptr);
                    copier = std::make_shared<golap::VarCopy<char>>(host_ptr, decomp_buf.ptr<char>(), compinfo.chunk_bytes);
                    copier->set_child(&(*loader));
                    root =  &(*copier);
                }else if(dataflow == "SSD2GPU"){
                    root = &(*decompressor);
                }else {
                    util::Log::get().error_fmt("Unknown dataflow in DecompressEnv, fix this!");
                }

            }

    std::vector<uint64_t> myblock_idxs;
    std::vector<golap::BlockInfo> myblocks;
    std::vector<uint64_t> mychunksizes;
    golap::CStream gstream;
    golap::CEvent gevent;
    std::shared_ptr<nvcomp::nvcompManagerBase> comp_manager;
    golap::DeviceMem comp_buf;
    golap::DeviceMem decomp_buf;

    std::shared_ptr<golap::Op> loader;
    std::shared_ptr<golap::ChunkDecompressor> decompressor;
    std::shared_ptr<golap::Op> copier;
    golap::Op *root;
};


struct DecompressEnvWOLoad{
    DecompressEnvWOLoad(const DecompressEnvWOLoad &obj) = delete;
    DecompressEnvWOLoad(DecompressEnvWOLoad&&) = default;
    DecompressEnvWOLoad(golap::CompInfo &compinfo, std::string dataflow, std::string stream_name,
                  nvcompType_t nvtype, std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block,
                  char* host_ptr = nullptr):
            gstream(stream_name),
            comp_manager(golap::build_gpucomp_manager(compinfo.comp_algo, compinfo.nvchunk, nvtype, gstream.stream,
                                                      compinfo.num_RLEs, compinfo.num_deltas, compinfo.use_bp)),
            comp_buf(golap::Tag<char>(), compinfo.max_uncomp_chunk()+(compinfo.max_uncomp_chunk()>>2), CUFILE_MAGIC_NUMBER, CUFILE_MAGIC_NUMBER),
            decomp_buf(golap::Tag<char>(), compinfo.max_uncomp_chunk(), 4096, 4096)
            {

                for(; start_block < past_end_block; ++start_block){
                    myblock_idxs.push_back(all_block_idxs[start_block]);
                    myblocks.push_back(compinfo.blocks[all_block_idxs[start_block]]);
                    if (compinfo.chunk_bytes != (uint64_t)-1) continue;
                    mychunksizes.push_back(compinfo.chunk_size_vec[all_block_idxs[start_block]]);
                }
                decompressor = std::make_shared<golap::ChunkDecompressor>(compinfo.chunk_bytes, comp_buf.ptr<char>(), decomp_buf.ptr<char>(), false, false,*comp_manager, mychunksizes);
                copier = std::make_shared<golap::DataCopy<char>>(decomp_buf.ptr<char>(), host_ptr, false, true, compinfo.chunk_bytes, mychunksizes);

                copier->set_child(&(*decompressor));
                if(dataflow == "SSD2GPU2CPU"){
                    if (host_ptr == nullptr) util::Log::get().error_fmt("DecompressEnv SSD2GPU2CPU, but host_ptr not set!");
                    root = &(*copier);
                }else if(dataflow == "SSD2GPU"){
                    root = &(*decompressor);
                }else {
                    util::Log::get().error_fmt("Unknown dataflow in DecompressEnv, fix this!");
                }

            }

    std::vector<uint64_t> myblock_idxs;
    std::vector<golap::BlockInfo> myblocks;
    std::vector<uint64_t> mychunksizes;
    golap::CStream gstream;
    golap::CEvent gevent;
    std::shared_ptr<nvcomp::nvcompManagerBase> comp_manager;
    golap::DeviceMem comp_buf;
    golap::DeviceMem decomp_buf;

    std::shared_ptr<golap::ChunkDecompressor> decompressor;
    std::shared_ptr<golap::DataCopy<char>> copier;
    golap::Op *root;
};


#ifdef WITH_CPU_COMP
struct LoadEnvCPU{
    LoadEnvCPU(const LoadEnvCPU &obj) = delete;
    LoadEnvCPU(LoadEnvCPU&&) = default;
    LoadEnvCPU(golap::CompInfo &compinfo, std::string dataflow, std::string stream_name, nvcompType_t nvtype,
              std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block, char* host_ptr = nullptr):
            gstream(stream_name)
            {
                for(; start_block < past_end_block; ++start_block){
                    myblock_idxs.push_back(all_block_idxs[start_block]);
                    myblocks.push_back(compinfo.blocks[all_block_idxs[start_block]]);
                    if (compinfo.chunk_bytes != (uint64_t)-1) continue;
                    mychunksizes.push_back(compinfo.chunk_size_vec[all_block_idxs[start_block]]);
                }
                loader = std::make_shared<golap::VarLoaderCPU<char>>(myblocks, host_ptr, true);

                if(dataflow != "SSD2CPU"){
                    util::Log::get().error_fmt("LoadEnvCPU with dataflow != SSD2CPU, fix this!");
                }
                if(host_ptr == nullptr){
                    util::Log::get().error_fmt("LoadEnvCPU with empty host_ptr, fix this!");
                }
                root = &(*loader);
            }
    std::vector<uint64_t> myblock_idxs;
    std::vector<golap::BlockInfo> myblocks;
    std::vector<uint64_t> mychunksizes;
    golap::CStream gstream;
    golap::CEvent gevent;

    std::shared_ptr<golap::VarLoaderCPU<char>> loader;
    golap::Op *root;
};

struct LoadChunkEnvCPU{
    LoadChunkEnvCPU(const LoadChunkEnvCPU &obj) = delete;
    LoadChunkEnvCPU(LoadChunkEnvCPU&&) = default;
    LoadChunkEnvCPU(golap::CompInfo &compinfo, std::string dataflow, std::string stream_name, nvcompType_t nvtype,
              std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block, char* host_ptr = nullptr):
            gstream(stream_name),
            decomp_buf(golap::Tag<char>(), compinfo.chunk_bytes, 4096, 4096)
            {
                for(; start_block < past_end_block; ++start_block){
                    myblock_idxs.push_back(all_block_idxs[start_block]);
                    myblocks.push_back(compinfo.blocks[all_block_idxs[start_block]]);
                    if (compinfo.chunk_bytes != (uint64_t)-1) continue;
                    mychunksizes.push_back(compinfo.chunk_size_vec[all_block_idxs[start_block]]);
                }
                loader = std::make_shared<golap::VarLoaderCPU<char>>(myblocks, decomp_buf.ptr<char>(), false);

                if(dataflow != "SSD2CPU"){
                    util::Log::get().error_fmt("LoadChunkEnvCPU with dataflow != SSD2CPU, fix this!");
                }
                if(host_ptr != nullptr){
                    util::Log::get().error_fmt("LoadChunkEnvCPU with set host_ptr, probably not what you want!");
                }
                root = &(*loader);
            }
    std::vector<uint64_t> myblock_idxs;
    std::vector<golap::BlockInfo> myblocks;
    std::vector<uint64_t> mychunksizes;
    golap::CStream gstream;
    golap::CEvent gevent;
    golap::HostMem decomp_buf;

    std::shared_ptr<golap::VarLoaderCPU<char>> loader;
    golap::Op *root;
};


struct DecompressEnvCPU{
    DecompressEnvCPU(const DecompressEnvCPU &obj) = delete;
    DecompressEnvCPU(DecompressEnvCPU&&) = default;
    DecompressEnvCPU(golap::CompInfo &compinfo, std::string dataflow, std::string stream_name, nvcompType_t nvtype,
                std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block, char* host_ptr = nullptr):
            gstream(stream_name),
            comp_manager(golap::build_cpucomp_manager(compinfo.comp_algo)),
            comp_buf(golap::Tag<char>(), compinfo.chunk_bytes+(compinfo.chunk_bytes>>2), 4096, 4096)

            {
                for(; start_block < past_end_block; ++start_block){
                    myblock_idxs.push_back(all_block_idxs[start_block]);
                    myblocks.push_back(compinfo.blocks[all_block_idxs[start_block]]);
                    if (compinfo.chunk_bytes != (uint64_t)-1) continue;
                    mychunksizes.push_back(compinfo.chunk_size_vec[all_block_idxs[start_block]]);
                }
                loader = std::make_shared<golap::VarLoaderCPU<char>>(myblocks, comp_buf.ptr<char>(), false);
                decompressor = std::make_shared<golap::CPUChunkDecompressor>(compinfo.chunk_bytes, comp_buf.ptr<char>(), host_ptr, false, true,*comp_manager, mychunksizes);

                decompressor->set_child(&(*loader));

                if(dataflow != "SSD2CPU"){
                    util::Log::get().error_fmt("DecompressEnvCPU with dataflow != SSD2CPU, fix this!");
                }
                if(host_ptr == nullptr){
                    util::Log::get().error_fmt("DecompressEnvCPU with empty host_ptr, fix this!");
                }
                root = &(*decompressor);

            }
    std::vector<uint64_t> myblock_idxs;
    std::vector<golap::BlockInfo> myblocks;
    std::vector<uint64_t> mychunksizes;
    golap::CStream gstream;
    golap::CEvent gevent;
    std::shared_ptr<golap::CPUCompManagerBase> comp_manager;
    golap::HostMem comp_buf;

    std::shared_ptr<golap::VarLoaderCPU<char>> loader;
    std::shared_ptr<golap::CPUChunkDecompressor> decompressor;
    golap::Op *root;
};

struct DecompressChunkEnvCPU{
    DecompressChunkEnvCPU(const DecompressChunkEnvCPU &obj) = delete;
    DecompressChunkEnvCPU(DecompressChunkEnvCPU&&) = default;
    DecompressChunkEnvCPU(golap::CompInfo &compinfo, std::string dataflow, std::string stream_name, nvcompType_t nvtype,
                    std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block, char* host_ptr = nullptr):
            gstream(stream_name),
            comp_manager(golap::build_cpucomp_manager(compinfo.comp_algo)),
            comp_buf(golap::Tag<char>(), compinfo.chunk_bytes+(compinfo.chunk_bytes>>2), 4096, 4096),
            decomp_buf(golap::Tag<char>(), compinfo.chunk_bytes, 4096, 4096)

            {
                for(; start_block < past_end_block; ++start_block){
                    myblock_idxs.push_back(all_block_idxs[start_block]);
                    myblocks.push_back(compinfo.blocks[all_block_idxs[start_block]]);
                    if (compinfo.chunk_bytes != (uint64_t)-1) continue;
                    mychunksizes.push_back(compinfo.chunk_size_vec[all_block_idxs[start_block]]);
                }
                loader = std::make_shared<golap::VarLoaderCPU<char>>(myblocks, comp_buf.ptr<char>(), false);
                decompressor = std::make_shared<golap::CPUChunkDecompressor>(compinfo.chunk_bytes, comp_buf.ptr<char>(), decomp_buf.ptr<char>(), false, false,*comp_manager, mychunksizes);

                decompressor->set_child(&(*loader));

                if(dataflow != "SSD2CPU"){
                    util::Log::get().error_fmt("DecompressChunkEnvCPU with dataflow != SSD2CPU, fix this!");
                }
                if(host_ptr != nullptr){
                    util::Log::get().error_fmt("DecompressChunkEnvCPU with set host_ptr, probably not what you want!");
                }
                root = &(*decompressor);

            }
    std::vector<uint64_t> myblock_idxs;
    std::vector<golap::BlockInfo> myblocks;
    std::vector<uint64_t> mychunksizes;
    golap::CStream gstream;
    golap::CEvent gevent;
    std::shared_ptr<golap::CPUCompManagerBase> comp_manager;
    golap::HostMem comp_buf;
    golap::HostMem decomp_buf;

    std::shared_ptr<golap::VarLoaderCPU<char>> loader;
    std::shared_ptr<golap::CPUChunkDecompressor> decompressor;
    golap::Op *root;
};
#endif //WITH_CPU_COMP


/**
 * A helper to bundle a bunch of the above BLOCKENV together, e.g. to load columns of one table
 */
template <typename BLOCKENV>
struct TableLoader{
    TableLoader(const TableLoader &obj) = delete;
    TableLoader(TableLoader&&) = default;
    TableLoader(uint64_t num){
        blockenvs.reserve(num);
    }

    void add(std::string name, std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block,
             golap::CompInfo &comp_info, nvcompType_t nvcomptype, std::string dataflow = "SSD2GPU"){
        blockenvs.emplace(std::piecewise_construct, std::forward_as_tuple(name),
                         std::forward_as_tuple(comp_info, dataflow, "Stream"+name, nvcomptype, all_block_idxs, start_block, past_end_block));
        rootop.add_child(blockenvs.at(name).root, blockenvs.at(name).gstream.stream);
    }

    std::unordered_map<std::string,BLOCKENV> blockenvs;
    golap::Collector rootop;
    golap::CStream rootstream{"Stream Root"};
    golap::CEvent rootevent;
};

template <typename BLOCKENV>
struct BatchTableLoader{
    BatchTableLoader(const BatchTableLoader &obj) = delete;
    BatchTableLoader(BatchTableLoader&&) = default;
    BatchTableLoader(uint64_t num):decollector(num),batch_loader(num){
        blockenvs.reserve(num);
        decollector.set_child(&batch_loader);
    }

    ~BatchTableLoader(){
        for (auto& [name,env] : blockenvs){
            cuFileBufDeregister(env.comp_buf.template ptr<char>());
        }
    }

    void add(std::string name, std::vector<uint64_t> &all_block_idxs, uint64_t start_block, uint64_t past_end_block,
             golap::CompInfo &comp_info, nvcompType_t nvcomptype, std::string dataflow = "SSD2GPU"){
        blockenvs.emplace(std::piecewise_construct, std::forward_as_tuple(name),
                         std::forward_as_tuple(comp_info, dataflow, "Stream"+name, nvcomptype, all_block_idxs, start_block, past_end_block));
        // rewire dataloading to decollector:
        blockenvs.at(name).decompressor->set_child(&decollector);

        // blockenvs[name].comp_buf is the area to load into
        batch_loader.out_ptrs.emplace_back(blockenvs.at(name).comp_buf.template ptr<char>());
        batch_loader.blockss.emplace_back(blockenvs.at(name).myblocks);
        // cuFileBufDeregister of these areas
        uint64_t max_block_size = 0;
        for (auto &block:blockenvs.at(name).myblocks){
            if (block.size > max_block_size){
                max_block_size = block.size;
            }
        }
        uint64_t reg_size = util::div_ceil(max_block_size,CUFILE_MAGIC_NUMBER)*CUFILE_MAGIC_NUMBER;
        checkCuFileError(cuFileBufRegister(blockenvs.at(name).comp_buf.template ptr<char>(), reg_size, 0));

        rootop.add_child(blockenvs.at(name).root, blockenvs.at(name).gstream.stream);
    }

    std::unordered_map<std::string,BLOCKENV> blockenvs;
    golap::BatchLoader batch_loader;
    golap::Decollector decollector;
    golap::Collector rootop;
    golap::CStream rootstream{"Stream Root"};
    golap::CEvent rootevent;
};


/**
 * Load (un)compressed column from offsets in blocks to GPU/CPU, possibly uncompress and copy to Host
 */
template <typename BLOCKENV, typename T, typename NVCOMPTYPE=T>
double load(uint64_t worker_num, std::string &core_pin, std::string dataflow, uint32_t cuda_device, T* host_ptr, golap::CompInfo &compinfo,
            uint64_t &host_allocated, uint64_t &device_allocated, int64_t simulate_compute_us=-1, golap::MirrorMem *prune_check = nullptr){

    std::vector<BLOCKENV> envs;
    envs.reserve(worker_num);
    std::vector<std::thread> threads;
    std::atomic<uint64_t> pruned_bytes{0};
    std::atomic<uint64_t> pruned_chunks{0};

    auto nvtype = [](){
        try{
            return nvcomp::TypeOf<NVCOMPTYPE>();
        }catch(nvcomp::NVCompException exc) {
            return nvcomp::TypeOf<uint8_t>();
        }
    }();

    std::vector<int> cpuids;
    auto cores = util::str_split(core_pin,"-");
    if (cores.size() != 0 && cores.size() != worker_num){
        std::cout << "core_pin ("<<core_pin<<") is not zero and doesnt fit the worker_num ("<<worker_num<<")! Wont pin anything.\n";
    }else{
        for (auto &cpu_str: cores){
            cpuids.push_back(std::stoi(cpu_str));
        }
    }


    util::SliceSeq workslice(compinfo.blocks.size(), worker_num);
    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfo.blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);


    for (uint32_t pipeline_idx=0; pipeline_idx<worker_num; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        envs.emplace_back(compinfo, dataflow, compinfo.comp_algo + " ENV", nvtype,
                          all_blocks_idxs, startblock, endblock, (char*)host_ptr + startblock * compinfo.chunk_bytes);
    }
    // fix sometimes slow start
    util::waiting_kernel<<<1,1>>>(100);
    cudaDeviceSynchronize();

    util::Timer timer;
    for (uint32_t pipeline_idx=0; pipeline_idx<worker_num; ++pipeline_idx){
        threads.emplace_back([&,pipeline_idx,cuda_device{cuda_device}]{

            cudaSetDevice(cuda_device);
            BLOCKENV &env = envs[pipeline_idx];
            uint64_t tuples_this_round,global_block_idx;
            uint64_t round = 0;

            if (pipeline_idx < cpuids.size()){
                if (util::pin_thread(cpuids[pipeline_idx])==0){
                    printf("Pinned thread[%u] to cpu[%d]\n",pipeline_idx,cpuids[pipeline_idx]);
                }
            }

            // while(env.root->step(env.gstream.stream,env.gevent.event)){
            while(round< env.myblocks.size()){

                tuples_this_round = env.myblocks[round].tuples;
                global_block_idx = env.myblock_idxs[round];
                // util::Log::get().info_fmt("Thread[%lu, round%lu] Block %lu, %lu tuples", pipeline_idx, round, global_block_idx, tuples_this_round);

                if (prune_check != nullptr && prune_check->hst.ptr<uint16_t>()[global_block_idx] == (uint16_t) 0){
                    util::Log::get().debug_fmt("Thread[%lu, round%lu] would skip the next chunk idx %lu...", pipeline_idx, round, global_block_idx);
                    env.root->skip_step(env.gstream.stream,env.gevent.event);
                    pruned_bytes.fetch_add(env.myblocks[round].size, std::memory_order_relaxed);
                    pruned_chunks.fetch_add(1, std::memory_order_relaxed);
                    round += 1;
                    continue;
                }
                if (!env.root->step(env.gstream.stream,env.gevent.event)){
                    util::Log::get().error_fmt("Shouldnt happen!");
                }
                // checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));

                if(simulate_compute_us > 0){
                    // scale simulate_compute_us
                    util::waiting_kernel<<<1,1,0,env.gstream.stream>>>(simulate_compute_us);
                }

                checkCudaErrors(cudaEventRecord(env.gevent.event, env.gstream.stream));
                round += 1;
            }

            checkCudaErrors(cudaStreamSynchronize(env.gstream.stream));

        });
    }

    for(auto &thread: threads) thread.join();

    auto time_ms = timer.elapsed();

    host_allocated = golap::HOST_ALLOCATED.load();
    device_allocated = golap::DEVICE_ALLOCATED.load();


    if (prune_check != nullptr){
        util::Log::get().info_fmt("Pruned: %lu of %lu chunks (%.2f), %lu of %lu bytes (%.2f)",
                                    pruned_chunks.load(),compinfo.blocks.size(),(double)pruned_chunks.load()/compinfo.blocks.size(),
                                    pruned_bytes.load(),compinfo.get_comp_bytes(),(double)pruned_bytes.load()/compinfo.get_comp_bytes());
    }


    return time_ms;
}


} // end of namespace