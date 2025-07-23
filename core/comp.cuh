#pragma once
#include <cstdint>
#include <iostream>
#include <cmath>
#include <optional>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "nvcomp/cascaded.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/snappy.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp.hpp"

#include "core.hpp"
#include "dev_structs.cuh"
#include "util.hpp"

namespace golap {

struct BlockInfo {
    // the offset of the block 
    uint64_t offset;
    // the size of the block
    uint64_t size;
    // the actual payload, e.g. the size of the compressed block (<size)
    uint64_t payload;
    // the number of tuples included in this block
    uint64_t tuples;
};


struct CompInfo{
    // in
    uint64_t chunk_bytes;
    uint64_t uncomp_bytes;
    std::string comp_algo;
    uint64_t nvchunk = (1<<16);
    // out
    uint64_t comp_bytes = (uint64_t) -1;

    // nvcomp cascaded specific
    int num_RLEs=2;
    int num_deltas=1;
    int use_bp=1;

    // chunk_sizes in bytes. Only relevant if chunk_bytes is -1
    std::vector<uint64_t> chunk_size_vec;
    // Information on the compressed blocks
    std::vector<BlockInfo> blocks;

    uint64_t max_uncomp_chunk(){
        if (chunk_bytes != (uint64_t)-1) return chunk_bytes;
        else return *std::max_element(chunk_size_vec.begin(),chunk_size_vec.end());
    }

    uint64_t start_offset(){
        if (blocks.size() == 0) return (uint64_t)-1;
        return blocks[0].offset;
    }

    uint64_t end_offset(){
        if (blocks.size() == 0) return (uint64_t)-1;
        return blocks[0].offset + get_comp_bytes();
    }

    uint64_t get_comp_bytes(){
        if (comp_bytes != (uint64_t) -1) return comp_bytes;
        comp_bytes = 0;
        for(auto &block:blocks){
            comp_bytes += block.size;
        }
        return comp_bytes;
    }

};


class ChunkCompressor : public Op{
public:
    ChunkCompressor(uint64_t chunk_bytes, char* in_ptr, char* out_ptr, bool advance_in, bool advance_out,
              nvcomp::nvcompManagerBase &comp_manager, std::vector<uint64_t> &chunk_size_vec):
            comp_manager(comp_manager),chunk_size_vec(chunk_size_vec),
            comp_config(comp_manager.configure_compression(512)),
            chunk_bytes(chunk_bytes),in_ptr(in_ptr),out_ptr(out_ptr),advance_in(advance_in),advance_out(advance_out){
        if (chunk_bytes != (uint64_t)-1){
            comp_config = comp_manager.configure_compression(chunk_bytes);
        }else{
            comp_config = comp_manager.configure_compression(*std::max_element(chunk_size_vec.begin(),chunk_size_vec.end()));
        }
        temp_mem.resize_num<uint8_t>(comp_manager.get_required_scratch_buffer_size());
        comp_manager.set_scratch_buffer(temp_mem.ptr<uint8_t>());
        if (advance_out) throw std::runtime_error("advance_out=true not supported currently for Compressor");
    }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (child!=nullptr && !child->step(stream,event)) return false;


        if(chunk_bytes == (uint64_t)-1){
            // for variable chunk sizes, update comp config every time
            comp_config = comp_manager.configure_compression(chunk_size_vec[cur_chunk]);
        }
        util::Log::get().debug_fmt("Step in ChunkCompressor. From %p, to %p, chunk_bytes %llu",in_ptr,out_ptr,comp_config.uncompressed_buffer_size);
        // util::Log::get().info_fmt("CONFIG=uncompressed_buffer_size=%llu, max_compressed_buffer_size=%llu,num_chunks=%llu",
        //                         comp_config->uncompressed_buffer_size,comp_config->max_compressed_buffer_size,comp_config->num_chunks);

        comp_manager.compress((const uint8_t*) in_ptr,(uint8_t*) out_ptr, comp_config);
        checkCudaErrors(cudaEventRecord(event,stream));

        if(advance_in){
            if (chunk_bytes != (uint64_t)-1) in_ptr = in_ptr+chunk_bytes;
            else in_ptr = in_ptr+chunk_size_vec[cur_chunk];
        }
        cur_chunk += 1;

        return true;
    }

    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if (child!=nullptr && !child->skip_step(stream,event)) return false;
        
        if(advance_in){
            if (chunk_bytes != (uint64_t)-1) in_ptr = in_ptr+chunk_bytes;
            else in_ptr = in_ptr+chunk_size_vec[cur_chunk];
        }
        cur_chunk += 1;
        return true;
    }
    void finish_step(cudaStream_t stream){
        Op::finish_step(stream);
        last_produced = comp_manager.get_compressed_output_size((uint8_t*)out_ptr);
        if (last_produced > (chunk_bytes + (chunk_bytes>>2))){
            util::Log::get().warn_fmt("Last step maybe wrote past end of buffer! %llu is more than %llu!",last_produced,chunk_bytes);
        }
    }
private:
    nvcomp::nvcompManagerBase &comp_manager;
    std::vector<uint64_t> &chunk_size_vec;
    nvcomp::CompressionConfig comp_config;
    DeviceMem temp_mem;
    uint64_t chunk_bytes,to_compress;
    uint64_t cur_chunk = 0;
    char *in_ptr,*out_ptr;
    bool advance_in,advance_out;
};


class ChunkDecompressor : public Op{
public:
    ChunkDecompressor(const ChunkDecompressor &obj) = delete;
    ChunkDecompressor(ChunkDecompressor&&) = default;
    ChunkDecompressor(uint64_t chunk_bytes, char* in_ptr, char* out_ptr, bool advance_in, bool advance_out,
                          nvcomp::nvcompManagerBase &comp_manager, std::vector<uint64_t> &chunk_size_vec):
        comp_manager(comp_manager),comp_config(comp_manager.configure_compression(512)),
        decomp_config(comp_manager.configure_decompression(comp_config)),chunk_size_vec(chunk_size_vec),chunk_bytes(chunk_bytes),in_ptr(in_ptr),out_ptr(out_ptr),advance_in(advance_in),advance_out(advance_out){
        if (chunk_bytes != (uint64_t)-1){
            comp_config = comp_manager.configure_compression(chunk_bytes);
            decomp_config= comp_manager.configure_decompression(comp_config);
        }else{
            comp_config = comp_manager.configure_compression(*std::max_element(chunk_size_vec.begin(),chunk_size_vec.end()));
        }
        temp_mem.resize_num<uint8_t>(comp_manager.get_required_scratch_buffer_size());
        comp_manager.set_scratch_buffer(temp_mem.ptr<uint8_t>());
        util::Log::get().debug_fmt("ChunkDecompressor for chunk_bytes=%lu, from %p, to %p. Using %lu bytes temp_mem",chunk_bytes,in_ptr,out_ptr,temp_mem.size_bytes());
        last_produced = chunk_bytes;
    }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (child!=nullptr && !child->step(stream,event)) return false;

        if(chunk_bytes == (uint64_t)-1){
            // for variable chunk sizes, update comp config every time
            comp_config = comp_manager.configure_compression(chunk_size_vec[cur_chunk]);
            decomp_config = comp_manager.configure_decompression(comp_config);
            last_produced = chunk_size_vec[cur_chunk];
        }

        util::Log::get().debug_fmt("Step in ChunkDecompressor. Expanding from %p to %p hopefully produced %llu",in_ptr,out_ptr,last_produced);

        // util::Log::get().debug_fmt("  CompressionConfig: %lu",comp_config.uncompressed_buffer_size);
        // util::Log::get().debug_fmt("DecompressionConfig: %lu",decomp_config.decomp_data_size);

        comp_manager.decompress((uint8_t*) out_ptr, (const uint8_t*) in_ptr,decomp_config);
        checkCudaErrors(cudaEventRecord(event,stream));

        if(advance_out){
            if (chunk_bytes != (uint64_t)-1) out_ptr = out_ptr+chunk_bytes;
            else out_ptr = out_ptr+chunk_size_vec[cur_chunk];
        }
        cur_chunk += 1;
        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if (child!=nullptr && !child->skip_step(stream,event)) return false;
        if(advance_out){
            if (chunk_bytes != (uint64_t)-1) out_ptr = out_ptr+chunk_bytes;
            else out_ptr = out_ptr+chunk_size_vec[cur_chunk];
        }
        cur_chunk += 1;
        return true;
    }
private:
    nvcomp::nvcompManagerBase &comp_manager;
    nvcomp::CompressionConfig comp_config;
    DeviceMem temp_mem;
    nvcomp::DecompressionConfig decomp_config;
    std::vector<uint64_t> &chunk_size_vec;
    uint64_t cur_chunk = 0;
    char *in_ptr, *out_ptr;
    bool advance_in,advance_out;
    uint64_t chunk_bytes;

};

// // LZ4Manager(size_t uncomp_chunk_size, nvcompType_t data_type, cudaStream_t user_stream, const int device_id)

// // CascadedManager(const nvcompBatchedCascadedOpts_t& options, cudaStream_t user_stream, int device_id)

// // SnappyManager(size_t uncomp_chunk_size, cudaStream_t user_stream, int device_id)

// // GdeflateManager(size_t uncomp_chunk_size, int algo, cudaStream_t user_stream, const int device_id)

// // BitcompManager(nvcompType_t data_type, int bitcomp_algo, cudaStream_t user_stream, const int device_id)

// // ANSManager(size_t uncomp_chunk_size, cudaStream_t user_stream, const int device_id)

static std::shared_ptr<nvcomp::nvcompManagerBase> build_gpucomp_manager(std::string name, uint64_t nv_chunk_size, nvcompType_t data_type, cudaStream_t stream, int num_RLEs=2, int num_deltas=1, int use_bp=1){
    if (name == "LZ4"){
        if (data_type == NVCOMP_TYPE_LONGLONG || data_type == NVCOMP_TYPE_ULONGLONG) {
            util::Log::get().debug_fmt("LongLong datatype not supported for LZ4 compression, using int32 instead ...");
            data_type = nvcomp::TypeOf<uint32_t>();
        }
        return std::make_shared<nvcomp::LZ4Manager>(nv_chunk_size, data_type, stream, 0);
    }else if(name == "Cascaded"){
        nvcompBatchedCascadedOpts_t options{4096, data_type, num_RLEs, num_deltas, use_bp};
        return std::make_shared<nvcomp::CascadedManager>(options,stream, 0);
    }else if(name == "Snappy"){
        return std::make_shared<nvcomp::SnappyManager>(nv_chunk_size,stream, 0);
    }else if(name == "Gdeflate" || name == "Gdeflate0"){
        return std::make_shared<nvcomp::GdeflateManager>(nv_chunk_size,0,stream, 0);
    }else if(name == "Gdeflate1"){
        return std::make_shared<nvcomp::GdeflateManager>(nv_chunk_size,1,stream, 0);
    }else if(name == "Gdeflate2"){
        return std::make_shared<nvcomp::GdeflateManager>(nv_chunk_size,2,stream, 0);
    }else if(name == "Bitcomp"){
        return std::make_shared<nvcomp::BitcompManager>(data_type,0,stream, 0);
    }else if(name == "ANS"){
        return std::make_shared<nvcomp::ANSManager>(nv_chunk_size,stream, 0);
    }else {
        util::Log::get().error_fmt("Received unknown comp type \"%s\"",name.c_str());
        return std::make_shared<nvcomp::LZ4Manager>(nv_chunk_size, data_type, stream, 0);
    }
}


}