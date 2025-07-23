#pragma once
#ifdef WITH_CPU_COMP
#include <cstdint>

#include "lz4.h"
#include "snappy.h"

namespace golap {

class CPUCompManagerBase {
public:
    /**
     * return the number of bytes of the compressed output
     */
    virtual int compress(const char* in_ptr,char* out_ptr, int in_size) = 0;
    /**
     * return the number of bytes of the uncompressed output
     */
    virtual int decompress(char* out_ptr, const char* in_ptr, int comp_size, int decomp_size) = 0;
};

class LZ4CPUManager : public CPUCompManagerBase{
public:
    int decompress(char* out_ptr, const char* in_ptr, int comp_size, int decomp_size){
        // return LZ4_decompress_safe(in_ptr, out_ptr, comp_size, decomp_size);
        return LZ4_decompress_safe_partial(in_ptr, out_ptr, comp_size, decomp_size, decomp_size);
    }
    int compress(const char* in_ptr,char* out_ptr, int in_size){
        // let the chunk compressor detect an overflow
        return LZ4_compress_default(in_ptr, out_ptr, in_size, in_size<<1);
    }
};

class SnappyCPUManager : public CPUCompManagerBase{
public:
    int decompress(char* out_ptr, const char* in_ptr, int comp_size, int decomp_size){
        util::Log::get().debug_fmt("In SnappyCPUManager: comp_size=%lu", comp_size);
        if(!snappy::RawUncompress(in_ptr, comp_size, out_ptr)){
            util::Log::get().warn_fmt("Could have failed!");
        }
        return decomp_size;
    }
    int compress(const char* in_ptr,char* out_ptr, int in_size){
        snappy::RawCompress(in_ptr, in_size, out_ptr, &last_compressed);
        util::Log::get().debug_fmt("In SnappyCPUManager: %lu to %lu",in_size,last_compressed);
        return last_compressed;
    }
    size_t last_compressed;
};

class CPUChunkCompressor : public Op{
public:
    CPUChunkCompressor(uint64_t chunk_bytes, char* in_ptr, char* out_ptr, bool advance_in, bool advance_out,
              CPUCompManagerBase &comp_manager, std::vector<uint64_t> &chunk_size_vec):
            comp_manager(comp_manager),chunk_size_vec(chunk_size_vec),
            chunk_bytes(chunk_bytes),in_ptr(in_ptr),out_ptr(out_ptr),advance_in(advance_in),advance_out(advance_out){
        if (advance_out) throw std::runtime_error("advance_out=true not supported currently for Compressor");
    }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (child!=nullptr && !child->step(stream,event)) return false;

        int ret_size = comp_manager.compress((const char*) in_ptr,(char*) out_ptr, chunk_bytes);

        util::Log::get().debug_fmt("Step in CPUChunkCompressor. From %p, to %p, chunk_bytes %lu, compressed to %lu",in_ptr,out_ptr,chunk_bytes,ret_size);


        if (ret_size > (chunk_bytes + (chunk_bytes>>2))){
            util::Log::get().error_fmt("Last step wrote past end of buffer! %d is more than %lu!",ret_size,chunk_bytes);
        }
        last_produced = ret_size;

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

private:
    CPUCompManagerBase &comp_manager;
    std::vector<uint64_t> &chunk_size_vec;
    uint64_t chunk_bytes;
    uint64_t cur_chunk = 0;
    char *in_ptr,*out_ptr;
    bool advance_in,advance_out;
};


class CPUChunkDecompressor : public Op{
public:
    CPUChunkDecompressor(uint64_t chunk_bytes, char* in_ptr, char* out_ptr, bool advance_in, bool advance_out,
                          CPUCompManagerBase &comp_manager, std::vector<uint64_t> &chunk_size_vec):
        comp_manager(comp_manager),chunk_size_vec(chunk_size_vec),chunk_bytes(chunk_bytes),in_ptr(in_ptr),out_ptr(out_ptr),advance_in(advance_in),advance_out(advance_out){
    }
    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (child!=nullptr && !child->step(stream,event)) return false;

        child->finish_step(stream);
        util::Log::get().debug_fmt("Step in CPUChunkDecompressor. Expanding %lu bytes from %p to %p hopefully produced %lu",child->last_produced,in_ptr,out_ptr,chunk_bytes);


        int ret_size = comp_manager.decompress((char*) out_ptr, (const char*) in_ptr, (int)child->last_produced, (int)chunk_bytes);
        // checkCudaErrors(cudaEventRecord(event,stream));

        if (ret_size != chunk_bytes){
            util::Log::get().warn_fmt("CPUChunkDecompressor did not expand the expected size %lu, got %d",chunk_bytes,ret_size);
        }

        if(advance_out) out_ptr += chunk_bytes;
        last_produced = chunk_bytes;
        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if (child!=nullptr && !child->skip_step(stream,event)) return false;
        return true;
    }
private:
    CPUCompManagerBase &comp_manager;
    std::vector<uint64_t> &chunk_size_vec;
    char *in_ptr, *out_ptr;
    bool advance_in,advance_out;
    uint64_t chunk_bytes;
};

static std::shared_ptr<CPUCompManagerBase> build_cpucomp_manager(std::string name){
    if (name == "LZ4"){
        return std::make_shared<LZ4CPUManager>();
    }else if (name == "Snappy"){
        return std::make_shared<SnappyCPUManager>();
    }else {
        util::Log::get().error_fmt("Received unknown comp type \"%s\"",name.c_str());
        return std::make_shared<LZ4CPUManager>();
    }
}

} // end of namespace

#endif //WITH_CPU_COMP
