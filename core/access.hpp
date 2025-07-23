#pragma once

#include <cstdint>
#include <algorithm>
#include <stdexcept>

#include "core.hpp"
#include "util.hpp"
#include "mem.hpp"
#include "storage.hpp"
#include "comp.cuh"

namespace golap{


template <typename T>
class DataCopy : public Op{
public:
    DataCopy(T *from, T *to, bool advance_from, bool advance_to, uint64_t chunk_bytes, std::vector<uint64_t> &chunk_size_vec, uint64_t total_bytes = 0):
        from(from),to(to),advance_from(advance_from),advance_to(advance_to),chunk_bytes(chunk_bytes),total_bytes(total_bytes),left_to_read(total_bytes != 0 ? total_bytes : chunk_bytes),chunk_size_vec(chunk_size_vec){
            last_produced = chunk_bytes;
        }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if((child != nullptr && !child->step(stream,event))
            || (total_bytes != 0 && left_to_read == 0) // total bytes limit set and reached
            || (chunk_bytes == (uint64_t) -1 && cur_chunk == chunk_size_vec.size()) // chunk_size_vec set and reached end
            ) return false;
        if (chunk_bytes != (uint64_t) -1) {
            if(total_bytes != 0){
                last_produced = std::min(left_to_read, chunk_bytes);
            }
        }else {
            last_produced = chunk_size_vec[cur_chunk];
        }
        left_to_read -= last_produced;
        util::Log::get().debug_fmt("Step in DataCopy. Copying %lu, from %p, to %p",last_produced,from,to);
        checkCudaErrors(cudaMemcpyAsync((void*) to, (void*) from, last_produced, cudaMemcpyDefault, stream));
        checkCudaErrors(cudaEventRecord(event,stream));
        if(advance_to) to = (T*) (((char*)to)+last_produced);
        if(advance_from) from = (T*) (((char*)from)+last_produced);
        cur_chunk += 1;
        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr || !child->skip_step(stream,event)) return false;
        if (chunk_bytes != (uint64_t) -1) {
            if(total_bytes != 0){
                last_produced = std::min(left_to_read, chunk_bytes);
            }
        }else {
            last_produced = chunk_size_vec[cur_chunk];
        }
        left_to_read -= last_produced;
        if(advance_to) to = (T*) (((char*)to)+last_produced);
        if(advance_from) from = (T*) (((char*)from)+last_produced);
        cur_chunk += 1;

        return true;
    }
private:
    T *from,*to;
    uint64_t chunk_bytes,total_bytes,left_to_read;
    std::vector<uint64_t> &chunk_size_vec;
    uint64_t cur_chunk = 0;
    bool advance_from,advance_to;
};


/**
 * Read from disk in chunk_size sized chunks, starting from store_offset. 
 */
template <typename T>
class ChunkLoader : public Op{
public:
    ChunkLoader(uint64_t chunk_bytes, uint64_t store_offset, T *out_ptr, bool advance_out):
            sm(StorageManager::get()),chunk_bytes(chunk_bytes),cur_offset(store_offset),
            out_ptr(out_ptr),advance_out(advance_out){
                last_produced = chunk_bytes;
            }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr || !child->step(stream,event)){
            return false;
        }

        // since the async version of cuFileRead is not yet available, we'll have to make sure manually that the last parent op is finished
        checkCudaErrors(cudaEventSynchronize(parent_event));

        util::Log::get().debug_fmt("Step in ChunkLoader. Size %llu, offset %llu, to %p",chunk_bytes,cur_offset,out_ptr);

        sm.read_bytes(out_ptr,chunk_bytes,cur_offset);

        if (advance_out) out_ptr = (T*) ((char*)out_ptr)+chunk_bytes;
        cur_offset += chunk_bytes;

        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr || !child->skip_step(stream,event)) return false;
        if (advance_out) out_ptr = (T*) ((char*)out_ptr)+chunk_bytes;
        cur_offset += chunk_bytes;

        return true;
    }
private:
    StorageManager &sm;
    uint64_t chunk_bytes;

    uint64_t cur_offset;
    T *out_ptr;
    bool advance_out;
};


template <typename T>
class DenseWriter : public Op{
public:
    DenseWriter(uint64_t store_offset, std::vector<BlockInfo> &blocks, T *in_ptr):
            sm(StorageManager::get()),blocks(blocks),cur_offset(store_offset),in_ptr(in_ptr){
                last_produced = 0;
            }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr ||  !child->step(stream,event)){
            return false;
        }
        child->finish_step(stream);
        uint64_t to_write = child->last_produced;

        // pwrite64 is unhappy if were not writing in multiples of 4kb ...
        to_write += util::to_next((long long)to_write,4096);

        // tuples have to be set in higher level
        blocks.push_back(BlockInfo{cur_offset, to_write, child->last_produced});

        sm.write_bytes(in_ptr,to_write,cur_offset);

        cur_offset += to_write;
        Op::last_produced = to_write;
        util::Log::get().debug_fmt("Step in DenseWriter. from %p, store_offset %lu, content %lu, wrote %lu",in_ptr,cur_offset-to_write,child->last_produced,Op::last_produced);

        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr || !child->skip_step(stream,event)) return false;

        return true;
    }

private:
    StorageManager &sm;

    std::vector<BlockInfo> &blocks;
    uint64_t cur_offset;
    T *in_ptr;
};

template <typename T>
class VarCopy : public Op{
public:
    VarCopy(T *from, T *to, uint64_t chunk_bytes):
        from(from),to(to),chunk_bytes(chunk_bytes){}

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child != nullptr && !child->step(stream,event)) return false;

        child->finish_step(stream);

        last_produced = child->last_produced;

        util::Log::get().debug_fmt("Step in VarCopy (chunk_size = %llu). copying %llu from %p, to %p",
                                   chunk_bytes,last_produced,from,to);
        checkCudaErrors(cudaMemcpyAsync((void*) to, (void*) from, last_produced, cudaMemcpyDefault, stream));
        checkCudaErrors(cudaEventRecord(event,stream));

        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr || !child->skip_step(stream,event)) return false;

        return true;
    }
private:
    T *from,*to;
    uint64_t chunk_bytes;
};

template <typename T>
class VarLoader : public Op{
public:
    VarLoader(const VarLoader &obj) = delete;
    VarLoader(VarLoader&&) = default;
    VarLoader(std::vector<BlockInfo> &blocks, T *out_ptr):
            sm(StorageManager::get()),blocks(blocks),out_ptr(out_ptr),cur_block(0){
                last_produced = 0;
                if (blocks.size() == 0) return;

                uint64_t max_block_size = 0;
                for (auto &block:blocks){
                    if (block.size > max_block_size){
                        max_block_size = block.size;
                    }
                }
                uint64_t reg_size = util::div_ceil(max_block_size,CUFILE_MAGIC_NUMBER)*CUFILE_MAGIC_NUMBER;
                util::Log::get().debug_fmt("Construct VarLoader. Out=%p, max_block_size=%lu, reg_size=%lu",out_ptr,max_block_size,reg_size);
                checkCuFileError(cuFileBufRegister(out_ptr, reg_size, 0));
            }
    ~VarLoader(){
        if (blocks.size() == 0) return;
        checkCuFileError(cuFileBufDeregister(out_ptr));
    }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (cur_block>=blocks.size()) return false;

        auto& block = blocks[cur_block];

        // Since the async version of cuFileRead is not yet available, we'll have to make sure manually that the last parent op is finished.
        if (EVENT_SYNC){
            // If we sync on the parent event, we allow overlap with operations further up (e.g. Decompression).
            checkCudaErrors(cudaEventSynchronize(parent_event));
        }else{
            // If we sync the whole stream, no intra pipeline overlap is possible.
            checkCudaErrors(cudaStreamSynchronize(stream));
        }

        util::Log::get().debug_fmt("Step in VarLoader. To %p, store offset %lu, reading %lu, payload %lu",out_ptr, block.offset, block.size, block.payload);

        int64_t ret = cuFileRead(sm.cfh, out_ptr, block.size, block.offset, 0);

        if(ret < 0) util::Log::get().error_fmt("CuFileError: %s",CUFILE_ERRSTR(ret));
        if(ret != block.size)util::Log::get().error_fmt("Read %ld instead of %lu bytes!", ret, block.size);

        Op::last_produced = block.payload;
        cur_block += 1;

        return true;
    }

    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if (cur_block>=blocks.size()) return false;
        auto& block = blocks[cur_block];
        util::Log::get().debug_fmt("SkipStep in VarLoader. Skipped: Store offset %lu, reading %lu, payload %lu",block.offset, block.size, block.payload);

        cur_block += 1;
        return true;
    }

    void set_child(Op *op){
        throw std::runtime_error("Can't set child on VarLoader!");
    }
    inline static bool EVENT_SYNC = true;
private:
    StorageManager &sm;

    uint64_t cur_block;
    std::vector<BlockInfo> &blocks;
    T *out_ptr;
};



template <typename T>
class VarLoaderAsync : public Op{
public:
    VarLoaderAsync(const VarLoaderAsync &obj) = delete;
    VarLoaderAsync(VarLoaderAsync&&) = default;
    VarLoaderAsync(std::vector<BlockInfo> &blocks, T *out_ptr):
            sm(StorageManager::get()),blocks(blocks),out_ptr(out_ptr),cur_block(0){
                last_produced = 0;
                if (blocks.size() == 0) return;

                uint64_t max_block_size = 0;
                for (auto &block:blocks){
                    if (block.size > max_block_size){
                        max_block_size = block.size;
                    }
                }
                uint64_t reg_size = util::div_ceil(max_block_size,CUFILE_MAGIC_NUMBER)*CUFILE_MAGIC_NUMBER;
                util::Log::get().debug_fmt("Construct VarLoaderAsync. Out=%p, max_block_size=%lu, reg_size=%lu",out_ptr,max_block_size,reg_size);
                checkCuFileError(cuFileBufRegister(out_ptr, reg_size, 0));
            }
    ~VarLoaderAsync(){
        if (blocks.size() == 0) return;
        checkCuFileError(cuFileBufDeregister(out_ptr));
    }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (cur_block>=blocks.size()) return false;

        auto& block = blocks[cur_block];

        util::Log::get().debug_fmt("Step in VarLoaderAsync. To %p, store offset %lu, reading %lu, payload %lu",out_ptr, block.offset, block.size, block.payload);

        // cuFileRead(CUfileHandle_tfh, void *bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset);
        // cuFileReadAsync(CUFileHandle_t fh,
        //                 void *bufPtr_base, 
        //                 size_t *size_p,
        //                 off_t file_offset_p, 
        //                 off_t bufPtr_offset_p,
        //                 int *bytes_read_p,
        //                 CUstream stream);
        cur_offset = block.offset;
        checkCuFileError(cuFileReadAsync(sm.cfh, out_ptr, &block.size, &cur_offset, &const_zero,
                                         async_read_bytes.ptr<ssize_t>(), stream));

        Op::last_produced = block.payload;
        cur_block += 1;

        return true;
    }

    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if (cur_block>=blocks.size()) return false;
        auto& block = blocks[cur_block];
        util::Log::get().debug_fmt("SkipStep in VarLoader. Skipped: Store offset %lu, reading %lu, payload %lu",block.offset, block.size, block.payload);

        cur_block += 1;
        return true;
    }

    void set_child(Op *op){
        throw std::runtime_error("Can't set child on VarLoader!");
    }
    inline static bool EVENT_SYNC = true;
private:
    StorageManager &sm;

    off_t cur_offset;
    off_t const_zero = 0;
    golap::HostMem async_read_bytes{golap::Tag<ssize_t>{}, 1};
    uint64_t cur_block;
    std::vector<BlockInfo> &blocks;
    T *out_ptr;
};

class BatchLoader : public Op{
public:
    BatchLoader(const BatchLoader &obj) = delete;
    BatchLoader(BatchLoader&&) = default;
    BatchLoader(uint64_t num_cols, bool advance_out = false) : sm(StorageManager::get()),num_cols(num_cols),cur_block(0),advance_out(advance_out){
        last_produced = 0;
        io_batch_params = new CUfileIOParams_t[num_cols];
        io_batch_events = new CUfileIOEvents_t[num_cols];
    }
    ~BatchLoader(){
        delete[] io_batch_params;
        delete[] io_batch_events;
    }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (cur_block>=blockss[0].size()) return false;

        util::Log::get().debug_fmt("A step in BatchLoader.");

        for (auto col_idx = 0; col_idx < blockss.size(); ++col_idx){
            auto &out_ptr = out_ptrs[col_idx];
            auto &block = blockss[col_idx][cur_block];
            util::Log::get().debug_fmt("A block in BatchLoader. To %p, store offset %lu, reading %lu, payload %lu", out_ptr, block.offset, block.size, block.payload);
            // put this info into a batch
            io_batch_params[col_idx].mode = CUFILE_BATCH;
            io_batch_params[col_idx].fh = sm.cfh;
            io_batch_params[col_idx].u.batch.devPtr_base = out_ptr;
            io_batch_params[col_idx].u.batch.file_offset = block.offset;
            io_batch_params[col_idx].u.batch.devPtr_offset = 0;
            io_batch_params[col_idx].u.batch.size = block.size;
            io_batch_params[col_idx].opcode = CUFILE_READ;
        }

        auto errorBatch = cuFileBatchIOSetUp(&batch_id, blockss.size());
        if(errorBatch.err != 0) {
            std::cerr << "Error in setting Up Batch" << std::endl;
            return false;
        }

        errorBatch = cuFileBatchIOSubmit(batch_id, blockss.size(), io_batch_params, 0);
        if(errorBatch.err != 0) {
            std::cerr << "Error in IO Batch Submit " << errorBatch.err<<  std::endl;
            return false;
        }

        unsigned nr;
        unsigned num_completed = 0;
        while(num_completed != blockss.size()) {
            memset(io_batch_events, 0, sizeof(CUfileIOEvents_t)*num_cols);
            nr = blockss.size();
            errorBatch = cuFileBatchIOGetStatus(batch_id, blockss.size(), &nr, io_batch_events, NULL);
            if(errorBatch.err != 0) {
                std::cerr << "Error in IO Batch Get Status" << std::endl;
                return false;
            }
            // std::cout << "Got events " << nr << std::endl;
            num_completed += nr;
            // for(unsigned i = 0; i < nr; i++) {
            //     uint64_t buf[blockss.size()];
            //     cudaMemcpyAsync(buf, io_batch_params[i].u.batch.devPtr_base, io_batch_events[i].ret, cudaMemcpyDefault, stream);
            //     std::cout << "Completed  IO, index" << i << "size: " << io_batch_events[i].ret << std::endl;
            // }
        }

        cuFileBatchIODestroy(batch_id);

        // int64_t ret = cuFileRead(sm.cfh, out_ptr, block.size, block.offset, 0);

        // if(ret < 0) util::Log::get().error_fmt("CuFileError: %s",CUFILE_ERRSTR(ret));
        // if(ret != block.size)util::Log::get().error_fmt("Read %ld instead of %lu bytes!", ret, block.size);

        // Op::last_produced = block.payload;
        cur_block += 1;

        return true;
    }

    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        // if (cur_block>=blocks.size()) return false;
        // auto& block = blocks[cur_block];
        // util::Log::get().debug_fmt("SkipStep in VarLoader. Skipped: Store offset %lu, reading %lu, payload %lu",block.offset, block.size, block.payload);

        cur_block += 1;
        return true;
    }
    std::vector<char*> out_ptrs;
    std::vector<std::vector<BlockInfo>> blockss;

private:
    uint64_t num_cols;
    CUfileBatchHandle_t batch_id;
    CUfileIOParams_t *io_batch_params;
    CUfileIOEvents_t *io_batch_events;
    StorageManager &sm;
    uint64_t cur_block;
    std::vector<char*> out_ptr;
    bool advance_out;
};


template <typename T>
class VarLoaderCPU : public Op{
public:
    VarLoaderCPU(std::vector<BlockInfo> &blocks, T *out_ptr, bool advance_out = false):
            sm(StorageManager::get()),blocks(blocks),out_ptr(out_ptr),cur_block(0),advance_out(advance_out){
                last_produced = 0;
            }

    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (cur_block>=blocks.size()) return false;

        auto& block = blocks[cur_block];

        // Since the async version of cuFileRead is not yet available, we'll have to make sure manually that the last parent op is finished.
        if (EVENT_SYNC){
            // If we sync on the parent event, we allow overlap with operations further up (e.g. Decompression).
            checkCudaErrors(cudaEventSynchronize(parent_event));
        }else{
            // If we sync the whole stream, no intra pipeline overlap is possible.
            checkCudaErrors(cudaStreamSynchronize(stream));
        }

        util::Log::get().debug_fmt("Step in VarLoaderCPU. To %p, store offset %lu, reading %lu, payload %lu",out_ptr,block.offset,block.size,block.payload);

        sm.host_read_bytes(out_ptr,block.size,block.offset);

        if (advance_out) out_ptr += block.size;

        Op::last_produced = block.payload;
        cur_block += 1;

        return true;
    }

    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        cur_block += 1;

        return true;
    }

    void set_child(Op *op){
        throw std::runtime_error("Can't set child on VarLoaderCPU!");
    }
    inline static bool EVENT_SYNC = true;
private:
    StorageManager &sm;
    uint64_t cur_block;
    std::vector<BlockInfo> &blocks;
    T *out_ptr;
    bool advance_out;
};






} // end of namespace

