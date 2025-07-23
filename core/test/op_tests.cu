#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>

#include "helper_cuda.h"

#include "test_common.hpp"
#include "access.hpp"
#include "mem.hpp"
#include "util.hpp"
#include "comp.cuh"



TEST_CASE("Test HL api", "[op]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    golap::CStream gstream{"Test Stream"};
    golap::CEvent gevent;

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);


    uint64_t BYTES = ((uint64_t)1<<31); // 4 Mio
    uint64_t CHUNK = ((uint64_t)1<<29);
    std::vector<uint64_t> dummy;
    uint64_t store_offset = 0;
    golap::MirrorMem mem0{golap::Tag<char>(), CHUNK};
    uint32_t* mem0_intptr = mem0.hst.ptr<uint32_t>();
    golap::DeviceMem mem1{golap::Tag<char>(), CHUNK};
    golap::HostMem final_host{golap::Tag<char>(), BYTES};

    for(uint64_t i=0;i<mem0.size<uint32_t>();++i){
        mem0_intptr[i] = i % 4096;
    }
    // write same chunk multiple times
    for(uint64_t offset = 0; offset<BYTES; offset+=CHUNK){
        sm.host_write_bytes(mem0_intptr,CHUNK,offset);
    }

    golap::DeviceMem compressed{golap::Tag<char>(), CHUNK<<1};

    golap::DoXTimes doxtimes{BYTES/CHUNK};
    golap::ChunkLoader chunk_loader{CHUNK, store_offset, mem0.dev.ptr<int>(), false};
    chunk_loader.set_child(&doxtimes);

    nvcomp::LZ4Manager comp_manager((1<<20), nvcomp::TypeOf<int>(), gstream.stream);

    golap::ChunkCompressor compressor_wrapper{CHUNK, mem0.dev.ptr<char>(), compressed.ptr<char>(),
                                                        false,false,comp_manager,dummy};
    compressor_wrapper.set_child(&chunk_loader);

    golap::ChunkDecompressor decompressor_wrapper{CHUNK, compressed.ptr<char>(), mem1.ptr<char>(),
                                    false,false,comp_manager,dummy};
    decompressor_wrapper.set_child(&compressor_wrapper);

    golap::DataCopy data_copy{mem1.ptr<char>(),final_host.ptr<char>(),false,true,CHUNK,dummy};
    data_copy.set_child(&decompressor_wrapper);

    while(data_copy.step(gstream.stream,gevent.event)){
        // checkCudaErrors(cudaStreamSynchronize(gstream.stream));
    }
    checkCudaErrors(cudaStreamSynchronize(gstream.stream));

    for(uint64_t i=0;i<final_host.size<uint32_t>();++i){
        REQUIRE(mem0_intptr[i % (mem0.size<uint32_t>())] == final_host.ptr<uint32_t>()[i]);
    }


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}



TEST_CASE("Dense compressed chunks", "[op]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    golap::CStream gstream{"Test Stream"};
    golap::CEvent gevent;

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);


    uint64_t BYTES = ((uint64_t)1<<31); // 2 GB
    uint64_t CHUNK = ((uint64_t)1<<28); // 256mb chunks
    std::vector<uint64_t> dummy;
    std::vector<golap::BlockInfo> blocks;
    uint64_t store_offset = 0;
    golap::MirrorMem mem0{golap::Tag<char>(), BYTES};
    golap::DeviceMem compressed{golap::Tag<char>(), CHUNK};

    checkCudaErrors(cudaMemset(mem0.dev.ptr<char>(),'*',BYTES));

    golap::DoXTimes dox{BYTES/CHUNK};
    nvcomp::LZ4Manager comp_manager((1<<16), nvcomp::TypeOf<int>(), gstream.stream);
    golap::ChunkCompressor compressor{CHUNK, mem0.dev.ptr<char>(), compressed.ptr<char>(),
                                            true,false,comp_manager, dummy};
    compressor.set_child(&dox);
    golap::DenseWriter writer{store_offset,blocks,compressed.ptr<char>()};
    writer.set_child(&compressor);

    writer.describe();

    uint64_t total_written = 0,i=0;
    while(writer.step(gstream.stream,gevent.event)){
        total_written += writer.last_produced;
        // blocks[i] = writer.last_produced;
        // printf("Offset: %lu, Size: %lu\n",blocks[i].offset,blocks[i].size);
        REQUIRE(blocks[i].size == writer.last_produced);
        i+=1;
    }

    REQUIRE(total_written < BYTES);


    checkCudaErrors(cudaMemset(compressed.ptr<char>(),'+',total_written));
    /**
     * Now read the compressed data from disk
     */
    golap::VarLoader loader{blocks, compressed.ptr<char>()};

    golap::ChunkDecompressor decompressor{CHUNK, compressed.ptr<char>(), mem0.dev.ptr<char>(),
                            false,true,comp_manager,dummy};
    decompressor.set_child(&loader);
    decompressor.describe();

    uint64_t total_expanded = 0;
    while(decompressor.step(gstream.stream,gevent.event)){
        total_expanded += decompressor.last_produced;
    }
    mem0.sync_to_host(gstream.stream);
    checkCudaErrors(cudaStreamSynchronize(gstream.stream));
    REQUIRE(total_expanded == BYTES);

    for(uint64_t i = 0; i<mem0.size_bytes(); ++i){
        REQUIRE(mem0.hst.ptr<char>()[i] == '*');
    }

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

