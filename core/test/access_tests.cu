#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "helper_cuda.h"

#include "test_common.hpp"

#include "access.hpp"
#include "storage.hpp"

TEST_CASE("Chunk Access Device", "[access]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    golap::CStream gstream{"Test Stream"};
    golap::CEvent gevent;

    uint64_t store_offset = 0;

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);

    golap::HostMem host_mem{golap::Tag<char>(), 8192};
    golap::MirrorMem mirror_mem{golap::Tag<char>(), 8192};
    for(int i=0; i<host_mem.size_bytes(); ++i){
        host_mem.ptr<char>()[i] = '^';
    }
    sm.host_write_bytes(host_mem.ptr<char>(),8192,store_offset);

    golap::DoXTimes doxtimes{8192/4096};
    golap::ChunkLoader chunk_loader{4096, store_offset, mirror_mem.dev.ptr<char>(), true};
    chunk_loader.set_child(&doxtimes);

    golap::CountPipe top_op;
    top_op.set_child(&chunk_loader);

    while(top_op.step(gstream.stream,gevent.event));
    REQUIRE(top_op.steps == 2);
    checkCudaErrors(cudaStreamSynchronize(gstream.stream));

    mirror_mem.sync_to_host();
    for(int i=0; i<mirror_mem.size_bytes(); ++i){
        REQUIRE(mirror_mem.hst.ptr<char>()[i] == '^');
    }

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("Find buf reg bug", "[bug]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);

    golap::HostMem hostmem{golap::Tag<char>{}, 1};
    golap::DeviceMem mem{golap::Tag<char>{}, (1<<22), 4096, (1<<16)};

    printf("Is mem divisible by (1<<16): (%d)\n",(uint64_t) mem.data % (1<<16) == 0);
    // if ((uint64_t) mem.data % (1<<16) == 0){
    //     mem.data = (void*)((uint64_t) mem.data + (1<<15));
    // }
    printf("Is mem divisible by (1<<16): (%d)\n",(uint64_t) mem.data % (1<<16) == 0);

    uint64_t seed = util::Timer::time_seed();


    const uint32_t magic_num = (1<<16);
    for(uint32_t dummy = 0; dummy < 10; dummy+=1){
        uint32_t io_size = util::uniform_int(seed, (1<<14), (1<<20));

        io_size = util::div_ceil(io_size,magic_num)*magic_num;

        printf("Trying to reg %u bytes, dataptr=%p, div=%d\n", io_size, mem.data, io_size % (1<<16)== 0);
        REQUIRE((uint64_t) mem.data % 4096 == 0);
        REQUIRE(io_size % 4096 == 0);
        auto error = cuFileBufRegister(mem.data, io_size, 0);
        if(error.err == CU_FILE_SUCCESS) printf("\t!!!!!!! Success\n");
        else {
            printf("\tXXXXXXX Failure %d\n",error.err);
            continue;
        }



        int64_t ret = cuFileRead(sm.cfh, mem.data, io_size, 0 /*read from start of file*/, 0);

        printf("\tLast 16 bytes: ");
        for (int i = 0; i < 16; i += 1){
            checkCudaErrors(cudaMemcpy(hostmem.ptr<char>(), &mem.ptr<char>()[io_size-1-i], 1, cudaMemcpyDefault));
            printf("%d ", hostmem.ptr<char>()[0]);
        }
        printf("\n");


        REQUIRE(ret == io_size);

        golap::checkCuFileError(cuFileBufDeregister(mem.data));
    }

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}
