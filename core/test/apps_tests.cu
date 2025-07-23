#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "helper_cuda.h"

#include "test_common.hpp"

#include "apps.cuh"
#include "util.hpp"

TEST_CASE("Prepare (un)compress", "[apps]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    golap::CStream gstream{"Test Stream"};
    golap::CEvent gevent;

    uint64_t ROW_NUM = 1000000;
    uint64_t CHUNK_BYTES = (1<<20);

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);
    uint64_t offset = 0;


    struct COLS{
        enum {ENTE=0,GANS=1};
    };

    golap::ColumnTable<golap::HostMem,uint32_t,double> table{"ente,gans", ROW_NUM};
    table.num_tuples = ROW_NUM;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
    }
    golap::CompInfo ente_comp{CHUNK_BYTES, ROW_NUM*sizeof(uint32_t), "UNCOMPRESSED"};
    golap::CompInfo gans_comp{CHUNK_BYTES, ROW_NUM*sizeof(double), "UNCOMPRESSED"};

    golap::prepare_uncompressed(table.col<COLS::ENTE>(), ROW_NUM, ente_comp);
    golap::prepare_uncompressed(table.col<COLS::GANS>(), ROW_NUM, gans_comp);

    uint64_t ente_row_num = ROW_NUM;
    REQUIRE(ente_comp.blocks.size() == util::div_ceil(ROW_NUM*sizeof(uint32_t), CHUNK_BYTES));
    for (auto& block : ente_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == std::min(ente_row_num,CHUNK_BYTES/sizeof(uint32_t)));
        REQUIRE(block.offset == offset);
        ente_row_num -= std::min(ente_row_num,CHUNK_BYTES/sizeof(uint32_t));
        offset += block.size;
    }

    uint64_t gans_row_num = ROW_NUM;
    REQUIRE(gans_comp.blocks.size() == util::div_ceil(ROW_NUM*sizeof(double), CHUNK_BYTES));
    for (auto& block : gans_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == std::min(gans_row_num,CHUNK_BYTES/sizeof(double)));
        REQUIRE(block.offset == offset);
        gans_row_num -= std::min(gans_row_num,CHUNK_BYTES/sizeof(double));
        offset += block.size;
    }


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("Variable chunk sizes, uncompressed", "[apps]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    golap::CStream gstream{"Test Stream"};
    golap::CEvent gevent;

    uint64_t ROW_NUM = 1000000;
    std::vector<uint64_t> chunk_size_vec{250000,250000,500000};

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);
    uint64_t offset = 0;


    struct COLS{
        enum {ENTE=0,GANS=1};
    };

    golap::ColumnTable<golap::HostMem,uint32_t,double> table{"ente,gans", ROW_NUM};
    table.num_tuples = ROW_NUM;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
    }
    golap::CompInfo ente_comp{(uint64_t)-1, ROW_NUM*sizeof(uint32_t), "UNCOMPRESSED"};
    for (auto &tup_count : chunk_size_vec) ente_comp.chunk_size_vec.push_back(tup_count*sizeof(uint32_t));
    golap::CompInfo gans_comp{(uint64_t)-1, ROW_NUM*sizeof(double), "UNCOMPRESSED"};
    for (auto &tup_count : chunk_size_vec) gans_comp.chunk_size_vec.push_back(tup_count*sizeof(double));

    golap::prepare_uncompressed(table.col<COLS::ENTE>(), ROW_NUM, ente_comp);
    golap::prepare_uncompressed(table.col<COLS::GANS>(), ROW_NUM, gans_comp);


    std::vector<uint64_t> all_blocks_idxs(ente_comp.blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);
    golap::TableLoader<golap::LoadEnv> table_loader{3};
    // golap::TableLoader<golap::DecompressEnv> table_loader{2};
    table_loader.add("ente", all_blocks_idxs, 0, ente_comp.blocks.size(), ente_comp, nvcomp::TypeOf<uint32_t>());
    table_loader.add("gans", all_blocks_idxs, 0, gans_comp.blocks.size(), gans_comp, nvcomp::TypeOf<uint64_t>());

    uint64_t ente_row_num = ROW_NUM;
    uint64_t chunk_idx = 0;
    REQUIRE(ente_comp.blocks.size() == chunk_size_vec.size());
    for (auto& block : ente_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == chunk_size_vec[chunk_idx]);
        REQUIRE(block.offset == offset);
        ente_row_num -= block.tuples;
        offset += block.size;
        chunk_idx += 1;
    }

    uint64_t gans_row_num = ROW_NUM;
    chunk_idx = 0;
    REQUIRE(gans_comp.blocks.size() == chunk_size_vec.size());
    for (auto& block : gans_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == chunk_size_vec[chunk_idx]);
        REQUIRE(block.offset == offset);
        gans_row_num -= block.tuples;
        offset += block.size;
        chunk_idx += 1;
    }

    // now try to actually load the columns
    uint64_t tuples_this_round;
    chunk_idx = 0;

    while(chunk_idx< table_loader.blockenvs.at("ente").myblocks.size()){
        // check predicate:
        if(!table_loader.rootop.step(table_loader.rootstream.stream,table_loader.rootevent.event)){
            printf("This shouldnt happen!\n");
        }

        tuples_this_round = table_loader.blockenvs.at("ente").myblocks[chunk_idx].tuples;

        checkCudaErrors(cudaStreamSynchronize(table_loader.rootstream.stream));

        printf("tuples_this_round %lu\n",tuples_this_round);

        chunk_idx += 1;
    }

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}


TEST_CASE("Variable chunk sizes compressed", "[apps]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    golap::CStream gstream{"Test Stream"};
    golap::CEvent gevent;

    uint64_t ROW_NUM = 1000000;
    std::vector<uint64_t> chunk_size_vec{250000,250000,500000};

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);
    uint64_t offset = 0;


    struct COLS{
        enum {ENTE=0,GANS=1};
    };

    golap::ColumnTable<golap::HostMem,uint32_t,double> table{"ente,gans", ROW_NUM};
    table.num_tuples = ROW_NUM;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
    }
    golap::CompInfo ente_comp{(uint64_t)-1, ROW_NUM*sizeof(uint32_t), "Gdeflate"};
    for (auto &tup_count : chunk_size_vec) ente_comp.chunk_size_vec.push_back(tup_count*sizeof(uint32_t));
    golap::CompInfo gans_comp{(uint64_t)-1, ROW_NUM*sizeof(double), "LZ4"};
    for (auto &tup_count : chunk_size_vec) gans_comp.chunk_size_vec.push_back(tup_count*sizeof(double));

    golap::prepare_compressed_device(table.col<COLS::ENTE>(), ROW_NUM, ente_comp);
    golap::prepare_compressed_device(table.col<COLS::GANS>(), ROW_NUM, gans_comp);


    std::vector<uint64_t> all_blocks_idxs(ente_comp.blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);
    // golap::TableLoader<golap::LoadEnv> table_loader{3};
    golap::TableLoader<golap::DecompressEnv> table_loader{2};
    table_loader.add("ente", all_blocks_idxs, 0, ente_comp.blocks.size(), ente_comp, nvcomp::TypeOf<uint32_t>());
    table_loader.add("gans", all_blocks_idxs, 0, gans_comp.blocks.size(), gans_comp, nvcomp::TypeOf<uint64_t>());


    uint64_t ente_row_num = ROW_NUM;
    uint64_t chunk_idx = 0;
    REQUIRE(ente_comp.blocks.size() == chunk_size_vec.size());
    for (auto& block : ente_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == chunk_size_vec[chunk_idx]);
        REQUIRE(block.offset == offset);
        ente_row_num -= block.tuples;
        offset += block.size;
        chunk_idx += 1;
    }

    uint64_t gans_row_num = ROW_NUM;
    chunk_idx = 0;
    REQUIRE(gans_comp.blocks.size() == chunk_size_vec.size());
    for (auto& block : gans_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == chunk_size_vec[chunk_idx]);
        REQUIRE(block.offset == offset);
        gans_row_num -= block.tuples;
        offset += block.size;
        chunk_idx += 1;
    }

    // now try to actually load the columns
    uint64_t tuples_this_round;
    chunk_idx = 0;

    while(chunk_idx< table_loader.blockenvs.at("ente").myblocks.size()){
        // check predicate:
        if(!table_loader.rootop.step(table_loader.rootstream.stream,table_loader.rootevent.event)){
            printf("This shouldnt happen!\n");
        }

        tuples_this_round = table_loader.blockenvs.at("ente").myblocks[chunk_idx].tuples;

        checkCudaErrors(cudaStreamSynchronize(table_loader.rootstream.stream));

        printf("tuples_this_round %lu\n",tuples_this_round);

        chunk_idx += 1;
    }


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("BatchIO: Variable chunk sizes compressed", "[apps]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck
    golap::CStream gstream{"Test Stream"};
    golap::CEvent gevent;

    uint64_t ROW_NUM = 1000000;
    std::vector<uint64_t> chunk_size_vec{250000,250000,500000};

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);
    uint64_t offset = 0;


    struct COLS{
        enum {ENTE=0,GANS=1};
    };

    golap::ColumnTable<golap::HostMem,uint32_t,double> table{"ente,gans", ROW_NUM};
    table.num_tuples = ROW_NUM;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
    }
    golap::CompInfo ente_comp{(uint64_t)-1, ROW_NUM*sizeof(uint32_t), "Gdeflate"};
    for (auto &tup_count : chunk_size_vec) ente_comp.chunk_size_vec.push_back(tup_count*sizeof(uint32_t));
    golap::CompInfo gans_comp{(uint64_t)-1, ROW_NUM*sizeof(double), "LZ4"};
    for (auto &tup_count : chunk_size_vec) gans_comp.chunk_size_vec.push_back(tup_count*sizeof(double));

    golap::prepare_compressed_device(table.col<COLS::ENTE>(), ROW_NUM, ente_comp);
    golap::prepare_compressed_device(table.col<COLS::GANS>(), ROW_NUM, gans_comp);


    std::vector<uint64_t> all_blocks_idxs(ente_comp.blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);
    // golap::TableLoader<golap::LoadEnv> table_loader{3};

    golap::BatchTableLoader<golap::DecompressEnvWOLoad> table_loader{2};
    table_loader.add("ente", all_blocks_idxs, 0, ente_comp.blocks.size(), ente_comp, nvcomp::TypeOf<uint32_t>());
    table_loader.add("gans", all_blocks_idxs, 0, gans_comp.blocks.size(), gans_comp, nvcomp::TypeOf<uint64_t>());


    uint64_t ente_row_num = ROW_NUM;
    uint64_t chunk_idx = 0;
    REQUIRE(ente_comp.blocks.size() == chunk_size_vec.size());
    for (auto& block : ente_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == chunk_size_vec[chunk_idx]);
        REQUIRE(block.offset == offset);
        ente_row_num -= block.tuples;
        offset += block.size;
        chunk_idx += 1;
    }

    uint64_t gans_row_num = ROW_NUM;
    chunk_idx = 0;
    REQUIRE(gans_comp.blocks.size() == chunk_size_vec.size());
    for (auto& block : gans_comp.blocks){
        printf("Offset=%lu, Size=%lu, Tuples=%lu\n",block.offset,block.size,block.tuples);
        REQUIRE(block.tuples == chunk_size_vec[chunk_idx]);
        REQUIRE(block.offset == offset);
        gans_row_num -= block.tuples;
        offset += block.size;
        chunk_idx += 1;
    }

    // now try to actually load the columns
    uint64_t tuples_this_round,tuple_idx=0;
    chunk_idx = 0;

    while(chunk_idx< table_loader.blockenvs.at("ente").myblocks.size()){
        // check predicate:
        if(!table_loader.rootop.step(table_loader.rootstream.stream,table_loader.rootevent.event)){
            printf("This shouldnt happen!\n");
        }

        tuples_this_round = table_loader.blockenvs.at("ente").myblocks[chunk_idx].tuples;

        checkCudaErrors(cudaStreamSynchronize(table_loader.rootstream.stream));

        printf("tuples_this_round %lu\n",tuples_this_round);

        // compare the tuples
        uint32_t entes[tuples_this_round];
        double ganss[tuples_this_round];
        checkCudaErrors(cudaMemcpy(entes, table_loader.blockenvs.at("ente").decomp_buf.template ptr<uint32_t>(), sizeof(uint32_t)*tuples_this_round, cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(ganss, table_loader.blockenvs.at("gans").decomp_buf.template ptr<double>(), sizeof(double)*tuples_this_round, cudaMemcpyDefault));

        for (uint64_t tuple_in_block=0; tuple_in_block<tuples_this_round; ++tuple_in_block){
            auto ente_truth = table.col<COLS::ENTE>().data()[tuple_idx];
            auto ente_loaded = entes[tuple_in_block];
            REQUIRE(ente_truth == ente_loaded);

            auto gans_truth = table.col<COLS::GANS>().data()[tuple_idx];
            auto gans_loaded = ganss[tuple_in_block];
            REQUIRE(gans_truth == gans_loaded);
            tuple_idx += 1;
        }
        chunk_idx += 1;
    }


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}
