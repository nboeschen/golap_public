#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "helper_cuda.h"

#include "test_common.hpp"

#include "mem.hpp"
#include "util.hpp"
#include "apps.cuh"
#include "metadata.cuh"



TEST_CASE("Histogram simple", "[metadata]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t ROW_NUM = 1000000, num_buckets = 10, chunk_tuples = 250000;
    uint64_t chunk_num = ROW_NUM/chunk_tuples;
    uint32_t min = 100,max = 200;

    golap::MirrorMem data{golap::Tag<uint32_t>{}, ROW_NUM};
    golap::MirrorMem qualified{golap::Tag<uint16_t>{}, chunk_num};

    uint64_t seed = 0xfefefafa;

    for (uint64_t chunk_idx = 0; chunk_idx < chunk_num; chunk_idx += 1){
        for (uint64_t i = 0; i< chunk_tuples; ++i){
            data.hst.ptr<uint32_t>()[chunk_idx*chunk_tuples + i] = util::uniform_int(seed, min+50*chunk_idx, max+50*chunk_idx);
        }
    }

    data.sync_to_device();

    golap::EqHistogram<uint32_t> histogram(chunk_num, num_buckets, chunk_tuples);


    for (uint64_t chunk_idx = 0; chunk_idx < chunk_num; chunk_idx += 1){
        uint32_t* ptr = data.dev.ptr<uint32_t>()+ chunk_idx*chunk_tuples;
        histogram.init_chunk(chunk_idx, ptr, chunk_tuples);
        golap::fill_hist<<<20,512>>>(histogram, ptr, chunk_idx, chunk_tuples);
        checkCudaErrors(cudaDeviceSynchronize());
    }



    golap::check_hist<<<20,512>>>(histogram, qualified.dev.ptr<uint16_t>(), (uint32_t)150, (uint32_t)180);
    checkCudaErrors(cudaDeviceSynchronize());

    qualified.sync_to_host();
    golap::HostMem hist_info{golap::Tag<uint64_t>{}, (chunk_num)*num_buckets};
    golap::HostMem minhost{golap::Tag<uint32_t>{}, (chunk_num)};
    golap::HostMem maxhost{golap::Tag<uint32_t>{}, (chunk_num)};

    checkCudaErrors(cudaMemcpy(hist_info.data, histogram.hist, (chunk_num)*num_buckets*sizeof(uint64_t), cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(minhost.data, histogram.mins, (chunk_num)*sizeof(uint32_t), cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(maxhost.data, histogram.maxs, (chunk_num)*sizeof(uint32_t), cudaMemcpyDefault));

    for (uint64_t chunk_idx = 0; chunk_idx < chunk_num; chunk_idx += 1){
        printf("Chunk[%lu]. Min=%u, max=%u, qualified=%u\n",chunk_idx, minhost.ptr<uint32_t>()[chunk_idx], maxhost.ptr<uint32_t>()[chunk_idx],
                                                            qualified.hst.ptr<uint16_t>()[chunk_idx]);
    }


    uint64_t total = 0;
    for (uint64_t i = 0; i < (chunk_num)*num_buckets; ++i){
        uint64_t cur = hist_info.ptr<uint64_t>()[i];
        // printf("Got %lu in bucket [%u,%u)\n", cur, min + (uint32_t)(i*10), min + (uint32_t)((i+1)*10));
        printf("Got %lu in bucket\n", cur);
        total += cur;
    }

    REQUIRE(total == ROW_NUM);

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("EqHistogram", "[metadata]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t chunk_num = 3;
    uint64_t bucket_num = 4;
    uint64_t max_tuples_in_chunk = 8;

    golap::MirrorMem rows{golap::Tag<int>{},chunk_num*max_tuples_in_chunk};
    golap::MirrorMem res{golap::Tag<uint16_t>{},chunk_num};
    res.dev.set(0);

    rows.hst.ptr<int>()[0] = 15;
    rows.hst.ptr<int>()[1] = 16;
    rows.hst.ptr<int>()[2] = 17;
    rows.hst.ptr<int>()[3] = 18;
    rows.hst.ptr<int>()[4] = 19;
    rows.hst.ptr<int>()[5] = 20;
    rows.hst.ptr<int>()[6] = 21;
    rows.hst.ptr<int>()[7] = 22;

    rows.hst.ptr<int>()[8] = 0;
    rows.hst.ptr<int>()[9] = 1;
    rows.hst.ptr<int>()[10] = 7;
    rows.hst.ptr<int>()[11] = 8;
    rows.hst.ptr<int>()[12] = 9;
    rows.hst.ptr<int>()[13] = 10;
    rows.hst.ptr<int>()[14] = 11;
    rows.hst.ptr<int>()[15] = 12;

    rows.hst.ptr<int>()[16] = 1993;
    rows.hst.ptr<int>()[17] = 1993;
    rows.hst.ptr<int>()[18] = 1993;
    rows.hst.ptr<int>()[19] = 1993;
    rows.hst.ptr<int>()[20] = 1993;
    rows.hst.ptr<int>()[21] = 1993;
    rows.hst.ptr<int>()[22] = 1993;
    rows.hst.ptr<int>()[23] = 1993;

    rows.sync_to_device();

    golap::EqHistogram<int> hist0(chunk_num,bucket_num,max_tuples_in_chunk);

    for (int i=0; i<chunk_num; ++i){
        hist0.init_chunk(i, rows.hst.ptr<int>()+i*max_tuples_in_chunk, max_tuples_in_chunk);
        golap::fill_hist<<<20,512>>>(hist0, rows.hst.ptr<int>()+i*max_tuples_in_chunk, i, max_tuples_in_chunk);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    hist0.print_debug();



    golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), 3, 5);
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    for (int i=0; i<chunk_num; ++i){
        // std::cout << res.hst.ptr<uint16_t>()[i] << "\n";
        REQUIRE(res.hst.ptr<uint16_t>()[i] == 0);
    }

    res.dev.set(0);
    golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), 12, 12);
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 1);


    res.dev.set(0);
    golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), 0, 0);
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 1);

    res.dev.set(0);
    golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), 15, 16);
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);

    res.dev.set(0);
    golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), 1993, 1993);
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 1);

    res.dev.set(0);
    golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), 1994, 1994);
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);

    res.dev.set(0);
    golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), 14, 15);
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}


TEST_CASE("BloomMeta", "[metadata]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck


    golap::BloomMeta<int> bloommeta0(0,0.01,4096,1000000);
    golap::BloomMeta<int> bloommeta1(0,0.05,8096,1000000);
    golap::BloomMeta<int> bloommeta2(0,0.02,256,1000000);


    REQUIRE(true);

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("String Metadata", "[metadata]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t chunk_num = 3;
    uint64_t bucket_num = 4;
    uint64_t max_tuples_in_chunk = 8;

    using PTYPE = util::Padded<char[10]>;

    golap::MirrorMem rows{golap::Tag<PTYPE>{},chunk_num*max_tuples_in_chunk};
    golap::MirrorMem res{golap::Tag<uint16_t>{},chunk_num};
    res.dev.set(0);
    for(int i = 0; i<16; ++i){
        snprintf(rows.hst.ptr<PTYPE>()[i].d, 10, "h%d", i);
    }
    for(int i = 16; i<24; ++i){
        if (i & 1) new (rows.hst.ptr<PTYPE>()+i) PTYPE("Dec1996");
        else       new (rows.hst.ptr<PTYPE>()+i) PTYPE("Feb1992");
    }


    rows.sync_to_device();


    golap::MinMaxMeta<PTYPE> minmax(chunk_num,max_tuples_in_chunk);

    for (int i=0; i<chunk_num; ++i){
        minmax.init_chunk(i, rows.hst.ptr<PTYPE>()+i*max_tuples_in_chunk, max_tuples_in_chunk);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    minmax.print_debug();


    res.dev.set(0);
    golap::check_mmmeta<<<20,512>>>(minmax, res.dev.ptr<uint16_t>(), PTYPE("h5"), PTYPE("h5"));
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);

    res.dev.set(0);
    golap::check_mmmeta<<<20,512>>>(minmax, res.dev.ptr<uint16_t>(), PTYPE("h15"), PTYPE("h15"));
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);

    res.dev.set(0);
    golap::check_mmmeta<<<20,512>>>(minmax, res.dev.ptr<uint16_t>(), PTYPE("Dec1997"), PTYPE("Dec1997"));
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 1);


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("String Metadata EqHistogram", "[metadata]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t chunk_num = 3;
    uint64_t bucket_num = 4;
    uint64_t max_tuples_in_chunk = 8;

    using PTYPE = util::Padded<char[10]>;

    std::cout << util::Padded<char[10]>("~~~~~~") - util::Padded<char[10]>("aaa") << "\n";


    golap::MirrorMem rows{golap::Tag<PTYPE>{},chunk_num*max_tuples_in_chunk};
    golap::MirrorMem res{golap::Tag<uint16_t>{},chunk_num};
    res.dev.set(0);
    for(int i = 0; i<16; ++i){
        snprintf(rows.hst.ptr<PTYPE>()[i].d, 10, "h%d", i);
    }
    for(int i = 16; i<24; ++i){
        if (i & 1) new (rows.hst.ptr<PTYPE>()+i) PTYPE("Dec1996");
        else       new (rows.hst.ptr<PTYPE>()+i) PTYPE("Feb1992");
    }


    // rows.sync_to_device();


    // golap::EqHistogram<PTYPE> hist0(chunk_num,bucket_num,max_tuples_in_chunk);

    // for (int i=0; i<chunk_num; ++i){
    //     hist0.init_chunk(i, rows.hst.ptr<PTYPE>()+i*max_tuples_in_chunk, max_tuples_in_chunk);
    //     golap::fill_hist<<<20,512>>>(hist0, rows.hst.ptr<PTYPE>()+i*max_tuples_in_chunk, i, max_tuples_in_chunk);
    // }
    // checkCudaErrors(cudaDeviceSynchronize());

    // hist0.print_debug();


    // res.dev.set(0);
    // golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), PTYPE("h5"), PTYPE("h5"));
    // checkCudaErrors(cudaDeviceSynchronize());
    // res.sync_to_host();
    // REQUIRE(res.hst.ptr<uint16_t>()[0] == 1);
    // REQUIRE(res.hst.ptr<uint16_t>()[1] == 1);
    // REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);

    // res.dev.set(0);
    // golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), PTYPE("h15"), PTYPE("h15"));
    // checkCudaErrors(cudaDeviceSynchronize());
    // res.sync_to_host();
    // REQUIRE(res.hst.ptr<uint16_t>()[0] == 1);
    // REQUIRE(res.hst.ptr<uint16_t>()[1] == 1);
    // REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);

    // res.dev.set(0);
    // golap::check_hist<<<20,512>>>(hist0, res.dev.ptr<uint16_t>(), PTYPE("Dec1997"), PTYPE("Dec1997"));
    // checkCudaErrors(cudaDeviceSynchronize());
    // res.sync_to_host();
    // REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    // REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    // REQUIRE(res.hst.ptr<uint16_t>()[2] == 1);


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("String Metadata Bloom", "[metadata]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t chunk_num = 3;
    uint64_t bucket_num = 4;
    uint64_t max_tuples_in_chunk = 8;

    using PTYPE = util::Padded<char[10]>;

    golap::MirrorMem rows{golap::Tag<PTYPE>{},chunk_num*max_tuples_in_chunk};
    golap::MirrorMem res{golap::Tag<uint16_t>{},chunk_num};
    res.dev.set(0);
    for(int i = 0; i<16; ++i){
        snprintf(rows.hst.ptr<PTYPE>()[i].d, 10, "h%d", i);
    }
    for(int i = 16; i<24; ++i){
        if (i & 1) new (rows.hst.ptr<PTYPE>()+i) PTYPE("Dec1996");
        else       new (rows.hst.ptr<PTYPE>()+i) PTYPE("Feb1992");
    }
    // for(int i = 0; i<24; ++i){
    //     printf("Hash of %s is %lu\n", rows.hst.ptr<PTYPE>()[i].d, rows.hst.ptr<PTYPE>()[i].hash());
    // }
    rows.sync_to_device();


    golap::BloomMeta<PTYPE> meta(chunk_num,0.01,256,max_tuples_in_chunk);

    for (int i=0; i<chunk_num; ++i){
        golap::fill_bloom<<<20,512>>>(meta, rows.hst.ptr<PTYPE>()+i*max_tuples_in_chunk, i, max_tuples_in_chunk);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // meta.print_debug();


    res.dev.set(0);
    golap::check_bloom<<<20,512>>>(meta, res.dev.ptr<uint16_t>(), PTYPE("h5"));
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);

    res.dev.set(0);
    golap::check_bloom<<<20,512>>>(meta, res.dev.ptr<uint16_t>(), PTYPE("h15"));
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 1);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);

    res.dev.set(0);
    golap::check_bloom<<<20,512>>>(meta, res.dev.ptr<uint16_t>(), PTYPE("Dec1996"));
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 1);

    res.dev.set(0);
    golap::check_bloom<<<20,512>>>(meta, res.dev.ptr<uint16_t>(), PTYPE("Dec1997"));
    checkCudaErrors(cudaDeviceSynchronize());
    res.sync_to_host();
    REQUIRE(res.hst.ptr<uint16_t>()[0] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[1] == 0);
    REQUIRE(res.hst.ptr<uint16_t>()[2] == 0);


    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

TEST_CASE("Pruning metadata", "[metadata]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t ROW_NUM = 256;
    uint64_t CHUNK_BYTES = (1<<20);

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);


    struct COLS{
        enum {ENTE=0,GANS=1,HUHN=2};
        uint32_t ente;
        double gans;
        util::Padded<char[16]> huhn;
    };

    golap::ColumnTable<golap::HostMem,decltype(COLS::ente),decltype(COLS::gans),decltype(COLS::huhn)> table{"ente,gans,huhn", ROW_NUM};
    table.num_tuples = ROW_NUM;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
        snprintf(table.col<COLS::HUHN>().data()[i].d, sizeof(decltype(COLS::huhn.d)), "h%lu", i);
    }
    table.to_csv(std::cout, ", ", 0, 100);

    // uint64_t tuples_per_chunk = CHUNK_BYTES / std::max({sizeof(decltype(COLS::ente)),sizeof(decltype(COLS::gans)),sizeof(decltype(COLS::huhn))});
    uint64_t tuples_per_chunk = 16;
    golap::CompInfo ente_comp{tuples_per_chunk*sizeof(decltype(COLS::ente)), ROW_NUM*sizeof(decltype(COLS::ente)), "Gdeflate"};
    golap::CompInfo gans_comp{tuples_per_chunk*sizeof(decltype(COLS::gans)), ROW_NUM*sizeof(decltype(COLS::gans)), "Gdeflate"};
    golap::CompInfo huhn_comp{tuples_per_chunk*sizeof(decltype(COLS::huhn)), ROW_NUM*sizeof(decltype(COLS::huhn)), "Gdeflate"};

    golap::MinMaxMeta<decltype(COLS::ente)> ente_minmax;
    golap::MinMaxMeta<decltype(COLS::gans)> gans_minmax;
    golap::MinMaxMeta<decltype(COLS::huhn)> huhn_minmax;

    golap::EqHistogram<decltype(COLS::ente)> ente_hist(10);
    golap::EqHistogram<decltype(COLS::gans)> gans_hist(10);

    golap::BloomMeta<decltype(COLS::ente)> ente_bloom(0.01,8192);
    golap::BloomMeta<decltype(COLS::gans)> gans_bloom(0.01,8192);

    golap::prepare_uncompressed(table.col<COLS::ENTE>(), ROW_NUM, ente_comp, &ente_minmax, &ente_hist, &ente_bloom);
    golap::prepare_uncompressed(table.col<COLS::GANS>(), ROW_NUM, gans_comp, &gans_minmax, &gans_hist, &gans_bloom);
    golap::prepare_uncompressed(table.col<COLS::HUHN>(), ROW_NUM, huhn_comp, &huhn_minmax);

    // golap::prepare_compressed_device<golap::HostMem, decltype(COLS::ente)>(table.col<COLS::ENTE>(), ROW_NUM, ente_comp, &ente_minmax, &ente_hist, &ente_bloom);
    // golap::prepare_compressed_device<golap::HostMem, decltype(COLS::gans), uint8_t>(.col<COLS::GANS>(), ROW_NUM, gans_comp, &gans_minmax, &gans_hist, &gans_bloom);
    // golap::prepare_compressed_device(table.col<COLS::HUHN>(), ROW_NUM, huhn_comp);

    std::cout << huhn_minmax.chunk_num << "\n";
    huhn_minmax.print_debug();
    // ente_minmax.print_debug();
    // ente_hist.print_debug();
    // gans_minmax.print_debug();
    // gans_hist.print_debug();

    golap::TableLoader<golap::LoadEnv> table_loader{3};
    // golap::TableLoader<golap::DecompressEnv> table_loader{3};
    std::vector<uint64_t> all_blocks_idxs(ente_comp.blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);
    table_loader.add("ente", all_blocks_idxs, 0, ente_comp.blocks.size(), ente_comp, nvcomp::TypeOf<decltype(COLS::ente)>());
    table_loader.add("gans", all_blocks_idxs, 0, gans_comp.blocks.size(), gans_comp, nvcomp::TypeOf<uint8_t>());
    table_loader.add("huhn", all_blocks_idxs, 0, huhn_comp.blocks.size(), huhn_comp, nvcomp::TypeOf<uint8_t>());


    uint64_t round = 0;
    uint64_t tuples_this_round;

    golap::HostMem ente_hist_host{golap::Tag<decltype(COLS::ente)>{},12};
    golap::HostMem gans_hist_host{golap::Tag<decltype(COLS::gans)>{},12};

    golap::MirrorMem ente_mm_check{golap::Tag<uint16_t>{}, ente_comp.blocks.size()};
    golap::MirrorMem gans_mm_check{golap::Tag<uint16_t>{}, gans_comp.blocks.size()};
    golap::MirrorMem combined_mm{golap::Tag<uint16_t>{}, gans_comp.blocks.size()};
    ente_mm_check.dev.set(0);
    gans_mm_check.dev.set(0);

    golap::MirrorMem ente_hist_check{golap::Tag<uint16_t>{}, ente_comp.blocks.size()};
    golap::MirrorMem gans_hist_check{golap::Tag<uint16_t>{}, gans_comp.blocks.size()};
    golap::MirrorMem combined_hist{golap::Tag<uint16_t>{}, gans_comp.blocks.size()};
    ente_hist_check.dev.set(0);
    gans_hist_check.dev.set(0);

    golap::MirrorMem ente_bloom_check{golap::Tag<uint16_t>{}, ente_comp.blocks.size()};
    golap::MirrorMem gans_bloom_check{golap::Tag<uint16_t>{}, gans_comp.blocks.size()};
    golap::MirrorMem combined_bloom{golap::Tag<uint16_t>{}, gans_comp.blocks.size()};
    ente_bloom_check.dev.set(0);
    gans_bloom_check.dev.set(0);

    golap::check_mmmeta<<<20,512>>>(ente_minmax, ente_mm_check.dev.ptr<uint16_t>(), (decltype(COLS::ente)) 50, (decltype(COLS::ente)) 400000);
    golap::check_mmmeta<<<20,512>>>(gans_minmax, gans_mm_check.dev.ptr<uint16_t>(), 500000.0, 550000.0);
    golap::combine_and<<<20,512>>>(combined_mm.dev.ptr<uint16_t>(), ente_mm_check.dev.ptr<uint16_t>(), gans_mm_check.dev.ptr<uint16_t>(), ente_comp.blocks.size());
    checkCudaErrors(cudaDeviceSynchronize());
    ente_mm_check.sync_to_host();
    gans_mm_check.sync_to_host();
    combined_mm.sync_to_host();

    golap::check_hist<<<20,512>>>(ente_hist,ente_hist_check.dev.ptr<uint16_t>(), (decltype(COLS::ente)) 50, (decltype(COLS::ente)) 400000);
    golap::check_hist<<<20,512>>>(gans_hist,gans_hist_check.dev.ptr<uint16_t>(), 500000.0, 550000.0);
    golap::combine_and<<<20,512>>>(combined_hist.dev.ptr<uint16_t>(), ente_hist_check.dev.ptr<uint16_t>(), gans_hist_check.dev.ptr<uint16_t>(), ente_comp.blocks.size());
    checkCudaErrors(cudaDeviceSynchronize());
    ente_hist_check.sync_to_host();
    gans_hist_check.sync_to_host();
    combined_hist.sync_to_host();

    golap::check_bloom<<<20,512>>>(ente_bloom,ente_bloom_check.dev.ptr<uint16_t>(), (decltype(COLS::ente)) 60);
    golap::check_bloom<<<20,512>>>(gans_bloom,gans_bloom_check.dev.ptr<uint16_t>(), 500000.0);
    golap::combine_and<<<20,512>>>(combined_bloom.dev.ptr<uint16_t>(), ente_bloom_check.dev.ptr<uint16_t>(), gans_bloom_check.dev.ptr<uint16_t>(), ente_comp.blocks.size());
    checkCudaErrors(cudaDeviceSynchronize());
    ente_bloom_check.sync_to_host();
    gans_bloom_check.sync_to_host();
    combined_bloom.sync_to_host();


    while(round< table_loader.blockenvs.at("ente").myblocks.size()){
        // check predicate:
        if(!table_loader.rootop.step(table_loader.rootstream.stream,table_loader.rootevent.event)){
            printf("This shouldnt happen!\n");
        }

        checkCudaErrors(cudaMemcpy((void*)&ente_hist_host.ptr<decltype(COLS::ente)>()[0], (void*) &ente_hist.mins[round],
                                    sizeof(decltype(COLS::ente)), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy((void*)&ente_hist_host.ptr<decltype(COLS::ente)>()[1], (void*) &ente_hist.maxs[round],
                                    sizeof(decltype(COLS::ente)), cudaMemcpyDefault));

        checkCudaErrors(cudaMemcpy((void*)&gans_hist_host.ptr<decltype(COLS::gans)>()[0], (void*) &gans_hist.mins[round],
                                    sizeof(decltype(COLS::gans)), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy((void*)&gans_hist_host.ptr<decltype(COLS::gans)>()[1], (void*) &gans_hist.maxs[round],
                                    sizeof(decltype(COLS::gans)), cudaMemcpyDefault));

        tuples_this_round = table_loader.blockenvs.at("ente").myblocks[round].tuples;

        checkCudaErrors(cudaStreamSynchronize(table_loader.rootstream.stream));

        printf("tuples_this_round %lu\n",tuples_this_round);
        printf("Ente Chunk [%u,%u], mm_qualifies=%hu, hist_qualifies=%hu, bloom_qualifies=%hu\n",ente_hist_host.ptr<decltype(COLS::ente)>()[0],
                        ente_hist_host.ptr<decltype(COLS::ente)>()[1], ente_mm_check.hst.ptr<uint16_t>()[round], ente_hist_check.hst.ptr<uint16_t>()[round], ente_bloom_check.hst.ptr<uint16_t>()[round]);
        printf("Gans Chunk [%.2f,%.2f], mm_qualifies=%hu, hist_qualifies=%hu, bloom_qualifies=%hu\n",gans_hist_host.ptr<decltype(COLS::gans)>()[0], gans_hist_host.ptr<decltype(COLS::gans)>()[1],
                                            gans_mm_check.hst.ptr<uint16_t>()[round], gans_hist_check.hst.ptr<uint16_t>()[round],
                                            gans_bloom_check.hst.ptr<uint16_t>()[round]);
        printf("Combined: MM=%hu, HIST=%hu, BLOOM=%hu\n", combined_mm.hst.ptr<uint16_t>()[round], combined_hist.hst.ptr<uint16_t>()[round],
                                combined_bloom.hst.ptr<uint16_t>()[round]);


        round += 1;
    }



    REQUIRE(true);

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}

