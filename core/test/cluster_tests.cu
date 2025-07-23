#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstring>
#include <fstream>

#include "helper_cuda.h"
#include "test_common.hpp"

#include "storage.hpp"
#include "util.hpp"
#include "comp.cuh"
#include "hl/cluster.cuh"

TEST_CASE("KMeans clustering", "[cluster]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    uint64_t store_offset = 0;
    uint64_t ROW_NUM = (1<<11);
    uint64_t CHUNK_BYTES = (1<<11);
    uint64_t k = 50;

    auto& sm = golap::StorageManager::get();
    sm.init(STORE_PATH,0);


    struct COLS{
        enum {ENTE=0,GANS=1,HAHN=2,HUHN=3};
        uint32_t ente;
        double gans;
        uint8_t hahn;
        char huhn[16]; 
    };

    uint64_t seed = 0xBADEE47E;


    golap::ColumnTable<golap::HostMem,decltype(COLS::ente),decltype(COLS::gans),decltype(COLS::hahn),decltype(COLS::huhn)> table{"ente,gans,hahn,huhn", ROW_NUM};
    table.num_tuples = ROW_NUM;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = util::uniform_int(seed, 0, 50000);
        table.col<COLS::GANS>().data()[i] = util::uniform_int(seed, 50000, 100000)*1.1;
        table.col<COLS::HAHN>().data()[i] = (uint8_t) util::uniform_int(seed, 0,50);
        snprintf(table.col<COLS::HUHN>().data()[i], sizeof(decltype(COLS::huhn)), "h%lu", i);

    }
    uint64_t tuples_per_chunk = CHUNK_BYTES / std::max({sizeof(decltype(COLS::ente)),sizeof(decltype(COLS::gans)),sizeof(decltype(COLS::huhn))});

    // do some clustering
    // kmeans of a number of columns
    

    golap::HostMem assignments_hst{golap::Tag<uint64_t>{}, table.num_tuples};
    golap::HostMem assigned_each_cluster_hst{golap::Tag<uint64_t>{}, k};


    golap::KMeansClustering kmeans(table.num_tuples, 1, k, true);
    kmeans.add_column_normalized(table.col<COLS::ENTE>().data());
    // kmeans.add_column_normalized(table.col<COLS::GANS>().data());
    // kmeans.add_column_normalized(table.col<COLS::HAHN>().data());
    kmeans.cluster(3, assignments_hst.ptr<uint64_t>(), assigned_each_cluster_hst.ptr<uint64_t>());
    // printf("order after:\n");
    // for(uint64_t i= 0; i< table.num_tuples; ++i){
    //     printf("%04lu, ", kmeans.sort_order_hst[i]);
    //     if(i % 16 == 15) printf("\n");
    // }


    std::vector<uint64_t> sort_order(kmeans.sort_order_hst,kmeans.sort_order_hst+table.num_tuples);
    std::vector<uint64_t> assigned_tuples(assigned_each_cluster_hst.ptr<uint64_t>(),assigned_each_cluster_hst.ptr<uint64_t>()+k);

    golap::HostMem centroids{golap::Tag<float>{},k};
    checkCudaErrors(cudaMemcpy(centroids.data, kmeans.centroids, k*sizeof(float), cudaMemcpyDefault));
    for (uint32_t i= 0; i<k;++i){

        printf("[%u] mean: %.3f, tuples: %lu\t",i,centroids.ptr<float>()[i] * 49,assigned_tuples[i]);
        if (i%5 == 4) printf("\n");
    }

    // write to file
    std::ofstream out_stream("clusters.csv", std::ofstream::out);
    out_stream << "ente;gans;hahn;cluster\n";
    for (uint32_t i= 0; i<table.num_tuples;++i){
        out_stream << table.col<COLS::ENTE>().data()[i] << ";" << table.col<COLS::GANS>().data()[i] << ";" << +table.col<COLS::HAHN>().data()[i] << ";" << assignments_hst.ptr<uint64_t>()[i] << "\n";
        // printf("tuple[%u] in cluster %lu\n", i, assignments_hst.ptr<uint64_t>()[i]);
    }
    out_stream.close();

    table.sort(sort_order);

    REQUIRE(std::accumulate(assigned_tuples.begin(), assigned_tuples.end(), (uint64_t)0) == table.num_tuples);

    golap::CompInfo ente_comp{tuples_per_chunk*sizeof(decltype(COLS::ente)), ROW_NUM*sizeof(decltype(COLS::ente)), "Gdeflate"};
    golap::CompInfo gans_comp{tuples_per_chunk*sizeof(decltype(COLS::gans)), ROW_NUM*sizeof(decltype(COLS::gans)), "Gdeflate"};
    golap::CompInfo huhn_comp{tuples_per_chunk*sizeof(decltype(COLS::huhn)), ROW_NUM*sizeof(decltype(COLS::huhn)), "Gdeflate"};

    // golap::prepare_uncompressed<golap::HostMem, uint32_t>(sm, store_offset, table.col<COLS::ENTE>(), ROW_NUM, ente_comp, &ente_minmax, &ente_hist);
    // golap::prepare_uncompressed<golap::HostMem, double>(sm, ente_comp.end_offset(), table.col<COLS::GANS>(), ROW_NUM, gans_comp, &gans_minmax, &gans_hist);
    // golap::prepare_uncompressed(sm, gans_comp.end_offset(), table.col<COLS::HUHN>(), ROW_NUM, huhn_comp);

    // golap::prepare_compressed_device<golap::HostMem, decltype(COLS::ente)>(sm, store_offset, table.col<COLS::ENTE>(), ROW_NUM, ente_comp, &ente_minmax, &ente_hist);
    // golap::prepare_compressed_device<golap::HostMem, decltype(COLS::gans), uint8_t>(sm, ente_comp.end_offset(), table.col<COLS::GANS>(), ROW_NUM, gans_comp, &gans_minmax, &gans_hist);
    // golap::prepare_compressed_device(sm, gans_comp.end_offset(), table.col<COLS::HUHN>(), ROW_NUM, huhn_comp);

    // // golap::TableLoader<golap::LoadEnv> table_loader{3};
    // golap::TableLoader<golap::DecompressEnv> table_loader{3};
    // table_loader.add(sm, "ente", 0, ente_comp.blocks.size(), ente_comp, nvcomp::TypeOf<decltype(COLS::ente)>());
    // table_loader.add(sm, "gans", 0, gans_comp.blocks.size(), gans_comp, nvcomp::TypeOf<uint8_t>());
    // table_loader.add(sm, "huhn", 0, huhn_comp.blocks.size(), huhn_comp, nvcomp::TypeOf<uint8_t>());

    REQUIRE(true);

    } // stack for cuda-memcheck
    checkCudaErrors(cudaDeviceReset());
}



