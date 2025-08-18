#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <fstream>
#include "helper_cuda.h"

#include "test_common.hpp"
#include "table.hpp"
#include "mem.hpp"
#include "types.hpp"


struct AType{
    int id;
    char name[11];
    float discount;
};


TEST_CASE("Column", "[table]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    // golap::Column<golap::HostMem,int> a_col(512);

    // REQUIRE(a_col.size_bytes() == sizeof(int)*512);
    // REQUIRE(a_col.size() == 512);
    struct COLS{
        enum {ENTE=0,GANS=1,KARTOFFEL=2};
    };

    uint64_t alloc_unit = (1<<15);

    golap::ColumnTable<golap::HostMem,int,float,util::Padded<char[11]>> table{"ente,gans,kartoffel", 0, 1};
    std::cout << table.col<COLS::ENTE>() << "\n";
    std::cout << table.col<COLS::GANS>() << "\n";
    std::cout << table.col<COLS::KARTOFFEL>() << "\n";

    table.resize(1024,alloc_unit);
    golap::Column<golap::HostMem,int> &entecol = table.col<COLS::ENTE>();
    // entecol.resize_col(512);
    std::cout << table.col<COLS::ENTE>() << "\n";
    std::cout << table.col<COLS::GANS>() << "\n";
    std::cout << table.col<COLS::KARTOFFEL>() << "\n";

    golap::ColumnTable<golap::DeviceMem,decltype(table.col<1>().data())> dev_table{"devente", 512, 1};
    std::cout << dev_table.col<0>() << "\n";

    auto int_ptr = table.col<0>().data();

    int_ptr[15] = 16234;
    snprintf(table.col<COLS::KARTOFFEL>().data()[15].d, 11, "%s", "potato");
    int val = table.col<0>().data()[15];
    table.col<0>().data()[16] = 16235;
    // std::cout << int_ptr[15] <<", "<<val<<", "<< int_ptr[16] <<  "\n";

    REQUIRE(table.size_bytes() == 3*alloc_unit);
    REQUIRE(table.num_slots == 1024);
    REQUIRE(table.col<COLS::ENTE>().size_bytes() == alloc_unit);
    REQUIRE(table.col<COLS::GANS>().size_bytes() == alloc_unit);
    REQUIRE(table.col<COLS::KARTOFFEL>().size_bytes() == alloc_unit);
    REQUIRE(true);
    table.num_tuples = 16;
    table.to_csv(std::cout,";");

    } // stack for cuda-memcheck
}


TEST_CASE("Column sort", "[table]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    struct COLS{
        enum {ENTE=0,GANS=1,KARTOFFEL=2};
    };

    uint64_t alloc_unit = (1<<15);

    golap::ColumnTable<golap::HostMem,int,float,util::Padded<char[11]>> table{"ente,gans,kartoffel", 1000, alloc_unit};
    table.num_tuples = 20;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
    }
    // table.to_csv(std::cout,";");

    table.sort_by<COLS::GANS>();

    // table.to_csv(std::cout,";");
    float last = 0.0;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        REQUIRE(table.col<COLS::GANS>().data()[i] >= last);
        last = table.col<COLS::GANS>().data()[i];
    }

    } // stack for cuda-memcheck
}


TEST_CASE("Column apply", "[table]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    struct COLS{
        enum {ENTE=0,GANS=1,KARTOFFEL=2};
    };


    golap::ColumnTable<golap::HostMem,int,float,util::Padded<char[11]>> table{"ente,gans,kartoffel", 1000};
    table.num_tuples = 20;
    double result = 0.0;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
        result += table.col<COLS::ENTE>().data()[i] + table.col<COLS::GANS>().data()[i];
    }

    double counter = 0.0;
    table.apply([&counter](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
        std::cout << "Col_idx: "<<col_idx << ", col_name: " <<a_col.attr_name << "\n";
        if constexpr (std::is_same_v<typename std::remove_reference<decltype(a_col)>::type::value_t, util::Padded<char[11]>>) return;
        else{
            for (uint64_t i =0; i<num_tuples;++i){
                counter += a_col.data()[i];
            }
        }
    });

    REQUIRE(counter == Catch::Approx(result));


    } // stack for cuda-memcheck
}

TEST_CASE("Column attr names", "[table]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    struct COLS{
        enum {ENTE=0,GANS=1,KARTOFFEL=2};
    };

    golap::ColumnTable<golap::HostMem,int,float,util::Padded<char[11]>> table{"ente,gans,kartoffel", 1000};
    table.num_tuples = 20;

    REQUIRE(table.col<COLS::ENTE>().attr_name == "ente");
    REQUIRE(table.col<COLS::GANS>().attr_name == "gans");
    REQUIRE(table.col<COLS::KARTOFFEL>().attr_name == "kartoffel");


    } // stack for cuda-memcheck
}

TEST_CASE("Column sort multi", "[table]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    struct COLS{
        enum {ENTE=0,GANS=1,KARTOFFEL=2};
    };

    uint64_t alloc_unit = (1<<15);

    golap::ColumnTable<golap::HostMem,int,float,util::Padded<char[11]>> table{"ente,gans,kartoffel", 1000, alloc_unit};
    table.num_tuples = 20;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i%3;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
        snprintf(table.col<COLS::KARTOFFEL>().data()[i].d, 11, "K %lu", i%3);
    }
    table.to_csv(std::cout,";");

    table.sort_by<COLS::KARTOFFEL,COLS::GANS>();

    table.to_csv(std::cout,";");
    float last = 0.0;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        // REQUIRE(table.col<COLS::GANS>().data()[i] >= last);
        // last = table.col<COLS::GANS>().data()[i];
    }

    } // stack for cuda-memcheck
}

TEST_CASE("To / From CSV", "[table]") {
    checkCudaErrors(cudaSetDevice(TEST_GPU_ID));
    { // stack for cuda-memcheck

    struct COLS{
        enum {ENTE=0,GANS=1,KARTOFFEL=2};
    };

    uint64_t alloc_unit = (1<<15);

    golap::ColumnTable<golap::HostMem,int,float,util::Padded<char[11]>> table{"ente,gans,kartoffel", 1000, alloc_unit};
    golap::ColumnTable<golap::HostMem,int,float,util::Padded<char[11]>> table_cpy{"ente,gans,kartoffel", 1000, alloc_unit};
    table.num_tuples = 50;
    for(uint64_t i = 0; i < table.num_tuples; ++i){
        table.col<COLS::ENTE>().data()[i] = i%3;
        table.col<COLS::GANS>().data()[i] = (table.num_tuples - i)*1.1;
        snprintf(table.col<COLS::KARTOFFEL>().data()[i].d, 11, "K %lu", i%3);
    }

    // table.to_csv(std::cout,";");
    std::ofstream out_stream("test_csv_write_read.csv", std::ofstream::out);
    table.to_csv(out_stream,";");
    out_stream.close();

    std::ifstream in_stream("test_csv_write_read.csv", std::ifstream::in);
    table_cpy.from_csv(in_stream,";");
    in_stream.close();
    // table_cpy.to_csv(std::cout,";");

    // clean up the file
    std::remove("test_csv_write_read.csv");

    for(uint64_t i = 0; i < table.num_tuples; ++i){
        REQUIRE(table.col<COLS::ENTE>().data()[i] == table_cpy.col<COLS::ENTE>().data()[i]);
        REQUIRE(table.col<COLS::GANS>().data()[i] == table_cpy.col<COLS::GANS>().data()[i]);
        REQUIRE(table.col<COLS::KARTOFFEL>().data()[i] == table_cpy.col<COLS::KARTOFFEL>().data()[i]);
    }

    } // stack for cuda-memcheck
}