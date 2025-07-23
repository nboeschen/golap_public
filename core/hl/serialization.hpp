#pragma once

#include <iostream>
#include <fstream>

#include "mem.hpp"
#include "util.hpp"

#if __GNUC_PREREQ(8,0)
// gcc 8+
#include <filesystem>
using namespace std::filesystem;
#else
// gcc < 8
#include <experimental/filesystem>
using namespace std::experimental::filesystem;
#endif //__GNUC_PREREQ(8,0)

namespace golap {

template <typename DBTYPE>
void write_col_db_csv(DBTYPE &tables, std::string fpath, std::string delimiter, std::string extension = ".csv"){
    path parent_path = path(fpath).parent_path();

    tables.apply([&](auto& table){
        std::string table_path = std::string(parent_path/table.name) + extension;

        std::cout << table.name << " " << table.num_tuples << " tuples\n";
        std::cout << table_path << "\n";
        std::ofstream stream(table_path, std::ofstream::out);
        if (!stream.good()){
            std::cout << "Couldnt open \"" << table_path << "\", exiting\n";
            std::exit(1);
        }
        table.to_csv(stream,delimiter);
        stream.close();
    });

}

template <typename DBTYPE>
void read_col_db_csv(DBTYPE &tables, std::string fpath, std::string delimiter, uint64_t start_idx=1,
                    std::string extension = ".csv"){
    path parent_path = path(fpath).parent_path();

    tables.apply([&](auto& table){
        std::string table_path = std::string(parent_path/table.name) + extension;

        std::ifstream stream(table_path, std::ifstream::in);
        if (!stream.good()){
            std::cout << "Couldnt open \"" << table_path << "\", exiting\n";
            std::exit(1);
        }
        table.from_csv(stream,delimiter,start_idx);
        stream.close();
    });

}

template <typename DBTYPE>
void write_col_db_bin(DBTYPE &tables, std::string fpath){

    std::ofstream outfile;
    outfile.open(fpath, std::ofstream::binary);
    outfile.seekp(4096);

    std::vector<std::pair<uint64_t,uint64_t>> table_tuples_n_bytes;

    tables.apply([&](auto& table){
        table_tuples_n_bytes.emplace_back(table.num_tuples, table.to_bin_col(outfile));
    });


    uint64_t end = outfile.tellp();
    outfile.seekp(0);
    for(auto [tuples,bytes] : table_tuples_n_bytes){
        outfile.write((char*)&tuples, sizeof(uint64_t));
        outfile.write((char*)&bytes, sizeof(uint64_t));
    }

    outfile.close();

    std::cout << "Wrote " << end << " bytes to \""<< fpath <<"\"\n";

}

template <typename DBTYPE>
void read_col_db_bin(DBTYPE &tables, std::string fpath){

    std::ifstream infile;
    infile.open(fpath, std::ifstream::binary);
    if(!infile.good()){
        std::cout << "Can't read file: " << fpath << "\n";
        return;
    }

    std::vector<std::pair<uint64_t,uint64_t>> table_tuples_n_bytes;

    uint64_t tuples,bytes,read,i=0;
    tables.apply([&](auto& table){
        infile.read((char*)&tuples, sizeof(uint64_t));
        infile.read((char*)&bytes, sizeof(uint64_t));
        table_tuples_n_bytes.emplace_back(tuples,bytes);
    });

    infile.seekg(4096);

    tables.apply([&](auto& table){
        std::tie(tuples,bytes) = table_tuples_n_bytes[i++];
        if(table.num_slots < tuples){
            std::cout << table.name << ": Slots " << table.num_slots << " not sufficient for " << tuples << " tuples\n";
            infile.close();
            std::exit(1);
        }
        table.num_tuples = tuples;
        read = table.from_bin_col(infile);
        if (bytes != read){
            std::cout << table.name << ": Expected " << bytes << " to be read, but actually read " << read << " bytes\n";
            infile.close();
            std::exit(1);
        }
    });

    util::Log::get().debug_fmt("Read %lu bytes from %s",infile.tellg(),fpath.c_str());
    infile.close();

}

} // end of namespace golap
