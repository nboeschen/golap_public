#include <iostream>
#include <helper_cuda.h>
#include <gflags/gflags.h>

#include "hl/serialization.hpp"
#include "taxi.hpp"

DEFINE_uint32(scale_factor, 10, "Scale factor.");
DEFINE_uint32(all_peek, 0, "Number of lines of all tables to peek at.");
DEFINE_string(path, "../bin/disk.dat", "Path to output file.");
DEFINE_string(op, "read", "Either read,write,bin2csv,csv2bin");
DEFINE_string(format, "binary", "binary or csv");
DEFINE_string(csv_delimiter, ";", "CSV delimiter");


int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // golap::StorageManager sm{FLAGS_path};
    TaxiVar var{(uint32_t)-1, (uint32_t)-1, "", "", FLAGS_scale_factor};

    TaxiColLayout db(var,(FLAGS_op == "write" ? "init_populate" : "init_only"));

    if (FLAGS_op == "csv2bin"){
        golap::read_col_db_csv(db.tables,FLAGS_path,FLAGS_csv_delimiter);
        util::Log::get().info_fmt("Read csv table data.");
        FLAGS_path += ".dat";
        golap::write_col_db_bin(db.tables,FLAGS_path);
    }else if (FLAGS_op == "bin2csv"){
        golap::read_col_db_bin(db.tables,FLAGS_path);
        util::Log::get().info_fmt("Read dat table data.");
        FLAGS_path += ".csv";
        golap::write_col_db_csv(db.tables,FLAGS_path,FLAGS_csv_delimiter);
    }else if(FLAGS_format == "binary"){
        if(FLAGS_op == "read"){
            golap::read_col_db_bin(db.tables,FLAGS_path);
        }else if(FLAGS_op == "write"){
            golap::write_col_db_bin(db.tables,FLAGS_path);
        }
    }else if (FLAGS_format == "csv"){
        if(FLAGS_op == "read"){
            golap::read_col_db_csv(db.tables,FLAGS_path,FLAGS_csv_delimiter);
        }else if(FLAGS_op == "write"){
            golap::write_col_db_csv(db.tables,FLAGS_path,FLAGS_csv_delimiter);
        }
    }


    if(FLAGS_all_peek != 0){
        db.tables.apply([&](auto& table){
            table.to_csv(std::cout, FLAGS_csv_delimiter, 0, FLAGS_all_peek);
        });
    }

    return 0;
}

