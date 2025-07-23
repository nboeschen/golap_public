#include <iostream>
#include <helper_cuda.h>
#include <gflags/gflags.h>

#include "hl/serialization.hpp"
#include "ssb.hpp"
#include "helper.hpp"

DEFINE_uint32(scale_factor, 10, "SSB scale factor.");
DEFINE_uint32(all_peek, 0, "Number of lines of all tables to peek at.");
DEFINE_uint32(lineorder_peek, 0, "Number of Lineorders to peek at.");
DEFINE_uint32(customer_peek, 0, "Number of Customers to peek at.");
DEFINE_uint32(supplier_peek, 0, "Number of suppliers to peek at.");
DEFINE_uint32(part_peek, 0, "Number of parts to peek at.");
DEFINE_uint32(date_peek, 0, "Number of dates to peek at.");
DEFINE_string(path, "../bin/ssb_disk.dat", "Path to output file.");
DEFINE_string(pre_aggr_path, "", "Path to pre_aggr disk db.");
DEFINE_string(sort_path, "", "Path to sorted disk db.");
DEFINE_string(op, "read", "Either read,write,bin2csv,csv2bin");
DEFINE_string(format, "binary", "binary or csv");
DEFINE_string(csv_delimiter, ";", "CSV delimiter");


int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // golap::StorageManager sm{FLAGS_path};
    SSBVar var{(uint32_t)-1, (uint32_t)-1, "", "", FLAGS_scale_factor};
    var.customer_factor = 1;

    SSBColLayout ssb(var,(FLAGS_op == "write" ? "init_populate" : "init_only"));

    if (FLAGS_op == "csv2bin"){
        golap::read_col_db_csv(ssb.tables,FLAGS_path,FLAGS_csv_delimiter);
        FLAGS_path += ".dat";
        golap::write_col_db_bin(ssb.tables,FLAGS_path);
    }else if (FLAGS_op == "bin2csv"){
        golap::read_col_db_bin(ssb.tables,FLAGS_path);
        FLAGS_path += ".csv";
        golap::write_col_db_csv(ssb.tables,FLAGS_path,FLAGS_csv_delimiter);
    }else if (FLAGS_op == "pre_aggr" && FLAGS_pre_aggr_path != "") {

        if (FLAGS_format == "binary"){
            golap::read_col_db_bin(ssb.tables,FLAGS_path);
        }else if (FLAGS_format == "csv") {
            golap::read_col_db_csv(ssb.tables,FLAGS_path,FLAGS_csv_delimiter);
        }

        util::Log::get().info_fmt("Preaggr tables ...");
        std::vector<uint64_t> dummy;
        SortHelper::get().preaggr_tables(ssb.tables, dummy);
        if (FLAGS_format == "binary"){
            golap::write_col_db_bin(*SortHelper::get().preaggrdb,FLAGS_pre_aggr_path);
        }else if (FLAGS_format == "csv") {
            golap::write_col_db_csv(*SortHelper::get().preaggrdb,FLAGS_pre_aggr_path,FLAGS_csv_delimiter);
        }
    }else if (FLAGS_op == "sort" && FLAGS_sort_path != "") {
        if (FLAGS_format == "binary"){
            golap::read_col_db_bin(ssb.tables,FLAGS_path);
        }else if (FLAGS_format == "csv") {
            golap::read_col_db_csv(ssb.tables,FLAGS_path,FLAGS_csv_delimiter);
        }

        util::Log::get().info_fmt("Prejoin and sorting tables ...");
        std::vector<uint64_t> dummy;
        SortHelper::get().prejoin_tables(ssb.tables, dummy);
        SortHelper::get().apply("general_dimsort", ssb.tables, dummy);
        util::Log::get().info_fmt("Writing Lineorders to : %s ...",FLAGS_sort_path.c_str());
        // golap::write_col_db_csv(*SortHelper::get().prejo,FLAGS_pre_aggr_path,FLAGS_csv_delimiter);

        std::ofstream stream(FLAGS_sort_path, std::ofstream::out);
        if (!stream.good()){
            std::cout << "Couldnt open \"" << FLAGS_sort_path << "\", exiting\n";
            std::exit(1);
        }
        ssb.tables.lineorder.to_csv(stream,FLAGS_csv_delimiter);
        stream.close();
    }else if(FLAGS_format == "binary"){
        if(FLAGS_op == "read"){
            golap::read_col_db_bin(ssb.tables,FLAGS_path);
        }else if(FLAGS_op == "write"){
            golap::write_col_db_bin(ssb.tables,FLAGS_path);
        }
    }else if (FLAGS_format == "csv"){
        if(FLAGS_op == "read"){
            golap::read_col_db_csv(ssb.tables,FLAGS_path,FLAGS_csv_delimiter);
        }else if(FLAGS_op == "write"){
            golap::write_col_db_csv(ssb.tables,FLAGS_path,FLAGS_csv_delimiter);
        }
    }else {
        util::Log::get().error_fmt("Dont know what to do, check arguments!");
    }

    if (FLAGS_lineorder_peek != 0 || FLAGS_all_peek != 0) ssb.tables.lineorder.to_csv(std::cout, FLAGS_csv_delimiter,
                                                                                      0, FLAGS_lineorder_peek!=0?FLAGS_lineorder_peek:FLAGS_all_peek);
    if (FLAGS_customer_peek != 0 || FLAGS_all_peek != 0) ssb.tables.customer.to_csv(std::cout, FLAGS_csv_delimiter,
                                                                                    0, FLAGS_customer_peek!=0?FLAGS_customer_peek:FLAGS_all_peek);
    if (FLAGS_supplier_peek != 0 || FLAGS_all_peek != 0) ssb.tables.supplier.to_csv(std::cout, FLAGS_csv_delimiter,
                                                                                    0, FLAGS_supplier_peek!=0?FLAGS_supplier_peek:FLAGS_all_peek);
    if (FLAGS_part_peek != 0 || FLAGS_all_peek != 0) ssb.tables.part.to_csv(std::cout, FLAGS_csv_delimiter,
                                                                            0, FLAGS_part_peek!=0?FLAGS_part_peek:FLAGS_all_peek);
    if (FLAGS_date_peek != 0 || FLAGS_all_peek != 0) ssb.tables.date.to_csv(std::cout, FLAGS_csv_delimiter,
                                                                            0, FLAGS_date_peek!=0?FLAGS_date_peek:FLAGS_all_peek);


    return 0;
}

