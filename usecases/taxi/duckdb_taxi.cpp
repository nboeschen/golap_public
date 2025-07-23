#include <iostream>
#include <unistd.h>
#include <cstdint>
#include <map>
#include "gflags/gflags.h"
#include "duckdb.hpp"

#include "util.hpp"
#include "queries/queries_str.hpp"

DEFINE_uint32(repeat, 3, "Experiment repetition num.");
DEFINE_string(dbpath, "./", "Path to directory containing the parquet files, load.sql and schema.sql.");
DEFINE_string(query, "query1.1", "The queries to execute.");
DEFINE_bool(print_res, false, "Print the actual output of duckdb.");
DEFINE_bool(drop_cache, true, "Drop pages of the parquet files from page cache before each query.");
DEFINE_uint64(mem_limit, ((uint64_t)1<<36), "Maximum memory size the duck is allowed to use.");
DEFINE_uint64(thread_limit, 32, "Maximum threads the duck is allowed to use.");
DEFINE_string(print, "csv", "Print output. Options are [csv_header,csv,duckdb_settings]");

struct DuckDBVar{
    //in
    std::string query;
    uint32_t scale_factor;
    bool drop_cache;

    //out
    std::string comp_algo;
    uint64_t mem_limit;
    uint64_t thread_limit;
    uint64_t uncomp_bytes; // could get this out of the parquet now, but not so important
    uint64_t comp_bytes;
    uint64_t io_bytes; // the bytes actually caused to be transferred from storage

    double time_ms;

    std::string to_pretty(){
        double comp_bw = (1000.0 / (1<<30)) * ((double)comp_bytes/time_ms);
        double io_bw = (1000.0 / (1<<30)) * ((double)io_bytes/time_ms);

        std::stringstream ss;
        ss << "DuckDB(query="<<query<<", comp_algo="<<comp_algo<<", comp_bytes="<< comp_bytes<<", comp_bw="<<comp_bw<<"GB/s, io_bytes="<<io_bytes<<", io_bw="<<io_bw<<"GB/s, time_ms="<<time_ms<<"ms)";
        return ss.str();
    }

    std::string to_csv(){
        std::stringstream ss;
        ss << query << "," <<scale_factor<<"," <<drop_cache<<","<<comp_algo << "," <<mem_limit << "," << thread_limit <<","<< uncomp_bytes << "," << comp_bytes <<","<<io_bytes<< "," << time_ms;
        return ss.str();
    }

    std::string repr(bool pretty){
        return pretty ? to_pretty() : to_csv();
    }

    static std::string csv_header(){
        return "query,scale_factor,drop_cache,comp_algo,mem_limit,thread_limit,uncomp_bytes,comp_bytes,io_bytes,time_ms";
    }

};


int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_print == "csv_header"){
        std::cout << DuckDBVar::csv_header() << '\n';
        return 0;
    }

    duckdb::DBConfig conf;
    conf.options.object_cache_enable = true;
    conf.options.maximum_memory = FLAGS_mem_limit;
    conf.options.maximum_threads = FLAGS_thread_limit;
    if(FLAGS_drop_cache) conf.options.use_direct_io = true;


    // use a memory db, since we'll include the parquets as views
    duckdb::DuckDB db(nullptr, &conf);
    duckdb::Connection con(db);

    con.Query(std::string("IMPORT DATABASE '")+FLAGS_dbpath+std::string("';"));
    // con.Query("PRAGMA profiling_mode=detailed;");

    if(FLAGS_print=="duckdb_settings"){
        con.Query("select * from duckdb_settings();")->Print();
        return 0;
    }

    auto queries = util::str_split(FLAGS_query,",");
    DuckDBVar var;
    var.drop_cache = FLAGS_drop_cache;
    var.scale_factor = std::stoul(util::str_split(util::str_split(FLAGS_dbpath,"parquet")[1], "/")[0]);
    var.mem_limit = FLAGS_mem_limit;
    var.thread_limit = FLAGS_thread_limit;


    for (uint32_t i = 0;i<FLAGS_repeat; ++i){
        for (auto &query: queries){
            // util::Log::get().info_fmt("Starting query execution: %s",query.c_str());
            var.query = query;
            var.uncomp_bytes = 0;
            var.comp_bytes = 0;

            if (var.drop_cache) util::drop_file_cache(FLAGS_dbpath);

            std::unique_ptr<duckdb::MaterializedQueryResult> parquet_meta = con.Query("SELECT path_in_schema,compression,SUM(total_uncompressed_size),SUM(total_compressed_size) FROM parquet_metadata('"+FLAGS_dbpath+"trips.parquet') GROUP BY path_in_schema,compression;");
            var.comp_algo = parquet_meta->GetValue(1,0).ToString();


            auto query_txt = get_query_text(query);
            util::Log::get().debug_fmt("Trying to run query: \"%s\"",query_txt.c_str());
            auto prepared = con.Prepare(query_txt);

            uint64_t rchar_before = util::get_proc_info("io","rchar");
            uint64_t io_bytes_before = util::get_proc_info("io","read_bytes");

            util::Timer timer;
            auto result = prepared->Execute();

            if (result->HasError()) {
                util::Log::get().error_fmt("Quack: %s", result->GetError());
                continue;
            }

            auto materialized = dynamic_cast<duckdb::StreamQueryResult*>(result.get())->Materialize();

            var.time_ms = timer.elapsed();

            util::Log::get().info_fmt("Res size %lu", materialized->RowCount());
            if (FLAGS_print_res){
                materialized->Print();
            }

            var.comp_bytes = util::get_proc_info("io","rchar")-rchar_before;
            var.io_bytes = util::get_proc_info("io","read_bytes")-io_bytes_before;

            std::cout << var.repr(FLAGS_print == "pretty") << "\n";
        }
        // std::cout << getpid() << "\n";
        // std::cin.ignore();
    }

    return 0;
}