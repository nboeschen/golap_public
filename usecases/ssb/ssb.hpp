#pragma once

#include <cstdint>
#include <string>
#include <thread>

#include "mem.hpp"
#include "table.hpp"
#include "storage.hpp"
#include "util.hpp"
#include "comp.cuh"
#include "comp_cpu.hpp"
#include "access.hpp"
#include "join.cuh"
#include "apps.cuh"

#include "core.hpp"
#include "data/SSB_def.hpp"


struct SSBVar : golap::Parameter{
    uint32_t customer_factor;
    uint32_t extra_workers;
    uint32_t add_cpu_threads;
    double bloomepsilon;    // no print
    bool join_prune;// no print
    bool extended_price_pred;// no print

    std::string to_csv(){
        std::stringstream ss;
        ss << golap::Parameter::to_csv();
        ss << ','<<customer_factor<<','<<extra_workers<<','<<add_cpu_threads;
        return ss.str();
    }

    std::string repr(bool pretty){
        return pretty ? to_pretty() : to_csv();
    }

    static std::string csv_header(){return golap::Parameter::csv_header() + ",customer_factor,extra_workers,add_cpu_threads";};
};

static const std::unordered_map<std::string,std::string> BEST_BW_COMP = {
    {"lo_commitdate", "Bitcomp"},
    {"lo_custkey", "Cascaded"},
    {"lo_discount", "ANS"},
    {"lo_extendedprice", "Bitcomp"},
    {"lo_key", "Cascaded"},
    {"lo_linenum", "Cascaded"},
    {"lo_linenumber", "ANS"},
    {"lo_orderdate", "Cascaded"},
    {"lo_orderpriority", "Gdeflate"},
    {"lo_ordtotalprice", "Cascaded"},
    {"lo_partkey", "Bitcomp"},
    {"lo_quantity", "ANS"},
    {"lo_revenue", "Bitcomp"},
    {"lo_shipmode", "Gdeflate"},
    {"lo_suppkey", "Bitcomp"},
    {"lo_supplycost", "Bitcomp"},
    {"lo_tax", "ANS"},
    {"c_key","Bitcomp"},
    {"c_city","Gdeflate"},
    {"c_nation","Gdeflate"},
    {"c_region","Gdeflate"},
};

static const std::unordered_map<std::string,std::string> BEST_RATIO_COMP = {
    {"lo_commitdate", "Bitcomp"},
    {"lo_custkey", "Cascaded"},
    {"lo_discount", "ANS"},
    {"lo_extendedprice", "Bitcomp"},
    {"lo_key", "Cascaded"},
    {"lo_linenum", "Cascaded"},
    {"lo_linenumber", "ANS"},
    {"lo_orderdate", "Cascaded"},
    {"lo_orderpriority", "Gdeflate"},
    {"lo_ordtotalprice", "Cascaded"},
    {"lo_partkey", "Bitcomp"},
    {"lo_quantity", "ANS"},
    {"lo_revenue", "Bitcomp"},
    {"lo_shipmode", "Gdeflate"},
    {"lo_suppkey", "Bitcomp"},
    {"lo_supplycost", "Bitcomp"},
    {"lo_tax", "ANS"},
    {"c_key","Bitcomp"},
    {"c_city","Gdeflate"},
    {"c_nation","LZ4"},
    {"c_region","LZ4"},
};


class SSBColLayout{
public:
    SSBColLayout(SSBVar &var, std::string variant):var(var),tables(var.scale_factor, var.customer_factor){
        if (variant == "init_populate"){
            tables.init();
            util::Timer timer;
            tables.populate();
            var.population_ms = timer.elapsed();
        }else if (variant == "init_only"){
            tables.init();
        }else{
            std::cout << "Inknown init variant: " << variant << ", exiting!\n";
            std::exit(1);
        }
    }


    SSB_Tables_col<golap::HostMem> tables;

    SSBVar &var;

    bool select_key();
    bool select_linenum();
    bool select_custkey();
    bool select_partkey();
    bool select_suppkey();
    bool select_orderdate();
    bool select_linenumber();
    bool select_orderpriority();
    bool select_shippriority();
    bool select_quantity();
    bool select_extendedprice();
    bool select_ordtotalprice();
    bool select_discount();
    bool select_revenue();
    bool select_supplycost();
    bool select_tax();
    bool select_commitdate();
    bool select_shipmode();
    bool select_c_key();
    bool select_c_name();
    bool select_c_address();
    bool select_c_city();
    bool select_c_nation();
    bool select_c_region();
    bool select_c_phone();
    bool select_c_mktsegment();

    bool filter_key();
    bool filter_linenum();
    bool filter_custkey();
    bool filter_partkey();
    bool filter_suppkey();
    bool filter_orderdate();
    bool filter_linenumber();
    bool filter_orderpriority();
    bool filter_shippriority();
    bool filter_quantity();
    bool filter_extendedprice();
    bool filter_ordtotalprice();
    bool filter_discount();
    bool filter_revenue();
    bool filter_supplycost();
    bool filter_tax();
    bool filter_commitdate();
    bool filter_shipmode();
    bool filter_c_key();
    bool filter_c_name();
    bool filter_c_address();
    bool filter_c_city();
    bool filter_c_nation();
    bool filter_c_region();
    bool filter_c_phone();
    bool filter_c_mktsegment();


    bool query1_1();
    bool query1_2();
    bool query1_3();
    bool query1_1inmem();
    bool query1_2inmem();
    bool query1_3inmem();

    bool query2_1();
    bool query2_2();
    bool query2_3();
    bool query2_1inmem();
    bool query2_2inmem();
    bool query2_3inmem();
    bool query2_1spd();
    bool query2_2spd();
    bool query2_3spd();
    bool query2_1sdp();
    bool query2_2sdp();
    bool query2_3sdp();
    bool query2_1dps();
    bool query2_2dps();
    bool query2_3dps();
    bool query2_1dsp();
    bool query2_2dsp();
    bool query2_3dsp();
    bool query2_1_pre_aggr();
    bool query2_2_pre_aggr();

    bool query3_1();
    bool query3_2();
    bool query3_3();
    bool query3_4();
    bool query3_1inmem();
    bool query3_2inmem();
    bool query3_3inmem();
    bool query3_4inmem();
    bool query3_1a();
    bool query3_2a();
    bool query3_3a();
    bool query3_4a();
    bool query3_1c();
    bool query3_2c();
    bool query3_3c();
    bool query3_4c();
    bool query3_5c();
    bool query3_1d();
    bool query3_2d();
    bool query3_3d();
    bool query3_4d();

    bool query4_1();
    bool query4_2();
    bool query4_3();
    bool query4_1inmem();
    bool query4_2inmem();
    bool query4_3inmem();


};


#include "queries/01.hpp"
#include "queries/01inmem.hpp"
#include "queries/02.hpp"
#include "queries/02_pre_aggr.hpp"
#include "queries/02inmem.hpp"
#include "queries/03.hpp"
#include "queries/03inmem.hpp"
#include "queries/03a.hpp"
#include "queries/03c.hpp"
#include "queries/03d.hpp"
#include "queries/04.hpp"
#include "queries/04inmem.hpp"
#include "queries/select.hpp"



const static std::unordered_map<std::string,decltype(&SSBColLayout::query1_1)> QUERY_FUNC_PTRS{
    {"select_key", &SSBColLayout::select_key},
    {"select_linenum", &SSBColLayout::select_linenum},
    {"select_custkey", &SSBColLayout::select_custkey},
    {"select_partkey", &SSBColLayout::select_partkey},
    {"select_suppkey", &SSBColLayout::select_suppkey},
    {"select_orderdate", &SSBColLayout::select_orderdate},
    {"select_linenumber", &SSBColLayout::select_linenumber},
    {"select_orderpriority", &SSBColLayout::select_orderpriority},
    {"select_shippriority", &SSBColLayout::select_shippriority},
    {"select_quantity", &SSBColLayout::select_quantity},
    {"select_extendedprice", &SSBColLayout::select_extendedprice},
    {"select_ordtotalprice", &SSBColLayout::select_ordtotalprice},
    {"select_discount", &SSBColLayout::select_discount},
    {"select_revenue", &SSBColLayout::select_revenue},
    {"select_supplycost", &SSBColLayout::select_supplycost},
    {"select_tax", &SSBColLayout::select_tax},
    {"select_commitdate", &SSBColLayout::select_commitdate},
    {"select_shipmode", &SSBColLayout::select_shipmode},
    {"select_c_key", &SSBColLayout::select_c_key},
    {"select_c_name", &SSBColLayout::select_c_name},
    {"select_c_address", &SSBColLayout::select_c_address},
    {"select_c_city", &SSBColLayout::select_c_city},
    {"select_c_nation", &SSBColLayout::select_c_nation},
    {"select_c_region", &SSBColLayout::select_c_region},
    {"select_c_phone", &SSBColLayout::select_c_phone},
    {"select_c_mktsegment", &SSBColLayout::select_c_mktsegment},

    {"filter_key", &SSBColLayout::filter_key},
    {"filter_linenum", &SSBColLayout::filter_linenum},
    {"filter_custkey", &SSBColLayout::filter_custkey},
    {"filter_partkey", &SSBColLayout::filter_partkey},
    {"filter_suppkey", &SSBColLayout::filter_suppkey},
    {"filter_orderdate", &SSBColLayout::filter_orderdate},
    {"filter_linenumber", &SSBColLayout::filter_linenumber},
    {"filter_orderpriority", &SSBColLayout::filter_orderpriority},
    {"filter_shippriority", &SSBColLayout::filter_shippriority},
    {"filter_quantity", &SSBColLayout::filter_quantity},
    {"filter_extendedprice", &SSBColLayout::filter_extendedprice},
    {"filter_ordtotalprice", &SSBColLayout::filter_ordtotalprice},
    {"filter_discount", &SSBColLayout::filter_discount},
    {"filter_revenue", &SSBColLayout::filter_revenue},
    {"filter_supplycost", &SSBColLayout::filter_supplycost},
    {"filter_tax", &SSBColLayout::filter_tax},
    {"filter_commitdate", &SSBColLayout::filter_commitdate},
    {"filter_shipmode", &SSBColLayout::filter_shipmode},
    {"filter_c_key", &SSBColLayout::filter_c_key},
    {"filter_c_name", &SSBColLayout::filter_c_name},
    {"filter_c_address", &SSBColLayout::filter_c_address},
    {"filter_c_city", &SSBColLayout::filter_c_city},
    {"filter_c_nation", &SSBColLayout::filter_c_nation},
    {"filter_c_region", &SSBColLayout::filter_c_region},
    {"filter_c_phone", &SSBColLayout::filter_c_phone},
    {"filter_c_mktsegment", &SSBColLayout::filter_c_mktsegment},

    {"query1.1", &SSBColLayout::query1_1},
    {"query1.2", &SSBColLayout::query1_2},
    {"query1.3", &SSBColLayout::query1_3},
    {"query1.1inmem", &SSBColLayout::query1_1inmem},
    {"query1.2inmem", &SSBColLayout::query1_2inmem},
    {"query1.3inmem", &SSBColLayout::query1_3inmem},
    {"query2.1", &SSBColLayout::query2_1},
    {"query2.2", &SSBColLayout::query2_2},
    {"query2.3", &SSBColLayout::query2_3},
    {"query2.1inmem", &SSBColLayout::query2_1inmem},
    {"query2.2inmem", &SSBColLayout::query2_2inmem},
    {"query2.3inmem", &SSBColLayout::query2_3inmem},
    {"query2.1spd", &SSBColLayout::query2_1spd},
    {"query2.2spd", &SSBColLayout::query2_2spd},
    {"query2.3spd", &SSBColLayout::query2_3spd},
    {"query2.1sdp", &SSBColLayout::query2_1sdp},
    {"query2.2sdp", &SSBColLayout::query2_2sdp},
    {"query2.3sdp", &SSBColLayout::query2_3sdp},
    {"query2.1dps", &SSBColLayout::query2_1dps},
    {"query2.2dps", &SSBColLayout::query2_2dps},
    {"query2.3dps", &SSBColLayout::query2_3dps},
    {"query2.1dsp", &SSBColLayout::query2_1dsp},
    {"query2.2dsp", &SSBColLayout::query2_2dsp},
    {"query2.3dsp", &SSBColLayout::query2_3dsp},
    {"query3.1", &SSBColLayout::query3_1},
    {"query3.2", &SSBColLayout::query3_2},
    {"query3.3", &SSBColLayout::query3_3},
    {"query3.4", &SSBColLayout::query3_4},
    {"query3.1inmem", &SSBColLayout::query3_1inmem},
    {"query3.2inmem", &SSBColLayout::query3_2inmem},
    {"query3.3inmem", &SSBColLayout::query3_3inmem},
    {"query3.4inmem", &SSBColLayout::query3_4inmem},
    {"query3.1c", &SSBColLayout::query3_1c},
    {"query3.2c", &SSBColLayout::query3_2c},
    {"query3.3c", &SSBColLayout::query3_3c},
    {"query3.4c", &SSBColLayout::query3_4c},
    {"query3.1a", &SSBColLayout::query3_1a},
    {"query3.2a", &SSBColLayout::query3_2a},
    {"query3.3a", &SSBColLayout::query3_3a},
    {"query3.4a", &SSBColLayout::query3_4a},
    {"query3.5c", &SSBColLayout::query3_5c},
    {"query3.1d", &SSBColLayout::query3_1d},
    {"query3.2d", &SSBColLayout::query3_2d},
    {"query3.3d", &SSBColLayout::query3_3d},
    {"query3.4d", &SSBColLayout::query3_4d},
    {"query4.1", &SSBColLayout::query4_1},
    {"query4.2", &SSBColLayout::query4_2},
    {"query4.3", &SSBColLayout::query4_3},
    {"query4.1inmem", &SSBColLayout::query4_1inmem},
    {"query4.2inmem", &SSBColLayout::query4_2inmem},
    {"query4.3inmem", &SSBColLayout::query4_3inmem},
    {"query2.1_pre_aggr", &SSBColLayout::query2_1_pre_aggr},
    {"query2.2_pre_aggr", &SSBColLayout::query2_2_pre_aggr},
};