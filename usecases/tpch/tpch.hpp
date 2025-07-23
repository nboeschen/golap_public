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
#include "data/TPCH_def.hpp"

static const std::unordered_map<std::string,std::string> BEST_BW_COMP = {
    {"l_orderkey", "Cascaded"},
    {"l_partkey", "Bitcomp"},
    {"l_suppkey", "Bitcomp"},
    {"l_extendedprice", "ANS"},
    {"l_discount", "ANS"},
    {"l_tax", "ANS"},
    {"l_shipdate", "Gdeflate"},
    {"l_commitdate", "Gdeflate"},
    {"l_receiptdate", "Gdeflate"},
    {"l_shipinstruct", "Gdeflate"},
    {"l_shipmode", "Gdeflate"},
};

struct TPCHVar : public golap::Parameter{
};

class TPCHColLayout{
public:
    TPCHColLayout(TPCHVar &var, std::string variant):var(var),tables(var.scale_factor){
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


    TPCH_Tables_col<golap::HostMem> tables;

    TPCHVar &var;

    bool select_l_orderkey();
    bool select_l_partkey();
    bool select_l_suppkey();
    bool select_l_linenumber();
    bool select_l_quantity();
    bool select_l_extendedprice();
    bool select_l_discount();
    bool select_l_tax();
    bool select_l_returnflag();
    bool select_l_linestatus();
    bool select_l_shipdate();
    bool select_l_commitdate();
    bool select_l_receiptdate();
    bool select_l_shipinstruct();
    bool select_l_shipmode();

    bool filter_l_orderkey();
    bool filter_l_partkey();
    bool filter_l_suppkey();
    bool filter_l_linenumber();
    bool filter_l_quantity();
    bool filter_l_extendedprice();
    bool filter_l_discount();
    bool filter_l_tax();
    bool filter_l_returnflag();
    bool filter_l_linestatus();
    bool filter_l_shipdate();
    bool filter_l_commitdate();
    bool filter_l_receiptdate();
    bool filter_l_shipinstruct();
    bool filter_l_shipmode();


    bool query1inmem();
    bool query3inmem();
    bool query5inmem();
    bool query1();
    bool query3();
    bool query5();
};


#include "queries/select.hpp"
#include "queries/01.hpp"
#include "queries/03.hpp"
#include "queries/05.hpp"
#include "queries/01inmem.hpp"
#include "queries/03inmem.hpp"
#include "queries/05inmem.hpp"


const static std::unordered_map<std::string,decltype(&TPCHColLayout::query1inmem)> QUERY_FUNC_PTRS{
    {"select_l_orderkey", &TPCHColLayout::select_l_orderkey},
    {"select_l_partkey", &TPCHColLayout::select_l_partkey},
    {"select_l_suppkey", &TPCHColLayout::select_l_suppkey},
    {"select_l_linenumber", &TPCHColLayout::select_l_linenumber},
    {"select_l_quantity", &TPCHColLayout::select_l_quantity},
    {"select_l_extendedprice", &TPCHColLayout::select_l_extendedprice},
    {"select_l_discount", &TPCHColLayout::select_l_discount},
    {"select_l_tax", &TPCHColLayout::select_l_tax},
    {"select_l_returnflag", &TPCHColLayout::select_l_returnflag},
    {"select_l_linestatus", &TPCHColLayout::select_l_linestatus},
    {"select_l_shipdate", &TPCHColLayout::select_l_shipdate},
    {"select_l_commitdate", &TPCHColLayout::select_l_commitdate},
    {"select_l_receiptdate", &TPCHColLayout::select_l_receiptdate},
    {"select_l_shipinstruct", &TPCHColLayout::select_l_shipinstruct},
    {"select_l_shipmode", &TPCHColLayout::select_l_shipmode},

    {"filter_l_orderkey", &TPCHColLayout::filter_l_orderkey},
    {"filter_l_partkey", &TPCHColLayout::filter_l_partkey},
    {"filter_l_suppkey", &TPCHColLayout::filter_l_suppkey},
    {"filter_l_linenumber", &TPCHColLayout::filter_l_linenumber},
    {"filter_l_quantity", &TPCHColLayout::filter_l_quantity},
    {"filter_l_extendedprice", &TPCHColLayout::filter_l_extendedprice},
    {"filter_l_discount", &TPCHColLayout::filter_l_discount},
    {"filter_l_tax", &TPCHColLayout::filter_l_tax},
    {"filter_l_returnflag", &TPCHColLayout::filter_l_returnflag},
    {"filter_l_linestatus", &TPCHColLayout::filter_l_linestatus},
    {"filter_l_shipdate", &TPCHColLayout::filter_l_shipdate},
    {"filter_l_commitdate", &TPCHColLayout::filter_l_commitdate},
    {"filter_l_receiptdate", &TPCHColLayout::filter_l_receiptdate},
    {"filter_l_shipinstruct", &TPCHColLayout::filter_l_shipinstruct},
    {"filter_l_shipmode", &TPCHColLayout::filter_l_shipmode},

    {"query1", &TPCHColLayout::query1},
    {"query3", &TPCHColLayout::query3},
    {"query5", &TPCHColLayout::query5},
    {"query1inmem", &TPCHColLayout::query1inmem},
    {"query3inmem", &TPCHColLayout::query3inmem},
    {"query5inmem", &TPCHColLayout::query5inmem},
};