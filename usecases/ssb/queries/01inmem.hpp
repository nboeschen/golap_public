#pragma once
#include "query_common.cuh"
#include "../ssb.hpp"
#include "01.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "host_structs.hpp"
#include "../data/SSB_def.hpp"

/*
select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key
and d_year = 1993 
and lo_discount between 1 and 3
and lo_quantity < 25;

select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key 
and d_yearmonthnum = 199401 
and lo_discount between 4 and 6
and lo_quantity between 26 and 35;

select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key 
and d_weeknuminyear = 6 
and d_year = 1994 
and lo_discount between 5 and 7 
and lo_quantity between 26 and 35;

 */

using Q1DATE_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Date::d_key),
                                        decltype(Date::d_year),decltype(Date::d_yearmonthnum),
                                        decltype(Date::d_weeknuminyear)>;

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using EXTENDEDPRICE_TYPE = decltype(Lineorder::lo_extendedprice);
using DISCOUNT_TYPE = decltype(Lineorder::lo_discount);
using QUANTITY_TYPE = decltype(Lineorder::lo_quantity);


enum class Q1TYPE{
    Q11,Q12,Q13
};

template <uint8_t DISCOUNT_LO, uint8_t DISCOUNT_HI, uint8_t QUANTITY_LO, uint8_t QUANTITY_HI, Q1TYPE VARIANT>
bool query1inmem(SSBColLayout &ssbobj){
    /**
     * Date: d_key, d_year, d_yearmonthnum, d_weeknuminyear (Keep full columns on device)
     * Lineorder: lo_orderdate, lo_extendedprice, lo_discount, lo_quantity
     * (streamed through)
     *
     */
    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = ssbobj.tables.lineorder.num_tuples * (
                                sizeof(ORDERDATE_TYPE)+sizeof(EXTENDEDPRICE_TYPE)
                                +sizeof(DISCOUNT_TYPE)+sizeof(QUANTITY_TYPE)
                                );

    golap::HostHashMap hash_map(ssbobj.tables.date.num_tuples, ssbobj.tables.date.col<Date::KEY>().data());
    std::atomic<uint64_t> agg_res{0};
    util::SliceSeq date_workslice{ssbobj.tables.date.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> date_slices(ssbobj.var.workers);
    util::SliceSeq lineorder_workslice{ssbobj.tables.lineorder.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> lineorder_slices(ssbobj.var.workers);

    for(auto& [start,end] : date_slices){
        date_workslice.get(start,end);
    }
    for(auto& [start,end] : lineorder_slices){
        lineorder_workslice.get(start,end);
    }
    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);

    util::Timer timer;

    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([worker_idx,&date_slices,&hash_map,&ssbobj]{
            auto[start, end] = date_slices[worker_idx];
            bool qualifies = true;
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                // check date_preds
                qualifies = true;

                if constexpr (VARIANT == Q1TYPE::Q11){
                    if (ssbobj.tables.date.col<Date::YEAR>().data()[tuple_id] != 1993) qualifies = false;
                }else if constexpr (VARIANT == Q1TYPE::Q12){
                    if (ssbobj.tables.date.col<Date::YEARMONTHNUM>().data()[tuple_id] != 199401) qualifies = false;
                }else if constexpr (VARIANT == Q1TYPE::Q13){
                    if (ssbobj.tables.date.col<Date::WEEKNUMINYEAR>().data()[tuple_id] != 6
                        || ssbobj.tables.date.col<Date::YEAR>().data()[tuple_id] != 1994) qualifies = false;
                }

                if (!qualifies) continue;

                hash_map.insert(tuple_id, ssbobj.tables.date.col<Date::KEY>().data()[tuple_id]);
            }
        });
    }
    for(auto &thread: threads) thread.join();
    threads.clear();

    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([worker_idx,&lineorder_slices,&agg_res,&hash_map,&ssbobj]{
            auto[start, end] = lineorder_slices[worker_idx];
            uint64_t join_res,partial=0;
            uint8_t lo_wo_warning = QUANTITY_LO; // silence the warning of comparing against zero
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                // check discount, quantity preds
                if (ssbobj.tables.lineorder.col<Lineorder::DISCOUNT>().data()[tuple_id] < DISCOUNT_LO ||
                    ssbobj.tables.lineorder.col<Lineorder::DISCOUNT>().data()[tuple_id] > DISCOUNT_HI ||
                    ssbobj.tables.lineorder.col<Lineorder::QUANTITY>().data()[tuple_id] < lo_wo_warning ||
                    ssbobj.tables.lineorder.col<Lineorder::QUANTITY>().data()[tuple_id] > QUANTITY_HI) continue;
                if (ssbobj.var.extended_price_pred && ((ssbobj.tables.lineorder.col<Lineorder::EXTENDEDPRICE>().data()[tuple_id] < 3990000) || (ssbobj.tables.lineorder.col<Lineorder::EXTENDEDPRICE>().data()[tuple_id] > 4000000))) continue;

                join_res = hash_map.probe(ssbobj.tables.lineorder.col<Lineorder::ORDERDATE>().data()[tuple_id]);
                if (join_res == (uint64_t) -1) continue;

                partial += ssbobj.tables.lineorder.col<Lineorder::EXTENDEDPRICE>().data()[tuple_id] * ssbobj.tables.lineorder.col<Lineorder::DISCOUNT>().data()[tuple_id];

            }
            agg_res.fetch_add(partial, std::memory_order_relaxed);
        });
    }
    for(auto &thread: threads) thread.join();


    ssbobj.var.time_ms = timer.elapsed();
    ssbobj.var.comp_ms = 0.0;
    ssbobj.var.prune_ms = -1.f;

    util::Log::get().info_fmt("The result is: %lu",agg_res.load());

    return true;
}

bool SSBColLayout::query1_1inmem(){
    return query1inmem<1,3,0,24,Q1TYPE::Q11>(*this);
}
bool SSBColLayout::query1_2inmem(){

    return query1inmem<4,6,26,35,Q1TYPE::Q12>(*this);
}
bool SSBColLayout::query1_3inmem(){
    return query1inmem<5,7,26,35,Q1TYPE::Q13>(*this);
}

