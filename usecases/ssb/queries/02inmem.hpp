#pragma once
#include "query_common.cuh"
#include "../ssb.hpp"
#include "02.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "host_structs.hpp"
#include "../data/SSB_def.hpp"

/*
select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
group by d_year, p_brand1
order by d_year, p_brand1

select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 between 'MFGR#2221'
and 'MFGR#2228'
and s_region = 'ASIA'
group by d_year, p_brand1
order by d_year, p_brand1;

select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1= 'MFGR#2239'
and s_region = 'EUROPE'
group by d_year, p_brand1
order by d_year, p_brand1;

*/

using Q2PART_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Part::p_key),
                                    decltype(Part::p_brand1),decltype(Part::p_category)>;

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using PARTKEY_TYPE = decltype(Lineorder::lo_partkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);

enum class Q2TYPE{
    Q21,Q22,Q23
};

template <typename REGION_PRED, Q2TYPE VARIANT>
bool query2(SSBColLayout &ssbobj, REGION_PRED region_pred){
    
    /**
     * Generally needed:
     * Lineorder: lo_orderdate, lo_partkey, lo_suppkey, lo_revenue
     * Supplier: s_key, s_region
     * Date: d_key, d_year
     * Parts: p_key, p_category, p_brand1
     */
    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = ssbobj.tables.lineorder.num_tuples * (
                                sizeof(ORDERDATE_TYPE)+sizeof(PARTKEY_TYPE)
                                +sizeof(SUPPKEY_TYPE)+sizeof(REVENUE_TYPE)
                                );
    ssbobj.var.comp_ms = 0.0;


    golap::HostHashMap part_hashmap(ssbobj.tables.part.num_tuples, ssbobj.tables.part.col<Part::KEY>().data());
    golap::HostHashMap supplier_hashmap(ssbobj.tables.supplier.num_tuples, ssbobj.tables.supplier.col<Supplier::KEY>().data());
    golap::HostHashMap date_hashmap(ssbobj.tables.date.num_tuples, ssbobj.tables.date.col<Date::KEY>().data());

    uint64_t num_groups = 500;
    golap::HostMem groups(golap::Tag<YEARBRAND_GROUP>{}, num_groups);
    auto aggs = new std::atomic<uint64_t>[num_groups]{};
    golap::HostHashAggregate hash_agg(num_groups, groups.ptr<YEARBRAND_GROUP>(), aggs);

    util::SliceSeq part_workslice{ssbobj.tables.part.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> part_slices(ssbobj.var.workers);
    util::SliceSeq supplier_workslice{ssbobj.tables.supplier.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> supplier_slices(ssbobj.var.workers);
    util::SliceSeq date_workslice{ssbobj.tables.date.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> date_slices(ssbobj.var.workers);
    util::SliceSeq lineorder_workslice{ssbobj.tables.lineorder.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> lineorder_slices(ssbobj.var.workers);
    
    for(auto& [start,end] : part_slices)        part_workslice.get(start,end);
    for(auto& [start,end] : supplier_slices)    supplier_workslice.get(start,end);
    for(auto& [start,end] : date_slices)        date_workslice.get(start,end);
    for(auto& [start,end] : lineorder_slices)   lineorder_workslice.get(start,end);


    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);
    
    util::Timer timer;

    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([&,worker_idx]{
            // build side, part table
            auto[start, end] = part_slices[worker_idx];
            bool qualifies;
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                // check part_preds
                qualifies = true;

                if constexpr (VARIANT == Q2TYPE::Q21){
                    if (!PartPred1()(&ssbobj.tables.part.col<Part::CATEGORY>().data()[tuple_id])) qualifies = false;
                }else if constexpr (VARIANT == Q2TYPE::Q22){
                    if (!PartPred2()(&ssbobj.tables.part.col<Part::BRAND1>().data()[tuple_id])) qualifies = false;
                }else if constexpr (VARIANT == Q2TYPE::Q23){
                    if (!PartPred3()(&ssbobj.tables.part.col<Part::BRAND1>().data()[tuple_id])) qualifies = false;
                }

                if (!qualifies) continue;

                part_hashmap.insert(tuple_id, ssbobj.tables.part.col<Part::KEY>().data()[tuple_id]);
            }
            // build side, supplier table
            std::tie(start, end) = supplier_slices[worker_idx];
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                // check part_preds
                qualifies = true;

                if (!region_pred(&ssbobj.tables.supplier.col<Supplier::REGION>().data()[tuple_id])) qualifies = false;

                if (!qualifies) continue;

                supplier_hashmap.insert(tuple_id, ssbobj.tables.supplier.col<Supplier::KEY>().data()[tuple_id]);
            }
            // build side, date table
            std::tie(start, end) = date_slices[worker_idx];
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                date_hashmap.insert(tuple_id, ssbobj.tables.date.col<Date::KEY>().data()[tuple_id]);
            }
        });
    }
    for(auto &thread: threads) thread.join();
    threads.clear();

    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([&,worker_idx]{
            auto[start, end] = lineorder_slices[worker_idx];
            uint64_t part_match,supplier_match,date_match;

            std::unordered_map<YEARBRAND_GROUP,uint64_t,HASHER<YEARBRAND_GROUP>> local;

            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                part_match = part_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::PARTKEY>().data()[tuple_id]);
                if (part_match == (uint64_t) -1) continue;

                supplier_match = supplier_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::SUPPKEY>().data()[tuple_id]);
                if (supplier_match == (uint64_t) -1) continue;

                date_match = date_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::ORDERDATE>().data()[tuple_id]);

                // hash_agg.add(YEARBRAND_GROUP{ssbobj.tables.date.col<Date::YEAR>().data()[date_match],
                //                             ssbobj.tables.part.col<Part::BRAND1>().data()[part_match]},
                //                             ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id],
                //                             HostSum());
                YEARBRAND_GROUP group{ssbobj.tables.date.col<Date::YEAR>().data()[date_match],
                                            ssbobj.tables.part.col<Part::BRAND1>().data()[part_match]};
                auto search = local.find(group);
                if (search == local.end()){
                    local.emplace(group,ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id]);
                } else {
                    search->second += ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id];
                }
            }

            for (auto& [key,val] : local) {
                hash_agg.add(key,val,HostSum());
            }

        });
    }
    for(auto &thread: threads) thread.join();

    ssbobj.var.time_ms = timer.elapsed();
    /**
     * Stopped timer
     */

    uint64_t num_actual_groups = 0;
    uint64_t complete_sum = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if ( hash_agg.wrote_group[i] == 0) continue;
        num_actual_groups += 1;

        auto& group = groups.ptr<YEARBRAND_GROUP>()[i];
        complete_sum += aggs[i];
        // std::cout << group <<"," << aggs[i] <<"\n";
    }

    util::Log::get().info_fmt("Sum of results            =%lu",complete_sum);
    util::Log::get().info_fmt("Number of results (groups)=%lu",num_actual_groups);
    delete[] aggs;
    return true;
}

bool SSBColLayout::query2_1inmem(){
    return query2<RegionAmericaPred,Q2TYPE::Q21>(*this, RegionAmericaPred());
}


bool SSBColLayout::query2_2inmem(){
    return query2<RegionAsiaPred,Q2TYPE::Q22>(*this, RegionAsiaPred());
}


bool SSBColLayout::query2_3inmem(){
    return query2<RegionEuropePred,Q2TYPE::Q23>(*this, RegionEuropePred());
}
