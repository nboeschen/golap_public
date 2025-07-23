#pragma once
#include "query_common.cuh"
#include "../ssb.hpp"
#include "04.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "../data/SSB_def.hpp"

/*
select d_year, c_nation,
sum(lo_revenue - lo_supplycost) as profit
from date, customer, supplier, part, lineorder
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_region = 'AMERICA'
and (p_mfgr = 'MFGR#1'
or p_mfgr = 'MFGR#2')
group by d_year, c_nation
order by d_year, c_nation;

*/

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using CUSTKEY_TYPE = decltype(Lineorder::lo_custkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using PARTKEY_TYPE = decltype(Lineorder::lo_partkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);
using SUPPLYCOST_TYPE = decltype(Lineorder::lo_supplycost);


template <typename GROUP, typename CUST_PRED, typename SUPP_PRED, typename DATE_PRED, typename PART_PRED>
bool query4inmem(SSBColLayout &ssbobj, GROUP group, CUST_PRED cust_pred, SUPP_PRED supp_pred, DATE_PRED date_pred,
                          PART_PRED part_pred){


    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = ssbobj.tables.lineorder.num_tuples * (
                                sizeof(ORDERDATE_TYPE)+sizeof(CUSTKEY_TYPE)
                                +sizeof(PARTKEY_TYPE)+sizeof(SUPPKEY_TYPE)
                                +sizeof(REVENUE_TYPE)+sizeof(SUPPLYCOST_TYPE)
                                );

    // prepare hashmaps of the four joins
    golap::HostHashMap customer_hashmap(ssbobj.tables.customer.num_tuples,ssbobj.tables.customer.col<Customer::KEY>().data());
    golap::HostHashMap supplier_hashmap(ssbobj.tables.supplier.num_tuples,ssbobj.tables.supplier.col<Supplier::KEY>().data());
    golap::HostHashMap date_hashmap(ssbobj.tables.date.num_tuples,ssbobj.tables.date.col<Date::KEY>().data());
    golap::HostHashMap part_hashmap(ssbobj.tables.part.num_tuples,ssbobj.tables.part.col<Part::KEY>().data());

    uint64_t num_groups = 1000;
    golap::HostMem groups(golap::Tag<GROUP>{}, num_groups);
    auto aggs = new std::atomic<uint64_t>[num_groups]{};
    golap::HostHashAggregate hash_agg(num_groups, groups.ptr<GROUP>(), aggs);


    util::SliceSeq cust_workslice{ssbobj.tables.customer.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> customer_slices(ssbobj.var.workers);
    util::SliceSeq supplier_workslice{ssbobj.tables.supplier.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> supplier_slices(ssbobj.var.workers);
    util::SliceSeq date_workslice{ssbobj.tables.date.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> date_slices(ssbobj.var.workers);
    util::SliceSeq part_workslice{ssbobj.tables.part.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> part_slices(ssbobj.var.workers);
    util::SliceSeq lineorder_workslice{ssbobj.tables.lineorder.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> lineorder_slices(ssbobj.var.workers);
    
    for(auto& [start,end] : customer_slices)    cust_workslice.get(start,end);
    for(auto& [start,end] : supplier_slices)    supplier_workslice.get(start,end);
    for(auto& [start,end] : date_slices)        date_workslice.get(start,end);
    for(auto& [start,end] : part_slices)        part_workslice.get(start,end);
    for(auto& [start,end] : lineorder_slices)   lineorder_workslice.get(start,end);

    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);

    /**
     * Start the timer
     */
    util::Timer timer;

    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([&,worker_idx]{
            // build side, customer table
            auto[start, end] = customer_slices[worker_idx];
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                // check customer_preds

                if (!cust_pred.pred(&cust_pred.col[tuple_id])) continue;

                customer_hashmap.insert(tuple_id, ssbobj.tables.customer.col<Part::KEY>().data()[tuple_id]);
            }
            // build side, supplier table
            std::tie(start, end) = supplier_slices[worker_idx];
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                // check part_preds
                if (!supp_pred.pred(&supp_pred.col[tuple_id])) continue;
                supplier_hashmap.insert(tuple_id, ssbobj.tables.supplier.col<Supplier::KEY>().data()[tuple_id]);
            }
            // build side, date table
            std::tie(start, end) = date_slices[worker_idx];
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                if (!date_pred.pred(&date_pred.col[tuple_id])) continue;
                date_hashmap.insert(tuple_id, ssbobj.tables.date.col<Date::KEY>().data()[tuple_id]);
            }
            // build side, part table
            std::tie(start, end) = part_slices[worker_idx];
            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                if (!part_pred.pred(&part_pred.col[tuple_id])) continue;
                part_hashmap.insert(tuple_id, ssbobj.tables.part.col<Part::KEY>().data()[tuple_id]);
            }
        });
    }
    for(auto &thread: threads) thread.join();
    threads.clear();
    
    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([&,worker_idx]{
            auto[start, end] = lineorder_slices[worker_idx];
            uint64_t customer_match,supplier_match,date_match,part_match;
            uint64_t local_sum;

            auto local = []{
                if constexpr(std::is_same_v<GROUP,Q41_GROUP>) return std::unordered_map<Q41_GROUP, uint64_t, HASHER<Q41_GROUP>>();
                else if constexpr(std::is_same_v<GROUP,Q42_GROUP>) return std::unordered_map<Q42_GROUP, uint64_t, HASHER<Q42_GROUP>>();
                else if constexpr(std::is_same_v<GROUP,Q43_GROUP>) return std::unordered_map<Q43_GROUP, uint64_t, HASHER<Q43_GROUP>>();
                __builtin_unreachable();
            }();

            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                customer_match = customer_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::CUSTKEY>().data()[tuple_id]);
                if (customer_match == (uint64_t) -1) continue;

                supplier_match = supplier_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::SUPPKEY>().data()[tuple_id]);
                if (supplier_match == (uint64_t) -1) continue;

                date_match = date_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::ORDERDATE>().data()[tuple_id]);
                if (date_match == (uint64_t) -1) continue;

                part_match = part_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::PARTKEY>().data()[tuple_id]);
                if (part_match == (uint64_t) -1) continue;

                local_sum = ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id] - ssbobj.tables.lineorder.col<Lineorder::SUPPLYCOST>().data()[tuple_id];

                if constexpr(std::is_same_v<GROUP,Q41_GROUP>){
                    auto group = Q41_GROUP{ssbobj.tables.date.col<Date::YEAR>().data()[date_match],
                                                ssbobj.tables.customer.col<Customer::NATION>().data()[customer_match]};
                    auto search = local.find(group);
                    if (search == local.end()) local.emplace(group,local_sum);
                    else search->second += local_sum;

                } else if constexpr(std::is_same_v<GROUP,Q42_GROUP>){
                    auto group = Q42_GROUP{ssbobj.tables.date.col<Date::YEAR>().data()[date_match],
                                                ssbobj.tables.supplier.col<Supplier::NATION>().data()[supplier_match],
                                                ssbobj.tables.part.col<Part::CATEGORY>().data()[part_match]};
                    auto search = local.find(group);
                    if (search == local.end()) local.emplace(group,local_sum);
                    else search->second += local_sum;
                } else if constexpr(std::is_same_v<GROUP,Q43_GROUP>){
                    auto group = Q43_GROUP{ssbobj.tables.date.col<Date::YEAR>().data()[date_match],
                                                ssbobj.tables.supplier.col<Supplier::CITY>().data()[supplier_match],
                                                ssbobj.tables.part.col<Part::BRAND1>().data()[part_match]};
                    auto search = local.find(group);
                    if (search == local.end()) local.emplace(group,local_sum);
                    else search->second += local_sum;
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
    
    ssbobj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    ssbobj.var.host_mem_used = golap::HOST_ALLOCATED.load();

    uint64_t num_actual_groups = 0;
    uint64_t complete_sum = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if ( hash_agg.wrote_group[i] == 0) continue;
        num_actual_groups += 1;

        // auto& group = groups.ptr<GROUP>()[i];
        complete_sum += aggs[i];
        // std::cout << group <<"," << aggs[i] <<"\n";
    }

    util::Log::get().info_fmt("Sum of results            =%lu",complete_sum);
    util::Log::get().info_fmt("Number of results (groups)=%lu",num_actual_groups);

    return true;
}


bool SSBColLayout::query4_1inmem(){
    return query4inmem(*this,Q41_GROUP{},
                            golap::PredInfo<decltype(Customer::c_region),RegionAmericaPred>{tables.customer.col<Customer::REGION>().data(), RegionAmericaPred()},
                            golap::PredInfo<decltype(Supplier::s_region),RegionAmericaPred>{tables.supplier.col<Supplier::REGION>().data(), RegionAmericaPred()},
                            golap::PredInfo<char,TruePred>{nullptr, TruePred()},
                            golap::PredInfo<decltype(Part::p_mfgr),MFGRPred>{tables.part.col<Part::MFGR>().data(), MFGRPred()}
                            );
}
bool SSBColLayout::query4_2inmem(){
    return query4inmem(*this,Q42_GROUP{},
                            golap::PredInfo<decltype(Customer::c_region),RegionAmericaPred>{tables.customer.col<Customer::REGION>().data(), RegionAmericaPred()},
                            golap::PredInfo<decltype(Supplier::s_region),RegionAmericaPred>{tables.supplier.col<Supplier::REGION>().data(), RegionAmericaPred()},
                            golap::PredInfo<decltype(Date::d_year),DATE_9798>{tables.date.col<Date::YEAR>().data(), DATE_9798()},
                            golap::PredInfo<decltype(Part::p_mfgr),MFGRPred>{tables.part.col<Part::MFGR>().data(), MFGRPred()}
                            );
}
bool SSBColLayout::query4_3inmem(){
    return query4inmem(*this,Q43_GROUP{},
                            golap::PredInfo<decltype(Customer::c_region),RegionAmericaPred>{tables.customer.col<Customer::REGION>().data(), RegionAmericaPred()},
                            golap::PredInfo<decltype(Supplier::s_nation),NationUSPred>{tables.supplier.col<Supplier::NATION>().data(), NationUSPred()},
                            golap::PredInfo<decltype(Date::d_year),DATE_9798>{tables.date.col<Date::YEAR>().data(), DATE_9798()},
                            golap::PredInfo<decltype(Part::p_category),Category14Pred>{tables.part.col<Part::CATEGORY>().data(), Category14Pred()}
                            );
}


