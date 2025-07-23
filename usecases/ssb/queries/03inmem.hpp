#pragma once
#include "query_common.cuh"
#include "../ssb.hpp"
#include "03.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "../data/SSB_def.hpp"

/*
select c_nation, s_nation, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and c_region = 'ASIA'
and s_region = 'ASIA'
and d_year >= 1992 and d_year <= 1997
group by c_nation, s_nation, d_year
order by d_year asc, revenue desc;

select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and c_nation = 'UNITED STATES'
and s_nation = 'UNITED STATES'
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc, revenue desc;

select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and (c_city='UNITED KI1' or c_city='UNITED KI5')
and (s_city='UNITED KI1' or s_city='UNITED KI5')
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc, revenue desc;

select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and (c_city='UNITED KI1' or c_city='UNITED KI5')
and (s_city='UNITED KI1' or s_city='UNITED KI5')
and d_yearmonth = 'Dec1997'
group by c_city, s_city, d_year
order by d_year asc, revenue desc;


*/

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using CUSTKEY_TYPE = decltype(Lineorder::lo_custkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);

using CKEY_TYPE = decltype(Customer::c_key);
using CCITY_TYPE = decltype(Customer::c_city);
using CNATION_TYPE = decltype(Customer::c_nation);
using CREGION_TYPE = decltype(Customer::c_region);

template <typename GROUP, typename CUST_PRED, typename SUPP_PRED, typename DATE_PRED>
bool query3_inmem(SSBColLayout &ssbobj, GROUP group, CUST_PRED cust_pred, SUPP_PRED supp_pred, DATE_PRED date_pred){

    /**
     * Generally needed:
     * Lineorder: lo_orderdate, lo_custkey, lo_suppkey, lo_revenue
     * Supplier: s_key, s_nation, s_region, s_city
     * Date: d_key, d_year
     * Customer: c_key, c_nation, c_region, c_city
     */
    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = ssbobj.tables.lineorder.num_tuples * (
                                sizeof(ORDERDATE_TYPE)+sizeof(CUSTKEY_TYPE)
                                +sizeof(SUPPKEY_TYPE)+sizeof(REVENUE_TYPE))
                              +ssbobj.tables.customer.num_tuples * (
                                sizeof(CKEY_TYPE)+sizeof(CCITY_TYPE)
                                +sizeof(CNATION_TYPE)+sizeof(CREGION_TYPE)
                                );

    golap::HostHashMap customer_hashmap(ssbobj.tables.customer.num_tuples,ssbobj.tables.customer.col<Customer::KEY>().data());
    golap::HostHashMap supplier_hashmap(ssbobj.tables.supplier.num_tuples,ssbobj.tables.supplier.col<Supplier::KEY>().data());
    golap::HostHashMap date_hashmap(ssbobj.tables.date.num_tuples,ssbobj.tables.date.col<Date::KEY>().data());

    uint64_t num_groups = 800;
    golap::HostMem groups(golap::Tag<GROUP>{}, num_groups);
    auto aggs = new std::atomic<uint64_t>[num_groups]{};
    golap::HostHashAggregate hash_agg(num_groups, groups.ptr<GROUP>(), aggs);

    util::SliceSeq cust_workslice{ssbobj.tables.customer.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> customer_slices(ssbobj.var.workers);
    util::SliceSeq supplier_workslice{ssbobj.tables.supplier.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> supplier_slices(ssbobj.var.workers);
    util::SliceSeq date_workslice{ssbobj.tables.date.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> date_slices(ssbobj.var.workers);
    util::SliceSeq lineorder_workslice{ssbobj.tables.lineorder.num_tuples, ssbobj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> lineorder_slices(ssbobj.var.workers);

    for(auto& [start,end] : customer_slices)    cust_workslice.get(start,end);
    for(auto& [start,end] : supplier_slices)    supplier_workslice.get(start,end);
    for(auto& [start,end] : date_slices)        date_workslice.get(start,end);
    for(auto& [start,end] : lineorder_slices)   lineorder_workslice.get(start,end);

    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);


    /**
     * Start the timer
     */
    util::Timer timer;

    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([worker_idx,&customer_slices,&supplier_slices,&date_slices,
                              &customer_hashmap,&supplier_hashmap,&date_hashmap,&ssbobj,&cust_pred,&supp_pred,
                              &date_pred]{
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
        });
    }
    for(auto &thread: threads) thread.join();
    threads.clear();
    std::cout << "Hashtable Builds took " << timer.elapsed() << " ms\n";

    for (uint32_t worker_idx=0; worker_idx<ssbobj.var.workers; ++worker_idx){
        threads.emplace_back([worker_idx,&lineorder_slices,&hash_agg,&ssbobj,
                              &customer_hashmap,&supplier_hashmap,&date_hashmap]{
            auto[start, end] = lineorder_slices[worker_idx];
            uint64_t customer_match,supplier_match,date_match;

            auto local = []{
                if constexpr(std::is_same_v<GROUP,Q3NATION_GROUP>){
                    return std::unordered_map<Q3NATION_GROUP,uint64_t, HASHER<Q3NATION_GROUP>>();
                }else return std::unordered_map<Q3CITY_GROUP,uint64_t, HASHER<Q3CITY_GROUP>>();
                __builtin_unreachable();
            }();

            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                customer_match = customer_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::CUSTKEY>().data()[tuple_id]);
                if (customer_match == (uint64_t) -1) continue;

                supplier_match = supplier_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::SUPPKEY>().data()[tuple_id]);
                if (supplier_match == (uint64_t) -1) continue;

                date_match = date_hashmap.probe(ssbobj.tables.lineorder.col<Lineorder::ORDERDATE>().data()[tuple_id]);
                if (date_match == (uint64_t) -1) continue;

                if constexpr(std::is_same_v<GROUP,Q3NATION_GROUP>){
                    Q3NATION_GROUP group{ssbobj.tables.supplier.col<Supplier::NATION>().data()[supplier_match],
                                                ssbobj.tables.customer.col<Customer::NATION>().data()[customer_match],
                                                ssbobj.tables.date.col<Date::YEAR>().data()[date_match]};
                    auto search = local.find(group);
                    if (search == local.end()){
                        local.emplace(group,ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id]);
                    } else {
                        search->second += ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id];
                    }
                }else{
                    // city cols
                    Q3CITY_GROUP group{ssbobj.tables.supplier.col<Supplier::CITY>().data()[supplier_match],
                                                ssbobj.tables.customer.col<Customer::CITY>().data()[customer_match],
                                                ssbobj.tables.date.col<Date::YEAR>().data()[date_match]};
                    auto search = local.find(group);
                    if (search == local.end()){
                        local.emplace(group,ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id]);
                    } else {
                        search->second += ssbobj.tables.lineorder.col<Lineorder::REVENUE>().data()[tuple_id];
                    }
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

bool SSBColLayout::query3_1inmem(){
    return query3_inmem(*this,Q3NATION_GROUP{},
                                            golap::PredInfo<CREGION_TYPE,RegionAsiaPred>{tables.customer.col<Customer::REGION>().data(),
                                                                                RegionAsiaPred()},
                                            golap::PredInfo<CREGION_TYPE,RegionAsiaPred>{tables.supplier.col<Supplier::REGION>().data(),
                                                                                RegionAsiaPred()},
                                            golap::PredInfo<decltype(Date::d_year),DATE_9297>{tables.date.col<Date::YEAR>().data(), DATE_9297()});
}
bool SSBColLayout::query3_2inmem(){
    return query3_inmem(*this,Q3CITY_GROUP{},
                                            golap::PredInfo<CNATION_TYPE,NationUSPred>{tables.customer.col<Customer::NATION>().data(),
                                                                                 NationUSPred()},
                                            golap::PredInfo<CNATION_TYPE,NationUSPred>{tables.supplier.col<Supplier::NATION>().data(),
                                                                                 NationUSPred()},
                                            golap::PredInfo<decltype(Date::d_year),DATE_9297>{tables.date.col<Date::YEAR>().data(), DATE_9297()});
}
bool SSBColLayout::query3_3inmem(){
    return query3_inmem(*this,Q3CITY_GROUP{},
                                            golap::PredInfo<CCITY_TYPE,CityKIPred>{tables.customer.col<Customer::CITY>().data(),
                                                                                 CityKIPred()},
                                            golap::PredInfo<CCITY_TYPE,CityKIPred>{tables.supplier.col<Supplier::CITY>().data(),
                                                                                 CityKIPred()},
                                            golap::PredInfo<decltype(Date::d_year),DATE_9297>{tables.date.col<Date::YEAR>().data(), DATE_9297()});
}

bool SSBColLayout::query3_4inmem(){
    return query3_inmem(*this,Q3CITY_GROUP{},
                                            golap::PredInfo<CCITY_TYPE,CityKIPred>{tables.customer.col<Customer::CITY>().data(),
                                                                                 CityKIPred()},
                                            golap::PredInfo<CCITY_TYPE,CityKIPred>{tables.supplier.col<Supplier::CITY>().data(),
                                                                                 CityKIPred()},
                                            golap::PredInfo<decltype(Date::d_yearmonth),YEARMONTHDec1997>{tables.date.col<Date::YEARMONTH>().data(), YEARMONTHDec1997()});
}
