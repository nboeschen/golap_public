#pragma once

#include "host_structs.hpp"
#include "common.cuh"

/*
-- Summed gift cost per month

SELECT EXTRACT('month' FROM tpep_pickup_datetime) as month, COUNT(*) trips 
FROM trips 
WHERE trip_distance >= 5 
GROUP BY month;
 */



using PICKUP_TIME_TYPE = decltype(Trips::tpep_pickup_datetime);
using DISTANCE_TYPE = decltype(Trips::Trip_distance);

bool query1inmem(TaxiColLayout &obj, DISTANCE_TYPE dist_lo, DISTANCE_TYPE dist_hi){
    obj.var.comp_bytes = 0;
    obj.var.uncomp_bytes = obj.tables.trips.num_tuples * (
                                sizeof(PICKUP_TIME_TYPE)+sizeof(DISTANCE_TYPE)
                                );

    uint64_t num_groups = 50;
    golap::HostMem groups(golap::Tag<MONTH_GROUP>{}, num_groups);
    auto aggs = new std::atomic<uint64_t>[num_groups]{};
    golap::HostHashAggregate hash_agg(num_groups, groups.ptr<MONTH_GROUP>(), aggs);

    util::SliceSeq exp_workslice{obj.tables.trips.num_tuples, obj.var.workers};
    std::vector<std::tuple<uint64_t,uint64_t>> exp_slices(obj.var.workers);

    for(auto& [start,end] : exp_slices){
        exp_workslice.get(start,end);
    }

    std::vector<std::thread> threads;
    threads.reserve(obj.var.workers);

    util::Timer timer;

    for (uint32_t worker_idx=0; worker_idx<obj.var.workers; ++worker_idx){
        threads.emplace_back([&,worker_idx]{
            auto[start, end] = exp_slices[worker_idx];
            int month;
            uint64_t local_aggs[12]{};

            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                if (obj.tables.trips.col<Trips::TRIP_DISTANCE>().data()[tuple_id] < dist_lo ||
                    obj.tables.trips.col<Trips::TRIP_DISTANCE>().data()[tuple_id] > dist_hi) continue;

                month = extract_month(obj.tables.trips.col<Trips::TPEP_PICKUP_DATETIME>().data()[tuple_id].t);
                local_aggs[month] += 1;
            }
            for(uint16_t i = 0; i<12; ++i){
                hash_agg.add(MONTH_GROUP{(int16_t)(i+1)}, local_aggs[i], HostSum());
            }
        });
    }
    for(auto &thread: threads) thread.join();


    obj.var.time_ms = timer.elapsed();
    obj.var.comp_ms = 0.0;
    obj.var.prune_ms = -1.f;

    uint64_t num_actual_groups = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if ( hash_agg.wrote_group[i] == 0) continue;
        num_actual_groups += 1;

        auto& group = groups.ptr<MONTH_GROUP>()[i];
        // std::cout << group <<"," << aggs[i] <<"\n";
        util::Log::get().info_fmt("Month: %d, Sum: %lu",group.month,aggs[i].load());
    }

    return true;
}


bool TaxiColLayout::query1_1inmem(){
    return query1inmem(*this, 0, 5);
}
bool TaxiColLayout::query1_2inmem(){
    return query1inmem(*this, 5, 15);
}
bool TaxiColLayout::query1_3inmem(){
    return query1inmem(*this, 15, 5000);
}

