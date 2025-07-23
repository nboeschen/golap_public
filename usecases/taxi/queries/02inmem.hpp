#pragma once

#include "host_structs.hpp"
#include "common.cuh"

/*
SELECT
  EXTRACT('dayofweek' FROM tpep_pickup_datetime),
  ROUND(AVG(trip_distance / (EXTRACT('epoch' FROM tpep_dropoff_datetime)-EXTRACT('epoch' FROM tpep_pickup_datetime)))*3600, 1) as speed
FROM '/mnt/labstore/nboeschen/taxi/*.parquet'
WHERE
  trip_distance > 0
  AND fare_amount/trip_distance BETWEEN 2 AND 10
  AND tpep_dropoff_datetime > tpep_pickup_datetime
GROUP BY EXTRACT('dayofweek' FROM tpep_pickup_datetime);

 */


using TIME_TYPE = decltype(Trips::tpep_pickup_datetime);
using DISTANCE_TYPE = decltype(Trips::Trip_distance);
using FARE_TYPE = decltype(Trips::Fare_amount);

bool query2inmem(TaxiColLayout& obj, FARE_TYPE fare_lo, FARE_TYPE fare_hi){
    obj.var.comp_bytes = 0;
    obj.var.uncomp_bytes = obj.tables.trips.num_tuples * (
                                2*sizeof(TIME_TYPE)+sizeof(DISTANCE_TYPE)+sizeof(FARE_TYPE)
                                );

    uint64_t num_groups = 50;
    golap::HostMem groups(golap::Tag<DAY_GROUP>{}, num_groups);
    auto aggs = new std::atomic<double>[num_groups]{};
    auto aggs_n = new std::atomic<int64_t>[num_groups]{};
    golap::HostHashAggregate hash_agg(num_groups, groups.ptr<DAY_GROUP>(), aggs);

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
            int day;
            int64_t local_agg_n[7]{};
            double local_aggs[7]{};
            double fare_amount;
            int64_t trip_seconds;

            for (uint64_t tuple_id = start; tuple_id < end; ++tuple_id){
                fare_amount = obj.tables.trips.col<Trips::FARE_AMOUNT>().data()[tuple_id];

                if (obj.tables.trips.col<Trips::TRIP_DISTANCE>().data()[tuple_id] <= 0.0 ||
                    fare_amount < fare_lo || fare_amount > fare_hi ||
                    obj.tables.trips.col<Trips::TPEP_DROPOFF_DATETIME>().data()[tuple_id].t <= obj.tables.trips.col<Trips::TPEP_PICKUP_DATETIME>().data()[tuple_id].t) continue;

                trip_seconds = obj.tables.trips.col<Trips::TPEP_DROPOFF_DATETIME>().data()[tuple_id].t - obj.tables.trips.col<Trips::TPEP_PICKUP_DATETIME>().data()[tuple_id].t;

                day = extract_day(obj.tables.trips.col<Trips::TPEP_PICKUP_DATETIME>().data()[tuple_id].t);

                // locally use standard rolling average
                local_aggs[day] += obj.tables.trips.col<Trips::TRIP_DISTANCE>().data()[tuple_id] / trip_seconds;
                local_agg_n[day] += 1;
            }
            for(uint16_t i = 0; i<7; ++i){
                hash_agg.add(DAY_GROUP{(int16_t)(i)}, local_aggs[i], HostFloatSum());
                aggs_n[i].fetch_add(local_agg_n[i], std::memory_order_relaxed);
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

        auto& group = groups.ptr<DAY_GROUP>()[i];
        // std::cout << group <<"," << aggs[i] <<"\n";
        util::Log::get().info_fmt("day: %d, SUM: %.2f, COUNT: %llu, Avg: %.2f", group.day, aggs[i].load(), aggs_n[group.day].load(), 3600*aggs[i].load() / aggs_n[group.day].load());
    }

    delete[] aggs;
    delete[] aggs_n;

    return true;
}

bool TaxiColLayout::query2_1inmem(){
    return query2inmem(*this, 0, 2);
}
bool TaxiColLayout::query2_2inmem(){
    return query2inmem(*this, 2, 10);
}
bool TaxiColLayout::query2_3inmem(){
    return query2inmem(*this, 10, 5000);
}

