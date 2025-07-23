#pragma once
#include <time.h>


__global__ void pipeline_q1(golap::HashAggregate<MONTH_GROUP, uint64_t> hashagg,
                            decltype(Trips::tpep_pickup_datetime) *tpep_pickup_datetime,
                            decltype(Trips::Trip_distance) *Trip_distance,
                            decltype(Trips::Trip_distance) dist_lo,
                            decltype(Trips::Trip_distance) dist_hi,
                            SumAgg agg_func,
                            uint64_t num){

    uint64_t r_id;
    __shared__ uint64_t block_agg[12];
    if (threadIdx.x < 12){
        block_agg[threadIdx.x] = 0;
    }
    __syncthreads();

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        // __syncwarp();
        if (Trip_distance[r_id] < dist_lo ||
            Trip_distance[r_id] > dist_hi) continue;

        // month needs to be extracted from tpep_pickup_datetime
        auto month = extract_month(tpep_pickup_datetime[r_id].t);

        agg_func(block_agg + month, 1);

    }
    __syncthreads();
    if (threadIdx.x < 12){
        hashagg.add(MONTH_GROUP{(int16_t)(threadIdx.x+1)}, block_agg[threadIdx.x], agg_func);
    }
}