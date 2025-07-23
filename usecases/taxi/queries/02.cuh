#pragma once
#include <time.h>


__global__ void pipeline_q2(golap::HashAggregate<DAY_GROUP, double> hashagg,
                            decltype(Trips::tpep_pickup_datetime) *tpep_pickup_datetime,
                            decltype(Trips::tpep_dropoff_datetime) *tpep_dropoff_datetime,
                            decltype(Trips::Trip_distance) *Trip_distance,
                            decltype(Trips::Fare_amount) *Fare_amount,
                            decltype(Trips::Fare_amount) fare_lo,
                            decltype(Trips::Fare_amount) fare_hi,
                            FloatSum agg_func,
                            uint64_t *n,
                            uint64_t num){

    uint64_t r_id;
    int64_t trip_seconds;
    __shared__ unsigned long long int block_agg_n[7];
    __shared__ double block_agg[7];
    unsigned long long int agg_n[7];
    double agg[7];
    if (threadIdx.x < 7){
        block_agg_n[threadIdx.x] = 0;
        block_agg[threadIdx.x] = 0.0;
    }
    __syncthreads();

    for (int i = 0; i< 7; ++i){
        agg_n[i] = 0;
        agg[i] = 0.0;
    }

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        // __syncwarp();
        if (Trip_distance[r_id] <= 0.0 ||
            Fare_amount[r_id] < fare_lo || Fare_amount[r_id]  > fare_hi ||
            tpep_dropoff_datetime[r_id] <= tpep_pickup_datetime[r_id]) continue;

        trip_seconds = tpep_dropoff_datetime[r_id].t - tpep_pickup_datetime[r_id].t;
        // day needs to be extracted from tpep_pickup_datetime
        auto day = extract_day(tpep_pickup_datetime[r_id].t);

        agg[day] += Trip_distance[r_id]/trip_seconds;
        agg_n[day] += 1;
    }
    for (int i = 0; i< 7; ++i){
        agg_func(block_agg + i, agg[i]);
        atomicAdd((unsigned long long *) block_agg_n + i, agg_n[i]);
    }

    __syncthreads();
    if (threadIdx.x < 7){
        hashagg.add(DAY_GROUP{(int16_t)(threadIdx.x)}, block_agg[threadIdx.x], agg_func);
        atomicAdd((unsigned long long *) n+threadIdx.x, (unsigned long long) block_agg_n[threadIdx.x]);
    }
}