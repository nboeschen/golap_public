#pragma once
#include <cuda_runtime.h>

// sum(l_quantity) as sum_qty,
// sum(l_extendedprice) as sum_base_price, "
// sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
// sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
// avg(l_quantity) as avg_qty, "
// avg(l_extendedprice) as avg_price, "
// avg(l_discount) as avg_disc, "
// count(*) as count_order "

// A, F --> 0
// R, F --> 1
// N, F --> 2
// N, O --> 3
inline __device__ int flag_status_lookup(FLAG_STATUS& flag_status){
    if (flag_status.hash() == 16710) return 0;
    else if (flag_status.hash() == 21062) return 1;
    else if (flag_status.hash() == 20038) return 2;
    else if (flag_status.hash() == 20047) return 3;
    else return 0;
}

__global__ void pipeline_q1(
                            golap::HashAggregate<FLAG_STATUS, uint64_t> hash_agg,
                            RETURNFLAG_TYPE *returnflag,
                            LINESTATUS_TYPE *linestatus,
                            QUANTITY_TYPE *quantity,
                            EXTENDEDPRICE_TYPE *extendedprice,
                            DISCOUNT_TYPE *discount,
                            TAX_TYPE *tax,
                            SHIPDATE_TYPE *shipdate,
                            util::Date date_hi,
                            uint64_t *other_aggs,
                            uint64_t tuples_this_round
                            ){
    uint64_t r_id,group_slot;

    const int total_cache = 8*6;
    __shared__ uint64_t local_cache[total_cache];
    if (threadIdx.x < total_cache){
        local_cache[threadIdx.x] = 0;
    }
    __syncthreads();
    int local_cache_idx;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        // __syncwarp();
        if(shipdate[r_id] > date_hi) continue;

        // use the hash_agg for the sum of discount, the given prepare memory locations for the other aggregations
        // group_slot = hash_agg.add(FLAG_STATUS{returnflag[r_id], linestatus[r_id]}, quantity[r_id], SumByte());

        // SumAgg()(other_aggs+8*group_slot+0, extendedprice[r_id].val);
        // SumAgg()(other_aggs+8*group_slot+1, (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])).val); // this includes a divison
        // SumAgg()(other_aggs+8*group_slot+2, (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])*(util::Decimal64{100}+tax[r_id])).val); // this includes a divison
        // SumAgg()(other_aggs+8*group_slot+3, extendedprice[r_id].val); // sum extended
        // SumAgg()(other_aggs+8*group_slot+4, discount[r_id].val); // sum discount
        // SumAgg()(other_aggs+8*group_slot+5, 1); // count


        FLAG_STATUS flag_status{returnflag[r_id], linestatus[r_id]};
        local_cache_idx = flag_status_lookup(flag_status);

        SumAgg()(local_cache+8*local_cache_idx+0, extendedprice[r_id].val);
        SumAgg()(local_cache+8*local_cache_idx+1, (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])).val); // this includes a divison
        SumAgg()(local_cache+8*local_cache_idx+2, (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])*(util::Decimal64{100}+tax[r_id])).val); // this includes a divison
        SumAgg()(local_cache+8*local_cache_idx+3, extendedprice[r_id].val); // sum extended
        SumAgg()(local_cache+8*local_cache_idx+4, discount[r_id].val); // sum discount
        SumAgg()(local_cache+8*local_cache_idx+5, 1); // count
        SumByte()(local_cache+8*local_cache_idx+6, quantity[r_id]); // count

    }
    __syncthreads();
    if (threadIdx.x < 4){
        // this could be further optimized

        if (threadIdx.x == 0) group_slot = hash_agg.add(FLAG_STATUS{'A', 'F'}, local_cache[8*0+6], SumAgg());
        else if (threadIdx.x == 1) group_slot = hash_agg.add(FLAG_STATUS{'R', 'F'}, local_cache[8*1+6], SumAgg());
        else if (threadIdx.x == 2) group_slot = hash_agg.add(FLAG_STATUS{'N', 'F'}, local_cache[8*2+6], SumAgg());
        else if (threadIdx.x == 3) group_slot = hash_agg.add(FLAG_STATUS{'N', 'O'}, local_cache[8*3+6], SumAgg());

        SumAgg()(other_aggs+8*group_slot+0, local_cache[8*threadIdx.x+0]);
        SumAgg()(other_aggs+8*group_slot+1, local_cache[8*threadIdx.x+1]);
        SumAgg()(other_aggs+8*group_slot+2, local_cache[8*threadIdx.x+2]);
        SumAgg()(other_aggs+8*group_slot+3, local_cache[8*threadIdx.x+3]);
        SumAgg()(other_aggs+8*group_slot+4, local_cache[8*threadIdx.x+4]);
        SumAgg()(other_aggs+8*group_slot+5, local_cache[8*threadIdx.x+5]);

    }
}

__global__ void pipeline_q11(
                            golap::HashAggregate<FLAG_STATUS, uint64_t> hash_agg,
                            RETURNFLAG_TYPE *returnflag,
                            LINESTATUS_TYPE *linestatus,
                            QUANTITY_TYPE *quantity,
                            EXTENDEDPRICE_TYPE *extendedprice,
                            DISCOUNT_TYPE *discount,
                            TAX_TYPE *tax,
                            SHIPDATE_TYPE *shipdate,
                            util::Date date_hi,
                            uint64_t *other_aggs,
                            uint64_t tuples_this_round
                            ){
    uint64_t r_id,group_slot;
    uint64_t local_cache[8]{};

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        // __syncwarp();
        if(shipdate[r_id] > date_hi) continue;

        if (returnflag[r_id] != 'A' or linestatus[r_id] != 'F') continue;

        local_cache[0] += extendedprice[r_id].val;
        local_cache[1] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])).val; // this includes a divison
        local_cache[2] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])*(util::Decimal64{100}+tax[r_id])).val; // this includes a divison
        local_cache[3] += extendedprice[r_id].val; // sum extended
        local_cache[4] += discount[r_id].val; // sum discount
        local_cache[5] += 1; // count
        local_cache[6] += quantity[r_id]; // count

    }

    group_slot = hash_agg.add(FLAG_STATUS{'A', 'F'}, local_cache[0+6], SumAgg());
    SumAgg()(other_aggs+8*group_slot+0, local_cache[0]);
    SumAgg()(other_aggs+8*group_slot+1, local_cache[1]);
    SumAgg()(other_aggs+8*group_slot+2, local_cache[2]);
    SumAgg()(other_aggs+8*group_slot+3, local_cache[3]);
    SumAgg()(other_aggs+8*group_slot+4, local_cache[4]);
    SumAgg()(other_aggs+8*group_slot+5, local_cache[5]);
}

__global__ void pipeline_q12(
                            golap::HashAggregate<FLAG_STATUS, uint64_t> hash_agg,
                            RETURNFLAG_TYPE *returnflag,
                            LINESTATUS_TYPE *linestatus,
                            QUANTITY_TYPE *quantity,
                            EXTENDEDPRICE_TYPE *extendedprice,
                            DISCOUNT_TYPE *discount,
                            TAX_TYPE *tax,
                            SHIPDATE_TYPE *shipdate,
                            util::Date date_hi,
                            uint64_t *other_aggs,
                            uint64_t tuples_this_round
                            ){
    uint64_t r_id,group_slot;
    uint64_t local_cache[8]{};

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        // __syncwarp();
        if(shipdate[r_id] > date_hi) continue;

        if (returnflag[r_id] != 'R' or linestatus[r_id] != 'F') continue;

        local_cache[0] += extendedprice[r_id].val;
        local_cache[1] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])).val; // this includes a divison
        local_cache[2] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])*(util::Decimal64{100}+tax[r_id])).val; // this includes a divison
        local_cache[3] += extendedprice[r_id].val; // sum extended
        local_cache[4] += discount[r_id].val; // sum discount
        local_cache[5] += 1; // count
        local_cache[6] += quantity[r_id]; // count

    }

    group_slot = hash_agg.add(FLAG_STATUS{'R', 'F'}, local_cache[0+6], SumAgg());
    SumAgg()(other_aggs+8*group_slot+0, local_cache[0]);
    SumAgg()(other_aggs+8*group_slot+1, local_cache[1]);
    SumAgg()(other_aggs+8*group_slot+2, local_cache[2]);
    SumAgg()(other_aggs+8*group_slot+3, local_cache[3]);
    SumAgg()(other_aggs+8*group_slot+4, local_cache[4]);
    SumAgg()(other_aggs+8*group_slot+5, local_cache[5]);
}

__global__ void pipeline_q13(
                            golap::HashAggregate<FLAG_STATUS, uint64_t> hash_agg,
                            RETURNFLAG_TYPE *returnflag,
                            LINESTATUS_TYPE *linestatus,
                            QUANTITY_TYPE *quantity,
                            EXTENDEDPRICE_TYPE *extendedprice,
                            DISCOUNT_TYPE *discount,
                            TAX_TYPE *tax,
                            SHIPDATE_TYPE *shipdate,
                            util::Date date_hi,
                            uint64_t *other_aggs,
                            uint64_t tuples_this_round
                            ){
    uint64_t r_id,group_slot;
    uint64_t local_cache[8]{};

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        // __syncwarp();
        if(shipdate[r_id] > date_hi) continue;

        if (returnflag[r_id] != 'N' or linestatus[r_id] != 'F') continue;

        local_cache[0] += extendedprice[r_id].val;
        local_cache[1] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])).val; // this includes a divison
        local_cache[2] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])*(util::Decimal64{100}+tax[r_id])).val; // this includes a divison
        local_cache[3] += extendedprice[r_id].val; // sum extended
        local_cache[4] += discount[r_id].val; // sum discount
        local_cache[5] += 1; // count
        local_cache[6] += quantity[r_id]; // count

    }

    group_slot = hash_agg.add(FLAG_STATUS{'N', 'F'}, local_cache[0+6], SumAgg());
    SumAgg()(other_aggs+8*group_slot+0, local_cache[0]);
    SumAgg()(other_aggs+8*group_slot+1, local_cache[1]);
    SumAgg()(other_aggs+8*group_slot+2, local_cache[2]);
    SumAgg()(other_aggs+8*group_slot+3, local_cache[3]);
    SumAgg()(other_aggs+8*group_slot+4, local_cache[4]);
    SumAgg()(other_aggs+8*group_slot+5, local_cache[5]);
}

__global__ void pipeline_q14(
                            golap::HashAggregate<FLAG_STATUS, uint64_t> hash_agg,
                            RETURNFLAG_TYPE *returnflag,
                            LINESTATUS_TYPE *linestatus,
                            QUANTITY_TYPE *quantity,
                            EXTENDEDPRICE_TYPE *extendedprice,
                            DISCOUNT_TYPE *discount,
                            TAX_TYPE *tax,
                            SHIPDATE_TYPE *shipdate,
                            util::Date date_hi,
                            uint64_t *other_aggs,
                            uint64_t tuples_this_round
                            ){
    uint64_t r_id,group_slot;
    uint64_t local_cache[8]{};

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        // __syncwarp();
        if(shipdate[r_id] > date_hi) continue;

        if (returnflag[r_id] != 'N' or linestatus[r_id] != 'O') continue;

        local_cache[0] += extendedprice[r_id].val;
        local_cache[1] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])).val; // this includes a divison
        local_cache[2] += (extendedprice[r_id]*(util::Decimal64{100}-discount[r_id])*(util::Decimal64{100}+tax[r_id])).val; // this includes a divison
        local_cache[3] += extendedprice[r_id].val; // sum extended
        local_cache[4] += discount[r_id].val; // sum discount
        local_cache[5] += 1; // count
        local_cache[6] += quantity[r_id]; // count

    }

    group_slot = hash_agg.add(FLAG_STATUS{'N', 'O'}, local_cache[0+6], SumAgg());
    SumAgg()(other_aggs+8*group_slot+0, local_cache[0]);
    SumAgg()(other_aggs+8*group_slot+1, local_cache[1]);
    SumAgg()(other_aggs+8*group_slot+2, local_cache[2]);
    SumAgg()(other_aggs+8*group_slot+3, local_cache[3]);
    SumAgg()(other_aggs+8*group_slot+4, local_cache[4]);
    SumAgg()(other_aggs+8*group_slot+5, local_cache[5]);
}
