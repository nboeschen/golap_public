#pragma once
#include <cuda_runtime.h>


__global__ void pipeline_q3_orders(
                            golap::HashMap<uint64_t, ORDERKEY_TYPE> hashmap,
                            golap::HashMap<uint64_t, CUSTKEY_TYPE> custorder_map,
                            ORDERKEY_TYPE *o_orderkey,
                            ORDERDATE_TYPE *o_orderdate,
                            CUSTKEY_TYPE *o_custkey,
                            util::Date odate_hi,
                            uint64_t tuples_this_round){

    uint64_t r_id,probe_res;
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        if(o_orderdate[r_id] >= odate_hi) continue;

        probe_res = custorder_map.probe(o_custkey[r_id]);
        if (probe_res == (uint64_t) -1) continue;

        hashmap.insert(r_id, o_orderkey[r_id]);
    }
}

__global__ void pipeline_q3_lineitem(
                            golap::HashAggregate<KEY_DATE_PRIO, uint64_t> hash_agg,
                            golap::HashMap<uint64_t, ORDERKEY_TYPE> hashmap,
                            ORDERKEY_TYPE *l_orderkey,
                            EXTENDEDPRICE_TYPE *l_extendedprice,
                            SHIPDATE_TYPE *l_shipdate,
                            DISCOUNT_TYPE *l_discount,
                            ORDERDATE_TYPE *o_orderdate,
                            SHIPPRIORITY_TYPE *o_shippriority,
                            util::Date ldate_lo,
                            uint64_t tuples_this_round
                            ){
    uint64_t r_id,probe_res;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        if(l_shipdate[r_id] < ldate_lo) continue;

        probe_res = hashmap.probe(l_orderkey[r_id]);
        if (probe_res == (uint64_t) -1) continue;

        hash_agg.add(KEY_DATE_PRIO{l_orderkey[r_id], o_orderdate[probe_res], o_shippriority[probe_res]}, (l_extendedprice[r_id]*(util::Decimal64{100}-l_discount[r_id])).val, SumAgg());

    }

}