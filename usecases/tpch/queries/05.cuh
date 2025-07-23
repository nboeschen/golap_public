#pragma once
#include <cuda_runtime.h>


__global__ void pipeline_q5_orders(
                            golap::HashMap<uint64_t, ORDERKEY_TYPE> order_build_side,
                            golap::HashMap<uint64_t, CUSTKEY_TYPE> cust_build_side,
                            ORDERKEY_TYPE *o_orderkey,
                            CUSTKEY_TYPE *o_custkey,
                            ORDERDATE_TYPE *o_orderdate,
                            uint64_t* joined_c_rid,
                            util::Date odate_lo,
                            util::Date odate_hi,
                            uint64_t tuples_this_round){

    uint64_t r_id,probe_res;
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        if(o_orderdate[r_id] < odate_lo || o_orderdate[r_id] >= odate_hi) continue;

        probe_res = cust_build_side.probe(o_custkey[r_id]);
        if (probe_res == (uint64_t) -1) continue;

        joined_c_rid[r_id] = probe_res;

        order_build_side.insert(r_id, o_orderkey[r_id]);
    }
}


inline __device__ int nation_lookup(NATIONKEY_TYPE& nationkey){
    if (nationkey == 8) return 0;
    else if (nationkey == 9) return 1;
    else if (nationkey == 12) return 2;
    else if (nationkey == 18) return 3;
    else if (nationkey == 21) return 4;
    else return 0;
}

inline __device__ uint64_t nationkey_lookup(const unsigned int& local_cache_idx){
    static uint64_t nationkey[5] = {8,9,12,18,21};
    return nationkey[local_cache_idx];
}

__global__ void pipeline_q5_lineitem(
                            golap::HashAggregate<NATION_GROUP, uint64_t> hash_agg,
                            golap::HashMap<uint64_t, ORDERKEY_TYPE> order_build_side,
                            golap::HashMap<uint64_t, SUPPKEY_TYPE> supp_map,
                            ORDERKEY_TYPE *l_orderkey,
                            SUPPKEY_TYPE *l_suppkey,
                            EXTENDEDPRICE_TYPE *l_extendedprice,
                            DISCOUNT_TYPE *l_discount,
                            NATIONKEY_TYPE *s_nationkey,
                            NATIONKEY_TYPE *c_nationkey,
                            uint64_t* joined_c_rid,
                            uint64_t tuples_this_round
                            ){
    uint64_t r_id,order_probe_res,supp_probe_res;

    __shared__ uint64_t local_cache[5];
    if (threadIdx.x < 5){
        local_cache[threadIdx.x] = 0;
    }
    __syncthreads();
    int local_cache_idx;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < tuples_this_round; r_id += blockDim.x * gridDim.x){
        // __syncwarp();

        order_probe_res = order_build_side.probe(l_orderkey[r_id]);
        if (order_probe_res == (uint64_t) -1) continue;

        supp_probe_res = supp_map.probe(l_suppkey[r_id]);
        if (supp_probe_res == (uint64_t) -1) continue;

        if (s_nationkey[supp_probe_res] != c_nationkey[joined_c_rid[order_probe_res]]) continue;

        local_cache_idx = nation_lookup(c_nationkey[joined_c_rid[order_probe_res]]);
        SumAgg()(local_cache + local_cache_idx, (l_extendedprice[r_id]*(util::Decimal64{100}-l_discount[r_id])).val);
        // hash_agg.add(NATION_GROUP{c_nationkey[joined_c_rid[order_probe_res]]}, (l_extendedprice[r_id]*(util::Decimal64{100}-l_discount[r_id])).val, SumAgg());

    }

    __syncthreads();
    if (threadIdx.x < 5){
        hash_agg.add(NATION_GROUP{nationkey_lookup(threadIdx.x)}, local_cache[threadIdx.x], SumAgg());
    }

}