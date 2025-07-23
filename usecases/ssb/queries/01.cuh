#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include "dev_structs.cuh"
#include "../data/SSB_def.hpp"

template <uint8_t DISCOUNT_LO, uint8_t DISCOUNT_HI>
struct DiscountPred{
    __host__ __device__ inline
    bool operator()(decltype(Lineorder::lo_discount)* tup){
        return *tup>=DISCOUNT_LO && *tup<=DISCOUNT_HI;
    }
};
template <uint8_t QUANTITY_LO, uint8_t QUANTITY_HI>
struct QuantityPred{
    __host__ __device__ inline
    bool operator()(decltype(Lineorder::lo_quantity)* tup){
        uint8_t lo_wo_warning = QUANTITY_LO; // silence the warning of comparing against zero
        return *tup>=lo_wo_warning && *tup<=QUANTITY_HI;
    }
};



template <typename PRED0, typename PRED1, typename AGGFUNC>
__global__ void pipeline_q1_lineorder(golap::HashMap<uint64_t, decltype(Date::d_key)> hashmap,
                                      decltype(Lineorder::lo_orderdate)*lo_orderdate,
                                      decltype(Lineorder::lo_discount)* lo_discount, PRED0 discount_pred,
                                      decltype(Lineorder::lo_quantity)* lo_quantity, PRED1 quantity_pred,
                                      decltype(Lineorder::lo_extendedprice)* lo_extendedprice,
                                      decltype(Lineorder::lo_extendedprice) lo,
                                      decltype(Lineorder::lo_extendedprice) hi,
                                      AGGFUNC agg_func,
                                      uint64_t *agg, uint64_t num
                                      ){

    uint64_t r_id,probe_res;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        if (!discount_pred(&lo_discount[r_id]) || !quantity_pred(&lo_quantity[r_id]) || lo_extendedprice[r_id] < lo || lo_extendedprice[r_id] > hi){
            continue;
        }
        probe_res = hashmap.probe(lo_orderdate[r_id]);
        if (probe_res == (uint64_t) -1) continue;

        agg_func(agg,lo_extendedprice[r_id] * lo_discount[r_id]);

    }


}
