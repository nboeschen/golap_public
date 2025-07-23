#pragma once
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#include "dev_structs.cuh"

namespace golap{


template <typename GROUP_TYPE, typename AGG_TYPE, typename COL0_TYPE, typename COL1_TYPE, typename AGG_COL_TYPE, typename AGG_FUNC>
__global__ void group_by_agg_2col(HashAggregate<GROUP_TYPE, AGG_TYPE> hashagg, COL0_TYPE *col0, COL1_TYPE *col1,
                                  AGG_COL_TYPE *agg_col,
                                  uint64_t num, AGG_FUNC agg_func){
    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= num) return;

    hashagg.add(GROUP_TYPE{col0[r_id],col1[r_id]}, agg_col[r_id], agg_func);
}

template <typename GROUP_TYPE, typename AGG_TYPE, typename COL0_TYPE, typename COL1_TYPE,
            typename COL2_TYPE, typename COL3_TYPE, typename AGG_COL_TYPE, typename AGG_FUNC>
__global__ void group_by_agg_4col(HashAggregate<GROUP_TYPE, AGG_TYPE> hashagg, COL0_TYPE *col0, COL1_TYPE *col1,
                                  COL2_TYPE *col2, COL3_TYPE *col3,
                                  AGG_COL_TYPE *agg_col,
                                  uint64_t num, AGG_FUNC agg_func){
    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= num) return;

    hashagg.add(GROUP_TYPE{col0[r_id],col1[r_id],col2[r_id],col3[r_id]}, agg_col[r_id], agg_func);
}


} // end of namespace