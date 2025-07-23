#pragma once
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#include "dev_structs.cuh"

namespace golap{

/**
 * Built the hashmap in parallel. All necessary information is in the hashmap object.
 * Adapt if r_id does not start from zero!
 */
template <typename BUILD_TUPLE, typename BUILD_KEY>
__global__ void hash_map_build(HashMap<BUILD_TUPLE, uint64_t> hashmap, uint64_t num_tuples,
                               BUILD_KEY build_key = DirectKey<BUILD_TUPLE>()){

    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= num_tuples) return;

    hashmap.insert(r_id, build_key(&hashmap.table[r_id]));

}
/**
 * Build the hashmap, after filtering with a predicate on the build key
 */
template <typename BUILD_TUPLE, typename BUILD_KEY, typename PREDICATE>
__global__ void hash_map_build_pred(HashMap<BUILD_TUPLE, uint64_t> hashmap, BUILD_KEY build_key, PREDICATE pred){

    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= hashmap.ht_size || !pred(&hashmap.table[r_id])) return;

    hashmap.insert(r_id, build_key(&hashmap.table[r_id]));

}

/**
 * Probe into a built hashtable and count the number of join partners.
 */
template <typename BUILD_TUPLE, typename PROBE_TUPLE, typename BUILD_KEY, typename PROBE_KEY>
__global__ void hash_join_count(HashMap<BUILD_TUPLE, uint64_t> hashmap, PROBE_TUPLE *probe_table, BUILD_KEY build_key, PROBE_KEY probe_key, uint64_t num, uint64_t *hits){

    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= num) return;

    auto res = hashmap.probe(probe_key(&probe_table[r_id]), build_key);

    if (res != (uint64_t) -1) atomicAdd((unsigned long long*) hits,(unsigned long long) 1);

}

/**
 * Probe into a built hashtable and materialize into a result table.
 * MAT_TUPLE should implement the ctor __host__ __device__ MAT_TUPLE(*BUILD_TUPLE,*PROBE_TUPLE)
 */
template <typename BUILD_TUPLE, typename PROBE_TUPLE, typename BUILD_KEY, typename PROBE_KEY, typename MAT_TUPLE>
__global__ void hash_join_mat(HashMap<BUILD_TUPLE, uint64_t> hashmap, PROBE_TUPLE *probe_table,
                                      BUILD_KEY build_key, PROBE_KEY probe_key, uint64_t num,
                                      MAT_TUPLE *mat_table, uint64_t *matches){

    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= num) return;

    auto res = hashmap.probe(probe_key(&probe_table[r_id]), build_key);

    if (res == (uint64_t) -1) return;

    uint64_t insert_idx = atomicAdd((unsigned long long*) matches,(unsigned long long) 1);

    mat_table[insert_idx] = MAT_TUPLE(&hashmap.table[res], &probe_table[r_id]);

}

template <typename BUILD_TUPLE, typename PROBE_TUPLE, typename BUILD_KEY, typename PROBE_KEY>
__global__ void hash_join_rids(HashMap<BUILD_TUPLE, uint64_t> hashmap, PROBE_TUPLE *probe_table,
                                      BUILD_KEY build_key, PROBE_KEY probe_key, uint64_t num,
                                      uint64_t *res_rids, uint64_t *matches){

    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (r_id >= num) return;

    auto res = hashmap.probe(probe_key(&probe_table[r_id]), build_key);

    if (res == (uint64_t) -1) return;

    uint64_t insert_idx = atomicAdd((unsigned long long*) matches,(unsigned long long) 1);

    res_rids[insert_idx] = r_id;
}

/**
 * Functions useful mainly for columnar layout.
 */

/**
 * Build a hash map after filtering on variable columns.
 * Base case, no columns left to filter.
 */
template <typename BUILD_TUPLE, typename BUILD_KEY>
__device__ inline void hash_map_build_pred_dev(HashMap<BUILD_TUPLE, uint64_t> &hashmap, uint64_t num_tuples,
                                               BUILD_KEY build_key){
    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (r_id >= num_tuples) return;
    hashmap.insert(r_id, build_key(&hashmap.table[r_id]));
    return;
}
/**
 * General case, filter first column, call recursively.
 */
template <typename BUILD_TUPLE, typename BUILD_KEY, typename PRED_COL, typename PREDICATE,
                                                    template<typename, typename> typename FIRST_PRED,
                                                    typename... REST>
__device__ inline void hash_map_build_pred_dev(HashMap<BUILD_TUPLE, uint64_t> &hashmap, uint64_t num_tuples,
                                                BUILD_KEY build_key,
                                                FIRST_PRED<PRED_COL, PREDICATE> pred_info, REST... rest){
    uint32_t r_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (r_id >= num_tuples) return;
    if (!pred_info.pred(&pred_info.col[r_id])) return;

    hash_map_build_pred_dev(hashmap,num_tuples,build_key,rest...);

}
template <typename BUILD_TUPLE, typename BUILD_KEY, typename... ALL>
__global__ void hash_map_build_pred_other(HashMap<BUILD_TUPLE, uint64_t> hashmap, uint64_t num_tuples,
                                          BUILD_KEY build_key, ALL... all){
    hash_map_build_pred_dev(hashmap,num_tuples,build_key, all...);
}



} // end of namespace