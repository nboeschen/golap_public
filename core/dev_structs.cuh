#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "mem.hpp"

namespace golap{


template <typename T>
struct DirectKey{
    __host__ __device__ inline
    T operator()(T *t){
        return *t;
    }
};


template <typename T, typename PREDICATE>
struct PredInfo{
    T* col;
    PREDICATE pred;
};

template <typename T, T C>
struct ConstPred{
    __host__ __device__ inline
    bool operator()(T* ptr){
        return *ptr == C;
    }
};


template <typename BUILD_TUPLE, typename KEY_TYPE = uint64_t>
class HashMap{
public:
    /**
     * @brief Construct a new HashMap object
     * FUTURE: This works well if a full column/table is added without r_id "holes".
     * If there's predicates, it would be beneficial to allow for smaller tables, where
     * r_ids are not directly translatable to array indexes.
     * @param ht_size size of hash table
     * @param table pointer to dense table with tuples
     */
    HashMap(uint64_t ht_size, BUILD_TUPLE* table) : ht_size(ht_size), table(table){
        checkCudaErrors(cudaMalloc(&RIDChains, ht_size * sizeof(uint64_t)));
        checkCudaErrors(cudaMalloc(&lastMatchingRIDs, ht_size * sizeof(uint64_t)));
        checkCudaErrors(cudaMemset(RIDChains, 0, ht_size * sizeof(uint64_t)));
        checkCudaErrors(cudaMemset(lastMatchingRIDs, 0, ht_size * sizeof(uint64_t)));
        checkCudaErrors(cudaMalloc(&filled, sizeof(uint64_t)));
        checkCudaErrors(cudaMemset(filled, 0, sizeof(uint64_t)));
        golap::DEVICE_ALLOCATED += ht_size * sizeof(uint64_t) *2 + sizeof(uint64_t);
    }
    /**
     * Constructor for already prepared pointers:
     */
    HashMap(uint64_t ht_size, BUILD_TUPLE* table, uint64_t *RIDChains,
            uint64_t *lastMatchingRIDs, uint64_t *filled):ht_size(ht_size), table(table),
            RIDChains(RIDChains),lastMatchingRIDs(lastMatchingRIDs),
            filled(filled){

        is_orig = false;
    }

    /**
     * This copy constructor lets you pass a HashMap object by value to a cuda kernel,
     * without double-frees of the dynamic memory.
     */
    HashMap(const HashMap& original){
        *this = original;
        is_orig = false;
    }

    ~HashMap(){
        if (!is_orig) return;
        cudaFree(RIDChains);
        cudaFree(lastMatchingRIDs);
        cudaFree(filled);
        golap::DEVICE_ALLOCATED -= ht_size * sizeof(uint64_t) *2 + sizeof(uint64_t);
    }

    /**
     * Insert into the hashmap, using tuples from "table".
     * @param rid   the row id of the inserted tuple in "table"
     * @param key   the key of the tuple
     */
    __device__ inline void insert(uint64_t rid, KEY_TYPE key){

        uint64_t hash = key % ht_size;
        uint64_t prev = atomicExch((unsigned long long*) &lastMatchingRIDs[hash], (unsigned long long) rid+1);
        // if (prev != 0){
        //     printf("collision on hash %llu, key %llu prev key %llu\n", hash, (unsigned long long)tuple->key(), (unsigned long long)(table[prev-1].key()));
        // }
        // printf("Inserted rid=%lu,key=%lu in slot=%lu, prev was %lu\n",rid,key,hash,prev);
        RIDChains[rid] = prev;
        atomicAdd((unsigned long long*) filled, (unsigned long long) 1);
    }

    /**
     * Probe into this hashmap, returning the rid of the found tuple in "table", or -1
     */
    template <typename BUILD_TUPLE_KEY = DirectKey<BUILD_TUPLE>>
    __device__ __host__ inline uint64_t probe(KEY_TYPE key, BUILD_TUPLE_KEY build_tuple_key = DirectKey<BUILD_TUPLE>()){
        uint64_t hash = key % ht_size;
        for(uint64_t hit = lastMatchingRIDs[hash]; hit > 0; hit = RIDChains[hit-1]){
            if(key == build_tuple_key(&table[hit-1])){
                return hit-1;
            }
        }
        return (uint64_t) -1;
    }

    template <typename BUILD_TUPLE_KEY = DirectKey<BUILD_TUPLE>>
    __device__ __host__ inline uint64_t pre_probe(KEY_TYPE key, BUILD_TUPLE_KEY build_tuple_key = DirectKey<BUILD_TUPLE>()){
        uint64_t hash = key % ht_size;
        if (lastMatchingRIDs[hash] > 0){
            return 1;
        }else {
            return (uint64_t) -1;
        }
    }


    uint64_t *RIDChains;
    uint64_t *lastMatchingRIDs;
    uint64_t *filled;
    uint64_t ht_size;
    BUILD_TUPLE *table;
    bool is_orig = true;
};


template <typename BUILD_TUPLE, typename KEY_TYPE>
__global__ void collect_hashmap_keys(HashMap<BUILD_TUPLE,KEY_TYPE> hash_map, KEY_TYPE *keys, uint64_t *counter){
    uint64_t idx,rid,insert;
    KEY_TYPE key;
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < hash_map.ht_size; idx += blockDim.x * gridDim.x){
        __syncwarp();
        rid = hash_map.lastMatchingRIDs[idx];
        
        while(rid != 0){
            key = DirectKey<BUILD_TUPLE>()(&hash_map.table[rid-1]);
            insert = atomicAdd((unsigned long long*) counter, (unsigned long long)1);
            keys[insert] = key;
            rid = hash_map.RIDChains[rid-1];
        }
    }
}



/**
 * Device GroupBy HashAggregate
 */
template <typename GROUP_ATTR, typename AGG_TYPE>
class HashAggregate{
public:
    /**
     * @brief Construct a new Hash Aggregate object
     * 
     * @param num_groups number of expected distinct groups
     * @param results pointer to aggregated group results (see below)
     */
    /*
    struct GROUP_ATTR{
        GROUP_ATTR(TUPLE_TYPE*){};
        GROUPBY_TYPE0 att0;
        GROUPBY_TYPE1 att1;
    };
    struct AGG_TYPE{};
    */
    HashAggregate(uint64_t num_groups, GROUP_ATTR* group_attr, AGG_TYPE* agg_res) : num_groups(num_groups), group_attr(group_attr), agg_res(agg_res){
        checkCudaErrors(cudaMalloc(&reserved, num_groups * sizeof(uint32_t)));
        checkCudaErrors(cudaMemset(reserved, 0, num_groups * sizeof(uint32_t)));
        checkCudaErrors(cudaMalloc(&wrote_group, num_groups * sizeof(uint32_t)));
        checkCudaErrors(cudaMemset(wrote_group, 0, num_groups * sizeof(uint32_t)));
        checkCudaErrors(cudaMalloc(&added, sizeof(uint64_t)));
        checkCudaErrors(cudaMemset(added, 0, sizeof(uint64_t)));
        golap::DEVICE_ALLOCATED += num_groups * sizeof(uint32_t) *2;
    }
    /**
     * Constructor for already prepared pointers:
     */
    HashAggregate(uint64_t num_groups, GROUP_ATTR* group_attr, AGG_TYPE* agg_res, uint32_t *reserved, 
                  uint32_t *wrote_group) : num_groups(num_groups), group_attr(group_attr), agg_res(agg_res),
                    reserved(reserved),wrote_group(wrote_group){
        is_orig = false;
    }

    /**
     * This copy constructor lets you pass a HashAggregate object by value to a cuda kernel,
     * without double-frees of the dynamic memory.
     */
    HashAggregate(const HashAggregate& original){
        *this = original;
        is_orig = false;
    }
    
    ~HashAggregate(){
        if (!is_orig) return;
        cudaFree(reserved);
        cudaFree(wrote_group);
        cudaFree(added);
        golap::DEVICE_ALLOCATED -= num_groups * sizeof(uint32_t) *2;
    }

    template <typename AGG_FUNC>
    __device__ inline uint64_t add(GROUP_ATTR group, AGG_TYPE agg_val, AGG_FUNC agg_func){

        uint64_t slot = group.hash() % num_groups;
        uint32_t old;

        // find a slot for the group
        while(true){
            old = (uint32_t) atomicCAS((unsigned int*) &reserved[slot],
                                       (unsigned int) 0, (unsigned int) 1);
            if(old == (unsigned int) 0){
                // case 1: slot was empty, we are the representing group
                group_attr[slot] = group;
                __threadfence();
                atomicExch(&wrote_group[slot], 1);
                // wrote_group[slot] = 1;
                break;
            }
            __threadfence();
            // wait until group_attr is filled ...
            while (reinterpret_cast<volatile uint32_t*>(wrote_group)[slot] == 0);
            __threadfence();

            if(group_attr[slot] == group){
                // case 2: slot is filled, and its the correct group
                break;
            }
            // case 3: slot was filled, groupattr is present, but not the correct one --> linear probing

            slot = (slot+1) % num_groups;
        }

        atomicAdd((unsigned long long*) added, (unsigned long long) 1);
        agg_func(&agg_res[slot],agg_val);
        return slot;
    }

    uint32_t *reserved;
    uint32_t *wrote_group;
    uint64_t *added;
    uint64_t num_groups;
    GROUP_ATTR* group_attr;
    AGG_TYPE *agg_res;
    bool is_orig = true;
};


/**
 * Device BloomFilter
 */
struct BloomFilter{
    /**
     * https://github.com/Claudenw/BloomFilter/wiki/Bloom-Filters----An-overview
     * The parameters of a bloom filter:
     * n - Expected number of items to be mapped into the filter [input]
     * p - Probability of false positive [input]
     * m - bits of the filter [derived]
     * k - number of hash functions [derived]
     */
    BloomFilter(uint64_t n, double p):n(n),p(p){
        m = util::next(ceil((n * log(p)) / log(1.0 / (pow(2.0, log(2.0))))), 8);
        // m = util::nextP2(ceil((n * log(p)) / log(1.0 / (pow(2.0, log(2.0))))));
        k = ceil(log(2.0) * m / n);

        util::Log::get().info_fmt("BloomFilter for n=%lu and p=%f: Calculated number of bits=%lu, k=%lu",n,p,m,k);

        checkCudaErrors(cudaMalloc(&state, m>>3));
        checkCudaErrors(cudaMemset(state, 0, m>>3));
        golap::DEVICE_ALLOCATED += m>>3;

    }

    BloomFilter(const BloomFilter& original){
        *this = original;
        is_orig = false;
    }
    
    ~BloomFilter(){
        if (!is_orig) return;
        cudaFree(state);
        golap::DEVICE_ALLOCATED -= m>>3;
    }
    __device__ inline uint64_t hash(uint64_t val){
        val ^= val >> 33;
        val *= 0xff51afd7ed558ccd;
        val ^= val >> 33;
        val *= 0xc4ceb9fe1a85ec53;
        val ^= val >> 33;
        // return val & (m-1);
        return val % m;
    }

    __device__ inline void map(uint64_t val){
        for (uint32_t i = 0; i<k; ++i){
            // k-th hash
            uint64_t bitidx = hash(val + i);
            uint64_t wordidx = bitidx>>5;
            bitidx &= 31; // idx in word

            atomicOr(state+wordidx, (uint32_t)(1<<bitidx));
        }
    }

    __device__ inline bool query(uint64_t val){
        auto res = true;
        for (uint32_t i = 0; i<k; ++i){
            // k-th hash
            uint64_t bitidx = hash(val + i);
            uint64_t wordidx = bitidx>>5;
            bitidx &= 31; // idx in word

            if ((state[wordidx] & (1<<bitidx)) == 0){
                // definitely not in set
                // return false; // let all thread do all k lookups less divergence?
                res = false;
            }
        }
        return res;
    }

    uint64_t n;
    double p;
    uint64_t m;
    uint32_t *state;

    bool is_orig = true;
    uint64_t k = 0;
};

} // end of namespace








