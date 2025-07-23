#pragma once

#include <atomic>

#include "util.hpp"

namespace golap{

template <typename BUILD_TUPLE, typename KEY_TYPE = uint64_t>
class HostHashMap{
public:
    /**
     * @brief Construct a new HostHashMap object
     * FUTURE: This works well if a full column/table is added without r_id "holes".
     * If there's predicates, it would be beneficial to allow for smaller tables, where
     * r_ids are not directly translatable to array indexes.
     * @param ht_size size of hash table
     * @param table pointer to dense table with tuples
     */
    HostHashMap(uint64_t ht_size, BUILD_TUPLE* table) : ht_size(ht_size), table(table){
        
        RIDChains = new uint64_t[ht_size]{};
        lastMatchingRIDs = new std::atomic<uint64_t>[ht_size]{};
    }

    /**
     * Constructor for already prepared pointers:
     */
    // HostHashMap(uint64_t ht_size, BUILD_TUPLE* table, uint64_t *RIDChains,
    //         uint64_t *lastMatchingRIDs):ht_size(ht_size), table(table),
    //         RIDChains(RIDChains),lastMatchingRIDs(lastMatchingRIDs){
    //     is_orig = false;
    // }

    ~HostHashMap(){
        if (!is_orig) return;
        delete[] RIDChains;
        delete[] lastMatchingRIDs;
    }

    /**
     * Insert into the hashmap, using tuples from "table".
     * @param rid   the row id of the inserted tuple in "table"
     */
    __host__ void insert(uint64_t rid, KEY_TYPE key){

        uint64_t hash = key % ht_size;
        // uint64_t prev = atomicExch((unsigned long long*) &lastMatchingRIDs[hash], (unsigned long long) rid+1);
        uint64_t prev = lastMatchingRIDs[hash].exchange(rid+1, std::memory_order_relaxed);
        // if (prev != 0){
        //     printf("collision on hash %llu, key %llu prev key %llu\n", hash, (unsigned long long)tuple->key(), (unsigned long long)(table[prev-1].key()));
        // }
        RIDChains[rid] = prev;
        filled.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * Probe into this hashmap, returning the rid of the found tuple in "table", or -1
     */
    template <typename BUILD_TUPLE_KEY = DirectKey<BUILD_TUPLE>>
    uint64_t probe(KEY_TYPE key, BUILD_TUPLE_KEY build_tuple_key = DirectKey<BUILD_TUPLE>()){
        uint64_t hash = key % ht_size;
        for(uint64_t hit = lastMatchingRIDs[hash].load(std::memory_order_relaxed); hit > 0; hit = RIDChains[hit-1]){
            if(key == build_tuple_key(&table[hit-1])){
                return hit-1;
            }
        }
        return (uint64_t) -1;
    }


    uint64_t *RIDChains;
    std::atomic<uint64_t> *lastMatchingRIDs;
    std::atomic<uint64_t> filled{};
    uint64_t ht_size;
    BUILD_TUPLE *table;
    bool is_orig = true;
};



template <typename GROUP_ATTR, typename AGG_TYPE>
class HostHashAggregate{
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
    HostHashAggregate(uint64_t num_groups, GROUP_ATTR* group_attr, AGG_TYPE* agg_res) : num_groups(num_groups), group_attr(group_attr), agg_res(agg_res){
        reserved = new std::atomic<uint32_t>[num_groups]{0};
        wrote_group = new std::atomic<uint32_t>[num_groups]{0};
    }
    
    ~HostHashAggregate(){
        delete[] reserved;
        delete[] wrote_group;
    }

    template <typename AGG_FUNC>
    void add(GROUP_ATTR group, AGG_TYPE agg_val, AGG_FUNC agg_func){

        uint64_t slot = group.hash() % num_groups;
        bool changed;
        uint32_t constant_zero = 0;

        // find a slot for the group
        while(true){
            constant_zero = 0;
            changed = reserved[slot].compare_exchange_strong(constant_zero,(uint32_t)1);
            if(changed){
                // case 1: slot was empty, we are the representing group
                group_attr[slot] = group;
                // memory barrier?
                wrote_group[slot] = 1;

                break;
            }

            // wait until group_attr is filled ...
            // while (reinterpret_cast<volatile uint32_t*>(wrote_group)[slot] == 0);
            while (wrote_group[slot].load() == 0) asm("pause");

            if(group_attr[slot] == group){
                // case 2: slot is filled, and its the correct group
                break;
            }
            // case 3: slot was filled, groupattr is present, but not the correct one --> linear probing

            slot = (slot+1) % num_groups;
        }
        added.fetch_add(1, std::memory_order_relaxed);
        agg_func(&agg_res[slot],agg_val);
    }

    std::atomic<uint32_t> *reserved;
    std::atomic<uint32_t> *wrote_group;
    std::atomic<uint64_t> added{0};
    uint64_t num_groups;
    GROUP_ATTR* group_attr;
    AGG_TYPE *agg_res;
    bool is_orig = true;
};




} // end of namespace