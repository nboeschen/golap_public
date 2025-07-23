#pragma once

#include <string.h>

#include "util.hpp"
#include "dev_structs.cuh"
#include "join.cuh"
#include "../data/SSB_def.hpp"
#include "query_common.cuh"

struct Q3Customer{
    decltype(Customer::c_key) c_key;
    decltype(Customer::c_city) c_city;
    decltype(Customer::c_nation) c_nation;
    decltype(Customer::c_region) c_region;
    friend std::ostream& operator<<(std::ostream &out, Q3Customer const& obj){
        out << obj.c_key << "," << obj.c_city.d<<","<<obj.c_nation.d<<","<<obj.c_region.d;
        return out;
    }
};

struct Q3CustomerKey{
    __host__ __device__
    uint64_t operator()(Q3Customer *t){
        return t->c_key;
    }
};

struct Q3NATION_GROUP{
    decltype(Customer::c_nation) c_nation;
    decltype(Supplier::s_nation) s_nation;
    decltype(Date::d_year) d_year;

    __host__ __device__ inline
    uint64_t hash() const {
        uint64_t hash = (*(uint32_t*)c_nation.d);
        hash <<= 32;
        hash |= (*(uint32_t*)s_nation.d);
        hash ^= d_year;
        return hash;
    }

    __host__ __device__ inline
    bool operator==(const Q3NATION_GROUP& other) const {
        for(int i = 0; i<16; ++i){
            if(c_nation.d[i] != other.c_nation.d[i]) return false;
            if(c_nation.d[i] == '\0') break;
        }
        for(int i = 0; i<16; ++i){
            if(s_nation.d[i] != other.s_nation.d[i]) return false;
            if(s_nation.d[i] == '\0') break;
        }
        return d_year == other.d_year;
    }
    friend std::ostream& operator<<(std::ostream &out, Q3NATION_GROUP const& obj){
        out << obj.c_nation << "," << obj.s_nation << "," << obj.d_year;
        return out;
    }
};

struct Q3CITY_GROUP{
    decltype(Customer::c_city) c_city;
    decltype(Supplier::s_city) s_city;
    decltype(Date::d_year) d_year;

    __host__ __device__ inline
    uint64_t hash() const {
        uint64_t hash = (*(uint32_t*)c_city.d);
        hash <<= 32;
        hash |= (*(uint32_t*)s_city.d);
        hash ^= d_year;
        return hash;
    }

    __host__ __device__ inline
    bool operator==(const Q3CITY_GROUP& other) const {
        for(int i = 0; i<11; ++i){
            if(c_city.d[i] != other.c_city.d[i]) return false;
            if(c_city.d[i] == '\0') break;
        }
        for(int i = 0; i<11; ++i){
            if(s_city.d[i] != other.s_city.d[i]) return false;
            if(s_city.d[i] == '\0') break;
        }
        return d_year == other.d_year;
    }
    friend std::ostream& operator<<(std::ostream &out, Q3CITY_GROUP const& obj){
        out << obj.c_city << "," << obj.s_city << "," << obj.d_year;
        return out;
    }
};

template <typename PRED_COL, typename PREDICATE, template<typename, typename> typename PRED>
__global__ void pipeline_customer_q3(golap::HashMap<Q3Customer, uint64_t> customer_hashmap,
                                      decltype(Customer::c_key) *c_key,
                                      decltype(Customer::c_city) *c_city,
                                      decltype(Customer::c_nation) *c_nation,
                                      decltype(Customer::c_region) *c_region,
                                      PRED<PRED_COL, PREDICATE> pred_info,
                                      Q3Customer *customer_buffer,
                                      uint64_t *customer_counter,
                                      uint64_t num
                                      ){
    uint64_t r_id,cur_idx;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        if (!pred_info.pred(&pred_info.col[r_id])) continue;
        
        cur_idx = atomicAdd((unsigned long long*)customer_counter, (unsigned long long) 1);

        customer_buffer[cur_idx] = Q3Customer{c_key[r_id],c_city[r_id],c_nation[r_id],c_region[r_id]};
        customer_hashmap.insert(cur_idx, c_key[r_id]);
    }
}

template <typename GROUP>
__global__ void pipeline_q3(golap::HashMap<Q3Customer, uint64_t> customer_hashmap,
                                      golap::HashMap<decltype(Supplier::s_key), uint64_t> supplier_hashmap,
                                      golap::HashMap<decltype(Date::d_key), uint64_t> date_hashmap,
                                      golap::HashAggregate<GROUP, uint64_t> hashagg,
                                      decltype(Lineorder::lo_custkey) *lo_custkey,
                                      decltype(Lineorder::lo_suppkey) *lo_suppkey,
                                      decltype(Lineorder::lo_orderdate) *lo_orderdate,
                                      decltype(Lineorder::lo_revenue) *lo_revenue,
                                      // decltype(Customer::c_nation) *c_nation,
                                      // decltype(Customer::c_city) *c_city,
                                      decltype(Supplier::s_nation) *s_nation,
                                      decltype(Supplier::s_city) *s_city,
                                      decltype(Date::d_year) *d_year,
                                      Q3Customer *customer_dev,
                                      SumAgg agg_func,
                                      uint64_t num,
                                      double *join_cycles,
                                      double *aggregate_cycles
                                      ){

    uint64_t r_id, cust_match,supp_match,date_match;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        // uint64_t startt = clock64();
        cust_match = customer_hashmap.probe(lo_custkey[r_id], Q3CustomerKey());
        if(cust_match == (uint64_t) -1) continue;

        supp_match = supplier_hashmap.probe(lo_suppkey[r_id]);
        if(supp_match == (uint64_t) -1) continue;

        date_match = date_hashmap.probe(lo_orderdate[r_id]);
        if(date_match == (uint64_t) -1) continue;

        // uint64_t joint = clock64();


        if constexpr(std::is_same_v<GROUP,Q3NATION_GROUP>){
            // nation cols
            hashagg.add(Q3NATION_GROUP{s_nation[supp_match], customer_dev[cust_match].c_nation, d_year[date_match]}, lo_revenue[r_id], agg_func);
        }else{
            // city cols
            hashagg.add(Q3CITY_GROUP{s_city[supp_match], customer_dev[cust_match].c_city, d_year[date_match]}, lo_revenue[r_id], agg_func);
        }

        // uint64_t endt = clock64();

        // atomicAdd(join_cycles, (joint-startt)/(double)(gridDim.x * blockDim.x));
        // atomicAdd(aggregate_cycles, (endt-joint)/(double)(gridDim.x * blockDim.x));
    }
}