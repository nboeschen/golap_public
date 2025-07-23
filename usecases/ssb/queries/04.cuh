#pragma once

#include <string.h>

#include "util.hpp"
#include "dev_structs.cuh"
#include "join.cuh"
#include "../data/SSB_def.hpp"
#include "query_common.cuh"


struct Q41_GROUP{
    decltype(Date::d_year) d_year;
    decltype(Customer::c_nation) c_nation;

    __host__ __device__ inline
    uint64_t hash() const{
        uint64_t hash = (*(uint32_t*)c_nation.d);
        hash <<= 32;
        hash ^= d_year;
        return hash;
    }
    __host__ __device__ inline
    bool operator==(const Q41_GROUP& other) const{

        for(int i = 0; i<16; ++i){
            if(c_nation.d[i] != other.c_nation.d[i]) return false;
            if(c_nation.d[i] == '\0') break;
        }
        return d_year == other.d_year;
    }
    friend std::ostream& operator<<(std::ostream &out, Q41_GROUP const& obj){
        out << obj.d_year << "," << obj.c_nation;
        return out;
    }
};

struct Q42_GROUP{
    decltype(Date::d_year) d_year;
    decltype(Supplier::s_nation) s_nation;
    decltype(Part::p_category) p_category;

    __host__ __device__ inline
    uint64_t hash() const{
        uint64_t hash = (*(uint32_t*)s_nation.d);
        hash <<= 32;
        hash |= (*(uint32_t*)p_category.d);
        hash ^= d_year;
        return hash;
    }
    __host__ __device__ inline
    bool operator==(const Q42_GROUP& other) const{
        for(int i = 0; i<16; ++i){
            if(s_nation.d[i] != other.s_nation.d[i]) return false;
            if(s_nation.d[i] == '\0') break;
        }
        for(int i = 0; i<8; ++i){
            if(p_category.d[i] != other.p_category.d[i]) return false;
            if(p_category.d[i] == '\0') break;
        }
        return d_year == other.d_year;
    }
    friend std::ostream& operator<<(std::ostream &out, Q42_GROUP const& obj){
        out << obj.d_year << "," << obj.s_nation<<","<<obj.p_category;
        return out;
    }
};
struct Q43_GROUP{
    decltype(Date::d_year) d_year;
    decltype(Supplier::s_city) s_city;
    decltype(Part::p_brand1) p_brand1;

    __host__ __device__ inline
    uint64_t hash() const{
        uint64_t hash = (*(uint32_t*)s_city.d);
        hash <<= 32;
        hash |= (*(uint32_t*)p_brand1.d);
        hash ^= d_year;
        return hash;
    }
    __host__ __device__ inline
    bool operator==(const Q43_GROUP& other) const {
        for(int i = 0; i<11; ++i){
            if(s_city.d[i] != other.s_city.d[i]) return false;
            if(s_city.d[i] == '\0') break;
        }
        for(int i = 0; i<10; ++i){
            if(p_brand1.d[i] != other.p_brand1.d[i]) return false;
            if(p_brand1.d[i] == '\0') break;
        }
        return d_year == other.d_year;
    }
    friend std::ostream& operator<<(std::ostream &out, Q43_GROUP const& obj){
        out << obj.d_year << "," << obj.s_city<<","<<obj.p_brand1;
        return out;
    }
};

template <typename GROUP>
__global__ void pipeline_q4(golap::HashMap<decltype(Customer::c_key), uint64_t> customer_hashmap,
                                golap::HashMap<decltype(Supplier::s_key), uint64_t> supplier_hashmap,
                                golap::HashMap<decltype(Date::d_key), uint64_t> date_hashmap,
                                golap::HashMap<decltype(Part::p_key), uint64_t> part_hashmap,
                                golap::HashAggregate<GROUP, uint64_t> hashagg,
                                decltype(Lineorder::lo_custkey) *lo_custkey,
                                decltype(Lineorder::lo_suppkey) *lo_suppkey,
                                decltype(Lineorder::lo_orderdate) *lo_orderdate,
                                decltype(Lineorder::lo_partkey) *lo_partkey,
                                decltype(Lineorder::lo_revenue) *lo_revenue,
                                decltype(Lineorder::lo_supplycost) *lo_supplycost,
                                decltype(Customer::c_nation) *c_nation,
                                decltype(Customer::c_region) *c_region,
                                decltype(Supplier::s_nation) *s_nation,
                                decltype(Supplier::s_region) *s_region,
                                decltype(Supplier::s_city) *s_city,
                                decltype(Date::d_year) *d_year,
                                decltype(Part::p_brand1) *p_brand1,
                                decltype(Part::p_category) *p_category,
                                SumAgg agg_func,
                                uint64_t num
                                // double *join_cycles,
                                // double *aggregate_cycles
                                ){

    uint64_t r_id,cust_match,supp_match,date_match,part_match,local_sum;
    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        supp_match = supplier_hashmap.probe(lo_suppkey[r_id]);
        if(supp_match == (uint64_t) -1) continue;

        cust_match = customer_hashmap.probe(lo_custkey[r_id]);
        if(cust_match == (uint64_t) -1) continue;

        part_match = part_hashmap.probe(lo_partkey[r_id]);
        if(part_match == (uint64_t) -1) continue;

        date_match = date_hashmap.probe(lo_orderdate[r_id]);
        if(date_match == (uint64_t) -1) continue;


        local_sum = lo_revenue[r_id] - lo_supplycost[r_id];

        if constexpr(std::is_same_v<GROUP,Q41_GROUP>){
            hashagg.add(Q41_GROUP{d_year[date_match], c_nation[cust_match]}, local_sum, agg_func);
        } else if constexpr(std::is_same_v<GROUP,Q42_GROUP>){
            hashagg.add(Q42_GROUP{d_year[date_match], s_nation[supp_match], p_category[part_match]}, local_sum, agg_func);
        } else if constexpr(std::is_same_v<GROUP,Q43_GROUP>){
            hashagg.add(Q43_GROUP{d_year[date_match], s_city[supp_match], p_brand1[part_match]}, local_sum, agg_func);
        }
    }

}
