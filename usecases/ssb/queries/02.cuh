#pragma once

#include "util.hpp"
#include "dev_structs.cuh"
#include "join.cuh"
#include "../data/SSB_def.hpp"
#include "query_common.cuh"


struct PartPred1{
    __host__ __device__ inline
    bool operator()(decltype(Part::p_category) *p_category){
        static const char cond[] = "MFGR#12";
        for(int i = 0; i<7; ++i){
            char cur = (p_category->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (p_category->d)[7] == '\0';
    }
};
struct PartPred2{
    __host__ __device__ inline
    bool operator()(decltype(Part::p_brand1) *p_brand1){
        static const char cond[] = "MFGR#222";
        for(int i = 0; i<8; ++i){
            char cur = (p_brand1->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (p_brand1->d)[8] >= '1' && (p_brand1->d)[8] <= '8' && (p_brand1->d)[9] == '\0';
    }
};
struct PartPred3{
    __host__ __device__ inline
    bool operator()(decltype(Part::p_brand1) *p_brand1){
        static const char cond[] = "MFGR#2239";
        for(int i = 0; i<9; ++i){
            char cur = (p_brand1->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (p_brand1->d)[9] == '\0';
    }
};


struct YEARBRAND_GROUP{
    decltype(Date::d_year) d_year;
    decltype(Part::p_brand1) p_brand1;

    __host__ __device__ inline
    uint64_t hash() const{
        return *((uint64_t*)&p_brand1) ^ d_year;
    }

    __host__ __device__ inline
    bool operator==(const YEARBRAND_GROUP& other) const {
        for(int i = 0; i<9; ++i){
            char cur = (p_brand1.d)[i];
            if(cur == '\0' || cur != other.p_brand1.d[i]) return false;
        }
        return d_year == other.d_year;
    }
};

struct YEARMONTH_GROUP{
    decltype(Date::d_year) d_year;
    decltype(Date::d_monthnuminyear) d_monthnuminyear;

    __host__ __device__ inline
    uint64_t hash() const{
        return (((uint64_t)d_monthnuminyear)<<16) | d_year;
    }

    __host__ __device__ inline
    bool operator==(const YEARMONTH_GROUP& other) const {
        return d_year == other.d_year && d_monthnuminyear == other.d_monthnuminyear;
    }
};

struct YEARMONTHWEEKSUPP_GROUP{
    decltype(Date::d_year) d_year;
    decltype(Date::d_monthnuminyear) d_monthnuminyear;
    decltype(Date::d_weeknuminyear) d_weeknuminyear;
    decltype(Lineorder::lo_suppkey) lo_suppkey;

    __host__ __device__ inline
    uint64_t hash() const{
        return lo_suppkey ^ ((((uint64_t)d_weeknuminyear)<<32) | (((uint64_t)d_monthnuminyear)<<16) | d_year);
    }

    __host__ __device__ inline
    bool operator==(const YEARMONTHWEEKSUPP_GROUP& other) const {
        return d_year == other.d_year && d_monthnuminyear == other.d_monthnuminyear
                && d_weeknuminyear == other.d_weeknuminyear && lo_suppkey == other.lo_suppkey;
    }
};


enum class Q2JO{
    PSD, SPD, SDP, DPS, DSP
};

template <Q2JO VARIANT = Q2JO::PSD>
__global__ void pipeline_q2(golap::HashMap<decltype(Part::p_key), uint64_t> part_hashmap,
                                      golap::HashMap<decltype(Supplier::s_key), uint64_t> supplier_hashmap,
                                      golap::HashMap<decltype(Date::d_key), uint64_t> date_hashmap,
                                      golap::HashAggregate<YEARBRAND_GROUP, uint64_t> hashagg,
                                      decltype(Lineorder::lo_partkey) *lo_partkey,
                                      decltype(Lineorder::lo_suppkey) *lo_suppkey,
                                      decltype(Lineorder::lo_orderdate) *lo_orderdate,
                                      decltype(Lineorder::lo_revenue) *lo_revenue,
                                      decltype(Part::p_brand1) *p_brand1,
                                      decltype(Date::d_year) *d_year,
                                      SumAgg agg_func,
                                      uint64_t num
                                      // ,uint64_t *raw_lineorder,
                                      ,uint64_t *after_part
                                      // uint64_t *after_supp,
                                      // uint64_t *after_date
                                      ){

    uint64_t r_id, part_match, supplier_match, date_match;
    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        if constexpr (VARIANT == Q2JO::PSD){
            part_match = part_hashmap.probe(lo_partkey[r_id]);
            if (part_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_part,1);

            supplier_match = supplier_hashmap.probe(lo_suppkey[r_id]);
            if (supplier_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_supp,1);

            date_match = date_hashmap.probe(lo_orderdate[r_id]);
            // if (date_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_date,1);
        }else if constexpr(VARIANT == Q2JO::SPD){
            supplier_match = supplier_hashmap.probe(lo_suppkey[r_id]);
            if (supplier_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_supp,1);

            part_match = part_hashmap.probe(lo_partkey[r_id]);
            if (part_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_part,1);

            date_match = date_hashmap.probe(lo_orderdate[r_id]);
            // if (date_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_date,1);
        }else if constexpr(VARIANT == Q2JO::SDP){
            supplier_match = supplier_hashmap.probe(lo_suppkey[r_id]);
            if (supplier_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_supp,1);

            date_match = date_hashmap.probe(lo_orderdate[r_id]);
            // if (date_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_date,1);

            part_match = part_hashmap.probe(lo_partkey[r_id]);
            if (part_match == (uint64_t) -1) continue;
            atomicAdd((unsigned long long*)after_part,1);

        }else if constexpr (VARIANT == Q2JO::DSP){
            date_match = date_hashmap.probe(lo_orderdate[r_id]);
            // if (date_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_date,1);

            supplier_match = supplier_hashmap.probe(lo_suppkey[r_id]);
            if (supplier_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_supp,1);

            part_match = part_hashmap.probe(lo_partkey[r_id]);
            if (part_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_part,1);

        }else if constexpr (VARIANT == Q2JO::DPS){
            date_match = date_hashmap.probe(lo_orderdate[r_id]);
            // if (date_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_date,1);

            part_match = part_hashmap.probe(lo_partkey[r_id]);
            if (part_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_part,1);

            supplier_match = supplier_hashmap.probe(lo_suppkey[r_id]);
            if (supplier_match == (uint64_t) -1) continue;
            // atomicAdd((unsigned long long*)after_supp,1);
        }


        hashagg.add(YEARBRAND_GROUP{d_year[date_match], p_brand1[part_match]}, lo_revenue[r_id], agg_func);
    }

}


__global__ void pipeline_q2_pre_aggr(
                                      golap::HashMap<decltype(Supplier::s_key), uint64_t> supplier_hashmap,
                                      golap::HashAggregate<YEARMONTH_GROUP, uint64_t> hashagg,
                                      decltype(Lineorder::lo_suppkey) *lo_suppkey,
                                      uint64_t *revenue_per_week,
                                      decltype(Date::d_year) *d_year,
                                      decltype(Date::d_monthnuminyear) *d_monthnuminyear,
                                      SumAgg agg_func,
                                      uint64_t num
                                      // ,uint64_t *raw_lineorder,
                                      // uint64_t *after_supp,
                                      ){

    uint64_t r_id, supplier_match;
    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();

        supplier_match = supplier_hashmap.probe(lo_suppkey[r_id]);
        if (supplier_match == (uint64_t) -1) continue;
        // atomicAdd((unsigned long long*)after_supp,1);

        hashagg.add(YEARMONTH_GROUP{d_year[r_id], d_monthnuminyear[r_id]}, revenue_per_week[r_id], agg_func);
    }

}