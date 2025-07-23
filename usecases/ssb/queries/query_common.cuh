#pragma once
#include <unordered_set>

#include "comp.cuh"
#include "storage.hpp"
#include "access.hpp"


template <typename T>
struct HASHER{
  std::size_t operator()(const T& k) const{
    return k.hash();
  }
};


struct TruePred {
    __host__ __device__ inline
    bool operator()(char* ptr){
        return true;
    }
};

struct SumAgg{
    __device__ inline
    void operator()(uint64_t *agg, uint64_t local) {
        atomicAdd((unsigned long long*) agg, (unsigned long long)local);
    }
};

struct SignedSumAgg{
    __device__ inline
    void operator()(int64_t *agg, int64_t local) {
        atomicAdd((unsigned long long*) agg, (unsigned long long)local);
    }
};

struct RegionAmericaPred {
    __host__ __device__ inline
    bool operator()(util::Padded<char[13]> *s_region){
        static const char cond[] = "AMERICA";
        for(int i = 0; i<7; ++i){
            char cur = (s_region->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (s_region->d)[7] == '\0';
    }
};

struct NationUSPred {
    __host__  __device__ inline
    bool operator()(util::Padded<char[16]> *nation){
        static const char cond[] = "UNITED STATES";
        for(int i = 0; i<13; ++i){
            char cur = (nation->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (nation->d)[13] == '\0';
    }
};
struct NationChinaPred {
    __host__  __device__ inline
    bool operator()(util::Padded<char[16]> *nation){
        static const char cond[] = "CHINA";
        for(int i = 0; i<5; ++i){
            char cur = (nation->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (nation->d)[5] == '\0';
    }
};

struct CityKIPred {
    __host__  __device__ inline
    bool operator()(util::Padded<char[11]> *city){
        static const char cond[] = "UNITED KI";
        for(int i = 0; i<9; ++i){
            char cur = (city->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return ((city->d)[9] == '1' || (city->d)[9] == '5') && (city->d)[10] == '\0';
    }
};
struct YEARMONTHDec1997{
    __host__ __device__ inline
    bool operator()(util::Padded<char[8]> *yearmonth){
        static const char cond[] = "Dec1997";
        for(int i = 0; i<7; ++i){
            char cur = (yearmonth->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (yearmonth->d)[7] == '\0';
    }
};
struct RegionAsiaPred {
    __host__  __device__ inline
    bool operator()(util::Padded<char[13]> *s_region){
        static const char cond[] = "ASIA";
        for(int i = 0; i<4; ++i){
            char cur = (s_region->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (s_region->d)[4] == '\0';
    }
};
struct RegionAPred {
    __host__  __device__ inline
    bool operator()(util::Padded<char[13]> *s_region){
        return (s_region->d)[0] == 'A';
    }
};
struct HostSum{
    __host__
    void operator()(std::atomic<uint64_t> *agg, uint64_t val){
        agg->fetch_add(val, std::memory_order_relaxed);
    }
};
struct RegionEuropePred {
    __host__ __device__ inline
    bool operator()(util::Padded<char[13]> *s_region){
        static const char cond[] = "EUROPE";
        for(int i = 0; i<6; ++i){
            char cur = (s_region->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (s_region->d)[6] == '\0';
    }
};

template <decltype(Date::d_year) LO, decltype(Date::d_year) HI>
struct DATE_YEAR{
    __host__ __device__ inline
    bool operator()(decltype(Date::d_year) *d_year){
        return *d_year >= LO && *d_year <= HI;
    }
};
using DATE_9297 = DATE_YEAR<1992,1997>;
using DATE_9597 = DATE_YEAR<1995,1997>;
using DATE_9798 = DATE_YEAR<1997,1998>;

struct MFGRPred{
    __host__ __device__ inline
    bool operator()(decltype(Part::p_mfgr) *p_mfgr){
        static const char cond[] = "MFGR#";
        for(int i = 0; i<5; ++i){
            char cur = (p_mfgr->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return ((p_mfgr->d)[5] == '1' || (p_mfgr->d)[5] == '2') && (p_mfgr->d)[6] == '\0';
    }
};
struct Category14Pred{
    __host__ __device__ inline
    bool operator()(decltype(Part::p_category) *p_category){
        static const char cond[] = "MFGR#14";
        for(int i = 0; i<7; ++i){
            char cur = (p_category->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (p_category->d)[7] == '\0';
    }
};

