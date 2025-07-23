#pragma once

#include "../data/Taxi_def.hpp"


struct SumAgg{
    __device__ inline
    void operator()(uint64_t *agg, uint64_t local) {
        atomicAdd((unsigned long long*) agg, (unsigned long long)local);
    }
};
struct FloatSum{
    __device__ inline
    void operator()(double *agg, double local) {
        unsigned long long int* address_as_ull = (unsigned long long int*)agg;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(local + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
    }
};



struct MONTH_GROUP{
    int16_t month;

    __host__ __device__ inline
    uint64_t hash(){
        return (uint64_t)month;
    }

    __host__ __device__ inline
    bool operator==(const MONTH_GROUP& other){
        return month == other.month;
    }
};

struct DAY_GROUP{
    int16_t day;

    __host__ __device__ inline
    uint64_t hash(){
        return (uint64_t)day;
    }

    __host__ __device__ inline
    bool operator==(const DAY_GROUP& other){
        return day == other.day;
    }
};

struct HostSum{
    __host__
    void operator()(std::atomic<uint64_t> *agg, uint64_t val){
        agg->fetch_add(val, std::memory_order_relaxed);
    }
};

struct HostFloatSum{
    __host__
    void operator()(std::atomic<double> *agg, double val){
        double expected = agg->load();
        while(!atomic_compare_exchange_weak(agg, &expected, expected + val));
        // agg->fetch_add(val, std::memory_order_relaxed); // cpp20
    }
};


/**
 * This is a very GPU unfriendly computation, but in that sense models more compute heavy query processing.
 * https://github.com/pts/minilibc686/blob/master/fyi/c_gmtime.c
 */
__host__ __device__ int extract_month(time_t &epoch_seconds){
    time_t t = epoch_seconds / 86400;
    unsigned hms = epoch_seconds % 86400;  /* -86399 <= hms <= 86399. This needs sizeof(int) >= 4. */
    time_t c, f;
    unsigned yday;  /* 0 <= yday <= 426. Also fits to an `unsigned short', but `int' is faster. */
    unsigned a;  /* 0 <= a <= 2133. Also fits to an `unsigned short', but `int' is faster. */
    if ((int)hms < 0) { --t;}  /* Fix quotient and negative remainder if epoch_seconds was negative (i.e. before year 1970 CE). */
    /* Now: -24856 <= t <= 24855. */
    if constexpr (sizeof(time_t) > 4) {  /* Optimization. For int32_t, this would keep t intact, so we won't have to do it. This produces unreachable code. */
        f = (t + 4) % 7;
        if (f < 0) f += 7;  /* Fix negative remainder if (t + 4) was negative. */
        /* Now 0 <= f <= 6. */
        c = (t << 2) + 102032;
        f = c / 146097;
        if (c % 146097 < 0) --f;  /* Fix negative remainder if c was negative. */
        --f;
        t += f;
        f >>= 2;
        t -= f;
        f = (t << 2) + 102035;
        c = f / 1461;
        if (f % 1461 < 0) --c;  /* Fix negative remainder if f was negative. */
    } else {
        /* Now: -24856 <= t <= 24855. */
        c = ((t << 2) + 102035) / 1461;
    }
    yday = t - 365 * c - (c >> 2) + 25568;
    /* Now: 0 <= yday <= 425. */
    a = (yday * 5 + 8)/153;
    a %= 12;
    return a;
}

__host__ __device__ int extract_day(time_t &epoch_seconds){
    int res;
    time_t t = epoch_seconds / 86400;
    unsigned hms = epoch_seconds % 86400;  /* -86399 <= hms <= 86399. This needs sizeof(int) >= 4. */
    time_t f;
    if ((int)hms < 0) { --t;}  /* Fix quotient and negative remainder if epoch_seconds was negative (i.e. before year 1970 CE). */

    if constexpr (sizeof(time_t) > 4) {  /* Optimization. For int32_t, this would keep t intact, so we won't have to do it. This produces unreachable code. */
        f = (t + 4) % 7;
        if (f < 0) f += 7;  /* Fix negative remainder if (t + 4) was negative. */
        /* Now 0 <= f <= 6. */
        res = f;
    } else {
        res = (t + 24861) % 7;  /* t + 24861 >= 0. */
    }
    return res;
}

