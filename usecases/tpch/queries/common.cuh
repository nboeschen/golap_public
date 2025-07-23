#pragma once

struct FLAG_STATUS{
    decltype(Lineitem::l_returnflag) l_returnflag;
    decltype(Lineitem::l_linestatus) l_linestatus;

    __host__ __device__ inline
    uint64_t hash() const{
        return (uint64_t)((((uint64_t)l_returnflag)<<8) | l_linestatus);
    }

    __host__ __device__ inline
    bool operator==(const FLAG_STATUS& other) const {
        return (l_returnflag == other.l_returnflag) && (l_linestatus == other.l_linestatus);
    }
};

struct KEY_DATE_PRIO{
    decltype(Lineitem::l_orderkey) l_orderkey;
    decltype(Order::o_orderdate) o_orderdate;
    decltype(Order::o_shippriority) o_shippriority;

    __host__ __device__ inline
    uint64_t hash() const{
        return l_orderkey ^ o_orderdate.t ^ (uint64_t) o_shippriority;
    }

    __host__ __device__ inline
    bool operator==(const KEY_DATE_PRIO& other) const {
        return (l_orderkey == other.l_orderkey) && (o_orderdate == other.o_orderdate) && (o_shippriority == other.o_shippriority);
    }
};

struct NATION_GROUP{
    // decltype(Nation::n_name) n_name;
    decltype(Customer::c_nationkey) c_nationkey;

    __host__ __device__ inline
    uint64_t hash() const{
        return c_nationkey;
    }

    __host__ __device__ inline
    bool operator==(const NATION_GROUP& other) const {
        return c_nationkey == other.c_nationkey;
    }
};

struct SumByte{
    __device__ inline
    void operator()(uint64_t *agg, uint8_t local) {
        atomicAdd((unsigned long long*) agg, (unsigned long long)local);
    }
};
struct SumAgg{
    __device__ inline
    void operator()(uint64_t *agg, uint64_t local) {
        atomicAdd((unsigned long long*) agg, (unsigned long long)local);
    }
};

struct MktSegmentPred{
    __host__ __device__ inline
    bool operator()(decltype(Customer::c_mktsegment) *c_mktsegment){
        static const char cond[] = "BUILDING";
        for(int i = 0; i<8; ++i){
            char cur = (c_mktsegment->d)[i];
            if(cur == '\0' || cur != cond[i]) return false;
        }
        return (c_mktsegment->d)[8] == '\0';
    }
};

struct NationKeyPred{
    __host__ __device__ inline
    bool operator()(decltype(Customer::c_nationkey) *c_nationkey){
        return (*c_nationkey == 8) || (*c_nationkey == 9) || (*c_nationkey == 12)
            || (*c_nationkey == 18) || (*c_nationkey == 21);
    }
};
