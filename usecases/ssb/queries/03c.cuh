#pragma once

struct Q3PreJoinedNation{
    decltype(Lineorder::lo_custkey) lo_custkey;
    decltype(Lineorder::lo_revenue) lo_revenue;

    decltype(Supplier::s_nation) s_nation;
    decltype(Date::d_year) d_year;
};
struct Q3PreJoinedCity{
    decltype(Lineorder::lo_custkey) lo_custkey;
    decltype(Lineorder::lo_revenue) lo_revenue;

    decltype(Supplier::s_city) s_city;
    decltype(Date::d_year) d_year;
};

template <typename Q3Joined>
__global__ void pipeline_q3a(
                             // golap::HashMap<Q3Customer, uint64_t> customer_hashmap,
                             golap::HashMap<decltype(Supplier::s_key), uint64_t> supplier_hashmap,
                                      golap::HashMap<decltype(Date::d_key), uint64_t> date_hashmap,
                                      // golap::HashAggregate<GROUP, uint64_t> hashagg,
                                      decltype(Lineorder::lo_custkey) *lo_custkey,
                                      decltype(Lineorder::lo_suppkey) *lo_suppkey,
                                      decltype(Lineorder::lo_orderdate) *lo_orderdate,
                                      decltype(Lineorder::lo_revenue) *lo_revenue,
                                      // decltype(Customer::c_nation) *c_nation,
                                      // decltype(Customer::c_city) *c_city,
                                      decltype(Supplier::s_nation) *s_nation,
                                      decltype(Supplier::s_city) *s_city,
                                      decltype(Date::d_year) *d_year,
                                      SumAgg agg_func,
                                      Q3Joined *lineorder_buffer,
                                      uint64_t *lineorder_counter,
                                      uint64_t num,
                                      double *join_cycles,
                                      double *aggregate_cycles
                                      ){

    uint64_t r_id,supp_match,date_match,insert_idx;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        // uint64_t startt = clock64();

        supp_match = supplier_hashmap.probe(lo_suppkey[r_id]);
        if(supp_match == (uint64_t) -1) continue;

        date_match = date_hashmap.probe(lo_orderdate[r_id]);
        if(date_match == (uint64_t) -1) continue;

        // uint64_t joint = clock64();
        // if(customer_hashmap.probe(lo_custkey[r_id], Q3CustomerKey()) == (uint64_t) -1) continue;

        // add to counter, and buffer
        insert_idx = atomicAdd((unsigned long long*) lineorder_counter, (unsigned long long) 1);


        if constexpr(std::is_same_v<Q3Joined,Q3PreJoinedNation>){
            // nation cols
            lineorder_buffer[insert_idx] = Q3PreJoinedNation{lo_custkey[r_id],lo_revenue[r_id],s_nation[supp_match],d_year[date_match]};
        }else if constexpr(std::is_same_v<Q3Joined,Q3PreJoinedCity>){
            // city cols
            lineorder_buffer[insert_idx] = Q3PreJoinedCity{lo_custkey[r_id],lo_revenue[r_id],s_city[supp_match],d_year[date_match]};
        }


        // uint64_t endt = clock64();

        // atomicAdd(join_cycles, (joint-startt)/(double)(gridDim.x * blockDim.x));
        // atomicAdd(aggregate_cycles, (endt-joint)/(double)(gridDim.x * blockDim.x));
    }
}

template <typename Q3Joined>
__global__ void pipeline_q3c(
                             golap::HashMap<Q3Customer, uint64_t> customer_hashmap,
                             golap::HashMap<decltype(Supplier::s_key), uint64_t> supplier_hashmap,
                                      golap::HashMap<decltype(Date::d_key), uint64_t> date_hashmap,
                                      // golap::HashAggregate<GROUP, uint64_t> hashagg,
                                      decltype(Lineorder::lo_custkey) *lo_custkey,
                                      decltype(Lineorder::lo_suppkey) *lo_suppkey,
                                      decltype(Lineorder::lo_orderdate) *lo_orderdate,
                                      decltype(Lineorder::lo_revenue) *lo_revenue,
                                      // decltype(Customer::c_nation) *c_nation,
                                      // decltype(Customer::c_city) *c_city,
                                      decltype(Supplier::s_nation) *s_nation,
                                      decltype(Supplier::s_city) *s_city,
                                      decltype(Date::d_year) *d_year,
                                      SumAgg agg_func,
                                      Q3Joined *lineorder_buffer,
                                      uint64_t *lineorder_counter,
                                      uint64_t num,
                                      double *join_cycles,
                                      double *aggregate_cycles
                                      ){

    uint64_t r_id,supp_match,date_match,insert_idx;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        // uint64_t startt = clock64();

        supp_match = supplier_hashmap.probe(lo_suppkey[r_id]);
        if(supp_match == (uint64_t) -1) continue;

        date_match = date_hashmap.probe(lo_orderdate[r_id]);
        if(date_match == (uint64_t) -1) continue;

        // uint64_t joint = clock64();
        if(customer_hashmap.pre_probe(lo_custkey[r_id], Q3CustomerKey()) == (uint64_t) -1) continue;

        // add to counter, and buffer
        insert_idx = atomicAdd((unsigned long long*) lineorder_counter, (unsigned long long) 1);


        if constexpr(std::is_same_v<Q3Joined,Q3PreJoinedNation>){
            // nation cols
            lineorder_buffer[insert_idx] = Q3PreJoinedNation{lo_custkey[r_id],lo_revenue[r_id],s_nation[supp_match],d_year[date_match]};
        }else if constexpr(std::is_same_v<Q3Joined,Q3PreJoinedCity>){
            // city cols
            lineorder_buffer[insert_idx] = Q3PreJoinedCity{lo_custkey[r_id],lo_revenue[r_id],s_city[supp_match],d_year[date_match]};
        }


        // uint64_t endt = clock64();

        // atomicAdd(join_cycles, (joint-startt)/(double)(gridDim.x * blockDim.x));
        // atomicAdd(aggregate_cycles, (endt-joint)/(double)(gridDim.x * blockDim.x));
    }
}

template <typename PRED_COL, typename PREDICATE, template<typename, typename> typename PRED>
__global__ void pipeline_customer_q3a(decltype(Customer::c_key) *c_key,
                                      decltype(Customer::c_city) *c_city,
                                      decltype(Customer::c_nation) *c_nation,
                                      decltype(Customer::c_region) *c_region,
                                      PRED<PRED_COL, PREDICATE> pred_info,
                                      Q3Customer *customer_buffer,
                                      uint64_t *customer_counter,
                                      uint64_t num
                                      ){
    uint64_t r_id, cur_idx;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        /*
        1) Predicate on the customer
        2) Put in buffer
         */
        if (!pred_info.pred(&pred_info.col[r_id])) continue;
        
        cur_idx = atomicAdd((unsigned long long*)customer_counter, (unsigned long long) 1);

        customer_buffer[cur_idx] = Q3Customer{c_key[r_id],c_city[r_id],c_nation[r_id],c_region[r_id]};
    }
}

template <typename PRED_COL, typename PREDICATE, template<typename, typename> typename PRED>
__global__ void pipeline_customer_q3b(decltype(Customer::c_key) *c_key,
                                      decltype(Customer::c_city) *c_city,
                                      decltype(Customer::c_nation) *c_nation,
                                      decltype(Customer::c_region) *c_region,
                                      PRED<PRED_COL, PREDICATE> pred_info,
                                      Q3Customer *customer_buffer,
                                      uint64_t *customer_counter,
                                      uint64_t num
                                      ){
    uint64_t r_id;
    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        customer_buffer[r_id] = Q3Customer{c_key[r_id],c_city[r_id],c_nation[r_id],c_region[r_id]};
    }
}

template <typename PRED_COL, typename PREDICATE, template<typename, typename> typename PRED>
__global__ void pipeline_customer_q3c_prerun(decltype(Customer::c_key) *c_key,
                                      decltype(Customer::c_city) *c_city,
                                      decltype(Customer::c_nation) *c_nation,
                                      decltype(Customer::c_region) *c_region,
                                      PRED<PRED_COL, PREDICATE> pred_info,
                                      Q3Customer *customer_buffer,
                                      uint64_t *customer_counter,
                                      uint64_t num
                                      ){
    uint32_t r_id;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        /*
        1) Predicate on the customer
        2) Put in buffer
         */
        if (!pred_info.pred(&pred_info.col[r_id])) continue;
        
        atomicAdd((unsigned long long*)customer_counter, (unsigned long long) 1);

        // customer_buffer[cur_idx] = Q3Customer{c_key[r_id],c_city[r_id],c_nation[r_id],c_region[r_id]};
    }

}

template <typename PRED_COL, typename PREDICATE, template<typename, typename> typename PRED>
__global__ void pipeline_customer_q3c(golap::HashMap<Q3Customer, uint64_t> customer_hashmap,
                                      decltype(Customer::c_key) *c_key,
                                      decltype(Customer::c_city) *c_city,
                                      decltype(Customer::c_nation) *c_nation,
                                      decltype(Customer::c_region) *c_region,
                                      PRED<PRED_COL, PREDICATE> pred_info,
                                      Q3Customer *customer_buffer,
                                      uint64_t *customer_counter,
                                      uint64_t customer_host_insert,
                                      uint64_t num
                                      ){
    uint64_t r_id,cur_idx;

    // grid stride loop over the tuples
    for (r_id = blockDim.x * blockIdx.x + threadIdx.x ; r_id < num; r_id += blockDim.x * gridDim.x){
        __syncwarp();
        /*
        1) Predicate on the customer
        2) Put in buffer
         */
        if (!pred_info.pred(&pred_info.col[r_id])) continue;
        
        cur_idx = atomicAdd((unsigned long long*)customer_counter, (unsigned long long) 1);

        customer_buffer[cur_idx-customer_host_insert] = Q3Customer{c_key[r_id],c_city[r_id],c_nation[r_id],c_region[r_id]};
        customer_hashmap.insert(cur_idx, c_key[r_id]);
    }
}

