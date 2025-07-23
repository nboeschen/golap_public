#pragma once

#include <cstdint>
#include <vector>
#include <tuple>
#include <memory>
#include "data/SSB_def.hpp"
#include "util.hpp"
#include "host_structs.hpp"
#include "queries/02.cuh"

struct SortHelper: public util::Singleton<SortHelper>{
    ~SortHelper(){
        // we dont actually need to clean up the cached columns, since the singleton will be deleted after the call to cudaDeviceReset,
        // freeing here will throw errors
    }
    /**
     * For simplicity, just cache the joined tables here.
     */
    using PREJOINED_TYPE = golap::ColumnTable<golap::HostMem, decltype(Lineorder::lo_discount), decltype(Lineorder::lo_quantity),
                    decltype(Date::d_year), decltype(Date::d_yearmonthnum), decltype(Date::d_weeknuminyear),
                    decltype(Date::d_yearmonth),
                    decltype(Customer::c_city), decltype(Customer::c_nation), decltype(Customer::c_region)>;
    static constexpr char prejoined_attrs[] = "meta_lo_discount,meta_lo_quantity,meta_d_year,meta_d_yearmonthnum,meta_d_weeknuminyear,meta_d_yearmonth,meta_c_city,meta_c_nation,meta_c_region";

    struct PRE_AGGR_DB {
    public:
        static constexpr char preaggr_attrs[] = "d_year,d_weeknuminyear,d_monthnuminyear,lo_suppkey,rev_per_week";

        PRE_AGGR_DB(uint64_t num_lines):pre_aggr("pre_aggr",preaggr_attrs,num_lines,4096){}


        golap::ColumnTable<golap::HostMem, decltype(Date::d_year), decltype(Date::d_weeknuminyear), decltype(Date::d_monthnuminyear),
                    decltype(Lineorder::lo_suppkey), uint64_t> pre_aggr;

        template<typename Fn>
        void apply(Fn&& f){
            f(pre_aggr);
        }

    };


    PREJOINED_TYPE* prejoined = nullptr;
    PRE_AGGR_DB* preaggrdb = nullptr;

    void init_pre_aggr(uint64_t num_tuples){
        if (!preaggrdb){
            preaggrdb = new PRE_AGGR_DB{num_tuples};
        }
    }

    void preaggr_tables(SSB_Tables_col<golap::HostMem> &tables, std::vector<uint64_t> &chunk_size_vec){
        if (!preaggrdb){
            init_pre_aggr(tables.lineorder.num_tuples);
        }

        uint64_t num_groups = tables.lineorder.num_tuples >> 2;
        golap::HostMem groups(golap::Tag<YEARMONTHWEEKSUPP_GROUP>{}, num_groups);
        std::atomic<uint64_t> tuple_idx{0};
        auto rev_per_week = new std::atomic<uint64_t>[num_groups]{};
        golap::HostHashAggregate hash_agg(num_groups, groups.ptr<YEARMONTHWEEKSUPP_GROUP>(), rev_per_week);

        golap::HostHashMap date_map(tables.date.num_tuples, tables.date.col<Date::KEY>().data());
        for (uint64_t tuple_id = 0; tuple_id < tables.date.num_tuples; ++tuple_id){
            date_map.insert(tuple_id, tables.date.col<Date::KEY>().data()[tuple_id]);
        }

        util::Timer timer;
        util::ThreadPool pool;
        pool.parallel_n(8, [&](int tid) {
            auto [start, stop] = util::RangeHelper::nth_chunk(0, tables.lineorder.num_tuples, 8, tid);
            for (uint64_t lo_tuple_id = start; lo_tuple_id < stop; ++lo_tuple_id){
                auto dkey = date_map.probe(tables.lineorder.col<Lineorder::ORDERDATE>().data()[lo_tuple_id]);
                if(dkey == (uint64_t) -1){
                    std::cout << "Failed in prejoin, this shouldnt ever happen\n";
                }

                hash_agg.add(YEARMONTHWEEKSUPP_GROUP{tables.date.col<Date::YEAR>().data()[dkey],
                                                 tables.date.col<Date::MONTHNUMINYEAR>().data()[dkey],
                                                 tables.date.col<Date::WEEKNUMINYEAR>().data()[dkey],
                                                 tables.lineorder.col<Lineorder::SUPPKEY>().data()[lo_tuple_id]
                                             }, tables.lineorder.col<Lineorder::REVENUE>().data()[lo_tuple_id], HostSum());
            }
        });
        pool.join();
        printf("HashAgg took %.2f ms\n",timer.elapsed());

        // update the preaggr table with actual values
        util::ThreadPool pool2;
        pool2.parallel_n(8, [&](int tid) {
            uint64_t insert_idx;
            auto [start, stop] = util::RangeHelper::nth_chunk(0, num_groups, 8, tid);
            for (uint64_t i = start; i < stop; ++i){
                if ( hash_agg.wrote_group[i] == 0) continue;
                insert_idx = tuple_idx.fetch_add(1, std::memory_order_relaxed);

                auto& group = groups.ptr<YEARMONTHWEEKSUPP_GROUP>()[i];

                preaggrdb->pre_aggr.col<0>().data()[insert_idx] = group.d_year;
                preaggrdb->pre_aggr.col<1>().data()[insert_idx] = group.d_weeknuminyear;
                preaggrdb->pre_aggr.col<2>().data()[insert_idx] = group.d_monthnuminyear;
                preaggrdb->pre_aggr.col<3>().data()[insert_idx] = group.lo_suppkey;
                preaggrdb->pre_aggr.col<4>().data()[insert_idx] = rev_per_week[i].load(std::memory_order_relaxed);

            }
        });
        pool2.join();
        preaggrdb->pre_aggr.num_tuples = tuple_idx.load();
        delete[] rev_per_week;

    }


    void prejoin_tables(SSB_Tables_col<golap::HostMem> &tables, std::vector<uint64_t> &chunk_size_vec){
        if (!prejoined){
            prejoined = new PREJOINED_TYPE{prejoined_attrs, tables.lineorder.num_tuples, 4096};
            prejoined->num_tuples = tables.lineorder.num_tuples;
            tables.lineorder.col<Lineorder::DISCOUNT>().transfer(prejoined->col<0>().data(), tables.lineorder.num_tuples);
            tables.lineorder.col<Lineorder::QUANTITY>().transfer(prejoined->col<1>().data(), tables.lineorder.num_tuples);
        }

        golap::HostHashMap date_map(tables.date.num_tuples, tables.date.col<Date::KEY>().data());
        golap::HostHashMap cust_map(tables.customer.num_tuples, tables.customer.col<Customer::KEY>().data());

        for (uint64_t tuple_id = 0; tuple_id < tables.date.num_tuples; ++tuple_id){
            date_map.insert(tuple_id, tables.date.col<Date::KEY>().data()[tuple_id]);
        }
        for (uint64_t tuple_id = 0; tuple_id < tables.customer.num_tuples; ++tuple_id){
            cust_map.insert(tuple_id, tables.customer.col<Customer::KEY>().data()[tuple_id]);
        }

        util::ThreadPool pool;
        pool.parallel_n(8, [&](int tid) {
            auto [start, stop] = util::RangeHelper::nth_chunk(0, prejoined->num_tuples, 8, tid);
            for (uint64_t tuple_id = start; tuple_id < stop; ++tuple_id){
                auto dkey = date_map.probe(tables.lineorder.col<Lineorder::ORDERDATE>().data()[tuple_id]);
                auto ckey = cust_map.probe(tables.lineorder.col<Lineorder::CUSTKEY>().data()[tuple_id]);
                if(dkey == (uint64_t) -1 || ckey == (uint64_t) -1){
                    std::cout << "Failed in prejoin, this shouldnt ever happen\n";
                }
                prejoined->col<2>().data()[tuple_id] = tables.date.col<Date::YEAR>().data()[dkey];
                prejoined->col<3>().data()[tuple_id] = tables.date.col<Date::YEARMONTHNUM>().data()[dkey];
                prejoined->col<4>().data()[tuple_id] = tables.date.col<Date::WEEKNUMINYEAR>().data()[dkey];
                prejoined->col<5>().data()[tuple_id] = tables.date.col<Date::YEARMONTH>().data()[dkey];
                prejoined->col<6>().data()[tuple_id] = tables.customer.col<Customer::CITY>().data()[ckey];
                prejoined->col<7>().data()[tuple_id] = tables.customer.col<Customer::NATION>().data()[ckey];
                prejoined->col<8>().data()[tuple_id] = tables.customer.col<Customer::REGION>().data()[ckey];
            }
        });
        pool.join();
    }

    void apply(std::string def_string, SSB_Tables_col<golap::HostMem> &tables, std::vector<uint64_t> &chunk_size_vec){
        std::vector<uint64_t> sort_order;
        bool prejoined_sorted = false;
        if (def_string == "natural"){
            sort_order = tables.lineorder.sort_by<Lineorder::KEY,Lineorder::LINENUMBER>();
        }else if (def_string == "random") sort_order = tables.lineorder.sort_random();
        else if (def_string == "lo_key") sort_order = tables.lineorder.sort_by<Lineorder::KEY>(); // almost natural/insertion order (shuffled linenumber)
        else if (def_string == "lo_discount") sort_order = tables.lineorder.sort_by<Lineorder::DISCOUNT>();
        else if (def_string == "lo_discount|lo_quantity") sort_order = tables.lineorder.sort_by<Lineorder::DISCOUNT,Lineorder::QUANTITY>();
        else if (def_string == "lo_quantity|lo_discount") sort_order = tables.lineorder.sort_by<Lineorder::QUANTITY,Lineorder::DISCOUNT>();
        else if (def_string == "lo_orderdate") sort_order = tables.lineorder.sort_by<Lineorder::ORDERDATE>();
        else if (def_string == "lo_quantity") sort_order = tables.lineorder.sort_by<Lineorder::QUANTITY>();
        else if (def_string == "lo_extendedprice") sort_order = tables.lineorder.sort_by<Lineorder::EXTENDEDPRICE>();
        else if (def_string == "sortq11"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // prejoined->to_csv(std::cout, ", ", 0, 25);
            // sort by year, discount, quantity
            sort_order = prejoined->sort_by<2,0,1>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);
        }else if (def_string == "sortq12"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // prejoined->to_csv(std::cout, ", ", 0, 25);
            // sort by yearmonthnum, discount, quantity
            sort_order = prejoined->sort_by<3,0,1>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);

        }else if (def_string == "sortq13"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // prejoined->to_csv(std::cout, ", ", 0, 25);
            // sort by year, weeknuminyear, discount, quantity
            sort_order = prejoined->sort_by<2,4,0,1>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);

        }else if (def_string == "sortq31"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // sort by d_year, c_region
            sort_order = prejoined->sort_by<2,8>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            tables.customer.sort_by<Customer::REGION>();

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);
        }else if (def_string == "sortq32"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // sort by d_year, c_nation
            sort_order = prejoined->sort_by<2,7>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            tables.customer.sort_by<Customer::NATION>();

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);
        }else if (def_string == "sortq33"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // sort by d_year, c_city
            sort_order = prejoined->sort_by<2,6>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            tables.customer.sort_by<Customer::CITY>();

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);
        }else if (def_string == "sortq34"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // sort by d_yearmonth, c_city
            sort_order = prejoined->sort_by<5,6>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            tables.customer.sort_by<Customer::CITY>();

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);
        }else if (def_string == "general_dimsort"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // #[INFO ] Values in column 8,meta_c_region: #vals=5
            // #[INFO ] Values in column 2,meta_d_year: #vals=7
            // #[INFO ] Values in column 0,meta_lo_discount: #vals=11
            // #[INFO ] Values in column 7,meta_c_nation: #vals=25
            // #[INFO ] Values in column 1,meta_lo_quantity: #vals=50
            // #[INFO ] Values in column 4,meta_d_weeknuminyear: #vals=54
            // #[INFO ] Values in column 3,meta_d_yearmonthnum: #vals=84
            // #[INFO ] Values in column 5,meta_d_yearmonth: #vals=84
            // #[INFO ] Values in column 6,meta_c_city: #vals=250
            // sort_order = prejoined->sort_by<2,4,3,8,7,6,0,1>();
            sort_order = prejoined->sort_by<8,0,7,1,3,6>();
            tables.lineorder.sort(sort_order);
            prejoined_sorted = true;

            tables.customer.sort_by<Customer::REGION,Customer::NATION,Customer::CITY>();

            // tables.lineorder.to_csv(std::cout, ", ", 0, 25);
        }else util::Log::get().error_fmt("Unknown sort order: %s",def_string.c_str());


        if (!prejoined_sorted){
            // to get meaningful metadata, the cached data also has to be sorted (at least the interesting columns)
            // otherwise the collected metadata is wrong!
            prejoined->sort(sort_order);
        }
    }
};
