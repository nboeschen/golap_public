#pragma once

#include <cstdint>
#include <vector>
#include <tuple>
#include <memory>
#include "data/TPCH_def.hpp"
#include "util.hpp"
#include "host_structs.hpp"

struct SortHelper: public util::Singleton<SortHelper>{
    ~SortHelper(){
    }
    /**
     * For simplicity, just cache the joined tables here.
     */
    using PREJOINED_TYPE = golap::ColumnTable<golap::HostMem, decltype(Lineitem::l_quantity), decltype(Lineitem::l_shipdate), decltype(Order::o_orderdate)>;
    static constexpr char prejoined_attrs[] = "meta_l_quantity,meta_l_shipdate,meta_o_orderdate";

    PREJOINED_TYPE* prejoined = nullptr;

    void prejoin_tables(TPCH_Tables_col<golap::HostMem> &tables, std::vector<uint64_t> &chunk_size_vec){
        if (!prejoined){
            prejoined = new PREJOINED_TYPE{prejoined_attrs, tables.lineitem.num_tuples, 4096};
            prejoined->num_tuples = tables.lineitem.num_tuples;
            tables.lineitem.col<Lineitem::QUANTITY>().transfer(prejoined->col<0>().data(), tables.lineitem.num_tuples);
            tables.lineitem.col<Lineitem::SHIPDATE>().transfer(prejoined->col<1>().data(), tables.lineitem.num_tuples);
        }

        golap::HostHashMap order_map(tables.orders.num_tuples, tables.orders.col<Order::ORDERKEY>().data());

        for (uint64_t tuple_id = 0; tuple_id < tables.orders.num_tuples; ++tuple_id){
            order_map.insert(tuple_id, tables.orders.col<Order::ORDERKEY>().data()[tuple_id]);
        }

        util::ThreadPool pool;
        pool.parallel_n(8, [&](int tid) {
            auto [start, stop] = util::RangeHelper::nth_chunk(0, prejoined->num_tuples, 8, tid);
            for (uint64_t tuple_id = start; tuple_id < stop; ++tuple_id){
                auto okey = order_map.probe(tables.lineitem.col<Lineitem::ORDERKEY>().data()[tuple_id]);
                if(okey == (uint64_t) -1){
                    std::cout << "Failed in prejoin, this shouldnt ever happen\n";
                }
                prejoined->col<2>().data()[tuple_id] = tables.orders.col<Order::ORDERDATE>().data()[okey];
            }
        });
        pool.join();
    }


    void apply(std::string def_string, TPCH_Tables_col<golap::HostMem> &tables, std::vector<uint64_t> &chunk_size_vec){
        std::vector<uint64_t> sort_order;
        bool prejoined_sorted = false;

        if (def_string == "general_dimsort"){
            if (!prejoined) prejoin_tables(tables, chunk_size_vec);

            // l_quantity, then o_orderdate and l_shipdate (correlated)
            sort_order = prejoined->sort_by<0,2,1>();
            tables.lineitem.sort(sort_order);
            prejoined_sorted = true;


            // tables.lineitem.to_csv(std::cout, ", ", 0, 25);
        }else std::cout << "Unknown sorting: " << def_string << "\n";

        if (!prejoined_sorted){
            // to get meaningful metadata, the cached data also has to be sorted (at least the interesting columns)
            // otherwise the collected metadata is wrong!
            prejoined->sort(sort_order);
        }
    }
};
