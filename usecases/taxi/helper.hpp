#pragma once

#include <cstdint>
#include <vector>
#include <tuple>
#include <memory>
#include "data/Taxi_def.hpp"
#include "util.hpp"
#include "host_structs.hpp"

struct SortHelper: public util::Singleton<SortHelper>{
    using PREJOINED_TYPE = golap::ColumnTable<golap::HostMem>;

    static constexpr char prejoined_attrs[] = "";

    PREJOINED_TYPE* prejoined = nullptr;

    void prejoin_tables(Taxi_Tables_col<golap::HostMem> &tables, std::vector<uint64_t> &chunk_size_vec){
    }

    void apply(std::string def_string, Taxi_Tables_col<golap::HostMem> &tables, std::vector<uint64_t> &chunk_size_vec){
        std::vector<uint64_t> sort_order;

        if (def_string == "Trip_distance"){
            sort_order = tables.trips.sort_by<Trips::TRIP_DISTANCE>();
            tables.trips.to_csv(std::cout, ", ", 0, 25);
        }else if (def_string == "Fare_amount"){
            sort_order = tables.trips.sort_by<Trips::FARE_AMOUNT>();
            tables.trips.to_csv(std::cout, ", ", 0, 25);
        }else if (def_string == "general_dimsort"){
            sort_order = tables.trips.sort_by<Trips::FARE_AMOUNT, Trips::TRIP_DISTANCE>();
            tables.trips.to_csv(std::cout, ", ", 0, 25);
        }else {
            std::cout << "Unknown sorting: " << def_string << "\n";
        }

    }
};
