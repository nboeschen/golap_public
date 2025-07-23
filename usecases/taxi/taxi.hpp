#pragma once

#include <cstdint>
#include <string>
#include <thread>

#include "mem.hpp"
#include "table.hpp"
#include "storage.hpp"
#include "util.hpp"
#include "comp.cuh"
#include "comp_cpu.hpp"
#include "access.hpp"
#include "join.cuh"
#include "apps.cuh"

#include "core.hpp"
#include "data/Taxi_def.hpp"


struct TaxiVar : public golap::Parameter {};

static const std::unordered_map<std::string,std::string> BEST_BW_COMP = {
    {"VendorID", "Bitcomp"},
    {"tpep_pickup_datetime", "ANS"},
    {"tpep_dropoff_datetime", "ANS"},
    {"Passenger_count", "Cascaded"},
    {"Trip_distance", "Gdeflate"},
    {"RateCodeID", "Bitcomp"},
    {"Store_and_fwd_flag", "Bitcomp"},
    {"PULocationID", "Bitcomp"},
    {"DOLocationID", "Bitcomp"},
    {"Payment_type", "Bitcomp"},
    {"Fare_amount", "ANS"},
    {"Extra", "Bitcomp"},
    {"MTA_tax", "Bitcomp"},
    {"Tip_amount", "Gdeflate"},
    {"Tolls_amount", "Cascaded"},
    {"Improvement_surcharge", "Bitcomp"},
    {"Total_amount", "Gdeflate"},
    {"Congestion_Surcharge", "Bitcomp"},
    {"Airport_fee", "Bitcomp"},
};

class TaxiColLayout{
public:
    TaxiColLayout(TaxiVar &var, std::string variant):var(var),tables(var.scale_factor){
        if (variant == "init_only"){
            tables.init();
        }else{
            std::cout << "Inknown init variant: " << variant << ", exiting!\n";
            std::exit(1);
        }
    }


    Taxi_Tables_col<golap::HostMem> tables;

    TaxiVar &var;

    bool select_VendorID();
    bool select_tpep_pickup_datetime();
    bool select_tpep_dropoff_datetime();
    bool select_Passenger_count();
    bool select_Trip_distance();
    bool select_RateCodeID();
    bool select_Store_and_fwd_flag();
    bool select_PULocationID();
    bool select_DOLocationID();
    bool select_Payment_type();
    bool select_Fare_amount();
    bool select_Extra();
    bool select_MTA_tax();
    bool select_Tip_amount();
    bool select_Tolls_amount();
    bool select_Improvement_surcharge();
    bool select_Total_amount();
    bool select_Congestion_Surcharge();
    bool select_Airport_fee();

    bool filter_VendorID();
    bool filter_tpep_pickup_datetime();
    bool filter_tpep_dropoff_datetime();
    bool filter_Passenger_count();
    bool filter_Trip_distance();
    bool filter_RateCodeID();
    bool filter_Store_and_fwd_flag();
    bool filter_PULocationID();
    bool filter_DOLocationID();
    bool filter_Payment_type();
    bool filter_Fare_amount();
    bool filter_Extra();
    bool filter_MTA_tax();
    bool filter_Tip_amount();
    bool filter_Tolls_amount();
    bool filter_Improvement_surcharge();
    bool filter_Total_amount();
    bool filter_Congestion_Surcharge();
    bool filter_Airport_fee();

    bool query1_1();
    bool query1_2();
    bool query1_3();
    bool query1_1inmem();
    bool query1_2inmem();
    bool query1_3inmem();

    bool query2_1();
    bool query2_2();
    bool query2_3();
    bool query2_1inmem();
    bool query2_2inmem();
    bool query2_3inmem();
};


#include "queries/select.hpp"
#include "queries/01.hpp"
#include "queries/01inmem.hpp"
#include "queries/02.hpp"
#include "queries/02inmem.hpp"


const static std::unordered_map<std::string,decltype(&TaxiColLayout::select_VendorID)> QUERY_FUNC_PTRS{
    {"select_VendorID", &TaxiColLayout::select_VendorID},
    {"select_tpep_pickup_datetime", &TaxiColLayout::select_tpep_pickup_datetime},
    {"select_tpep_dropoff_datetime", &TaxiColLayout::select_tpep_dropoff_datetime},
    {"select_Passenger_count", &TaxiColLayout::select_Passenger_count},
    {"select_Trip_distance", &TaxiColLayout::select_Trip_distance},
    {"select_RateCodeID", &TaxiColLayout::select_RateCodeID},
    {"select_Store_and_fwd_flag", &TaxiColLayout::select_Store_and_fwd_flag},
    {"select_PULocationID", &TaxiColLayout::select_PULocationID},
    {"select_DOLocationID", &TaxiColLayout::select_DOLocationID},
    {"select_Payment_type", &TaxiColLayout::select_Payment_type},
    {"select_Fare_amount", &TaxiColLayout::select_Fare_amount},
    {"select_Extra", &TaxiColLayout::select_Extra},
    {"select_MTA_tax", &TaxiColLayout::select_MTA_tax},
    {"select_Tip_amount", &TaxiColLayout::select_Tip_amount},
    {"select_Tolls_amount", &TaxiColLayout::select_Tolls_amount},
    {"select_Improvement_surcharge", &TaxiColLayout::select_Improvement_surcharge},
    {"select_Total_amount", &TaxiColLayout::select_Total_amount},
    {"select_Congestion_Surcharge", &TaxiColLayout::select_Congestion_Surcharge},
    {"select_Airport_fee", &TaxiColLayout::select_Airport_fee},

    {"filter_VendorID", &TaxiColLayout::filter_VendorID},
    {"filter_tpep_pickup_datetime", &TaxiColLayout::filter_tpep_pickup_datetime},
    {"filter_tpep_dropoff_datetime", &TaxiColLayout::filter_tpep_dropoff_datetime},
    {"filter_Passenger_count", &TaxiColLayout::filter_Passenger_count},
    {"filter_Trip_distance", &TaxiColLayout::filter_Trip_distance},
    {"filter_RateCodeID", &TaxiColLayout::filter_RateCodeID},
    {"filter_Store_and_fwd_flag", &TaxiColLayout::filter_Store_and_fwd_flag},
    {"filter_PULocationID", &TaxiColLayout::filter_PULocationID},
    {"filter_DOLocationID", &TaxiColLayout::filter_DOLocationID},
    {"filter_Payment_type", &TaxiColLayout::filter_Payment_type},
    {"filter_Fare_amount", &TaxiColLayout::filter_Fare_amount},
    {"filter_Extra", &TaxiColLayout::filter_Extra},
    {"filter_MTA_tax", &TaxiColLayout::filter_MTA_tax},
    {"filter_Tip_amount", &TaxiColLayout::filter_Tip_amount},
    {"filter_Tolls_amount", &TaxiColLayout::filter_Tolls_amount},
    {"filter_Improvement_surcharge", &TaxiColLayout::filter_Improvement_surcharge},
    {"filter_Total_amount", &TaxiColLayout::filter_Total_amount},
    {"filter_Congestion_Surcharge", &TaxiColLayout::filter_Congestion_Surcharge},
    {"filter_Airport_fee", &TaxiColLayout::filter_Airport_fee},

    {"query1.1", &TaxiColLayout::query1_1},
    {"query1.2", &TaxiColLayout::query1_2},
    {"query1.3", &TaxiColLayout::query1_3},
    {"query1.1inmem", &TaxiColLayout::query1_1inmem},
    {"query1.2inmem", &TaxiColLayout::query1_2inmem},
    {"query1.3inmem", &TaxiColLayout::query1_3inmem},
    {"query2.1", &TaxiColLayout::query2_1},
    {"query2.2", &TaxiColLayout::query2_2},
    {"query2.3", &TaxiColLayout::query2_3},
    {"query2.1inmem", &TaxiColLayout::query2_1inmem},
    {"query2.2inmem", &TaxiColLayout::query2_2inmem},
    {"query2.3inmem", &TaxiColLayout::query2_3inmem},
};