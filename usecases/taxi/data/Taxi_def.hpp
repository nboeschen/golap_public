#pragma once

#include <cstdint>
#include <memory>
#include <cstdlib>

#include "mem.hpp"
#include "util.hpp"
#include "types.hpp"
#include "table.hpp"

struct alignas(8) Trips{
    static constexpr char ATTR[] = "VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,Passenger_count,Trip_distance,RateCodeID,Store_and_fwd_flag,PULocationID,DOLocationID,Payment_type,Fare_amount,Extra,MTA_tax,Tip_amount,Tolls_amount,Improvement_surcharge,Total_amount,Congestion_Surcharge,Airport_fee";
    // VendorID|tpep_pickup_datetime|tpep_dropoff_datetime|passenger_count|trip_distance|RatecodeID|store_and_fwd_flag|PULocationID|DOLocationID|payment_type|fare_amount|extra|mta_tax|tip_amount|tolls_amount|improvement_surcharge|total_amount|congestion_surcharge|airport_fee
    enum : uint64_t {
            VENDORID=0,TPEP_PICKUP_DATETIME=1,TPEP_DROPOFF_DATETIME=2,PASSENGER_COUNT=3,TRIP_DISTANCE=4,RATECODEID=5,STORE_AND_FWD_FLAG=6,PULOCATIONID=7,DOLOCATIONID=8,PAYMENT_TYPE=9,FARE_AMOUNT=10,EXTRA=11,MTA_TAX=12,TIP_AMOUNT=13,TOLLS_AMOUNT=14,IMPROVEMENT_SURCHARGE=15,TOTAL_AMOUNT=16,CONGESTION_SURCHARGE=17,AIRPORT_FEE=18};
    // Datatypes from the provided parquet files
    int64_t VendorID;
    util::Datetime tpep_pickup_datetime;
    util::Datetime tpep_dropoff_datetime;
    int64_t Passenger_count;
    double Trip_distance;
    int64_t RateCodeID;
    char Store_and_fwd_flag;
    int64_t PULocationID;
    int64_t DOLocationID;
    int64_t Payment_type;
    double Fare_amount;
    double Extra;
    double MTA_tax;
    double Tip_amount;
    double Tolls_amount;
    double Improvement_surcharge;
    double Total_amount;
    int32_t Congestion_Surcharge;
    int32_t Airport_fee;
};


template <typename MEM_TYPE>
class Taxi_Tables_col{
public:
    Taxi_Tables_col(uint64_t scale_factor):scale_factor(scale_factor),
            trips("trips", Trips::ATTR, 0)
        {
            // 15 million per month
        trips_num = scale_factor * 15000000;
    }
    void init(){
        init_trips();
    }

    void init_trips() { trips.resize(trips_num); }

    uint64_t size_bytes(){
        return trips.size_bytes();
    }

    template<typename Fn>
    void apply(Fn&& f){
        f(trips);
    }

    uint32_t scale_factor;
    uint64_t rand_seed = 0xBEE5BEE5;

    uint64_t trips_num;

    golap::ColumnTable<MEM_TYPE,
        decltype(Trips::VendorID),
        decltype(Trips::tpep_pickup_datetime),
        decltype(Trips::tpep_dropoff_datetime),
        decltype(Trips::Passenger_count),
        decltype(Trips::Trip_distance),
        decltype(Trips::RateCodeID),
        decltype(Trips::Store_and_fwd_flag),
        decltype(Trips::PULocationID),
        decltype(Trips::DOLocationID),
        decltype(Trips::Payment_type),
        decltype(Trips::Fare_amount),
        decltype(Trips::Extra),
        decltype(Trips::MTA_tax),
        decltype(Trips::Tip_amount),
        decltype(Trips::Tolls_amount),
        decltype(Trips::Improvement_surcharge),
        decltype(Trips::Total_amount),
        decltype(Trips::Congestion_Surcharge),
        decltype(Trips::Airport_fee)
    > trips;



};


