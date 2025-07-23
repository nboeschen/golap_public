#pragma once

#include "hl/select.cuh"
#include "util.hpp"
#include "comp.cuh"
#include "comp_cpu.hpp"

#include "../taxi.hpp"
#include "../data/Taxi_def.hpp"




bool TaxiColLayout::select_VendorID() { return golap::select(var,tables.trips.col<Trips::VENDORID>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_tpep_pickup_datetime() { return golap::select(var,tables.trips.col<Trips::TPEP_PICKUP_DATETIME>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_tpep_dropoff_datetime() { return golap::select(var,tables.trips.col<Trips::TPEP_DROPOFF_DATETIME>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Passenger_count() { return golap::select(var,tables.trips.col<Trips::PASSENGER_COUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Trip_distance() { return golap::select(var,tables.trips.col<Trips::TRIP_DISTANCE>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_RateCodeID() { return golap::select(var,tables.trips.col<Trips::RATECODEID>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Store_and_fwd_flag() { return golap::select(var,tables.trips.col<Trips::STORE_AND_FWD_FLAG>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_PULocationID() { return golap::select(var,tables.trips.col<Trips::PULOCATIONID>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_DOLocationID() { return golap::select(var,tables.trips.col<Trips::DOLOCATIONID>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Payment_type() { return golap::select(var,tables.trips.col<Trips::PAYMENT_TYPE>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Fare_amount() { return golap::select(var,tables.trips.col<Trips::FARE_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Extra() { return golap::select(var,tables.trips.col<Trips::EXTRA>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_MTA_tax() { return golap::select(var,tables.trips.col<Trips::MTA_TAX>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Tip_amount() { return golap::select(var,tables.trips.col<Trips::TIP_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Tolls_amount() { return golap::select(var,tables.trips.col<Trips::TOLLS_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Improvement_surcharge() { return golap::select(var,tables.trips.col<Trips::IMPROVEMENT_SURCHARGE>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Total_amount() { return golap::select(var,tables.trips.col<Trips::TOTAL_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Congestion_Surcharge() { return golap::select(var,tables.trips.col<Trips::CONGESTION_SURCHARGE>(),tables.trips.num_tuples); }
bool TaxiColLayout::select_Airport_fee() { return golap::select(var,tables.trips.col<Trips::AIRPORT_FEE>(),tables.trips.num_tuples); }


bool TaxiColLayout::filter_VendorID() { return golap::filter(var,tables.trips.col<Trips::VENDORID>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_tpep_pickup_datetime() { return golap::filter(var,tables.trips.col<Trips::TPEP_PICKUP_DATETIME>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_tpep_dropoff_datetime() { return golap::filter(var,tables.trips.col<Trips::TPEP_DROPOFF_DATETIME>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Passenger_count() { return golap::filter(var,tables.trips.col<Trips::PASSENGER_COUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Trip_distance() { return golap::filter(var,tables.trips.col<Trips::TRIP_DISTANCE>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_RateCodeID() { return golap::filter(var,tables.trips.col<Trips::RATECODEID>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Store_and_fwd_flag() { return golap::filter(var,tables.trips.col<Trips::STORE_AND_FWD_FLAG>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_PULocationID() { return golap::filter(var,tables.trips.col<Trips::PULOCATIONID>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_DOLocationID() { return golap::filter(var,tables.trips.col<Trips::DOLOCATIONID>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Payment_type() { return golap::filter(var,tables.trips.col<Trips::PAYMENT_TYPE>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Fare_amount() { return golap::filter(var,tables.trips.col<Trips::FARE_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Extra() { return golap::filter(var,tables.trips.col<Trips::EXTRA>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_MTA_tax() { return golap::filter(var,tables.trips.col<Trips::MTA_TAX>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Tip_amount() { return golap::filter(var,tables.trips.col<Trips::TIP_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Tolls_amount() { return golap::filter(var,tables.trips.col<Trips::TOLLS_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Improvement_surcharge() { return golap::filter(var,tables.trips.col<Trips::IMPROVEMENT_SURCHARGE>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Total_amount() { return golap::filter(var,tables.trips.col<Trips::TOTAL_AMOUNT>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Congestion_Surcharge() { return golap::filter(var,tables.trips.col<Trips::CONGESTION_SURCHARGE>(),tables.trips.num_tuples); }
bool TaxiColLayout::filter_Airport_fee() { return golap::filter(var,tables.trips.col<Trips::AIRPORT_FEE>(),tables.trips.num_tuples); }
