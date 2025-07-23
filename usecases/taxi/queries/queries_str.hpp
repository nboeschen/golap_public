#pragma once
#include <string>
#include <unordered_map>

// 237492807
const std::string query1_1_str =
"SELECT EXTRACT('month' FROM tpep_pickup_datetime) as month, COUNT(*) as trips "
"FROM trips "
"WHERE trip_distance >= 2 AND trip_distance <= 5 "
"GROUP BY month;";

// 32492434
const std::string query1_2_str =
"SELECT EXTRACT('month' FROM tpep_pickup_datetime) as month, COUNT(*) as trips "
"FROM trips "
"WHERE trip_distance >= 5 AND trip_distance <= 15 "
"GROUP BY month;";

// 7742055
const std::string query1_3_str =
"SELECT EXTRACT('month' FROM tpep_pickup_datetime) as month, COUNT(*) as trips "
"FROM trips "
"WHERE trip_distance >= 15 and trip_distance <= 5000"
"GROUP BY month;";


const std::string query2_1_str =
"SELECT EXTRACT('dayofweek' FROM tpep_pickup_datetime), "
"ROUND(AVG(trip_distance / (EXTRACT('epoch' FROM tpep_dropoff_datetime)-EXTRACT('epoch' FROM tpep_pickup_datetime)))*3600, 1) as speed "
"FROM trips "
"WHERE trip_distance > 0 AND fare_amount BETWEEN 0 AND 2 "
"AND tpep_dropoff_datetime > tpep_pickup_datetime "
"GROUP BY EXTRACT('dayofweek' FROM tpep_pickup_datetime);";

const std::string query2_2_str =
"SELECT EXTRACT('dayofweek' FROM tpep_pickup_datetime), "
"ROUND(AVG(trip_distance / (EXTRACT('epoch' FROM tpep_dropoff_datetime)-EXTRACT('epoch' FROM tpep_pickup_datetime)))*3600, 1) as speed "
"FROM trips "
"WHERE trip_distance > 0 AND fare_amount BETWEEN 2 AND 10 "
"AND tpep_dropoff_datetime > tpep_pickup_datetime "
"GROUP BY EXTRACT('dayofweek' FROM tpep_pickup_datetime);";

const std::string query2_3_str =
"SELECT EXTRACT('dayofweek' FROM tpep_pickup_datetime), "
"ROUND(AVG(trip_distance / (EXTRACT('epoch' FROM tpep_dropoff_datetime)-EXTRACT('epoch' FROM tpep_pickup_datetime)))*3600, 1) as speed "
"FROM trips "
"WHERE trip_distance > 0 AND fare_amount BETWEEN 10 AND 5000 "
"AND tpep_dropoff_datetime > tpep_pickup_datetime "
"GROUP BY EXTRACT('dayofweek' FROM tpep_pickup_datetime);";

std::unordered_map<std::string,std::string> query_strs = {
    {"query1.1",query1_1_str},
    {"query1.2",query1_2_str},
    {"query1.3",query1_3_str},
    {"query2.1",query2_1_str},
    {"query2.2",query2_2_str},
    {"query2.3",query2_3_str},
};

static std::string get_query_text(std::string query){
    if(query_strs.find(query) != query_strs.end()){ // the classic double lookup
        return query_strs[query];
    }else {
        return "Got a wrong query input!";
    }
}