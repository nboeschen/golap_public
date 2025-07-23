#pragma once
#include <string>
#include <unordered_map>

const std::string query1_1_str =
"select sum(lo_extendedprice*lo_discount) as revenue "
"from lineorder, date "
"where lo_orderdate = d_key "
"and d_year = 1993 "
"and lo_discount between 1 and 3 "
"and lo_quantity < 25;";

const std::string query1_2_str =
"select sum(lo_extendedprice*lo_discount) as revenue "
"from lineorder, date "
"where lo_orderdate = d_key "
"and d_yearmonthnum = 199401 "
"and lo_discount between 4 and 6 "
"and lo_quantity between 26 and 35;";

const std::string query1_3_str =
"select sum(lo_extendedprice*lo_discount) as revenue "
"from lineorder, date "
"where lo_orderdate = d_key "
"and d_weeknuminyear = 6 "
"and d_year = 1994 "
"and lo_discount between 5 and 7 "
"and lo_quantity between 26 and 35;";

const std::string query2_1_str =
"select sum(lo_revenue), d_year, p_brand1 "
"from lineorder, date, part, supplier "
"where lo_orderdate = d_key "
"and lo_partkey = p_key "
"and lo_suppkey = s_key "
"and p_category = 'MFGR#12' "
"and s_region = 'AMERICA' "
"group by d_year, p_brand1;";
// "order by d_year, p_brand1;";

const std::string query2_2_str =
"select sum(lo_revenue), d_year, p_brand1 "
"from lineorder, date, part, supplier "
"where lo_orderdate = d_key "
"and lo_partkey = p_key "
"and lo_suppkey = s_key "
"and p_brand1 between 'MFGR#2221' "
"and 'MFGR#2228' "
"and s_region = 'ASIA' "
"group by d_year, p_brand1;";
// "order by d_year, p_brand1;";

const std::string query2_3_str =
"select sum(lo_revenue), d_year, p_brand1 "
"from lineorder, date, part, supplier "
"where lo_orderdate = d_key "
"and lo_partkey = p_key "
"and lo_suppkey = s_key "
"and p_brand1= 'MFGR#2239' "
"and s_region = 'EUROPE' "
"group by d_year, p_brand1;";
// "order by d_year, p_brand1;";

const std::string query3_1_str = "select c_nation, s_nation, d_year, sum(lo_revenue) as revenue "
"from customer, lineorder, supplier, date "
"where lo_custkey = c_key "
"and lo_suppkey = s_key "
"and lo_orderdate = d_key "
"and c_region = 'ASIA' "
"and s_region = 'ASIA' "
"and d_year >= 1992 and d_year <= 1997 "
"group by c_nation, s_nation, d_year;";
// "order by d_year asc, revenue desc;";

const std::string query3_2_str = "select c_city, s_city, d_year, sum(lo_revenue) as revenue "
"from customer, lineorder, supplier, date "
"where lo_custkey = c_key "
"and lo_suppkey = s_key "
"and lo_orderdate = d_key "
"and c_nation = 'UNITED STATES' "
"and s_nation = 'UNITED STATES' "
"and d_year >= 1992 and d_year <= 1997 "
"group by c_city, s_city, d_year;";
// "order by d_year asc, revenue desc;";

const std::string query3_3_str = "select c_city, s_city, d_year, sum(lo_revenue) as revenue "
"from customer, lineorder, supplier, date "
"where lo_custkey = c_key "
"and lo_suppkey = s_key "
"and lo_orderdate = d_key "
"and (c_city='UNITED KI1' or c_city='UNITED KI5') "
"and (s_city='UNITED KI1' or s_city='UNITED KI5') "
"and d_year >= 1992 and d_year <= 1997 "
"group by c_city, s_city, d_year;";
// "order by d_year asc, revenue desc;";

const std::string query3_4_str = "select c_city, s_city, d_year, sum(lo_revenue) as revenue "
"from customer, lineorder, supplier, date "
"where lo_custkey = c_key "
"and lo_suppkey = s_key "
"and lo_orderdate = d_key "
"and (c_city='UNITED KI1' or c_city='UNITED KI5') "
"and (s_city='UNITED KI1' or s_city='UNITED KI5') "
"and d_yearmonth = 'Dec1997' "
"group by c_city, s_city, d_year;";
// "order by d_year asc, revenue desc;";

const std::string query4_1_str = "select d_year, c_nation, "
"sum(lo_revenue - lo_supplycost) as profit "
"from date, customer, supplier, part, lineorder "
"where lo_custkey = c_key "
"and lo_suppkey = s_key "
"and lo_partkey = p_key "
"and lo_orderdate = d_key "
"and c_region = 'AMERICA' "
"and s_region = 'AMERICA' "
"and (p_mfgr = 'MFGR#1' "
"or p_mfgr = 'MFGR#2') "
"group by d_year, c_nation;";
// "order by d_year, c_nation; "

const std::string query4_2_str = "select d_year, s_nation, p_category, "
"sum(lo_revenue - lo_supplycost) as profit "
"from date, customer, supplier, part, lineorder "
"where lo_custkey = c_key "
"and lo_suppkey = s_key "
"and lo_partkey = p_key "
"and lo_orderdate = d_key "
"and c_region = 'AMERICA' "
"and s_region = 'AMERICA' "
"and (d_year = 1997 or d_year = 1998) "
"and (p_mfgr = 'MFGR#1' "
"or p_mfgr = 'MFGR#2') "
"group by d_year, s_nation, p_category;";
// "order by d_year, s_nation, p_category; "

const std::string query4_3_str = 
"select d_year, s_city, p_brand1, "
"sum(lo_revenue - lo_supplycost) as profit "
"from date, customer, supplier, part, lineorder "
"where lo_custkey = c_key "
"and lo_suppkey = s_key "
"and lo_partkey = p_key "
"and lo_orderdate = d_key "
"and c_region = 'AMERICA' "
"and s_nation = 'UNITED STATES' "
"and (d_year = 1997 or d_year = 1998) "
"and p_category = 'MFGR#14' "
"group by d_year, s_city, p_brand1;";
// "order by d_year, s_city, p_brand1; "


const std::string query2_1_pre_aggr_str =
"SELECT sum(rev_per_week), d_year, d_monthnuminyear "
"from lineorder, supplier "
"where lo_suppkey = s_key and s_region = 'AMERICA' "
"group by d_year, d_monthnuminyear; ";

const std::string query2_2_pre_aggr_str =
"SELECT sum(rev_per_week), d_year, d_monthnuminyear "
"from lineorder, supplier "
"where lo_suppkey = s_key and s_nation = 'CHINA' "
"group by d_year, d_monthnuminyear; ";

const std::string scan_only =
"SELECT COUNT(*) "
"from lineorder "
"where lo_orderdate < 20101212; ";

const std::string scan_only2 =
"SELECT COUNT(*) "
"from lineorder "
"where lo_orderdate < 20101212 and lo_custkey > 1500000; ";

std::unordered_map<std::string,std::string> query_strs = {
    {"query1.1",query1_1_str},
    {"query1.2",query1_2_str},
    {"query1.3",query1_3_str},
    {"query2.1",query2_1_str},
    {"query2.2",query2_2_str},
    {"query2.3",query2_3_str},
    {"query3.1",query3_1_str},
    {"query3.2",query3_2_str},
    {"query3.3",query3_3_str},
    {"query3.4",query3_4_str},
    {"query4.1",query4_1_str},
    {"query4.2",query4_2_str},
    {"query4.3",query4_3_str},
    {"query2.1_pre_aggr",query2_1_pre_aggr_str},
    {"query2.2_pre_aggr",query2_2_pre_aggr_str},
    {"scan_only",scan_only},
    {"scan_only2",scan_only2},
};

static std::string get_query_text(std::string query){
    if (query.substr(0,7) == "select_"){
        std::cout << "#[WARN ] Select query will be transformed to a count!\n";
        return "SELECT COUNT(lo_"+query.substr(7)+") FROM lineorder;";
    }else if (query.substr(0,6) == "count_"){
        return "SELECT COUNT(lo_"+query.substr(6)+") FROM lineorder;";
    }else if(query.substr(0,4) == "sum_"){
        return "SELECT SUM(lo_"+query.substr(4)+") FROM lineorder;";
    }else if(query_strs.find(query) != query_strs.end()){ // the classic double lookup
        return query_strs[query];
    }else {
        return "Got a wrong query input!";
    }
}