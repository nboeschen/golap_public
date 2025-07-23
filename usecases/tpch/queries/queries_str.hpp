#pragma once
#include <string>
#include <unordered_map>

const std::string query1_str =
"select "
"        l_returnflag, "
"        l_linestatus, "
"        sum(l_quantity) as sum_qty, "
"        sum(l_extendedprice) as sum_base_price, "
"        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
"        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
"        avg(l_quantity) as avg_qty, "
"        avg(l_extendedprice) as avg_price, "
"        avg(l_discount) as avg_disc, "
"        count(*) as count_order "
"from "
"        lineitem "
"where "
"        l_shipdate <= date '1998-12-01' - interval '90' day "
"group by "
"        l_returnflag, "
"        l_linestatus;";

const std::string query3_str =
"select "
"        l_orderkey, "
"        sum(l_extendedprice * (1 - l_discount)) as revenue, "
"        o_orderdate, "
"        o_shippriority "
"from "
"        customer, "
"        orders, "
"        lineitem "
"where "
"        c_mktsegment = 'BUILDING' "
"        and c_custkey = o_custkey "
"        and l_orderkey = o_orderkey "
"        and o_orderdate < date '1995-03-15' "
"        and l_shipdate > date '1995-03-15' "
"group by "
"        l_orderkey, "
"        o_orderdate, "
"        o_shippriority;";


const std::string query5opt_str =
"select "
"        c_nationkey, "
"        sum(l_extendedprice * (1 - l_discount)) as revenue "
"from "
"        customer, "
"        orders, "
"        lineitem, "
"        supplier "
"where "
"        c_custkey = o_custkey "
"        and l_orderkey = o_orderkey "
"        and l_suppkey = s_suppkey "
"        and c_nationkey = s_nationkey "
"        and s_nationkey in (8, 9, 12, 18, 21) "
"        and o_orderdate >= date '1994-01-01' "
"        and o_orderdate < date '1994-01-01' + interval '1' year "
"group by "
"        c_nationkey; ";

const std::string query5_str =
"select "
"        n_name, "
"        sum(l_extendedprice * (1 - l_discount)) as revenue "
"from "
"        customer, "
"        orders, "
"        lineitem, "
"        supplier, "
"        nation, "
"        region "
"where "
"        c_custkey = o_custkey "
"        and l_orderkey = o_orderkey "
"        and l_suppkey = s_suppkey "
"        and c_nationkey = s_nationkey "
"        and s_nationkey = n_nationkey "
"        and n_regionkey = r_regionkey "
"        and r_name = 'ASIA' "
"        and o_orderdate >= date '1994-01-01' "
"        and o_orderdate < date '1994-01-01' + interval '1' year "
"group by "
"        n_name; ";

std::unordered_map<std::string,std::string> query_strs = {
    {"query1",query1_str},
    {"query3",query3_str},
    {"query5",query5_str},
    {"query5opt",query5opt_str},
};

static std::string get_query_text(std::string query){
    if(query_strs.find(query) != query_strs.end()){
        return query_strs[query];
    }else {
        return "Got a wrong query input!";
    }
}