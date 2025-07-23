
query1_1 = """
select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key 
and d_year = 1993 
and lo_discount between 1 and 3 
and lo_quantity < 25 """

query1_2 = """
select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key 
and d_yearmonthnum = 199401 
and lo_discount between 4 and 6 
and lo_quantity between 26 and 35"""

query1_3 = """
select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key 
and d_weeknuminyear = 6 
and d_year = 1994 
and lo_discount between 5 and 7 
and lo_quantity between 26 and 35"""

query2_1 = """
select sum(lo_revenue), d_year, p_brand1 
from lineorder, date, part, supplier 
where lo_orderdate = d_key 
and lo_partkey = p_key 
and lo_suppkey = s_key 
and p_category = 'MFGR#12' 
and s_region = 'AMERICA' 
group by d_year, p_brand1 """
# order by d_year, p_brand1;;

query2_2 = """
select sum(lo_revenue), d_year, p_brand1 
from lineorder, date, part, supplier 
where lo_orderdate = d_key 
and lo_partkey = p_key 
and lo_suppkey = s_key 
and p_brand1 between 'MFGR#2221' 
and 'MFGR#2228' 
and s_region = 'ASIA' 
group by d_year, p_brand1 """
# order by d_year, p_brand1;;

query2_3 = """
select sum(lo_revenue), d_year, p_brand1 
from lineorder, date, part, supplier 
where lo_orderdate = d_key 
and lo_partkey = p_key 
and lo_suppkey = s_key 
and p_brand1 = 'MFGR#2239' 
and s_region = 'EUROPE' 
group by d_year, p_brand1 """
# order by d_year, p_brand1;;

query3_1 = """
select c_nation, s_nation, d_year, sum(lo_revenue) as revenue 
from customer, lineorder, supplier, date 
where lo_custkey = c_key 
and lo_suppkey = s_key 
and lo_orderdate = d_key 
and c_region = 'ASIA' 
and s_region = 'ASIA' 
and d_year >= 1992 and d_year <= 1997 
group by c_nation, s_nation, d_year """
# order by d_year asc, revenue desc;;

query3_2 = """
select c_city, s_city, d_year, sum(lo_revenue) as revenue 
from customer, lineorder, supplier, date 
where lo_custkey = c_key 
and lo_suppkey = s_key 
and lo_orderdate = d_key 
and c_nation = 'UNITED STATES' 
and s_nation = 'UNITED STATES' 
and d_year >= 1992 and d_year <= 1997 
group by c_city, s_city, d_year """
# order by d_year asc, revenue desc;;

query3_3 = """
select c_city, s_city, d_year, sum(lo_revenue) as revenue 
from customer, lineorder, supplier, date 
where lo_custkey = c_key 
and lo_suppkey = s_key 
and lo_orderdate = d_key 
and (c_city='UNITED KI1' or c_city='UNITED KI5') 
and (s_city='UNITED KI1' or s_city='UNITED KI5') 
and d_year >= 1992 and d_year <= 1997 
group by c_city, s_city, d_year """
# order by d_year asc, revenue desc;;

query3_4 = """
select c_city, s_city, d_year, sum(lo_revenue) as revenue 
from customer, lineorder, supplier, date 
where lo_custkey = c_key 
and lo_suppkey = s_key 
and lo_orderdate = d_key 
and (c_city='UNITED KI1' or c_city='UNITED KI5') 
and (s_city='UNITED KI1' or s_city='UNITED KI5') 
and d_yearmonth = 'Dec1997' 
group by c_city, s_city, d_year """
# order by d_year asc, revenue desc;;

query4_1 = """
select d_year, c_nation, 
sum(lo_revenue - lo_supplycost) as profit 
from date, customer, supplier, part, lineorder 
where lo_custkey = c_key 
and lo_suppkey = s_key 
and lo_partkey = p_key 
and lo_orderdate = d_key 
and c_region = 'AMERICA' 
and s_region = 'AMERICA' 
and (p_mfgr = 'MFGR#1' 
or p_mfgr = 'MFGR#2') 
group by d_year, c_nation """
# order by d_year, c_nation; 

query4_2 = """
select d_year, s_nation, p_category, 
sum(lo_revenue - lo_supplycost) as profit 
from date, customer, supplier, part, lineorder 
where lo_custkey = c_key 
and lo_suppkey = s_key 
and lo_partkey = p_key 
and lo_orderdate = d_key 
and c_region = 'AMERICA' 
and s_region = 'AMERICA' 
and (d_year = 1997 or d_year = 1998) 
and (p_mfgr = 'MFGR#1' 
or p_mfgr = 'MFGR#2') 
group by d_year, s_nation, p_category """
# order by d_year, s_nation, p_category; 

query4_3 = """
select d_year, s_city, p_brand1, 
sum(lo_revenue - lo_supplycost) as profit 
from date, customer, supplier, part, lineorder 
where lo_custkey = c_key 
and lo_suppkey = s_key 
and lo_partkey = p_key 
and lo_orderdate = d_key 
and c_region = 'AMERICA' 
and s_nation = 'UNITED STATES' 
and (d_year = 1997 or d_year = 1998) 
and p_category = 'MFGR#14' 
group by d_year, s_city, p_brand1 """
# order by d_year, s_city, p_brand1; 


QUERY_STR = {
            "query1.1":query1_1.replace(" date", " date_table"),
            "query1.2":query1_2.replace(" date", " date_table"),
            "query1.3":query1_3.replace(" date", " date_table"),
            "query2.1":query2_1.replace(" date", " date_table"),
            "query2.2":query2_2.replace(" date", " date_table"),
            "query2.3":query2_3.replace(" date", " date_table"),
            "query3.1":query3_1.replace(" date", " date_table"),
            "query3.2":query3_2.replace(" date", " date_table"),
            "query3.3":query3_3.replace(" date", " date_table"),
            "query3.4":query3_4.replace(" date", " date_table"),
            "query4.1":query4_1.replace(" date", " date_table"),
            "query4.2":query4_2.replace(" date", " date_table"),
            "query4.3":query4_3.replace(" date", " date_table"),
            }