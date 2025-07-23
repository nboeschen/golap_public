USE SSB
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED
SET STATISTICS IO on
SET STATISTICS time on
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE

-- Q1.1
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder
left join date on lo_orderdate = d_key
where d_year = 1993
and lo_discount between 1 and 3
and lo_quantity < 25;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE

-- Q1.2

select sum(lo_extendedprice*lo_discount) as revenue
from lineorder
left join date on lo_orderdate = d_key
where d_yearmonthnum = 199401
and lo_discount between 4 and 6
and lo_quantity between 26 and 35;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q1.3

select sum(lo_extendedprice*lo_discount) as revenue
from lineorder
left join date on lo_orderdate = d_key
where d_weeknuminyear = 6 and d_year = 1994
and lo_discount between 5 and 7
and lo_quantity between 26 and 35;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q2.1

select sum(lo_revenue) as lo_revenue, d_year, p_brand1
from lineorder
left join date on lo_orderdate = d_key
left join part on lo_partkey = p_key
left join supplier on lo_suppkey = s_key
where p_category = 'MFGR#12' and s_region = 'AMERICA'
group by d_year, p_brand1
order by d_year, p_brand1;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q2.2

select sum(lo_revenue) as lo_revenue, d_year, p_brand1
from lineorder
left join date on lo_orderdate = d_key
left join part on lo_partkey = p_key
left join supplier on lo_suppkey = s_key
where p_brand1 between 'MFGR#2221' and 'MFGR#2228' and s_region = 'ASIA'
group by d_year, p_brand1
order by d_year, p_brand1;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q2.3

select sum(lo_revenue) as lo_revenue, d_year, p_brand1
from lineorder
left join date on lo_orderdate = d_key
left join part on lo_partkey = p_key
left join supplier on lo_suppkey = s_key
where p_brand1 = 'MFGR#2239' and s_region = 'EUROPE'
group by d_year, p_brand1
order by d_year, p_brand1;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q3.1

select c_nation, s_nation, d_year, sum(lo_revenue) as lo_revenue
from lineorder
left join date on lo_orderdate = d_key
left join customer on lo_custkey = c_key
left join supplier on lo_suppkey = s_key
where c_region = 'ASIA' and s_region = 'ASIA'and d_year >= 1992 and d_year <= 1997
group by c_nation, s_nation, d_year
order by d_year asc, lo_revenue desc;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q3.2

select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue
from lineorder
left join date on lo_orderdate = d_key
left join customer on lo_custkey = c_key
left join supplier on lo_suppkey = s_key
where c_nation = 'UNITED STATES' and s_nation = 'UNITED STATES'
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc, lo_revenue desc;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q3.3

select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue
from lineorder
left join date on lo_orderdate = d_key
left join customer on lo_custkey = c_key
left join supplier on lo_suppkey = s_key
where (c_city='UNITED KI1' or c_city='UNITED KI5')
and (s_city='UNITED KI1' or s_city='UNITED KI5')
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc, lo_revenue desc;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q3.4

select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue
from lineorder
left join date on lo_orderdate = d_key
left join customer on lo_custkey = c_key
left join supplier on lo_suppkey = s_key
where (c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_yearmonth = 'Dec1997'
group by c_city, s_city, d_year
order by d_year asc, lo_revenue desc;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q4.1

select d_year, c_nation, sum(lo_revenue) - sum(lo_supplycost) as profit
from lineorder
left join date on lo_orderdate = d_key
left join customer on lo_custkey = c_key
left join supplier on lo_suppkey = s_key
left join part on lo_partkey = p_key
where c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
group by d_year, c_nation
order by d_year, c_nation;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q4.2

select d_year, s_nation, p_category, sum(lo_revenue) - sum(lo_supplycost) as profit
from lineorder
left join date on lo_orderdate = d_key
left join customer on lo_custkey = c_key
left join supplier on lo_suppkey = s_key
left join part on lo_partkey = p_key
where c_region = 'AMERICA' and s_region = 'AMERICA'
and (d_year = 1997 or d_year = 1998)
and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
group by d_year, s_nation, p_category
order by d_year, s_nation, p_category;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
-- Q4.3

select d_year, s_city, p_brand1, sum(lo_revenue) - sum(lo_supplycost) as profit
from lineorder
left join date on lo_orderdate = d_key
left join customer on lo_custkey = c_key
left join supplier on lo_suppkey = s_key
left join part on lo_partkey = p_key
where c_region = 'AMERICA' and s_nation = 'UNITED STATES'
and (d_year = 1997 or d_year = 1998)
and p_category = 'MFGR#14'
group by d_year, s_city, p_brand1
order by d_year, s_city, p_brand1;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE

