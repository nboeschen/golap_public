USE TPCH
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED
SET STATISTICS IO on
SET STATISTICS time on
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

-- Q1
-- TPC-H Query 1
select 
        l_returnflag, 
        l_linestatus, 
        sum(l_quantity) as sum_qty, 
        sum(l_extendedprice) as sum_base_price, 
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, 
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, 
        avg(l_quantity) as avg_qty, 
        avg(l_extendedprice) as avg_price, 
        avg(l_discount) as avg_disc, 
        count(*) as count_order 
from 
        lineitem 
where 
        l_shipdate <= DATEADD(day, -90, cast('1998-12-01' as date)) 
group by 
        l_returnflag, 
        l_linestatus;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

-- Q3
select 
        top 100000 
        l_orderkey, 
        sum(l_extendedprice * (1 - l_discount)) as revenue, 
        o_orderdate, 
        o_shippriority 
from 
        customer, 
        orders, 
        lineitem 
where 
        c_mktsegment = 'BUILDING' 
        and c_custkey = o_custkey 
        and l_orderkey = o_orderkey 
        and o_orderdate < cast('1995-03-15' as date) 
        and l_shipdate > cast('1995-03-15' as date) 
group by 
        l_orderkey, 
        o_orderdate, 
        o_shippriority;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

-- Q5
-- TPC-H Query 5 (with nation and region optimization)
select 
        c_nationkey, 
        sum(l_extendedprice * (1 - l_discount)) as revenue 
from 
        customer, 
        orders, 
        lineitem, 
        supplier 
where 
        c_custkey = o_custkey 
        and l_orderkey = o_orderkey 
        and l_suppkey = s_suppkey 
        and c_nationkey = s_nationkey 
        and c_nationkey in (8,9,12,18,21) 
        and o_orderdate >= cast('1994-01-01' as date) 
        and o_orderdate < DATEADD(year, 1, cast('1994-01-01' as date)) 
group by 
        c_nationkey;
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO
