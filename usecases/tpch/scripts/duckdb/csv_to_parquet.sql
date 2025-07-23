CREATE TABLE IF NOT EXISTS customer (
    c_custkey UBIGINT primary key,
    c_name VARCHAR(25),
    c_address VARCHAR(40),
    c_nationkey UBIGINT,
    c_phone VARCHAR(15),
    c_acctbal DECIMAL(18, 2),
    c_mktsegment VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS lineitem (
    l_orderkey UBIGINT,
    l_partkey UBIGINT,
    l_suppkey UBIGINT,
    l_linenumber BIGINT,
    l_quantity BIGINT,
    l_extendedprice DECIMAL(18, 2),
    l_discount DECIMAL(18, 2),
    l_tax DECIMAL(18, 2),
    l_returnflag VARCHAR(1),
    l_linestatus VARCHAR(1),
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct VARCHAR(20),
    l_shipmode VARCHAR(20),
    primary key(l_orderkey,l_linenumber)
);

CREATE TABLE IF NOT EXISTS nation (
    n_nationkey UBIGINT primary key,
    n_name VARCHAR(15),
    n_regionkey UBIGINT
);

CREATE TABLE IF NOT EXISTS orders (
    o_orderkey UBIGINT primary key,
    o_custkey UBIGINT,
    o_orderstatus VARCHAR(1),
    o_totalprice DECIMAL(18, 2),
    o_orderdate DATE,
    o_orderpriority VARCHAR(20),
    o_clerk VARCHAR(15),
    o_shippriority BIGINT
);

CREATE TABLE IF NOT EXISTS part (
    p_partkey UBIGINT primary key,
    p_name VARCHAR(55),
    p_mfgr VARCHAR(25),
    p_brand VARCHAR(10),
    p_type VARCHAR(25),
    p_size BIGINT,
    p_container VARCHAR(10),
    p_retailprice DECIMAL(18, 2)
);

CREATE TABLE IF NOT EXISTS partsupp (
    ps_partkey UBIGINT,
    ps_suppkey UBIGINT,
    ps_availqty BIGINT,
    ps_supplycost DECIMAL(18, 2),
    primary key(ps_partkey,ps_suppkey)
);

CREATE TABLE IF NOT EXISTS region (
    r_regionkey UBIGINT primary key,
    r_name VARCHAR(12)
);

CREATE TABLE IF NOT EXISTS supplier (
    s_suppkey UBIGINT primary key,
    s_name VARCHAR(25),
    s_address VARCHAR(40),
    s_nationkey UBIGINT,
    s_phone VARCHAR(15),
    s_acctbal DECIMAL(18, 2)
);

SET threads TO 16;
SET temp_directory = '/tmp/duck.tmp/';
SET preserve_insertion_order = false;
SET enable_progress_bar = true;

COPY customer FROM 'CSV_FOLDER/customer.tbl' (DELIMITER '|');
COPY lineitem FROM 'CSV_FOLDER/lineitem.tbl' (DELIMITER '|');
COPY nation FROM 'CSV_FOLDER/nation.tbl' (DELIMITER '|');
COPY orders FROM 'CSV_FOLDER/orders.tbl' (DELIMITER '|');
COPY part FROM 'CSV_FOLDER/part.tbl' (DELIMITER '|');
COPY partsupp FROM 'CSV_FOLDER/partsupp.tbl' (DELIMITER '|');
COPY region FROM 'CSV_FOLDER/region.tbl' (DELIMITER '|');
COPY supplier FROM 'CSV_FOLDER/supplier.tbl' (DELIMITER '|');
