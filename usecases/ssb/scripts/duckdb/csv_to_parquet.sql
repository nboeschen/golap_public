CREATE TABLE IF NOT EXISTS lineorder (
    lo_key UBIGINT,
    lo_linenum UTINYINT,
    lo_custkey UBIGINT,
    lo_partkey UBIGINT,
    lo_suppkey UBIGINT,
    lo_orderdate UBIGINT,
    lo_linenumber UTINYINT,
    lo_orderpriority VARCHAR(15),
    lo_shippriority VARCHAR(1),
    lo_quantity UTINYINT,
    lo_extendedprice UINTEGER,
    lo_ordtotalprice UINTEGER,
    lo_discount UTINYINT,
    lo_revenue UINTEGER,
    lo_supplycost UINTEGER,
    lo_tax UTINYINT,
    lo_commitdate UBIGINT,
    lo_shipmode VARCHAR(16),
    primary key(lo_key,lo_linenumber)
);

CREATE TABLE IF NOT EXISTS part (
    p_key UBIGINT primary key,
    p_name VARCHAR(22),
    p_mfgr VARCHAR(6),
    p_category VARCHAR(7),
    p_brand1 VARCHAR(9),
    p_color VARCHAR(11),
    p_type VARCHAR(25),
    p_size UTINYINT,
    p_container VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS supplier (
    s_key UBIGINT primary key,
    s_name VARCHAR(25),
    s_address VARCHAR(25),
    s_city VARCHAR(10),
    s_nation VARCHAR(15),
    s_region VARCHAR(12),
    s_phone VARCHAR(15)
);

CREATE TABLE IF NOT EXISTS customer (
    c_key UBIGINT primary key,
    c_name VARCHAR(25),
    c_address VARCHAR(25),
    c_city VARCHAR(10),
    c_nation VARCHAR(15),
    c_region VARCHAR(12),
    c_phone VARCHAR(15),
    c_mktsegment VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS date (
    d_key UBIGINT primary key,
    d_date VARCHAR(18),
    d_dayofweek VARCHAR(9),
    d_month VARCHAR(9),
    d_year USMALLINT,
    d_yearmonthnum UINTEGER,
    d_yearmonth VARCHAR(7),
    d_daynuminweek USMALLINT,
    d_daynuminmonth USMALLINT,
    d_daynuminyear USMALLINT,
    d_monthnuminyear USMALLINT,
    d_weeknuminyear USMALLINT,
    d_sellingseason VARCHAR(12),
    d_lastdayinweekfl UTINYINT,
    d_lastdayinmonthfl UTINYINT,
    d_holidayfl UTINYINT,
    d_weekdayfl UTINYINT
);

SET threads TO 16;
SET temp_directory = '/tmp/duck.tmp/';
SET preserve_insertion_order = false;
SET enable_progress_bar = true;

COPY part FROM 'CSV_FOLDER/part.csv' (DELIMITER ';',HEADER 1);
COPY supplier FROM 'CSV_FOLDER/supplier.csv' (DELIMITER ';',HEADER 1);
COPY customer FROM 'CSV_FOLDER/customer.csv' (DELIMITER ';',HEADER 1);
COPY date FROM 'CSV_FOLDER/date.csv' (DELIMITER ';',HEADER 1);
COPY lineorder FROM 'CSV_FOLDER/lineorder.csv' (DELIMITER ';',HEADER 1);
