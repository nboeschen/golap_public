#!/bin/bash
# set -x

# go to this files directory
DIR="$(dirname $(readlink -f "$0"))"
cd $DIR

scale_factor=200
row_group_size=1048576
duckdb_exec="/home/nboeschen/duckdb"
parquet_folder="/raid/gds/ssb/parquet${scale_factor}/"
encodings=("snappy" "uncompressed")

for enc in "${encodings[@]}"; do
    mkdir -p "${parquet_folder}/${enc}_pre_aggr"
    SQL="
    IMPORT database '$parquet_folder/uncompressed/';

    COPY (SELECT d_year, d_weeknuminyear, d_monthnuminyear, lo_suppkey, sum(lo_revenue) as rev_per_week FROM lineorder, date WHERE lo_orderdate = d_key GROUP BY (d_year, d_weeknuminyear, d_monthnuminyear, lo_suppkey)) to '${parquet_folder}/${enc}_pre_aggr/lineorder.parquet' (FORMAT 'PARQUET', CODEC '${enc^^}', ROW_GROUP_SIZE ${row_group_size});
    "
    echo $SQL
    $duckdb_exec -c "$SQL"

    # link the supplier parquet file
    ln -s "${parquet_folder}/${enc}/supplier.parquet" "${parquet_folder}/${enc}_pre_aggr/supplier.parquet"
    # create the schema and load files
    printf "
    CREATE VIEW lineorder as SELECT * FROM read_parquet('${parquet_folder}/${enc}_pre_aggr/lineorder.parquet');
    CREATE VIEW supplier as SELECT * FROM read_parquet('${parquet_folder}/${enc}_pre_aggr/supplier.parquet');

    " > "${parquet_folder}/${enc}_pre_aggr/schema.sql"

    touch "${parquet_folder}/${enc}_pre_aggr/load.sql"
done





# SELECT COUNT(*),sum(rev_per_week), d_year
# from (
#         SELECT d_year, d_weeknuminyear, lo_suppkey, sum(lo_revenue) as rev_per_week FROM lineorder, date WHERE lo_orderdate = d_datekey GROUP BY (d_year, d_weeknuminyear, lo_suppkey)
# ), supplier
# where lo_suppkey = s_suppkey and s_region = 'AMERICA'
# group by d_year;


# SELECT COUNT(*),sum(lo_revenue), d_year
# from lineorder, supplier, date
# where lo_suppkey = s_suppkey and lo_orderdate = d_datekey and s_region = 'AMERICA'
# group by d_year;


# SELECT sum(rev_per_week), d_year from lineorder, supplier where lo_suppkey = s_key and s_region = 'AMERICA' group by d_year;  