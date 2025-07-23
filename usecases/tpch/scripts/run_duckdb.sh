#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "No socket/cuda device supplied"
    exit 1
fi

command_prefix=""
now=$(date +"%Y-%m-%dT%H.%M.%S")


cd "$(dirname $(readlink -f "$0"))/../"
mkdir -p "build"
mkdir -p "results"
cd "build"

# Experiment parameters
repeat=5
mem_limit=$((1<<38))
thread_limit=64
scale_factor=200
encodings=("snappy" "zstd" "gzip" "uncompressed")


outfile="../results/duckdb_query_$now.csv"
query="query1,query3,query5"
# query="query5"

# outfile="../results/duckdb_count_$now.csv"
# query="count_key,count_linenum,count_custkey,count_partkey,count_suppkey,count_orderdate,count_linenumber,count_orderpriority,count_shippriority,count_quantity,count_extendedprice,count_ordtotalprice,count_discount,count_revenue,count_supplycost,count_tax,count_commitdate,count_shipmode"

$command_prefix ../../../bin/duckdb_tpch --print=csv_header >> $outfile

for enc in "${encodings[@]}"; do
    dbpath="/raid/gds/tpch/parquet${scale_factor}/${enc}/"
    command="$command_prefix ../../../bin/duckdb_tpch --repeat=$repeat --dbpath=$dbpath --query=$query --mem_limit=$mem_limit --thread_limit=$thread_limit"
    echo "#$command" >> $outfile
    $command >> $outfile
done

