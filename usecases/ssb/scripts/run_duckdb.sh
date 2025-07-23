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
explain_analyze="false"
encodings=("snappy" "zstd" "gzip" "uncompressed")
query="query1.1,query1.2,query1.3,query2.1,query2.2,query2.3,query3.1,query3.2,query3.3,query3.4,query4.1,query4.2,query4.3"

# scan only, different row group sizes
outfile="../results/duckdb_scan_$now.csv"
encodings=("snappy" "uncompressed")
row_group_sizes=("61440" "122880" "524288" "2097152")
query="scan_only"



# query on sorted parquet files
# query="query1.1,query1.2,query1.3,query2.1,query2.2,query2.3,query3.1,query3.2,query3.3,query3.4,query4.1,query4.2,query4.3"
# outfile="../results/duckdb_query_pruning_$now.csv"
# encodings=("uncompressed_sorted" "snappy_sorted")
# outfile="../results/duckdb_query_$now.csv"
# encodings=("uncompressed" "snappy")


# count queries
# outfile="../results/duckdb_count_$now.csv"
# query="count_key,count_linenum,count_custkey,count_partkey,count_suppkey,count_orderdate,count_linenumber,count_orderpriority,count_shippriority,count_quantity,count_extendedprice,count_ordtotalprice,count_discount,count_revenue,count_supplycost,count_tax,count_commitdate,count_shipmode"

# pre_aggr queries
# outfile="../results/duckdb_pre_aggr_$now.csv"
# encodings=("snappy_pre_aggr" "uncompressed_pre_aggr")
# query="query2.1_pre_aggr,query2.2_pre_aggr"

$command_prefix ../../../bin/duckdb_ssb1.0 --print=csv_header >> $outfile

for enc in "${encodings[@]}"; do
    dbpath="/raid/gds/ssb/parquet${scale_factor}/${enc}/"
    command="$command_prefix ../../../bin/duckdb_ssb1.0 --repeat=$repeat --dbpath=$dbpath --query=$query --mem_limit=$mem_limit --thread_limit=$thread_limit --explain_analyze=$explain_analyze"
    echo "#$command" >> $outfile
    $command >> $outfile
done

# additional row group sizes
for row_group_size in "${row_group_sizes[@]}"; do
    for enc in "${encodings[@]}"; do
        dbpath="/raid/gds/ssb/parquet${scale_factor}/${enc}_${row_group_size}/"
        command="$command_prefix ../../../bin/duckdb_ssb --repeat=$repeat --dbpath=$dbpath --query=$query --mem_limit=$mem_limit --thread_limit=$thread_limit --explain_analyze=$explain_analyze"
        echo "#$command" >> $outfile
        $command >> $outfile
    done
done

echo "#DONE" >> $outfile