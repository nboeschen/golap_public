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
cuda_device=0
# ssdpath="/dev/nvme0n1"
ssdpath="/raid/gds/300G.file"
# ssdpath="/dev/md0"
scale_factor=200
dbpath=/mnt/labstore/nboeschen/ssb/diskdb200.dat
init_variant=init_only
repeat=2
store_offset=0
workers=32
nvchunk=$((1<<16))
# query="select_key,select_linenum,select_custkey,select_partkey,select_suppkey,select_orderdate,select_linenumber,select_orderpriority,select_quantity,select_extendedprice,select_ordtotalprice,select_discount,select_revenue,select_supplycost,select_tax,select_commitdate,select_shipmode,select_c_key,select_c_city,select_c_nation,select_c_region"
query="select_key,select_linenum,select_custkey,select_partkey,select_suppkey,select_orderdate,select_linenumber,select_orderpriority,select_quantity,select_extendedprice,select_ordtotalprice,select_discount,select_revenue,select_supplycost,select_tax,select_commitdate,select_shipmode"
# query="select_key,select_orderpriority,select_revenue,select_c_city"

# chunk_bytes="$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28)),$((1<<29))"
# chunk_bytes="$((1<<14)),$((1<<16)),$((1<<18)),$((1<<20)),$((1<<22)),$((1<<24)),$((1<<26)),$((1<<28))"
chunk_bytes="$((1<<16)),$((1<<17)),$((1<<18)),$((1<<19)),$((1<<20))"

# CPU setup
customer_factor=1
dataflow="SSD2CPU"
comp_algos="LZ4,Snappy,UNCOMPRESSED"
outfile="../results/select_cpu_$now.csv"


$command_prefix ../../../bin/ssb --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/ssb --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --customer_factor=$customer_factor"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
