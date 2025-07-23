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
cuda_device="$1"
# ssdpath="/dev/nvme0n1"
ssdpath="/raid/gds/300G.file"
scale_factor=100
dbpath=/mnt/labstore/nboeschen/tpch/csv200/orders.csv
csv_delimiter="|"
init_variant=init_only
repeat=1
store_offset=0
workers=16
nvchunk=$((1<<16))
dataflow="SSD2GPU"

query="select_l_orderkey,select_l_partkey,select_l_suppkey,select_l_linenumber,select_l_quantity,select_l_extendedprice,select_l_discount,select_l_tax,select_l_returnflag,select_l_linestatus,select_l_shipdate,select_l_commitdate,select_l_receiptdate,select_l_shipinstruct,select_l_shipmode"
comp_algos_gpu="UNCOMPRESSED,LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS"

# chunk_bytes="$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28))"
chunk_bytes="$((1<<22)),$((1<<24)),$((1<<26))"
outfile="../results/select_$now.csv"

$command_prefix ../../../bin/tpch --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/tpch --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos_gpu --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --csv_delimiter=$csv_delimiter"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
