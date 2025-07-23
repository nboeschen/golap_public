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
# ssdpath="/raid/gds/300G.file"
ssdpath="/dev/nvme1n1"
# ssdpath="/dev/md0"
scale_factor=50
# dbpath="/mnt/labstore/nboeschen/ssb/diskdb${scale_factor}.dat"
dbpath=""
init_variant=init_only
repeat=1
store_offset=0
workers=8
nvchunk=$((1<<16))
sim_compute_us=-1
query="select_key,select_linenum,select_custkey,select_partkey,select_suppkey,select_orderdate,select_linenumber,select_orderpriority,select_quantity,select_extendedprice,select_ordtotalprice,select_discount,select_revenue,select_supplycost,select_tax,select_commitdate,select_shipmode"
outfile="../results/select_$now.csv"
comp_algos="UNCOMPRESSED,LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS"
customer_factor=1
chunk_bytes="$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28))"
# comp_algos="BEST_BW_COMP,BEST_RATIO_COMP,UNCOMPRESSED"
dataflow="SSD2GPU,SSD2GPU2CPU"
event_sync=true
sort_by="natural"
# aws_endpoint="http://10.0.2.15:9000"
aws_endpoint=""

##################################################

workers=8
chunk_bytes="$((1<<16)),$((1<<18)),$((1<<20)),$((1<<22)),$((1<<24)),$((1<<26))"
dataflow="SSD2GPU"
# sort_by="general_dimsort"
comp_algos="BEST_BW_COMP"

# chunk_bytes="$((1<<24)),$((1<<26)),$((1<<28))"
# comp_algos="UNCOMPRESSED,LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS"
# pipeline experiment
# dataflow="SSD2GPU,SSD2GPU2CPU"
# comp_algos="BEST_BW_COMP,UNCOMPRESSED"
# chunk_bytes="$((1<<20)),$((1<<22)),$((1<<24)),$((1<<26)),$((1<<28))"
# query="select_key,select_linenum,select_custkey,select_orderdate,select_orderpriority,select_revenue"
# workers="1,2,4,8,16"
# outfile="../results/workers_$now.csv"

# (almost) same as cpu experiment
# workers=8
# chunk_bytes="$((1<<20)),$((1<<22)),$((1<<24)),$((1<<26)),$((1<<28))"
# query="select_key,select_orderpriority,select_revenue,select_c_city"
# customer_factor=100
# dataflow="SSD2GPU,SSD2GPU2CPU"
# comp_algos="BEST_BW_COMP,UNCOMPRESSED"

# q3 customer setup
# customer_factor=50
# query="select_c_key,select_c_city,select_c_nation,select_c_region"
# dataflow="SSD2GPU"
# comp_algos="LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS,UNCOMPRESSED"

# uncompressed BW
# query="select_key,select_custkey,select_orderpriority,select_quantity"
# query="select_orderpriority"
# chunk_bytes="$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28)),$((1<<29))"
# comp_algos="UNCOMPRESSED,BEST_BW_COMP"
# dataflow="SSD2GPU,SSD2GPU2CPU"
# workers="1,2,4,8,16"

# same as sample
# chunk_bytes="$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28))"
# comp_algos="LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS"
# outfile="../results/select_all_algos_$now.csv"
# dataflow="SSD2GPU"

# chunk_bytes="$((1<<16)),$((1<<17)),$((1<<18)),$((1<<19)),$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27))"
# chunk_bytes="$((1<<16)),$((1<<18)),$((1<<20)),$((1<<22)),$((1<<24))"
# workers="16"
# comp_algos="UNCOMPRESSED,BEST_BW_COMP"
# query="select_orderpriority,select_commitdate,select_key"
# dataflow="SSD2GPU,SSD2CPU2GPU"
# outfile="../results/select_$now.csv"
# sim_compute_us=50
# event_sync=false


# chunk_bytes="$((1<<22)),$((1<<24)),$((1<<26))"
chunk_bytes="$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28))"
workers="8"
comp_algos="BEST_BW_COMP"
query="select_orderpriority,select_commitdate,select_key,select_discount"
dataflow="S32CPU2GPU"
outfile="../results/select_s3_remote_$now.csv"

# with and without GPUDirect
# dataflow="SSD2GPU,SSD2GPU2CPU,SSD2CPU2GPU,SSD2CPU"
# comp_algos="UNCOMPRESSED"
# query="select_orderpriority"
# outfile="../results/select_GDS_$now.csv"
# chunk_bytes="$((1<<18)),$((1<<19)),$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26))"

$command_prefix ../../../bin/ssb --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/ssb --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --customer_factor=$customer_factor --sim_compute_us=$sim_compute_us --event_sync=$event_sync --sort_by=$sort_by --aws_endpoint=$aws_endpoint"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile