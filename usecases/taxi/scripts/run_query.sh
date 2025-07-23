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
# ssdpath="/raid/gds/300G.file"
ssdpath="/dev/md0"
scale_factor=60
dbpath=/home/ubuntu/trips.csv.dat
init_variant=init_only
repeat=3
store_offset=0
workers=6
nvchunk=$((1<<16))
dataflow="SSD2GPU2CPU"

query="query1.1,query1.2,query1.3,query2.1,query2.2,query2.3"
# query="query1.1"

comp_algos_gpu="BEST_BW_COMP,UNCOMPRESSED"

# outfile="../results/query_$now.csv"

chunk_bytes="$((1<<22)),$((1<<24)),$((1<<26))"
# workers="1,2,4,8"
outfile="../results/query12_$now.csv"

###
# dataflow="INMEM"
# comp_algos_gpu="UNCOMPRESSED"
# chunk_bytes="$((1<<20))"
# workers="4,8,16,32"
# outfile="../results/query1_inmem_$now.csv"
###


$command_prefix ../../../bin/taxi --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/taxi --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos_gpu --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
