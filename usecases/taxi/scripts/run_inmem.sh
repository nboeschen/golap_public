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
scale_factor=60
dbpath=/mnt/labstore/nboeschen/taxi/trips.csv.dat
init_variant=init_only
repeat=2
store_offset=0
nvchunk=$((1<<16))

query="query1.1,query1.2,query1.3,query2.1,query2.2,query2.3"


comp_algos_gpu="-"
chunk_bytes="0"
dataflow="INMEM"
workers="32"
outfile="../results/inmem_query_$now.csv"

$command_prefix ../../../bin/taxi --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/taxi --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos_gpu --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
