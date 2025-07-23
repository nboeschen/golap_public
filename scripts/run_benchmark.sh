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
# ssdpath="/dev/nvme1n1"
# ssdpath="/dev/md0"
scale_factor=6
repeat=1
store_offset=0
workers=8
nvchunk=$((1<<16))
sim_compute_us=-1
# query="uniform,iota,uniform200,uniform100"
query="uniform,iota,uniform200"
outfile="../results/benchmark_$now.csv"
comp_algos="UNCOMPRESSED,LZ4,Gdeflate,Cascaded,Bitcomp,ANS,Snappy"
chunk_bytes="$((1<<22)),$((1<<24)),$((1<<26))"
dataflow="SSD2GPU"
event_sync=true
sort_by="natural"
aws_endpoint="http://172.18.94.40:9000"

##################################################
outfile="../results/benchmark_$now.csv"

# test ratio
scale_factor=1
query="ratio5"
workers="1"
chunk_bytes="$((1<<14))"
comp_algos="Gdeflate"
# test ratio


$command_prefix ../bin/test_benchmark --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../bin/test_benchmark --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos --cuda_device=$cuda_device --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --store_offset=$store_offset --workers=$workers --sim_compute_us=$sim_compute_us --event_sync=$event_sync --sort_by=$sort_by --aws_endpoint=$aws_endpoint"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile