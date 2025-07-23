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
workers=8
nvchunk=$((1<<16))
dataflow="SSD2GPU2CPU"
sort_by="Fare_amount"
shuffle_ratio=0.0
pruning="HIST"
pruning_param=256
pruning_m=2048
pruning_p=0.01


comp_algos_gpu="UNCOMPRESSED,BEST_BW_COMP"
cluster_param_max_tuples=$((1<<22))
cluster_param_k=10000
cluster_param_rounds=3
# sort_by="cluster|Trip_distance|Fare_amount"
query="query2.1,query2.2,query2.3"
# chunk_bytes="-1"

chunk_bytes="$((1<<18)),$((1<<19)),$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24))"
outfile="../results/pruning_$now.csv"

$command_prefix ../../../bin/taxi --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/taxi --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos_gpu --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --sort_by=$sort_by --pruning=$pruning --shuffle_ratio=$shuffle_ratio --pruning_param=$pruning_param --pruning_m=$pruning_m --pruning_p=$pruning_p --cluster_param_max_tuples=$cluster_param_max_tuples --cluster_param_k=$cluster_param_k --cluster_param_rounds=$cluster_param_rounds"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
