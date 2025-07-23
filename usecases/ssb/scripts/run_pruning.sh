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
ssdpath="/dev/md0"
scale_factor=100
# dbpath="/mnt/labstore/nboeschen/ssb/diskdb${scale_factor}.dat"
dbpath="/home/ubuntu/diskdb100.dat"
init_variant=init_only
repeat=2
store_offset=0
workers=8
nvchunk=$((1<<16))
dataflow="SSD2GPU2CPU"
chunk_bytes="$((1<<18)),$((1<<19)),$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24))"
comp_algos="UNCOMPRESSED,BEST_BW_COMP"
query="query1.1,query1.2,query1.3"
# sort_by="natural,lo_discount,lo_quantity"
# sort_by="lo_quantity|lo_discount,lo_discount|lo_quantity"
sort_by="general_dimsort"
shuffle_ratio=0.0
pruning="DONTPRUNE,HIST"
# pruning="COMBINED,BLOOM"
pruning_param=128
pruning_m=2048
pruning_p=0.01
outfile="../results/pruning_$now.csv"


# sort_by="natural"
# dataflow="SSD2GPU"
# # query="filter_quantity"
# # 20-45, 30-40, 40-45, 10-12
# # col_filter_lo="20,30,40,10"
# # col_filter_hi="45,40,45,12"
# query="filter_extendedprice"
# col_filter_lo="90110,1000000,5000000"
# col_filter_hi="110000,1001000,5000100"
# pruning="HIST"
# pruning_param="1024"
# outfile="../results/pruning_lo_extendedprice_$now.csv"


$command_prefix ../../../bin/ssb --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/ssb --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --sort_by=$sort_by --pruning=$pruning --shuffle_ratio=$shuffle_ratio --pruning_param=$pruning_param --pruning_m=$pruning_m --pruning_p=$pruning_p --col_filter_lo=$col_filter_lo --col_filter_hi=$col_filter_hi"
echo "#$command" >> $outfile
$command >> $outfile

echo "#DONE" >> $outfile
