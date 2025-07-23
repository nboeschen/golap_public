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
scale_factor=200
dbpath=/mnt/labstore/nboeschen/ssb/diskdb200.dat
init_variant=init_only
repeat=1
store_offset=0
workers=8
extra_workers=8
add_cpu_threads=32
customer_factor="50,100"
nvchunk=$((1<<16))
max_gpu_um_memory="0"

comp_algos="BEST_BW_COMP"
chunk_bytes="$((1<<18)),$((1<<19)),$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24))"
dataflow="SSD2GPU2CPU"
# outfile="../results/coexec_$now.csv"

# normal setup
query="query3.1,query3.2,query3.3,query3.4,query3.1a,query3.2a,query3.3a,query3.4a,query3.1c,query3.2c,query3.3c,query3.4c,query3.1d,query3.2d,query3.3d,query3.4d"
query="query3.1a,query3.2a,query3.3a,query3.4a,query3.1c,query3.2c,query3.3c,query3.4c,query3.1d,query3.2d,query3.3d,query3.4d"
outfile="../results/query3_coexec_oom_$now.csv"


# inmem handcoded
# dataflow="INMEM"
# chunk_bytes="0"
# workers=16
# query="query3.1,query3.2,query3.3,query3.4"

# filter on cpu
# query="query3.1b,query3.2b,query3.3b,query3.4b"
# outfile="../results/query3b_$now.csv"

# query 3cd:
# workers=8
# extra_workers=4
# query="query3.1d,query3.2d,query3.3d,query3.4d,query3.1c,query3.2c,query3.3c,query3.4c"
# max_gpu_um_memory="$((1<<28)),$((1<<30)),0"
# outfile="../results/query3cd_$now.csv"

# query 3d:
# workers=8
# extra_workers=32
# query="query3.1d,query3.2d,query3.3d,query3.4d"
# outfile="../results/query3d_$now.csv"


$command_prefix ../../../bin/ssb --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/ssb --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --extra_workers=$extra_workers --add_cpu_threads=$add_cpu_threads --customer_factor=$customer_factor --max_gpu_um_memory=$max_gpu_um_memory"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
