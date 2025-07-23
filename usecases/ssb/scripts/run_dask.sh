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

export LIBCUDF_NVCOMP_POLICY="ALWAYS"
export LIBCUDF_CUFILE_POLICY="GDS"

# Experiment parameters
cuda_device="$1"
repeat=2
query="query1.1,query1.2,query1.3,query2.1,query2.2,query2.3,query3.1,query3.2,query3.3,query3.4,query4.1,query4.2,query4.3"
scale_factor=200
split_row_groups=128
encodings="uncompressed,snappy,zstd,gzip"
proc="GPU"
# proc="CPU"

outfile="../results/dask_$now.csv"

$command_prefix python ../dask_ssb.py --csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi



command="$command_prefix python ../dask_ssb.py --repeat=$repeat --encoding=$encodings --proc=$proc --split_row_groups=$split_row_groups --scale_factor=$scale_factor --query=$query"
echo "#$command" >> $outfile
$command >> $outfile



