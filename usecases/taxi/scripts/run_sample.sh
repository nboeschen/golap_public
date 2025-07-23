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
scale_factor=60
dbpath=/mnt/labstore/nboeschen/taxi/trips.csv.dat
init_variant=init_only
repeat=2
store_offset=0
workers=8
nvchunk=$((1<<16))
outfile="../results/sample_$now.csv"
# chunk_bytes="$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28)),$((1<<29)),$((1<<30))"
chunk_bytes="$((1<<16)),$((1<<18)),$((1<<20)),$((1<<22)),$((1<<24)),$((1<<26)),$((1<<28))"
comp_algos="LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS"
columns="VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,Passenger_count,Trip_distance,RateCodeID,Store_and_fwd_flag,PULocationID,DOLocationID,Payment_type,Fare_amount,Extra,MTA_tax,Tip_amount,Tolls_amount,Improvement_surcharge,Total_amount,Congestion_Surcharge,Airport_fee"
sample_ratio="0.01,0.05,0.1,0.2"

$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/sample_taxi --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --columns=$columns --sample_ratio=$sample_ratio"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
