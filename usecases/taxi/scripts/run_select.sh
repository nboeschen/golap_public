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
repeat=1
store_offset=0
workers=8
nvchunk=$((1<<16))
dataflow="SSD2GPU"

query="select_VendorID,select_tpep_pickup_datetime,select_tpep_dropoff_datetime,select_Passenger_count,select_Trip_distance,select_RateCodeID,select_Store_and_fwd_flag,select_PULocationID,select_DOLocationID,select_Payment_type,select_Fare_amount,select_Extra,select_MTA_tax,select_Tip_amount,select_Tolls_amount,select_Improvement_surcharge,select_Total_amount,select_Congestion_Surcharge,select_Airport_fee"
comp_algos_gpu="LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS"

# chunk_bytes="$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28))"
chunk_bytes="$((1<<18)),$((1<<20)),$((1<<22)),$((1<<24)),$((1<<26))"
comp_algos_gpu="LZ4,Cascaded,Snappy,Gdeflate,Bitcomp,ANS"
workers=16
outfile="../results/select_$now.csv"

$command_prefix ../../../bin/taxi --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/taxi --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos_gpu --cuda_device=$cuda_device --init_variant=$init_variant --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
