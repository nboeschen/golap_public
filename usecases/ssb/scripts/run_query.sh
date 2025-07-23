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
# ssdpath="/dev/md0"
scale_factor=200
dbpath="/mnt/labstore/nboeschen/ssb/diskdb100.dat"
# dbpath=""
# dbpath="/home/ubuntu/diskdb10.dat"
# init_variant="init_only"
repeat=2
store_offset=0
workers=8
nvchunk=$((1<<16))
pre_aggr_path=""
csv_delimiter=","

# query="query1.1,query1.2,query1.3"
query="query1.1,query1.2,query1.3,query2.1,query2.2,query2.3,query3.1,query3.2,query3.3,query3.4,query4.1,query4.2,query4.3"
# query="query3.1,query3.2,query3.3,query3.4"

# query="query2.1,query2.1spd,query2.1sdp,query2.1dps,query2.1dsp,query2.2,query2.2spd,query2.2sdp,query2.2dps,query2.2dsp,query2.3,query2.3spd,query2.3sdp,query2.3dps,query2.3dsp"

comp_algos_gpu="UNCOMPRESSED,Gdeflate,Bitcomp"
chunk_bytes="$((1<<20)),$((1<<22)),$((1<<24)),$((1<<26))"
pre_aggr_path="/mnt/labstore/nboeschen/ssb/csv200_pre_aggr/pre_aggr.csv"
query="query2.1_pre_aggr,query2.2_pre_aggr"
dataflow="SSD2GPU2CPU"
outfile="../results/query_pre_aggr_$now.csv"

# comp_algos_gpu="UNCOMPRESSED,BEST_BW_COMP"
# query="query1.1,query1.2,query1.3"
# dataflow="SSD2GPU2CPU"
# chunk_bytes="$((1<<22)),$((1<<24)),$((1<<26))"
# outfile="../results/query_$now.csv"

# comp_algos_gpu="BEST_BW_COMP"
# query="query1.1,query1.2,query1.3"
# chunk_bytes="$((1<<20)),$((1<<21)),$((1<<22)),$((1<<23)),$((1<<24)),$((1<<25)),$((1<<26)),$((1<<27)),$((1<<28))"
# dataflow="SSD2GPU2CPU"
# workers="1,2,4,8"
# outfile="../results/query1_many_chunk_$now.csv"

$command_prefix ../../../bin/ssb --print=csv_header >> $outfile
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

command="$command_prefix ../../../bin/ssb --dataflow=$dataflow --chunk_bytes=$chunk_bytes --comp_algo=$comp_algos_gpu --cuda_device=$cuda_device --nvchunk=$nvchunk --print=csv --query=$query --repeat=$repeat --scale_factor=$scale_factor --ssdpath=$ssdpath --dbpath=$dbpath --store_offset=$store_offset --workers=$workers --pre_aggr_path=$pre_aggr_path --csv_delimiter=$csv_delimiter"
echo "#$command" >> $outfile
$command >> $outfile


echo "#DONE" >> $outfile
