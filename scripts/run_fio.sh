#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "No socket/cuda device supplied"
    exit 1
fi

command_prefix=""
now=$(date +"%Y-%m-%dT%H.%M.%S")

cd "$(dirname $(readlink -f "$0"))/../"
mkdir -p "plot"
cd "plot"

cuda_device="$1"
ssdpath="/dev/md0"
sizes="$((1<<27)) $((1<<28)) $((1<<29))"
repeat=1
# workerss="1 2 4 8 16 32"
workers="16"

outfile="../results/fio_$now.csv"
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

for (( run = 1; run <= $repeat; run+=1 )); do
    for size in $sizes; do
        # command="$command_prefix gdsio -d 0 -w $workers -s 32G -i $io_size -x 6 -I 0 -f $ssdpath"
        command="fio --size=${size}GB --filename=/dev/md0 --io_size=2GB --name=bla --rw=read --iodepth=8 --ioengine=io_uring --direct=1 --blocksize=4MB  --numjobs=16 --thread --group_reporting"
        $command >> $outfile
    done
done
echo "#DONE" >> $outfile
