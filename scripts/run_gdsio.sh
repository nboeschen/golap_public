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
chunk_bytess="$((1<<16)) $((1<<17)) $((1<<18)) $((1<<19)) $((1<<20)) $((1<<21)) $((1<<22)) $((1<<23)) $((1<<24)) $((1<<25)) $((1<<26)) $((1<<27)) $((1<<28)) $((1<<29))"
repeat=2
# workerss="1 2 4 8 16 32"
workerss="4 16"

outfile="../results/gdsio_$now.csv"
$command_prefix echo -n "# GPU   Info: " >> $outfile && nvidia-smi -i $cuda_device --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader >> $outfile
if [[ -n "$SLURM_JOB_ID" ]]; then
    $command_prefix echo -n "# Slurm Info: " >> $outfile && (squeue -o "%A %C %D %b" | grep $SLURM_JOB_ID) >> $outfile
fi

for (( run = 1; run <= $repeat; run+=1 )); do
    for workers in $workerss; do
        for chunk_bytes in $chunk_bytess; do
            command="$command_prefix gdsio -d 0 -w $workers -s 32G -i $chunk_bytes -x 0 -I 0 -f $ssdpath"
            $command >> $outfile
        done
    done

    # for workers in $workerss; do
    #     for chunk_bytes in $chunk_bytess; do
    #         command="$command_prefix gdsio -d 0 -w $workers -s 32G -i $chunk_bytes -x 5 -I 0 -f $ssdpath"
    #         $command >> $outfile
    #     done
    # done

    # for workers in $workerss; do
    #     for chunk_bytes in $chunk_bytess; do
    #         command="$command_prefix gdsio -d 0 -w $workers -s 32G -i $chunk_bytes -x 6 -I 0 -f $ssdpath"
    #         $command >> $outfile
    #     done
    # done
done
echo "#DONE" >> $outfile
