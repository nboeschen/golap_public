#!/bin/bash


files=("avg_ssb.py" "front.py" "hybrid_exec.py" "hybrid_exec_new.py" "select_plot.py" "sample_plot.py" "chunksize_workers.py" "pruning_time.py" "overlap_scan.py" "comp_ratio.py" "comp_prune_ablation.py" "chunksize_comp_prune_ablation.py" "pre_aggr.py" "price_performance.py" "scaling.py" "rowgroupsize.py")

mkdir -p pdf
for file in "${files[@]}"
do
   python $file print
done

# copy them all in the paper dir
rsync pdf/* ../../../../golap_paper/plots/ssb/
