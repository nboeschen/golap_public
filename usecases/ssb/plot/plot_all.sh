#!/bin/bash


files=("avg_ssb.py" "hybrid_exec_new.py" "sample_plot.py" "chunksize_workers.py" "overlap_scan.py" "comp_ratio.py" "chunksize_comp_prune_ablation.py" "pre_aggr.py" "scaling.py" "rowgroupsize.py")

mkdir -p pdf
for file in "${files[@]}"
do
   python $file print
done

# copy them all in the paper dir
# rsync pdf/* ../../../../golap_paper/plots/ssb/
