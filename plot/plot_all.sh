#!/bin/bash


files=("datasets.py" "comp_ratio_bandwidth.py" "comp_prune_ablation.py" "comp_prune_ablation_duck_db.py" "front.py" "storageio.py" "price.py")

mkdir -p pdf
for file in "${files[@]}"
do
   python $file print
done

# copy them all in the paper dir
rsync pdf/* ../../golap_paper/plots/
