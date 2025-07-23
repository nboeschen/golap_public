import sys
sys.path.append('../../../plot/')
import pandas as pd
import numpy as np
from behindbars import BehindBars

from matplotlib import pyplot as plt
from seaborn import color_palette

import common

PRUNE = pd.read_csv("../results/pruning_query1123_export_2.csv",comment="#")
MINMAX = pd.read_csv("../results/pruning_query1123_export_3.csv",comment="#")
BLOOM_Q1 = pd.read_csv("../results/pruning_query1123_export_4.csv",comment="#")

AVGD = pd.concat([PRUNE,MINMAX,BLOOM_Q1]).copy()
# AVGD = pd.concat([PRUNE,MINMAX]).copy()
AVGD["query"] = AVGD["query"].str.slice(stop=6)


nocomp_noprune_files = [
            ("Query1 avg", AVGD[(AVGD.comp_algo == "UNCOMPRESSED") & (AVGD.pruning == "DONTPRUNE")]),
            ]

nocomp_prune_files = [
            ("Query1 avg", AVGD[(AVGD.comp_algo == "UNCOMPRESSED") & (AVGD.pruning != "DONTPRUNE")]),
            ]

comp_noprune_files = [
            ("Query1 avg", AVGD[(AVGD.comp_algo != "UNCOMPRESSED") & (AVGD.pruning == "DONTPRUNE")]),
            ]

comp_prune_files = [
            ("Query1 avg", AVGD[(AVGD.comp_algo != "UNCOMPRESSED") & (AVGD.pruning != "DONTPRUNE")]),
            ]


def plot_pruning(files,name,metric):
    bars = BehindBars(0.7)
    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))

    for title, df in files:

        data = df.copy()
        # data = pd.read_csv(path,comment="#")

        groupby_cols = ["query","comp_algo","dataflow","pruning","chunk_bytes","uncomp_bytes"]
        # if in data.columns:
        #     groupby_cols.append)

        # print(data[["query","comp_algo","chunk_bytes","pruning","uncomp_bytes","comp_bytes","pruned_bytes"]].to_string(index=False))

        data = data.groupby(groupby_cols,as_index=False).agg(prune_ms=("prune_ms","mean"), time_ms=("time_ms","mean"), time_ms_std=("time_ms","std"), pruned_bytes=("pruned_bytes","mean"), comp_bytes=("comp_bytes","mean"))

        for (metastrat,comp_algo),cur_data in data.groupby(["pruning","comp_algo"]):
            # if metastrat not in set(["HIST", "MINMAX", "BLOOM"]):
            #     continue
            cur_data = cur_data.copy()
            if comp_algo != "UNCOMPRESSED":
                # continue
                pass
            if comp_algo == "UNCOMPRESSED":
                # continue
                # plot_params["hatch"] = "//"
                pass

            prune_ref_size = cur_data["uncomp_bytes"] if comp_algo == "UNCOMPRESSED" else cur_data["comp_bytes"]
            cur_data["pruned_ratio"] = cur_data["pruned_bytes"]/prune_ref_size
            cur_data["prune_time_ratio"] = cur_data["prune_ms"]/cur_data["time_ms"]
            # cur_data["speedup"] = data[(data.pruning == "DONTPRUNE") & (data.comp_algo == comp_algo)].time_ms.values/cur_data.time_ms.values

            # label = f"{title} {metastrat}"
            labels = {"BLOOM": "Bloom Filter", "HIST": "Histogram", "MINMAX": "MinMax"}
            hatch = {"BLOOM": "", "HIST": "/", "MINMAX": "o"}
            plot_params = {"hatch":hatch[metastrat]}
            label = labels[metastrat]
            bars.add_cat(cur_data.chunk_bytes, cur_data[metric],label=label, part_of=label, args=plot_params)

    # bars.sort(prefixes=set(name for name,_ in files))
    bars.do_plot(shape="bar",axes=axes, edgecolor="black", linewidth=2)

    # axes.set_xscale('log')
    # axes.set_yscale('log')
    # xticks = np.power(2,np.linspace(20,29,10))
    axes.set_xticks(axes.get_xticks(),labels=[common.hrsize(int(x._text),sep='\n') for x in axes.get_xticklabels()])
    axes.set_xlabel("Chunksize")
    axes.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)


    if metric == "time_ms" or metric == "prune_ms":
        axes.set_ylabel("Time [ms]")
        if "comp_prune" in name:
            axes.set_ylim([0,500])
            if name == "comp_prune":
                axes.text(0.75,488,"//",fontsize=24)
                axes.text(0.68,468,">1k",fontsize=22,c="#fafafa")
        if name == "comp_noprune":
            axes.set_ylim([0,1500])
            axes.text(0.97,1465,"//",fontsize=24)
            axes.text(0.85,1400,">5000",fontsize=22,c="#fafafa")
    elif metric == "pruned_ratio":
        axes.set_ylim([0,1.2])
        axes.set_ylabel("Pruned data [ratio]")

    axes.set_axisbelow(True)
    fig.tight_layout()
    plt.grid()

    if len(sys.argv) > 1 and sys.argv[1] == "print":
        plt.savefig(f"pdf/pruning_{name}_{metric}.pdf",metadata=common.EMPTY_PDF_META)
        # plt.savefig(f"pdf/pruning_{name}_{metric}_wobloom.pdf",metadata=common.EMPTY_PDF_META)
    else:
        plt.show()



plot_pruning(comp_prune_files, "comp_prune", "pruned_ratio")
# plot_pruning(comp_prune_files, "comp_prune", "prune_ms")
