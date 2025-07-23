import sys
sys.path.append('../../../plot/')
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from seaborn import color_palette

import common

QUERY = "select_commitdate"
COMP_ALGO = "BEST_BW_COMP"

DATA = pd.read_csv("../results/select_export_5.csv",comment="#") # with event sync
DATA = DATA[(DATA.comp_algo == COMP_ALGO) & (DATA["query"] == QUERY)]
# DATA["marker"] = "o-"
SYNCHRONOUS = pd.read_csv("../results/select_STREAMSYNC_export.csv",comment="#") # with stream sync
SYNCHRONOUS = SYNCHRONOUS[(SYNCHRONOUS.comp_algo == COMP_ALGO) & (SYNCHRONOUS["query"] == QUERY)]
# SYNCHRONOUS["marker"] = "x:"

OVERLAP_50 = pd.read_csv("../results/select_sim_compute_export_1.csv",comment="#") # with event sync, 50us
OVERLAP_50 = OVERLAP_50[(OVERLAP_50.comp_algo == COMP_ALGO) & (OVERLAP_50["query"] == QUERY)].copy()
# OVERLAP_50["marker"] = "o-"

OVERLAP_100 = pd.read_csv("../results/select_sim_compute_export_2.csv",comment="#") # with event sync, 100us
OVERLAP_100 = OVERLAP_100[(OVERLAP_100.comp_algo == COMP_ALGO) & (OVERLAP_100["query"] == QUERY)].copy()
# OVERLAP_100["marker"] = "o-"

OVERLAP_600 = pd.read_csv("../results/select_sim_compute_export_3.csv",comment="#") # with event sync, 600us
OVERLAP_600 = OVERLAP_600[(OVERLAP_600.comp_algo == COMP_ALGO) & (OVERLAP_600["query"] == QUERY)].copy()
# OVERLAP_600["marker"] = "o-"

SYNCHRONOUS_50 = pd.read_csv("../results/select_sim_compute_export_4.csv",comment="#") # with stream sync, 50us
SYNCHRONOUS_50 = SYNCHRONOUS_50[(SYNCHRONOUS_50.comp_algo == COMP_ALGO) & (SYNCHRONOUS_50["query"] == QUERY)].copy()
# SYNCHRONOUS_50["marker"] = "x:"

SYNCHRONOUS_100 = pd.read_csv("../results/select_sim_compute_export_5.csv",comment="#") # with stream sync, 100us
SYNCHRONOUS_100 = SYNCHRONOUS_100[(SYNCHRONOUS_100.comp_algo == COMP_ALGO) & (SYNCHRONOUS_100["query"] == QUERY)].copy()
# SYNCHRONOUS_100["marker"] = "x:"

SYNCHRONOUS_600 = pd.read_csv("../results/select_sim_compute_export_6.csv",comment="#") # with stream sync, 600us
SYNCHRONOUS_600 = SYNCHRONOUS_600[(SYNCHRONOUS_600.comp_algo == COMP_ALGO) & (SYNCHRONOUS_600["query"] == QUERY)].copy()
# SYNCHRONOUS_600["marker"] = "x:"




def fun(title, sim_compute_us, df, dataflow):

    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))

    # data = pd.read_csv(path,comment="#")
    data = df[:]

    metric = "effective_bw"
    # metric = "ratio"
    # metric = "time_ms"

    groupby_cols = ["query","dataflow","workers","chunk_bytes","uncomp_bytes", "event_sync"]

    data = data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"), time_ms_std=("time_ms","std"), comp_bytes=("comp_bytes","mean"))

    data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
    data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
    data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]
    data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]

    # data = data[data["query"] == "select_commitdate"]
    data = data[data["dataflow"] == dataflow]
    data = data[data["workers"] != 16]

    data["inefficient"] = data["uncomp_bytes"] < (data.workers * data.chunk_bytes)
    data.loc[data.inefficient,metric] = "NaN"

    for worker_num, cur_data in data.groupby("workers"):
        marker = {1:"o",2:"*",4:"P",8:"^",16:"H"}[worker_num]

        overlap_data = cur_data[(cur_data.event_sync == True)]
        plotted, = axes.plot(overlap_data.chunk_bytes, overlap_data[metric], marker+"-", ms=16, lw=4, label=f"{worker_num}")

        # if worker_num == 1 and dataflow == "SSD2GPU" and title != "UNCOMPRESSED":
        #     # optimal BW doesnt change with worker num
        #     axes.plot(cur_data.chunk_bytes, cur_data["ratio_alt"]*common.CONFIG["DGXSSDBW"], "--", lw=6, c="#3EAA4E", label="_")

        stream_sync_data = cur_data[(cur_data.event_sync == False)]
        axes.plot(stream_sync_data.chunk_bytes, stream_sync_data[metric], marker+":", ms=16, lw=4, color=plotted.get_color(), label="_")

    axes.set_xscale('log')
    xticks = np.power(2,np.linspace(16,27,12))
    axes.set_xticks(xticks,labels=[common.hrsize(x,sep='\n') for x in xticks])
    axes.set_xticks([],minor=True)
    axes.set_xlabel("Chunksize")

    axes.plot([],[],"-",c="black",ms=16, lw=4, label="Overlap")
    axes.plot([],[],":",c="black",ms=16, lw=4, label="No Overlap")

    axes.axhline(common.CONFIG["DGXSSDBW"],lw=6,ls="--",c="#FF4E5B",zorder=0)
    axes.text((1<<16)+(1<<12),17.5,"SSD BW",c="#FF4E5B")

    if dataflow == "SSD2GPU2CPU" or title == "UNCOMPRESSED":
        axes.set_ylim([0,30])
        axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=6,ls="--",c="#1AA1F1",zorder=0)
        axes.text((1<<16)+(1<<12),21.5,"GPU-CPU BW",c="#1AA1F1")
    axes.set_ylabel("Effective Bandwidth [GiB/s]")

    axes.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.,title="In-Flight Requests / Data Pipelines                          Intra Pipeline Parallelism")
    plt.grid()
    fig.tight_layout()

    if len(sys.argv) > 1 and sys.argv[1] == "print":
        plt.savefig(f"pdf/{title}_intra_worker_overlap_{sim_compute_us}_{dataflow}.pdf",metadata=common.EMPTY_PDF_META)
    else:
        plt.show()


# fun("SCAN", 50, pd.concat([OVERLAP_50,SYNCHRONOUS_50]), "SSD2GPU")
# fun("SCAN", 100, pd.concat([OVERLAP_100,SYNCHRONOUS_100]), "SSD2GPU")
# fun("SCAN", 600, pd.concat([OVERLAP_600,SYNCHRONOUS_600]), "SSD2GPU")
fun("SCAN", 0, pd.concat([DATA,SYNCHRONOUS]), "SSD2GPU2CPU")


