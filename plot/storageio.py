import common
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from seaborn import color_palette

from behindbars import BehindBars
from math import inf

def parse_line(line):
    x = line.split(" ")
    rename = {"GPUD": "SYNC", "ASYNC":"ASYNC", "GPU_BATCH":"BATCH"}
    dataflow = rename[x[x.index("XferType:")+1]]
    return (x[x.index("IoType:")+1],
            dataflow,
            int(x[x.index("Threads:")+1]) if dataflow != "BATCH" else int(x[x.index("IoDepth:")+1]),
            int(x[x.index("DataSetSize:")+1].split("/")[0])*1024,
            int(x[x.index("IOSize:")+1][:-5])*1024,
            float(x[x.index("total_time")+1])*1000)

def parse_file(path):
    DATA_raw = []

    with open(path) as file:
        for line in file:
            if line.strip()[0] == "#":
                continue
            DATA_raw.append(parse_line(line))

    return pd.DataFrame(DATA_raw,columns=["query","dataflow","workers","uncomp_bytes","chunk_bytes","time_ms"])

GDSIO_DATA = parse_file("gdsio_export.csv")
GDSIO_DATA = GDSIO_DATA.query("workers == 4 or workers == 16")

# GDSIO_DATA = parse_file("../results/gdsio_2024-08-06T12.27.46.csv")

######################################
def plot(data):
    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))
    bars = BehindBars(0.7)

    metric = "effective_bw"

    groupby_cols = ["query","dataflow","workers","chunk_bytes"]

    data = data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"), time_ms_std=("time_ms","std"),
                                                         uncomp_bytes=("uncomp_bytes","mean"))
    data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
    # data = data.query("workers < 32")


    for worker_num,_ in data.groupby("workers"):
        for dataflow,cur_data in _.groupby("dataflow"):


            marker = f"{'s--' if worker_num == 4 else 'P-'}"
            c = {"ASYNC":"tab:orange","SYNC":"tab:blue","BATCH":"tab:green"}[dataflow]

            axes.plot(cur_data.chunk_bytes, cur_data[metric], marker, c=c, ms=15, lw=4)
            # bars.add_cat(cur_data.chunk_bytes, cur_data[metric], label=f"{worker_num} {dataflow}")

    # bars.do_plot(shape="bar",axes=axes, edgecolor="black", linewidth=2)
    axes.set_xscale('log')
    xticks = np.power(2,np.linspace(16,30,15))
    axes.set_xticks(xticks,labels=[f"{common.hrsize(x)}" for x in xticks])
    axes.set_xticks([],minor=True)
    axes.set_xlabel("Chunksize")

    axes.plot([],[],c="tab:blue",lw=15, label="Synchronous GDS Read")
    axes.plot([],[],c="tab:orange",lw=15, label="Asynchronous GDS Read")
    # axes.plot([],[],c="tab:green",lw=15, label="Batch GDS Read")
    axes.plot([],[],"P-",c="black", ms=15, label="16 Threads")
    axes.plot([],[],"s--",c="black", ms=15, label="4 Threads")

    if metric == "effective_bw":
        axes.axhline(common.CONFIG["DGXSSDBW"],lw=6,ls="--",c="#FF4E5B",zorder=0)
        axes.plot([],[],"--",lw=6,c="#FF4E5B",label="SSD Bandwidth")
        # if dataflow == "SSD2GPU2CPU":
        #     axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=6,ls="--",c="#1AA1F1",zorder=0)
        # if dataflow == "SSD2GPU2CPU" or dataflow == "SSD2CPU" or title == "UNCOMPRESSED":
        #     axes.set_ylim([0,30])
        # if dataflow == "SSD2GPU" and title == "COMPRESSED":
        #     axes.plot([],[],"--",lw=6,c="#3EAA4E",label="Optimal Bandwidth")
        axes.set_ylabel("Bandwidth [GiB/s]")
        axes.set_ylim([0,23])
        # axes.set_ylim([0,3])
    elif metric == "ratio":
        axes.set_ylim([0,1.2])
        axes.set_ylabel("Compression [ratio]")



    plt.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
    plt.grid()

    plt.tight_layout()
    if len(sys.argv) > 1 and sys.argv[1] == "print":
        plt.savefig(f"pdf/gds_storage_io.pdf",metadata=common.EMPTY_PDF_META)
    else:
        plt.show()
##############################
plot(GDSIO_DATA)

