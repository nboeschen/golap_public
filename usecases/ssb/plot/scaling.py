import sys
sys.path.append('../../../plot/')
import pandas as pd
import numpy as np

import common
from matplotlib import pyplot as plt
from seaborn import color_palette
from behindbars import BehindBars

def plot(files):
    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))
    bars = BehindBars(0.7)

    bws = []

    for title,cur_data in files:
        groupby_cols = ["scale_factor","comp_algo"]
        cur_data["effective_bw"] = (1000.0 / (1<<30)) * cur_data["uncomp_bytes"]/cur_data["time_ms"]
        cur_data = cur_data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"),time_ms_std=("time_ms","std"),effective_bw=("effective_bw","mean"))

        for comp_algo,comp_data in cur_data.groupby("comp_algo"):
            # bars.add_cat(comp_data.scale_factor,comp_data.effective_bw,label=f"{title}")
            bars.add_cat(comp_data.scale_factor,comp_data.time_ms,label=f"{title}")
            # to annotate effective bandwidth
            bws += list(comp_data.effective_bw)

    xss,yss = bars.do_plot(shape="bar",axes=axes, edgecolor="black", linewidth=2)
    i = 0
    for xs,ys in zip(xss,yss):
        for x,y in zip(xs,ys):
            axes.text(x-0.15,y+10,f"BW:{bws[i]:.1f}",fontsize=22)
            i += 1
    axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
    axes.set_axisbelow(True)
    return axes

SF10 = pd.read_csv("../results/query_sf10_export.csv",comment="#")
SF10 = SF10.loc[SF10.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SF10["query"] = SF10["query"].str.slice(stop=5)

SF50 = pd.read_csv("../results/query_sf50_export.csv",comment="#")
SF50 = SF50.loc[SF50.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SF50["query"] = SF50["query"].str.slice(stop=5)

# SF100 = pd.concat([pd.read_csv("../results/query_10.28-08.18.29.csv",comment="#"),
#                     pd.read_csv("../results/query_10.28-13.11.09.csv",comment="#")])
SF100 = pd.read_csv("../results/query_sf100_export.csv",comment="#")
SF100 = SF100.loc[SF100.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SF100["query"] = SF100["query"].str.slice(stop=5)

SF200 = pd.read_csv("../results/query_export.csv",comment="#")
SF200 = SF200.loc[SF200.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SF200["query"] = SF200["query"].str.slice(stop=5)

Uncompressed = pd.concat([SF10,SF50,SF100,SF200]).query("comp_algo == 'UNCOMPRESSED'")
Compressed = pd.concat([SF10,SF50,SF100,SF200]).query("comp_algo != 'UNCOMPRESSED'")

files = [
             ["Golap Uncompressed", Uncompressed],
             ["Golap Compressed", Compressed],
            ]
axes = plot(files)


# axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
# axes.text(0.48,20.2,"SSD",fontsize=26,c="#FF4E5B")
axes.set_ylabel("Avg Query Runtime [ms]")
axes.set_xlabel("SSB Scale Factor")
axes.set_axisbelow(True)

plt.grid()
plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/scaling.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

# same for time
# axes = plot(files,"Select",x_val,show,"time_ms",sort=True,shape=shape)
# t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
# axes.set_xticklabels(t,rotation=25)

# plt.tight_layout()
# if len(sys.argv) > 1 and sys.argv[1] == "print":
#     plt.savefig(f"pdf/avg_ssb_time.pdf",metadata=common.EMPTY_PDF_META)
# else:
#     plt.show()
