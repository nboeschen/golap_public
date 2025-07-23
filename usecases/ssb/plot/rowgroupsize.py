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
        groupby_cols = ["row_group_size","comp_algo"]
        cur_data["io_bw"] = (1000.0 / (1<<30)) * cur_data["io_bytes"]/cur_data["time_ms"]
        cur_data["effective_bw"] = (1000.0 / (1<<30)) * cur_data["uncomp_bytes"]/cur_data["time_ms"]
        print(cur_data.to_string())
        cur_data = cur_data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"),time_ms_std=("time_ms","std"),effective_bw=("effective_bw","mean"),
                                                                     io_bw=("io_bw","mean"))

        for comp_algo,comp_data in cur_data.groupby("comp_algo"):
            print(comp_data.to_string())
            # bars.add_cat(comp_data.row_group_size,comp_data.effective_bw,label=f"{title}")
            # bars.add_cat(comp_data.row_group_size,comp_data.io_bw,label=f"{title}")
            bars.add_cat(comp_data.row_group_size,comp_data.time_ms,label=f"{title}")
            # to annotate effective bandwidth
            bws += list(comp_data.io_bw)

    xss,yss = bars.do_plot(shape="bar",axes=axes, edgecolor="black", linewidth=2)
    i = 0
    for xs,ys in zip(xss,yss):
        for x,y in zip(xs,ys):
            axes.text(x-0.15,y+10,f"I/O BW:\n{bws[i]:.1f} GiB/s",fontsize=22)
            i += 1
    axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
    axes.set_axisbelow(True)
    return axes

SF100 = pd.read_csv("../results/duckdb_scan_export.csv",comment="#")
SF100 = SF100.loc[SF100.groupby(["query","io_bytes","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
# SF100["query"] = SF100["query"].str.slice(stop=5)
# SF100["uncomp_bytes"] = 4800171307
SF100["uncomp_bytes"] = 7203315712
SF100["comp_ratio_alt"] = SF100["io_bytes"]/7203315712


Uncompressed = SF100.query("comp_algo == 'UNCOMPRESSED'")
Compressed = SF100.query("comp_algo != 'UNCOMPRESSED'")


files = [
             ["DuckDB Uncompressed", Uncompressed],
             ["DuckDB Compressed", Compressed],
            ]
axes = plot(files)


# axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
# axes.text(0.48,20.2,"SSD",fontsize=26,c="#FF4E5B")
axes.set_ylim([0,3000])
axes.set_ylabel("Scan Run Time [ms]")
axes.set_xlabel("Row Group Size")
axes.set_axisbelow(True)

plt.grid()
plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/row_group_size.pdf",metadata=common.EMPTY_PDF_META)
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
