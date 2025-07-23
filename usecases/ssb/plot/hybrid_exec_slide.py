import sys
sys.path.append('../../../plot/')
import common
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from plot_various import plot
from seaborn import color_palette
from behindbars import BehindBars

ALL_DATA = pd.concat([pd.read_csv("../results/query3_coexec_export_1.csv",comment="#"),
            pd.read_csv("../results/query3_coexec_export_2.csv",comment="#")])
# ALL_DATA["query"] = ALL_DATA["query"].str.slice(stop=6)  # remove suffix from e.g. 'query3.1a'
ALL_DATA = ALL_DATA.loc[ALL_DATA.groupby(["query","dataflow","customer_factor"],as_index=False)["time_ms"].transform("idxmin").unique()]

INMEM_DATA = pd.read_csv("../results/scaled_dim_query3_inmem_export.csv", comment="#")
# print(INMEM_DATA.to_string())
INMEM_DATA = INMEM_DATA.loc[INMEM_DATA.groupby(["query","customer_factor"],as_index=False)["time_ms"].transform("idxmin").unique()]
# INMEM_DATA["query"] = INMEM_DATA["query"].str.slice(stop=6)
INMEM_DATA['comp_algo'] = "In-Memory"

OPTIMAL = pd.DataFrame(data={'customer_factor':[1,25,50,100],
                       'uncomp_bytes':[3936429884,42000429884,50400429884,67200429884],
                       'comp_bytes':[8678387712,10195423232,11639992320,14472077312],
                       })

files = [
             ["CPU Naive", ALL_DATA[ALL_DATA["query"].str.endswith("d")]],
             ["All GPU Scan", ALL_DATA[ALL_DATA["query"].str.endswith("a")]],
             ["Hybrid Join", ALL_DATA[ALL_DATA["query"].str.endswith("c")]], # Hybrid UM c inf
             # ["Handcoded In-Memory", INMEM_DATA],
             ]

# order = ["[CPU] In-Memory","[GPU]", "[GPU+CPU] CPU Fallback", "[GPU+CPU] Hybrid Join", "[OPTIMAL]"]
plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
fig,axes = plt.subplots(1,3, figsize=(20,10), dpi=100, sharex=True,sharey=False,squeeze=True)
axes = axes.flatten()

for idx,query_name in enumerate(["Query3.1 - Least Selective","Query3.2 - More Selective",
                                "Query3.3 - Most Selective",
                                # "Query3.4 - Most Selective"
                                ]):
    axes[idx].set_prop_cycle('color', color_palette("colorblind"))
    bars = BehindBars(0.7)

    for title,data in files:
        data["query"] = data["query"].str.slice(stop=8)
        cur_data = data[((data["query"] == query_name.split()[0].casefold()) & (data.comp_algo != 'UNCOMPRESSED'))].copy()

        if "effective_bw" not in cur_data.columns:
            cur_data["effective_bw"] = (1000.0 / (1<<30)) * cur_data["uncomp_bytes"]/cur_data["time_ms"]

        cur_data = cur_data.groupby(["query","customer_factor"],as_index=False).agg(time_ms=("time_ms","mean"),
                                                                      debug_0=("debug_0","mean"),
                                                                      effective_bw=("effective_bw","mean"),
                                                                        effective_bw_std=("effective_bw","std"))

        # bars.add_cat(cur_data["customer_factor"],cur_data["effective_bw"],label=title)
        args={}
        if title == "All GPU Scan" or title == "Hybrid Join":
        # if title == "Hybrid Join":
            args["alpha"] = 0
        bars.add_cat(cur_data["customer_factor"],cur_data["time_ms"]/1000,label=title,part_of=title, args=args)
        # bars.add_cat(cur_data["customer_factor"],cur_data["debug_0"]/1000,label=title+" cust",part_of=title,
        #                 args={"hatch":"//","color":"none"})
    # bars.add_cat(OPTIMAL.customer_factor, OPTIMAL.comp_bytes/(1<<30)/common.CONFIG["DGXSSDBW"], label="Optimal Compressed")
    # bars.add_cat(OPTIMAL.customer_factor, OPTIMAL.uncomp_bytes/(1<<30)/common.CONFIG["DGXSSDBW"], label="Optimal UnCompressed")
    bars.do_plot(axes=axes[idx],edgecolor="black",linewidth=2)
    # axes[idx].set_ylim([0,70])
    # axes[idx].axvline(2.5,lw=5,ls="--",c="#FF4E5B",zorder=0)
    # axes[idx].text(2.1, 12000, 'Out of GPU Mem', c="#FF4E5B", rotation='vertical')
    axes[idx].set_title(query_name,fontsize=28)
    axes[idx].grid(axis="y")
    axes[idx].set_axisbelow(True)

    axes[idx].set_xlabel("Customer table scaling", fontsize=30)

fig.tight_layout(h_pad=0.1)
fig.text(0.0, 0.5, 'Query Runtime [s]', va='center', rotation='vertical')
handles, labels = plt.gca().get_legend_handles_labels()
# fig.legend(handles[::2]+
#     [mpatches.Patch(edgecolor="black",facecolor="none",hatch='//')],
#     labels[::2]+['Customer HT Build Time'], ncol=2, loc='upper center', mode="expand")
fig.legend(handles, labels, ncol=3, loc='upper center', mode="expand")
# plt.subplots_adjust(top=0.72)
plt.subplots_adjust(top=0.85)
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/hybrid_exec_time_simple0.pdf",metadata=common.EMPTY_PDF_META)
    # plt.savefig(f"pdf/hybrid_exec_time_simple1.pdf",metadata=common.EMPTY_PDF_META)
    # plt.savefig(f"pdf/hybrid_exec_time_simple.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()
