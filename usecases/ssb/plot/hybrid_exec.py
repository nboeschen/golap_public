import sys
sys.path.append('../../../plot/')
import common
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from plot_various import plot
from seaborn import color_palette
from behindbars import BehindBars

ALL_GPU_DATA = pd.read_csv("../results/query3_export.csv",comment="#")
ALL_GPU_DATA = ALL_GPU_DATA.query("query.str.startswith('query3')")
best_gpu = ALL_GPU_DATA.groupby(["query","dataflow","customer_factor"],as_index=False)["time_ms"].transform("idxmin").unique()
# ALL_GPU_DATA["query"] = ALL_GPU_DATA["query"].str.slice(stop=6)  # remove suffix from e.g. 'query3.1a'
ALL_GPU_DATA = ALL_GPU_DATA.loc[best_gpu]
ALL_GPU_DATA['comp_algo'] = ALL_GPU_DATA['comp_algo'].str.replace('BEST_BW_COMP', '')

INMEM_DATA = pd.read_csv("../results/scaled_dim_query3_inmem_export.csv", comment="#")
INMEM_DATA = INMEM_DATA.loc[INMEM_DATA.groupby(["query","customer_factor"],as_index=False)["time_ms"].transform("idxmin").unique()]
# INMEM_DATA["query"] = INMEM_DATA["query"].str.slice(stop=6)
INMEM_DATA['comp_algo'] = "In-Memory"

CD_DATA = pd.read_csv("../results/query3cd_export.csv",comment="#")
BASELINE_3D = CD_DATA[CD_DATA["query"].str.endswith("d")].copy()
BASELINE_3D["query"] = BASELINE_3D["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'
best_3D = BASELINE_3D.groupby(["query","dataflow","customer_factor"],as_index=False)["time_ms"].transform("idxmin").unique()
# BASELINE_3D["query"] = BASELINE_3D["query"].str.slice(stop=6)  # remove suffix from e.g. 'query3.1'
BASELINE_3D = BASELINE_3D.loc[best_3D]
BASELINE_3D['comp_algo'] = BASELINE_3D['comp_algo'].str.replace('BEST_BW_COMP', '')

CF_DATA = pd.read_csv("../results/query3cf_export.csv",comment="#")


DATA3C = CF_DATA[CF_DATA["query"].str.endswith("c")].copy()
DATA3C["query"] = DATA3C["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'
best_3c = DATA3C.groupby(["query","dataflow","customer_factor", "max_gpu_um_memory"],as_index=False)["time_ms"].transform("idxmin").unique()
# DATA3C["query"] = DATA3C["query"].str.slice(stop=6)  # remove suffix from e.g. 'query3.1a'
DATA3C = DATA3C.loc[best_3c]
DATA3C['comp_algo'] = DATA3C['comp_algo'].str.replace('BEST_BW_COMP', '')
DATA3C_16MB = DATA3C[DATA3C.max_gpu_um_memory == (1<<24)]
DATA3C_64MB = DATA3C[DATA3C.max_gpu_um_memory == (1<<26)]
DATA3C_265MB = DATA3C[DATA3C.max_gpu_um_memory == (1<<28)]
DATA3C_1GB = DATA3C[DATA3C.max_gpu_um_memory == (1<<30)]
DATA3C_INF = DATA3C[DATA3C.max_gpu_um_memory == 0]

DATA3F = CF_DATA[CF_DATA["query"].str.endswith("f")].copy()
DATA3F["query"] = DATA3F["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'
best_3c = DATA3F.groupby(["query","dataflow","customer_factor", "max_gpu_um_memory"],as_index=False)["time_ms"].transform("idxmin").unique()
# DATA3F["query"] = DATA3F["query"].str.slice(stop=6)  # remove suffix from e.g. 'query3.1a'
DATA3F = DATA3F.loc[best_3c]
DATA3F_16MB = DATA3F[DATA3F.max_gpu_um_memory == (1<<24)]
DATA3F_64MB = DATA3F[DATA3F.max_gpu_um_memory == (1<<26)]
DATA3F_265MB = DATA3F[DATA3F.max_gpu_um_memory == (1<<28)]
DATA3F_1GB = DATA3F[DATA3F.max_gpu_um_memory == (1<<30)]
DATA3F_INF = DATA3F[DATA3F.max_gpu_um_memory == 0]


OPTIMAL = DATA3C_INF.copy()
OPTIMAL.time_ms = 1000*OPTIMAL.comp_bytes / (common.CONFIG["DGXSSDBW"]*(1<<30))


files = [
             ["[GPU+CPU] CPU Fallback", BASELINE_3D],
             # ["[GPU]", ALL_GPU_DATA],
             # ["[CPU] In-Memory", INMEM_DATA],
             # ["[Hybrid UM c inf", DATA3C_INF], # Hybrid UM c inf
             # ["[OPTIMAL]", OPTIMAL], # Hybrid UM c inf
             ["[GPU+CPU] Hybrid Join", DATA3C_INF], # Hybrid UM c inf
             # ["[Hybrid UM 16MB]", DATA3C_16MB],
             # ["[Hybrid UM 64MB]", DATA3C_64MB],
             # ["[Hybrid UM c 265MB]", DATA3C_265MB],
             # ["[Hybrid UM f inf]", DATA3F_INF],
             # ["[Hybrid UM f 265MB]", DATA3F_265MB],
             # ["[Hybrid UM 1GB]", DATA3C_1GB],
             # ["[Hybrid UVA]", DATA3C_UVA],
             ]

order = ["[CPU] In-Memory","[GPU]", "[GPU+CPU] CPU Fallback", "[GPU+CPU] Hybrid Join", "[OPTIMAL]"]
plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
fig,axes = plt.subplots(2,2, figsize=(20,10), dpi=100, sharex=True,sharey=True,squeeze=True)
axes = axes.flatten()

for idx,query_name in enumerate(["Query3.1 - Least Selective","Query3.2 - More Selective",
                                "Query3.3 - Most Selective","Query3.4 - Most Selective"]):
    axes[idx].set_prop_cycle('color', color_palette("colorblind"))
    bars = BehindBars(0.7)

    for title,data in files:
        cur_data = data[((data["query"] == query_name.split()[0].casefold()) & (data.comp_algo != 'UNCOMPRESSED'))].copy()

        if "effective_bw" not in cur_data.columns:
            cur_data["effective_bw"] = (1000.0 / (1<<30)) * cur_data["uncomp_bytes"]/cur_data["time_ms"]

        cur_data = cur_data.groupby(["query","customer_factor"],as_index=False).agg(time_ms=("time_ms","mean"),
                                                                      effective_bw=("effective_bw","mean"),
                                                                        effective_bw_std=("effective_bw","std"))
        # axes[idx].bar(cur_data["customer_factor"],cur_data["effective_bw"],width=1,label=title,edgecolor="black",
        #              linewidth=2)
        bars.add_cat(cur_data["customer_factor"],cur_data["effective_bw"],label=title)

    bars.do_plot(axes=axes[idx],edgecolor="black",linewidth=2)
    axes[idx].set_ylim([0,70])
    axes[idx].axvline(2.5,lw=5,ls="--",c="#FF4E5B",zorder=0)
    axes[idx].set_title(query_name,fontsize=28)
    axes[idx].grid(axis="y")
    axes[idx].set_axisbelow(True)

    if idx > 1:
        axes[idx].set_xlabel("Customer table scaling", fontsize=30)

fig.tight_layout(h_pad=0.1)
fig.text(0.0, 0.5, 'Effective Bandwidth [GB/s]', va='center', rotation='vertical')
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, ncol=2,loc='upper center', mode="expand")
plt.subplots_adjust(top=0.85)
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/hybrid_exec.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()
