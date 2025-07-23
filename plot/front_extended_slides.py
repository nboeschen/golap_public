import common
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from seaborn import color_palette

from behindbars import BehindBars
from math import inf


def plot(files,title,x_var,metric="time_ms",sort=True,shape="bar",with_optimal_ssd_scan=False,with_optimal_inmem_scan=False):
    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))

    order={
            'Golap\nSSD UNCOMPRESSED': 0,
            'Golap\nSSD BEST_BW_COMP': 1.5,
            " SSD\nCompressed\n":2.5,}
    alpha={
            'Golap\nSSD UNCOMPRESSED': 1,
            # 'Golap\nSSD BEST_BW_COMP': 0,
            # " SSD\nCompressed\n":0,
            'Golap\nSSD BEST_BW_COMP': 1,
            " SSD\nCompressed\n":1,
            }

    for label,data_or_path in files:
        if type(data_or_path) == str:
            data = pd.read_csv(data_or_path,comment="#")
        else:
            data = data_or_path.copy()

        if "effective_bw" not in data.columns:
            data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
            # data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
            # data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]
            data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]
            # data["optimal_bw"] = data["ratio_alt"]*common.CONFIG["DGXSSDBW"]
            # data["optimal_time"] = data["uncomp_bytes"]/common.CONFIG["DGXSSDBW"]
            # data["comp_bytes_gb"] = data["comp_bytes"]/(1<<30)
            # print(data[["query","ratio_alt","effective_bw"]].to_string())

            data = data.groupby(["query","comp_algo"],as_index=False).agg(time_ms=("time_ms","mean"),
                                                                      effective_bw=("effective_bw","mean"),
                                                                      # optimal_bw=("optimal_bw","mean"),
                                                                      # optimal_time=("optimal_time","mean"),
                                                                        effective_bw_std=("effective_bw","std"))
        # print(data)
        for comp_algo,cur_data in data.groupby("comp_algo"):
            if f"{label} {comp_algo}" == " SSD\n+Pruning":
                continue
            axes.bar(order[f"{label} {comp_algo}"],cur_data[metric],width=1,edgecolor="black",alpha=alpha[f"{label} {comp_algo}"],
                     linewidth=2, **common.CONFIG[f"{label} {comp_algo}"])

    common.axes_prepare(axes,metric)

    axes.set_xticks(list(order.values()))
    axes.set_xticklabels(["SSD Uncompressed\n(CPU or GPU)", "GPU + SSD\n+ Compression", "CPU + SSD\n+ Compression"],fontsize=26)

    # axes.set_xlabel(x_var)
    # axes.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
    # axes.grid(axis="y")
    axes.set_axisbelow(True)

    return axes



SSB_GOLAP_DATA = pd.read_csv("../usecases/ssb/results/query_export.csv",comment="#")
# SSB_GOLAP_DATA = SSB_GOLAP_DATA[SSB_GOLAP_DATA.chunk_bytes == (1<<24)]
SSB_GOLAP_DATA = SSB_GOLAP_DATA[SSB_GOLAP_DATA["query"] == "query1.1"]
SSB_GOLAP_DATA = SSB_GOLAP_DATA.loc[SSB_GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_GOLAP_UNCOMPRESSED = SSB_GOLAP_DATA[SSB_GOLAP_DATA.comp_algo == "UNCOMPRESSED"]
# SSB_GOLAP_DATA = SSB_GOLAP_DATA.query("comp_algo != 'UNCOMPRESSED'")
# SSB_GOLAP_DATA["query"] = "SSB/" + SSB_GOLAP_DATA["query"].str.slice(stop=5)
# SSB_GOLAP_DATA["query"] = "Avg over SSB queries"
# SSB_GOLAP_DATA["query"] = "SSB (Synth Data)" # avg over all queries
# SSB_GOLAP_DATA.comp_algo = ""
# SSB_GOLAP_DATA.loc[SSB_GOLAP_DATA.comp_algo != 'UNCOMPRESSED', "comp_algo"] = ""

SSB_PRUNED_DATA = pd.read_csv("../usecases/ssb/results/pruning_query1123_export.csv",comment="#")
SSB_PRUNED_DATA = SSB_PRUNED_DATA[(SSB_PRUNED_DATA["query"] == "query1.1") & (SSB_PRUNED_DATA.comp_algo != "UNCOMPRESSED")]
SSB_PRUNED_DATA = SSB_PRUNED_DATA.loc[SSB_PRUNED_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_PRUNED_DATA.comp_algo = ""

SSB_DUCKDB = pd.concat([
                pd.read_csv("../usecases/ssb/results/duckdb_query_export_1.csv",comment="#"),
                pd.read_csv("../usecases/ssb/results/duckdb_query_export_2.csv",comment="#"),
                ])
# SSB_DUCKDB.comp_bytes = SSB_DUCKDB.io_bytes
# SSB_DUCKDB.uncomp_bytes = SSB_DUCKDB.io_bytes
# duckdb best compression is snappy
SSB_DUCKDB = SSB_DUCKDB.query("comp_algo != 'ZSTD' & comp_algo != 'GZIP' & query == 'query1.1'")
# use known uncompressed sizes from golap:
for idx,row in SSB_DUCKDB.iterrows():
    if row.uncomp_bytes == 0:
        # print(row)
        SSB_DUCKDB.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query)
SSB_DUCKDB = SSB_DUCKDB.loc[SSB_DUCKDB.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
# SSB_DUCKDB["query"] = "SSB/" + SSB_DUCKDB["query"].str.slice(stop=5)
SSB_DUCKDB.loc[SSB_DUCKDB.comp_algo == "UNCOMPRESSED","comp_algo"] = "DISK"
SSB_DUCKDB["query"] = "Avg over SSB queries"
# SSB_DUCKDB["query"] = "SSB (Synth Data)" # avg over all queries

SSB_INMEM = pd.read_csv("../usecases/ssb/results/inmem_query_export.csv",comment="#")
SSB_INMEM = SSB_INMEM.loc[SSB_INMEM.groupby(["query"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_INMEM["comp_algo"] = "INMEM"
# SSB_INMEM["query"] = "SSB/" + SSB_INMEM["query"].str.slice(stop=5)
SSB_INMEM["query"] = "Avg over SSB queries"
# SSB_INMEM["query"] = "SSB (Synth Data)" # avg over all queries

SSB_HANDCODED_OPT = pd.read_csv("./handcoded_optimized_export_4.csv",comment="#")
SSB_HANDCODED_OPT["effective_bw"] = SSB_HANDCODED_OPT.bw
SSB_HANDCODED_OPT["comp_algo"] = ""
# ran in parallel
SSB_HANDCODED_OPT = SSB_HANDCODED_OPT.groupby(["strategy","compress","prune"],as_index=False).agg(time_ms=("time_ms","max"),effective_bw=("effective_bw","sum"))
# SSB_HANDCODED_OPT["comp_algo"] = SSB_HANDCODED_OPT.strategy.str.split().apply('\n'.join)
# SSB_HANDCODED_OPT = SSB_HANDCODED_OPT[SSB_HANDCODED_OPT.sockets == 2]
for index, row in SSB_HANDCODED_OPT.iterrows():
    pstr = ["","+Pruning"]
    name = { (True, "InMemory"): "In-Memory", (False, "InMemory"): "In-Memory",
        (True,"SSD"): "SSD\nCompressed",(False,"SSD"): "SSD"}

    SSB_HANDCODED_OPT.at[index,"comp_algo"] = f"{name[(row.compress,row.strategy)]}\n{pstr[int(row.prune)]}"

SSB_HANDCODED_OPT = SSB_HANDCODED_OPT[(SSB_HANDCODED_OPT.strategy == "SSD") & (SSB_HANDCODED_OPT.prune == False) & (SSB_HANDCODED_OPT.compress == True)]

files = [
             # ["DuckDB", SSB_DUCKDB],
             # ["Golap (GPU, ours)", SSB_GOLAP_UNCOMPRESSED],
             ["Golap\nSSD", SSB_GOLAP_DATA],
             # ["+Pruning", SSB_PRUNED_DATA],
             ["", SSB_HANDCODED_OPT],
             # ["Handcoded", SSB_INMEM],
            ]

# axes = plot(files,"Select","query","comp_bytes_gb",sort=False,shape="bar")
# axes = plot(files,"Select","query","ratio",sort=False,shape="bar")
axes = plot(files,"Select","query","effective_bw",sort=True,shape="bar")
# axes = plot(files,"Select","query","time_ms",sort=False,shape="bar")
# axes = plot(files,"Select","query","actual_bw",sort=False,shape="bar")
# axes = plot(files,"Select","query","speedup",sort=False,shape="bar")
# XLABEL = """└──────  CPU  ──────┘ └───────  CPU  ──────┘ └───────  GPU  ──────┘
#        In-Memory                SSD                  SSD
# """

# axes.set_xlabel(XLABEL,loc="left",weight="bold")
axes.axhline(common.CONFIG["DGXSSDBW"],lw=6,ls="--",c="#FF4E5B",zorder=0)
axes.text(-0.65,21,"SSD Read BW",fontsize=36,c="#FF4E5B")

# axes.text(2.28,70,"UNCOMPRESSED\n(I/O bound)",fontsize=30,rotation=90,c="#444444")
# axes.text(2.0,60,"└────┘",fontsize=30,rotation=180,c="#444444")
# axes.text(3.8,70,"COMPRESSED\n(CPU bound)",fontsize=30,rotation=90,c="#444444")
# axes.text(3.1,60,"└──────────┘",fontsize=30,rotation=180,c="#444444")


plt.grid()
plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    # plt.savefig(f"pdf/front_extended_slides0.pdf",metadata=common.EMPTY_PDF_META)
    plt.savefig(f"pdf/front_extended_slides.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()


