import sys
import pandas as pd
import numpy as np

import common
from matplotlib import pyplot as plt
from seaborn import color_palette
from behindbars import BehindBars


def plot(files,title,x_var,show="display",metric="time_ms",filter_out=[],sort=True,shape="bar",order=None):
    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))

    bars = BehindBars(0.7)

    for label,data_or_path in files:
        if type(data_or_path) == str:
            data = pd.read_csv(data_or_path,comment="#")
        else:
            data = data_or_path.copy()

        data["query"] = data["query"].str.replace("^count","select",regex=True)
        for col_name,fun in filter_out.items():
            if col_name not in data:
                continue
            data = data[data[col_name].apply(fun)]

        groupby_cols = [x_var,"comp_algo","scale_factor"]
        # if metric in data.columns and metric not in groupby_cols:
        #     groupby_cols.append(metric)
        if "prune_ms" not in data.columns:
            data["prune_ms"] = 0
        data = data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"),
                                                             prune_ms=("prune_ms","mean"),
                                                                                                    time_ms_std=("time_ms","std"))
        if "pruned_bytes" not in data.columns:
            data["pruned_bytes"] = 0

        # use known uncompressed sizes from golap:
        # for idx,row in data.iterrows():
        #     if row.uncomp_bytes == 0:
        #         data.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query)

        # data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
        # data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
        # data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]
        # data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]
        # data["comp_bytes_gb"] = data["comp_bytes"]/(1<<30)

        for comp_algo,cur_data in data.groupby("comp_algo"):
            if shape=="bar":
                x = cur_data[x_var]
            else:
                x = list(np.ones(len(cur_data.shape)))
            hatch = {"✘ Pruning ✘ Compression":"",
                        "✓ Pruning ✘ Compression": "/",
                        "✘ Pruning ✓ Compression": "o",
                        "✓ Pruning ✓ Compression": "+"}
            plabel = f"{label} {comp_algo}"
            bars.add_cat(x,cur_data[metric],label=plabel,args={"hatch":hatch[plabel]})
    if sort:
        bars.sort(prefixes=set(name for name,_ in files))
    if order:
        bars.order(order)
    bars.do_plot(shape=shape,axes=axes, edgecolor="black", linewidth=2)

    common.axes_prepare(axes,metric)

    axes.set_xlabel(x_var)
    axes.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
    axes.grid(axis="y")
    axes.set_axisbelow(True)

    return axes


SSB_DUCKDB = pd.read_csv("../usecases/ssb/results/duckdb_query_export.csv",comment="#")
SSB_DUCKDB = SSB_DUCKDB.query('comp_algo == "UNCOMPRESSED" or comp_algo == "SNAPPY"')
SSB_DUCKDB = SSB_DUCKDB.query("query.str.startswith('query1')")
SSB_DUCKDB = SSB_DUCKDB.loc[SSB_DUCKDB.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
# SSB_DUCKDB["query"] = SSB_DUCKDB["query"].str.slice(stop=5)
SSB_DUCKDB["query"] = "SSB\n" + SSB_DUCKDB["query"]
SSB_DUCKDB.loc[SSB_DUCKDB.comp_algo == "UNCOMPRESSED","comp_algo"] = "✘ Compression"
SSB_DUCKDB.loc[SSB_DUCKDB.comp_algo == "SNAPPY","comp_algo"] = "✓ Compression"

SSB_DUCKDB_PRUNED = pd.read_csv("../usecases/ssb/results/duckdb_query_pruning_export.csv",comment="#")
SSB_DUCKDB_PRUNED = SSB_DUCKDB_PRUNED.query('comp_algo == "UNCOMPRESSED" or comp_algo == "SNAPPY"')
SSB_DUCKDB_PRUNED = SSB_DUCKDB_PRUNED.query("query.str.startswith('query1')")
SSB_DUCKDB_PRUNED = SSB_DUCKDB_PRUNED.loc[SSB_DUCKDB_PRUNED.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_DUCKDB_PRUNED["query"] = "SSB\n" + SSB_DUCKDB_PRUNED["query"]
SSB_DUCKDB_PRUNED.loc[SSB_DUCKDB_PRUNED.comp_algo == "UNCOMPRESSED","comp_algo"] = "✘ Compression"
SSB_DUCKDB_PRUNED.loc[SSB_DUCKDB_PRUNED.comp_algo == "SNAPPY","comp_algo"] = "✓ Compression"

files = [
             ["✘ Pruning", pd.concat([SSB_DUCKDB])],
             ["✓ Pruning", pd.concat([SSB_DUCKDB_PRUNED])]
            ]
filter_out = {}
show = "return"
# show = "print"
x_val = "query"
# x_val = "chunk_bytes"
shape = "bar"
order = ["✘ Pruning ✘ Compression","✓ Pruning ✘ Compression", "✘ Pruning ✓ Compression", "✓ Pruning ✓ Compression"]

# axes = plot(files,"Select",x_val,show,"comp_bytes_gb",filter_out,sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"ratio",filter_out,sort=False,shape=shape)
axes = plot(files,"Select",x_val,show,"time_ms",filter_out,sort=False,shape=shape,order=order)
# axes = plot(files,"Select",x_val,show,"effective_bw",filter_out,sort=False,shape=shape,order=order)
# axes = plot(files,"Select",x_val,show,"pruned_bytes",filter_out,sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"actual_bw",filter_out,sort=False,shape=shape,order=order)
# axes = plot(files,"Select",x_val,show,"speedup",filter_out,sort=False,shape=shape)

# axes.axhline(common.CONFIG["DGXSSDBW"],lw=6,ls="--",c="#FF4E5B",zorder=0)
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
# axes.text(0.48,16.2,"SSD",fontsize=18,c="#FF4E5B")
# axes.text(0.48,26.5,"CPU\n<->\nGPU",fontsize=18,c="#1AA1F1")
# axes.set_yscale("log")
# axes.set_ylim([0,50])
axes.set_xlabel("")
# axes.plot([],[],"--",lw=6,c="#FF4E5B",label="SSD")
# axes.scatter([],[],color='none',label="Bandwidth")
axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/comp_prune_ablation_duck_db.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

