import sys
sys.path.append('../../../plot/')
from matplotlib import pyplot as plt
import pandas as pd
import warnings
import numpy as np
from seaborn import color_palette

from behindbars import BehindBars

dataflow_names = {"SSD2CPU": "HOST: ","SSD2GPU2CPU":"GiD: ", "SSD2GPU":"GPU: "}

def plot_hist(path,queries,type_,metric,show,dataflow_filter=[]):
    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))

    data = pd.read_csv(path,comment="#")
    data = data[data["query"].isin(queries)]
    assert len(data["uncomp_bytes"].unique()) == 1
    total_uncomp_bytes = data["uncomp_bytes"].unique()[0]

    data = data[data["chunk_bytes"]<805306368]
    data = data[data["dataflow"].isin(dataflow_filter)]
    warnings.simplefilter(action='ignore', category=UserWarning)
    data = data.groupby(["query","comp_algo","chunk_bytes","dataflow"],as_index=False).agg({"time_ms":"mean","comp_bytes":"mean"})

    dataflow_names = {"SSD2CPU": "HOST: ","SSD2GPU2CPU":"GiD: ", "SSD2GPU":"GPU: "}

    data["bw"] = (1000.0 / (1<<30)) * total_uncomp_bytes/data["time_ms"]
    data["ratio"] = data["comp_bytes"]/total_uncomp_bytes

    # chunk_bytes gets aggregated out here:
    # data = data.groupby(["query","comp_algo","dataflow"],as_index=False).agg({"bw":np.max,"ratio":np.min})

    bins = None
    if metric == "bw":
        maxbw = 200
        bins = list(np.linspace(0,maxbw,int(maxbw/2.5)+1))
        # bins.append(np.inf)
        data.bw = data.bw.clip(0,maxbw)
    elif metric == "ratio":
        bins = np.linspace(0,1,11)
    elif metric == "time_ms":
        maxtime = 500
        bins = list(np.linspace(0,maxtime,int(maxtime/10)+1))
        # bins.append(np.inf)
        data.bw = data.bw.clip(0,maxtime)

    # cats = ["ANS","Bitcomp","Cascaded","Gdeflate","LZ4","Snappy","UNCOMPRESSED"]
    cats = ["BEST_BW_COMP","UNCOMPRESSED"]
    data["comp_algo"] = pd.Categorical(data.comp_algo, categories=cats, ordered=True)
    data.sort_values(by="comp_algo")
    data.hist(metric,by="comp_algo",sharey=True,bins=bins,ax=axes,edgecolor="black", linewidth=2)
    # data.hist(metric,sharey=True,bins=bins,ax=axes,edgecolor="black", linewidth=2)

    for axes in fig.axes:
        axes.set_ylabel("frequency [#]")
        axes.grid(axis="y")
        axes.set_axisbelow(True)
        if metric == "bw":
            axes.set_xlabel("BW [GB]")
        elif metric == "ratio":
            axes.set_xlabel("compression [ratio]")
        elif metric == "time_ms":
            axes.set_xlabel("Time [ms]")
    # labelax = fig.add_subplot(338)
    # labelax.set_title(f"{type_}, total_bytes={total_uncomp_bytes/(1<<20):.2f}MB")
    fig.suptitle(f"{type_}, total_bytes={total_uncomp_bytes/(1<<20):.2f}MB")
    fig.tight_layout()


    means = data.groupby(["comp_algo"]).agg({metric:"mean"})
    if metric == "ratio" or metric == "time_ms":
        the_best_idx = means.idxmin()
    else:
        the_best_idx = means.idxmax()

    print(f"Best in {metric=} for {type_}: {the_best_idx[0]}\t{means.loc[the_best_idx[0], metric]:.2f}")
    table_ax = fig.add_axes([0.5, 0.05, 0.3, 0.25])
    table_ax.set_axis_off()
    table = table_ax.table(rowLabels=cats, colLabels=["Mean"], 
                  cellText=[[f"{means.loc[cat][0]:.2f}"] for cat in cats],
                  rowColours =["palegreen" for _ in cats], colColours =["palegreen"], cellLoc ='center',loc ='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(23)
    table.scale(1, 1.5)

    if show:
        plt.show()
    else:
        plt.savefig(f"pdf/hist_{metric}_{type_}.pdf",metadata=common.EMPTY_PDF_META)

if __name__ == '__main__':

    # path = "../results/select_04.06-11.41.59.csv"
    # path = "../results/select_05.05-10.55.12.csv" # single ssd
    # path = "../results/select_05.05-12.14.23.csv" # raid0
    path = "../results/select_10.07-11.04.24.csv" # raid0
    # query1_cols = [(["select_orderdate"], "uint64"),
                    # (["select_extendedprice"], "uint32"),
                    # (["select_discount"], "uint8"),
                    # (["select_quantity"], "uint8"),
                    # ]
    all_cols = [([q],q) for q in "select_key,select_linenum,select_custkey,select_partkey,select_suppkey,select_orderdate,select_linenumber,select_orderpriority,select_shippriority,select_quantity,select_extendedprice,select_ordtotalprice,select_discount,select_revenue,select_supplycost,select_tax,select_commitdate,select_shipmode".split(",")]
    cols_gb_type = [
                     (["select_key","select_custkey","select_partkey","select_suppkey","select_orderdate","select_commitdate"], "uint64"),
                     # (["select_extendedprice","select_ordtotalprice","select_revenue","select_supplycost"], "uint32"),
                     # (["select_linenum","select_linenumber","select_quantity","select_discount","select_tax"], "uint8"),
                     # (["select_orderpriority"], "char[16]"),
                     # (["select_shipmode"], "char[11]"),
                     # (["select_shippriority"], "char"),
                    ]
    dataflow_filter = [
                        "SSD2GPU",
                        # "SSD2GPU2CPU",
                        # "SSD2CPU"
                        ]
    for queries,type_ in all_cols:
        plot_hist(path,queries,type_,"bw",True,dataflow_filter)
        # plot_hist(path,queries,type_,"ratio",True,dataflow_filter)