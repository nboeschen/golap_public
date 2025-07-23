import common
import re

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from seaborn import color_palette

from behindbars import BehindBars


def plot(files,title,x_var,show="display",metric="time_ms",filter_out={},sort=True,shape="bar",order=None):
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

        groupby_cols = [x_var,"comp_algo","uncomp_bytes","scale_factor"]
        if metric in data.columns and metric not in groupby_cols:
            groupby_cols.append(metric)
        data = data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"),
                                                                                                    time_ms_std=("time_ms","std"),
                                                                                                    comp_bytes=("comp_bytes","mean"))
        #
        if "pruned_bytes" not in data.columns:
            data["pruned_bytes"] = 0

        # use known uncompressed sizes from golap:
        for idx,row in data.iterrows():
            if row.uncomp_bytes == 0:
                data.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query)


        data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
        data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
        data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]
        data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]
        data["comp_bytes_gb"] = data["comp_bytes"]/(1<<30)


        for comp_algo,cur_data in data.groupby("comp_algo"):
            if shape=="bar":
                x = cur_data[x_var]
            else:
                x = list(np.ones(len(cur_data.shape)))
            bars.add_cat(x,cur_data[metric],label=f"{label} {comp_algo}",args=common.CONFIG[f"{label} {comp_algo}"])
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

    fig.tight_layout()
    if show == "display":
        plt.show()
    elif show == "print":
        plt.savefig(f"pdf/{title}_{metric}.pdf",metadata=common.EMPTY_PDF_META)
    elif show == "return":
        return axes
