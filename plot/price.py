import sys
sys.path.append('../../../plot/')
import pandas as pd

import common
from matplotlib import pyplot as plt

from behindbars import BehindBars
from seaborn import color_palette

def plot(files,title,x_var,show="display",metric="time_ms",filter_out={},sort=True,shape="bar",order=None):
    plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
    fig = plt.figure(figsize=(20,10), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_prop_cycle('color', color_palette("colorblind"))

    times = []
    bws = []
    perfs = []
    prices = []
    bars = BehindBars(0.85)


    for label,data_or_path in files:
        if type(data_or_path) == str:
            data = pd.read_csv(data_or_path,comment="#")
        else:
            data = data_or_path.copy()

        for col_name,fun in filter_out.items():
            if col_name not in data:
                continue
            data = data[data[col_name].apply(fun)]

        # groupby_cols.append(metric)
        groupby_cols = [x_var,"comp_algo","uncomp_bytes","scale_factor"]
        # if metric in data.columns and metric not in groupby_cols:
        #     groupby_cols.append(metric)
        data = data.groupby(groupby_cols,as_index=False).agg(price=("price","mean"),priceperf=("priceperf","mean"),timeprice=("timeprice","mean"),
                                                            time_ms=("time_ms","mean"),time_ms_std=("time_ms","std"),
                                                            comp_bytes=("comp_bytes","mean"))
        #
        if "pruned_bytes" not in data.columns:
            data["pruned_bytes"] = 0

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
            alpha = {"Golap":1,
                        "Optimal Uncompressed GPU": 1,
                        "Optimal Uncompressed CPU": 1}
            real_label = label
            # if label == "Golap":
            #     label = "     "
            bars.add_cat(x,cur_data[metric],label=f"{label} {comp_algo}",args={"alpha":alpha[real_label]})
            times += list(cur_data["time_ms"])
            bws += list(cur_data["effective_bw"])
            perfs += list(cur_data["priceperf"])
            prices += list(cur_data["price"])
    # if sort:
    #     bars.sort(prefixes=set(name for name,_ in files))
    # if order:
    #     bars.order(order)

    xss,yss = bars.do_plot(shape="bar",axes=axes, edgecolor="black", linewidth=2)
    i = 0
    for xs,ys in zip(xss,yss):
        for x,y in zip(xs,ys):
            timestr = f"{times[i]:.0f}".rjust(4)
            axes.text(x-0.13,y+0.00005,f"{timestr}\n   ms",fontsize=20)
            # axes.text(x-0.13,y+0.00005,f"{bws[i]:.1f}\nGB/s",fontsize=18)
            # axes.text(x-0.1,y+0.0001,f"{perfs[i]:.1f}\nGB/s/$",fontsize=18,rotation=90)
            # axes.text(x-0.1,y+1000,f"{prices[i]*100:.2f}\nct",fontsize=18,rotation=0)
            i += 1

    common.axes_prepare(axes,metric)

    axes.set_xlabel(x_var)
    axes.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
    axes.grid(axis="y")
    axes.set_axisbelow(True)

    fig.tight_layout()

    return axes




ASSUMED_COMP_RATIO = 3


COMPARABLE_CPU_DATA = []
MAX_CPU_DATA = []
MAX_GPU_DATA = []

for dataset,queries in [("SSB",["query1","query2","query3","query4"]),
                        ("TAXI",["query1","query2"])]:
    for query in queries:
        UNCOMP_BYTES = common.uncomp_bytes(100,query+".1",dataset)

        COMP_CPU_TIME = UNCOMP_BYTES /(1<<30) / common.CONFIG["CPUNODE_SSD_BW_GIBS_ALT"] * 1000
        COMP_CPU_TIME_COMPRESSED = UNCOMP_BYTES/ASSUMED_COMP_RATIO /(1<<30) / common.CONFIG["CPUNODE_SSD_BW_GIBS_ALT"] * 1000
        MIN_CPU_TIME = UNCOMP_BYTES /(1<<30) / common.CONFIG["CPUNODE_SSD_BW_GIBS"] * 1000
        MIN_CPU_TIME_COMPRESSED = UNCOMP_BYTES/ASSUMED_COMP_RATIO /(1<<30) / common.CONFIG["CPUNODE_SSD_BW_GIBS"] * 1000
        MIN_GPU_TIME = UNCOMP_BYTES /(1<<30) / common.CONFIG["GPUNODE_SSD_BW_GIBS"] * 1000
        MIN_GPU_TIME_COMPRESSED = UNCOMP_BYTES/ASSUMED_COMP_RATIO /(1<<30) / common.CONFIG["GPUNODE_SSD_BW_GIBS"] * 1000

        COMPARABLE_CPU_DATA.append([dataset+"\n"+query,"",UNCOMP_BYTES,100,UNCOMP_BYTES,
                            COMP_CPU_TIME,(COMP_CPU_TIME/(1000*60*60))*common.CONFIG["CPUNODE_ONDEMAND_PRICE_H_ALT"]])
        MAX_CPU_DATA.append([dataset+"\n"+query,"",UNCOMP_BYTES,100,UNCOMP_BYTES,
                            MIN_CPU_TIME,(MIN_CPU_TIME/(1000*60*60))*common.CONFIG["CPUNODE_ONDEMAND_PRICE_H"]])
        # MAX_CPU_DATA.append([dataset+"\n"+query,"COMPRESSED",UNCOMP_BYTES,100,UNCOMP_BYTES/ASSUMED_COMP_RATIO,
        #                     MIN_CPU_TIME_COMPRESSED,(MIN_CPU_TIME_COMPRESSED/(1000*60*60))*common.CONFIG["CPUNODE_ONDEMAND_PRICE_H_ALT"]])
        MAX_GPU_DATA.append([dataset+"\n"+query,"",UNCOMP_BYTES,100,UNCOMP_BYTES,
                            MIN_GPU_TIME,(MIN_GPU_TIME/(1000*60*60))*common.CONFIG["GPUNODE_ONDEMAND_PRICE_H"]])
        # MAX_GPU_DATA.append([dataset+"\n"+query,"COMPRESSED",UNCOMP_BYTES,100,UNCOMP_BYTES/ASSUMED_COMP_RATIO,
        #                     MIN_GPU_TIME_COMPRESSED,(MIN_GPU_TIME_COMPRESSED/(1000*60*60))*common.CONFIG["GPUNODE_ONDEMAND_PRICE_H"]])
COMPARABLE_CPU_DATA = pd.DataFrame(COMPARABLE_CPU_DATA,columns=["query","comp_algo","uncomp_bytes","scale_factor","comp_bytes","time_ms","price"])
MAX_CPU_DATA = pd.DataFrame(MAX_CPU_DATA,columns=["query","comp_algo","uncomp_bytes","scale_factor","comp_bytes","time_ms","price"])
MAX_GPU_DATA = pd.DataFrame(MAX_GPU_DATA,columns=["query","comp_algo","uncomp_bytes","scale_factor","comp_bytes","time_ms","price"])

COMPARABLE_CPU_DATA["effective_bw"] = (1000.0 / (1<<30)) * COMPARABLE_CPU_DATA["uncomp_bytes"]/COMPARABLE_CPU_DATA["time_ms"]
COMPARABLE_CPU_DATA["priceperf"] = COMPARABLE_CPU_DATA.effective_bw / COMPARABLE_CPU_DATA.price
COMPARABLE_CPU_DATA["timeprice"] = (1/COMPARABLE_CPU_DATA.time_ms) / COMPARABLE_CPU_DATA.price
MAX_CPU_DATA["effective_bw"] = (1000.0 / (1<<30)) * MAX_CPU_DATA["uncomp_bytes"]/MAX_CPU_DATA["time_ms"]
MAX_CPU_DATA["priceperf"] = MAX_CPU_DATA.effective_bw / MAX_CPU_DATA.price
MAX_CPU_DATA["timeprice"] = (1/MAX_CPU_DATA.time_ms) / MAX_CPU_DATA.price
MAX_GPU_DATA["effective_bw"] = (1000.0 / (1<<30)) * MAX_GPU_DATA["uncomp_bytes"]/MAX_GPU_DATA["time_ms"]
MAX_GPU_DATA["priceperf"] = MAX_GPU_DATA.effective_bw / MAX_GPU_DATA.price
MAX_GPU_DATA["timeprice"] = (1/MAX_GPU_DATA.time_ms) / MAX_GPU_DATA.price


GOLAP_DATA = pd.read_csv("../usecases/ssb/aws_results/query_export_2.csv",comment="#") # ssd2gpu2cpu (compat)
GOLAP_DATA = GOLAP_DATA.loc[GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
GOLAP_DATA = GOLAP_DATA.query("comp_algo != 'UNCOMPRESSED'")
GOLAP_DATA.comp_algo = ""
GOLAP_DATA["query"] = "SSB\n"+GOLAP_DATA["query"].str.slice(stop=6)
GOLAP_DATA["effective_bw"] = (1000.0 / (1<<30)) * GOLAP_DATA["uncomp_bytes"]/GOLAP_DATA["time_ms"]
GOLAP_DATA["ratio_alt"] = GOLAP_DATA["uncomp_bytes"]/GOLAP_DATA["comp_bytes"]
GOLAP_DATA["price"] = (GOLAP_DATA.time_ms / 1000 / 60 / 60) * common.CONFIG["GPUNODE_ONDEMAND_PRICE_H"]
GOLAP_DATA["priceperf"] = GOLAP_DATA.effective_bw / GOLAP_DATA.price
GOLAP_DATA["timeprice"] = (1/GOLAP_DATA.time_ms) / GOLAP_DATA.price


GOLAP_TAXI_DATA = pd.read_csv("../usecases/taxi/aws_results/query12_export.csv",comment="#") # ssd2gpu2cpu (compat)
GOLAP_TAXI_DATA = GOLAP_TAXI_DATA.loc[GOLAP_TAXI_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
GOLAP_TAXI_DATA = GOLAP_TAXI_DATA.query("comp_algo != 'UNCOMPRESSED'")
GOLAP_TAXI_DATA.comp_algo = ""
GOLAP_TAXI_DATA["query"] = "TAXI\n"+GOLAP_TAXI_DATA["query"].str.slice(stop=6)
GOLAP_TAXI_DATA["effective_bw"] = (1000.0 / (1<<30)) * GOLAP_TAXI_DATA["uncomp_bytes"]/GOLAP_TAXI_DATA["time_ms"]
GOLAP_TAXI_DATA["ratio_alt"] = GOLAP_TAXI_DATA["uncomp_bytes"]/GOLAP_TAXI_DATA["comp_bytes"]
GOLAP_TAXI_DATA["price"] = (GOLAP_TAXI_DATA.time_ms / 1000 / 60 / 60) * common.CONFIG["GPUNODE_ONDEMAND_PRICE_H"]
GOLAP_TAXI_DATA["priceperf"] = GOLAP_TAXI_DATA.effective_bw / GOLAP_TAXI_DATA.price
GOLAP_TAXI_DATA["timeprice"] = (1/GOLAP_TAXI_DATA.time_ms) / GOLAP_TAXI_DATA.price

# print("GOLAP_DATA:\n",GOLAP_DATA[["query","uncomp_bytes","comp_bytes","comp_algo","pruning","time_ms","ratio_alt","effective_bw","price"]].to_string())
# print("GOLAP_TAXI_DATA:\n",GOLAP_TAXI_DATA[["query","uncomp_bytes","comp_bytes","comp_algo","pruning","time_ms","ratio_alt","effective_bw","price"]].to_string())
# print("MAX_GPU_DATA:\n",MAX_GPU_DATA[["query","uncomp_bytes","comp_algo","time_ms","price"]])
# print("MAX_CPU_DATA:\n",MAX_CPU_DATA[["query","uncomp_bytes","comp_algo","time_ms","price"]])



files = [
             ["Golap",pd.concat([GOLAP_DATA,GOLAP_TAXI_DATA])],
             # ["Best GPU -", MAX_GPU_DATA_COMPRESSED],
             ["Optimal Uncompressed GPU", MAX_GPU_DATA],
             # ["Golap CPU2GPU",GOLAP_CPU2GPU_DATA],
             # ["Optimal Uncompressed CPU", MAX_CPU_DATA],
             ["Optimal Uncompressed CPU", COMPARABLE_CPU_DATA],
             # ["Best CPU -", MAX_CPU_DATA_COMPRESSED],
            ]
show = "return"
x_val = "query"
shape = "bar"

# axes = plot(files,"Select",x_val,show,"comp_bytes_gb",sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"ratio",sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"effective_bw",sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"time_ms",sort=False,shape=shape)
axes = plot(files,"Select",x_val,show,"price",sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"actual_bw",sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"speedup",sort=False,shape=shape)

# axes.axhline(common.CONFIG["GPUNODE_SSD_BW_GIBS"],lw=2,ls="--",c="#FF4E5B",zorder=0)
# axes.axhline(common.CONFIG["CPUNODE_SSD_BW_GIBS"],lw=2,ls="--",c="#1AA1F1",zorder=0)
# axes.text(0.48,16.2,"SSD",fontsize=18,c="#FF4E5B")
# axes.text(0.48,26.5,"CPU\n<->\nGPU",fontsize=18,c="#1AA1F1")
# axes.set_yscale("log")

# axes.set_ylim([0,90000])
# axes.set_ylabel("Performance per Dollar [GiB/s/$]")

axes.set_ylim([0,0.0075])
axes.set_ylabel("Price per Query [$]")

axes.set_xlabel("")
# t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
# axes.set_xticklabels(t,rotation=25)
# axes.set_xticklabels(["select"])

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/price.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()
