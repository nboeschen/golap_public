import sys
import common
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullFormatter
from seaborn import color_palette

plt.rcParams.update({'font.size': 32,"pdf.fonttype": 42,"ps.fonttype": 42})
fig = plt.figure(figsize=(20,10), dpi=100)
axes = fig.add_subplot(1, 1, 1)
axes.set_prop_cycle('color', color_palette("colorblind"))



SSB = pd.read_csv("../usecases/ssb/results/select_export.csv",comment="#") # select * all columns
SSB = SSB.query("dataflow == 'SSD2GPU'")
SSB = SSB.query("uncomp_bytes >= 800000000")
SSB.loc[SSB['comp_algo'] != "UNCOMPRESSED", "comp_algo"] = "COMPRESSED"
SSB = SSB.loc[SSB.groupby(["query","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
# SSB = SSB.loc[SSB.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
# SSB["query"] = SSB["query"].str.slice(stop=6)
# print(SSB.to_string())

TAXI = pd.read_csv("../usecases/taxi/results/select_export.csv",comment="#") # select * all columns
TAXI = TAXI.query("dataflow == 'SSD2GPU'")
TAXI = TAXI.query("uncomp_bytes >= 800000000")
TAXI.loc[TAXI['comp_algo'] != "UNCOMPRESSED", "comp_algo"] = "COMPRESSED"
TAXI = TAXI.loc[TAXI.groupby(["query","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
# TAXI = TAXI.loc[TAXI.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
# TAXI["query"] = TAXI["query"].str.slice(stop=6)
# print(TAXI.to_string())

TPCH = pd.read_csv("../usecases/tpch/results/select_export_3.csv", comment="#")
TPCH = TPCH.query("dataflow == 'SSD2GPU'")
TPCH = TPCH.query("uncomp_bytes >= 800000000")
TPCH.loc[TPCH['comp_algo'] != "UNCOMPRESSED", "comp_algo"] = "COMPRESSED"
TPCH = TPCH.loc[TPCH.groupby(["query","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
# TPCH = TPCH.loc[TPCH.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]


# CPU_SELECT_DATA = pd.read_csv("../results/select_cpu_export.csv",comment="#")
# # CPU_SELECT_DATA = CPU_SELECT_DATA.query("comp_algo == 'UNCOMPRESSED'")
# CPU_SELECT_DATA = CPU_SELECT_DATA.query("uncomp_bytes >= 2000000000")
# CPU_SELECT_DATA = CPU_SELECT_DATA.query("comp_algo == 'LZ4'")
# CPU_SELECT_DATA = CPU_SELECT_DATA.loc[CPU_SELECT_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]


for label,data in [
        ("Taxi", TAXI), ("SSB", SSB),
        ("TPC-H", TPCH),
        ]:
    data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
    data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]

    for dataflow,cur_data in data.groupby("dataflow"):

        # sns.regplot(x=cur_data["ratio_alt"], y=cur_data["effective_bw"], ci=50, label=f"ColumnScan {dataflow}", line_kws={"ls":(0, (5, 20))});
        marker = {"Taxi":"o","SSB":"^","TPC-H":"*"}[label]
        plt.scatter(cur_data["ratio_alt"], cur_data["effective_bw"], 300, edgecolors="#00000000", linewidths=2, marker=marker, alpha=0.7, label=f"{label} Column")

        # plot column names and comp_bytes
        # for query,query_data in cur_data.groupby("query"):
        #     short_text = "_".join(query.split("_")[1:])
        #     plt.text(query_data["ratio_alt"], query_data["effective_bw"],short_text
        #                 + " " + common.hrsize(query_data.comp_bytes.values[0]))


# ["Memory Bandwidth", pd.DataFrame([["scan","UNCOMPRESSED",1<<30,1<<30,1000]],columns=["query","comp_algo","uncomp_bytes","comp_bytes","time_ms"])],

# plt.plot(1,100,"x",label="Memory Bandwidth (Seq)")


# for data in [GPU_SELECT_DATA]:
#     data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
#     data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]
#     for comp_algo,cur_data in data.groupby("comp_algo"):
#         plt.plot(cur_data["ratio_alt"], cur_data["effective_bw"], "o", label=f"ColumnScan {comp_algo}")

x = np.linspace(1,23,1000)
plt.plot(x,np.array(x)*common.CONFIG["DGXSSDBW"],"--",lw=6,c="#72C771",zorder=0,label="_")

# axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
plt.plot([0.8,23],[common.CONFIG["DGXSSDBW"],common.CONFIG["DGXSSDBW"]],lw=6,ls="--",c="#FF4E5B",zorder=0,label="_")
plt.text(6,23,"SSD BW",c="#FF4E5B",fontsize=35)
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
axes.set_yscale("log")

# paper variant
# axes.set_xlim([0.5,23])
# axes.set_xticks(range(1,24))
# plt.text(9,250,"Optimal Decompression BW",c="#72C771",fontsize=35,rotation=10)
# axes.set_yticks([10,20,50,100,200,300])

# shorter variant for poster
axes.set_xlim([0.5,11])
axes.set_ylim([10,350])
axes.set_yticks([10,20,50,100,200])
plt.text(5,130,"Optimal Decompression BW",c="#72C771",fontsize=35,rotation=8)

# axes.set_xticks(range(1,10))
# axes.set_xscale("log")
# axes.set_xticks([1,2,3,5,10,20])
# axes.xaxis.set_major_formatter(ScalarFormatter())
# axes.xaxis.set_minor_formatter(NullFormatter())


axes.yaxis.set_major_formatter(ScalarFormatter())
axes.yaxis.set_minor_formatter(NullFormatter())

axes.set_xlabel("Compression [ratio]")
axes.set_ylabel("Effective Bandwidth [GiB/s]")
axes.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
axes.set_axisbelow(True)

plt.grid()
plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/datasets_comp_ratio_vs_decompression_speed_compact.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

