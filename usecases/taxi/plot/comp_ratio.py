import sys
sys.path.append('../../../plot/')
import common
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullFormatter
from seaborn import color_palette

plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
fig = plt.figure(figsize=(20,10), dpi=100)
axes = fig.add_subplot(1, 1, 1)
axes.set_prop_cycle('color', color_palette("colorblind"))

GPU_SELECT_DATA = pd.read_csv("../results/select_export.csv",comment="#") # select * all columns
GPU_SELECT_DATA = GPU_SELECT_DATA.query("dataflow == 'SSD2GPU'")
# GPU_SELECT_DATA = GPU_SELECT_DATA.query("uncomp_bytes >= 200000000")
GPU_SELECT_DATA = GPU_SELECT_DATA.loc[GPU_SELECT_DATA.groupby(["query","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
# GPU_SELECT_DATA["query"] = GPU_SELECT_DATA["query"].str.slice(stop=6)
# print(GPU_SELECT_DATA.to_string())


for data in [GPU_SELECT_DATA]:
    data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
    data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]

    for dataflow,cur_data in data.groupby("dataflow"):

        cur_data["efficiency"] = cur_data["effective_bw"] / (cur_data["ratio_alt"]*common.CONFIG["DGXSSDBW"])
        # print(cur_data[["query","ratio_alt","chunk_bytes","efficiency"]])
        # sns.regplot(x=cur_data["ratio_alt"], y=cur_data["effective_bw"], ci=50, label=f"ColumnScan {dataflow}", line_kws={"ls":(0, (5, 20))});
        plt.plot(cur_data["ratio_alt"], cur_data["effective_bw"], "o", markersize=10, markeredgecolor="#000000", label=f"ColumnScan {dataflow}")


# ["Memory Bandwidth", pd.DataFrame([["scan","UNCOMPRESSED",1<<30,1<<30,1000]],columns=["query","comp_algo","uncomp_bytes","comp_bytes","time_ms"])],

# plt.plot(1,100,"x",label="Memory Bandwidth (Seq)")


# for data in [GPU_SELECT_DATA]:
#     data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
#     data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]
#     for comp_algo,cur_data in data.groupby("comp_algo"):
#         plt.plot(cur_data["ratio_alt"], cur_data["effective_bw"], "o", label=f"ColumnScan {comp_algo}")

x = np.linspace(1,22.7,1000)
plt.plot(x,np.array(x)*common.CONFIG["DGXSSDBW"],"--",lw=2,c="#0EB417",zorder=0,label="Optimal Decompression BW")

# axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
plt.plot([0.8,22.7],[common.CONFIG["DGXSSDBW"],common.CONFIG["DGXSSDBW"]],lw=2,ls="--",c="#FF4E5B",zorder=0,label="SSD Bandwidth")
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)

axes.set_xlim([0.5,15])
axes.set_xticks(range(1,16))
# axes.set_xscale("log")
# axes.set_xticks([1,2,3,5,10,20])
# axes.xaxis.set_major_formatter(ScalarFormatter())
# axes.xaxis.set_minor_formatter(NullFormatter())

axes.set_yscale("log")

axes.set_yticks([1,5,10,20,50,100,200,300])
axes.yaxis.set_major_formatter(ScalarFormatter())
axes.yaxis.set_minor_formatter(NullFormatter())

axes.set_xlabel("Compression [ratio]")
axes.set_ylabel("Effective Bandwidth [GB/s, Log Scale]")
axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)


plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/comp_ratio_vs_decompression_speed.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

