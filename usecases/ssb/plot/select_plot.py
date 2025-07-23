import sys
sys.path.append('../../../plot/')
import common
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullFormatter
from plot_various import plot

show = "return"
# show = "print"
x_val = "query"
# shape = "box"
# shape = "violin"
shape = "bar"

GPU_SELECT_DATA = pd.read_csv("../results/select_export.csv",comment="#")
GPU_SELECT_DATA = GPU_SELECT_DATA.query("comp_algo != 'BEST_RATIO_COMP'")
best_gpu = GPU_SELECT_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()
GPU_SELECT_DATA = GPU_SELECT_DATA.loc[best_gpu]
# GPU_SELECT_DATA["query"] = GPU_SELECT_DATA["query"].str.slice(stop=6)


CPU_SELECT_DATA = pd.read_csv("../results/select_cpu_export.csv",comment="#") # select * all columns
# CPU_SELECT_DATA["query"] = CPU_SELECT_DATA["query"].str.slice(stop=6)
best_cpu = CPU_SELECT_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()
CPU_SELECT_DATA = CPU_SELECT_DATA.loc[best_cpu]


files = [
             ["SSD2GPU2CPU", GPU_SELECT_DATA[GPU_SELECT_DATA.dataflow == "SSD2GPU2CPU"]],
             ["SSD2GPU", GPU_SELECT_DATA[GPU_SELECT_DATA.dataflow == "SSD2GPU"]],
             ["SSD2CPU", CPU_SELECT_DATA],

            ]
filter_out = {}


# axes = plot(files,"Select",x_val,show,"time_ms",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"comp_bytes_gb",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"ratio",filter_out,shape=shape)
axes = plot(files,"Select",x_val,show,"effective_bw",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"actual_bw",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"speedup",filter_out,shape=shape)


axes.text(0.605,16.2,"SSD BW",fontsize=18,c="#FF4E5B")
axes.text(0.605,25,"CPU<->GPU BW",fontsize=18,c="#1AA1F1")
axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
axes.set_yscale("log")
axes.set_yticks([5,10,20,50,100,200,300])
axes.yaxis.set_major_formatter(ScalarFormatter())
axes.yaxis.set_minor_formatter(NullFormatter())
# axes.set_ylim([0,120])
axes.set_xlim([0.6,1.35])
t  = [xtick.get_text()[7:] for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t,rotation=25)
axes.set_xlabel("Column Scan")

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/select_comparison.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()
