import sys
sys.path.append('../../../plot/')
import pandas as pd

from matplotlib import pyplot as plt
from plot_various import plot

GPU_SELECT_DATA = pd.read_csv("../results/select_export.csv",comment="#") # select * all columns
# GPU_SELECT_DATA["query"] = GPU_SELECT_DATA["query"].str.slice(stop=6)
# GPU_SELECT_DATA = GPU_SELECT_DATA[GPU_SELECT_DATA.chunk_bytes == (1<<28)]

best_gpu = GPU_SELECT_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()
GPU_SELECT_DATA = GPU_SELECT_DATA.loc[best_gpu]


files = [
             # ["SSD2GPU2CPU", GPU_SELECT_DATA[GPU_SELECT_DATA.dataflow == "SSD2GPU2CPU"]],
             ["", GPU_SELECT_DATA[GPU_SELECT_DATA.dataflow == "SSD2GPU"]],
            ]
filter_out = {}
show = "return"
# show = "print"
x_val = "query"
# x_val = "chunk_bytes"
shape = "bar"

# axes = plot(files,"Select",x_val,show,"time_ms",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"comp_bytes_gb",filter_out,shape=shape)
axes = plot(files,"Select",x_val,show,"ratio",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"effective_bw",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"actual_bw",filter_out,shape=shape)
# axes = plot(files,"Select",x_val,show,"speedup",filter_out,shape=shape)


# axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
# axes.set_yscale("log")
# axes.set_ylim([0,50])
t  = [xtick.get_text()[7:] for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t,rotation=25)
# axes.set_xticklabels(["select"])

plt.tight_layout()
plt.show()
