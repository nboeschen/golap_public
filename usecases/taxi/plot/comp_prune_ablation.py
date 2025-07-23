import sys
sys.path.append('../../../plot/')
import pandas as pd

import common
from matplotlib import pyplot as plt
from plot_various import plot

GOLAP_DATA = pd.read_csv("../results/query_export.csv",comment="#")
# GOLAP_DATA = GOLAP_DATA[GOLAP_DATA.chunk_bytes == (1<<24)]
# GOLAP_DATA = GOLAP_DATA.query("query.str.startswith('query1')")
GOLAP_DATA = GOLAP_DATA.loc[GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
# GOLAP_DATA["query"] = GOLAP_DATA["query"].str.slice(stop=5)
# GOLAP_DATA = GOLAP_DATA.loc[best_gpu]
GOLAP_DATA.loc[GOLAP_DATA.comp_algo == "UNCOMPRESSED","comp_algo"] = "No Compression"
GOLAP_DATA.loc[GOLAP_DATA.comp_algo == "BEST_BW_COMP","comp_algo"] = "Compression"


PRUNED_DATA = pd.concat([ pd.read_csv("../results/pruning_export_1.csv",comment="#").query("query.str.startswith('query1')"),
    pd.read_csv("../results/pruning_export_2.csv",comment="#")
    ])
# PRUNED_DATA = PRUNED_DATA.query("query.str.startswith('query1')")
PRUNED_DATA = PRUNED_DATA.loc[PRUNED_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
PRUNED_DATA.loc[PRUNED_DATA.comp_algo != "UNCOMPRESSED","comp_algo"] = "Compression"
PRUNED_DATA.loc[PRUNED_DATA.comp_algo == "UNCOMPRESSED","comp_algo"] = "No Compression"

PRUNED_DATA["pruned_ratio"] =  PRUNED_DATA["comp_bytes"] / (PRUNED_DATA["comp_bytes"] - PRUNED_DATA["pruned_bytes"])
PRUNED_DATA["effective_bw"] = (1000.0 / (1<<30)) * PRUNED_DATA["uncomp_bytes"]/PRUNED_DATA["time_ms"]
# print(PRUNED_DATA[["query","comp_algo","chunk_bytes", "effective_bw","pruned_ratio"]])

files = [
             ["No Pruning +", GOLAP_DATA],
             ["Pruning +", PRUNED_DATA]
            ]
filter_out = {}
show = "return"
# show = "print"
x_val = "query"
# x_val = "chunk_bytes"
shape = "bar"
order = ["No Pruning + No Compression","Pruning + No Compression", "No Pruning + Compression", "Pruning + Compression"]

# axes = plot(files,"Select",x_val,show,"comp_bytes_gb",filter_out,sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"ratio",filter_out,sort=False,shape=shape)
axes = plot(files,"Select",x_val,show,"effective_bw",filter_out,sort=False,shape=shape,order=order)
# axes = plot(files,"Select",x_val,show,"time_ms",filter_out,sort=False,shape=shape,order=order)
# axes = plot(files,"Select",x_val,show,"prune_ms",filter_out,sort=False,shape=shape,order=order)
# axes = plot(files,"Select",x_val,show,"pruned_bytes",filter_out,sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"actual_bw",filter_out,sort=False,shape=shape,order=order)
# axes = plot(files,"Select",x_val,show,"speedup",filter_out,sort=False,shape=shape)

axes.axhline(common.CONFIG["DGXSSDBW"],lw=6,ls="--",c="#FF4E5B",zorder=0)
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
# axes.text(0.48,16.2,"SSD",fontsize=18,c="#FF4E5B")
# axes.text(0.48,26.5,"CPU\n<->\nGPU",fontsize=18,c="#1AA1F1")
# axes.set_yscale("log")
# axes.set_ylim([0,50])
t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t,rotation=25)
axes.plot([],[],"--",lw=6,c="#FF4E5B",label="SSD Bandwidth")
axes.scatter([],[],color='none',label=" ")
axes.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/comp_prune_ablation.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

