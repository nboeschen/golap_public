import sys
sys.path.append('../../../plot/')
import common
import pandas as pd

from matplotlib import pyplot as plt
from plot_various import plot


GOLAP_DATA = pd.read_csv("../results/query_export.csv",comment="#")
# GOLAP_DATA["query"] = GOLAP_DATA["query"].str.slice(stop=6)
GOLAP_DATA = GOLAP_DATA.loc[GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]


files = [
             ["Golap", GOLAP_DATA],
            
            ]
filter_out = {}
show = "return"
# show = "print"
x_val = "query"
# x_val = "chunk_bytes"
shape = "bar"

# axes = plot(files,"Select",x_val,show,"comp_bytes_gb",filter_out,sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"ratio",filter_out,sort=False,shape=shape)
axes = plot(files,"Select",x_val,show,"effective_bw",filter_out,sort=True,shape=shape)
# axes = plot(files,"Select",x_val,show,"actual_bw",filter_out,sort=False,shape=shape)
# axes = plot(files,"Select",x_val,show,"speedup",filter_out,sort=False,shape=shape)

axes.axhline(19.5,lw=2,ls="--",c="#FF4E5B",zorder=0)
axes.axhline(24,lw=2,ls="--",c="#1AA1F1",zorder=0)
axes.text(0.48,16.2,"SSD",fontsize=18,c="#FF4E5B")
axes.text(0.48,26.5,"CPU\n<->\nGPU",fontsize=18,c="#1AA1F1")
# axes.set_yscale("log")
# axes.set_ylim([0,50])
t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t,rotation=25)
# axes.set_xticklabels(["select"])

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/avg_taxi_bw.pdf")
else:
    plt.show()


# same for time
axes = plot(files,"Select",x_val,show,"time_ms",filter_out,sort=True,shape=shape)
t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t,rotation=25)

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/avg_taxi_time.pdf",common.EMPTY_PDF_META)
else:
    plt.show()
