import sys
sys.path.append('../../../plot/')
import pandas as pd

import common
from matplotlib import pyplot as plt
from plot_various import plot

GOLAP_DATA = pd.read_csv("../results/query_export.csv",comment="#")
# GOLAP_DATA["query"] = GOLAP_DATA["query"].str.slice(stop=6)
GOLAP_DATA = GOLAP_DATA[GOLAP_DATA.chunk_bytes == (1<<24)]
# best_gpu = GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()
# GOLAP_DATA = GOLAP_DATA.loc[best_gpu]


DUCKDB = pd.concat([
                pd.read_csv("../results/duckdb_query_export_1.csv",comment="#"),
                pd.read_csv("../results/duckdb_query_export_2.csv",comment="#"),
                ])
# duckdb best compression is snappy
DUCKDB = DUCKDB.query("comp_algo != 'ZSTD' & comp_algo != 'GZIP'")
# DUCKDB.comp_bytes = DUCKDB.io_bytes
# DUCKDB["query"] = DUCKDB["query"].str.slice(stop=6)

INMEM = pd.read_csv("../results/inmem_query_export.csv",comment="#")
# INMEM["query"] = INMEM["query"].str.slice(stop=6)
# best_inmem = INMEM.groupby(["query"],as_index=False)["time_ms"].transform("idxmin").unique()
# INMEM = INMEM.loc[best_inmem]

files = [
             ["Golap", GOLAP_DATA],
             ["DuckDB", DUCKDB],
             ["Handcoded INMEM", INMEM],
            ]
filter_out = {
                # "query": lambda x: not x.endswith("_sql") and x.startswith("query1")
                # "query": lambda x: x.startswith("query3"),
                # "scale_factor": lambda x: x == 200,
                # "chunk_bytes": lambda x: x == (1<<26), # sub optimal chunk sizes
                # "query": lambda x: not x.endswith("_shippriority"), # empty column
                # "dataflow": lambda x: x == "SSD2GPU" or x == "SSD2CPU",
                # "chunk_bytes": lambda x: x == (1<<28)
                    }
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

axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
# axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
axes.text(0.48,16.2,"SSD",fontsize=18,c="#FF4E5B")
# axes.text(0.48,26.5,"CPU\n<->\nGPU",fontsize=18,c="#1AA1F1")
# axes.set_yscale("log")
# axes.set_ylim([0,50])
t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t,rotation=25)
# axes.set_xticklabels(["select"])

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/avg_ssb_bw.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

# same for time


axes = plot(files,"Select",x_val,show,"time_ms",filter_out,sort=True,shape=shape)
t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t,rotation=25)

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/avg_ssb_time.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()
