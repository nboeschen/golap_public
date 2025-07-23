import sys
sys.path.append('../../../plot/')
import common

import pandas as pd
from matplotlib import pyplot as plt
from plot_various import plot


GOLAP_DATA = pd.read_csv("../results/query_export.csv",comment="#")
GOLAP_DATA["query"] = GOLAP_DATA["query"].str.slice(stop=6)
best_gpu = GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()
GOLAP_DATA = GOLAP_DATA.loc[best_gpu]

DASK_DATA = pd.read_csv("../results/dask_export.csv",comment="#") # gpu dataframes
DASK_DATA["query"] = DASK_DATA["query"].str.slice(stop=6)
best_dask = DASK_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()
DASK_DATA = DASK_DATA.loc[best_dask]

DUCKDB = pd.read_csv("../results/duckdb_query_export_1.csv",comment="#")
DUCKDB["query"] = DUCKDB["query"].str.slice(stop=6)
best_duckdb = DUCKDB.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()
DUCKDB = DUCKDB.loc[best_duckdb]

INMEM =pd.read_csv("../results/inmem_query_export.csv",comment="#")
INMEM["query"] = INMEM["query"].str.slice(stop=6)
best_inmem = INMEM.groupby(["query"],as_index=False)["time_ms"].transform("idxmin").unique()
INMEM = INMEM.loc[best_inmem]


files = [
         ["Golap", GOLAP_DATA],
         ["Dask", DASK_DATA],
         ["DuckDB", DUCKDB],
         ["Handcoded INMEM", INMEM],
        ]
filter_out = {
                # "query": lambda x: not x.endswith("_sql") and x.startswith("query1")
                # "query": lambda x: x.startswith("query1")
                # "comp_algo": lambda x: x != "GZIP" and x != "ZSTD"
                }

x_var = "query"

axes = plot(files,"Front",x_var,"return","time_ms",filter_out)
# axes = plot(files,"Front",x_var,"return","comp_bytes_gb",filter_out)
# axes = plot(files,"Front",x_var,"return","ratio",filter_out)
# axes = plot(files,"Front",x_var,"return","effective_bw",filter_out)
# axes = plot(files,"Front",x_var,"return","actual_bw",filter_out)
# axes = plot(files,"Front",x_var,"return","speedup",filter_out)

t  = [f"AVG {xtick.get_text()}" for xtick in axes.get_xticklabels()]
axes.set_xticklabels(t)

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/front.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()