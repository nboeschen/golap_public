import sys
sys.path.append('../../../plot/')
import pandas as pd

from matplotlib import pyplot as plt
from plot_various import plot


GOLAP_SELECT = pd.read_csv("../results/select_10.07-11.04.24.csv",comment="#") # select * all columns
# GOLAP_SELECT["query"] = GOLAP_SELECT["query"].str.slice(stop=6)
# GOLAP_SELECT["uncomp_bytes"] = -1

GOLAP_SOME_COLUMNS = pd.read_csv("../results/select_11.24-16.03.56.csv",comment="#") # select * all columns

DATA3C = pd.read_csv("../results/coexec_11.23-11.37.16.csv",comment="#") # 1GB
DATA3C["query"] = DATA3C["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'

GOLAP_QUERIES = pd.read_csv("../results/query_export.csv",comment="#")


files = [

             # ["DuckDB" , "../results/duckdb_count_09.30-11.14.37.csv"], # select/count *
             # ["CPU", "../results/select_cpu_11.24-13.49.52.csv"], # SSD2CPU many chunksizes
             # ["GPU", GOLAP_SOME_COLUMNS], # same as above setup
             ["GPU", GOLAP_SELECT],
             # ["GPU", GOLAP_QUERIES],
             # ["Golap", "../results/select_10.28-09.27.32.csv"], # select_orderpriority, 5 pipelines
             # ["Golap", "../results/query_10.28-13.11.09.csv"], # queries
             # ["Golap", "../results/select_cpu_11.15-12.41.06.csv"], # SSD2CPU larger chunksizes
             # ["Golap", "../results/select_cpu_11.15-14.42.29.csv"], # SSD2CPU smaller chunksizes

            ]
filter_out = {
                # "query": lambda x: not x.endswith("_sql") and x.startswith("query1")
                "query": lambda x: x.endswith("_orderpriority"),
                # "query": lambda x: x.startswith("query3.2"),
                # "query": lambda x: x.startswith("query3.3"),
                # "dataflow": lambda x: x == "SSD2GPU"
                    }
show = "display"
x_val = "chunk_bytes"
title="chunksizes_query3.3_3.png"

# axis = plot(files,title,x_val,show,"time_ms",filter_out)
# axis = plot(files,title,x_val,show,"comp_bytes_gb",filter_out)
# axis = plot(files,title,x_val,show,"ratio",filter_out)
axis = plot(files,title,x_val,show,"effective_bw",filter_out)
# axis = plot(files,title,x_val,show,"actual_bw",filter_out)
# axis = plot(files,title,x_val,show,"speedup",filter_out)

# plt.show()
