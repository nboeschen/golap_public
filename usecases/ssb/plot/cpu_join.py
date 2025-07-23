import sys
sys.path.append('../../../plot/')
import pandas as pd
from matplotlib import pyplot as plt
from plot_various import plot

DATA3A = pd.read_csv("../results/coexec_10.27-13.44.29.csv",comment="#")
DATA3A["query"] = DATA3A["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'

DATA3B = pd.read_csv("../results/coexec_10.31-14.39.12.csv",comment="#")
DATA3B["query"] = DATA3B["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'

DATA3D = pd.read_csv("../results/query3d_11.28-09.45.04.csv",comment="#")
DATA3D["query"] = DATA3D["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'

DATA3C = pd.read_csv("../results/coexec_11.23-11.37.16.csv",comment="#") # 1GB
DATA3C["query"] = DATA3C["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'

DATA3C500 = pd.read_csv("../results/coexec_11.24-12.07.40.csv",comment="#") # 500 mb
DATA3C500["query"] = DATA3C500["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'

DATA3C_UVA = pd.read_csv("../results/coexec_11.25-12.08.15.csv",comment="#") # 1GB
# DATA3C_UVA["query"] = DATA3C_UVA["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'

DUCKDB = pd.read_csv("../results/duckdb_query_09.07-12.07.09.csv",comment="#")
DUCKDB["customer_factor"] = 1
DUCKDB["scale_factor"] = 100

DATA3X = pd.read_csv("../results/coexec_11.17-14.46.46.csv",comment="#")
DATA3X["query"] = DATA3X["query"].str.slice(stop=8)  # remove suffix from e.g. 'query3.1a'


files = [
             ["[3a: Customer GPU Decomp + Pred]", DATA3A],
             ["[3:  GPU only]", "../results/query3_11.25-13.59.29.csv"],
             # ["[3b: GPU Decomp Only]", DATA3B],
             # ["[DuckDB]", DUCKDB],
             ["[3c: Customer HT in CPU Memory, UM 1GB]", DATA3C], # without additional host thread
             ["[3c: Customer HT in CPU Memory, UVA]", DATA3C_UVA], # without additional host thread
             # ["[3c500: HT in CPU Memory, UM 500mb]", DATA3C500], # without additional host thread
             ["[3d: Customer Pipeline on Host]", DATA3D], # without host compression!
             # ["[3x: Customer Pipeline on Host]", DATA3X], # with host compression

             ]
filter_out = {
                # "query": lambda x: not x.endswith("_sql") and x.startswith("query1")
                "query": lambda x: x.startswith("query3"),
                # "customer_factor": lambda x: x == 100,
                "customer_factor": lambda x: x in [1,50,100],
                "comp_algo": lambda x: x != "UNCOMPRESSED",
                    }
show = "return"
x_val = "query"
title="CPU Join"
x_val = "customer_factor"
sort = True

axis = plot(files,title,x_val,show,"time_ms",filter_out, sort=sort)
# axis = plot(files,title,x_val,show,"comp_bytes_gb",filter_out, sort=sort)
# axis = plot(files,title,x_val,show,"ratio",filter_out, sort=sort)
# axis = plot(files,title,x_val,show,"effective_bw",filter_out, sort=sort)
# axis = plot(files,title,x_val,show,"device_mem_used",filter_out, sort=sort)
# axis = plot(files,title,x_val,show,"actual_bw",filter_out, sort=sort)
# axis = plot(files,title,x_val,show,"speedup",filter_out, sort=sort)

axis.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.show()
