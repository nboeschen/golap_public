import common
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullFormatter
from seaborn import color_palette

from behindbars import BehindBars
from math import inf
from functools import partial



def parse_commercial(file_path,dataset="SSB"):
    df = pd.DataFrame()
    queries = None
    if dataset == "SSB":
        queries = ["query1.1","query1.2","query1.3","query2.1","query2.2","query2.3",
                    "query3.1","query3.2","query3.3","query3.4","query4.1","query4.2","query4.3"]
    elif dataset == "TAXI":
        queries = ["query1.1","query1.2","query1.3","query2.1","query2.2","query2.3"]
    elif dataset == "TPCH":
        queries = ["query1","query3","query5"]
    idx = 0
    seen_table = False
    cur_size = 0

    with open(file_path) as file:
        for line in file:
            x = line.strip().split()
            if "Table" in line:
                if not seen_table:
                    seen_table = True
                if "Segment" not in line:
                    # cur_size += int(x[x.index("logical")+2][:-1])
                    cur_size += sum([int(s[:-1]) for s in x if s[:-1].isdigit()]) * (1<<12)

            if seen_table and "elapsed time" in line:
                seen_table = False
                df.loc[idx, "query"] = queries[idx%len(queries)]
                df.loc[idx, "scale_factor"] = 200
                df.loc[idx, "time_ms"] = int(x[-2])
                df.loc[idx, "uncomp_bytes"] = 0
                df.loc[idx, "comp_bytes"] = cur_size
                df.loc[idx, "comp_algo"] = "(Light Comp.)"
                idx += 1
                cur_size = 0
    return df

plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
fig, (axes1,axes2) = plt.subplots(1,2,sharey=True,figsize=(20,10), dpi=100)
axes1.set_prop_cycle('color', color_palette("colorblind"))
colors = color_palette("colorblind")
hatch = {   "Golap ":"",
                    "Golap (Heavy Comp.)":"",
                    "System X (Light Comp.)": "+",
                    "DuckDB (Uncompressed)": "o",
                    "DuckDB (Heavy Comp.)": "/",
                    "DuckDB (Heavy Comp.) +Pruning": "\\",
                    "Mordred (GPU/CPU In-Memory)": ".",
                    }
col = {   "Golap ":"",
                    "Golap (Heavy Comp.)":colors[0],
                    "System X (Light Comp.)": colors[1],
                    "DuckDB (Uncompressed)": colors[2],
                    "DuckDB (Heavy Comp.)": colors[3],
                    "DuckDB (Heavy Comp.) +Pruning": colors[4],
                    "Mordred (GPU/CPU In-Memory)": colors[5],
                    }

def plot(files,ax,title,x_var,metric="time_ms",sort=True,shape="bar",with_optimal_ssd_scan=False,with_optimal_inmem_scan=False):

    bars = BehindBars(0.8)


    ref = None
    for label,data_or_path in files:
        if type(data_or_path) == str:
            data = pd.read_csv(data_or_path,comment="#")
        else:
            data = data_or_path.copy()

        data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
        data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
        data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]
        data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]
        data["optimal_bw"] = data["ratio_alt"]*common.CONFIG["DGXSSDBW"]
        data["optimal_time"] = data["uncomp_bytes"]/common.CONFIG["DGXSSDBW"]
        data["comp_bytes_gb"] = data["comp_bytes"]/(1<<30)

        # print(label)
        # print(data[["query","comp_algo","uncomp_bytes","comp_bytes","optimal_bw","time_ms","effective_bw","ratio_alt"]].to_string())

        data = data.groupby(["query","comp_algo"],as_index=False).agg(time_ms=("time_ms","mean"),
                                                                      effective_bw=("effective_bw","mean"),
                                                                      optimal_bw=("optimal_bw","mean"),
                                                                      optimal_time=("optimal_time","mean"),
                                                                        effective_bw_std=("effective_bw","std"))
        if label == "Golap":
            ref = data.time_ms.copy()

        for comp_algo,cur_data in data.groupby("comp_algo"):
            if shape=="bar":
                x = cur_data[x_var]
            else:
                x = list(np.ones(len(cur_data.shape)))
            # print(cur_data.time_ms.values/ref.values)
            bars.add_cat(x,cur_data[metric],label=f"{label} {comp_algo}",args={"hatch":hatch[f"{label} {comp_algo}"], "color":col[f"{label} {comp_algo}"]})
            if with_optimal_ssd_scan and any(cur_data["optimal_bw"] != inf):
                bars.add_cat(x,cur_data["optimal_bw"],label="Ideal Maximum Scan BW",args=common.CONFIG["Achievable Maximum Scan BW"])
    if with_optimal_inmem_scan:
        bars.add_cat(x,common.CONFIG["DGXCPUBW"],label=f"INMEM Scan Only (Optimal)")
    if sort:
        bars.sort(prefixes=set(name for name,_ in files))
    bars.order({k:i for i,(k,v) in enumerate(hatch.items())})
    bars.do_plot(shape=shape,axes=ax, edgecolor="black", linewidth=2)

    common.axes_prepare(ax,metric)

    # axes.set_xlabel(x_var)
    # ax.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    return ax


# SSB_COMM_DATA = parse_commercial("../usecases/ssb/results/systemx_export.csv")
# for idx,row in SSB_COMM_DATA.iterrows():
#     if row.uncomp_bytes == 0:
#         SSB_COMM_DATA.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query)
# SSB_COMM_DATA["query"] = "SSB\n" + SSB_COMM_DATA["query"].str.slice(stop=6)


SSB_GOLAP_DATA = pd.read_csv("../usecases/ssb/results/query_export.csv",comment="#")
# SSB_GOLAP_DATA = SSB_GOLAP_DATA[SSB_GOLAP_DATA.chunk_bytes == (1<<24)]
SSB_GOLAP_DATA = SSB_GOLAP_DATA.loc[SSB_GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_GOLAP_DATA = SSB_GOLAP_DATA.query("comp_algo != 'UNCOMPRESSED'")
SSB_GOLAP_DATA["query"] = "SSB\n" + SSB_GOLAP_DATA["query"].str.slice(stop=6)
# SSB_GOLAP_DATA["query"] = "SSB (Synth Data)" # avg over all queries
SSB_GOLAP_DATA.comp_algo = "(Heavy Comp.)"

SSB_DUCKDB = pd.read_csv("../usecases/ssb/results/duckdb_query_export.csv",comment="#")
# SSB_DUCKDB.comp_bytes = SSB_DUCKDB.io_bytes
# SSB_DUCKDB.uncomp_bytes = SSB_DUCKDB.io_bytes
# duckdb best compression is snappy
SSB_DUCKDB = SSB_DUCKDB.query("comp_algo != 'ZSTD' & comp_algo != 'GZIP' & comp_algo != 'INMEM'")
# use known uncompressed sizes from golap:
for idx,row in SSB_DUCKDB.iterrows():
    if row.uncomp_bytes == 0:
        SSB_DUCKDB.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query)
SSB_DUCKDB = SSB_DUCKDB.loc[SSB_DUCKDB.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_DUCKDB["query"] = "SSB\n" + SSB_DUCKDB["query"].str.slice(stop=6)
SSB_DUCKDB.loc[SSB_DUCKDB.comp_algo == "SNAPPY", "comp_algo"] = "(Heavy Comp.)"
SSB_DUCKDB.loc[SSB_DUCKDB.comp_algo == "UNCOMPRESSED", "comp_algo"] = "(Uncompressed)"
# SSB_DUCKDB["query"] = "SSB (Synth Data)" # avg over all queries

SSB_DUCKDB_PRUNED = pd.read_csv("../usecases/ssb/results/duckdb_query_pruning_export.csv",comment="#")
SSB_DUCKDB_PRUNED = SSB_DUCKDB_PRUNED.query('comp_algo == "SNAPPY"')
SSB_DUCKDB_PRUNED = SSB_DUCKDB_PRUNED.loc[SSB_DUCKDB_PRUNED.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_DUCKDB_PRUNED.comp_algo = "(Heavy Comp.) +Pruning"
SSB_DUCKDB_PRUNED["query"] = "SSB\n" + SSB_DUCKDB_PRUNED["query"].str.slice(stop=6)

SSB_INMEM = pd.read_csv("../usecases/ssb/results/inmem_query_export.csv",comment="#")
SSB_INMEM = SSB_INMEM.loc[SSB_INMEM.groupby(["query"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_INMEM["comp_algo"] = "INMEM"
SSB_INMEM["query"] = "SSB\n" + SSB_INMEM["query"].str.slice(stop=6)
# SSB_INMEM["query"] = "SSB (Synth Data)" # avg over all queries

SSB_MORDRED = pd.read_csv("./mordred_all_queries_export.csv",comment="#")
SSB_MORDRED = SSB_MORDRED.loc[SSB_MORDRED.groupby(["query"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB_MORDRED["uncomp_bytes"] = 0
SSB_MORDRED["comp_bytes"] = 0
SSB_MORDRED["comp_algo"] = "(GPU/CPU In-Memory)"
SSB_MORDRED["query"] = "SSB\n" + SSB_MORDRED["query"].str.slice(stop=6)


# TAXI DATASET
# TAXI_COMM_DATA = parse_commercial("../usecases/taxi/results/systemx_export.csv","TAXI")
# for idx,row in TAXI_COMM_DATA.iterrows():
#     if row.uncomp_bytes == 0:
#         TAXI_COMM_DATA.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query)
# TAXI_COMM_DATA["query"] = "TAXI\n" + TAXI_COMM_DATA["query"].str.slice(stop=6)


TAXI_GOLAP_DATA = pd.read_csv("../usecases/taxi/results/query_export.csv",comment="#")

# TAXI_GOLAP_DATA = TAXI_GOLAP_DATA[TAXI_GOLAP_DATA.chunk_bytes == (1<<24)]
TAXI_GOLAP_DATA = TAXI_GOLAP_DATA.query("comp_algo != 'UNCOMPRESSED'")
TAXI_GOLAP_DATA = TAXI_GOLAP_DATA.loc[TAXI_GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
TAXI_GOLAP_DATA["query"] = "TAXI\n" + TAXI_GOLAP_DATA["query"].str.slice(stop=6)
# TAXI_GOLAP_DATA["query"] = "Taxi (Real World)"
TAXI_GOLAP_DATA.comp_algo = "(Heavy Comp.)"


# TAXI_DUCKDB = pd.read_csv("../usecases/taxi/results/duckdb_query_export.csv",comment="#")
TAXI_DUCKDB = pd.read_csv("../usecases/taxi/results/duckdb_query_export_2.csv",comment="#")
# TAXI_DUCKDB.comp_bytes = TAXI_DUCKDB.io_bytes
# TAXI_DUCKDB.uncomp_bytes = TAXI_DUCKDB.io_bytes
# duckdb best compression is snappy
TAXI_DUCKDB = TAXI_DUCKDB.query("comp_algo != 'ZSTD' & comp_algo != 'GZIP' & comp_algo != 'INMEM'")
# use known uncompressed sizes from golap:
for idx,row in TAXI_DUCKDB.iterrows():
    if row.uncomp_bytes == 0:
        TAXI_DUCKDB.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query,"TAXI")
TAXI_DUCKDB = TAXI_DUCKDB.loc[TAXI_DUCKDB.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
TAXI_DUCKDB["query"] = "TAXI\n" + TAXI_DUCKDB["query"].str.slice(stop=6)
TAXI_DUCKDB.loc[TAXI_DUCKDB.comp_algo == "SNAPPY", "comp_algo"] = "(Heavy Comp.)"
TAXI_DUCKDB.loc[TAXI_DUCKDB.comp_algo == "UNCOMPRESSED", "comp_algo"] = "(Uncompressed)"
# TAXI_DUCKDB["query"] = "TAXI (Synth Data)" # avg over all queries

TAXI_INMEM = pd.read_csv("../usecases/taxi/results/inmem_query_export.csv",comment="#")
TAXI_INMEM = TAXI_INMEM.loc[TAXI_INMEM.groupby(["query"],as_index=False)["time_ms"].transform("idxmin").unique()]
TAXI_INMEM["comp_algo"] = "INMEM"
TAXI_INMEM["query"] = "TAXI\n" + TAXI_INMEM["query"].str.slice(stop=6)
# TAXI_INMEM["query"] = "Taxi (Real World)"


#tpch dataset
# TPCH_COMM_DATA = parse_commercial("../usecases/tpch/results/systemx_export.csv","TPCH")
# for idx,row in TPCH_COMM_DATA.iterrows():
#     if row.uncomp_bytes == 0:
#         TPCH_COMM_DATA.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query,"TPCH")
# TPCH_COMM_DATA["query"] = "TPC-H\n" + TPCH_COMM_DATA["query"].str.slice(stop=6)

TPCH_GOLAP_DATA = pd.concat([pd.read_csv("../usecases/tpch/results/query_export.csv",comment="#"),
                        pd.read_csv("../usecases/tpch/results/query13_export.csv",comment="#").query("not query.str.startswith('query1')"),
                             pd.read_csv("../usecases/tpch/results/query5_export.csv",comment="#")])
TPCH_GOLAP_DATA = TPCH_GOLAP_DATA.loc[TPCH_GOLAP_DATA.groupby(["query","comp_algo","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
TPCH_GOLAP_DATA = TPCH_GOLAP_DATA.query("comp_algo != 'UNCOMPRESSED'")
TPCH_GOLAP_DATA["query"] = "TPC-H\n" + TPCH_GOLAP_DATA["query"].str.slice(stop=6)
# TPCH_GOLAP_DATA["query"] = "TPCH (Synth Data)" # avg over all queries
TPCH_GOLAP_DATA.comp_algo = "(Heavy Comp.)"


TPCH_DUCKDB = pd.read_csv("../usecases/tpch/results/duckdb_query_export.csv",comment="#")
# TPCH_DUCKDB.comp_bytes = TPCH_DUCKDB.io_bytes
# TPCH_DUCKDB.uncomp_bytes = TPCH_DUCKDB.io_bytes
# duckdb best compression is snappy
TPCH_DUCKDB = TPCH_DUCKDB.query("comp_algo != 'ZSTD' & comp_algo != 'GZIP' & comp_algo != 'INMEM'")
# use known uncompressed sizes from golap:
for idx,row in TPCH_DUCKDB.iterrows():
    if row.uncomp_bytes == 0:
        TPCH_DUCKDB.loc[idx,"uncomp_bytes"] = common.uncomp_bytes(row.scale_factor,row.query,"TPCH")
TPCH_DUCKDB = TPCH_DUCKDB.loc[TPCH_DUCKDB.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
TPCH_DUCKDB["query"] = "TPC-H\n" + TPCH_DUCKDB["query"].str.slice(stop=6)
TPCH_DUCKDB.loc[TPCH_DUCKDB.comp_algo == "SNAPPY", "comp_algo"] = "(Heavy Comp.)"
TPCH_DUCKDB.loc[TPCH_DUCKDB.comp_algo == "UNCOMPRESSED", "comp_algo"] = "(Uncompressed)"

files = [
             ["Golap", pd.concat([SSB_GOLAP_DATA])],
             ["DuckDB", pd.concat([SSB_DUCKDB,])],
             # ["Handcoded", pd.concat([SSB_INMEM,TAXI_INMEM])],
             # ["System X", pd.concat([SSB_COMM_DATA])],
             ["DuckDB", SSB_DUCKDB_PRUNED],
             ["Mordred", SSB_MORDRED],
            ]
files2 = [
             ["Golap", pd.concat([TAXI_GOLAP_DATA,TPCH_GOLAP_DATA])],
             ["DuckDB", pd.concat([TAXI_DUCKDB,TPCH_DUCKDB])],
             # ["Handcoded", pd.concat([TAXI_INMEM])],
             # ["System X", pd.concat([TAXI_COMM_DATA,TPCH_COMM_DATA])],
            ]

# axes = plot(files,"Select","query","comp_bytes_gb",sort=False,shape="bar")
# axes = plot(files,"Select","query","ratio",sort=False,shape="bar")
# axes = plot(files,"Select","query","effective_bw",sort=False,shape="bar")
axes1 = plot(files,axes1,"Select","query","time_ms",sort=False,shape="bar")
axes2 = plot(files2,axes2,"Select","query","time_ms",sort=False,shape="bar")
# axes = plot(files,"Select","query","actual_bw",sort=False,shape="bar")
# axes = plot(files,"Select","query","speedup",sort=False,shape="bar")

axes1.set_xlabel("All Systems",color="#025F98")
axes2.set_xlabel("W/o Mordred and DuckDB+ pruning",color="#025F98")


axes1.set_yticks([0,2000,4000,6000,8000,10000,12000])
axes1.set_ylim([0,12500])

axes1.legend(ncol=2,bbox_to_anchor=(0, 1.01, 2.06, 0), loc='lower left', mode="expand", borderaxespad=0.)
axes2.set_ylabel("")
plt.tight_layout()
fig.subplots_adjust(wspace = 0.05)
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/systems.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()
