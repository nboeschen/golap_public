import sys
sys.path.append('../../../plot/')
import pandas as pd
from seaborn import color_palette
from behindbars import BehindBars
import common
from matplotlib import pyplot as plt

GOLAP_FULL_DATA = pd.read_csv("../results/query_export.csv",comment="#")
GOLAP_FULL_DATA = GOLAP_FULL_DATA.loc[GOLAP_FULL_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
GOLAP_FULL_DATA.loc[GOLAP_FULL_DATA.comp_algo != "UNCOMPRESSED","comp_algo"] = "✓ Compression"
GOLAP_FULL_DATA.loc[GOLAP_FULL_DATA.comp_algo == "UNCOMPRESSED","comp_algo"] = "✘ Compression"
GOLAP_FULL_DATA = GOLAP_FULL_DATA.loc[GOLAP_FULL_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
GOLAP_FULL_DATA.set_index(["query","comp_algo"],inplace=True)

GOLAP_DATA = pd.read_csv("../results/query_pre_aggr_export.csv",comment="#")
# GOLAP_DATA["query"] = GOLAP_DATA["query"].str.slice(stop=6)
# GOLAP_DATA.comp_algo = GOLAP_DATA["query"].str.slice(stop=6)
GOLAP_DATA = GOLAP_DATA.loc[GOLAP_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
GOLAP_DATA.loc[GOLAP_DATA.comp_algo != "UNCOMPRESSED","comp_algo"] = "✓ Compression"
GOLAP_DATA.loc[GOLAP_DATA.comp_algo == "UNCOMPRESSED","comp_algo"] = "✘ Compression"
GOLAP_DATA = GOLAP_DATA.loc[GOLAP_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]

for idx,row in GOLAP_DATA.iterrows():
    GOLAP_DATA.at[idx,"speedup"] = GOLAP_FULL_DATA.loc[row["query"].split("_")[0],row.comp_algo].time_ms / row.time_ms


DUCKDB_FULL_DATA = pd.read_csv("../results/duckdb_query_export.csv",comment="#")
DUCKDB_FULL_DATA = DUCKDB_FULL_DATA.loc[DUCKDB_FULL_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
DUCKDB_FULL_DATA.loc[DUCKDB_FULL_DATA.comp_algo != "UNCOMPRESSED","comp_algo"] = "✓ Compression"
DUCKDB_FULL_DATA.loc[DUCKDB_FULL_DATA.comp_algo == "UNCOMPRESSED","comp_algo"] = "✘ Compression"
DUCKDB_FULL_DATA = DUCKDB_FULL_DATA.loc[DUCKDB_FULL_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
DUCKDB_FULL_DATA.set_index(["query","comp_algo"],inplace=True)

DUCKDB_DATA = pd.read_csv("../results/duckdb_pre_aggr_export.csv",comment="#")
# DUCKDB_DATA = DUCKDB_DATA.query("comp_algo != 'ZSTD' & comp_algo != 'GZIP'")
DUCKDB_DATA = DUCKDB_DATA.loc[DUCKDB_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
DUCKDB_DATA.loc[DUCKDB_DATA.comp_algo != "UNCOMPRESSED","comp_algo"] = "✓ Compression"
DUCKDB_DATA.loc[DUCKDB_DATA.comp_algo == "UNCOMPRESSED","comp_algo"] = "✘ Compression"
DUCKDB_DATA = DUCKDB_DATA.loc[DUCKDB_DATA.groupby(["query","comp_algo"],as_index=False)["time_ms"].transform("idxmin").unique()]
# DUCKDB_DATA.comp_bytes = DUCKDB_DATA.io_bytes
# DUCKDB_DATA["query"] = DUCKDB_DATA["query"].str.slice(stop=6)
for idx,row in DUCKDB_DATA.iterrows():
    DUCKDB_DATA.at[idx,"speedup"] = DUCKDB_FULL_DATA.loc[row["query"].split("_")[0],row.comp_algo].time_ms / row.time_ms

plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
fig = plt.figure(figsize=(20,10), dpi=100)
axes = fig.add_subplot(1, 1, 1)
axes.set_prop_cycle('color', color_palette("colorblind"))

bars = BehindBars(0.7)
time_mss = []
for label,data in [
             ["Golap", GOLAP_DATA],
             ["DuckDB", DUCKDB_DATA],
            ]:
    for comp_algo,cur_data in data.groupby("comp_algo"):
        time_mss += list(cur_data["time_ms"])
        bars.add_cat(cur_data["query"],cur_data["speedup"],label=f"{label} {comp_algo}")

xss,yss = bars.do_plot(shape="bar",axes=axes, edgecolor="black", linewidth=2)
i = 0
for xs,ys in zip(xss,yss):
    for x,y in zip(xs,ys):
        axes.text(x-0.08,y+0.1,f"{time_mss[i]:.1f} ms",fontsize=22)
        i += 1
axes.grid(axis="y")
axes.set_ylabel("Speedup over Non-Pre-Aggr. [Ratio]")
axes.set_ylim([0,11])
axes.set_axisbelow(True)
axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)

plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/pre_aggr_bw.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

