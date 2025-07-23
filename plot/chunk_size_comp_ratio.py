import sys
import common
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullFormatter
from seaborn import color_palette

plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
fig = plt.figure(figsize=(20,10), dpi=100)
axes = fig.add_subplot(1, 1, 1)
axes.set_prop_cycle('color', color_palette("colorblind"))



SSB = pd.read_csv("../usecases/ssb/results/select_export.csv",comment="#") # select * all columns
SSB = SSB.query("dataflow == 'SSD2GPU'")
# SSB = SSB.query("uncomp_bytes >= 200000000")
SSB = SSB.loc[SSB.groupby(["query","dataflow"],as_index=False)["comp_bytes"].transform("idxmin").unique()]
# SSB["query"] = SSB["query"].str.slice(stop=6)
# print(SSB.to_string())

TAXI = pd.read_csv("../usecases/taxi/results/select_export.csv",comment="#") # select * all columns
TAXI = TAXI.query("dataflow == 'SSD2GPU'")
# TAXI = TAXI.query("uncomp_bytes >= 200000000")
TAXI = TAXI.loc[TAXI.groupby(["query","dataflow"],as_index=False)["comp_bytes"].transform("idxmin").unique()]
# TAXI["query"] = TAXI["query"].str.slice(stop=6)
# print(TAXI.to_string())


for label,data in [("Taxi", TAXI), ("SSB", SSB)]:
    data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
    data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]

    for dataflow,cur_data in data.groupby("dataflow"):

        # sns.regplot(x=cur_data["ratio_alt"], y=cur_data["effective_bw"], ci=50, label=f"ColumnScan {dataflow}", line_kws={"ls":(0, (5, 20))});
        plt.plot(cur_data["chunk_bytes"], cur_data["ratio_alt"], "o", markersize=13, markeredgecolor="#000000", alpha=0.5, label=f"{label} Dataset Column Scan")


axes.set_xscale('log')
xticks = np.power(2,np.linspace(20,28,9))
axes.set_xticks(xticks,labels=[f"{common.hrsize(x)}" for x in xticks])
axes.set_xticks([],minor=True)

axes.set_ylim([0,11])
# axes.set_yscale("log")
# axes.set_yticks([10,20,50,100,200,300])
# axes.yaxis.set_major_formatter(ScalarFormatter())
# axes.yaxis.set_minor_formatter(NullFormatter())

axes.set_xlabel("Chunk Bytes [MB]")
axes.set_ylabel("Compression Ratio [ratio]")
axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)

plt.grid()
plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/datasets_comp_ratio_vs_decompression_speed.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

