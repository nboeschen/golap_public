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
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', None)


SSB = pd.concat([pd.read_csv("../results/benchmark_2024-08-09T12.07.09.csv",comment="#"),
                pd.read_csv("../results/benchmark_2024-08-06T12.45.59.csv",comment="#")])
SSB = SSB.query("dataflow == 'SSD2GPU'")
# SSB = SSB.query("comp_algo == 'UNCOMPRESSED'")
SSB = SSB.query("comp_algo == 'Gdeflate'")
# SSB = SSB.query("comp_algo == 'Cascaded'")
SSB = SSB.loc[SSB.groupby(["query","comp_algo","chunk_bytes","dataflow"],as_index=False)["time_ms"].transform("idxmin").unique()]
SSB = SSB.sort_values(by=["query","chunk_bytes"])


for label,data in [("SSB", SSB)]:
    data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
    data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
    data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]
    data["optimality"] = data["effective_bw"] / ( data["ratio_alt"] * 2.74)
    data["max_bw"] = ( data["ratio_alt"] * 2.74)
    print(data[["query","comp_algo","chunk_bytes","uncomp_bytes","comp_bytes","time_ms","effective_bw","actual_bw","optimality"]])

    for (query,comp_algo),cur_data in data.groupby(["query","comp_algo"]):

        # lines = plt.plot(cur_data["chunk_bytes"], cur_data["effective_bw"], "o-", markersize=13, markeredgecolor="#000000", alpha=0.9, label=f"{query} {comp_algo}")
        lines = plt.plot(cur_data["chunk_bytes"], cur_data["actual_bw"], "o-", markersize=13, markeredgecolor="#000000", alpha=0.9, label=f"{query} {comp_algo}")
        color = lines[0]._color
        plt.plot(cur_data["chunk_bytes"], cur_data["max_bw"], "^--", markersize=13, c=color, markeredgecolor="#000000", alpha=0.9, label=f"_")
        # plt.plot(cur_data["chunk_bytes"], cur_data["optimality"], "o-", markersize=13, markeredgecolor="#000000", alpha=0.9, label=f"{query} {comp_algo}")


axes.set_xscale('log')
xticks = np.power(2,np.linspace(18,26,9))
axes.set_xticks(xticks,labels=[f"{common.hrsize(x)}" for x in xticks])
axes.set_xticks([],minor=True)

# axes.set_ylim([0,5])
# axes.set_yscale("log")
# axes.set_yticks([10,20,50,100,200,300])
# axes.yaxis.set_major_formatter(ScalarFormatter())
# axes.yaxis.set_minor_formatter(NullFormatter())

axes.set_xlabel("Chunk Bytes [MB]")
axes.set_ylabel("Effective Bandwidth [GiB/s]")
axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)

plt.grid()
plt.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "print":
    plt.savefig(f"pdf/datasets_comp_ratio_vs_decompression_speed.pdf",metadata=common.EMPTY_PDF_META)
else:
    plt.show()

