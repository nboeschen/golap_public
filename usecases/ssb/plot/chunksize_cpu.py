import sys
sys.path.append('../../../plot/')
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from seaborn import color_palette

import common


files = [
            ("CPU","../results/select_cpu_export.csv"),
    ]

for title, path in files:
    for dataflow in [
                        # "SSD2GPU",
                        # "SSD2GPU2CPU",
                        "SSD2CPU",
                    ]:
        plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
        fig = plt.figure(figsize=(20,10), dpi=100)
        axes = fig.add_subplot(1, 1, 1)
        axes.set_prop_cycle('color', color_palette("colorblind"))

        data = pd.read_csv(path,comment="#")


        metric = "effective_bw"
        # metric = "ratio"
        # metric = "time_ms"

        groupby_cols = ["query","dataflow","workers","chunk_bytes","uncomp_bytes","comp_algo"]

        data = data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"), time_ms_std=("time_ms","std"), comp_bytes=("comp_bytes","mean"))

        data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
        data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
        data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]

        data = data[data["query"] == "select_commitdate"]
        data = data[data["dataflow"] == dataflow]

        # data["inefficient"] = data["uncomp_bytes"] < (data.workers * data.chunk_bytes)
        # data.loc[data.inefficient,metric] = "NaN"

        for comp_algo in data.comp_algo.unique():
            cur_data = data[data.comp_algo == comp_algo]

            axes.plot(cur_data.chunk_bytes, cur_data[metric], "o-", lw=2, label=f"{comp_algo}")

        axes.set_xscale('log')
        xticks = np.power(2,np.linspace(20,29,10))
        axes.set_xticks(xticks,labels=[f"{x/(1024*1024):.0f}" for x in xticks])
        axes.set_xlabel("Chunksize [MB]")

        if metric == "effective_bw":
            axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
            axes.axhline(24,lw=2,ls="--",c="#1AA1F1",zorder=0)
            if dataflow == "SSD2GPU2CPU" or dataflow == "SSD2CPU" or title == "UNCOMPRESSED":
                axes.set_ylim([0,30])
            axes.set_ylabel("Bandwidth [GiB/s]")
        elif metric == "ratio":
            axes.set_ylim([0,1.2])
            axes.set_ylabel("Compression [ratio]")

        fig.tight_layout()
        plt.grid()
        plt.legend()

        if len(sys.argv) > 1 and sys.argv[1] == "print":
            plt.savefig(f"pdf/{title}_column_scan_{dataflow}.pdf",metadata=common.EMPTY_PDF_META)
        else:
            plt.show()



