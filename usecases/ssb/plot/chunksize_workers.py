import sys
sys.path.append('../../../plot/')
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from seaborn import color_palette

import common

DATA = pd.concat([
                 pd.read_csv("../results/select_export_6.csv",comment="#"), # UNCOMPRESSED, BEST_BW, SSD2GPU
                 pd.read_csv("../results/select_export_7.csv",comment="#"), # UNCOMPRESSED, BEST_BW, SSD2GPU
                 pd.read_csv("../results/select_export_2.csv",comment="#"), # BEST_BW, SSD2GPU2CPU
                 pd.read_csv("../results/select_export_4.csv",comment="#"), # UNCOMPRESSED, SSD2GPU2CPU
                 ])

files = [
            ("UNCOMPRESSED",DATA[DATA.comp_algo == "UNCOMPRESSED"]),
            ("COMPRESSED",DATA[DATA.comp_algo == "BEST_BW_COMP"]),
    ]

for title, data_or_path in files:
    for dataflow in [
                        # "SSD2GPU2CPU",
                        "SSD2GPU",
                        # "SSD2CPU",
                    ]:


        plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
        fig = plt.figure(figsize=(20,10), dpi=100)
        axes = fig.add_subplot(1, 1, 1)
        axes.set_prop_cycle('color', color_palette("colorblind"))

        if type(data_or_path) == str:
            data = pd.read_csv(data_or_path,comment="#")
        else:
            data = data_or_path.copy()

        metric = "effective_bw"
        # metric = "ratio"
        # metric = "time_ms"

        groupby_cols = ["query","dataflow","workers","chunk_bytes","uncomp_bytes"]
        # data = data[data["query"] == "select_key"]
        data = data[data["query"] == "select_commitdate"]
        # data = data[data["query"] == "select_orderpriority"]
        # data = data[data["query"] == "select_revenue"]

        data = data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"), time_ms_std=("time_ms","std"), comp_bytes=("comp_bytes","mean"))

        data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
        data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
        data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]
        data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]

        data = data[data["dataflow"] == dataflow]

        data["inefficient"] = data["uncomp_bytes"] < (data.workers * data.chunk_bytes)
        data.loc[data.inefficient,metric] = "NaN"


        for worker_num in data.workers.unique():
            cur_data = data[data.workers == worker_num]
            if worker_num == 1 and dataflow == "SSD2GPU" and title != "UNCOMPRESSED":
                # optimal BW doesnt change with worker num
                axes.plot(cur_data.chunk_bytes, cur_data["ratio_alt"]*common.CONFIG["DGXSSDBW"], "D:",ms=12,lw=6, c="#3EAA4E", label="_")
                # axes.plot([],[],"--",lw=6,c="#3EAA4E",label="Optimal Bandwidth")
            # if dataflow == "SSD2GPU" and title == "COMPRESSED":
            #     print(data[["query","chunk_bytes","ratio_alt"]])
            marker = {1:"o",2:"*",4:"P",8:"^",16:"H"}[worker_num]
            # axes.plot(cur_data.chunk_bytes, cur_data[metric], marker+"-", ms=16, lw=4, alpha=0, label=f"{worker_num}")
            axes.plot(cur_data.chunk_bytes, cur_data[metric], marker+"-", ms=16, lw=4, label=f"{worker_num}")

        axes.set_xscale('log')
        xticks = np.power(2,np.linspace(16,27,12))
        axes.set_xticks(xticks,labels=[common.hrsize(x,sep='\n') for x in xticks])
        axes.set_xticks([],minor=True)
        axes.set_xlabel("Chunksize")
        # axes.set_title(title+" "+dataflow)

        if metric == "effective_bw":
            axes.axhline(common.CONFIG["DGXSSDBW"],lw=6,ls="--",c="#FF4E5B",zorder=0)
            # axes.plot([],[],"--",lw=6,c="#FF4E5B",label="SSD Bandwidth")
            axes.text((1<<16)+(1<<12),13.5,"SSD BW",c="#FF4E5B")
            if dataflow == "SSD2GPU2CPU":
                axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=6,ls="--",c="#1AA1F1",zorder=0)
            if dataflow == "SSD2GPU2CPU" or dataflow == "SSD2CPU" or title == "UNCOMPRESSED":
                axes.set_ylim([0,30])
            if dataflow == "SSD2GPU" and title == "COMPRESSED":
                axes.text((1<<16)+(1<<12),55,"Optimal BW",c="#3EAA4E")
            #     axes.plot([],[],"D:",ms=12,lw=6,c="#3EAA4E",label="Optimal Bandwidth")
            axes.set_ylabel("Effective Bandwidth [GiB/s]")
        elif metric == "ratio":
            axes.set_ylim([0,1.2])
            axes.set_ylabel("Compression [ratio]")

        plt.legend(ncol=3,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.,title="In-Flight Requests / Data Pipelines")
        plt.grid()
        fig.tight_layout()

        if len(sys.argv) > 1 and sys.argv[1] == "print":
            # plt.savefig(f"pdf/{title}_column_scan_{dataflow}_empty.pdf",metadata=common.EMPTY_PDF_META)
            plt.savefig(f"pdf/{title}_column_scan_{dataflow}.pdf",metadata=common.EMPTY_PDF_META)
        else:
            plt.show()



