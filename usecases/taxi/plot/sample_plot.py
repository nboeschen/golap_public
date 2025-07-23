import sys
sys.path.append('../../../plot/')
import common
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from seaborn import color_palette

from behindbars import BehindBars


for OPTIMIZE_FOR in ["BW", "RATIO"]:
    for PLOTVARIANT in ["RELATIVE","ABSOLUTE"]:
        GPU_SELECT_DATA = pd.read_csv("../../taxi/results/select_export.csv",comment="#") # all comp_algos
        # GPU_SELECT_DATA = GPU_SELECT_DATA[GPU_SELECT_DATA.chunk_bytes == (1<<26)]
        GPU_SELECT_DATA = GPU_SELECT_DATA[GPU_SELECT_DATA.dataflow == "SSD2GPU"]
        GPU_SELECT_DATA["query"] = GPU_SELECT_DATA["query"].str.slice(start=7)

        SAMPLE_DATA = pd.read_csv("../results/sample_export.csv",comment="#") # select * all columns
        SAMPLE_DATA.rename(columns = {"col_name":"query","decomp_time":"time_ms"}, inplace = True)
        SAMPLE_DATA = SAMPLE_DATA[SAMPLE_DATA["query"].str.endswith(OPTIMIZE_FOR)]
        # SAMPLE_DATA["query"] = SAMPLE_DATA["query"].str.slice(start=3,stop=-8-len(OPTIMIZE_FOR))
        SAMPLE_DATA["query"] = SAMPLE_DATA["query"].str.slice(stop=-8-len(OPTIMIZE_FOR))

        x_val = "query"
        # x_val = "chunk_bytes"
        if OPTIMIZE_FOR == "BW":
            metric = "effective_bw"
        elif OPTIMIZE_FOR == "RATIO":
            metric = "ratio"
        plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
        fig = plt.figure(figsize=(20,10), dpi=100)
        axes = fig.add_subplot(1, 1, 1)
        axes.set_prop_cycle('color', color_palette("colorblind"))
        bars = BehindBars(0.7)

        ALL_SELECT_NUMS = {}
        TRUE_BEST = {}

        for label,cur_data in [["Actual", GPU_SELECT_DATA]]:
            groupby_cols = [x_val,"comp_algo","chunk_bytes","uncomp_bytes"]
            if metric in cur_data.columns:
                groupby_cols.append(metric)
            cur_data = cur_data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"),time_ms_std=("time_ms","std"),comp_bytes=("comp_bytes","mean"))
            cur_data["effective_bw"] = (1000.0 / (1<<30)) * cur_data["uncomp_bytes"]/cur_data["time_ms"]
            cur_data["actual_bw"] = (1000.0 / (1<<30)) * cur_data["comp_bytes"]/cur_data["time_ms"]
            cur_data["ratio"] = cur_data["comp_bytes"]/cur_data["uncomp_bytes"]
            cur_data["comp_bytes_gb"] = cur_data["comp_bytes"]/(1<<30)

            for rowidx,row in cur_data.iterrows():
                ALL_SELECT_NUMS[(row["query"],row.comp_algo,row.chunk_bytes)] = row

            xs = []
            ys = []
            for query in cur_data["query"].unique():
                query_data = cur_data[cur_data["query"] == query]
                if OPTIMIZE_FOR == "BW":
                    the_row = query_data.loc[query_data[metric].idxmax()]
                elif OPTIMIZE_FOR == "RATIO":
                    the_row = query_data.loc[query_data[metric].idxmin()]
                xs.append(query)
                ys.append(the_row[metric])
                TRUE_BEST[query] = the_row

            if PLOTVARIANT == "ABSOLUTE":
                bars.add_cat(xs,ys,label=f"{label}")


        # print(ALL_SELECT_NUMS.keys())
        # for row in TRUE_BEST.values():
        #     print(f'{{"lo_{row.query}", "{row.comp_algo}"}},')


        for label,cur_data in [["SAMPLE 1%", SAMPLE_DATA[SAMPLE_DATA.sample_ratio == 0.01]],
                                 ["SAMPLE 5%", SAMPLE_DATA[SAMPLE_DATA.sample_ratio == 0.05]],
                                 ["SAMPLE 10%", SAMPLE_DATA[SAMPLE_DATA.sample_ratio == 0.1]],
                                 ["SAMPLE 20%", SAMPLE_DATA[SAMPLE_DATA.sample_ratio == 0.2]],]:
            groupby_cols = [x_val,"comp_algo","chunk_bytes","uncomp_bytes"]
            if metric in cur_data.columns:
                groupby_cols.append(metric)
            cur_data = cur_data.groupby(groupby_cols,as_index=False).agg(time_ms=("time_ms","mean"),time_ms_std=("time_ms","std"),comp_bytes=("comp_bytes","mean"))
            cur_data["effective_bw"] = (1000.0 / (1<<30)) * cur_data["uncomp_bytes"]/cur_data["time_ms"]
            cur_data["actual_bw"] = (1000.0 / (1<<30)) * cur_data["comp_bytes"]/cur_data["time_ms"]
            cur_data["ratio"] = cur_data["comp_bytes"]/cur_data["uncomp_bytes"]
            cur_data["comp_bytes_gb"] = cur_data["comp_bytes"]/(1<<30)

            xs = []
            ys = []
            for query in cur_data["query"].unique():
                query_data = cur_data[cur_data["query"] == query]
                if OPTIMIZE_FOR == "BW":
                    the_row = query_data.loc[query_data[metric].idxmax()]
                elif OPTIMIZE_FOR == "RATIO":
                    the_row = query_data.loc[query_data[metric].idxmin()]
                xs.append(query)
                if PLOTVARIANT == "ABSOLUTE":
                    ys.append(ALL_SELECT_NUMS[(query,the_row.comp_algo,the_row.chunk_bytes)][metric])
                elif PLOTVARIANT == "RELATIVE":
                    truth = TRUE_BEST[query][metric]
                    sampled = ALL_SELECT_NUMS[(query,the_row.comp_algo,the_row.chunk_bytes)][metric]
                    ys.append(max([truth,sampled])/min([truth,sampled]))

            print(label, "AVG", np.mean(ys))
            print(label, "MAX", np.max(ys))
            bars.add_cat(xs,ys,label=f"{label}")

        bars.do_plot(shape="bar",axes=axes, edgecolor="black", linewidth=2)


        if OPTIMIZE_FOR == "BW":
            # axes.axhline(common.CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
            # axes.axhline(common.CONFIG["DGXGPUCPUBW"],lw=2,ls=":",c="#1AA1F1",zorder=0)
            # axes.set_yscale("log")
            # axes.set_ylim([0,50])
            axes.set_ylabel("Effective BW (uncomp) [GiB/s]")
        elif OPTIMIZE_FOR == "RATIO":
            axes.set_ylim([0,1.2])
            axes.set_ylabel("Compression [ratio]")

        t  = [xtick.get_text() for xtick in axes.get_xticklabels()]
        axes.set_xticklabels(t,rotation=90)
        axes.legend(ncol=2,bbox_to_anchor=(0, 1.01, 1, 0), loc='lower left', mode="expand", borderaxespad=0.)
        axes.set_axisbelow(True)

        plt.grid()
        plt.tight_layout()
        if len(sys.argv) > 1 and sys.argv[1] == "print":
            plt.savefig(f"pdf/sample_plot_{OPTIMIZE_FOR}_{PLOTVARIANT}.pdf",metadata=common.EMPTY_PDF_META)
        else:
            plt.show()

