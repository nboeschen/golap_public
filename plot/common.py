from datetime import datetime
from seaborn import color_palette
from collections import defaultdict

_palette = color_palette("muted")

CONFIG = defaultdict(dict)
# CONFIG["Golap BEST_BW_COMP"] = {"color":_palette[0]}
# CONFIG["Golap UNCOMPRESSED"] = {"color":_palette[1],"hatch":"//"}
# CONFIG["GPU BEST_BW_COMP"] = {"color":_palette[0]}
# CONFIG["GPU UNCOMPRESSED"] = {"color":_palette[1],"hatch":"//"}

# CONFIG["DuckDB UNCOMPRESSED"] = {"color":_palette[2],"hatch":"//"}
# CONFIG["DuckDB SNAPPY"] = {"color":_palette[3]}
# CONFIG["DuckDB GZIP"] = {"color":_palette[4]}
# CONFIG["DuckDB ZSTD"] = {"color":_palette[5]}

# CONFIG["CPU UNCOMPRESSED"] = {"color":_palette[2],"hatch":"//"}
# CONFIG["CPU SNAPPY"] = {"color":_palette[3]}
# CONFIG["CPU GZIP"] = {"color":_palette[4]}
# CONFIG["CPU ZSTD"] = {"color":_palette[5]}

# CONFIG["Dask UNCOMPRESSED"] = {"color":_palette[6],"hatch":"//"}
# CONFIG["Handcoded INMEM -"] = {"color":_palette[8],"hatch":"--"}
CONFIG["Achievable Maximum Scan BW"] = {"hatch":"/"}

# fixed measurements
# CONFIG["AWSSSDBW"] = 3.3
# CONFIG["AWSGPUCPUBW"] = 6.6
CONFIG["10Gb"] = 1.25
CONFIG["50Gb"] = 6.7
CONFIG["100Gb"] = 12.5

CONFIG["GPUNODE_ONDEMAND_PRICE_H"] = 2.0144
CONFIG["CPUNODE_ONDEMAND_PRICE_H"] = 4.9920
# CONFIG["CPUNODE_ONDEMAND_PRICE_H_ALT"] = 0.624
CONFIG["CPUNODE_ONDEMAND_PRICE_H_ALT"] = 1.2480
CONFIG["CPUNODE_SSD_BW_GIBS"] = 16
# CONFIG["CPUNODE_SSD_BW_GIBS_ALT"] = 1.573563*1.3
CONFIG["CPUNODE_SSD_BW_GIBS_ALT"] = 3.147125*1.3
CONFIG["GPUNODE_SSD_BW_GIBS"] = 2.5

CONFIG["DGXSSDBW"] = 19.5
CONFIG["DGXSSDBW_ALT"] = 14.5
CONFIG["DGXGPUCPUBW"] = 23.7
CONFIG["DGXCPUBW"] = 200

EMPTY_PDF_META = {"Creator": "", "Producer": "", "Title": "", "CreationDate": datetime.fromisoformat('2024-01-01T00:00:00')}


def parse_hrsize(s,factors={"KiB": (1<<10), "MiB": (1<<20), "GiB": (1<<30), "TiB":(1<<40), "PiB": (1<<50)}):
    """
        Roughly from:
        https://github.com/borgbackup/borg/blob/6a5feaffaccd3e2835b86cc42749095dc7312e11/src/borg/helpers/parseformat.py
    """
    if s[-1].isdigit():
        return int(s)
    else:
        factor = factors[s[-3:]]
        return int(float(s[:-3]) * factor)


def hrsize(num, power=1024, sep="", precision=0, sign=False):
    """
        Human readable size
        https://github.com/borgbackup/borg/blob/6a5feaffaccd3e2835b86cc42749095dc7312e11/src/borg/helpers/parseformat.py
    """
    if power == 1024:
        suffix = "B"
        units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"]
    else:
        suffix = ""
        units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    sign = "+" if sign and num > 0 else ""
    fmt = "{0:{1}.{2}f}{3}{4}{5}"
    prec = 0
    for unit in units[:-1]:
        if abs(round(num, precision)) < power:
            break
        num /= float(power)
        prec = precision
    else:
        unit = units[-1]
    return fmt.format(num, sign, prec, sep, unit, suffix)

def tuple_size(q, dataset="SSB"):
    """ Not used """
    if dataset == "SSB":
        if q == "query1.1":
            return 16
        elif q == "query1.2":
            return 18
        elif q == "query1.3":
            return 18
        elif q == "query2.1":
            return 0
        elif q == "query2.2":
            return 0
        elif q == "query2.3":
            return 0
        elif q == "query3.1":
            return 0
        elif q == "query3.2":
            return 0
        elif q == "query3.3":
            return 0
        elif q == "query3.4":
            return 0
        elif q == "query4.1":
            return 0
        elif q == "query4.2":
            return 0
        elif q == "query4.3":
            return 0
        else:
            print("Unknown query for tuple_size lookup: ", q)
            return 0
    else:
        print("Unknown dataset: ", dataset)
        return 0

def uncomp_bytes(scale_factor, q, dataset="SSB"):
    if dataset == "SSB":
        if q.startswith("query1"):
            return int((16800214942 / 200) * scale_factor)
        elif q == "query2.1" or q == "query2.2" or q == "query2.3":
            return int((33600429884 / 200) * scale_factor)
        elif q == "query2.1_pre_aggr" or q == "query2.2_pre_aggr":
            return int((1874777410 / 100) * scale_factor)
        elif q.startswith("query3"):
            return int((33600429884 / 200) * scale_factor)
        elif q.startswith("query4"):
            return int((48000614120 / 200) * scale_factor)
        elif q == "select_linenum" or q == "select_linenumber" or q == "select_shippriority" or q == "select_quantity" or q == "select_discount" or q == "select_tax":
            return int((1200015353 / 200) * scale_factor)
        elif q == "select_key" or q == "select_custkey" or q == "select_partkey" or q == "select_suppkey" or q == "select_orderdate" or q == "select_commitdate":
            return int((9600122824 / 200) * scale_factor)
        elif q == "select_orderpriority" or q == "select_shipmode":
            return int((19200245648 / 200) * scale_factor)
        elif q == "select_extendedprice" or q == "select_ordtotalprice" or q == "select_revenue" or q == "select_supplycost":
            return int((4800061412 / 200) * scale_factor)
        else:
            print("Unknown query for uncomp_bytes lookup: ", q)
            return 0
    elif dataset == "TAXI":
        if q.startswith("query1"):
            return int((9250259104/60)*scale_factor)
        elif q.startswith("query2"):
            return int((18500518208/60)*scale_factor)
        else:
            print("Unknown query for uncomp_bytes lookup: ", q)
            return 0
    elif dataset == "TPCH":
        if q == "query1":
            return int((42000645190/200)*scale_factor)
        elif q == "query3":
            return int((38400589888/200)*scale_factor)
        elif q == "query5":
            return int((24000000000/100)*scale_factor)
    else:
        print("Unknown dataset: ", dataset)


def axes_prepare(axes,metric):
    if metric == "time_ms":
        axes.set_ylabel("run time [ms]")
    elif metric == "effective_bw":
        # axes.set_yticks(np.linspace(0,50,11))
        # axes.axhline(CONFIG["DGXSSDBW"],lw=2,ls="--",c="#FF4E5B",zorder=0)
        # axes.axhline(CONFIG["DGXGPUCPUBW"],lw=2,ls="--",c="#1AA1F1",zorder=0)
        axes.set_ylabel("Effective BW [GiB/s]")
    elif metric == "actual_bw":
        # axes.axhline(CONFIG["DGXSSDBW"],lw=5,ls="--",c="#FF4E5B",zorder=0)
        axes.set_ylabel("Actual BW (comp/read) [GiB/s]")
    elif metric == "comp_bytes_gb":
        axes.set_ylabel("Compressed Data Size [GiB]")
    elif metric == "ratio":
        axes.set_ylim([0,1.2])
        axes.set_ylabel("Compression [ratio]")
    elif metric == "speedup":
        axes.axhline(1,ls="--",c="r",zorder=0)
        axes.set_ylabel("Speed Up [ratio]")
    else:
        axes.set_ylabel(metric)