# import dask.dataframe as dd
# from dask_sql import Context
import dask.dataframe
import dask_cudf

import sys
sys.path.insert(0, "../queries")
from queries import QUERY_STR

import argparse
import time

class Timer:
    def __enter__(self):
        self.start = time.monotonic()
        return self
    def __exit__(self, *args):
        self.end = time.monotonic()
        self.ms = (self.end - self.start)*1000

def get_proc_io(query):
    with open("/proc/self/io") as file:
        for line in file:
            key,val = line.split(": ")
            if key == query:
                return int(val)
    return -1


def query1_1(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_extendedprice","lo_discount","lo_quantity","lo_orderdate"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()

    lineorder_filtered = lineorder[(lineorder.lo_discount>=1)&(lineorder.lo_discount<=3)&(lineorder.lo_quantity < 25)]
    date_filtered = date[date.d_year == 1993]

    joined = lineorder_filtered.merge(date_filtered,left_on="lo_orderdate",right_index=True)
    prod = joined.lo_extendedprice.mul(joined.lo_discount)

    res = prod.sum()
    return res

def query1_2(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_extendedprice","lo_discount","lo_quantity","lo_orderdate"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_yearmonthnum"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()

    lineorder_filtered = lineorder[(lineorder.lo_discount>=4)&(lineorder.lo_discount<=6)&
                                  (lineorder.lo_quantity >= 26)&(lineorder.lo_quantity <= 35)]
    date_filtered = date[date.d_yearmonthnum == 199401]

    joined = lineorder_filtered.merge(date_filtered,left_on="lo_orderdate",right_index=True)
    prod = joined.lo_extendedprice.mul(joined.lo_discount)

    res = prod.sum()
    return res

def query1_3(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_extendedprice","lo_discount","lo_quantity","lo_orderdate"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year","d_weeknuminyear"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()

    lineorder_filtered = lineorder[(lineorder.lo_discount>=5)&(lineorder.lo_discount<=7)&
                                  (lineorder.lo_quantity >= 26)&(lineorder.lo_quantity <= 35)]
    date_filtered = date[(date.d_year == 1994)&(date.d_weeknuminyear == 6)]

    joined = lineorder_filtered.merge(date_filtered,left_on="lo_orderdate",right_index=True)
    prod = joined.lo_extendedprice.mul(joined.lo_discount)

    res = prod.sum()
    return res


def query2_1(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_orderdate","lo_partkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    part = reader.read_parquet(f"{PATH}/part.parquet", columns=["p_key","p_brand1","p_category"],split_row_groups=split_row_groups)
    part = part.set_index("p_key")
    part = part.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_region"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    part_filtered = part[part.p_category == "MFGR#12"]
    supplier_filtered = supplier[supplier.s_region == "AMERICA"]

    joined0 = lineorder.merge(part_filtered,left_on="lo_partkey",right_index=True)
    joined1 = joined0.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined2 = joined1.merge(date,left_on="lo_orderdate",right_index=True)

    res = joined2.groupby(["d_year","p_brand1"]).lo_revenue.sum()
    return res

def query2_2(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_orderdate","lo_partkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    part = reader.read_parquet(f"{PATH}/part.parquet", columns=["p_key","p_brand1"],split_row_groups=split_row_groups)
    part = part.set_index("p_key")
    part = part.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_region"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    part_filtered = part[(part.p_brand1 >= "MFGR#2221") & (part.p_brand1 <= "MFGR#2228")]
    supplier_filtered = supplier[supplier.s_region == "ASIA"]

    joined0 = lineorder.merge(part_filtered,left_on="lo_partkey",right_index=True)
    joined1 = joined0.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined2 = joined1.merge(date,left_on="lo_orderdate",right_index=True)

    res = joined2.groupby(["d_year","p_brand1"]).lo_revenue.sum()
    return res

def query2_3(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_orderdate","lo_partkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    part = reader.read_parquet(f"{PATH}/part.parquet", columns=["p_key","p_brand1"],split_row_groups=split_row_groups)
    part = part.set_index("p_key")
    part = part.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_region"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    part_filtered = part[part.p_brand1 == "MFGR#2239"]
    supplier_filtered = supplier[supplier.s_region == "EUROPE"]

    joined0 = lineorder.merge(part_filtered,left_on="lo_partkey",right_index=True)
    joined1 = joined0.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined2 = joined1.merge(date,left_on="lo_orderdate",right_index=True)

    res = joined2.groupby(["d_year","p_brand1"]).lo_revenue.sum()
    return res

def query3_1(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_orderdate","lo_custkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    customer = reader.read_parquet(f"{PATH}/customer.parquet", columns=["c_key","c_region","c_nation"],split_row_groups=split_row_groups)
    customer = customer.set_index("c_key")
    customer = customer.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_region","s_nation"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    customer_filtered = customer[customer.c_region == "ASIA"]
    supplier_filtered = supplier[supplier.s_region == "ASIA"]
    date_filtered = date[(date.d_year >= 1992) & (date.d_year <= 1997)]

    joined0 = lineorder.merge(customer_filtered,left_on="lo_custkey",right_index=True)
    joined1 = joined0.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined2 = joined1.merge(date_filtered,left_on="lo_orderdate",right_index=True)

    res = joined2.groupby(["c_nation","s_nation","d_year"]).lo_revenue.sum()
    return res

def query3_2(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_orderdate","lo_custkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    customer = reader.read_parquet(f"{PATH}/customer.parquet", columns=["c_key","c_city","c_nation"],split_row_groups=split_row_groups)
    customer = customer.set_index("c_key")
    customer = customer.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_city","s_nation"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    customer_filtered = customer[customer.c_nation == "UNITED STATES"]
    supplier_filtered = supplier[supplier.s_nation == "UNITED STATES"]
    date_filtered = date[(date.d_year >= 1992) & (date.d_year <= 1997)]

    joined0 = lineorder.merge(customer_filtered,left_on="lo_custkey",right_index=True)
    joined1 = joined0.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined2 = joined1.merge(date_filtered,left_on="lo_orderdate",right_index=True)

    res = joined2.groupby(["c_city","s_city","d_year"]).lo_revenue.sum()
    return res


def query3_3(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_orderdate","lo_custkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    customer = reader.read_parquet(f"{PATH}/customer.parquet", columns=["c_key","c_city"],split_row_groups=split_row_groups)
    customer = customer.set_index("c_key")
    customer = customer.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_city"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    customer_filtered = customer[(customer.c_city == "UNITED KI1") | (customer.c_city == "UNITED KI5")]
    supplier_filtered = supplier[(supplier.s_city == "UNITED KI1") | (supplier.s_city == "UNITED KI5")]
    date_filtered = date[(date.d_year >= 1992) & (date.d_year <= 1997)]

    joined0 = lineorder.merge(customer_filtered,left_on="lo_custkey",right_index=True)
    joined1 = joined0.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined2 = joined1.merge(date_filtered,left_on="lo_orderdate",right_index=True)

    res = joined2.groupby(["c_city","s_city","d_year"]).lo_revenue.sum()
    return res

def query3_4(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_orderdate","lo_custkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year","d_yearmonth"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    customer = reader.read_parquet(f"{PATH}/customer.parquet", columns=["c_key","c_city"],split_row_groups=split_row_groups)
    customer = customer.set_index("c_key")
    customer = customer.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_city"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    customer_filtered = customer[(customer.c_city == "UNITED KI1") | (customer.c_city == "UNITED KI5")]
    supplier_filtered = supplier[(supplier.s_city == "UNITED KI1") | (supplier.s_city == "UNITED KI5")]
    date_filtered = date[date.d_yearmonth == "Dec1997"]

    joined0 = lineorder.merge(customer_filtered,left_on="lo_custkey",right_index=True)
    joined1 = joined0.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined2 = joined1.merge(date_filtered,left_on="lo_orderdate",right_index=True)

    res = joined2.groupby(["c_city","s_city","d_year"]).lo_revenue.sum()
    return res

def query4_1(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_supplycost","lo_orderdate",
                                       "lo_custkey","lo_partkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year","d_yearmonth"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    customer = reader.read_parquet(f"{PATH}/customer.parquet", columns=["c_key","c_region","c_nation"],split_row_groups=split_row_groups)
    customer = customer.set_index("c_key")
    customer = customer.persist()
    part = reader.read_parquet(f"{PATH}/part.parquet", columns=["p_key","p_mfgr"],split_row_groups=split_row_groups)
    part = part.set_index("p_key")
    part = part.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_region"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    customer_filtered = customer[customer.c_region == "AMERICA"]
    supplier_filtered = supplier[supplier.s_region == "AMERICA"]
    part_filtered = part[(part.p_mfgr == "MFGR#1") | (part.p_mfgr == "MFGR#2")]

    joined0 = lineorder.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined1 = joined0.merge(customer_filtered,left_on="lo_custkey",right_index=True)
    joined2 = joined1.merge(part_filtered,left_on="lo_partkey",right_index=True)
    joined3 = joined2.merge(date,left_on="lo_orderdate",right_index=True)

    grouped = joined3.groupby(["d_year","c_nation"]).agg({"lo_revenue":"sum","lo_supplycost":"sum"})
    # res = grouped.assign(grouped.loc[:,["lo_revenue","lo_supplycost"]].sum(1))
    return grouped

def query4_2(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_supplycost","lo_orderdate",
                                       "lo_custkey","lo_partkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year","d_yearmonth"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    customer = reader.read_parquet(f"{PATH}/customer.parquet", columns=["c_key","c_region"],split_row_groups=split_row_groups)
    customer = customer.set_index("c_key")
    customer = customer.persist()
    part = reader.read_parquet(f"{PATH}/part.parquet", columns=["p_key","p_mfgr","p_category"],split_row_groups=split_row_groups)
    part = part.set_index("p_key")
    part = part.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_region","s_nation"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    customer_filtered = customer[customer.c_region == "AMERICA"]
    supplier_filtered = supplier[supplier.s_region == "AMERICA"]
    part_filtered = part[(part.p_mfgr == "MFGR#1") | (part.p_mfgr == "MFGR#2")]
    date_filtered = date[(date.d_year == 1997) | (date.d_year == 1998)]

    joined0 = lineorder.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined1 = joined0.merge(customer_filtered,left_on="lo_custkey",right_index=True)
    joined2 = joined1.merge(part_filtered,left_on="lo_partkey",right_index=True)
    joined3 = joined2.merge(date_filtered,left_on="lo_orderdate",right_index=True)

    grouped = joined3.groupby(["d_year","s_nation","p_category"]).agg({"lo_revenue":"sum","lo_supplycost":"sum"})
    # res = grouped.assign(grouped.loc[:,["lo_revenue","lo_supplycost"]].sum(1))
    return grouped

def query4_3(reader,split_row_groups):
    lineorder = reader.read_parquet(f"{PATH}/lineorder.parquet", columns=["lo_revenue","lo_supplycost","lo_orderdate",
                                       "lo_custkey","lo_partkey","lo_suppkey"],
                                  split_row_groups=split_row_groups
                                  )
    date = reader.read_parquet(f"{PATH}/date.parquet", columns=["d_key","d_year","d_yearmonth"],split_row_groups=split_row_groups)
    date = date.set_index("d_key")
    date = date.persist()
    customer = reader.read_parquet(f"{PATH}/customer.parquet", columns=["c_key","c_region"],split_row_groups=split_row_groups)
    customer = customer.set_index("c_key")
    customer = customer.persist()
    part = reader.read_parquet(f"{PATH}/part.parquet", columns=["p_key","p_brand1","p_category"],split_row_groups=split_row_groups)
    part = part.set_index("p_key")
    part = part.persist()
    supplier = reader.read_parquet(f"{PATH}/supplier.parquet", columns=["s_key","s_nation","s_city"],split_row_groups=split_row_groups)
    supplier = supplier.set_index("s_key")
    supplier = supplier.persist()

    customer_filtered = customer[customer.c_region == "AMERICA"]
    supplier_filtered = supplier[supplier.s_nation == "UNITED STATES"]
    part_filtered = part[(part.p_category == "MFGR#14")]
    date_filtered = date[(date.d_year == 1997) | (date.d_year == 1998)]

    joined0 = lineorder.merge(supplier_filtered,left_on="lo_suppkey",right_index=True)
    joined1 = joined0.merge(customer_filtered,left_on="lo_custkey",right_index=True)
    joined2 = joined1.merge(part_filtered,left_on="lo_partkey",right_index=True)
    joined3 = joined2.merge(date_filtered,left_on="lo_orderdate",right_index=True)

    grouped = joined3.groupby(["d_year","s_city","p_brand1"]).agg({"lo_revenue":"sum","lo_supplycost":"sum"})
    # res = grouped.assign(grouped.loc[:,["lo_revenue","lo_supplycost"]].sum(1))
    return grouped


QUERY_FUN = {
    "query1.1":query1_1,
    "query1.2":query1_2,
    "query1.3":query1_3,
    "query2.1":query2_1,
    "query2.2":query2_2,
    "query2.3":query2_3,
    "query3.1":query3_1,
    "query3.2":query3_2,
    "query3.3":query3_3,
    "query3.4":query3_4,
    "query4.1":query4_1,
    "query4.2":query4_2,
    "query4.3":query4_3,
}


###########################################################################
###########################################################################
###########################################################################


parser = argparse.ArgumentParser(description="Dask sql SSB.")
parser.add_argument("--scale_factor", type=int, default=10, help="Querie(s)")
parser.add_argument("--csv_header", action='store_const', const=True, default=False, help="Print header and exit.")
parser.add_argument("--repeat", type=int, default=1, help="Repeat queries x times.")
parser.add_argument("--proc", type=str, default="GPU", help="Either GPU or CPU.")
parser.add_argument("--split_row_groups", type=int, default=8, help="Split row groups.")
parser.add_argument("--query", help="Querie(s).")
parser.add_argument("--encoding", help="Encoding(s).")
args = parser.parse_args()
reader = dask_cudf if args.proc == "GPU" else dask.dataframe

if args.csv_header:
    print("query,scale_factor,comp_algo,comp_bytes,uncomp_bytes,io_bytes,time_ms")
    exit(0)

# create a context to register tables
# context = Context()

for _ in range(args.repeat):
    for encoding in args.encoding.split(","):
        for query_name in args.query.split(","):
            PATH = f"/raid/gds/ssb/parquet{args.scale_factor}/{encoding}"
            # PATH = f"/raid/gds/ssb/parquet{args.scale_factor}_hugerowgroups/{encoding}"

            # context.create_table("lineorder", f"{PATH}/lineorder.parquet", persist=False, gpu=True, split_row_groups=args.split_row_groups)
            # context.create_table("customer", f"{PATH}/customer.parquet", persist=True, gpu=True, split_row_groups=args.split_row_groups)
            # context.create_table("supplier", f"{PATH}/supplier.parquet", persist=True, gpu=True, split_row_groups=args.split_row_groups)
            # context.create_table("part", f"{PATH}/part.parquet", persist=True, gpu=True, split_row_groups=args.split_row_groups)
            # context.create_table("date_table", f"{PATH}/date.parquet", persist=True, gpu=True, split_row_groups=args.split_row_groups)

            # print(PATH)
            # print(QUERY_STR[query_name])
            # print(context.explain(QUERY_STR[query_name]))
            # context.visualize(QUERY_STR[query_name], filename=f"{query_name}")
            # plan = context.sql(QUERY_STR[query_name])
            # import os
            # print(os.getpid())
            # input("Waiting ...")
            start = get_proc_io("rchar")


            plan = QUERY_FUN[query_name](reader,args.split_row_groups)

            # actually compute the query...
            with Timer() as timer:
                res = plan.compute()


            if hasattr(res,"__len__"):
                print("#[INFO ] Results#: ",len(res))
            else:
                print("#[INFO ] Result  : ",res)

            print(f"{query_name},{args.scale_factor},{encoding.upper()},0,0,0,{timer.ms:.3f}")

            print("# read total=",get_proc_io("rchar")-start)
            # input("Waiting ...")
            # context.drop_table("lineorder")
            # context.drop_table("customer")
            # context.drop_table("supplier")
            # context.drop_table("part")
            # context.drop_table("date_table")


# to execute: conda activate rapids-env