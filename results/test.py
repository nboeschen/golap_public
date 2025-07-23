import pandas as pd


data = pd.read_csv("./benchmark_export.csv",comment="#")

data["effective_bw"] = (1000.0 / (1<<30)) * data["uncomp_bytes"]/data["time_ms"]
data["actual_bw"] = (1000.0 / (1<<30)) * data["comp_bytes"]/data["time_ms"]
data["ratio"] = data["comp_bytes"]/data["uncomp_bytes"]
data["ratio_alt"] = data["uncomp_bytes"]/data["comp_bytes"]

print(data[["query","comp_algo","actual_bw","ratio_alt","effective_bw"]].to_string())