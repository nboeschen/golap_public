# Golap Public Repo
Code of the paper [GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP](https://dl.acm.org/doi/10.1145/3698812).
If you have any questions/comments, please contact nils.boeschen@cs.tu-darmstadt.de

# Install packages:
```bash
./scripts/init.sh
```
Packages will be installed in "third_party" folder

# Execute
E.g. for an application in the Star Schema Benchmark context. 
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j ssb_disk_db ssb
../bin/ssb_disk_db --scale_factor=10 --path="/PATH/diskdb10.dat" --op="write" --format=binary
../ssb/scripts/run_ssb.sh 0
```

## Create parquet files, and comparison to DuckDB (installed in third_party)
```bash
../ssb/scripts/duckdb_ssb/init_duckdb.sh
make -j duckdb_ssb
../ssb/scripts/run_duckdb.sh 0
```


