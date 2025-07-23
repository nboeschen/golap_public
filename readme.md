# Golap Public Repo
Code of the paper [GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP](https://dl.acm.org/doi/10.1145/3698812)

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

## Comparison to Dask cudf
```bash
conda create -n rapids-env -c rapidsai -c nvidia -c conda-forge cudf=22.10 dask-cudf=22.10 python=3.8 cudatoolkit=11.4
conda activate rapids-env
../ssb/scripts/run_dask.sh 0
```

