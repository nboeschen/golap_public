# Golap Public Repo
Code of the paper [GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP](https://dl.acm.org/doi/10.1145/3698812).
If you have any questions/comments, please contact nils.boeschen@cs.tu-darmstadt.de

# Contents of this repo:
- Folder `core`: General project code, e.g.  
    E.g. have a look at ...  
    - ... `core/access.hpp` & `core/storage.hpp` for data copying as well as direct I/O primitives
    - ... `core/apps.hpp` & `core/comp.cuh` &  `core/core.cuh` for decompression and higher-level data pipelines
    - ... `core/mem.hpp` & `core/dev_structs.cuh` & `core/metadata.cuh` for device-side data structures for memory layouts, query processing and pruning by metadata 

- Folder `plot`: Plotting scripts
- Folder `scripts`: Scripts for installing packages and general maintenance
- Folder `usecases`: Query processing applications in SSB/TAXI/TPCH context 

# Plotting:
Needs Python 3, Matplotlib and Pandas
```bash
cd <golap_public_folder>/plots
chmod +x plot_all.sh
./plot_all.sh
cd <golap_public_folder>/usecases/ssb
chmod +x plot_all.sh
./plot_all.sh
```

# Executing
## Installing necessary packages
```bash
./scripts/init.sh
```
Packages will be installed in "third_party" folder

## Query processing
E.g. for an application in the Star Schema Benchmark context:
```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j ssb_disk_db ssb
../bin/ssb_disk_db --scale_factor=10 --path="/PATH/diskdb10.dat" --op="write" --format=binary
../ssb/scripts/run_ssb.sh 0 # adapt parameters in file before, e.g. SSD path, queries etc.
```

E.g. for TPC-H queries:
```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j tpch_disk_db tpch
../tpch/scripts/run_tpch.sh 0 # adapt parameters in file before, e.g. path to TPCH csvs, SSD path, queries etc.
```

## Comparison to DuckDB
Create parquet files, and comparison to DuckDB (installed in third_party).

```bash
../ssb/scripts/duckdb_ssb/init_duckdb.sh
make -j duckdb_ssb
../ssb/scripts/run_duckdb.sh 0
```


