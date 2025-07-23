#!/bin/bash
# set -x

# go to this files directory
DIR="$(dirname $(readlink -f "$0"))"
cd $DIR

# 1) First read the csvs into a duckdb database (e.g. a transient memory db)
# 2) Export the database in parquet format with different compressions (Snappy, uncompressed, etc.)
# 3) Fix load.sql and schema.sql in the parquets folder, so an import creates a view from the parquet files (which doesnt read the complete files from disk)
# 4) Run the queries on the view, which will do projection pushdown/decompression/chunking etc. on the parquets.

encodings=("snappy" "zstd" "gzip" "uncompressed")
# encodings=("zstd" "gzip")

scale_factor=200
duckdb_exec="/home/nboeschen/duckdb"
csv_folder="/mnt/labstore/nboeschen/tpch/csv${scale_factor}"
parquet_folder="/raid/gds/tpch/parquet${scale_factor}"
sql_to_execute=$(sed -e "s|CSV_FOLDER|${csv_folder}|g" csv_to_parquet.sql)


for enc in "${encodings[@]}"; do
    mkdir -p "${parquet_folder}/${enc}"
    sql_to_execute="${sql_to_execute}
    EXPORT DATABASE '${parquet_folder}/${enc}/' (FORMAT 'PARQUET', CODEC '${enc^^}', ROW_GROUP_SIZE 1048576);"
    # EXPORT DATABASE '${parquet_folder}/${enc}/' (FORMAT 'PARQUET', CODEC '${enc^^}', ROW_GROUP_SIZE 1000000000);"
done
echo "$sql_to_execute"
# exit 0


$duckdb_exec -c "$sql_to_execute"

if [ $? -eq 0 ]; then
    echo "Success, fixing load and schema files..."
else
    echo "DuckDB import failed, exiting."
    exit 1
fi

# now fix load and schema: schema creates views, load is empty
for enc in "${encodings[@]}"; do
    rm "${parquet_folder}/${enc}/schema.sql"
    sed -e "s|PARQUET_FOLDER|${parquet_folder}/${enc}|g" schema.sql > "${parquet_folder}/${enc}/schema.sql"
    truncate -s 0 "${parquet_folder}/${enc}/load.sql"
done



