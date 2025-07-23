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
sorted_encodings=("snappy")
tables=("lineorder" "part" "supplier" "customer" "date")


scale_factor=100
row_group_size=1048576
duckdb_exec="/home/nboeschen/duckdb"
csv_folder="/mnt/labstore/nboeschen/ssb/csv${scale_factor}"
parquet_folder="/raid/gds/ssb/parquet${scale_factor}"
sql_to_execute=$(sed -e "s|CSV_FOLDER|${csv_folder}|g" csv_to_parquet.sql)



# for enc in "${encodings[@]}"; do
#     mkdir -p "${parquet_folder}/${enc}"
#     for table in "${tables[@]}"; do
#         sql_to_execute="${sql_to_execute}
#         COPY (SELECT * FROM ${table}) to '${parquet_folder}/${enc}/${table}.parquet' (FORMAT 'PARQUET', CODEC '${enc^^}', ROW_GROUP_SIZE ${row_group_size});"
#     done
# done

# sorted:
for enc in "${sorted_encodings[@]}"; do
    mkdir -p "${parquet_folder}/${enc}_sorted"
    for table in "${tables[@]}"; do
        #     sql_to_execute="${sql_to_execute}
        #     COPY (SELECT * FROM ${table} ORDER BY (lo_discount, lo_quantity)) to '${parquet_folder}/${enc}_sorted/${table}.parquet' (FORMAT 'PARQUET', PER_THREAD_OUTPUT, CODEC '${enc^^}', ROW_GROUP_SIZE ${row_group_size});"
        # else
        if [[ "$table" == "lineorder" ]]; then
            sql_to_execute="${sql_to_execute}
            COPY (SELECT * FROM ${table}) to '${parquet_folder}/${enc}_sorted/${table}.parquet' (FORMAT 'PARQUET', PER_THREAD_OUTPUT, CODEC '${enc^^}', ROW_GROUP_SIZE ${row_group_size});"
        fi
    done
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
# for enc in "${encodings[@]}"; do
#     # rm "${parquet_folder}/${enc}/schema.sql"
#     sed -e "s|PARQUET_FOLDER|${parquet_folder}/${enc}|g" schema.sql > "${parquet_folder}/${enc}/schema.sql"
#     # truncate -s 0 "${parquet_folder}/${enc}/load.sql"
#     touch "${parquet_folder}/${enc}/load.sql"
# done

# sorted
for enc in "${sorted_encodings[@]}"; do
    sed -e "s|PARQUET_FOLDER|${parquet_folder}/${enc}_sorted|g" schema.sql > "${parquet_folder}/${enc}_sorted/schema.sql"
    for table in "${tables[@]}"; do
        if [[ "$table" != "lineorder" ]]; then
            ln -s "${parquet_folder}/${enc}/${table}.parquet" "${parquet_folder}/${enc}_sorted/${table}.parquet"
        fi
    done
    # empty load.sql
    touch "${parquet_folder}/${enc}_sorted/load.sql"
done