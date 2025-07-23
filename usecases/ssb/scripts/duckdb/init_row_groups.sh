#!/bin/bash
# set -x

# go to this files directory
DIR="$(dirname $(readlink -f "$0"))"
cd $DIR

# 1) First read the csvs into a duckdb database (e.g. a transient memory db)
# 2) Export the database in parquet format with different compressions (Snappy, uncompressed, etc.)
# 3) Fix load.sql and schema.sql in the parquets folder, so an import creates a view from the parquet files (which doesnt read the complete files from disk)
# 4) Run the queries on the view, which will do projection pushdown/decompression/chunking etc. on the parquets.

encodings=("snappy" "uncompressed")
tables=("lineorder" "part" "supplier" "customer" "date")


scale_factor=100
row_group_sizes=("61440" "122880" "524288" "2097152")
duckdb_exec="/home/nboeschen/duckdb"
csv_folder="/mnt/labstore/nboeschen/ssb/csv${scale_factor}"
parquet_folder="/raid/gds/ssb/parquet${scale_factor}"
sql_to_execute=$(sed -e "s|CSV_FOLDER|${csv_folder}|g" csv_to_parquet.sql)



for row_group_size in "${row_group_sizes[@]}"; do
    for enc in "${encodings[@]}"; do
        mkdir -p "${parquet_folder}/${enc}_${row_group_size}"
        for table in "${tables[@]}"; do
            if [[ "$table" == "lineorder" ]]; then
            sql_to_execute="${sql_to_execute}
            COPY (SELECT * FROM ${table}) to '${parquet_folder}/${enc}_${row_group_size}/${table}.parquet' (FORMAT 'PARQUET', CODEC '${enc^^}', ROW_GROUP_SIZE ${row_group_size});"
            fi
        done
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
for row_group_size in "${row_group_sizes[@]}"; do
    for enc in "${encodings[@]}"; do
        # rm "${parquet_folder}/${enc}/schema.sql"
        sed -e "s|PARQUET_FOLDER|${parquet_folder}/${enc}_${row_group_size}|g" schema.sql > "${parquet_folder}/${enc}_${row_group_size}/schema.sql"
        # truncate -s 0 "${parquet_folder}/${enc}/load.sql"
        for table in "${tables[@]}"; do
            if [[ "$table" != "lineorder" ]]; then
                ln -s "${parquet_folder}/${enc}/${table}.parquet" "${parquet_folder}/${enc}_${row_group_size}/${table}.parquet"
            fi
        done
        touch "${parquet_folder}/${enc}_${row_group_size}/load.sql"
    done
done
