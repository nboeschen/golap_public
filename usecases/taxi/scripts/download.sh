#!/bin/bash

# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2015-01.parquet
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2015-02.parquet


for month in {01..12}; do
    wget --inet4-only -c "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2015-${month}.parquet" -P ./parquets/
    wget --inet4-only -c "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-${month}.parquet" -P ./parquets/
    wget --inet4-only -c "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2017-${month}.parquet" -P ./parquets/
    wget --inet4-only -c "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2018-${month}.parquet" -P ./parquets/
    wget --inet4-only -c "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-${month}.parquet" -P ./parquets/
done

# COPY (SELECT * FROM read_parquet('/mnt/labstore/nboeschen/taxi/*.parquet')) TO '/mnt/labstore/nboeschen/taxi/trips.csv' (FORMAT CSV, DELIMITER '|', HEADER);
