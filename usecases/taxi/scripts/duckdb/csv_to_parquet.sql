CREATE TABLE IF NOT EXISTS trips (
    VendorID BIGINT,
    tpep_pickup_datetime DATETIME,
    tpep_dropoff_datetime DATETIME,
    passenger_count BIGINT,
    trip_distance DOUBLE,
    RatecodeID BIGINT,
    store_and_fwd_flag CHAR,
    PULocationID BIGINT,
    DOLocationID BIGINT,
    payment_type BIGINT,
    fare_amount DOUBLE,
    extra DOUBLE,
    mta_tax DOUBLE,
    tip_amount DOUBLE,
    tolls_amount DOUBLE,
    improvement_surcharge DOUBLE,
    total_amount DOUBLE,
    congestion_surcharge INTEGER,
    airport_fee INTEGER
);


SET enable_progress_bar=true;

COPY trips FROM 'CSV_FOLDER/trips.csv' (DELIMITER '|',HEADER 1);
