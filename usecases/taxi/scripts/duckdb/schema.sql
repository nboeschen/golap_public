
CREATE VIEW trips AS SELECT * FROM read_parquet('PARQUET_FOLDER/trips.parquet');
