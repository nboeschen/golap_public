
CREATE VIEW date AS SELECT * FROM read_parquet('PARQUET_FOLDER/date.parquet');
CREATE VIEW customer as SELECT * FROM read_parquet('PARQUET_FOLDER/customer.parquet');
CREATE VIEW lineorder as SELECT * FROM read_parquet('PARQUET_FOLDER/lineorder.parquet');
CREATE VIEW supplier as SELECT * FROM read_parquet('PARQUET_FOLDER/supplier.parquet');
CREATE VIEW part as SELECT * FROM read_parquet('PARQUET_FOLDER/part.parquet');
