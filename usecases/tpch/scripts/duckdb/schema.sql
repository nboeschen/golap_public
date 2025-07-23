
CREATE VIEW customer AS SELECT * FROM read_parquet('PARQUET_FOLDER/customer.parquet');
CREATE VIEW lineitem AS SELECT * FROM read_parquet('PARQUET_FOLDER/lineitem.parquet');
CREATE VIEW nation AS SELECT * FROM read_parquet('PARQUET_FOLDER/nation.parquet');
CREATE VIEW orders AS SELECT * FROM read_parquet('PARQUET_FOLDER/orders.parquet');
CREATE VIEW part AS SELECT * FROM read_parquet('PARQUET_FOLDER/part.parquet');
CREATE VIEW partsupp AS SELECT * FROM read_parquet('PARQUET_FOLDER/partsupp.parquet');
CREATE VIEW region AS SELECT * FROM read_parquet('PARQUET_FOLDER/region.parquet');
CREATE VIEW supplier AS SELECT * FROM read_parquet('PARQUET_FOLDER/supplier.parquet');
