USE TAXI
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED
SET STATISTICS IO on
SET STATISTICS time on
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

SELECT MONTH(tpep_pickup_datetime) as month, COUNT(*) trips 
FROM trips 
WHERE trip_distance >= 2 and trip_distance <= 5 
GROUP BY MONTH(tpep_pickup_datetime);
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

SELECT MONTH(tpep_pickup_datetime) as month, COUNT(*) trips 
FROM trips 
WHERE trip_distance >= 5 and trip_distance <= 15 
GROUP BY MONTH(tpep_pickup_datetime);
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

SELECT MONTH(tpep_pickup_datetime) as month, COUNT(*) trips 
FROM trips 
WHERE trip_distance >= 15 and trip_distance <= 5000 
GROUP BY MONTH(tpep_pickup_datetime);
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO


SELECT DATEPART(weekday, tpep_pickup_datetime), 
  ROUND(AVG(trip_distance / DATEDIFF(second, tpep_pickup_datetime, tpep_dropoff_datetime))*3600, 1) as speed 
FROM trips 
WHERE trip_distance > 0 AND fare_amount BETWEEN 0 AND 2 
AND tpep_dropoff_datetime > tpep_pickup_datetime 
GROUP BY DATEPART(weekday, tpep_pickup_datetime);
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

SELECT DATEPART(weekday, tpep_pickup_datetime), 
  ROUND(AVG(trip_distance / DATEDIFF(second, tpep_pickup_datetime, tpep_dropoff_datetime))*3600, 1) as speed 
FROM trips 
WHERE trip_distance > 0 AND fare_amount BETWEEN 2 AND 10 
AND tpep_dropoff_datetime > tpep_pickup_datetime 
GROUP BY DATEPART(weekday, tpep_pickup_datetime);
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

SELECT DATEPART(weekday, tpep_pickup_datetime), 
  ROUND(AVG(trip_distance / DATEDIFF(second, tpep_pickup_datetime, tpep_dropoff_datetime))*3600, 1) as speed 
FROM trips 
WHERE trip_distance > 0 AND fare_amount BETWEEN 10 AND 5000 
AND tpep_dropoff_datetime > tpep_pickup_datetime 
GROUP BY DATEPART(weekday, tpep_pickup_datetime);
GO
DBCC DROPCLEANBUFFERS
DBCC FREEPROCCACHE
GO

