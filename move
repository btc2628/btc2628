SELECT 
    OBJECT_NAME(ius.object_id) AS TableName,
    MAX(ius.last_user_update) AS LastUpdated
FROM 
    sys.dm_db_index_usage_stats ius
INNER JOIN 
    sys.tables t ON ius.object_id = t.object_id
WHERE 
    ius.database_id = DB_ID('YourDatabaseName') AND 
    OBJECT_NAME(ius.object_id) = 'YourTableName'
GROUP BY 
    ius.object_id;
