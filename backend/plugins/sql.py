import asyncio
from typing import Any, Dict, List, Optional

import aiomysql
import aiopg
import aiosqlite
import pandas as pd

from backend.core.cell import Cell, SQLCell
from backend.core.execution import ExecutionContext
from backend.plugins.base import ConnectionConfig, PluginBase, PluginConfig


class SQLPlugin(PluginBase):
    """Plugin for SQL database connections"""
    
    def __init__(self):
        super().__init__(
            config=PluginConfig(
                name="sql",
                description="SQL database connector",
                version="0.1.0",
                config={
                    "default_timeout": 30,  # Default query timeout in seconds
                }
            )
        )
        self.connections = {}
    
    async def validate_connection(self, connection_config: ConnectionConfig) -> bool:
        """Validate a SQL connection configuration"""
        try:
            # Test the connection
            conn = await self._get_connection(connection_config)
            async with conn as conn:
                return True
        except Exception as e:
            print(f"Error validating SQL connection: {e}")
            return False
    
    async def execute_query(
        self,
        query: str,
        connection_config: ConnectionConfig,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute an SQL query"""
        try:
            dialect = connection_config.config.get("dialect", "sqlite")
            timeout = connection_config.config.get("timeout", self.get_config("default_timeout"))
            
            # Execute the query
            conn = await self._get_connection(connection_config)
            async with conn as conn:
                # SQLite
                if dialect == "sqlite":
                    results = await self._execute_sqlite(conn, query, parameters, timeout)
                
                # PostgreSQL
                elif dialect == "postgresql":
                    results = await self._execute_postgres(conn, query, parameters, timeout)
                
                # MySQL
                elif dialect == "mysql":
                    results = await self._execute_mysql(conn, query, parameters, timeout)
                
                else:
                    raise ValueError(f"Unsupported SQL dialect: {dialect}")
                
                return {
                    "data": results,
                    "query": query,
                    "parameters": parameters,
                    "dialect": dialect,
                }
        
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "parameters": parameters,
            }
    
    async def get_schema(self, connection_config: ConnectionConfig) -> Dict[str, Any]:
        """Get the database schema"""
        dialect = connection_config.config.get("dialect", "sqlite")
        schema = {}
        
        try:
            conn = await self._get_connection(connection_config)
            async with conn as conn:
                # SQLite
                if dialect == "sqlite":
                    schema = await self._get_sqlite_schema(conn)
                
                # PostgreSQL
                elif dialect == "postgresql":
                    schema = await self._get_postgres_schema(conn)
                
                # MySQL
                elif dialect == "mysql":
                    schema = await self._get_mysql_schema(conn)
                
                else:
                    raise ValueError(f"Unsupported SQL dialect: {dialect}")
                
                return schema
        
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_connection(self, connection_config: ConnectionConfig):
        """Get a database connection based on the configuration"""
        dialect = connection_config.config.get("dialect", "sqlite")
        connection_string = connection_config.config.get("connection_string", "")
        
        # SQLite
        if dialect == "sqlite":
            return await aiosqlite.connect(connection_string)
        
        # PostgreSQL
        elif dialect == "postgresql":
            return await aiopg.connect(connection_string)
        
        # MySQL
        elif dialect == "mysql":
            return await aiomysql.connect(
                host=connection_config.config.get("host", "localhost"),
                port=connection_config.config.get("port", 3306),
                user=connection_config.config.get("user", ""),
                password=connection_config.config.get("password", ""),
                db=connection_config.config.get("database", ""),
            )
        
        else:
            raise ValueError(f"Unsupported SQL dialect: {dialect}")
    
    async def _execute_sqlite(self, conn, query, parameters, timeout):
        """Execute a query on SQLite"""
        try:
            cursor = await conn.execute(query, parameters or {})
            columns = [description[0] for description in cursor.description] if cursor.description else []
            rows = await cursor.fetchall()
            await conn.commit()
            
            # Convert to list of dicts
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            return results
        
        except Exception as e:
            await conn.rollback()
            raise e
    
    async def _execute_postgres(self, conn, query, parameters, timeout):
        """Execute a query on PostgreSQL"""
        async with conn.cursor() as cursor:
            try:
                await cursor.execute(query, parameters or {})
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = await cursor.fetchall()
                
                # Convert to list of dicts
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                return results
            
            except Exception as e:
                conn.rollback()
                raise e
    
    async def _execute_mysql(self, conn, query, parameters, timeout):
        """Execute a query on MySQL"""
        async with conn.cursor() as cursor:
            try:
                await cursor.execute(query, parameters or {})
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = await cursor.fetchall()
                
                # Convert to list of dicts
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                return results
            
            except Exception as e:
                await conn.rollback()
                raise e
    
    async def _get_sqlite_schema(self, conn):
        """Get the schema for a SQLite database"""
        tables = {}
        
        # Get tables
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_rows = await cursor.fetchall()
        
        for (table_name,) in table_rows:
            # Get columns for each table
            cursor = await conn.execute(f"PRAGMA table_info({table_name})")
            column_rows = await cursor.fetchall()
            
            columns = []
            for col in column_rows:
                columns.append({
                    "name": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                    "primary_key": bool(col[5]),
                })
            
            tables[table_name] = {
                "columns": columns,
                "primary_key": next((col["name"] for col in columns if col["primary_key"]), None),
            }
        
        return {
            "tables": tables,
            "dialect": "sqlite",
        }
    
    async def _get_postgres_schema(self, conn):
        """Get the schema for a PostgreSQL database"""
        tables = {}
        
        async with conn.cursor() as cursor:
            # Get all tables
            await cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            table_rows = await cursor.fetchall()
            
            for (table_name,) in table_rows:
                # Get columns for each table
                await cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default,
                           (SELECT constraint_type FROM information_schema.table_constraints tc
                            JOIN information_schema.constraint_column_usage ccu 
                            ON tc.constraint_name = ccu.constraint_name
                            WHERE tc.constraint_type = 'PRIMARY KEY' 
                            AND tc.table_name = c.table_name 
                            AND ccu.column_name = c.column_name
                            LIMIT 1) as is_primary
                    FROM information_schema.columns c
                    WHERE table_name = %s
                """, (table_name,))
                column_rows = await cursor.fetchall()
                
                columns = []
                for row in column_rows:
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3],
                        "primary_key": row[4] == "PRIMARY KEY",
                    })
                
                tables[table_name] = {
                    "columns": columns,
                    "primary_key": next((col["name"] for col in columns if col["primary_key"]), None),
                }
        
        return {
            "tables": tables,
            "dialect": "postgresql",
        }
    
    async def _get_mysql_schema(self, conn):
        """Get the schema for a MySQL database"""
        tables = {}
        
        async with conn.cursor() as cursor:
            # Get all tables
            await cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = DATABASE()
            """)
            table_rows = await cursor.fetchall()
            
            for (table_name,) in table_rows:
                # Get columns for each table
                await cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default,
                           column_key = 'PRI' as is_primary
                    FROM information_schema.columns
                    WHERE table_schema = DATABASE() AND table_name = %s
                """, (table_name,))
                column_rows = await cursor.fetchall()
                
                columns = []
                for row in column_rows:
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3],
                        "primary_key": bool(row[4]),
                    })
                
                tables[table_name] = {
                    "columns": columns,
                    "primary_key": next((col["name"] for col in columns if col["primary_key"]), None),
                }
        
        return {
            "tables": tables,
            "dialect": "mysql",
        }


async def execute_sql_cell(cell: SQLCell, context: ExecutionContext) -> Dict[str, Any]:
    """
    Execute an SQL cell
    
    Args:
        cell: The SQL cell to execute
        context: The execution context
        
    Returns:
        The results of the SQL query
    """
    # This would normally be injected or retrieved from a registry
    from backend.services.connection_manager import get_connection_manager
    connection_manager = get_connection_manager()
    
    # Get the connection configuration
    connection_id = cell.connection_id
    if not connection_id:
        return {
            "error": "No connection specified for SQL cell",
            "query": cell.content,
        }
    
    # Get the connection
    connection_config = connection_manager.get_connection(connection_id)
    if not connection_config:
        return {
            "error": f"Connection not found: {connection_id}",
            "query": cell.content,
        }
    
    # Get the plugin
    plugin = connection_manager.get_plugin(connection_config.plugin_name)
    if not plugin or not isinstance(plugin, SQLPlugin):
        return {
            "error": f"SQL plugin not found for connection: {connection_id}",
            "query": cell.content,
        }
    
    # Execute the query
    parameters = context.variables if context.variables else {}
    result = await plugin.execute_query(cell.content, connection_config, parameters)
    
    # Convert to pandas DataFrame for easier analysis in dependent cells
    if "data" in result and not "error" in result:
        try:
            result["dataframe"] = pd.DataFrame(result["data"])
        except Exception as e:
            print(f"Error converting SQL result to DataFrame: {e}")
    
    return result