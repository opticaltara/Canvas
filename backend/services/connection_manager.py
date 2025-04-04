"""
Connection Manager Service

Manages connections to data sources via MCP servers
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Type
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import aiofiles

from backend.config import get_settings
from backend.context.engine import get_context_engine


class ConnectionConfig(BaseModel):
    """Configuration for a connection"""
    id: str
    name: str
    type: str  # "grafana", "postgres", "prometheus", "loki", "s3", etc.
    config: Dict[str, str]  # Configuration details (API keys, URLs, etc.)


# Singleton instance
_connection_manager_instance = None


def get_connection_manager():
    """Get the singleton ConnectionManager instance"""
    global _connection_manager_instance
    if _connection_manager_instance is None:
        _connection_manager_instance = ConnectionManager()
    return _connection_manager_instance


class ConnectionManager:
    """
    Manages connections to data sources via MCP servers
    """
    def __init__(self):
        self.connections: Dict[str, ConnectionConfig] = {}
        self.default_connections: Dict[str, str] = {}
        self.settings = get_settings()
        self.context_engine = get_context_engine()
        self.active_mcp_servers = {}
    
    async def initialize(self) -> None:
        """Initialize and load connections"""
        # Load connections from storage
        await self._load_connections()
    
    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        # Stop any running MCP servers
        for server_id, server in self.active_mcp_servers.items():
            if hasattr(server, 'stop') and callable(server.stop):
                await server.stop()
        
        # Close the context engine
        await self.context_engine.close()
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionConfig]:
        """
        Get a connection by ID
        
        Args:
            connection_id: The ID of the connection
            
        Returns:
            The connection, or None if not found
        """
        return self.connections.get(connection_id)
    
    def get_connections_by_type(self, connection_type: str) -> List[ConnectionConfig]:
        """
        Get all connections for a specific type
        
        Args:
            connection_type: The type of connection
            
        Returns:
            List of connections for the type
        """
        return [
            conn for conn in self.connections.values()
            if conn.type == connection_type
        ]
    
    def get_all_connections(self) -> List[Dict]:
        """
        Get all connections (with sensitive fields redacted)
        
        Returns:
            List of connection information
        """
        return [
            {
                "id": conn.id,
                "name": conn.name,
                "type": conn.type,
                "config": self._redact_sensitive_fields(conn.config)
            }
            for conn in self.connections.values()
        ]
    
    def get_default_connection(self, connection_type: str) -> Optional[ConnectionConfig]:
        """
        Get the default connection for a type
        
        Args:
            connection_type: The type of connection
            
        Returns:
            The default connection, or None if not found
        """
        connection_id = self.default_connections.get(connection_type)
        if connection_id:
            return self.get_connection(connection_id)
        
        # If no default is set, try to find any connection for this type
        connections = self.get_connections_by_type(connection_type)
        if connections:
            return connections[0]
        
        return None
    
    async def create_connection(self, name: str, type: str, config: Dict) -> ConnectionConfig:
        """
        Create a new connection
        
        Args:
            name: The name of the connection
            type: The type of connection
            config: The connection configuration
            
        Returns:
            The created connection
            
        Raises:
            ValueError: If connection validation fails
        """
        # Create a unique ID for the connection
        connection_id = str(uuid4())
        
        # Create the connection config
        connection = ConnectionConfig(
            id=connection_id,
            name=name,
            type=type,
            config=config
        )
        
        # Validate the connection
        is_valid = await self.test_connection(type, config)
        if not is_valid:
            raise ValueError(f"Connection validation failed for {name}")
        
        # Add to connections
        self.connections[connection_id] = connection
        
        # If this is the first connection for this type, set it as default
        if not self.get_default_connection(type):
            self.default_connections[type] = connection_id
        
        # Save connections
        await self._save_connections()
        
        # Index the connection schema/metadata for RAG
        await self._index_connection_for_rag(connection_id)
        
        return connection
    
    async def update_connection(self, connection_id: str, name: str, config: Dict) -> ConnectionConfig:
        """
        Update an existing connection
        
        Args:
            connection_id: The ID of the connection to update
            name: The new name of the connection
            config: The new connection configuration
            
        Returns:
            The updated connection
            
        Raises:
            ValueError: If the connection is not found or validation fails
        """
        connection = self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Update the connection
        updated_connection = ConnectionConfig(
            id=connection_id,
            name=name,
            type=connection.type,
            config=config
        )
        
        # Validate the connection
        is_valid = await self.test_connection(connection.type, config)
        if not is_valid:
            raise ValueError(f"Connection validation failed for {name}")
        
        # Update the connection
        self.connections[connection_id] = updated_connection
        
        # Save connections
        await self._save_connections()
        
        # Re-index the connection schema/metadata for RAG
        await self._index_connection_for_rag(connection_id)
        
        return updated_connection
    
    async def delete_connection(self, connection_id: str) -> None:
        """
        Delete a connection
        
        Args:
            connection_id: The ID of the connection to delete
            
        Raises:
            ValueError: If the connection is not found
        """
        connection = self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Remove the connection
        del self.connections[connection_id]
        
        # Update default connections if needed
        for conn_type, default_id in list(self.default_connections.items()):
            if default_id == connection_id:
                # Find another connection for this type
                connections = self.get_connections_by_type(conn_type)
                if connections:
                    self.default_connections[conn_type] = connections[0].id
                else:
                    del self.default_connections[conn_type]
        
        # Save connections
        await self._save_connections()
    
    async def set_default_connection(self, connection_id: str) -> None:
        """
        Set a connection as the default for its type
        
        Args:
            connection_id: The ID of the connection to set as default
            
        Raises:
            ValueError: If the connection is not found
        """
        connection = self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Set as default
        self.default_connections[connection.type] = connection_id
        
        # Save connections
        await self._save_connections()
    
    async def test_connection(self, connection_type: str, config: Dict) -> bool:
        """
        Test a connection configuration
        
        Args:
            connection_type: The type of connection
            config: The connection configuration to test
            
        Returns:
            True if the connection is valid, False otherwise
        """
        try:
            # Basic validation for each type
            if connection_type == "grafana":
                if not config.get("url") or not config.get("api_key"):
                    return False
                
                # TODO: Actually test the connection to Grafana
                # For now, just return True if the required fields are present
                return True
            
            elif connection_type == "postgres":
                if not config.get("connection_string"):
                    return False
                
                # TODO: Actually test the connection to Postgres
                # For now, just return True if the required fields are present
                return True
            
            elif connection_type == "prometheus":
                if not config.get("url"):
                    return False
                
                # TODO: Actually test the connection to Prometheus
                # For now, just return True if the required fields are present
                return True
            
            elif connection_type == "loki":
                if not config.get("url"):
                    return False
                
                # TODO: Actually test the connection to Loki
                # For now, just return True if the required fields are present
                return True
            
            elif connection_type == "s3":
                if not config.get("endpoint") or not config.get("access_key") or not config.get("secret_key"):
                    return False
                
                # TODO: Actually test the connection to S3
                # For now, just return True if the required fields are present
                return True
            
            else:
                # Unknown connection type
                return False
        
        except Exception as e:
            print(f"Error testing connection: {e}")
            return False
    
    async def get_connection_schema(self, connection_id: str) -> Dict:
        """
        Get the schema for a connection
        
        Args:
            connection_id: The ID of the connection
            
        Returns:
            The schema information
            
        Raises:
            ValueError: If the connection is not found
        """
        connection = self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Placeholder schema extraction logic
        # In a real implementation, this would connect to the MCP server
        # and retrieve the schema information
        
        if connection.type == "grafana":
            return {"message": "Grafana schema retrieval not implemented yet"}
        
        elif connection.type == "postgres":
            return {"message": "Postgres schema retrieval not implemented yet"}
        
        elif connection.type == "prometheus":
            return {"message": "Prometheus schema retrieval not implemented yet"}
        
        elif connection.type == "loki":
            return {"message": "Loki schema retrieval not implemented yet"}
        
        elif connection.type == "s3":
            return {"message": "S3 schema retrieval not implemented yet"}
        
        else:
            return {"message": f"Unknown connection type: {connection.type}"}
    
    async def _index_connection_for_rag(self, connection_id: str) -> None:
        """
        Index connection schema/metadata for RAG
        
        Args:
            connection_id: The ID of the connection
        """
        try:
            connection = self.get_connection(connection_id)
            if not connection:
                return
            
            # Placeholder for schema extraction and indexing
            # Note: A real implementation would extract the schema and
            # use the context engine to index it
            
            # Get schema data (placeholder)
            schema_data = await self.get_connection_schema(connection_id)
            
            # Index based on connection type
            if connection.type == "postgres" or connection.type == "sql":
                # Convert schema to expected format and index
                sql_schema = {"tables": {}}  # Placeholder
                await self.context_engine.index_sql_schema(
                    connection_id, connection.name, sql_schema
                )
            
            elif connection.type == "prometheus":
                # Convert schema to expected format and index
                prometheus_data = {"metrics": []}  # Placeholder
                await self.context_engine.index_prometheus_metrics(
                    connection_id, connection.name, prometheus_data
                )
            
            elif connection.type == "loki":
                # Convert schema to expected format and index
                loki_data = {"labels": []}  # Placeholder
                await self.context_engine.index_loki_logs(
                    connection_id, connection.name, loki_data
                )
            
            elif connection.type == "s3":
                # Convert schema to expected format and index
                s3_data = {"buckets": []}  # Placeholder
                await self.context_engine.index_s3_buckets(
                    connection_id, connection.name, s3_data
                )
            
        except Exception as e:
            print(f"Error indexing connection for RAG: {e}")
    
    def _redact_sensitive_fields(self, config: Dict) -> Dict:
        """
        Redact sensitive fields from a connection configuration
        
        Args:
            config: The configuration to redact
            
        Returns:
            The redacted configuration
        """
        sensitive_fields = [
            "password", "secret", "key", "token", "api_key", 
            "aws_secret_access_key", "private_key"
        ]
        
        redacted_config = config.copy()
        
        for key in redacted_config:
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                redacted_config[key] = "********"
        
        return redacted_config
    
    async def _save_connections(self) -> None:
        """Save connections to storage"""
        # Convert to serializable format
        connections_data = {
            conn_id: {
                "id": conn.id,
                "name": conn.name,
                "type": conn.type,
                "config": conn.config
            }
            for conn_id, conn in self.connections.items()
        }
        
        data = {
            "connections": connections_data,
            "default_connections": self.default_connections
        }
        
        # Check storage type
        storage_type = self.settings.connection_storage_type
        
        if storage_type == "file":
            # Save to local file
            await self._save_connections_to_file(data)
        elif storage_type == "env":
            # Connections are loaded from environment, no need to save
            pass
        else:
            # Default to in-memory only (no persistence)
            pass
    
    async def _load_connections(self) -> None:
        """Load connections from storage"""
        # Check storage type
        storage_type = self.settings.connection_storage_type
        
        if storage_type == "file":
            # Load from local file
            data = await self._load_connections_from_file()
        elif storage_type == "env":
            # Load from environment variables
            data = self._load_connections_from_env()
        else:
            # Default to empty
            data = {"connections": {}, "default_connections": {}}
        
        # Convert to ConnectionConfig objects
        self.connections = {
            conn_id: ConnectionConfig(
                id=conn["id"],
                name=conn["name"],
                type=conn["type"],
                config=conn["config"]
            )
            for conn_id, conn in data.get("connections", {}).items()
        }
        
        self.default_connections = data.get("default_connections", {})
        
        # Index all connections for RAG
        for connection_id in self.connections:
            await self._index_connection_for_rag(connection_id)
    
    async def _save_connections_to_file(self, data: Dict) -> None:
        """Save connections to a local file"""
        file_path = self.settings.connection_file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to file (use async file operations)
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(data, indent=2))
    
    async def _load_connections_from_file(self) -> Dict:
        """Load connections from a local file"""
        file_path = self.settings.connection_file_path
        
        if not os.path.exists(file_path):
            return {"connections": {}, "default_connections": {}}
        
        # Load from file (use async file operations)
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            print(f"Error loading connections from file: {e}")
            return {"connections": {}, "default_connections": {}}
    
    def _load_connections_from_env(self) -> Dict:
        """Load connections from environment variables"""
        connections = {}
        default_connections = {}
        
        # Example format: SHERLOG_CONNECTION_GRAFANA_MYSERVER_NAME=My Grafana Server
        # Example format: SHERLOG_CONNECTION_GRAFANA_MYSERVER_CONFIG_URL=https://grafana.example.com
        # Example format: SHERLOG_CONNECTION_DEFAULT_GRAFANA=myserver
        
        prefix = "SHERLOG_CONNECTION_"
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            parts = key[len(prefix):].split("_")
            if len(parts) < 2:
                continue
            
            # Check if this is a default connection
            if parts[0] == "DEFAULT" and len(parts) >= 2:
                connection_type = parts[1].lower()
                default_connections[connection_type] = value
                continue
            
            connection_type = parts[0].lower()
            conn_id = parts[1].lower()
            
            if len(parts) < 3:
                continue
            
            # Get the field name and value
            if parts[2] == "NAME":
                # Connection name
                if conn_id not in connections:
                    connections[conn_id] = {
                        "id": conn_id,
                        "name": value,
                        "type": connection_type,
                        "config": {}
                    }
                else:
                    connections[conn_id]["name"] = value
            
            elif parts[2] == "CONFIG" and len(parts) >= 4:
                # Connection config
                config_key = "_".join(parts[3:]).lower()
                
                if conn_id not in connections:
                    connections[conn_id] = {
                        "id": conn_id,
                        "name": f"{connection_type.capitalize()} Connection",
                        "type": connection_type,
                        "config": {config_key: value}
                    }
                else:
                    if "config" not in connections[conn_id]:
                        connections[conn_id]["config"] = {}
                    connections[conn_id]["config"][config_key] = value
        
        return {
            "connections": connections,
            "default_connections": default_connections
        }