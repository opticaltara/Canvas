import asyncio
import json
import os
from typing import Dict, List, Optional, Type
from uuid import UUID, uuid4

from backend.config import get_settings
from backend.context.engine import get_context_engine
from backend.plugins.base import ConnectionConfig, PluginBase
from backend.plugins.log import GrafanaLokiPlugin, LokiPlugin
from backend.plugins.metric import GrafanaPrometheusPlugin, PrometheusPlugin
from backend.plugins.python import PythonExecutor
from backend.plugins.s3 import S3Plugin
from backend.plugins.sql import SQLPlugin


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
    Manages connections to data sources and their plugins
    """
    def __init__(self):
        self.plugins: Dict[str, PluginBase] = {}
        self.connections: Dict[str, ConnectionConfig] = {}
        self.default_connections: Dict[str, str] = {}
        self.settings = get_settings()
        self.context_engine = get_context_engine()
    
    async def initialize(self) -> None:
        """Initialize plugins and load connections"""
        # Register plugins
        self._register_plugins()
        
        # Load connections from storage
        await self._load_connections()
    
    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        # Close the context engine
        await self.context_engine.close()
    
    def _register_plugins(self) -> None:
        """Register all available plugins"""
        # SQL plugin
        self.plugins["sql"] = SQLPlugin()
        
        # Log plugins
        self.plugins["loki"] = LokiPlugin()
        self.plugins["grafana_loki"] = GrafanaLokiPlugin()
        
        # Metric plugins
        self.plugins["prometheus"] = PrometheusPlugin()
        self.plugins["grafana_prometheus"] = GrafanaPrometheusPlugin()
        
        # S3 plugin
        self.plugins["s3"] = S3Plugin()
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """
        Get a plugin by name
        
        Args:
            plugin_name: The name of the plugin
            
        Returns:
            The plugin, or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def get_all_plugins(self) -> List[Dict]:
        """
        Get all registered plugins
        
        Returns:
            List of plugin information
        """
        return [
            {
                "name": plugin.name,
                "description": plugin.description,
                "version": plugin.version,
                "enabled": plugin.is_enabled
            }
            for plugin in self.plugins.values()
        ]
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionConfig]:
        """
        Get a connection by ID
        
        Args:
            connection_id: The ID of the connection
            
        Returns:
            The connection, or None if not found
        """
        return self.connections.get(connection_id)
    
    def get_connections_by_plugin(self, plugin_name: str) -> List[ConnectionConfig]:
        """
        Get all connections for a specific plugin
        
        Args:
            plugin_name: The name of the plugin
            
        Returns:
            List of connections for the plugin
        """
        return [
            conn for conn in self.connections.values()
            if conn.plugin_name == plugin_name
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
                "plugin_name": conn.plugin_name,
                "config": self._redact_sensitive_fields(conn.config)
            }
            for conn in self.connections.values()
        ]
    
    def get_default_connection(self, plugin_name: str) -> Optional[ConnectionConfig]:
        """
        Get the default connection for a plugin
        
        Args:
            plugin_name: The name of the plugin
            
        Returns:
            The default connection, or None if not found
        """
        connection_id = self.default_connections.get(plugin_name)
        if connection_id:
            return self.get_connection(connection_id)
        
        # If no default is set, try to find any connection for this plugin
        connections = self.get_connections_by_plugin(plugin_name)
        if connections:
            return connections[0]
        
        return None
    
    async def create_connection(self, name: str, plugin_name: str, config: Dict) -> ConnectionConfig:
        """
        Create a new connection
        
        Args:
            name: The name of the connection
            plugin_name: The name of the plugin
            config: The connection configuration
            
        Returns:
            The created connection
            
        Raises:
            ValueError: If the plugin is not found or connection validation fails
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Create the connection
        connection_id = str(uuid4())
        connection = ConnectionConfig(
            id=connection_id,
            name=name,
            plugin_name=plugin_name,
            config=config
        )
        
        # Validate the connection
        is_valid = await plugin.validate_connection(connection)
        if not is_valid:
            raise ValueError(f"Connection validation failed for {name}")
        
        # Add to connections
        self.connections[connection_id] = connection
        
        # If this is the first connection for this plugin, set it as default
        if not self.get_default_connection(plugin_name):
            self.default_connections[plugin_name] = connection_id
        
        # Save connections
        await self._save_connections()
        
        # Index the connection schema/metadata for RAG
        await self._index_connection_for_rag(connection_id, connection, plugin)
        
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
        
        plugin = self.get_plugin(connection.plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {connection.plugin_name}")
        
        # Update the connection
        updated_connection = ConnectionConfig(
            id=connection_id,
            name=name,
            plugin_name=connection.plugin_name,
            config=config
        )
        
        # Validate the connection
        is_valid = await plugin.validate_connection(updated_connection)
        if not is_valid:
            raise ValueError(f"Connection validation failed for {name}")
        
        # Update the connection
        self.connections[connection_id] = updated_connection
        
        # Save connections
        await self._save_connections()
        
        # Re-index the connection schema/metadata for RAG
        await self._index_connection_for_rag(connection_id, updated_connection, plugin)
        
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
        for plugin_name, default_id in list(self.default_connections.items()):
            if default_id == connection_id:
                # Find another connection for this plugin
                connections = self.get_connections_by_plugin(plugin_name)
                if connections:
                    self.default_connections[plugin_name] = connections[0].id
                else:
                    del self.default_connections[plugin_name]
        
        # Save connections
        await self._save_connections()
    
    async def set_default_connection(self, connection_id: str) -> None:
        """
        Set a connection as the default for its plugin
        
        Args:
            connection_id: The ID of the connection to set as default
            
        Raises:
            ValueError: If the connection is not found
        """
        connection = self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Set as default
        self.default_connections[connection.plugin_name] = connection_id
        
        # Save connections
        await self._save_connections()
    
    async def test_connection(self, plugin_name: str, config: Dict) -> bool:
        """
        Test a connection configuration
        
        Args:
            plugin_name: The name of the plugin
            config: The connection configuration to test
            
        Returns:
            True if the connection is valid, False otherwise
            
        Raises:
            ValueError: If the plugin is not found
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Create a temporary connection
        connection = ConnectionConfig(
            id="test",
            name="Test Connection",
            plugin_name=plugin_name,
            config=config
        )
        
        # Validate the connection
        return await plugin.validate_connection(connection)
    
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
        
        plugin = self.get_plugin(connection.plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {connection.plugin_name}")
        
        # Get the schema
        return await plugin.get_schema(connection)
    
    async def _index_connection_for_rag(
        self, connection_id: str, connection: ConnectionConfig, plugin: PluginBase
    ) -> None:
        """
        Index connection schema/metadata for RAG
        
        Args:
            connection_id: The ID of the connection
            connection: The connection configuration
            plugin: The plugin to use
        """
        try:
            # Get schema data
            schema_data = await plugin.get_schema(connection)
            
            # Handle different plugin types appropriately
            if connection.plugin_name == "sql":
                await self.context_engine.index_sql_schema(
                    connection_id, connection.name, schema_data
                )
            
            elif connection.plugin_name in ["prometheus", "grafana_prometheus"]:
                await self.context_engine.index_prometheus_metrics(
                    connection_id, connection.name, schema_data
                )
            
            elif connection.plugin_name in ["loki", "grafana_loki"]:
                await self.context_engine.index_loki_logs(
                    connection_id, connection.name, schema_data
                )
            
            elif connection.plugin_name == "s3":
                await self.context_engine.index_s3_buckets(
                    connection_id, connection.name, schema_data
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
                "plugin_name": conn.plugin_name,
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
                plugin_name=conn["plugin_name"],
                config=conn["config"]
            )
            for conn_id, conn in data.get("connections", {}).items()
        }
        
        self.default_connections = data.get("default_connections", {})
        
        # Index all connections for RAG
        for connection_id, connection in self.connections.items():
            plugin = self.get_plugin(connection.plugin_name)
            if plugin:
                await self._index_connection_for_rag(connection_id, connection, plugin)
    
    async def _save_connections_to_file(self, data: Dict) -> None:
        """Save connections to a local file"""
        file_path = self.settings.connection_file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to file (use async file operations)
        async with open(file_path, "w") as f:
            await f.write(json.dumps(data, indent=2))
    
    async def _load_connections_from_file(self) -> Dict:
        """Load connections from a local file"""
        file_path = self.settings.connection_file_path
        
        if not os.path.exists(file_path):
            return {"connections": {}, "default_connections": {}}
        
        # Load from file (use async file operations)
        try:
            async with open(file_path, "r") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            print(f"Error loading connections from file: {e}")
            return {"connections": {}, "default_connections": {}}
    
    def _load_connections_from_env(self) -> Dict:
        """Load connections from environment variables"""
        connections = {}
        default_connections = {}
        
        # Example format: SHERLOG_CONNECTION_SQL_DEFAULT=connection_id
        # Example format: SHERLOG_CONNECTION_SQL_MYDB_NAME=My Database
        # Example format: SHERLOG_CONNECTION_SQL_MYDB_CONFIG_URL=postgresql://user:pass@localhost/db
        
        prefix = "SHERLOG_CONNECTION_"
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            parts = key[len(prefix):].split("_")
            if len(parts) < 2:
                continue
            
            plugin_name = parts[0].lower()
            
            # Check if this is a default connection
            if parts[1] == "DEFAULT":
                default_connections[plugin_name] = value
                continue
            
            # This is a connection config
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
                        "plugin_name": plugin_name,
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
                        "name": f"{plugin_name.capitalize()} Connection",
                        "plugin_name": plugin_name,
                        "config": {config_key: value}
                    }
                else:
                    connections[conn_id]["config"][config_key] = value
        
        return {
            "connections": connections,
            "default_connections": default_connections
        }