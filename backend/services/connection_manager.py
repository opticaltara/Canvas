"""
Connection Manager Service

Manages connections to data sources via MCP servers
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from uuid import uuid4

from pydantic import BaseModel, Field
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.db.database import get_db_session
from backend.db.repositories import ConnectionRepository

# Configure logging
logger = logging.getLogger(__name__)

# Make sure correlation_id is set for all logs
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

# Add the filter to our logger
logger.addFilter(CorrelationIdFilter())

# Base connection configuration model
class BaseConnectionConfig(BaseModel):
    """Base connection configuration"""
    id: str
    name: str
    type: str
    
class GrafanaConnectionConfig(BaseConnectionConfig):
    """Grafana connection configuration"""
    type: str = "grafana"
    url: str
    api_key: str
    
class KubernetesConnectionConfig(BaseConnectionConfig):
    """Kubernetes connection configuration"""
    type: str = "kubernetes"
    kubeconfig: str
    context: Optional[str] = None
    
class S3ConnectionConfig(BaseConnectionConfig):
    """S3 connection configuration"""
    type: str = "s3"
    bucket: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    
class PostgresConnectionConfig(BaseConnectionConfig):
    """PostgreSQL connection configuration"""
    type: str = "postgres"
    host: str
    port: str
    database: str
    username: str
    password: str
    
# Generic fallback for other types
class GenericConnectionConfig(BaseConnectionConfig):
    """Generic connection configuration"""
    config: Dict[str, str] = Field(default_factory=dict)

# Main connection config that can be any specific type
class ConnectionConfig(BaseModel):
    """Connection configuration"""
    id: str
    name: str
    type: str  # "grafana", "sql", "prometheus", "loki", "s3", "kubernetes", etc.
    config: Dict[str, Any] = Field(default_factory=dict)
    
    def to_specific_config(self) -> Union[
        GrafanaConnectionConfig, 
        KubernetesConnectionConfig,
        S3ConnectionConfig,
        PostgresConnectionConfig,
        GenericConnectionConfig
    ]:
        """Convert to a type-specific configuration"""
        if self.type == "grafana":
            return GrafanaConnectionConfig(
                id=self.id,
                name=self.name,
                url=self.config.get("url", ""),
                api_key=self.config.get("api_key", "")
            )
        elif self.type == "kubernetes":
            return KubernetesConnectionConfig(
                id=self.id,
                name=self.name,
                kubeconfig=self.config.get("kubeconfig", ""),
                context=self.config.get("context")
            )
        elif self.type == "s3":
            return S3ConnectionConfig(
                id=self.id,
                name=self.name,
                bucket=self.config.get("bucket", ""),
                aws_access_key_id=self.config.get("aws_access_key_id"),
                aws_secret_access_key=self.config.get("aws_secret_access_key"),
                region=self.config.get("region"),
                endpoint=self.config.get("endpoint")
            )
        elif self.type == "postgres":
            return PostgresConnectionConfig(
                id=self.id,
                name=self.name,
                host=self.config.get("host", ""),
                port=self.config.get("port", ""),
                database=self.config.get("database", ""),
                username=self.config.get("username", ""),
                password=self.config.get("password", "")
            )
        else:
            return GenericConnectionConfig(
                id=self.id,
                name=self.name,
                type=self.type,
                config=self.config
            )
    
    @classmethod
    def from_specific_config(cls, config: Union[
        GrafanaConnectionConfig, 
        KubernetesConnectionConfig,
        S3ConnectionConfig,
        PostgresConnectionConfig,
        GenericConnectionConfig
    ]) -> 'ConnectionConfig':
        """Create from a type-specific configuration"""
        if isinstance(config, GrafanaConnectionConfig):
            return cls(
                id=config.id,
                name=config.name,
                type=config.type,
                config={
                    "url": config.url,
                    "api_key": config.api_key
                }
            )
        elif isinstance(config, KubernetesConnectionConfig):
            return cls(
                id=config.id,
                name=config.name,
                type=config.type,
                config={
                    "kubeconfig": config.kubeconfig,
                    "context": config.context if config.context else ""
                }
            )
        elif isinstance(config, S3ConnectionConfig):
            config_dict = {
                "bucket": config.bucket
            }
            
            # Add optional fields if they exist
            if config.aws_access_key_id:
                config_dict["aws_access_key_id"] = config.aws_access_key_id
            if config.aws_secret_access_key:
                config_dict["aws_secret_access_key"] = config.aws_secret_access_key
            if config.region:
                config_dict["region"] = config.region
            if config.endpoint:
                config_dict["endpoint"] = config.endpoint
                
            return cls(
                id=config.id,
                name=config.name,
                type=config.type,
                config=config_dict
            )
        elif isinstance(config, PostgresConnectionConfig):
            return cls(
                id=config.id,
                name=config.name,
                type=config.type,
                config={
                    "host": config.host,
                    "port": config.port,
                    "database": config.database,
                    "username": config.username,
                    "password": config.password
                }
            )
        elif isinstance(config, GenericConnectionConfig):
            return cls(
                id=config.id,
                name=config.name,
                type=config.type,
                config=config.config
            )
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

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
        self.active_mcp_servers = {}
    
    async def initialize(self) -> None:
        """Initialize and load connections"""
        correlation_id = str(uuid4())
        logger.info("Initializing ConnectionManager", extra={'correlation_id': correlation_id})
        # Load connections from storage
        await self._load_connections()
        logger.info("ConnectionManager initialized successfully", extra={'correlation_id': correlation_id})
    
    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        correlation_id = str(uuid4())
        logger.info("Closing ConnectionManager and cleaning up resources", extra={'correlation_id': correlation_id})
        # Stop any running MCP servers
        for server_id, server in self.active_mcp_servers.items():
            if hasattr(server, 'stop') and callable(server.stop):
                logger.debug(f"Stopping MCP server {server_id}", extra={'correlation_id': correlation_id})
                await server.stop()
        
        logger.info("ConnectionManager closed successfully", extra={'correlation_id': correlation_id})
    
    async def get_connection(self, connection_id: str) -> Optional[ConnectionConfig]:
        """Get a connection by ID"""
        logger.info(f"Getting connection {connection_id}")
        
        # First check the in-memory cache
        if connection_id in self.connections:
            return self.connections[connection_id]
            
        # If not found, query the database
        try:
            async with get_db_session() as session:
                repo = ConnectionRepository(session)
                db_connection = await repo.get_by_id(connection_id)
                
                if db_connection:
                    # Convert to ConnectionConfig and cache it
                    connection_dict = db_connection.to_dict()
                    connection = ConnectionConfig(
                        id=connection_dict["id"],
                        name=connection_dict["name"],
                        type=connection_dict["type"],
                        config=connection_dict["config"]
                    )
                    # Update the cache
                    self.connections[connection_id] = connection
                    return connection
                    
            # Not found in database either
            return None
        except Exception as e:
            logger.error(f"Error retrieving connection from database: {str(e)}")
            return None
    
    async def get_connections_by_type(self, connection_type: str) -> List[ConnectionConfig]:
        """
        Get all connections for a specific type
        
        Args:
            connection_type: The type of connection
            
        Returns:
            List of connections for the type
        """
        try:
            # Query the database directly
            async with get_db_session() as session:
                repo = ConnectionRepository(session)
                db_connections = await repo.get_by_type(connection_type)
                
                connections = []
                for db_conn in db_connections:
                    conn_dict = db_conn.to_dict()
                    connection = ConnectionConfig(
                        id=conn_dict["id"],
                        name=conn_dict["name"],
                        type=conn_dict["type"],
                        config=conn_dict["config"]
                    )
                    # Update cache
                    self.connections[conn_dict["id"]] = connection
                    connections.append(connection)
                
                return connections
        except Exception as e:
            logger.error(f"Error retrieving connections by type from database: {str(e)}")
            return []
    
    async def get_all_connections(self) -> List[Dict]:
        """
        Get all connections (with sensitive fields redacted)
        
        Returns:
            List of connection information
        """
        try:
            # Query the database directly
            async with get_db_session() as session:
                repo = ConnectionRepository(session)
                db_connections = await repo.get_all()
                
                # Update cache with any new connections
                for db_conn in db_connections:
                    conn_dict = db_conn.to_dict()
                    self.connections[conn_dict["id"]] = ConnectionConfig(
                        id=conn_dict["id"],
                        name=conn_dict["name"],
                        type=conn_dict["type"],
                        config=conn_dict["config"]
                    )
                
                return [
                    {
                        "id": conn.to_dict()["id"],
                        "name": conn.to_dict()["name"],
                        "type": conn.to_dict()["type"],
                        "config": self._redact_sensitive_fields(conn.to_dict()["config"])
                    }
                    for conn in db_connections
                ]
        except Exception as e:
            logger.error(f"Error retrieving all connections from database: {str(e)}")
            # Fallback to in-memory cache
            return [
                {
                    "id": conn.id,
                    "name": conn.name,
                    "type": conn.type,
                    "config": self._redact_sensitive_fields(conn.config)
                }
                for conn in self.connections.values()
            ]
    
    async def get_default_connection(self, connection_type: str) -> Optional[ConnectionConfig]:
        """
        Get the default connection for a given type
        
        Args:
            connection_type: The connection type to find
            
        Returns:
            The default connection or None if not found
        """
        # Check environment variables for default connection
        env_var_name = f"SHERLOG_CONNECTION_DEFAULT_{connection_type.upper()}"
        default_conn_name = os.environ.get(env_var_name)
        
        # If we have a default connection name from env vars, try to get it
        if default_conn_name:
            try:
                async with get_db_session() as session:
                    repo = ConnectionRepository(session)
                    db_connection = await repo.get_by_name_and_type(default_conn_name, connection_type)
                    
                    if db_connection:
                        conn_dict = db_connection.to_dict()
                        connection = ConnectionConfig(
                            id=conn_dict["id"],
                            name=conn_dict["name"],
                            type=conn_dict["type"],
                            config=conn_dict["config"]
                        )
                        # Update cache
                        self.connections[conn_dict["id"]] = connection
                        return connection
            except Exception as e:
                logger.error(f"Error finding default connection by name: {str(e)}")
        
        # Otherwise, check if we have a default connection stored in db
        try:
            async with get_db_session() as session:
                repo = ConnectionRepository(session)
                db_connection = await repo.get_default_connection(connection_type)
                
                if db_connection:
                    conn_dict = db_connection.to_dict()
                    connection = ConnectionConfig(
                        id=conn_dict["id"],
                        name=conn_dict["name"],
                        type=conn_dict["type"],
                        config=conn_dict["config"]
                    )
                    # Update cache
                    self.connections[conn_dict["id"]] = connection
                    # Update in-memory default tracking
                    self.default_connections[connection_type] = conn_dict["id"]
                    return connection
                    
                # If still not found, try to return the first connection of the type
                connections = await self.get_connections_by_type(connection_type)
                if connections:
                    return connections[0]
        except Exception as e:
            logger.error(f"Error retrieving default connection from database: {str(e)}")
        
        # No connection found
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
        correlation_id = str(uuid4())
        logger.info(f"Creating new {type} connection: {name}", extra={'correlation_id': correlation_id})
        
        # Validate the connection
        logger.debug(f"Validating connection {name}", extra={'correlation_id': correlation_id})
        is_valid, message = self._validate_and_prepare_connection(None, type, config)
        if not is_valid:
            logger.error(f"Connection validation failed for {name}: {message}", extra={'correlation_id': correlation_id})
            raise ValueError(message)
        
        # Create the connection in the database
        async with get_db_session() as session:
            repo = ConnectionRepository(session)
            db_connection = await repo.create(name, type, config)
            
            # Convert to ConnectionConfig model
            connection_dict = db_connection.to_dict()
            connection = ConnectionConfig(
                id=connection_dict["id"],
                name=connection_dict["name"],
                type=connection_dict["type"],
                config=connection_dict["config"]
            )
        
        # Add to in-memory connections
        self.connections[connection.id] = connection
        
        # If this is the first connection for this type, set it as default
        if not self.get_default_connection(type):
            logger.info(f"Setting {name} ({connection.id}) as default {type} connection", extra={'correlation_id': correlation_id})
            self.default_connections[type] = connection.id
            await self.set_default_connection(connection.id)
        
        logger.info(f"Successfully created connection {name} ({connection.id})", extra={'correlation_id': correlation_id})
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
        logger.info(f"Updating connection {connection_id} with new name: {name}")
        connection = await self.get_connection(connection_id)
        if not connection:
            logger.error(f"Connection not found for update: {connection_id}")
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Validate the connection
        logger.debug(f"Validating updated connection {name} ({connection_id})")
        is_valid, message = self._validate_and_prepare_connection(connection_id, connection.type, config)
        if not is_valid:
            logger.error(f"Connection validation failed for update {name} ({connection_id}): {message}")
            raise ValueError(message)
        
        # Update the connection in the database
        async with get_db_session() as session:
            repo = ConnectionRepository(session)
            db_connection = await repo.update(connection_id, name, config)
            if not db_connection:
                logger.error(f"Failed to update connection {connection_id} in database")
                raise ValueError(f"Failed to update connection: {connection_id}")
            
            # Convert to ConnectionConfig model
            connection_dict = db_connection.to_dict()
            updated_connection = ConnectionConfig(
                id=connection_dict["id"],
                name=connection_dict["name"],
                type=connection_dict["type"],
                config=connection_dict["config"]
            )
        
        # Update the in-memory connection
        self.connections[connection_id] = updated_connection
        
        logger.info(f"Successfully updated connection {name} ({connection_id})")
        return updated_connection
    
    async def delete_connection(self, connection_id: str) -> None:
        """
        Delete a connection
        
        Args:
            connection_id: The ID of the connection to delete
            
        Raises:
            ValueError: If the connection is not found
        """
        logger.info(f"Deleting connection {connection_id}")
        connection = self.get_connection(connection_id)
        if not connection:
            logger.error(f"Connection not found for deletion: {connection_id}")
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Delete from database
        async with get_db_session() as session:
            repo = ConnectionRepository(session)
            success = await repo.delete(connection_id)
            if not success:
                logger.error(f"Failed to delete connection {connection_id} from database")
                raise ValueError(f"Failed to delete connection: {connection_id}")
        
        # Remove from in-memory connections
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        # Update default connections if needed
        for conn_type, default_id in list(self.default_connections.items()):
            if default_id == connection_id:
                logger.info(f"Removing {connection_id} as default connection for type {conn_type}")
                # Find another connection for this type
                connections = await self.get_connections_by_type(conn_type)
                if connections:
                    new_default = connections[0].id
                    logger.info(f"Setting new default connection for type {conn_type}: {new_default}")
                    self.default_connections[conn_type] = new_default
                    await self.set_default_connection(new_default)
                else:
                    logger.info(f"No remaining connections for type {conn_type}, removing default")
                    if conn_type in self.default_connections:
                        del self.default_connections[conn_type]
        
        logger.info(f"Successfully deleted connection {connection_id}")
    
    async def set_default_connection(self, connection_id: str) -> None:
        """
        Set a connection as the default for its type
        
        Args:
            connection_id: The ID of the connection to set as default
            
        Raises:
            ValueError: If the connection is not found
        """
        connection = await self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Set as default in memory
        self.default_connections[connection.type] = connection_id
        
        # Set as default in database
        async with get_db_session() as session:
            repo = ConnectionRepository(session)
            await repo.set_default_connection(connection.type, connection_id)
    
    async def test_connection(self, connection_type: str, config: Dict) -> bool:
        """
        Test a connection configuration
        
        Args:
            connection_type: The type of connection
            config: The connection configuration to test
            
        Returns:
            True if the connection is valid, False otherwise
        """
        logger.debug(f"Testing {connection_type} connection")
        try:
            # Basic validation for each type
            if connection_type == "grafana":
                if not config.get("url") or not config.get("api_key"):
                    logger.warning("Grafana connection missing required fields: url or api_key")
                    return False
                
                # Test connection by contacting MCP server
                from backend.mcp.manager import get_mcp_server_manager
                
                # Create a temporary connection config to test
                test_conn = ConnectionConfig(
                    id=str(uuid4()),
                    name="Test Connection", 
                    type="grafana",
                    config=config
                )
                
                # Start MCP server for test connection
                mcp_manager = get_mcp_server_manager()
                try:
                    server_addresses = await mcp_manager.start_mcp_servers([test_conn])
                    if test_conn.id in server_addresses:
                        # Successfully started MCP server
                        await mcp_manager.stop_server(test_conn.id)
                        return True
                    return False
                except Exception as e:
                    logger.error(f"Failed to test Grafana connection: {str(e)}")
                    return False
            
            elif connection_type == "kubernetes":
                if not config.get("kubeconfig"):
                    logger.warning("Kubernetes connection missing required field: kubeconfig")
                    return False
                    
                # TODO: Verify the kubeconfig is valid
                return True
            
            elif connection_type == "s3":
                if not config.get("bucket"):
                    logger.warning("S3 connection missing required field: bucket")
                    return False
                    
                # TODO: Verify the S3 bucket is accessible
                return True
            
            else:
                logger.error(f"Unknown connection type: {connection_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            return False
    
    def _validate_and_prepare_connection(self, connection_id, connection_type, config) -> Tuple[bool, str]:
        """
        Validate a connection configuration
        """
        try:
            if connection_type == "grafana":
                if not config.get("url") or not config.get("api_key"):
                    logger.warning("Grafana connection missing required fields: url, api_key")
                    return False, "Grafana connection missing required fields: url, api_key"
                
                # TODO: Actually test the connection to Grafana
                return True, "Connection validated"
            elif connection_type == "kubernetes":
                if not config.get("kubeconfig"):
                    logger.warning("Kubernetes connection missing required field: kubeconfig")
                    return False, "Kubernetes connection missing required field: kubeconfig"
                    
                # TODO: Verify the kubeconfig is valid
                return True, "Connection validated"
            elif connection_type == "s3":
                if not config.get("bucket"):
                    logger.warning("S3 connection missing required field: bucket")
                    return False, "S3 connection missing required field: bucket"
                    
                # TODO: Verify the S3 bucket is accessible
                return True, "Connection validated"
            else:
                return True, "Connection type not validated"
                
        except Exception as e:
            logger.error(f"Error validating connection: {e}")
            return False, f"Error validating connection: {str(e)}"

    async def get_connection_schema(self, connection_id: str) -> Dict:
        """
        Get the schema for a connection (tables, fields, etc.)
        
        Args:
            connection_id: The connection ID
            
        Returns:
            The connection schema or an error message
        """
        connection = await self.get_connection(connection_id)
        if not connection:
            return {"error": f"Connection not found: {connection_id}"}
            
        try:
            if connection.type == "grafana":
                # Fetch dashboards and datasources from Grafana
                return {"message": "Grafana schema retrieval not implemented yet"}
            elif connection.type == "kubernetes":
                # Fetch list of resources 
                return {"message": "Kubernetes schema retrieval not implemented yet"}
            else:
                return {"message": f"Schema retrieval not implemented for {connection.type} connections"}
        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            return {"error": f"Error retrieving schema: {str(e)}"}
    
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
    
    async def _load_connections(self) -> None:
        """Load connections from storage"""
        correlation_id = str(uuid4())
        try:
            # Always use database storage
            logger.info("Loading connections from database storage", extra={'correlation_id': correlation_id})
            await self._load_connections_from_db()
            logger.info(f"Successfully loaded {len(self.connections)} connections", extra={'correlation_id': correlation_id})
        except Exception as e:
            logger.error(f"Error loading connections: {str(e)}", extra={'correlation_id': correlation_id})
            raise
    
    async def _load_connections_from_db(self) -> None:
        """Load connections from database"""
        async with get_db_session() as session:
            repo = ConnectionRepository(session)
            
            # Load all connections
            db_connections = await repo.get_all()
            self.connections = {
                conn.to_dict()["id"]: ConnectionConfig(
                    id=conn.to_dict()["id"],
                    name=conn.to_dict()["name"],
                    type=conn.to_dict()["type"],
                    config=conn.to_dict()["config"]
                )
                for conn in db_connections
            }
            
            # Load default connections
            for conn_type in set(conn.to_dict()["type"] for conn in db_connections):
                default_conn = await repo.get_default_connection(conn_type)
                if default_conn:
                    self.default_connections[conn_type] = default_conn.to_dict()["id"]
    
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
            return {"connections": {}, "defaults": {}}
        
        # Load from file (use async file operations)
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            print(f"Error loading connections from file: {e}")
            return {"connections": {}, "defaults": {}}
    
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
            "defaults": default_connections
        }

    def can_execute_sql(self, connection: ConnectionConfig) -> bool:
        """
        Check if a connection can execute SQL
        
        Args:
            connection: The connection to check
            
        Returns:
            True if the connection can execute SQL, False otherwise
        """
        if connection.type == "sql":
            return True
        return False