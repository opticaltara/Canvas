"""
Connection Manager Service

Manages connections to data sources via MCP servers
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from uuid import uuid4
import asyncio

from pydantic import BaseModel, Field
import aiofiles

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from backend.config import get_settings
from backend.db.database import get_db_session
from backend.db.repositories import ConnectionRepository

from backend.core.types import MCPToolInfo

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

# Define the command/args template for GitHub MCP stdio
GITHUB_MCP_COMMAND = ["docker", "run", "-i", "--rm"]
GITHUB_MCP_ARGS_TEMPLATE = [
    # PAT environment variable will be added dynamically during validation/agent execution
    "ghcr.io/github/github-mcp-server"
]
GRAFANA_MCP_COMMAND = ["mcp-grafana"] # New Grafana command

class BaseConnectionConfig(BaseModel):
    """Base connection configuration"""
    id: str
    name: str
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)

class GrafanaConnectionConfig(BaseConnectionConfig):
    """Grafana stdio connection configuration"""
    type: str = "grafana"
    # url and api_key are now stored within the base 'config' dict, specifically under 'env'
    # config field is inherited from BaseConnectionConfig

class GithubConnectionConfig(BaseConnectionConfig):
    """GitHub connection configuration (uses stdio MCP server)"""
    type: str = "github"
    # Stores stdio command/args template in base 'config', PAT is handled separately within 'config'
    # config field is inherited from BaseConnectionConfig

# Generic fallback for other types
class GenericConnectionConfig(BaseConnectionConfig):
    """Generic connection configuration"""
    config: Dict[str, str] = Field(default_factory=dict)

class ConnectionConfig(BaseConnectionConfig):
    """Connection configuration"""
    def to_specific_config(self) -> Union[
        GrafanaConnectionConfig, 
        GithubConnectionConfig,
        GenericConnectionConfig
    ]:
        """Convert to a type-specific configuration"""
        if self.type == "grafana":
            # Returns the Grafana config, the base 'config' dict holds the details (command, args, env, url, api_key)
            return GrafanaConnectionConfig(
                id=self.id,
                name=self.name,
                config=self.config
            )
        elif self.type == "github":
            # Returns the GitHub config, the base 'config' dict holds the details (command, args template, pat)
            return GithubConnectionConfig(
                id=self.id,
                name=self.name,
                config=self.config # Contains command and args template now
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
        GithubConnectionConfig,
        GenericConnectionConfig
    ]) -> 'ConnectionConfig':
        """Create from a type-specific configuration"""
        # For stdio types (Grafana, GitHub) and Generic, the relevant details are already in the specific config's 'config' dict
        if isinstance(config, (GrafanaConnectionConfig, GithubConnectionConfig, GenericConnectionConfig)):
             return cls(
                id=config.id,
                name=config.name,
                type=config.type,
                config=config.config # The specific config's 'config' dict holds everything needed for the base config
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
        logger.info("Closing ConnectionManager", extra={'correlation_id': correlation_id})
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
                        "config": self._redact_sensitive_fields(conn.to_dict()["config"], conn_dict["type"])
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
                    "config": self._redact_sensitive_fields(conn.config, conn.type)
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
    
    async def create_connection(self, name: str, type: str, config: Dict, **kwargs) -> ConnectionConfig:
        """
        Create a new connection
        
        Args:
            name: The name of the connection
            type: The type of connection
            config: The base connection configuration (may be empty for some types)
            **kwargs: Additional type-specific arguments (e.g., github_pat)
            
        Returns:
            The created connection
            
        Raises:
            ValueError: If connection validation fails
        """
        correlation_id = str(uuid4())
        logger.info(f"Creating new {type} connection: {name}", extra={'correlation_id': correlation_id})

        # Validate the connection and prepare final config
        logger.info(f"Validating connection {name}", extra={'correlation_id': correlation_id})
        
        # Pass kwargs for type-specific validation data (like PAT)
        is_valid, message, prepared_config = await self._validate_and_prepare_connection(
            connection_id=None, 
            connection_type=type, 
            initial_config=config, 
            **kwargs 
        )
        
        if not is_valid:
            logger.error(f"Connection validation failed for {name}: {message}", extra={'correlation_id': correlation_id})
            raise ValueError(message)

        # Create the connection in the database with the prepared config
        async with get_db_session() as session:
            repo = ConnectionRepository(session)
            # Use prepared_config which contains validated/structured data (like command/args for github)
            db_connection = await repo.create(name, type, prepared_config) 
            
            # Convert to ConnectionConfig model
            connection_dict = db_connection.to_dict()
            connection = ConnectionConfig(
                id=connection_dict["id"],
                name=connection_dict["name"],
                type=connection_dict["type"],
                config=connection_dict["config"] # This is the prepared_config
            )

        # Add to in-memory connections
        self.connections[connection.id] = connection

        # If this is the first connection for this type, set it as default
        # Check default *after* potential creation
        existing_default = await self.get_default_connection(type)
        if not existing_default:
            logger.info(f"Setting {name} ({connection.id}) as default {type} connection", extra={'correlation_id': correlation_id})
            self.default_connections[type] = connection.id
            await self.set_default_connection(connection.id)

        logger.info(f"Successfully created connection {name} ({connection.id})", extra={'correlation_id': correlation_id})
        return connection
    
    async def update_connection(self, connection_id: str, name: str, config: Dict, **kwargs) -> ConnectionConfig:
        """
        Update an existing connection
        
        Args:
            connection_id: The ID of the connection to update
            name: The new name of the connection
            config: The new base connection configuration
            **kwargs: Additional type-specific arguments (e.g., github_pat for re-validation)
            
        Returns:
            The updated connection
            
        Raises:
            ValueError: If the connection is not found or validation fails
        """
        correlation_id = str(uuid4())
        logger.info(f"Updating connection {connection_id} with new name: {name}", extra={'correlation_id': correlation_id})
        connection = await self.get_connection(connection_id)
        if not connection:
            logger.error(f"Connection not found for update: {connection_id}", extra={'correlation_id': correlation_id})
            raise ValueError(f"Connection not found: {connection_id}")

        # Validate the connection and prepare final config
        logger.info(f"Validating updated connection {name} ({connection_id})", extra={'correlation_id': correlation_id})
        
        is_valid, message, prepared_config = await self._validate_and_prepare_connection(
            connection_id=connection_id, 
            connection_type=connection.type, 
            initial_config=config, 
            **kwargs # Pass PAT if provided for re-validation
        )
        
        if not is_valid:
            logger.error(f"Connection validation failed for update {name} ({connection_id}): {message}", extra={'correlation_id': correlation_id})
            raise ValueError(message)

        # Update the connection in the database with the prepared config
        async with get_db_session() as session:
            repo = ConnectionRepository(session)
            db_connection = await repo.update(connection_id, name, prepared_config)
            if not db_connection:
                logger.error(f"Failed to update connection {connection_id} in database", extra={'correlation_id': correlation_id})
                raise ValueError(f"Failed to update connection: {connection_id}")

            # Convert to ConnectionConfig model
            connection_dict = db_connection.to_dict()
            updated_connection = ConnectionConfig(
                id=connection_dict["id"],
                name=connection_dict["name"],
                type=connection_dict["type"],
                config=connection_dict["config"] # This is the prepared_config
            )

        # Update the in-memory connection
        self.connections[connection_id] = updated_connection

        logger.info(f"Successfully updated connection {name} ({connection_id})", extra={'correlation_id': correlation_id})
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
    
    async def test_connection(self, connection_type: str, config: Dict, **kwargs) -> Tuple[bool, str]:
        """
        Test a connection configuration without saving it.
        
        Args:
            connection_type: The type of connection
            config: The connection configuration to test
            **kwargs: Additional type-specific arguments (e.g., github_pat)
            
        Returns:
            Tuple (is_valid: bool, message: str)
        """
        logger.info(f"Testing {connection_type} connection configuration")
        try:
            # Delegate to the validation logic
            is_valid, message, _ = await self._validate_and_prepare_connection(
                connection_id=None, # No ID for testing
                connection_type=connection_type,
                initial_config=config,
                is_test_run=True, # Indicate this is just a test
                **kwargs # Pass PAT etc.
            )
            return is_valid, message
        except Exception as e:
            logger.error(f"Error testing connection configuration: {str(e)}")
            return False, f"Error testing connection: {str(e)}"

    async def _validate_and_prepare_connection(
        self, 
        connection_id: Optional[str], 
        connection_type: str, 
        initial_config: Dict,
        is_test_run: bool = False, # Flag to indicate if this is just a test
        **kwargs
    ) -> Tuple[bool, str, Dict]:
        """
        Validate a connection configuration and prepare it for saving.
        
        Args:
            connection_id: ID if updating, None if creating or testing.
            connection_type: Type of connection.
            initial_config: Configuration dictionary from the request.
            is_test_run: If True, don't prepare for saving, just validate.
            **kwargs: Additional type-specific arguments (e.g., github_pat).

        Returns:
            Tuple (is_valid: bool, message: str, prepared_config: Dict)
            prepared_config is the config to be saved to the DB.
        """
        correlation_id = kwargs.get('correlation_id', str(uuid4()))
        logger.info(f"Validating connection type {connection_type}", extra={'correlation_id': correlation_id})
        
        try:
            if connection_type == "grafana":
                grafana_url = kwargs.get("grafana_url")
                grafana_api_key = kwargs.get("grafana_api_key")

                # Attempt to retrieve from existing config if not provided (e.g., during a test or update without changes)
                if not grafana_url and connection_id:
                    existing_conn = await self.get_connection(connection_id)
                    if existing_conn and existing_conn.config.get("grafana_url"):
                         grafana_url = existing_conn.config["grafana_url"]
                         logger.info("Using existing Grafana URL for validation.", extra={'correlation_id': correlation_id})
                if not grafana_api_key and connection_id:
                     existing_conn = await self.get_connection(connection_id)
                     # Check both the direct key and inside env for robustness
                     if existing_conn and existing_conn.config.get("grafana_api_key"):
                         grafana_api_key = existing_conn.config["grafana_api_key"]
                         logger.info("Using existing Grafana API Key (top-level) for validation.", extra={'correlation_id': correlation_id})
                     elif existing_conn and isinstance(existing_conn.config.get("env"), dict) and existing_conn.config["env"].get("GRAFANA_API_KEY"):
                         grafana_api_key = existing_conn.config["env"]["GRAFANA_API_KEY"]
                         logger.info("Using existing Grafana API Key (from env) for validation.", extra={'correlation_id': correlation_id})


                if not grafana_url or not grafana_api_key:
                    raise ValueError("Grafana URL and API Key are required.")

                is_valid, message = await self._validate_grafana_stdio(grafana_url, grafana_api_key, correlation_id)
                if not is_valid:
                    return False, message, {}

                # Prepare config for saving (command, args, env, and original url/key for reference)
                prepared_config = {
                    "mcp_command": GRAFANA_MCP_COMMAND,
                    "mcp_args": [], # Grafana MCP takes no command line args
                    "env": {
                        "GRAFANA_URL": grafana_url,
                        "GRAFANA_API_KEY": grafana_api_key # Needs redaction before display
                    },
                    # Store original values for easier reference/updates, mark api_key for redaction
                    "grafana_url": grafana_url,
                    "grafana_api_key": grafana_api_key
                }
                logger.warning("Storing Grafana API Key directly in config. Needs encryption/redaction!", extra={'correlation_id': correlation_id})
                return True, "Grafana stdio connection validated successfully", prepared_config

            elif connection_type == "github":
                # GitHub stdio validation
                github_pat = kwargs.get("github_personal_access_token")
                existing_config = None
                
                if not github_pat:
                    # If updating/testing and PAT not provided, try to get it from existing config
                    if connection_id:
                        existing_conn = await self.get_connection(connection_id)
                        if existing_conn and existing_conn.config.get("github_pat"):
                            # Ensure retrieved PAT is a string
                            potential_pat = existing_conn.config.get("github_pat")
                            if isinstance(potential_pat, str):
                                github_pat = potential_pat
                            else:
                                # Log error if PAT exists but is not a string?
                                logger.error(f"Stored github_pat for {connection_id} is not a string.", extra={'correlation_id': correlation_id})
                                # Proceed as if PAT was not found
                                pass 
                            existing_config = existing_conn.config # Keep existing config if validation passes
                            logger.info("Using existing GitHub PAT from DB for validation.", extra={'correlation_id': correlation_id})
                        else:
                            # If no PAT provided and none exists in DB, fail validation unless just updating name
                            if not is_test_run and not kwargs.get("config"): # Allow name-only updates without PAT
                                logger.info("GitHub PAT not provided or found, but allowing name-only update.", extra={'correlation_id': correlation_id})
                                existing_conn_config = existing_conn.config if existing_conn else {}
                                return True, "GitHub PAT not provided, name update allowed.", existing_conn_config
                            else:
                                msg = "GitHub connection requires Personal Access Token for validation (not provided and not found in existing config)."
                                logger.warning(msg, extra={'correlation_id': correlation_id})
                                return False, msg, {}
                    else: # Create or Test requires PAT if not updating
                        msg = "GitHub connection requires Personal Access Token for validation."
                        logger.warning(msg, extra={'correlation_id': correlation_id})
                        return False, msg, {}

                # Check if github_pat is now a valid string before validating
                if not isinstance(github_pat, str) or not github_pat:
                    # This case should ideally be caught earlier, but double-check
                    msg = "Failed to obtain a valid GitHub PAT for validation."
                    logger.error(msg, extra={'correlation_id': correlation_id})
                    return False, msg, {}
                
                # Perform validation using the obtained PAT
                is_valid, message = await self._validate_github_stdio(github_pat, correlation_id)
                
                if not is_valid:
                    return False, message, {}
                
                # If validation passed using an existing PAT, return the existing config
                if existing_config is not None:
                    logger.info("Validation with existing PAT successful, returning existing config.", extra={'correlation_id': correlation_id})
                    return True, message, existing_config
                    
                # Prepare config for saving (command, args template, AND PAT)
                # !!! SECURITY WARNING: Storing PAT plaintext is not recommended! Encrypt in production. !!!
                prepared_config = {
                    "mcp_command": GITHUB_MCP_COMMAND,
                    "mcp_args_template": GITHUB_MCP_ARGS_TEMPLATE,
                    "github_pat": github_pat # Store PAT directly (NEEDS ENCRYPTION IN PROD)
                }
                logger.warning("Storing GitHub PAT directly in config. Needs encryption!", extra={'correlation_id': correlation_id})
                return True, "GitHub stdio connection validated successfully", prepared_config
                
            elif connection_type == "python":
                logger.info("No specific validation for Python connection type.", extra={'correlation_id': correlation_id})
                return True, "Python connection validated", initial_config # Return initial config

            else:
                # Allow generic/unknown through without specific validation for now
                logger.warning(f"No specific validation logic for connection type: {connection_type}", extra={'correlation_id': correlation_id})
                return True, f"Connection validated (or validation not implemented for type {connection_type})", initial_config # Return initial config

        except Exception as e:
            logger.error(f"Error validating connection: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
            return False, f"Error validating connection: {str(e)}", {}

    async def _validate_github_stdio(self, github_pat: str, correlation_id: str) -> Tuple[bool, str]:
        """
        Validate GitHub connection by running a test MCP stdio session.
        
        Args:
            github_pat: The GitHub Personal Access Token.
            correlation_id: For logging.

        Returns:
            Tuple (is_valid: bool, message: str)
        """
        logger.info("Performing GitHub stdio validation", extra={'correlation_id': correlation_id})
        
        # Construct arguments for docker run, including the -e flag for the PAT
        # Args should be: ['run', '-i', '--rm', '-e', f'GITHUB_PERSONAL_ACCESS_TOKEN={pat}', 'ghcr.io/github/github-mcp-server']
        dynamic_env_arg = ["-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={github_pat}"]
        full_args = GITHUB_MCP_COMMAND[1:] + dynamic_env_arg + GITHUB_MCP_ARGS_TEMPLATE
        command = GITHUB_MCP_COMMAND[0]
        
        # Do not pass env here; the -e arg handles it for the container
        params = StdioServerParameters(
            command=command,
            args=full_args
        )
        
        logger.info(f"Created StdioServerParameters for GitHub. Command: '{params.command}', Args: {params.args}")
        return True, "GitHub stdio connection validated successfully."

    async def _validate_grafana_stdio(self, grafana_url: str, grafana_api_key: str, correlation_id: str) -> Tuple[bool, str]:
        """Validate Grafana connection by running a test MCP stdio session."""
        logger.info("Performing Grafana stdio validation", extra={'correlation_id': correlation_id})

        env_vars = {
            "GRAFANA_URL": grafana_url,
            "GRAFANA_API_KEY": grafana_api_key
        }

        # mcp-grafana takes no command line args, config is via env vars
        server_params = StdioServerParameters(
            command=GRAFANA_MCP_COMMAND[0], # 'mcp-grafana'
            args=[],
            env=env_vars
        )

        try:
            logger.info(f"Attempting stdio_client connection with Grafana params: {server_params}", extra={'correlation_id': correlation_id})
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.info("Initializing MCP session for Grafana validation...", extra={'correlation_id': correlation_id})
                    # Set a timeout for initialization
                    init_task = asyncio.create_task(session.initialize())
                    try:
                        await asyncio.wait_for(init_task, timeout=30.0) # 30 second timeout
                    except asyncio.TimeoutError:
                         logger.error("Timeout during Grafana MCP session initialization", extra={'correlation_id': correlation_id})
                         return False, "Validation failed: Timeout initializing Grafana MCP session. Check mcp-grafana binary, URL, API key, and network."
                    except Exception as init_err:
                        logger.error(f"Error during Grafana MCP session initialization: {init_err}", extra={'correlation_id': correlation_id})
                        # Provide more specific error if possible, e.g., connection refused
                        return False, f"Validation failed: Error initializing Grafana MCP session: {init_err}"

                    logger.info("Listing tools via MCP session for Grafana validation...", extra={'correlation_id': correlation_id})
                    # Set a timeout for list_tools
                    list_tools_task = asyncio.create_task(session.list_tools())
                    try:
                         tools = await asyncio.wait_for(list_tools_task, timeout=15.0) # 15 second timeout
                    except asyncio.TimeoutError:
                         logger.error("Timeout during Grafana MCP list_tools call", extra={'correlation_id': correlation_id})
                         return False, "Validation failed: Timeout listing tools via Grafana MCP. Check server logs."
                    except Exception as list_err:
                        logger.error(f"Error during Grafana MCP list_tools call: {list_err}", extra={'correlation_id': correlation_id})
                        return False, f"Validation failed: Error listing tools via Grafana MCP: {list_err}"

                    # Basic check: Did we get tools? Grafana MCP should provide some.
                    if tools and tools.tools:
                        logger.info(f"Grafana stdio validation successful. Found {len(tools.tools)} tools.", extra={'correlation_id': correlation_id})
                        return True, "Grafana stdio connection validated successfully."
                    else:
                        logger.warning("Grafana stdio validation potentially failed: No tools listed.", extra={'correlation_id': correlation_id})
                        # Consider this a failure, as we expect tools (e.g., query_prometheus, query_loki)
                        return False, "Validation failed: Grafana MCP server connected but reported no tools. Check Grafana setup or mcp-grafana logs."

        except FileNotFoundError:
             logger.error(f"Validation failed: Command '{GRAFANA_MCP_COMMAND[0]}' not found. Was it installed correctly in the Docker image?", extra={'correlation_id': correlation_id})
             return False, f"Validation failed: Required command '{GRAFANA_MCP_COMMAND[0]}' not found. Ensure it's installed and in PATH."
        except Exception as e:
            # Catch potential OS errors during process start e.g. permission denied
            logger.error(f"Error during Grafana stdio validation process startup or communication: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
            return False, f"Validation failed due to system error: {str(e)}"
        # Fallback if logic doesn't return explicitly
        return False, "Unknown validation error for Grafana stdio."

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
            else:
                return {"message": f"Schema retrieval not implemented for {connection.type} connections"}
        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            return {"error": f"Error retrieving schema: {str(e)}"}
    
    def _redact_sensitive_fields(self, config: Dict, connection_type: Optional[str] = None) -> Dict:
        """
        Redact sensitive fields from a connection configuration
        
        Args:
            config: The configuration to redact
            connection_type: The type of connection (for context, e.g., github should show PAT)
            
        Returns:
            The redacted configuration
        """
        sensitive_fields = [
            "password", "secret", "key", "token", "api_key", 
            "aws_secret_access_key", "private_key",
            "github_pat", # Add github_pat to the sensitive list
            "github_personal_access_token" # Keep this for redacting input if ever present
        ]
        
        redacted_config = config.copy()
        
        # Remove PAT if present, regardless of type, before general redaction
        if "github_pat" in redacted_config:
            redacted_config["github_pat"] = "********"
        if "github_personal_access_token" in redacted_config: # Also redact this input key if present
            redacted_config["github_personal_access_token"] = "********"

        # General redaction based on key names
        for key in list(redacted_config.keys()): # Iterate over keys list for safe removal
            # Skip keys already redacted
            if redacted_config[key] == "********":
                continue
            if isinstance(redacted_config[key], str) and any(sensitive_part in key.lower() for sensitive_part in sensitive_fields):
                redacted_config[key] = "********"

        # Type-specific redaction (e.g., nested keys)
        if connection_type == "github": # No change needed here if github_pat is already covered
            pass
        elif connection_type == "grafana":
            # Redact the top-level key if present
            if "grafana_api_key" in redacted_config:
                 redacted_config["grafana_api_key"] = "***REDACTED***"
            # Redact the key within the 'env' dict
            if "env" in redacted_config and isinstance(redacted_config["env"], dict):
                 if "GRAFANA_API_KEY" in redacted_config["env"]:
                     # Ensure we modify a copy if 'env' might be shared or reused
                     if redacted_config["env"] is config.get("env"): # Check if it's the same dict instance
                          redacted_config["env"] = redacted_config["env"].copy()
                     redacted_config["env"]["GRAFANA_API_KEY"] = "***REDACTED***"

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

    # --- Tool Discovery --- 
    async def get_tools_for_connection_type(self, connection_type: str) -> List[MCPToolInfo]:
        """
        Retrieves the list of available tools for a given connection type by querying its MCP.
        Returns basic tool info including the raw inputSchema.
        """
        correlation_id = str(uuid4())
        logger.info(f"Fetching tools for connection type: {connection_type}", extra={'correlation_id': correlation_id})
        
        # --- Start Implementation (Modified for MCPToolInfo) --- 
        connection_config = await self.get_default_connection(connection_type)
        if not connection_config: 
            logger.warning(f"No default connection found for {connection_type}", extra={'correlation_id': correlation_id})
            return []
        config_dict = connection_config.config
        if not isinstance(config_dict, dict): raise ValueError(f"Invalid config for {connection_config.id}")

        mcp_command_list = config_dict.get("mcp_command")
        mcp_args_template = config_dict.get("mcp_args_template", []) 
        mcp_env = config_dict.get("env", {})
        if not mcp_command_list or not isinstance(mcp_command_list, list) or not mcp_command_list: raise ValueError(f"Missing MCP command config for {connection_type}")
        command = mcp_command_list[0]
        base_args = mcp_command_list[1:]
        full_args = base_args + mcp_args_template

        # Handle GitHub PAT injection (same as before)
        if connection_type == "github":
            github_pat = config_dict.get("github_pat")
            if not github_pat or not isinstance(github_pat, str): raise ValueError("Missing GitHub PAT")
            if command == "docker" and "run" in base_args:
                pat_arg_index = -1
                if mcp_args_template: 
                    try: pat_arg_index = full_args.index(mcp_args_template[0])
                    except ValueError: pass 
                if pat_arg_index != -1: full_args.insert(pat_arg_index, f"GITHUB_PERSONAL_ACCESS_TOKEN={github_pat}"); full_args.insert(pat_arg_index, "-e")
                else: 
                    if mcp_args_template: full_args = base_args + ["-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={github_pat}"] + mcp_args_template
                    else: logger.warning("Cannot determine where to insert GitHub PAT env var, appending.", extra={'correlation_id': correlation_id}); full_args.extend(["-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={github_pat}"])
            else: logger.warning("GitHub not using 'docker run', PAT injection might fail.", extra={'correlation_id': correlation_id}); mcp_env["GITHUB_PERSONAL_ACCESS_TOKEN"] = github_pat

        server_params = StdioServerParameters(command=command, args=full_args, env=mcp_env)
        
        try:
            logger.info(f"Attempting MCP connection for {connection_type} with params: {server_params}", extra={'correlation_id': correlation_id})
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.info(f"Initializing MCP session for {connection_type} tool discovery...", extra={'correlation_id': correlation_id})
                    try: await asyncio.wait_for(session.initialize(), timeout=30.0)
                    except asyncio.TimeoutError: raise ValueError(f"Timeout initializing MCP session for {connection_type}")
                    except Exception as init_err: raise ValueError(f"Error initializing MCP session for {connection_type}: {init_err}")

                    logger.info(f"Listing tools via MCP session for {connection_type}...", extra={'correlation_id': correlation_id})
                    try:
                         response = await asyncio.wait_for(session.list_tools(), timeout=15.0) 
                         
                         # --- Simplified MCP Tool Parsing --- 
                         parsed_tools: List[MCPToolInfo] = [] # Updated type
                         mcp_tools_list = getattr(response, 'tools', [])
                         if not isinstance(mcp_tools_list, list): 
                             logger.warning(f"MCP list_tools response for {connection_type} lacked a list in 'tools' attribute.", extra={'correlation_id': correlation_id}); mcp_tools_list = []

                         for mcp_tool in mcp_tools_list:
                             try:
                                 # Extract basic info + raw schema
                                 tool_name = getattr(mcp_tool, 'name', 'Unknown Tool')
                                 tool_description = getattr(mcp_tool, 'description', None)
                                 input_schema = getattr(mcp_tool, 'inputSchema', {})
                                 
                                 # Ensure inputSchema is a dict
                                 if not isinstance(input_schema, dict):
                                     logger.warning(f"Tool '{tool_name}' has invalid inputSchema format (not a dict), skipping.", extra={'correlation_id': correlation_id})
                                     continue
                                 
                                 # Create MCPToolInfo instance
                                 tool_info = MCPToolInfo(
                                     name=tool_name,
                                     description=tool_description,
                                     inputSchema=input_schema # Store the raw schema
                                 )
                                 parsed_tools.append(tool_info)
                             except Exception as tool_parse_err:
                                 logger.error(f"Skipping tool due to error: {tool_parse_err}", extra={'correlation_id': correlation_id}, exc_info=True)
                                 continue # Skip this tool if basic extraction fails
                         # --- End Simplified Parsing --- 
                         
                         logger.info(f"Retrieved {len(parsed_tools)} tools info for {connection_type}", extra={'correlation_id': correlation_id})
                         return parsed_tools
                         
                    except asyncio.TimeoutError: raise ValueError(f"Timeout listing tools via MCP for {connection_type}")
                    except Exception as list_err: raise ValueError(f"Error listing tools via MCP for {connection_type}: {list_err}")
        except FileNotFoundError: raise ValueError(f"Command '{command}' not found for {connection_type}")
        except Exception as e: raise ValueError(f"Failed to communicate with MCP for {connection_type}: {str(e)}")
    # --- End Tool Discovery ---

# --- Helper Function for GitHub Stdio Params --- 

async def get_github_stdio_params() -> Optional[StdioServerParameters]:
    """
    Fetches the default GitHub connection and constructs StdioServerParameters.

    Returns:
        StdioServerParameters instance if successful, None otherwise.
    """
    logger = logging.getLogger("connection_manager") # Use appropriate logger
    try:
        connection_manager = get_connection_manager() # Get manager instance
        github_conn = await connection_manager.get_default_connection("github")
        
        if not github_conn:
            logger.error("No default GitHub connection found.")
            return None
        
        config = github_conn.config
        if not isinstance(config, dict):
            logger.error(f"GitHub connection {github_conn.id} config is not a dictionary: {type(config)}")
            return None

        command_list = config.get("mcp_command")
        args_template = config.get("mcp_args_template", [])
        pat = config.get("github_pat")
        
        if not command_list or not isinstance(command_list, list) or not command_list:
            logger.error(f"Invalid or missing 'mcp_command' in GitHub connection {github_conn.id}")
            return None
        if not args_template or not isinstance(args_template, list):
             logger.warning(f"Missing or invalid 'mcp_args_template' in GitHub connection {github_conn.id}. Using empty list.")
             args_template = []
        if not pat or not isinstance(pat, str):
            logger.error(f"Invalid or missing 'github_pat' in GitHub connection {github_conn.id}")
            return None
            
        dynamic_env_arg = ["-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={pat}"]
        full_args = command_list[1:] + dynamic_env_arg + args_template
        command = command_list[0]
        
        params = StdioServerParameters(
            command=command,
            args=full_args
        )
        
        logger.info(f"Created StdioServerParameters for GitHub. Command: '{params.command}', Args: {params.args}")
        return params
        
    except Exception as e:
        logger.error(f"Error getting/configuring GitHub stdio parameters: {e}", exc_info=True)
        return None