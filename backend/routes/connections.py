"""
Connection API Endpoints
"""

from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.services.connection_manager import (
    ConnectionManager, 
    get_connection_manager,
    GrafanaConnectionConfig,
    GenericConnectionConfig,
    BaseConnectionConfig,
    GithubConnectionConfig
)
import logging
from backend.core.types import MCPToolInfo

# Initialize logger
connection_logger = logging.getLogger("routes.connections")

# Add correlation ID filter to the connection logger
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

connection_logger.addFilter(CorrelationIdFilter())

# Ensure at least a basic console handler is attached if none exists
# This is a safeguard in case this module is imported before logging is configured
if not connection_logger.handlers:
    # Check if logger has a parent with handlers
    has_parent_handlers = connection_logger.parent and hasattr(connection_logger.parent, 'handlers') and connection_logger.parent.handlers
    
    if not has_parent_handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        connection_logger.addHandler(console_handler)
        connection_logger.setLevel(logging.INFO)

router = APIRouter()


class ConnectionConfig(BaseModel):
    """Configuration for a connection"""
    id: str
    name: str
    type: str  # "grafana", "sql", "prometheus", "loki", "s3", "kubernetes", etc.
    config: Dict[str, str]  # Configuration details (API keys, URLs, etc.)


class ConnectionRequest(BaseModel):
    """Request to create a new connection"""
    name: str
    type: str  # "grafana", "sql", "prometheus", "loki", "s3", "kubernetes", etc.
    config: Dict[str, str]


# Type-specific connection creation models
class GrafanaConnectionCreate(BaseModel):
    """Grafana connection creation parameters"""
    name: str
    url: str
    api_key: str

class GithubConnectionCreate(BaseModel):
    """GitHub connection creation parameters"""
    name: str
    github_personal_access_token: str

# Generic fallback for connection create
class ConnectionCreate(BaseModel):
    """Parameters for creating a connection"""
    name: str
    type: str  # "grafana", "sql", "prometheus", "loki", "s3", "kubernetes", etc.
    config: Dict


class ConnectionUpdate(BaseModel):
    """Parameters for updating a connection"""
    name: Optional[str] = None
    config: Optional[Dict] = None
    github_personal_access_token: Optional[str] = None


class ConnectionTest(BaseModel):
    """Parameters for testing a connection"""
    type: str  # "grafana", "sql", "prometheus", "loki", "s3", "kubernetes", etc.
    config: Dict
    github_personal_access_token: Optional[str] = None


@router.get("/")
async def list_connections(
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> List[Dict]:
    """
    Get a list of all connections
    
    Returns:
        List of connection information
    """
    start_time = time.time()
    try:
        connections = await connection_manager.get_all_connections()
        process_time = time.time() - start_time
        connection_logger.info(
            "Listed all connections",
            extra={
                'connection_count': len(connections),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        return connections
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error listing connections",
            extra={
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise


@router.get("/types")
async def list_connection_types(
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> List[str]:
    """
    Get a list of all available connection types
    
    Returns:
        List of connection types
    """
    start_time = time.time()
    try:
        # Return supported connection types
        # Data connection types
        data_connections = ["grafana", "github"]
        # Utility connection types
        utility_connections = ["python"]
        
        connection_types = data_connections + utility_connections
        process_time = time.time() - start_time
        
        connection_logger.info(
            "Listed connection types",
            extra={
                'type_count': len(connection_types),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        return connection_types
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error listing connection types",
            extra={
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise


@router.get("/{connection_id}")
async def get_connection(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Get a connection by ID
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        The connection information
    """
    start_time = time.time()
    try:
        connection = await connection_manager.get_connection(connection_id)
        if not connection:
            process_time = time.time() - start_time
            connection_logger.warning(
                "Connection not found",
                extra={
                    'connection_id': connection_id,
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Convert to type-specific configuration
        # Note: to_specific_config helps with type hinting but the actual data is in connection.config
        specific_config = connection.to_specific_config() 
        
        # Create response based on connection type
        response: Dict[str, Any] = {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type
        }
        
        # Add type-specific fields based on connection type, reading from the stored config
        if isinstance(specific_config, GrafanaConnectionConfig):
            # For Grafana stdio, get url/key from the stored config dictionary
            response["url"] = connection.config.get("grafana_url", "")
            response["api_key"] = "***REDACTED***" # Always redact API key
            response["config"] = connection_manager._redact_sensitive_fields(
                connection.config, # Pass the full config for redaction
                connection_type="grafana"
            )
        elif isinstance(specific_config, GithubConnectionConfig):
            # GitHub config holds command/args template/pat
            response["config"] = connection_manager._redact_sensitive_fields(
                connection.config, # Pass the full config for redaction
                connection_type="github"
            )
        elif isinstance(specific_config, GenericConnectionConfig):
            # For GenericConnectionConfig, include the full redacted config dictionary
            response["config"] = connection_manager._redact_sensitive_fields(connection.config)
        else:
            # Ultimate fallback - redact the original config
            response["config"] = connection_manager._redact_sensitive_fields(connection.config)
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Retrieved connection",
            extra={
                'connection_id': connection_id,
                'connection_name': connection.name,
                'connection_type': connection.type,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error retrieving connection",
            extra={
                'connection_id': connection_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise


@router.post("/grafana")
async def create_grafana_connection(
    connection_data: GrafanaConnectionCreate,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Create a new Grafana connection (now uses stdio MCP)
    
    Args:
        connection_data: Parameters for creating the Grafana connection
        
    Returns:
        The created connection information
    """
    start_time = time.time()
    try:
        # Pass url and api_key via kwargs for validation and inclusion in the prepared config
        config = {} # Start with an empty config, manager prepares it
        cm_kwargs = {
            "grafana_url": connection_data.url,
            "grafana_api_key": connection_data.api_key
        }
        
        connection = await connection_manager.create_connection(
            name=connection_data.name,
            type="grafana",
            config=config, # Empty, prepared by manager
            **cm_kwargs
        )
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Created Grafana connection",
            extra={
                'connection_id': connection.id,
                'connection_name': connection.name,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            # Redact the final prepared config, specifying the type for correct redaction
            "config": connection_manager._redact_sensitive_fields(connection.config, connection_type="grafana")
        }
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error creating Grafana connection",
            extra={
                'connection_name': connection_data.name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error creating Grafana connection",
            extra={
                'connection_name': connection_data.name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise

@router.post("/github")
async def create_github_connection(
    connection_data: GithubConnectionCreate,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Create a new GitHub connection (Docker MCP)
    
    Args:
        connection_data: Parameters for creating the GitHub connection
        
    Returns:
        The created connection information
    """
    start_time = time.time()
    try:
        # Config dict itself is now prepared by connection manager during validation
        # We just pass the PAT via kwargs
        config = {}
        cm_kwargs = {
            "github_personal_access_token": connection_data.github_personal_access_token
        }
        
        connection = await connection_manager.create_connection(
            name=connection_data.name,
            type="github",
            config=config,
            **cm_kwargs
        )
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Created GitHub connection",
            extra={
                'connection_id': connection.id,
                'connection_name': connection.name,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            "config": connection_manager._redact_sensitive_fields(connection.config, connection_type="github")
        }
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error creating GitHub connection",
            extra={
                'connection_name': connection_data.name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error creating GitHub connection",
            extra={
                'connection_name': connection_data.name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise

# Keep the generic endpoint for backward compatibility and other connection types
@router.post("/")
async def create_connection(
    connection_data: ConnectionCreate,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Create a new connection (generic method)
    
    Args:
        connection_data: Parameters for creating the connection
        
    Returns:
        The created connection information
    """
    start_time = time.time()
    try:
        # Prepare kwargs for potential PAT update
        cm_kwargs = {}
        if connection_data.type == "github" and connection_data.config.get("github_personal_access_token"):
            cm_kwargs["github_personal_access_token"] = connection_data.config["github_personal_access_token"]
        
        # Ensure name and config are provided if they exist in the request
        update_name = connection_data.name if connection_data.name is not None else None
        update_config = connection_data.config if connection_data.config is not None else {}
        
        # Get current connection details to provide original name if not updated
        current_connection = await connection_manager.get_connection(connection_data.name)
        if not current_connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_data.name} not found")
        
        final_name = update_name if update_name is not None else current_connection.name
        
        connection = await connection_manager.create_connection(
            name=final_name,
            type=connection_data.type,
            config=update_config,
            **cm_kwargs
        )
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Created connection",
            extra={
                'connection_id': connection.id,
                'connection_name': connection.name,
                'connection_type': connection.type,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            "config": connection_manager._redact_sensitive_fields(connection.config, connection_type=connection.type)
        }
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error creating connection",
            extra={
                'connection_name': connection_data.name,
                'connection_type': connection_data.type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error creating connection",
            extra={
                'connection_name': connection_data.name,
                'connection_type': connection_data.type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{connection_id}")
async def update_connection(
    connection_id: str,
    connection_data: ConnectionUpdate,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Update a connection
    
    Args:
        connection_id: The ID of the connection
        connection_data: Parameters for updating the connection
        
    Returns:
        The updated connection information
    """
    start_time = time.time()
    try:
        # Prepare kwargs for potential PAT update
        cm_kwargs = {}
        if connection_data.github_personal_access_token:
            cm_kwargs["github_personal_access_token"] = connection_data.github_personal_access_token
        
        # Ensure name and config are provided if they exist in the request
        update_name = connection_data.name if connection_data.name is not None else None
        update_config = connection_data.config if connection_data.config is not None else {}
        
        # Get current connection details to provide original name if not updated
        current_connection = await connection_manager.get_connection(connection_id)
        if not current_connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        final_name = update_name if update_name is not None else current_connection.name
        
        connection = await connection_manager.update_connection(
            connection_id=connection_id,
            name=final_name,
            config=update_config,
            **cm_kwargs
        )
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Updated connection",
            extra={
                'connection_id': connection.id,
                'connection_name': connection.name,
                'connection_type': connection.type,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            "config": connection_manager._redact_sensitive_fields(connection.config, connection_type=connection.type)
        }
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error updating connection",
            extra={
                'connection_id': connection_id,
                'connection_name': connection_data.name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error updating connection",
            extra={
                'connection_id': connection_id,
                'connection_name': connection_data.name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise


@router.delete("/{connection_id}", status_code=204)
async def delete_connection(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> None:
    """
    Delete a connection
    
    Args:
        connection_id: The ID of the connection
    """
    start_time = time.time()
    try:
        await connection_manager.delete_connection(connection_id)
        process_time = time.time() - start_time
        connection_logger.info(
            "Deleted connection",
            extra={
                'connection_id': connection_id,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error deleting connection",
            extra={
                'connection_id': connection_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error deleting connection",
            extra={
                'connection_id': connection_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise


@router.post("/test")
async def test_connection(
    connection_data: ConnectionTest,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict[str, Any]:
    """
    Test a connection configuration
    
    Args:
        connection_data: Parameters for testing the connection
        
    Returns:
        The test result
    """
    start_time = time.time()
    try:
        # Prepare kwargs for type-specific test data (like PAT)
        test_kwargs = {}
        if connection_data.type == "github" and connection_data.github_personal_access_token:
            test_kwargs["github_personal_access_token"] = connection_data.github_personal_access_token
        
        is_valid, message = await connection_manager.test_connection(
            connection_type=connection_data.type,
            config=connection_data.config,
            **test_kwargs
        )
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Tested connection configuration",
            extra={
                'connection_type': connection_data.type,
                'is_valid': is_valid,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return {
            "valid": is_valid,
            "message": message
        }
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error testing connection",
            extra={
                'connection_type': connection_data.type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error testing connection",
            extra={
                'connection_type': connection_data.type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connection_id}/default")
async def set_default_connection(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Set a connection as the default for its type
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        Success message
    """
    start_time = time.time()
    try:
        await connection_manager.set_default_connection(connection_id)
        process_time = time.time() - start_time
        connection_logger.info(
            "Set default connection",
            extra={
                'connection_id': connection_id,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        return {"message": "Default connection set successfully"}
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error setting default connection",
            extra={
                'connection_id': connection_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error setting default connection",
            extra={
                'connection_id': connection_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise


@router.get("/{connection_id}/schema")
async def get_connection_schema(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Get the schema for a connection
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        The schema information
    """
    try:
        connection = await connection_manager.get_connection(connection_id)
        if not connection:
             raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        schema = await connection_manager.get_connection_schema(connection_id)
        return schema
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: # Catch other potential errors
        connection_logger.error(f"Error getting schema for {connection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve connection schema")

@router.get("/{connection_type}/tools")
async def get_tools_for_connection_type(
    connection_type: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> List[MCPToolInfo]:
    """
    Get a list of available tools (name, description, inputSchema) for a specific connection type.
    
    Args:
        connection_type: The type of connection (e.g., "github", "grafana")
        
    Returns:
        List of tool definitions for the connection type
    """
    start_time = time.time()
    try:
        tools = await connection_manager.get_tools_for_connection_type(connection_type)
        process_time = time.time() - start_time
        connection_logger.info(
            f"Retrieved tools for connection type: {connection_type}",
            extra={
                'connection_type': connection_type,
                'tool_count': len(tools),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        return tools
    except ValueError as e:
        process_time = time.time() - start_time
        connection_logger.error(
            f"Error fetching tools for connection type: {connection_type}",
            extra={
                'connection_type': connection_type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            f"Error fetching tools for connection type: {connection_type}",
            extra={
                'connection_type': connection_type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tools for {connection_type}")