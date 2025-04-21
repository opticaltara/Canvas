"""
Connection API Endpoints
"""

from typing import Dict, List, Optional, Any
from uuid import uuid4
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.services.connection_manager import (
    ConnectionManager, 
    get_connection_manager,
    GrafanaConnectionConfig,
    GenericConnectionConfig,
    BaseConnectionConfig,
    GithubConnectionConfig
)
import logging

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
    name: str
    config: Dict


class ConnectionTest(BaseModel):
    """Parameters for testing a connection"""
    type: str  # "grafana", "sql", "prometheus", "loki", "s3", "kubernetes", etc.
    config: Dict


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
        specific_config = connection.to_specific_config()
        
        # Create response based on connection type
        response: Dict[str, Any] = {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type
        }
        
        # Add type-specific fields based on connection type
        if isinstance(specific_config, GrafanaConnectionConfig):
            response["url"] = specific_config.url
            response["api_key"] = "********" if specific_config.api_key else ""
        elif isinstance(specific_config, GithubConnectionConfig):
            response["github_personal_access_token"] = "********" if specific_config.github_personal_access_token else ""
        elif isinstance(specific_config, GenericConnectionConfig):
            # For GenericConnectionConfig, include the full config dictionary
            response["config"] = connection_manager._redact_sensitive_fields(specific_config.config)
        else:
            # Ultimate fallback - use the original config
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
    Create a new Grafana connection
    
    Args:
        connection_data: Parameters for creating the Grafana connection
        
    Returns:
        The created connection information
    """
    start_time = time.time()
    try:
        # Convert to the generic format used by connection manager
        config = {
            "url": connection_data.url,
            "api_key": connection_data.api_key
        }
        
        connection = await connection_manager.create_connection(
            name=connection_data.name,
            type="grafana",
            config=config
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
            "config": connection_manager._redact_sensitive_fields(connection.config)
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
        # Convert to the generic format used by connection manager
        config = {
            "github_personal_access_token": connection_data.github_personal_access_token
        }
        
        connection = await connection_manager.create_connection(
            name=connection_data.name,
            type="github",
            config=config
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
            "config": connection_manager._redact_sensitive_fields(connection.config)
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
        connection = await connection_manager.create_connection(
            name=connection_data.name,
            type=connection_data.type,
            config=connection_data.config
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
            "config": connection_manager._redact_sensitive_fields(connection.config)
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
        connection = await connection_manager.update_connection(
            connection_id=connection_id,
            name=connection_data.name,
            config=connection_data.config
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
            "config": connection_manager._redact_sensitive_fields(connection.config)
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
) -> Dict:
    """
    Test a connection configuration
    
    Args:
        connection_data: Parameters for testing the connection
        
    Returns:
        The test result
    """
    start_time = time.time()
    try:
        is_valid = await connection_manager.test_connection(
            connection_type=connection_data.type,
            config=connection_data.config
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
            "message": "Connection test successful" if is_valid else "Connection test failed"
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


@router.get("/{connection_id}/mcp/status")
async def get_mcp_server_status(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Get the status of the MCP server for a connection
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        Status information for the MCP server
    """
    try:
        # Get the connection
        connection = await connection_manager.get_connection(connection_id)
        
        if not connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Convert to BaseConnectionConfig
        base_connection = BaseConnectionConfig(
            id=connection.id,
            name=connection.name,
            type=connection.type,
            config=connection.config
        )
        
        # Get the status from the connection
        status = await base_connection.get_mcp_status()
        
        return {
            "connection_id": connection_id,
            "connection_name": connection.name,
            "connection_type": connection.type,
            "status": status["status"],
            "address": status.get("address"),
            "error": status.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connection_id}/mcp/start")
async def start_mcp_server(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Start the MCP server for a connection
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        Status information for the MCP server
    """
    correlation_id = str(uuid4())
    connection_logger.info(
        "Starting MCP server for connection",
        extra={
            'correlation_id': correlation_id,
            'connection_id': connection_id
        }
    )
    
    try:
        # Get the connection
        connection = await connection_manager.get_connection(connection_id)
        
        if not connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Convert to BaseConnectionConfig
        base_connection = BaseConnectionConfig(
            id=connection.id,
            name=connection.name,
            type=connection.type,
            config=connection.config
        )
        
        # Start the MCP server
        await base_connection.start_mcp_server()
        
        # Get the updated status
        status = await base_connection.get_mcp_status()
        
        return {
            "connection_id": connection_id,
            "connection_name": connection.name,
            "connection_type": connection.type,
            "status": status["status"],
            "address": status.get("address"),
            "error": status.get("error")
        }
    except Exception as e:
        connection_logger.error(
            f"Failed to start MCP server",
            extra={
                'correlation_id': correlation_id,
                'connection_id': connection_id,
                'error': str(e)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connection_id}/mcp/stop")
async def stop_mcp_server(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Stop the MCP server for a connection
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        Status information for the MCP server
    """
    correlation_id = str(uuid4())
    connection_logger.info(
        "Stopping MCP server for connection",
        extra={
            'correlation_id': correlation_id,
            'connection_id': connection_id
        }
    )
    
    try:
        # Get the connection
        connection = await connection_manager.get_connection(connection_id)
        
        if not connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Convert to BaseConnectionConfig
        base_connection = BaseConnectionConfig(
            id=connection.id,
            name=connection.name,
            type=connection.type,
            config=connection.config
        )
        
        # Stop the MCP server
        await base_connection.stop_mcp_server()
        
        # Get the updated status
        status = await base_connection.get_mcp_status()
        
        return {
            "connection_id": connection_id,
            "connection_name": connection.name,
            "connection_type": connection.type,
            "status": status["status"],
            "address": status.get("address"),
            "error": status.get("error")
        }
    except Exception as e:
        connection_logger.error(
            f"Failed to stop MCP server",
            extra={
                'correlation_id': correlation_id,
                'connection_id': connection_id,
                'error': str(e)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mcp/status")
async def get_all_mcp_server_statuses(
    request: Request
) -> Dict:
    """
    Get the status of all MCP servers
    
    Returns:
        Status information for all MCP servers
    """
    try:
        # Get managers from the application state
        mcp_manager = request.app.state.mcp_manager
        connection_manager = request.app.state.connection_manager
        
        # Get statuses of locally managed MCPs
        local_mcp_statuses = mcp_manager.get_all_server_statuses()
        
        # Get all connections
        all_connections = await connection_manager.get_all_connections() # Assuming this returns List[Dict] or similar
        
        all_statuses = {}
        for conn_dict in all_connections:
            conn_id = conn_dict['id']
            conn_type = conn_dict['type']
            conn_name = conn_dict['name']
            
            status_info = {
                "connection_id": conn_id,
                "connection_name": conn_name,
                "connection_type": conn_type,
                "status": "unknown", # Default status
                "address": None,
                "error": None
            }

            if conn_id in local_mcp_statuses:
                # Use status from the MCP manager for locally managed servers
                local_status = local_mcp_statuses[conn_id]
                status_info["status"] = local_status.get("status", "unknown")
                status_info["address"] = local_status.get("address")
                status_info["error"] = local_status.get("error")
            else:
                # Locally managed type but not in mcp_manager status (e.g., never started)
                status_info["status"] = "stopped"
                
            all_statuses[conn_id] = status_info
            
        return {"servers": all_statuses}
    except Exception as e:
        connection_logger.error(
            "Error retrieving all MCP server statuses",
            extra={
                'error': str(e),
                'error_type': type(e).__name__
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))