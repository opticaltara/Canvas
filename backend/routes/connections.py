"""
Connection API Endpoints
"""

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.services.connection_manager import ConnectionManager, get_connection_manager

router = APIRouter()


class ConnectionCreate(BaseModel):
    """Parameters for creating a connection"""
    name: str
    type: str  # "grafana", "postgres", "prometheus", "loki", "s3", etc.
    config: Dict


class ConnectionUpdate(BaseModel):
    """Parameters for updating a connection"""
    name: str
    config: Dict


class ConnectionTest(BaseModel):
    """Parameters for testing a connection"""
    type: str  # "grafana", "postgres", "prometheus", "loki", "s3", etc.
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
    return connection_manager.get_all_connections()


@router.get("/types")
async def list_connection_types(
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> List[str]:
    """
    Get a list of all available connection types
    
    Returns:
        List of connection types
    """
    # Return supported connection types
    return ["grafana", "postgres", "prometheus", "loki", "s3", "kubernetes"]


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
    connection = connection_manager.get_connection(connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
    
    return {
        "id": connection.id,
        "name": connection.name,
        "type": connection.type,
        "config": connection_manager._redact_sensitive_fields(connection.config)
    }


@router.post("/")
async def create_connection(
    connection_data: ConnectionCreate,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """
    Create a new connection
    
    Args:
        connection_data: Parameters for creating the connection
        
    Returns:
        The created connection information
    """
    try:
        connection = await connection_manager.create_connection(
            name=connection_data.name,
            type=connection_data.type,
            config=connection_data.config
        )
        
        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            "config": connection_manager._redact_sensitive_fields(connection.config)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    try:
        connection = await connection_manager.update_connection(
            connection_id=connection_id,
            name=connection_data.name,
            config=connection_data.config
        )
        
        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            "config": connection_manager._redact_sensitive_fields(connection.config)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    try:
        await connection_manager.delete_connection(connection_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    try:
        is_valid = await connection_manager.test_connection(
            connection_type=connection_data.type,
            config=connection_data.config
        )
        
        return {
            "valid": is_valid,
            "message": "Connection test successful" if is_valid else "Connection test failed"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    try:
        await connection_manager.set_default_connection(connection_id)
        return {"message": "Default connection set successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
        schema = await connection_manager.get_connection_schema(connection_id)
        return schema
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{connection_id}/mcp/status")
async def get_mcp_server_status(
    connection_id: str,
    request: Request
) -> Dict:
    """
    Get the status of the MCP server for a connection
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        Status information for the MCP server
    """
    try:
        # Get the MCP server manager from the application state
        mcp_manager = request.app.state.mcp_manager
        
        # Get the connection manager to verify the connection exists
        connection_manager = request.app.state.connection_manager
        connection = connection_manager.get_connection(connection_id)
        
        if not connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Get the status
        status = mcp_manager.get_server_status(connection_id)
        
        return {
            "connection_id": connection_id,
            "connection_name": connection.name,
            "connection_type": connection.type,
            "status": status["status"],
            "address": status["address"],
            "error": status["error"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connection_id}/mcp/start")
async def start_mcp_server(
    connection_id: str,
    request: Request
) -> Dict:
    """
    Start the MCP server for a connection
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        Status information for the MCP server
    """
    try:
        # Get the MCP server manager from the application state
        mcp_manager = request.app.state.mcp_manager
        
        # Get the connection manager to get the connection
        connection_manager = request.app.state.connection_manager
        connection = connection_manager.get_connection(connection_id)
        
        if not connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Start the MCP server
        address = await mcp_manager.start_mcp_server(connection)
        
        # Get the updated status
        status = mcp_manager.get_server_status(connection_id)
        
        return {
            "connection_id": connection_id,
            "connection_name": connection.name,
            "connection_type": connection.type,
            "status": status["status"],
            "address": status["address"],
            "error": status["error"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connection_id}/mcp/stop")
async def stop_mcp_server(
    connection_id: str,
    request: Request
) -> Dict:
    """
    Stop the MCP server for a connection
    
    Args:
        connection_id: The ID of the connection
        
    Returns:
        Status information for the MCP server
    """
    try:
        # Get the MCP server manager from the application state
        mcp_manager = request.app.state.mcp_manager
        
        # Get the connection manager to verify the connection exists
        connection_manager = request.app.state.connection_manager
        connection = connection_manager.get_connection(connection_id)
        
        if not connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Stop the MCP server
        success = await mcp_manager.stop_server(connection_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop MCP server")
        
        # Get the updated status
        status = mcp_manager.get_server_status(connection_id)
        
        return {
            "connection_id": connection_id,
            "connection_name": connection.name,
            "connection_type": connection.type,
            "status": status["status"],
            "error": status["error"]
        }
    except Exception as e:
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
        # Get the MCP server manager from the application state
        mcp_manager = request.app.state.mcp_manager
        
        # Get the connection manager
        connection_manager = request.app.state.connection_manager
        
        # Get all server statuses
        statuses = mcp_manager.get_all_server_statuses()
        
        # Enrich with connection information
        enriched_statuses = {}
        for connection_id, status in statuses.items():
            connection = connection_manager.get_connection(connection_id)
            if connection:
                enriched_statuses[connection_id] = {
                    "connection_id": connection_id,
                    "connection_name": connection.name,
                    "connection_type": connection.type,
                    "status": status["status"],
                    "address": status["address"],
                    "error": status["error"]
                }
            else:
                # This is a connection that no longer exists
                enriched_statuses[connection_id] = {
                    "connection_id": connection_id,
                    "connection_name": "Unknown",
                    "connection_type": "Unknown",
                    "status": status["status"],
                    "address": status["address"],
                    "error": status["error"]
                }
        
        return {"servers": enriched_statuses}