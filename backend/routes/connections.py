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
    plugin_name: str
    config: Dict


class ConnectionUpdate(BaseModel):
    """Parameters for updating a connection"""
    name: str
    config: Dict


class ConnectionTest(BaseModel):
    """Parameters for testing a connection"""
    plugin_name: str
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


@router.get("/plugins")
async def list_plugins(
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> List[Dict]:
    """
    Get a list of all available plugins
    
    Returns:
        List of plugin information
    """
    return connection_manager.get_all_plugins()


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
        "plugin_name": connection.plugin_name,
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
            plugin_name=connection_data.plugin_name,
            config=connection_data.config
        )
        
        return {
            "id": connection.id,
            "name": connection.name,
            "plugin_name": connection.plugin_name,
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
            "plugin_name": connection.plugin_name,
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
            plugin_name=connection_data.plugin_name,
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
    Set a connection as the default for its plugin
    
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