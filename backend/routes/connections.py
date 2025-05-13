"""
Connection API Endpoints
"""

from typing import Dict, List, Optional, Any, Union, Annotated
from uuid import uuid4
import time
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Body
from pydantic import BaseModel, Field, model_validator

from backend.services.connection_manager import (
    ConnectionManager, 
    get_connection_manager,
    ConnectionConfig
)
import logging
from backend.core.types import MCPToolInfo

# Import handler-specific Pydantic models for Union
from backend.services.connection_handlers.github_handler import (
    GithubConnectionCreate, GithubConnectionUpdate, GithubConnectionTest
)
from backend.services.connection_handlers.jira_handler import (
    JiraConnectionCreate, JiraConnectionUpdate, JiraConnectionTest
)
# Import filesystem models
from backend.services.connection_handlers.filesystem_handler import (
    FileSystemConnectionCreate, FileSystemConnectionUpdate, FileSystemConnectionBase
    # FileSystemConnectionTest is skipped for now, using FileSystemConnectionBase for test
)
# Import git_repo model
from backend.services.connection_handlers.git_repo_handler import (
    GitRepoConnectionCreate,
    GitRepoConnectionHandler # Import handler class
)
# Import function to get registered types and handler getter
from backend.services.connection_handlers.registry import get_all_handler_types, get_handler

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


# --- Define Discriminated Unions for Payloads --- 

# Union for specific config fields during creation
CreatePayloadUnion = Annotated[
    Union[
        GithubConnectionCreate,
        JiraConnectionCreate,
        FileSystemConnectionCreate, # Add Filesystem create model
        GitRepoConnectionCreate # Add Git Repo create model
    ],
    Field(discriminator="type")
]

# Union for specific config fields during update
# Note: Update models in handlers allow optional fields
UpdatePayloadUnion = Annotated[
    Union[
        GithubConnectionUpdate,
        JiraConnectionUpdate,
        FileSystemConnectionUpdate, # Add Filesystem update model
        GitRepoConnectionCreate # Reuse Create model for Git Repo update for now
    ],
    Field(discriminator="type")
]

# Union for specific config fields during testing - Filesystem skipped for now
TestPayloadUnion = Annotated[
    Union[
        GithubConnectionTest,
        JiraConnectionTest,
        FileSystemConnectionBase, # Added for filesystem connection testing
        GitRepoConnectionCreate # Reuse Create model for Git Repo test for now
    ],
    Field(discriminator="type")
]


# --- Define New Request Body Models --- 

class CreateRequest(BaseModel):
    """Request body for creating a new connection."""
    name: str = Field(..., description="Unique name for the connection.")
    # The payload contains the type discriminator and type-specific fields
    payload: CreatePayloadUnion

class UpdateRequest(BaseModel):
    """Request body for updating a connection."""
    name: Optional[str] = Field(None, description="Optional new name for the connection.")
    # Payload is optional; if omitted, only the name might be updated.
    # If provided, it contains type discriminator and optional fields to update.
    payload: Optional[UpdatePayloadUnion] = None 

    # Ensure at least one field is provided for an update
    @model_validator(mode='before')
    @classmethod
    def check_at_least_one_value(cls, values):
        if not values.get('name') and not values.get('payload'):
            raise ValueError('At least name or payload must be provided for update')
        return values

class TestRequest(BaseModel):
    """Request body for testing a connection configuration."""
    # The payload contains the type discriminator and type-specific fields for testing
    payload: TestPayloadUnion 


# --- API Endpoints --- 

@router.get("/")
async def list_connections(
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> List[Dict]: # Return list of dicts matching DB structure + redaction
    """
    Get a list of all connections (sensitive fields redacted).
    
    Returns:
        List of connection information including id, name, type, is_default, and redacted config.
    """
    start_time = time.time()
    try:
        # ConnectionManager now handles fetching and redaction via handlers
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
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        # Reraise as internal server error or specific type?
        raise HTTPException(status_code=500, detail=f"Failed to list connections: {str(e)}")


@router.get("/types")
async def list_connection_types() -> List[str]:
    """Get a list of all available connection types."""
    start_time = time.time()
    try:
        handler_types = get_all_handler_types()
        # "log_ai" and "python" are virtual connection types not backed by handlers
        connection_types = sorted(list(set(handler_types) | {"python", "log_ai"}))
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Listed connection types",
            extra={'type_count': len(connection_types), 'processing_time_ms': round(process_time * 1000, 2)}
        )
        return connection_types
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error listing connection types",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to list connection types")


@router.get("/{connection_id}")
async def get_connection(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict[str, Any]: # Return a dict representing the connection
    """
    Get a specific connection by ID (sensitive fields redacted).
    """
    start_time = time.time()
    try:
        connection = await connection_manager.get_connection(connection_id)
        if not connection:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
        
        # Redact the config using the manager method (which uses handlers)
        redacted_config = connection_manager._redact_sensitive_fields(connection.config, connection.type)
        
        # Check if default (requires fetching default separately or adding to ConnectionConfig)
        # Let's add a step to check if it's default
        is_default = False
        default_conn = await connection_manager.get_default_connection(connection.type)
        if default_conn and default_conn.id == connection_id:
            is_default = True

        response = {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            "is_default": is_default,
            "config": redacted_config
        }
        
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
    except HTTPException: # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            "Error retrieving connection",
            extra={'connection_id': connection_id, 'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve connection {connection_id}")


# --- Refactored Generic Create Endpoint --- 
@router.post("/", status_code=201)
async def create_connection(
    request: CreateRequest, # Use the new request model with union
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict[str, Any]: # Return the created connection (redacted)
    """
    Create a new connection (handles different types via discriminated union).
    """
    start_time = time.time()
    connection_name = request.name
    connection_type = request.payload.type # Type comes from the payload
    
    try:
        # Pass the validated payload model directly to the manager
        connection = await connection_manager.create_connection(
            name=connection_name,
            type=connection_type,
            connection_data=request.payload 
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
        
        # Redact the config before returning
        redacted_config = connection_manager._redact_sensitive_fields(connection.config, connection.type)
        
        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            # Newly created might be default if it's the first of its type
            "is_default": connection_manager.default_connections.get(connection.type) == connection.id,
            "config": redacted_config
        }
    except (ValueError, TypeError) as e:
        # Catch validation/config errors from ConnectionManager/Handlers
        process_time = time.time() - start_time
        connection_logger.error(
            f"Error creating {connection_type} connection '{connection_name}'",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True # Include traceback for unexpected TypeErrors
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected errors
        process_time = time.time() - start_time
        connection_logger.error(
            f"Unexpected error creating {connection_type} connection '{connection_name}'",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- Refactored Update Endpoint --- 
@router.put("/{connection_id}")
async def update_connection(
    connection_id: str,
    request: UpdateRequest, # Use the new request model with optional payload
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict[str, Any]: # Return the updated connection (redacted)
    """
    Update a connection (name and/or config fields).
    Uses discriminated union for config updates.
    """
    start_time = time.time()
    
    # If payload is None, create a dummy empty model for the manager call? 
    # No, the manager's update logic handles None payload correctly if only name is updated.
    # However, the manager expects *some* pydantic model. Let's reconsider.
    # Option 1: Pass None payload if None.
    # Option 2: If payload is None, pass an empty dict? No, needs model.
    # Option 3: Handler's update model should accept potentially all None values.
    # Option 4: ConnectionManager detects None payload and only updates name. <-- Simpler

    # Let's refine ConnectionManager.update_connection to handle this:
    # It should accept `name: Optional[str]` and `connection_data: Optional[BaseModel]`. 
    # We need to modify ConnectionManager first.

    # *** TEMPORARY HACK: If payload is None, create an empty base model ***
    # This avoids changing ConnectionManager for now, but isn't ideal.
    # We'll assume a base update model exists or handlers handle empty update models.
    # Let's assume handler update models work with no fields set.
    connection_data_to_pass = request.payload 
    
    # Check if payload is provided and has actual update values
    has_config_updates = request.payload is not None and request.payload.model_dump(exclude_unset=True)
    
    # If only name is provided and payload is empty/None, we still need to call update, 
    # but the manager logic should handle the empty payload.
    if not request.name and not has_config_updates:
         # Pydantic validator in UpdateRequest should prevent this, but double-check
         raise HTTPException(status_code=400, detail="Update requires at least a name or config changes.")

    # Fetch connection type *before* calling update, in case update fails early
    existing_conn = await connection_manager.get_connection(connection_id)
    if not existing_conn:
         raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
    connection_type = existing_conn.type

    try:
        # If payload is None, we need to instantiate the correct type for the manager
        if connection_data_to_pass is None:
            # Get the handler to find the appropriate update model type
            try:
                 handler = get_handler(connection_type)
                 UpdateModel = handler.get_update_model()
                 # Instantiate with type only, assuming other fields are optional
                 connection_data_to_pass = UpdateModel(type=connection_type) 
            except (ValueError, NotImplementedError) as e:
                 raise HTTPException(status_code=400, detail=f"Cannot process name-only update for type '{connection_type}': {e}")

        connection = await connection_manager.update_connection(
            connection_id=connection_id,
            name=request.name, # Pass optional name
            connection_data=connection_data_to_pass # Pass validated payload or dummy model
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
        
        # Redact the config before returning
        redacted_config = connection_manager._redact_sensitive_fields(connection.config, connection.type)
        
        # Check if default
        is_default = False
        default_conn = await connection_manager.get_default_connection(connection.type)
        if default_conn and default_conn.id == connection_id:
            is_default = True

        return {
            "id": connection.id,
            "name": connection.name,
            "type": connection.type,
            "is_default": is_default,
            "config": redacted_config
        }
    except (ValueError, TypeError) as e:
        # Catch validation/config errors from ConnectionManager/Handlers
        process_time = time.time() - start_time
        connection_logger.error(
            f"Error updating connection {connection_id}",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException: # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch unexpected errors
        process_time = time.time() - start_time
        connection_logger.error(
            f"Unexpected error updating connection {connection_id}",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.delete("/{connection_id}", status_code=204)
async def delete_connection(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> None:
    """Delete a connection."""
    start_time = time.time()
    try:
        await connection_manager.delete_connection(connection_id)
        process_time = time.time() - start_time
        connection_logger.info(
            "Deleted connection",
            extra={'connection_id': connection_id, 'processing_time_ms': round(process_time * 1000, 2)}
        )
    except ValueError as e:
        # Typically "Connection not found"
        process_time = time.time() - start_time
        connection_logger.warning(
            f"Failed to delete connection {connection_id}: {e}",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)}
        )
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            f"Error deleting connection {connection_id}",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")


# --- Refactored Test Endpoint --- 
@router.post("/test")
async def test_connection(
    request: TestRequest, # Use the new request model with union
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict[str, Any]:
    """
    Test a connection configuration without saving it.
    """
    start_time = time.time()
    connection_type = request.payload.type
    
    try:
        # Pass the validated payload model directly to the manager
        is_valid, message = await connection_manager.test_connection(
            connection_type=connection_type,
            config_data=request.payload 
        )
        
        process_time = time.time() - start_time
        connection_logger.info(
            "Tested connection configuration",
            extra={
                'connection_type': connection_type,
                'is_valid': is_valid,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        # Return validation result
        # If invalid, return 400 status? Test endpoint usually returns 200 with validity flag.
        status_code = 200 # Keep 200 even for invalid config
        return {"valid": is_valid, "message": message}

    except (ValueError, TypeError) as e:
        # Catch errors from ConnectionManager/Handlers during test setup/execution
        process_time = time.time() - start_time
        connection_logger.warning(
            f"Error testing {connection_type} connection",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)}
            # No need for full traceback on validation errors
        )
        # Return validation failure with error message
        return {"valid": False, "message": str(e)}
        # raise HTTPException(status_code=400, detail=str(e)) # Or raise 400?
    except Exception as e:
        # Catch unexpected errors
        process_time = time.time() - start_time
        connection_logger.error(
            f"Unexpected error testing {connection_type} connection",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during testing: {str(e)}")


@router.post("/{connection_id}/default")
async def set_default_connection(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Dict:
    """Set a connection as the default for its type."""
    start_time = time.time()
    try:
        await connection_manager.set_default_connection(connection_id)
        process_time = time.time() - start_time
        connection_logger.info(
            "Set default connection",
            extra={'connection_id': connection_id, 'processing_time_ms': round(process_time * 1000, 2)}
        )
        return {"message": "Default connection set successfully"}
    except ValueError as e:
        # Handles "Connection not found" or DB errors from manager
        process_time = time.time() - start_time
        connection_logger.warning(
            f"Failed to set default connection {connection_id}: {e}",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)}
        )
        # Return 404 if not found, 400/500 for other errors?
        if "not found" in str(e).lower():
             raise HTTPException(status_code=404, detail=str(e))
        else:
             raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            f"Error setting default connection {connection_id}",
            extra={'error': str(e), 'error_type': type(e).__name__, 'processing_time_ms': round(process_time * 1000, 2)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to set default connection: {str(e)}")


@router.get("/{connection_type}/tools")
async def get_tools_for_connection_type(
    connection_type: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> List[MCPToolInfo]:
    """Get available tools for a specific connection type via its default connection."""
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
        # Catch errors from manager (no default conn, config error, MCP comms error)
        process_time = time.time() - start_time
        connection_logger.error(
            f"Error fetching tools for connection type: {connection_type}",
            extra={
                'connection_type': connection_type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            # exc_info=True # Maybe not needed for expected ValueErrors like 'no default'
        )
        # Return 404 if no default, 400/500 otherwise?
        if "no default connection found" in str(e).lower():
             raise HTTPException(status_code=404, detail=str(e))
        else: # Other config/MCP errors
             raise HTTPException(status_code=500, detail=f"Failed to retrieve tools for {connection_type}: {str(e)}")
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.error(
            f"Unexpected error fetching tools for connection type: {connection_type}",
            extra={
                'connection_type': connection_type,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred retrieving tools for {connection_type}")


# --- Re-index Endpoint --- 

@router.post(
    "/{connection_id}/reindex", 
    status_code=202, # 202 Accepted for async operation
    summary="Re-index Git Repository Connection",
    description="Triggers a background re-indexing process for a specific Git repository connection."
)
async def reindex_git_repo_connection(
    connection_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    Triggers a re-indexing process for a specific git_repo connection.
    This fetches the latest data from the Git repository and updates the Qdrant index.
    """
    # Relying on module-level imports for get_db_session and ConnectionRepository
    
    correlation_id = str(uuid4())
    start_time = time.time()
    connection_logger.info(f"Received request to re-index connection {connection_id}", extra={'correlation_id': correlation_id})

    # Fetch the full connection object using the manager (uses DB)
    # We need the actual DB model instance for post_create_actions
    connection = await connection_manager.get_connection(connection_id)
    
    # Use explicit None check
    if connection is None:
        connection_logger.error(f"Re-index failed: Connection {connection_id} not found", extra={'correlation_id': correlation_id})
        raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")

    # Ensure it's the correct type - cast to str for type checker
    connection_type_str = str(connection.type)
    if connection_type_str != "git_repo":
        connection_logger.error(f"Re-index failed: Connection {connection_id} is type '{connection_type_str}', not 'git_repo'", extra={'correlation_id': correlation_id})
        raise HTTPException(
            status_code=400,
            detail=f"Re-indexing is only supported for 'git_repo' connections. Found type '{connection_type_str}'."
        )

    try:
        # Get the specific handler using the casted string type
        handler = get_handler(connection_type_str)
        if not isinstance(handler, GitRepoConnectionHandler):
            connection_logger.error(f"Re-index failed: Could not get GitRepoConnectionHandler for {connection_id}", extra={'correlation_id': correlation_id})
            raise HTTPException(status_code=500, detail="Internal server error: Could not get appropriate handler.")

        # Call post_create_actions as a background task
        connection_logger.info(f"Scheduling re-indexing task for {connection_id} ('{connection.name}')", extra={'correlation_id': correlation_id})
        # Pass the SQLAlchemy model instance directly
        asyncio.create_task(handler.post_create_actions(connection))
        
        process_time = time.time() - start_time
        connection_logger.info(
            f"Successfully scheduled re-indexing for connection {connection_id}",
            extra={
                'correlation_id': correlation_id,
                'connection_name': connection.name,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        return {"message": f"Re-indexing process started for connection '{connection.name}' ({connection_id})."}

    except (ValueError, RuntimeError) as e:
        process_time = time.time() - start_time
        # Catch specific errors that might be raised during handler setup or if post_create_actions was awaited and failed
        connection_logger.error(
            f"Re-index failed for {connection_id}: {e}", 
            extra={
                'correlation_id': correlation_id,
                'error': str(e), 
                'error_type': type(e).__name__, 
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True # Include traceback for unexpected RuntimeErrors
        )
        # Return 400 for configuration/validation type errors, 500 for runtime issues
        status_code = 400 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        process_time = time.time() - start_time
        connection_logger.exception(
            f"Unexpected error during re-indexing request for {connection_id}: {e}", 
            extra={
                'correlation_id': correlation_id, 
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while trying to re-index: {str(e)}"
        )

# --- End Re-index Endpoint ---
