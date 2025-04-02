import asyncio
import json
import os
from typing import Dict, List, Optional, Set
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import logfire
from logfire.integrations.fastapi import LogfireMiddleware

from backend.config import get_settings
from backend.core.cell import Cell, CellType
from backend.core.execution import CellExecutor, ExecutionQueue
from backend.core.notebook import Notebook
from backend.routes.connections import router as connections_router
from backend.routes.notebooks import router as notebooks_router
from backend.services.connection_manager import ConnectionManager
from backend.services.notebook_manager import NotebookManager

# Initialize Logfire
settings = get_settings()
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN", ""),
    service_name="sherlog-canvas-backend",
    service_version="0.1.0",
    environment=os.getenv("SHERLOG_ENV", "development"),
    enable_console_logging=True,
)

# Instrument AI providers (Anthropic is used by the app)
try:
    logfire.instrument_anthropic()
    logfire.instrument_openai()  # In case OpenAI is used later
except Exception as e:
    print(f"Warning: Could not instrument AI providers: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Sherlog Canvas",
    description="Reactive notebook for software engineering investigations",
    version="0.1.0"
)

# Add Logfire middleware
app.add_middleware(LogfireMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with production origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger = logfire.getLogger("request")
    request_id = request.headers.get("X-Request-ID", str(UUID.uuid4()))
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client=request.client.host if request.client else None,
    )
    
    try:
        response = await call_next(request)
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
        )
        return response
    except Exception as e:
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise

# Initialize execution queue
execution_queue = ExecutionQueue()

# Client connection manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[UUID, Set[WebSocket]] = {}
        self.logger = logfire.getLogger("websocket")
    
    async def connect(self, websocket: WebSocket, notebook_id: UUID):
        await websocket.accept()
        if notebook_id not in self.active_connections:
            self.active_connections[notebook_id] = set()
        self.active_connections[notebook_id].add(websocket)
        self.logger.info("WebSocket connected", notebook_id=str(notebook_id), connections=len(self.active_connections[notebook_id]))
    
    def disconnect(self, websocket: WebSocket, notebook_id: UUID):
        if notebook_id in self.active_connections:
            self.active_connections[notebook_id].discard(websocket)
            self.logger.info("WebSocket disconnected", notebook_id=str(notebook_id), 
                           remaining_connections=len(self.active_connections[notebook_id]))
            if not self.active_connections[notebook_id]:
                del self.active_connections[notebook_id]
                self.logger.info("All connections closed for notebook", notebook_id=str(notebook_id))
    
    async def broadcast(self, notebook_id: UUID, message: Dict):
        if notebook_id in self.active_connections:
            message_type = message.get("type", "unknown")
            self.logger.debug("Broadcasting message", 
                             notebook_id=str(notebook_id),
                             message_type=message_type,
                             connections=len(self.active_connections[notebook_id]))
            for connection in self.active_connections[notebook_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    self.logger.error("Error broadcasting to client", error=str(e))


# Create instances
websocket_manager = WebSocketManager()

# Include routers
app.include_router(notebooks_router, prefix="/api/notebooks", tags=["notebooks"])
app.include_router(connections_router, prefix="/api/connections", tags=["connections"])

@app.get("/api/health")
def health_check():
    """Health check endpoint for the API"""
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    # Start the execution queue
    await execution_queue.start()
    
    # Initialize services
    settings = get_settings()
    
    # Initialize notebook manager
    notebook_manager = NotebookManager()
    app.state.notebook_manager = notebook_manager
    
    # Initialize connection manager
    connection_manager = ConnectionManager()
    await connection_manager.initialize()
    app.state.connection_manager = connection_manager
    
    # Initialize cell executor
    cell_executor = CellExecutor(notebook_manager)
    app.state.cell_executor = cell_executor
    
    # Attach the broadcast function to the notebook manager
    notebook_manager.set_notify_callback(websocket_manager.broadcast)
    
    # Initialize MCP server manager and start servers
    from backend.mcp.manager import get_mcp_server_manager
    mcp_manager = get_mcp_server_manager()
    app.state.mcp_manager = mcp_manager
    
    # Start MCP servers for all connections
    connections = list(connection_manager.connections.values())
    if connections:
        print(f"Starting MCP servers for {len(connections)} connections...")
        server_addresses = await mcp_manager.start_mcp_servers(connections)
        app.state.mcp_server_addresses = server_addresses
        print(f"Started {len(server_addresses)} MCP servers")


@app.on_event("shutdown")
async def shutdown_event():
    # Stop the execution queue
    await execution_queue.stop()
    
    # Close connections
    connection_manager = app.state.connection_manager
    await connection_manager.close()
    
    # Stop MCP servers
    if hasattr(app.state, "mcp_manager"):
        mcp_manager = app.state.mcp_manager
        await mcp_manager.stop_all_servers()


@app.websocket("/ws/notebook/{notebook_id}")
async def websocket_endpoint(websocket: WebSocket, notebook_id: UUID):
    try:
        # Connect the client
        await websocket_manager.connect(websocket, notebook_id)
        
        # Get the notebook manager
        notebook_manager = app.state.notebook_manager
        
        # Send initial notebook state
        try:
            notebook = notebook_manager.get_notebook(notebook_id)
            await websocket.send_json({
                "type": "notebook_state",
                "notebook_id": str(notebook_id),
                "data": notebook.serialize()
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Error loading notebook: {str(e)}"
            })
        
        # Handle messages from the client
        while True:
            try:
                message = await websocket.receive_json()
                message_type = message.get("type")
                
                if message_type == "execute_cell":
                    # Execute a cell
                    cell_id = UUID(message.get("cell_id"))
                    notebook_id = UUID(message.get("notebook_id"))
                    
                    # Add to execution queue
                    await execution_queue.add_cell(
                        notebook_id,
                        cell_id,
                        app.state.cell_executor
                    )
                    
                    await websocket.send_json({
                        "type": "cell_queued",
                        "cell_id": str(cell_id)
                    })
                
                elif message_type == "update_cell":
                    # Update a cell's content
                    cell_id = UUID(message.get("cell_id"))
                    notebook_id = UUID(message.get("notebook_id"))
                    content = message.get("content")
                    
                    notebook = notebook_manager.get_notebook(notebook_id)
                    cell = notebook.update_cell_content(cell_id, content)
                    
                    # Notify clients of the update
                    await websocket_manager.broadcast(notebook_id, {
                        "type": "cell_updated",
                        "cell_id": str(cell_id),
                        "notebook_id": str(notebook_id),
                        "data": cell.dict()
                    })
                
                elif message_type == "add_cell":
                    # Add a new cell
                    notebook_id = UUID(message.get("notebook_id"))
                    cell_type = CellType(message.get("cell_type"))
                    content = message.get("content", "")
                    position = message.get("position")
                    
                    notebook = notebook_manager.get_notebook(notebook_id)
                    cell = notebook.create_cell(cell_type, content, position=position)
                    
                    # Notify clients of the new cell
                    await websocket_manager.broadcast(notebook_id, {
                        "type": "cell_added",
                        "cell_id": str(cell.id),
                        "notebook_id": str(notebook_id),
                        "data": cell.dict()
                    })
                
                elif message_type == "delete_cell":
                    # Delete a cell
                    cell_id = UUID(message.get("cell_id"))
                    notebook_id = UUID(message.get("notebook_id"))
                    
                    notebook = notebook_manager.get_notebook(notebook_id)
                    notebook.remove_cell(cell_id)
                    
                    # Notify clients of the deletion
                    await websocket_manager.broadcast(notebook_id, {
                        "type": "cell_deleted",
                        "cell_id": str(cell_id),
                        "notebook_id": str(notebook_id)
                    })
                
                elif message_type == "add_dependency":
                    # Add a dependency between cells
                    dependent_id = UUID(message.get("dependent_id"))
                    dependency_id = UUID(message.get("dependency_id"))
                    notebook_id = UUID(message.get("notebook_id"))
                    
                    notebook = notebook_manager.get_notebook(notebook_id)
                    notebook.add_dependency(dependent_id, dependency_id)
                    
                    # Notify clients of the new dependency
                    await websocket_manager.broadcast(notebook_id, {
                        "type": "dependency_added",
                        "dependent_id": str(dependent_id),
                        "dependency_id": str(dependency_id),
                        "notebook_id": str(notebook_id)
                    })
                
                elif message_type == "remove_dependency":
                    # Remove a dependency between cells
                    dependent_id = UUID(message.get("dependent_id"))
                    dependency_id = UUID(message.get("dependency_id"))
                    notebook_id = UUID(message.get("notebook_id"))
                    
                    notebook = notebook_manager.get_notebook(notebook_id)
                    notebook.remove_dependency(dependent_id, dependency_id)
                    
                    # Notify clients of the removed dependency
                    await websocket_manager.broadcast(notebook_id, {
                        "type": "dependency_removed",
                        "dependent_id": str(dependent_id),
                        "dependency_id": str(dependency_id),
                        "notebook_id": str(notebook_id)
                    })
                
                elif message_type == "save_notebook":
                    # Save the notebook
                    notebook_id = UUID(message.get("notebook_id"))
                    notebook = notebook_manager.get_notebook(notebook_id)
                    
                    # Update metadata if provided
                    if "metadata" in message:
                        notebook.update_metadata(message["metadata"])
                    
                    # Save the notebook
                    notebook_manager.save_notebook(notebook_id)
                    
                    await websocket.send_json({
                        "type": "notebook_saved",
                        "notebook_id": str(notebook_id)
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message"
                })
            
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
    
    except WebSocketDisconnect:
        # Client disconnected
        websocket_manager.disconnect(websocket, notebook_id)
    
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)