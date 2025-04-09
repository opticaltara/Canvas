import asyncio
import json
import os
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, Set, Union, Optional
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
# import logfire

from backend.config import get_settings
from backend.core.cell import CellType
from backend.core.execution import CellExecutor, ExecutionQueue
from backend.routes.connections import router as connections_router
from backend.routes.notebooks import router as notebooks_router
from backend.services.connection_manager import ConnectionManager
from backend.services.notebook_manager import NotebookManager

# Add correlation ID to log records
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, we'll provide a default if it's missing
        # This will allow logs that don't have it to still work properly
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

# Function to get log level from environment variable
def get_log_level() -> int:
    """Get log level from environment variable LOG_LEVEL, defaults to INFO"""
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, log_level_name, logging.INFO)

# Enhanced logging configuration
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Determine if we're running in Docker
    is_docker = os.path.exists('/.dockerenv')
    
    # Get log level from environment
    log_level = get_log_level()
    
    # Define formatter and filter
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    correlation_filter = CorrelationIdFilter()

    # Create shared file handler
    file_handler = RotatingFileHandler(
        'logs/sherlog_canvas.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)
    
    # Create console handler for Docker/development environment
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)

    # Loggers to configure - make sure to include more loggers
    app_logger_names = ['app', 'request', 'websocket', 'execution', 'notebook', 'connection', 'mcp', 'routes.connections']
    # Include all uvicorn loggers 
    uvicorn_logger_names = ["uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"]
    # Include fastapi loggers
    fastapi_logger_names = ["fastapi"]
    # Include other third-party libraries that might log
    third_party_logger_names = ["asyncio"]
    
    all_logger_names = app_logger_names + uvicorn_logger_names + fastapi_logger_names + third_party_logger_names
    
    # Also configure the root logger to catch any unconfigured loggers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addFilter(correlation_filter)
    root_logger.setLevel(log_level)

    app_loggers = {}

    # Configure all loggers
    for name in all_logger_names:
        logger = logging.getLogger(name)
        # Remove existing handlers to prevent duplicate logs
        logger.handlers.clear()
        # Add both handlers for all environments
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addFilter(correlation_filter)
        logger.setLevel(log_level)
        # Enable propagation only for third-party loggers
        if name in app_logger_names:
            logger.propagate = False
            app_loggers[name] = logger

    if is_docker:
        # In Docker, log this important environment information
        root_logger.info(f"Running in Docker environment, logs will be sent to console and file. Log level: {logging.getLevelName(log_level)}")
    else:
        root_logger.info(f"Running in local environment, logs will be sent to console and {os.path.abspath('logs/sherlog_canvas.log')}. Log level: {logging.getLevelName(log_level)}")

    return app_loggers

# Initialize loggers
loggers = setup_logging()
app_logger = loggers['app']
request_logger = loggers['request']
websocket_logger = loggers['websocket']
execution_logger = loggers['execution']
notebook_logger = loggers['notebook']
connection_logger = loggers['connection']

# Apply filter - MOVED INSIDE setup_logging
for logger in loggers.values():
    logger.addFilter(CorrelationIdFilter())

settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Sherlog Canvas",
    description="Reactive notebook for software engineering investigations",
    version="0.1.0",
)
# logfire.instrument_fastapi(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with production origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start_time = time.time()
    
    # Add correlation ID to request state
    request.state.correlation_id = request_id
    
    request_logger.info(
        f"Request started",
        extra={
            'correlation_id': request_id,
            'method': request.method,
            'url': str(request.url),
            'client_host': request.client.host if request.client else None,
            'headers': dict(request.headers)
        }
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        request_logger.info(
            f"Request completed",
            extra={
                'correlation_id': request_id,
                'status_code': response.status_code,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        request_logger.error(
            f"Request failed",
            extra={
                'correlation_id': request_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise

# Initialize execution queue
execution_queue = ExecutionQueue()

# Enhanced WebSocket manager with detailed logging
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[UUID, Set[WebSocket]] = {}
        self.logger = websocket_logger
    
    async def connect(self, websocket: WebSocket, notebook_id: UUID):
        connection_id = str(uuid4())
        websocket.state.connection_id = connection_id
        
        await websocket.accept()
        if notebook_id not in self.active_connections:
            self.active_connections[notebook_id] = set()
        self.active_connections[notebook_id].add(websocket)
        
        self.logger.info(
            f"WebSocket connected",
            extra={
                'correlation_id': connection_id,
                'notebook_id': str(notebook_id),
                'connection_count': len(self.active_connections[notebook_id]),
                'client': websocket.client.host if websocket.client else None
            }
        )
    
    def disconnect(self, websocket: WebSocket, notebook_id: UUID):
        connection_id = getattr(websocket.state, 'connection_id', 'unknown')
        
        if notebook_id in self.active_connections:
            self.active_connections[notebook_id].discard(websocket)
            self.logger.info(
                f"WebSocket disconnected",
                extra={
                    'correlation_id': connection_id,
                    'notebook_id': str(notebook_id),
                    'remaining_connections': len(self.active_connections[notebook_id])
                }
            )
            
            if not self.active_connections[notebook_id]:
                del self.active_connections[notebook_id]
                self.logger.info(
                    f"All connections closed for notebook",
                    extra={
                        'correlation_id': connection_id,
                        'notebook_id': str(notebook_id)
                    }
                )
    
    async def broadcast(self, notebook_id: UUID, message: Dict):
        if notebook_id in self.active_connections:
            message_type = message.get("type", "unknown")
            correlation_id = str(uuid4())
            
            self.logger.debug(
                f"Broadcasting message",
                extra={
                    'correlation_id': correlation_id,
                    'notebook_id': str(notebook_id),
                    'message_type': message_type,
                    'connection_count': len(self.active_connections[notebook_id])
                }
            )
            
            for connection in self.active_connections[notebook_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    self.logger.error(
                        f"Error broadcasting to client",
                        extra={
                            'correlation_id': correlation_id,
                            'notebook_id': str(notebook_id),
                            'error': str(e),
                            'error_type': type(e).__name__
                        },
                        exc_info=True
                    )


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
    correlation_id = str(uuid4())
    app_logger.info(
        "Starting application",
        extra={'correlation_id': correlation_id}
    )
    
    try:
        # Initialize database
        from backend.db.database import init_db
        await init_db()
        app_logger.info(
            "Database initialized",
            extra={'correlation_id': correlation_id}
        )
        
        # Start the execution queue
        await execution_queue.start()
        app_logger.info(
            "Execution queue started",
            extra={'correlation_id': correlation_id}
        )
        
        # Initialize services
        settings = get_settings()
        
        # Initialize notebook manager
        notebook_manager = NotebookManager()
        app.state.notebook_manager = notebook_manager
        app_logger.info(
            "Notebook manager initialized",
            extra={'correlation_id': correlation_id}
        )
        
        # Initialize connection manager
        connection_manager = ConnectionManager()
        await connection_manager.initialize()
        app.state.connection_manager = connection_manager
        connection_logger.info(
            "Connection manager initialized",
            extra={'correlation_id': correlation_id}
        )
        
        # Initialize cell executor
        cell_executor = CellExecutor(notebook_manager)
        app.state.cell_executor = cell_executor
        execution_logger.info(
            "Cell executor initialized",
            extra={'correlation_id': correlation_id}
        )
        
        # Attach the broadcast function to the notebook manager
        notebook_manager.set_notify_callback(websocket_manager.broadcast)
        
        # Initialize MCP server manager but don't start servers automatically
        from backend.mcp.manager import get_mcp_server_manager_async
        mcp_manager = await get_mcp_server_manager_async()
        app.state.mcp_manager = mcp_manager
        app_logger.info(
            "MCP server manager initialized",
            extra={'correlation_id': correlation_id}
        )
        
        app_logger.info(
            "Application startup completed successfully",
            extra={'correlation_id': correlation_id}
        )
    except Exception as e:
        app_logger.error(
            "Application startup failed",
            extra={
                'correlation_id': correlation_id,
                'error': str(e),
                'error_type': type(e).__name__
            },
            exc_info=True
        )
        raise


@app.on_event("shutdown")
async def shutdown_event():
    correlation_id = str(uuid4())
    app_logger.info(
        "Starting application shutdown",
        extra={'correlation_id': correlation_id}
    )
    
    try:
        # Stop the execution queue
        await execution_queue.stop()
        app_logger.info(
            "Execution queue stopped",
            extra={'correlation_id': correlation_id}
        )
        
        # Close connections
        connection_manager = app.state.connection_manager
        await connection_manager.close()
        connection_logger.info(
            "Connection manager closed",
            extra={'correlation_id': correlation_id}
        )
        
        # Stop MCP servers
        if hasattr(app.state, "mcp_manager"):
            mcp_manager = app.state.mcp_manager
            await mcp_manager.stop_all_servers()
            app_logger.info(
                "MCP servers stopped",
                extra={'correlation_id': correlation_id}
            )
        
        app_logger.info(
            "Application shutdown completed successfully",
            extra={'correlation_id': correlation_id}
        )
    except Exception as e:
        app_logger.error(
            "Application shutdown failed",
            extra={
                'correlation_id': correlation_id,
                'error': str(e),
                'error_type': type(e).__name__
            },
            exc_info=True
        )
        raise


@app.websocket("/ws/notebook/{notebook_id}")
async def websocket_endpoint(websocket: WebSocket, notebook_id: UUID):
    connection_id = str(uuid4())
    websocket_logger.info(
        "WebSocket connection attempt",
        extra={
            'correlation_id': connection_id,
            'notebook_id': str(notebook_id),
            'client': websocket.client.host if websocket.client else None
        }
    )
    
    try:
        # Connect the client
        await websocket_manager.connect(websocket, notebook_id)
        
        # Get the notebook manager
        notebook_manager = app.state.notebook_manager
        
        # Send initial notebook state
        try:
            start_time = time.time()
            notebook = notebook_manager.get_notebook(notebook_id)
            
            await websocket.send_json({
                "type": "notebook_state",
                "notebook_id": str(notebook_id),
                "data": notebook.serialize()
            })
            
            process_time = time.time() - start_time
            websocket_logger.info(
                "Initial notebook state sent",
                extra={
                    'correlation_id': connection_id,
                    'notebook_id': str(notebook_id),
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
        except Exception as e:
            websocket_logger.error(
                "Error loading notebook",
                extra={
                    'correlation_id': connection_id,
                    'notebook_id': str(notebook_id),
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            await websocket.send_json({
                "type": "error",
                "message": f"Error loading notebook: {str(e)}"
            })
        
        # Handle messages from the client
        while True:
            try:
                message = await websocket.receive_json()
                message_type = message.get("type")
                
                websocket_logger.debug(
                    "Received WebSocket message",
                    extra={
                        'correlation_id': connection_id,
                        'notebook_id': str(notebook_id),
                        'message_type': message_type
                    }
                )
                
                if message_type == "execute_cell":
                    # Execute a cell
                    cell_id = UUID(message.get("cell_id"))
                    notebook_id = UUID(message.get("notebook_id"))
                    
                    execution_logger.info(
                        "Cell execution requested",
                        extra={
                            'correlation_id': connection_id,
                            'notebook_id': str(notebook_id),
                            'cell_id': str(cell_id)
                        }
                    )
                    
                    # Add to execution queue
                    start_time = time.time()
                    await execution_queue.add_cell(
                        notebook_id,
                        cell_id,
                        app.state.cell_executor
                    )
                    process_time = time.time() - start_time
                    
                    execution_logger.info(
                        "Cell added to execution queue",
                        extra={
                            'correlation_id': connection_id,
                            'notebook_id': str(notebook_id),
                            'cell_id': str(cell_id),
                            'queue_time_ms': round(process_time * 1000, 2)
                        }
                    )
                
                elif message_type == "update_cell":
                    # Update a cell's content and settings
                    cell_id = UUID(message.get("cell_id"))
                    notebook_id = UUID(message.get("notebook_id"))
                    content = message.get("content")
                    settings = message.get("settings")
                    
                    notebook = notebook_manager.get_notebook(notebook_id)
                    cell = notebook.update_cell_content(cell_id, content)
                    
                    # Update settings if provided
                    if settings is not None:
                        cell.settings = settings
                    
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
        websocket_logger.info(
            "WebSocket disconnected",
            extra={
                'correlation_id': connection_id,
                'notebook_id': str(notebook_id)
            }
        )
        websocket_manager.disconnect(websocket, notebook_id)
    except Exception as e:
        websocket_logger.error(
            "WebSocket error",
            extra={
                'correlation_id': connection_id,
                'notebook_id': str(notebook_id),
                'error': str(e),
                'error_type': type(e).__name__
            },
            exc_info=True
        )
        websocket_manager.disconnect(websocket, notebook_id)
        raise


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.environ.get("SHERLOG_API_HOST", "0.0.0.0")
    port = int(os.environ.get("SHERLOG_API_PORT", "8000"))
    reload_mode = os.environ.get("SHERLOG_DEBUG", "false").lower() == "true"
    log_level_name = os.environ.get("LOG_LEVEL", "info").lower()
    
    # Configure Uvicorn with log settings to ensure it uses our configuration
    uvicorn.run(
        "backend.server:app", 
        host=host,
        port=port,
        reload=reload_mode,
        log_config=None,  # Disable Uvicorn's default logging config
        log_level=log_level_name,
        access_log=True,  # Enable access logs
        use_colors=True,  # Colorized output on terminals
    )