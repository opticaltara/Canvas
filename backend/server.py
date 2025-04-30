import asyncio
import json
import os
import time
from typing import Dict, Set
from uuid import UUID, uuid4
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis

from backend.config import get_settings
from backend.core.execution import CellExecutor, ExecutionQueue
from backend.routes.connections import router as connections_router
from backend.routes.notebooks import router as notebooks_router
from backend.routes.chat import router as chat_router
from backend.routes.models import router as models_router
from backend.services.connection_manager import ConnectionManager
from backend.services.notebook_manager import NotebookManager
from backend.db.chat_db import ChatDatabase
from backend.core.logging import setup_logging, get_logger

# Initialize loggers
app_loggers = setup_logging()
app_logger = get_logger('app')
request_logger = get_logger('request')
websocket_logger = get_logger('websocket')
execution_logger = get_logger('execution')
notebook_logger = get_logger('notebook')
connection_logger = get_logger('connection')

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown event handler
    """
    app_logger.info("Application startup initiated")
    
    # Initialize shared state
    app.state.chat_agents = {} # Cache for active ChatAgentService instances
    app.state.redis = None # Initialize redis state
    app.state.chat_db = None # Initialize chat_db state
    app.state.notebook_manager = NotebookManager() # Initialize notebook manager
    app.state.connection_manager = ConnectionManager() # Initialize connection manager
    app.state.execution_queue = ExecutionQueue() # Initialize execution queue
    app_logger.info("Initialized shared application state attributes.")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    app_logger.info(f"Ensured data directory exists at {data_dir.absolute()}")
    
    # Initialize database
    app_logger.info("Initializing database tables")
    try:
        from backend.db.database import init_db
        await init_db()
        app_logger.info("Database tables initialized successfully")
    except Exception as e:
        app_logger.error(f"Failed to initialize database tables: {str(e)}", exc_info=True)
        raise
    
    # Initialize chat database
    app_logger.info("Initializing chat database connection")
    try:
        # Create the chat database connection
        chat_db_ctx = ChatDatabase.connect(data_dir)
        app.state.chat_db = await chat_db_ctx.__aenter__()
        app_logger.info("Chat database initialized successfully")
    except Exception as e:
        app_logger.error(f"Failed to initialize chat database: {str(e)}", exc_info=True)
        raise
    
    # Initialize Redis client
    app_logger.info("Initializing Redis connection")
    try:
        redis_client = await redis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
            password=settings.redis_password,
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping() # Verify connection
        app.state.redis = redis_client
        app_logger.info("Redis connection pool created and attached to app state.")
    except redis.RedisError as e:
        app_logger.error(f"Failed to initialize Redis connection: {str(e)}", exc_info=True)
        # Depending on requirements, you might want to raise here to prevent startup
        app.state.redis = None # Ensure state is None if failed
        # raise # Uncomment to prevent startup if Redis is critical
    except Exception as e: # Catch other potential errors
        app_logger.error(f"An unexpected error occurred during Redis initialization: {str(e)}", exc_info=True)
        app.state.redis = None
        # raise # Uncomment to prevent startup
    
    # Pass managers to the executor if needed (or configure executor differently)
    # Example: app.state.execution_queue.set_managers(app.state.notebook_manager, app.state.connection_manager)
    
    # Create task for cell execution
    app_logger.info("Starting cell execution worker")
    try:
        # Pass the queue from app.state to the task
        app.state.execution_task = asyncio.create_task(
            app.state.execution_queue.start(),
            name="cell_execution_worker"
        )
        app_logger.info("Cell execution worker started")
    except Exception as e:
        app_logger.error(f"Failed to start cell execution worker: {str(e)}", exc_info=True)
        raise
    
    app_logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    app_logger.info("Application shutdown initiated")
    
    # Cancel the cell execution task
    if hasattr(app.state, "execution_task"):
        app_logger.info("Cancelling cell execution worker")
        try:
            app.state.execution_task.cancel()
            try:
                await app.state.execution_task
            except asyncio.CancelledError:
                pass
            app_logger.info("Cell execution worker cancelled")
        except Exception as e:
            app_logger.error(f"Error cancelling cell execution worker: {str(e)}", exc_info=True)
    
    # Close chat database connection
    if hasattr(app.state, "chat_db"):
        app_logger.info("Closing chat database connection")
        try:
            await app.state.chat_db._asyncify(app.state.chat_db.con.close)
            app_logger.info("Chat database connection closed")
        except Exception as e:
            app_logger.error(f"Error closing chat database connection: {str(e)}", exc_info=True)
    
    # Close Redis connection
    if hasattr(app.state, "redis") and app.state.redis:
        app_logger.info("Closing Redis connection pool")
        try:
            await app.state.redis.close()
            app_logger.info("Redis connection pool closed")
        except Exception as e:
            app_logger.error(f"Error closing Redis connection pool: {str(e)}", exc_info=True)
    
    # Clear agent cache on shutdown (optional, helps release resources)
    if hasattr(app.state, "chat_agents"):
         app_logger.info(f"Clearing chat agent cache ({len(app.state.chat_agents)} instances).")
         app.state.chat_agents.clear()
    
    app_logger.info("Application shutdown completed")

# Initialize FastAPI app
app = FastAPI(
    title="Sherlog Canvas",
    description="Reactive notebook for software engineering investigations",
    version="0.1.0",
    lifespan=lifespan
)
# logfire.instrument_fastapi(app)

# Remove global manager instances - now managed in app.state via lifespan
# notebook_manager = NotebookManager()
# connection_manager = ConnectionManager()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with production origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remove direct state assignment - handled by lifespan
# app.state.notebook_manager = notebook_manager
# app.state.connection_manager = connection_manager

# Register routers
app.include_router(notebooks_router, prefix="/api/notebooks", tags=["notebooks"])
app.include_router(connections_router, prefix="/api/connections", tags=["connections"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(models_router, prefix="/api/models", tags=["models"])

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
        await websocket.accept()
        
        if notebook_id not in self.active_connections:
            self.active_connections[notebook_id] = set()
        
        self.active_connections[notebook_id].add(websocket)
        
        conn_count = len(self.active_connections[notebook_id])
        self.logger.info(
            f"WebSocket connected",
            extra={
                'correlation_id': str(uuid4()),
                'notebook_id': str(notebook_id),
                'client': str(websocket.client),
                'connection_count': conn_count
            }
        )
    
    def disconnect(self, websocket: WebSocket, notebook_id: UUID):
        if notebook_id in self.active_connections:
            if websocket in self.active_connections[notebook_id]:
                self.active_connections[notebook_id].remove(websocket)
                
                # Log the disconnection
                conn_count = len(self.active_connections[notebook_id])
                self.logger.info(
                    f"WebSocket disconnected",
                    extra={
                        'correlation_id': str(uuid4()),
                        'notebook_id': str(notebook_id),
                        'client': str(websocket.client),
                        'connection_count': conn_count
                    }
                )
                
                # Remove the notebook entry if no connections remain
                if not self.active_connections[notebook_id]:
                    del self.active_connections[notebook_id]
                    self.logger.info(
                        f"Removed notebook from active connections",
                        extra={
                            'correlation_id': str(uuid4()),
                            'notebook_id': str(notebook_id)
                        }
                    )
    
    async def broadcast(self, notebook_id: UUID, message: Dict):
        if notebook_id in self.active_connections:
            # Add a timestamp for client-side ordering
            if 'timestamp' not in message:
                message['timestamp'] = time.time()
                
            # Convert to JSON once for all clients
            data = json.dumps(message)
            
            # Create a list to store failures
            disconnected = []
            
            # Broadcast to all connected clients
            for websocket in self.active_connections[notebook_id]:
                try:
                    await websocket.send_text(data)
                    self.logger.info(
                        f"Message sent to client",
                        extra={
                            'correlation_id': str(uuid4()),
                            'notebook_id': str(notebook_id),
                            'client': str(websocket.client),
                            'message_type': message.get('type', 'unknown')
                        }
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to send message",
                        extra={
                            'correlation_id': str(uuid4()),
                            'notebook_id': str(notebook_id),
                            'client': str(websocket.client),
                            'error': str(e)
                        },
                        exc_info=True
                    )
                    # Add to disconnected list
                    disconnected.append(websocket)
            
            # Clean up disconnected clients
            for websocket in disconnected:
                self.disconnect(websocket, notebook_id)
                
            return len(self.active_connections[notebook_id])
        return 0

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# Remove global cell executor - needs access to managers from app.state if used
# cell_executor = CellExecutor(notebook_manager)
# Instead, CellExecutor might need to be instantiated where needed or accept managers
# from request state or dependency injection.

@app.get("/api/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "running", "version": "0.1.0"}


@app.websocket("/ws/notebook/{notebook_id}")
async def websocket_endpoint(websocket: WebSocket, notebook_id: UUID):
    """
    WebSocket endpoint for real-time notebook updates
    
    Clients connect to this endpoint to receive updates about notebook changes,
    cell execution status, and results.
    """
    request_id = str(uuid4())
    websocket_logger.info(
        f"WebSocket connection request",
        extra={
            'correlation_id': request_id,
            'notebook_id': str(notebook_id),
            'client': str(websocket.client)
        }
    )
    
    try:
        # Accept the connection
        await websocket_manager.connect(websocket, notebook_id)
        
        # Keep connection alive until client disconnects
        try:
            while True:
                # Wait for messages from the client
                try:
                    data = await websocket.receive_text()
                    
                    # Log received message
                    websocket_logger.info(
                        f"Received WebSocket message",
                        extra={
                            'correlation_id': request_id,
                            'notebook_id': str(notebook_id),
                            'client': str(websocket.client),
                            'message_length': len(data)
                        }
                    )
                    
                    # Echo back for now
                    await websocket.send_text(f"Message received: {data}")
                    
                except WebSocketDisconnect:
                    websocket_logger.info(
                        f"WebSocket disconnect detected during receive",
                        extra={
                            'correlation_id': request_id,
                            'notebook_id': str(notebook_id),
                            'client': str(websocket.client)
                        }
                    )
                    break
                    
        except Exception as e:
            websocket_logger.error(
                f"Error in WebSocket connection",
                extra={
                    'correlation_id': request_id,
                    'notebook_id': str(notebook_id),
                    'client': str(websocket.client),
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
    
    finally:
        # Ensure connection is closed and removed from manager
        websocket_manager.disconnect(websocket, notebook_id)


if __name__ == "__main__":
    # Check if inside Docker to determine host
    if os.path.exists('/.dockerenv'):
        host = "0.0.0.0"
    else:
        host = "127.0.0.1"
        
    # Run the uvicorn server
    uvicorn.run(
        "backend.server:app",
        host=host,
        port=8000,
        reload=True,
        log_level=os.environ.get("LOG_LEVEL", "info").lower()
    )