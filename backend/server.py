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
from backend.services.connection_handlers.registry import get_all_handler_types

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
    
    # Import connection handlers to trigger registration
    try:
        # These imports execute the handler files, which call register_handler
        from backend.services.connection_handlers import github_handler
        from backend.services.connection_handlers import jira_handler
        app_logger.info(f"Successfully imported and registered connection handlers: {get_all_handler_types()}")
    except ImportError as e:
        app_logger.error(f"Failed to import connection handlers: {e}", exc_info=True)
        # Depending on requirements, you might want to raise here to prevent startup
        # if connection handlers are critical.

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

@app.get("/api/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "running", "version": "0.1.0"}

# Remove old websocket endpoint (lines 360-434)

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