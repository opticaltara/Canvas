import logging
import os
from logging.handlers import RotatingFileHandler

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
    file_handler.addFilter(correlation_filter)  # Apply filter at the handler level
    
    # Create console handler for Docker/development environment
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)
    console_handler.addFilter(correlation_filter)  # Apply filter at the handler level

    # Loggers to configure - make sure to include more loggers
    app_logger_names = [
        'app', 'request', 'websocket', 'execution', 'notebook', 
        'connection', 'mcp', 'routes.connections', 'chat.db', 
        'routes.chat', 'ai.chat_agent', 'ai.chat_tools'
    ]
    # Include all uvicorn loggers 
    uvicorn_logger_names = ["uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"]
    # Include fastapi loggers
    fastapi_logger_names = ["fastapi"]
    # Include other third-party libraries that might log
    third_party_logger_names = ["asyncio", "httpx", "httpcore"]
    
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

# Function to get a logger with correlation_id filter
def get_logger(name: str) -> logging.Logger:
    """Get a logger with correlation_id filter applied"""
    logger = logging.getLogger(name)
    # Add correlation_id filter if not already present
    if not any(isinstance(f, CorrelationIdFilter) for f in logger.filters):
        logger.addFilter(CorrelationIdFilter())
    return logger 