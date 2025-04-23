"""
MCP Server Manager - Deprecated

This module previously managed the lifecycle of central MCP servers.
With the shift to agent-managed stdio MCPs, this manager is no longer needed
and the related code has been removed.
"""

import asyncio
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional
from uuid import uuid4

# Remove import related to BaseConnectionConfig if MCPServerManager doesn't use it anymore
# from backend.services.connection_manager import BaseConnectionConfig 
from backend.core.logging import get_logger

# Initialize logger
mcp_logger = get_logger('mcp')

# Make sure correlation_id is set for all logs
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

# Add the filter to our logger if not already added by server.py
mcp_logger.addFilter(CorrelationIdFilter())

if not mcp_logger.handlers:
    # Check if logger has a parent with handlers
    has_parent_handlers = mcp_logger.parent and hasattr(mcp_logger.parent, 'handlers') and mcp_logger.parent.handlers
    
    if not has_parent_handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        mcp_logger.addHandler(console_handler)
        mcp_logger.setLevel(logging.INFO)