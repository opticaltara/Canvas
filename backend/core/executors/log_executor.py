"""
Log Cell Executor
"""

from typing import Any

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.ai.agent import LogQueryParams
from backend.services.connection_manager import get_connection_manager

from pydantic import BaseModel
from typing import List, Dict, Optional

class LogQueryResult(BaseModel):
    data: List[Dict]
    query: str 
    error: Optional[str] = None
    metadata: Dict = {}

async def execute_log_cell(cell: Cell, context: ExecutionContext) -> LogQueryResult:
    """
    Execute a Log query cell using MCP servers
    
    Args:
        cell: The Log query cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Get the source and time range from cell metadata if available
    source = cell.metadata.get("source", "loki")
    time_range = cell.metadata.get("time_range", None)
    
    # Set up query parameters
    query_params = LogQueryParams(
        query=cell.content,
        source=source,
        time_range=time_range
    )
    
    # Execute the query directly using the MCP clients
    connection_manager = get_connection_manager()
    
    # Get the Grafana connection
    connection = await connection_manager.get_default_connection("grafana")
    
    if not connection:
        return LogQueryResult(
            data=[],
            query=cell.content,
            error="No Grafana connection available",
            metadata={}
        )
    
    # Execute the query using the connection
    try:
        # Get the MCP client for this connection
        mcp_client = context.mcp_clients.get(f"mcp_{connection.id}")
        if not mcp_client:
            raise ValueError(f"No MCP client found for connection {connection.id}")
            
        # Execute the query using the MCP client
        result = mcp_client["query_logs"](cell.content, query_params.model_dump())
        return LogQueryResult(
            data=result.get("data", []),
            query=cell.content,
            error=None,
            metadata={"source": source, "connection_id": connection.id}
        )
    except Exception as e:
        return LogQueryResult(
            data=[],
            query=cell.content,
            error=str(e),
            metadata={"source": source, "connection_id": connection.id}
        )