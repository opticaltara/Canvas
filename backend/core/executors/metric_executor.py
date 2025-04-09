"""
Metric Cell Executor
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel
from typing import List

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.ai.agent import MetricQueryParams
from backend.services.connection_manager import get_connection_manager


class MetricQueryResult(BaseModel):
    data: List[Dict]
    query: str 
    error: Optional[str] = None
    metadata: Dict = {}


async def execute_metric_cell(cell: Cell, context: ExecutionContext) -> MetricQueryResult:
    """
    Execute a Metric query cell using MCP servers
    
    Args:
        cell: The Metric query cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Get the source, time range, and instant query preference from cell metadata if available
    source = cell.metadata.get("source", "prometheus")
    time_range = cell.metadata.get("time_range", None)
    instant = cell.metadata.get("instant", False)
    
    # Set up query parameters
    query_params = MetricQueryParams(
        query=cell.content,
        source=source,
        time_range=time_range,
        instant=instant
    )
    
    # Execute the query directly using the MCP clients
    connection_manager = get_connection_manager()
    
    # Get the Grafana connection
    connection = await connection_manager.get_default_connection("grafana")
    
    if not connection:
        return MetricQueryResult(
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
        result = mcp_client["query_metrics"](cell.content, query_params.model_dump())
        return MetricQueryResult(
            data=result.get("data", []),
            query=cell.content,
            error=None,
            metadata={"source": source, "connection_id": connection.id}
        )
    except Exception as e:
        return MetricQueryResult(
            data=[],
            query=cell.content,
            error=str(e),
            metadata={"source": source, "connection_id": connection.id}
        )