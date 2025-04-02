"""
Log Cell Executor
"""

from typing import Any, Dict, Optional
from pydantic_ai import RunContext

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.ai.agent import LogQueryParams, query_logs, InvestigationDependencies


async def execute_log_cell(cell: Cell, context: ExecutionContext) -> Any:
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
    
    # Set up dependencies for the AI tool to use
    dependencies = InvestigationDependencies(
        user_query="",  # Not needed for direct execution
        notebook_id=context.notebook.id if context.notebook else None
    )
    
    # Execute the query using the AI agent's query_logs tool
    run_context = RunContext(dependencies)
    
    # Add MCP clients to the run context state
    run_context.state.update(context.mcp_clients)
    
    result = await query_logs(run_context, query_params)
    
    # Return the query result
    return {
        "data": result.data,
        "query": result.query,
        "error": result.error,
        "metadata": result.metadata
    }