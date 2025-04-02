"""
SQL Cell Executor
"""

from typing import Any, Dict, Optional
from pydantic_ai import RunContext

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.ai.agent import SQLQueryParams, query_sql, InvestigationDependencies


async def execute_sql_cell(cell: Cell, context: ExecutionContext) -> Any:
    """
    Execute an SQL cell using MCP servers
    
    Args:
        cell: The SQL cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Get the connection ID from the cell metadata if available
    connection_id = cell.metadata.get("connection_id", None)
    
    # Set up query parameters
    query_params = SQLQueryParams(
        query=cell.content,
        connection_id=connection_id,
        parameters={}
    )
    
    # Set up dependencies for the AI tool to use
    dependencies = InvestigationDependencies(
        user_query="",  # Not needed for direct execution
        notebook_id=context.notebook.id if context.notebook else None
    )
    
    # Execute the query using the AI agent's query_sql tool
    run_context = RunContext(dependencies)
    
    # Add MCP clients to the run context state
    run_context.state.update(context.mcp_clients)
    
    result = await query_sql(run_context, query_params)
    
    # Return the query result
    return {
        "data": result.data,
        "query": result.query,
        "error": result.error,
        "metadata": result.metadata
    }