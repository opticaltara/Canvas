"""
S3 Cell Executor
"""

from typing import Any, Dict, Optional
from pydantic_ai import RunContext

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.ai.agent import S3QueryParams, query_s3, InvestigationDependencies


async def execute_s3_cell(cell: Cell, context: ExecutionContext) -> Any:
    """
    Execute an S3 operation cell using MCP servers
    
    Args:
        cell: The S3 operation cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Get the S3 operation parameters from cell metadata if available
    bucket = cell.metadata.get("bucket", "")
    prefix = cell.metadata.get("prefix", None)
    operation = cell.metadata.get("operation", "list_objects")
    
    if not bucket:
        return {
            "error": "S3 bucket not specified in cell metadata",
            "data": [],
            "query": cell.content
        }
    
    # Set up query parameters
    query_params = S3QueryParams(
        query=cell.content,
        bucket=bucket,
        prefix=prefix,
        operation=operation
    )
    
    # Set up dependencies for the AI tool to use
    dependencies = InvestigationDependencies(
        user_query="",  # Not needed for direct execution
        notebook_id=context.notebook.id if context.notebook else None
    )
    
    # Execute the query using the AI agent's query_s3 tool
    run_context = RunContext(dependencies)
    
    # Add MCP clients to the run context state
    run_context.state.update(context.mcp_clients)
    
    result = await query_s3(run_context, query_params)
    
    # Return the query result
    return {
        "data": result.data,
        "query": result.query,
        "error": result.error,
        "metadata": result.metadata
    }