"""
SQL Cell Executor
"""

import logging
from typing import Any, Dict, Optional
from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.services.connection_manager import get_connection_manager


from pydantic import BaseModel
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class SQLQueryResult(BaseModel):
    data: List[Dict]
    query: str
    error: Optional[str] = None
    metadata: Dict = {}

async def execute_sql_cell(cell: Cell, context: ExecutionContext) -> SQLQueryResult:
    """
    Execute an SQL cell using MCP servers
    
    Args:
        cell: The SQL cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Get the connection ID from the cell
    connection_id = str(cell.connection_id) if cell.connection_id is not None else None
    
    # Get connection manager
    connection_manager = get_connection_manager()
    
    # Get the connection
    if connection_id:
        connection = await connection_manager.get_connection(connection_id)
    else:
        try:
            connection = await connection_manager.get_default_connection("sql")
            if connection:
                connection_id = connection.id
        except Exception as e:
            logger.error(f"Error getting default SQL connection: {str(e)}", extra={'correlation_id': 'N/A'})
            connection = None
    
    if not connection:
        return SQLQueryResult(
            data=[],
            query=cell.content,
            error="No SQL connection available",
            metadata={}
        )
    
    # Get MCP client for this connection
    mcp_client = context.mcp_clients.get(f"mcp_{connection.id}")
    if not mcp_client:
        return SQLQueryResult(
            data=[],
            query=cell.content,
            error=f"No MCP client available for connection: {connection.id}",
            metadata={}
        )
    
    try:
        # Execute the query using the MCP client
        result = await mcp_client["query_db"](cell.content, {})
        
        return SQLQueryResult(
            data=result.get("rows", []),
            query=cell.content,
            error=None,
            metadata={
                "connection_id": connection.id,
                "connection_name": connection.name,
                "columns": result.get("columns", [])
            }
        )
    except Exception as e:
        return SQLQueryResult(
            data=[],
            query=cell.content,
            error=f"MCP query error: {str(e)}",
            metadata={
                "connection_id": connection.id,
                "connection_name": connection.name
            }
        )