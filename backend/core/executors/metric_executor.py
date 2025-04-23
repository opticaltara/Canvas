"""
Metric Cell Executor
"""

import logging
import json
from typing import Any, Dict, Optional, List, cast
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.core.query_result import MetricQueryResult
from backend.services.connection_manager import get_connection_manager


async def execute_metric_cell(cell: Cell, context: ExecutionContext) -> MetricQueryResult:
    """
    Execute a Metric query cell using MCP servers
    
    Args:
        cell: The Metric query cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    logger = logging.getLogger("metric_executor")
    correlation_id = context.variables.get("correlation_id", "N/A") # Get correlation ID if available
    logger.info(f"Executing metric cell {cell.id}", extra={'correlation_id': correlation_id, 'cell_id': str(cell.id)})
    
    # Get the source, time range, and instant query preference from cell metadata if available
    source = cell.metadata.get("source", "prometheus")
    time_range = cell.metadata.get("time_range", None)
    instant = cell.metadata.get("instant", False)
    
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
    
    # Execute the query using the connection config via stdio
    try:
        # Extract stdio parameters from the connection config
        config = connection.config
        command_list = config.get("mcp_command")
        args_list = config.get("mcp_args")
        env_dict = config.get("env")
        
        if not command_list or args_list is None or env_dict is None:
            logger.error(f"Incomplete stdio configuration for Grafana connection {connection.id}", extra={'correlation_id': correlation_id})
            raise ValueError(f"Incomplete stdio configuration for connection {connection.id}")
            
        command = command_list[0]
        args = command_list[1:] + args_list # Combine base command args with specific args
            
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env_dict
        )
        
        logger.info(f"Attempting to connect to Grafana MCP via stdio: {command} {' '.join(args)}", extra={'correlation_id': correlation_id})
        
        metric_data = []
        error_message = None
        
        # Execute via stdio client
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info(f"Grafana MCP session initialized. Calling query_prometheus for cell {cell.id}", extra={'correlation_id': correlation_id})
                
                # Prepare parameters for the tool call
                # TODO: Adapt parameters based on mcp-grafana's actual tool spec (e.g., time range, instant query)
                tool_params = {"query": cell.content}
                if time_range:
                    tool_params["time_range"] = time_range # Assuming mcp-grafana accepts this
                if instant:
                    tool_params["instant"] = instant # Assuming mcp-grafana accepts this
                    
                result_message = await session.call_tool("query_prometheus", tool_params)
                
                # Process the result message
                if result_message.content:
                    # Assuming the first content part is the JSON result
                    # TODO: Verify the exact result structure from mcp-grafana spec
                    try:
                        # Expecting text content containing JSON
                        content_dict = cast(Dict[str, Any], result_message.content[0])
                        result_data_str = content_dict.get('text', '{}')
                        result_data = json.loads(result_data_str)
                        metric_data = result_data.get("data", []) # Adapt based on actual structure
                        logger.info(f"Successfully received data from query_prometheus for cell {cell.id}", extra={'correlation_id': correlation_id})
                    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as json_err:
                         logger.error(f"Failed to parse result from query_prometheus: {json_err}. Raw content: {result_message.content}", extra={'correlation_id': correlation_id})
                         error_message = f"Failed to parse result: {json_err}"
                else:
                    logger.warning(f"Received empty content from query_prometheus for cell {cell.id}", extra={'correlation_id': correlation_id})
                    error_message = "Received no data from Grafana query."
        
        return MetricQueryResult(
            data=metric_data,
            query=cell.content,
            error=error_message,
            metadata={"source": source, "connection_id": connection.id}
        )

    except FileNotFoundError as fnf_err:
        logger.error(f"MCP command not found for Grafana: {fnf_err}", extra={'correlation_id': correlation_id}, exc_info=True)
        return MetricQueryResult(
            data=[],
            query=cell.content,
            error=f"MCP command not found: {fnf_err.filename}. Ensure mcp-grafana is installed.",
            metadata={"source": source, "connection_id": connection.id}
        )
    except ConnectionRefusedError as conn_err:
         logger.error(f"MCP connection refused for Grafana: {conn_err}", extra={'correlation_id': correlation_id}, exc_info=True)
         return MetricQueryResult(
             data=[],
             query=cell.content,
             error=f"MCP connection refused. Is the mcp-grafana process running correctly? Error: {conn_err}",
             metadata={"source": source, "connection_id": connection.id}
         )
    except Exception as e:
        logger.error(f"Error executing metric cell via stdio: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
        return MetricQueryResult(
            data=[],
            query=cell.content,
            error=f"Error during MCP execution: {str(e)}",
            metadata={"source": source, "connection_id": connection.id}
        )