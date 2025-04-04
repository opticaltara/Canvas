"""
Python Cell Executor

Supports two execution modes:
1. Local execution (in-process)
2. Sandboxed execution using Pydantic MCP Python Run server
"""

import ast
import sys
import traceback
import json
import aiohttp
from io import StringIO
from typing import Any, Dict, List, Optional

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext
from backend.mcp.manager import get_mcp_server_manager, MCPServerStatus
from backend.services.connection_manager import ConnectionConfig, get_connection_manager


from pydantic import BaseModel


class PythonCellResult(BaseModel):
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


async def execute_python_cell(cell: Cell, context: ExecutionContext) -> PythonCellResult:
    """
    Execute a Python cell
    
    Args:
        cell: The Python cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Check if we should use the sandbox environment
    use_sandbox = context.settings.get("use_sandbox", False)
    
    if use_sandbox:
        return await execute_python_cell_sandboxed(cell, context)
    else:
        return await execute_python_cell_local(cell, context)


async def execute_python_cell_sandboxed(cell: Cell, context: ExecutionContext) -> PythonCellResult:
    """
    Execute a Python cell in a sandboxed environment using Pydantic MCP Python Run
    
    Args:
        cell: The Python cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Get the Python MCP server
    mcp_manager = get_mcp_server_manager()
    connection_manager = get_connection_manager()
    
    # Get or create a Python connection
    python_connections = [
        conn for conn in connection_manager.get_all_connections() 
        if conn.get("type") == "python"
    ]
    
    if not python_connections:
        # Create a Python connection
        connection = await connection_manager.create_connection(
            name="Python Sandbox",
            type="python",
            config={}  # No config needed for Python MCP
        )
        connection_id = connection.id
    else:
        # Use the first Python connection
        connection_id = python_connections[0]["id"]
        connection = connection_manager.get_connection(connection_id)
    
    # Ensure the Python MCP server is running
    server_status = mcp_manager.get_server_status(connection_id)
    if server_status["status"] != MCPServerStatus.RUNNING:
        # Start the server
        if not connection:
            raise RuntimeError("No valid Python connection found")
        address = await mcp_manager.start_mcp_server(connection)
        if not address:
            raise RuntimeError("Failed to start the Python MCP server")
    else:
        address = server_status["address"]
    
    # Prepare the code for execution
    # Format the code with dependencies section if needed
    dependencies = context.settings.get("dependencies", [])
    code = cell.content
    
    if dependencies:
        # Format the code with dependencies using PEP 723 syntax
        dependencies_str = ", ".join([f'"{dep}"' for dep in dependencies])
        code = f'''# /// script
# dependencies = [{dependencies_str}]
# ///

{code}
'''
    
    # Define the input variables
    variables = {}
    for var_name, var_value in context.variables.items():
        # Only pass serializable values to the sandbox
        try:
            json.dumps(var_value)
            variables[var_name] = var_value
        except (TypeError, ValueError):
            # Skip non-serializable values
            pass
    
    # Prepare the request payload
    payload = {
        "code": code,
        "vars": variables
    }
    
    # Call the Python MCP API to execute the code
    async with aiohttp.ClientSession() as session:
        try:
            url = f"http://{address}/run"
            async with session.post(url, json=payload) as response:
                result_data = await response.json()
                
                # Extract the execution results
                if "error" in result_data and result_data["error"]:
                    return PythonCellResult(
                        result=None,
                        stdout=result_data.get("stdout", ""),
                        stderr=result_data.get("stderr", ""),
                        error=result_data["error"]
                    )
                    
                # Process the successful result
                return PythonCellResult(
                    result=result_data.get("result"),
                    stdout=result_data.get("stdout", ""),
                    stderr=result_data.get("stderr", ""),
                    error=None
                )
                
        except Exception as e:
            return PythonCellResult(
                result=None,
                stdout="",
                stderr="",
                error=f"Failed to execute code in sandbox: {str(e)}"
            )


async def execute_python_cell_local(cell: Cell, context: ExecutionContext) -> PythonCellResult:
    """
    Execute a Python cell in the local process (non-sandboxed)
    
    Args:
        cell: The Python cell to execute
        context: The execution context
        
    Returns:
        The result of the execution
    """
    # Create a new globals dictionary with common modules
    globals_dict = {
        "__builtins__": __builtins__,
        "print": print,
    }
    
    # Add dependency results as variables
    locals_dict = context.variables.copy()
    
    # Capture stdout
    stdout = StringIO()
    stderr = StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout
    sys.stderr = stderr
    
    result = None
    
    try:
        # Execute the code
        code = compile(cell.content, f"cell_{cell.id}", "exec")
        exec(code, globals_dict, locals_dict)
        
        # Check for the last expression to capture its value
        try:
            # Parse the code into an AST
            parsed = ast.parse(cell.content)
            
            # Check if the last node is an expression
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                # Get the last expression
                last_expr = parsed.body[-1].value
                
                # Convert it to source code
                last_expr_src = ast.unparse(last_expr)
                
                # Evaluate it to get the result
                result = eval(last_expr_src, globals_dict, locals_dict)
        except:
            # If there's any error in capturing the last expression, ignore it
            pass
        
        # If we didn't get a result from the last expression,
        # use the stdout as the result
        if result is None:
            result = stdout.getvalue().strip()
            if not result and stderr.getvalue().strip():
                # If there's no stdout but there is stderr, use stderr
                result = stderr.getvalue().strip()
    
    except Exception:
        # Capture the traceback
        error = traceback.format_exc()
        raise RuntimeError(error)
    
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return PythonCellResult(
        result=result,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        error=None
    )