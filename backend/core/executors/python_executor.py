"""
Python Cell Executor
"""

import ast
import sys
import traceback
from io import StringIO
from typing import Any, Dict

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext


from pydantic import BaseModel
from typing import Any, Optional

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