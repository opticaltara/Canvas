import asyncio
import builtins
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from backend.core.cell import Cell, PythonCell
from backend.core.execution import ExecutionContext
from backend.plugins.base import PluginBase


class PythonExecutor:
    """
    Executes Python code with safety restrictions and resource limitations
    """
    
    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 1024):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        
        # Allowed modules for imports
        self.allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'datetime', 'json', 'csv', 
            're', 'math', 'statistics', 'collections', 'itertools',
            'io', 'os.path', 'functools', 'typing'
        }
        
        # Disallowed builtins
        self.disallowed_builtins = {
            'exec', 'eval', 'compile', '__import__', 'open', 'input',
            'getattr', 'setattr', 'delattr', '__getattribute__',
            'globals', 'locals', 'vars'
        }
    
    async def execute(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Python code in a safe environment
        
        Args:
            code: The Python code to execute
            context: Optional dictionary of variables to include in the execution context
            
        Returns:
            A dictionary containing the execution results, including:
            - stdout: Captured standard output
            - stderr: Captured standard error
            - result: The result of the last expression (if any)
            - variables: Dictionary of variables defined during execution
            - figures: List of matplotlib figures created during execution
        """
        # Set up execution environment
        loc = {} if context is None else context.copy()
        glob = {
            "__builtins__": {
                name: getattr(builtins, name) for name in dir(builtins)
                if name not in self.disallowed_builtins
            },
            "pd": pd,
            "np": np,
            "plt": plt,
        }
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Keep track of created figures
        created_figures = []
        
        try:
            # Override the plt.figure function to track figures
            original_figure = plt.figure
            
            def tracked_figure(*args, **kwargs):
                fig = original_figure(*args, **kwargs)
                created_figures.append(fig)
                return fig
            
            plt.figure = tracked_figure
            
            # Run the code with timeout
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Set up task to run with timeout
                exec_task = asyncio.create_task(
                    self._exec_with_memory_limit(code, glob, loc, self.max_memory_mb)
                )
                
                await asyncio.wait_for(exec_task, timeout=self.timeout_seconds)
                
                last_expr_value = exec_task.result()
            
            # Restore original plt.figure
            plt.figure = original_figure
            
            # Get any matplotlib figures that were created but not explicitly saved
            if not created_figures and plt.get_fignums():
                created_figures = [plt.figure(n) for n in plt.get_fignums()]
            
            # Convert figures to base64 PNG images
            figure_images = []
            for fig in created_figures:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                
                import base64
                png_base64 = base64.b64encode(buf.read()).decode('utf-8')
                figure_images.append({
                    "format": "png",
                    "data": png_base64
                })
                plt.close(fig)
            
            # Close any remaining figures
            plt.close('all')
            
            # Prepare the result
            result = {
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "result": last_expr_value,
                "variables": {k: v for k, v in loc.items() if not k.startswith('_')},
                "figures": figure_images
            }
            
            # Convert any pandas DataFrames to HTML for better display
            if isinstance(last_expr_value, pd.DataFrame):
                result["dataframe_html"] = last_expr_value.to_html(max_rows=100)
            
            return result
        
        except asyncio.TimeoutError:
            return {
                "stdout": stdout_capture.getvalue(),
                "stderr": "Execution timed out after {} seconds".format(self.timeout_seconds),
                "error": "TimeoutError",
                "variables": {}
            }
        
        except Exception as e:
            return {
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue() + "\n" + traceback.format_exc(),
                "error": str(e),
                "variables": {}
            }
        
        finally:
            # Close captures
            stdout_capture.close()
            stderr_capture.close()
    
    async def _exec_with_memory_limit(self, code: str, glob: Dict, loc: Dict, max_memory_mb: int) -> Any:
        """
        Execute code with a memory limit
        
        Args:
            code: The Python code to execute
            glob: Global variables
            loc: Local variables
            max_memory_mb: Maximum memory usage in MB
            
        Returns:
            The result of the last expression (if any)
        """
        # TODO: Implement actual memory limiting
        # This is a placeholder for future implementation
        # For now, we just execute the code
        
        # Parse the code to capture the last expression value if present
        lines = code.strip().split('\n')
        if not lines:
            return None
        
        last_line = lines[-1].strip()
        
        # Check if the last line is an expression (not an assignment or statement)
        if (not last_line.startswith(('def ', 'class ', 'import ', 'from ', 'return ')) and
            '=' not in last_line and
            not last_line.endswith(':')):
            
            # Execute all but the last line
            if len(lines) > 1:
                exec('\n'.join(lines[:-1]), glob, loc)
            
            # Evaluate the last line as an expression
            try:
                return eval(last_line, glob, loc)
            except SyntaxError:
                # If it's not a valid expression, execute it as a statement
                exec(last_line, glob, loc)
                return None
        else:
            # Execute the entire code as statements
            exec(code, glob, loc)
            return None


async def execute_python_cell(cell: PythonCell, context: ExecutionContext) -> Dict[str, Any]:
    """
    Execute a Python cell
    
    Args:
        cell: The Python cell to execute
        context: The execution context
        
    Returns:
        The results of the Python code execution
    """
    # This would be set from configuration
    timeout_seconds = 30
    max_memory_mb = 1024
    
    # Create the Python executor
    executor = PythonExecutor(timeout_seconds=timeout_seconds, max_memory_mb=max_memory_mb)
    
    # Include context variables
    exec_context = {}
    if context.variables:
        exec_context.update(context.variables)
    
    # Execute the code
    result = await executor.execute(cell.content, exec_context)
    
    return result