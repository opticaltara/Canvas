import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.cell import Cell, CellStatus, CellType, GitHubCell
from backend.core.notebook import Notebook
from backend.core.dependency import ToolCallID
from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.core.cell import GitHubCell
from mcp.client.stdio import stdio_client # Reverted import style
from mcp import ClientSession
from mcp.shared.exceptions import McpError
import logging
import os # Import os module

# Import StdioServerParameters
from mcp import StdioServerParameters
# Import handler registry
from backend.services.connection_handlers.registry import get_handler
# Import settings
from backend.config import get_settings
# Assuming ErrorData is here (adjust if necessary)
from mcp.types import ErrorData
# Import DB session management
from backend.db.database import get_db_session 

# Initialize loggers
execution_logger = logging.getLogger("execution")
cell_executor_logger = logging.getLogger("cell_executor")


# --- Tool Call Representation ---


# --- Existing Execution Context and Classes ---

@dataclass
class ExecutionContext:
    """Context for cell execution with access to notebook state"""
    notebook: Notebook
    cell_id: UUID
    variables: Dict[str, Any] = field(default_factory=dict)
    mcp_clients: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


class ExecutionQueue:
    """
    Manages the queue of cells to be executed
    Ensures cells are executed in the correct order
    """
    def __init__(self):
        self.queue = asyncio.Queue()
        self._running = False
        self._task = None
        self.active_tasks: Set[UUID] = set()
        self.logger = execution_logger
    
    async def add_cell(self, notebook_id: UUID, cell_id: UUID, executor: 'CellExecutor') -> None:
        """Add a cell to the execution queue"""
        correlation_id = str(uuid4())
        
        if cell_id in self.active_tasks:
            self.logger.warning(
                "Cell already in execution queue",
                extra={
                    'correlation_id': correlation_id,
                    'notebook_id': str(notebook_id),
                    'cell_id': str(cell_id)
                }
            )
            return
        
        self.logger.info(
            "Adding cell to execution queue",
            extra={
                'correlation_id': correlation_id,
                'notebook_id': str(notebook_id),
                'cell_id': str(cell_id),
                'queue_size': self.queue.qsize()
            }
        )
        await self.queue.put((notebook_id, cell_id, executor, correlation_id))
    
    async def start(self) -> None:
        """Start processing the execution queue"""
        correlation_id = str(uuid4())
        
        if self._running:
            self.logger.warning(
                "Execution queue already running",
                extra={'correlation_id': correlation_id}
            )
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
        self.logger.info(
            "Execution queue started",
            extra={'correlation_id': correlation_id}
        )
    
    async def stop(self) -> None:
        """Stop processing the execution queue"""
        correlation_id = str(uuid4())
        
        if not self._running:
            self.logger.warning(
                "Execution queue already stopped",
                extra={'correlation_id': correlation_id}
            )
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None
        
        self.logger.info(
            "Execution queue stopped",
            extra={
                'correlation_id': correlation_id,
                'active_tasks': len(self.active_tasks),
                'remaining_queue_size': self.queue.qsize()
            }
        )
    
    async def _process_queue(self) -> None:
        """Process items in the execution queue"""
        while self._running:
            notebook_id, cell_id, executor, correlation_id = None, None, None, None # Initialize for finally block
            try:
                # Get the next cell to execute
                notebook_id, cell_id, executor, correlation_id = await self.queue.get()
                
                # Skip if cell is already being processed
                if cell_id in self.active_tasks:
                    self.logger.warning(
                        "Cell already being processed, skipping",
                        extra={
                            'correlation_id': correlation_id,
                            'notebook_id': str(notebook_id),
                            'cell_id': str(cell_id)
                        }
                    )
                    # self.queue.task_done() # Moved to finally
                    continue
                
                # Track active task
                self.active_tasks.add(cell_id)
                start_time = time.time()
                
                try:
                    self.logger.info(
                        "Starting cell execution within DB session scope",
                        extra={
                            'correlation_id': correlation_id,
                            'notebook_id': str(notebook_id),
                            'cell_id': str(cell_id),
                            'active_tasks': len(self.active_tasks),
                            'queue_size': self.queue.qsize()
                        }
                    )
                    
                    # Execute the cell within a database session scope
                    async with get_db_session() as db:
                        await executor.execute_cell(db, notebook_id, cell_id, correlation_id)
                    
                    process_time = time.time() - start_time
                    self.logger.info(
                        "Cell execution completed",
                        extra={
                            'correlation_id': correlation_id,
                            'notebook_id': str(notebook_id),
                            'cell_id': str(cell_id),
                            'execution_time_ms': round(process_time * 1000, 2)
                        }
                    )
                except Exception as e:
                    process_time = time.time() - start_time
                    self.logger.error(
                        "Cell execution failed",
                        extra={
                            'correlation_id': correlation_id,
                            'notebook_id': str(notebook_id),
                            'cell_id': str(cell_id),
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'execution_time_ms': round(process_time * 1000, 2)
                        },
                        exc_info=True
                    )
                # finally: # Moved outer finally
                #     # Remove from active tasks
                #     if cell_id in self.active_tasks:
                #         self.active_tasks.remove(cell_id)
                #     self.queue.task_done()
            
            except asyncio.CancelledError:
                # Queue processing has been cancelled
                self.logger.info("Execution queue processing cancelled.")
                break
            except Exception as e:
                # Log unexpected errors in the queue processing loop itself
                self.logger.error(
                    "Unexpected error in execution queue processing loop",
                    extra={
                        'error': str(e), 
                        'error_type': type(e).__name__, 
                        'notebook_id': str(notebook_id), # Log context if available
                        'cell_id': str(cell_id), 
                        'correlation_id': correlation_id
                        },
                    exc_info=True
                )
            finally:
                # Ensure task is marked done and removed from active set regardless of outcome
                if cell_id is not None and cell_id in self.active_tasks:
                    self.active_tasks.remove(cell_id)
                if notebook_id is not None: # Check if item was successfully dequeued
                    self.queue.task_done()


class CellExecutor:
    """
    Handles the execution of individual cells
    Orchestrates interaction with MCP clients
    Updates cell status and results
    """
    def __init__(self, notebook_manager, connection_manager: Optional[ConnectionManager] = None):
        self.notebook_manager = notebook_manager
        self.connection_manager = connection_manager or get_connection_manager()
        self.logger = cell_executor_logger

    async def execute_cell(self, db: Session, notebook_id: UUID, cell_id: UUID, correlation_id: Optional[str] = None) -> Any:
        """
        Execute a single cell and update its state
        
        Args:
            db: The database session.
            notebook_id: ID of the notebook containing the cell
            cell_id: ID of the cell to execute
            correlation_id: Optional correlation ID for logging
            
        Returns:
            The result of the cell execution, or None if execution failed
        """
        start_time = time.time()
        correlation_id = correlation_id or str(uuid4()) # Ensure correlation_id exists
        log_extra = {'correlation_id': correlation_id, 'notebook_id': str(notebook_id), 'cell_id': str(cell_id)}
        
        self.logger.info("Starting execution for cell", extra=log_extra)
        
        # 1. Get Notebook and Cell
        notebook = await self.notebook_manager.get_notebook(db, notebook_id)
        if not notebook:
            self.logger.error("Notebook not found", extra=log_extra)
            return None # Or raise an error
        
        cell = notebook.get_cell(cell_id)
        if not cell:
            self.logger.error("Cell not found in notebook object", extra=log_extra)
            return None # Or raise an error

        # 2. Check if already running or completed
        if cell.status == CellStatus.RUNNING:
            self.logger.warning("Cell execution requested but already running", extra=log_extra)
            return None # Or potentially wait/poll? For now, just return
        
        # Allow re-execution of SUCCESS/ERROR cells if explicitly requested (e.g., by user action)
        # For now, assume execute_cell is the start of a new execution attempt
        
        # 3. Update Status to RUNNING
        await self.notebook_manager.update_cell_status(db, notebook_id, cell_id, CellStatus.RUNNING, correlation_id)
        
        # 4. Prepare Execution Context
        # Collect results from dependencies
        dependency_results = self._get_dependency_results(notebook, cell_id)
        
        context = ExecutionContext(
            notebook=notebook,
            cell_id=cell_id,
            variables=dependency_results,
            settings=get_settings().dict() # Pass current settings
        )
        
        # 5. Execute Based on Type
        result = None
        error = None
        try:
            result = await self._execute_by_type(cell, context, correlation_id)
            status = CellStatus.SUCCESS
            self.logger.info(f"Cell executed successfully. Result type: {type(result).__name__}", extra=log_extra)
        except Exception as e:
            status = CellStatus.ERROR
            error_message = f"{type(e).__name__}: {str(e)}"
            error = { "type": type(e).__name__, "message": str(e), "traceback": str(traceback.format_exc())}
            self.logger.error(f"Cell execution failed: {error_message}", exc_info=True, extra=log_extra)
            result = None # Ensure result is None on error
            
        # 6. Update Cell State (Status, Result/Error, Timestamp)
        end_time = time.time()
        await self.notebook_manager.set_cell_result(
            db=db, 
            notebook_id=notebook_id,
            cell_id=cell_id,
            result=result,
            error=error.get("message") if error else None,
            execution_time=round((end_time - start_time), 3)
        )
        
        execution_time_ms = round((end_time - start_time) * 1000, 2)
        self.logger.info(f"Finished execution for cell. Status: {status.value}, Time: {execution_time_ms}ms", extra={**log_extra, 'status': status.value, 'execution_time_ms': execution_time_ms})
        
        # 7. Trigger Dependent Cells (if successful)
        if status == CellStatus.SUCCESS:
            # In a full dependency system, we might notify downstream cells.
            # Currently handled by ExecutionQueue/NotebookExecutor polling readiness.
            pass
            
        return result # Return the execution result or None

    async def execute_notebook(self, db: Session, notebook_id: UUID) -> None:
        """
        Executes all cells in a notebook according to their dependencies.
        (This assumes a more sophisticated execution model than just queueing)
        
        Args:
            db: The database session.
            notebook_id: ID of the notebook to execute
        """
        correlation_id = str(uuid4()) # Unique ID for the whole notebook run
        self.logger.info(f"Starting execution for entire notebook {notebook_id}", extra={'correlation_id': correlation_id})
        
        try:
            notebook = await self.notebook_manager.get_notebook(db, notebook_id)
            if not notebook:
                self.logger.error(f"Notebook {notebook_id} not found for execution.", extra={'correlation_id': correlation_id})
                return

            # Get topologically sorted order if possible, fallback to notebook order
            try:
                execution_order = notebook.get_execution_order()
                self.logger.info(f"Using dependency graph execution order for notebook {notebook_id}.", extra={'correlation_id': correlation_id})
            except Exception as e:
                self.logger.warning(f"Failed to get dependency graph order for notebook {notebook_id}: {e}. Falling back to cell_order.", extra={'correlation_id': correlation_id})
                execution_order = notebook.cell_order

            completed_cells: Set[UUID] = set()
            failed_cells: Set[UUID] = set()
            runnable_cells = set(execution_order)

            execution_cycles = 0
            max_cycles = len(runnable_cells) + 1 # Safety break

            while runnable_cells and execution_cycles < max_cycles:
                execution_cycles += 1
                executed_in_cycle = False
                cells_to_run_this_cycle = list(runnable_cells) # Copy to iterate while modifying set

                for cell_id in cells_to_run_this_cycle:
                    if cell_id in completed_cells or cell_id in failed_cells:
                        runnable_cells.discard(cell_id) # Already processed
                        continue

                    cell = notebook.get_cell(cell_id) # Assume get_cell fetches the latest
                    if not cell:
                        self.logger.warning(f"Cell {cell_id} not found during notebook execution cycle.", extra={'correlation_id': correlation_id})
                        runnable_cells.discard(cell_id)
                        failed_cells.add(cell_id) # Mark as failed if not found
                        continue

                    # Check dependencies
                    dependencies = cell.dependencies
                    all_deps_met = True
                    dependency_failed = False
                    for dep_id in dependencies:
                        if dep_id not in completed_cells:
                            all_deps_met = False
                            if dep_id in failed_cells:
                                dependency_failed = True
                            break # Stop checking dependencies for this cell
                    
                    if dependency_failed:
                        self.logger.warning(f"Skipping cell {cell_id} because dependency failed.", extra={'correlation_id': correlation_id})
                        runnable_cells.discard(cell_id)
                        failed_cells.add(cell_id)
                        continue # Skip to next cell in this cycle

                    if all_deps_met:
                        self.logger.info(f"Executing cell {cell_id} (Cycle {execution_cycles})", extra={'correlation_id': correlation_id})
                        try:
                            # Ensure cell_id is passed as UUID, although it likely already is
                            await self.execute_cell(db, notebook_id, cell_id, correlation_id)
                            # Re-fetch cell to check status after execution
                            # NOTE: get_notebook refetches the whole notebook, maybe inefficient?
                            # Consider adding get_cell to NotebookManager if needed.
                            updated_notebook = await self.notebook_manager.get_notebook(db, notebook_id)
                            updated_cell = updated_notebook.get_cell(cell_id) 
                            if updated_cell.status == CellStatus.SUCCESS:
                                completed_cells.add(cell_id)
                                self.logger.info(f"Cell {cell_id} completed successfully.", extra={'correlation_id': correlation_id})
                            else: # Includes ERROR, potentially STALE if execution was interrupted
                                failed_cells.add(cell_id)
                                self.logger.warning(f"Cell {cell_id} did not complete successfully (Status: {updated_cell.status.value}).", extra={'correlation_id': correlation_id})
                        except Exception as exec_err:
                            self.logger.error(f"Error during isolated execution of cell {cell_id}: {exec_err}", exc_info=True, extra={'correlation_id': correlation_id})
                            failed_cells.add(cell_id)
                        
                        runnable_cells.discard(cell_id)
                        executed_in_cycle = True
                        # Break inner loop to re-evaluate readiness after one execution
                        # This ensures sequential execution respecting dependencies within the cycle manager
                        break 
                
                if not executed_in_cycle and runnable_cells:
                    # If no cells were executed in a cycle but some are still runnable, implies a deadlock or issue
                    self.logger.warning(f"Notebook {notebook_id} execution stalled. Runnable but no progress: {runnable_cells}", extra={'correlation_id': correlation_id})
                    break # Avoid infinite loop

            if execution_cycles >= max_cycles:
                self.logger.error(f"Notebook {notebook_id} execution exceeded max cycles. Aborting.", extra={'correlation_id': correlation_id})

            remaining_cells = set(execution_order) - completed_cells - failed_cells
            if remaining_cells:
                self.logger.warning(f"Notebook {notebook_id} execution finished, but some cells did not run: {remaining_cells}", extra={'correlation_id': correlation_id})
            elif failed_cells:
                self.logger.warning(f"Notebook {notebook_id} execution finished with failed cells: {failed_cells}", extra={'correlation_id': correlation_id})
            else:
                self.logger.info(f"Notebook {notebook_id} execution finished successfully.", extra={'correlation_id': correlation_id})

        except Exception as outer_err:
            # NOTE: This method also needs the 'db' session passed to it
            self.logger.error(f"Error during notebook execution process for {notebook_id}: {outer_err}", exc_info=True, extra={'correlation_id': correlation_id})
    
    async def _execute_by_type(self, cell: Cell, context: ExecutionContext, correlation_id: Optional[str] = None) -> Any:
        """
        Execute a cell based on its type
        
        Args:
            cell: The cell to execute
            context: The execution context
            correlation_id: Optional correlation ID for logging
            
        Returns:
            The result of the execution
        """
        if cell.type == CellType.MARKDOWN:
            return cell.content
        
        elif cell.type == CellType.SUMMARIZATION:
            return cell.content

        elif cell.type == CellType.PYTHON:
            self.logger.info(f"Executing Python cell {cell.id} using mcp-server-data-exploration.", extra={'correlation_id': correlation_id, 'cell_id': cell.id})
            python_code = cell.content
            if not python_code:
                self.logger.warning(f"Python cell {cell.id} is empty.", extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                return {"status": "success", "stdout": "", "stderr": None} # Or an error?

            # Get StdioServerParameters for mcp-server-data-exploration
            # This is similar to PythonAgent._get_stdio_server_params
            # TODO: Consider refactoring to a shared utility if this pattern repeats
            env = os.environ.copy()
            server_params = StdioServerParameters(
                command='/root/.local/bin/uv',
                args=[
                    '--directory',
                    '/opt/mcp-server-data-exploration',
                    'run',
                    'mcp-server-ds' 
                ],
                env=env
            )
            
            # Create a client session for this execution
            # The MCPServerStdio instance from pydantic-ai is for agent use.
            # For direct client calls, we use mcp.client.stdio.stdio_client
            try:
                async with stdio_client(server_params) as (read, write): # Reverted usage
                    async with ClientSession(read, write) as session: # Reverted usage
                        self.logger.info(f"Calling 'run-script' for Python cell {cell.id}", extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                        # The mcp-server-data-exploration 'run-script' tool expects a 'script' argument.
                        await session.initialize()
                        response = await session.call_tool(
                            "run-script", # Tool name
                            arguments={"script": python_code}, # Tool arguments
                        )
                        self.logger.info(f"Received response from 'run-script' for cell {cell.id}: {response}", extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                        
                        # Assuming the response from 'run-script' is a dict like:
                        # {"stdout": "...", "stderr": "...", "status": "success/error", ...}
                        # We can adapt PythonAgent's _parse_python_mcp_output or simplify
                        if isinstance(response, dict):
                            # Basic parsing, similar to a simplified _parse_python_mcp_output
                            parsed_result = {
                                "status": response.get("status", "success" if response.get("stdout") is not None else "error"),
                                "stdout": response.get("stdout"),
                                "stderr": response.get("stderr"),
                                # Add other fields if mcp-server-data-exploration provides them
                            }
                            if parsed_result["status"] == "error" and parsed_result["stderr"] and not response.get("error_details"):
                                parsed_result["error_details"] = {
                                    "type": "PythonExecutionError",
                                    "message": str(parsed_result["stderr"]).splitlines()[-1] if parsed_result["stderr"] else "Unknown Python error",
                                    "traceback": str(parsed_result["stderr"])
                                }
                            return parsed_result
                        else:
                            # Fallback if response is not a dict (e.g., just a string)
                            self.logger.warning(f"Unexpected response type from 'run-script' for cell {cell.id}: {type(response)}. Treating as stdout.", extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                            return {"status": "success", "stdout": str(response), "stderr": None}

            except McpError as mcp_e:
                self.logger.error(f"McpError during Python cell {cell.id} execution: {mcp_e}", exc_info=True, extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                raise # Re-raise to be caught by the outer try-except
            except Exception as e:
                self.logger.error(f"Unexpected error during Python cell {cell.id} execution with mcp-server-data-exploration: {e}", exc_info=True, extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                raise # Re-raise to be caught by the outer try-except
        
        elif cell.type == CellType.LOG_AI:
            # Execute Log-AI tool calls (FastMCP server)
            self.logger.info(f"Executing Log-AI cell {cell.id} via logai MCP server.", extra={'correlation_id': correlation_id, 'cell_id': cell.id})

            # For Log-AI cells we expect they encode a tool call directly (tool_name + tool_arguments)
            if not cell.tool_name:
                raise ValueError("Log-AI cell requires tool_name to be set.")

            docker_args = [
                'run',
                '--rm',
                '-i', # Keep STDIN open for MCP protocol
                "--volume=/var/run/docker.sock:/var/run/docker.sock",
            ]

            host_fs_root = os.environ.get("SHERLOG_HOST_FS_ROOT")
            if host_fs_root:
                docker_args.append("-v")
                docker_args.append(f"{host_fs_root}:{host_fs_root}:ro")

            docker_args.append('ghcr.io/navneet-mkr/logai-mcp:0.1.3')

            server_params = StdioServerParameters(
                command='docker',
                args=docker_args,
                env=os.environ.copy()
            )

            try:
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self.logger.info(f"Calling Log-AI tool '{cell.tool_name}' for cell {cell.id}…", extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                        mcp_result = await session.call_tool(cell.tool_name, arguments=cell.tool_arguments or {})
                        self.logger.info(f"Log-AI tool '{cell.tool_name}' call completed for cell {cell.id}.", extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                        return mcp_result
            except Exception as e:
                self.logger.error(f"Log-AI MCP execution error for cell {cell.id}: {e}", exc_info=True, extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                raise
        
        # Check if this cell represents a tool call (This is the original logic for other tool calls)
        elif cell.tool_name and cell.tool_arguments is not None:
            tool_name = cell.tool_name
            tool_args = cell.tool_arguments
            connection_id = cell.connection_id 

            if not connection_id:
                error_msg = f"Cell {cell.id} (type: {cell.type}) requires a connection_id but none was found."
                self.logger.error(error_msg, extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                raise ValueError(error_msg)

            # Fetch connection details using ConnectionManager
            connection_config = await self.connection_manager.get_connection(str(connection_id))
            if not connection_config:
                error_msg = f"Failed to get connection details for ID {connection_id}."
                self.logger.error(error_msg, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'connection_id': connection_id})
                raise ConnectionError(error_msg)
            
            if not connection_config.config:
                error_msg = f"Connection config is missing for ID {connection_id}."
                self.logger.error(error_msg, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'connection_id': connection_id})
                raise ConnectionError(error_msg)

            # Get the appropriate handler based on connection type
            try:
                handler = get_handler(connection_config.type)
            except ValueError as e:
                error_msg = f"No connection handler found for type '{connection_config.type}' (connection: {connection_id}): {e}"
                self.logger.error(error_msg, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'connection_id': connection_id, 'connection_type': connection_config.type})
                raise NotImplementedError(error_msg)

            # Delegate execution to the handler
            try:
                self.logger.info(f"Delegating tool execution to handler: {handler.__class__.__name__}", extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'tool_name': tool_name, 'connection_type': connection_config.type})
                mcp_result = await handler.execute_tool_call(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    db_config=connection_config.config,
                    correlation_id=correlation_id
                )
                self.logger.info(f"Handler execution successful for cell {cell.id}. Result type: {type(mcp_result)}", extra={'correlation_id': correlation_id})
                return mcp_result
            except (ValueError, ConnectionError, McpError, NotImplementedError) as handler_error:
                 # Errors related to config, connection, or tool execution within the handler
                 error_msg = f"Handler error executing tool '{tool_name}' for cell {cell.id}: {handler_error}"
                 self.logger.error(error_msg, exc_info=True, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'tool_name': tool_name, 'error_type': type(handler_error).__name__})
                 raise # Re-raise the specific error from the handler
            except Exception as unexpected_handler_error:
                 # Catch-all for truly unexpected errors within the handler
                 error_msg = f"Unexpected handler error executing tool '{tool_name}' for cell {cell.id}: {unexpected_handler_error}"
                 self.logger.error(error_msg, exc_info=True, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'tool_name': tool_name})
                 raise Exception(error_msg) from unexpected_handler_error # Wrap in generic Exception
            
            # --- Remove Old GitHub specific logic --- 
            # if cell.type == CellType.GITHUB: 
            #    ... (old logic was here) ...
            # else:
            #     raise NotImplementedError(f"Execution via MCP not implemented for cell type/source: {cell.type}")

        else:
            self.logger.warning(f"Cell {cell.id} of type {cell.type} has no execution logic defined and is not a tool call. Returning content.", extra={'cell_id': cell.id})
            return cell.content # Default fallback
    
    def _get_dependency_results(self, notebook: Notebook, cell_id: UUID) -> Dict[str, Any]:
        """
        Get the results of cells that this cell depends on
        
        Args:
            notebook: The notebook containing the cell
            cell_id: The ID of the cell
            
        Returns:
            Dictionary mapping dependency cell IDs to their results
        """
        cell = notebook.get_cell(cell_id)
        variables = {}
        
        for dep_id in cell.dependencies:
            if dep_id in notebook.cells:
                dep_cell = notebook.cells[dep_id]
                if dep_cell.result:
                    variables[str(dep_id)] = dep_cell.result.content
        
        return variables

# --- Utility ---
import traceback # Moved import here to be closer to usage
