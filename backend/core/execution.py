import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from backend.core.cell import Cell, CellStatus, CellType, GitHubCell
from backend.core.notebook import Notebook
from backend.core.dependency import ToolCallID
from backend.services.connection_manager import get_github_stdio_params, ConnectionManager, get_connection_manager
from backend.core.cell import GitHubCell
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from mcp.shared.exceptions import McpError
import logging

# Import StdioServerParameters
from mcp import StdioServerParameters

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
                    self.queue.task_done()
                    continue
                
                # Track active task
                self.active_tasks.add(cell_id)
                start_time = time.time()
                
                try:
                    self.logger.info(
                        "Starting cell execution",
                        extra={
                            'correlation_id': correlation_id,
                            'notebook_id': str(notebook_id),
                            'cell_id': str(cell_id),
                            'active_tasks': len(self.active_tasks),
                            'queue_size': self.queue.qsize()
                        }
                    )
                    
                    # Execute the cell
                    await executor.execute_cell(notebook_id, cell_id, correlation_id)
                    
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
                finally:
                    # Remove from active tasks
                    self.active_tasks.remove(cell_id)
                    self.queue.task_done()
            
            except asyncio.CancelledError:
                # Queue processing has been cancelled
                break
            except Exception as e:
                self.logger.error(
                    "Error in execution queue processing",
                    extra={
                        'correlation_id': str(uuid4()),
                        'error': str(e),
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )
                await asyncio.sleep(1)  # Prevent tight loop on error


class CellExecutor:
    """
    Executes cells and manages their execution state
    Different cell types will have different execution strategies
    """
    def __init__(self, notebook_manager, connection_manager: Optional[ConnectionManager] = None):
        self.notebook_manager = notebook_manager
        self.connection_manager = connection_manager or get_connection_manager() # Use injected or singleton
        self.logger = cell_executor_logger
    
    async def execute_cell(self, notebook_id: UUID, cell_id: UUID, correlation_id: Optional[str] = None) -> Any:
        """
        Execute a cell and update its state
        
        Args:
            notebook_id: ID of the notebook containing the cell
            cell_id: ID of the cell to execute
            correlation_id: Optional correlation ID for tracing execution
            
        Returns:
            The result of the cell execution
        """
        correlation_id = correlation_id if correlation_id else str(uuid4())
        
        notebook = self.notebook_manager.get_notebook(notebook_id)
        cell = notebook.get_cell(cell_id)
        
        # Update cell status
        cell.status = CellStatus.RUNNING
        
        # Notify that the cell is running
        await self.notebook_manager.notify_cell_update(notebook_id, cell_id)
        
        # Track execution time
        start_time = time.time()
        result = None
        error = None
        
        self.logger.info(
            "Starting cell execution",
            extra={
                'correlation_id': correlation_id,
                'notebook_id': str(notebook_id),
                'cell_id': str(cell_id),
                'cell_type': cell.type.value,
                'dependencies_count': len(cell.dependencies) if hasattr(cell, 'dependencies') else 0
            }
        )
        
        try:
            # Create execution context
            # Get cell settings
            settings = {}
            if hasattr(cell, 'settings') and cell.settings:
                settings = cell.settings
            
            # Get dependencies
            dependency_start = time.time()
            variables = self._get_dependency_results(notebook, cell_id)
            dependency_time = time.time() - dependency_start
            
            self.logger.info(
                "Dependencies resolved",
                extra={
                    'correlation_id': correlation_id,
                    'notebook_id': str(notebook_id),
                    'cell_id': str(cell_id),
                    'dependency_count': len(variables),
                    'resolution_time_ms': round(dependency_time * 1000, 2)
                }
            )
            
            context = ExecutionContext(
                notebook=notebook,
                cell_id=cell_id,
                variables=variables,
                settings=settings
            )
            
            # Execute the cell based on its type
            execution_start = time.time()
            result = await self._execute_by_type(cell, context, correlation_id)
            execution_time = time.time() - execution_start
            
            self.logger.info(
                "Cell execution successful",
                extra={
                    'correlation_id': correlation_id,
                    'notebook_id': str(notebook_id),
                    'cell_id': str(cell_id),
                    'execution_time_ms': round(execution_time * 1000, 2)
                }
            )
            
        except Exception as e:
            # Capture execution error
            error = str(e)
            self.logger.error(
                "Cell execution failed",
                extra={
                    'correlation_id': correlation_id,
                    'notebook_id': str(notebook_id),
                    'cell_id': str(cell_id),
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
        
        # Calculate total execution time
        total_execution_time = time.time() - start_time
        
        # Update cell status and result
        cell.set_result(result, error, total_execution_time)
        
        # Notify that the cell has been updated
        await self.notebook_manager.notify_cell_update(notebook_id, cell_id)
        
        self.logger.info(
            "Cell processing completed",
            extra={
                'correlation_id': correlation_id,
                'notebook_id': str(notebook_id),
                'cell_id': str(cell_id),
                'status': cell.status.value,
                'total_time_ms': round(total_execution_time * 1000, 2),
                'has_error': error is not None
            }
        )
        
        return result
    
    async def execute_notebook(self, notebook_id: UUID) -> None:
        """
        Execute all runnable cells in a notebook, respecting dependencies.
        
        Args:
            notebook_id: ID of the notebook to execute
        """
        correlation_id = str(uuid4())
        self.logger.info(f"Starting execution for notebook {notebook_id}", extra={'correlation_id': correlation_id})

        try:
            notebook = self.notebook_manager.get_notebook(notebook_id)
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
                            await self.execute_cell(notebook_id, cell_id, correlation_id)
                            # Re-fetch cell to check status after execution
                            updated_cell = notebook.get_cell(cell_id) 
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
        # TODO: Inject or retrieve MCP clients based on cell.source or type
        # mcp_clients = context.mcp_clients # Assuming clients are passed in context
        
        if cell.type == CellType.MARKDOWN:
            return cell.content # Markdown just displays content
        
        elif cell.type == CellType.SUMMARIZATION:
            # Currently, summarization cells just hold content/results generated by the agent.
            # If they needed active execution (e.g., calling a summarization API), 
            # that logic would go here, likely using an MCP client.
            return cell.content
        
        # Check if this cell represents a tool call
        elif cell.tool_name and cell.tool_arguments is not None:
            tool_name = cell.tool_name
            tool_args = cell.tool_arguments
            # Determine which MCP client to use based on cell type or source
            # For now, hardcoding for GitHub as it's the only tool type left
            connection_id = cell.connection_id # Get connection ID from cell

            if cell.type == CellType.GITHUB:
                self.logger.info(f"Executing GitHub tool via MCP: cell={cell.id}, tool='{tool_name}', args={tool_args}, connection_id={connection_id}", extra={'correlation_id': correlation_id})
                
                if not connection_id:
                    error_msg = f"GitHub cell {cell.id} is missing a connection_id."
                    self.logger.error(error_msg, extra={'correlation_id': correlation_id, 'cell_id': cell.id})
                    raise ValueError(error_msg)

                # Fetch connection details using ConnectionManager
                connection_config = await self.connection_manager.get_connection(str(connection_id))
                if not connection_config or connection_config.type != 'github':
                    error_msg = f"Failed to get valid GitHub connection details for ID {connection_id}."
                    self.logger.error(error_msg, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'connection_id': connection_id})
                    raise ConnectionError(error_msg)

                # Prepare StdioServerParameters from connection config
                cmd = connection_config.config.get('command')
                args = connection_config.config.get('args')
                env = connection_config.config.get('env') # Get env vars which might include PAT

                if not cmd or not args:
                    error_msg = f"GitHub connection {connection_id} is missing command or args in config."
                    self.logger.error(error_msg, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'connection_id': connection_id})
                    raise ConnectionError(error_msg)

                stdio_params = StdioServerParameters(command=cmd, args=args, env=env)

                # Execute using stdio_client context manager
                mcp_result = None
                try:
                    async with stdio_client(stdio_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            mcp_result = await session.call_tool(tool_name, arguments=tool_args)
                            self.logger.info(f"GitHub MCP call successful for cell {cell.id}. Result type: {type(mcp_result)}", extra={'correlation_id': correlation_id})
                    return mcp_result
                except McpError as e:
                    error_msg = f"MCP Error executing tool '{tool_name}' for cell {cell.id}: {e}"
                    self.logger.error(error_msg, exc_info=True, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'tool_name': tool_name})
                    raise # Re-raise to be caught by execute_cell
                except Exception as e:
                    error_msg = f"Unexpected Error executing tool '{tool_name}' for cell {cell.id}: {e}"
                    self.logger.error(error_msg, exc_info=True, extra={'correlation_id': correlation_id, 'cell_id': cell.id, 'tool_name': tool_name})
                    raise # Re-raise
            else:
                # Handle other potential tool sources if added later
                raise NotImplementedError(f"Execution via MCP not implemented for cell type/source: {cell.type}")
        
        # If not Markdown, Summarization, or a recognized tool call cell
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