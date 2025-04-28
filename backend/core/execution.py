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
from backend.services.connection_manager import get_github_stdio_params
from backend.core.cell import GitHubCell
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from mcp.shared.exceptions import McpError
import logging

from backend.core.types import ToolCallRecord

# Initialize loggers
execution_logger = logging.getLogger("execution")
cell_executor_logger = logging.getLogger("cell_executor")


# --- Tool Call Representation ---

def register_tool_dependencies(
    tool_call_record: ToolCallRecord, 
    notebook: Notebook, 
    step_id_to_record_id: Dict[str, UUID]
) -> None:
    """
    Inspects a tool call record's parameters for the special '__ref__' structure 
    indicating a dependency on a previous step's output, and registers the 
    corresponding dependencies in the notebook's dependency graph using ToolCallIDs.

    Args:
        tool_call_record: The ToolCallRecord whose parameters should be inspected.
        notebook: The notebook instance containing the dependency graph.
        step_id_to_record_id: Mapping from the planner's step_id to the ToolCallRecord UUID.
    """
    if not tool_call_record.parameters:
        return

    dependency_graph = notebook.dependency_graph
    dependent_tool_id = tool_call_record.id

    def find_and_register_refs(value: Any):
        """Recursive helper to find __ref__ dicts and register dependencies."""
        if isinstance(value, dict):
            if '__ref__' in value:
                ref_data = value['__ref__']
                if isinstance(ref_data, dict) and 'step_id' in ref_data and 'output_name' in ref_data:
                    source_step_id = ref_data['step_id']
                    output_name = ref_data['output_name']
                    
                    # Look up the actual ToolCallID using the step_id
                    if source_step_id in step_id_to_record_id:
                        source_tool_id = step_id_to_record_id[source_step_id]
                        
                        # Ensure the source tool record exists (optional, for robustness)
                        if source_tool_id not in notebook.tool_call_records:
                            execution_logger.warning(
                                f"Source tool call record {source_tool_id} (from step {source_step_id}) not found when registering dependency for {dependent_tool_id}",
                                extra={'notebook_id': str(notebook.id), 'dependent_tool_id': str(dependent_tool_id), 'source_step_id': source_step_id, 'source_tool_id': str(source_tool_id)}
                            )
                            # Continue checking other parts of the structure even if one source is missing
                        else:
                            # Register the dependency if not already registered (avoid duplicates from nested calls)
                            if source_tool_id not in dependency_graph.get_tool_dependencies(dependent_tool_id):
                                dependency_graph.add_tool_dependency(
                                    dependent_id=dependent_tool_id,
                                    dependency_id=source_tool_id
                                )
                                execution_logger.info(
                                    f"Registered tool dependency: {dependent_tool_id} depends on {source_tool_id} (step: {source_step_id}, output: {output_name})",
                                    extra={'notebook_id': str(notebook.id), 'dependent_tool_id': str(dependent_tool_id), 'source_tool_id': str(source_tool_id), 'source_step_id': source_step_id, 'output_name': output_name}
                                )
                    else:
                        execution_logger.warning(
                            f"Step ID '{source_step_id}' referenced by tool {dependent_tool_id} not found in step_id_to_record_id map.",
                             extra={'notebook_id': str(notebook.id), 'dependent_tool_id': str(dependent_tool_id), 'source_step_id': source_step_id}
                        )
            else: # Recurse into dict values if it's not a ref dict itself
                for k, v in value.items():
                    find_and_register_refs(v)
        elif isinstance(value, list):
            for item in value:
                find_and_register_refs(item)

    # Iterate through the top-level parameters
    for param_name, param_value in tool_call_record.parameters.items():
        # TODO: Handle potential nested references (e.g., in lists or dicts within params)
        # Check for the specific reference structure
        find_and_register_refs(param_value)
        # The old code below is now handled by the recursive helper
        # if isinstance(param_value, dict) and '__ref__' in param_value:
        #     ref_data = param_value['__ref__']
        #     ... rest of the previous logic ...


def propagate_tool_results(
    completed_record: ToolCallRecord, 
    notebook: Notebook, 
    step_id_to_record_id: Dict[str, UUID] # Added map, though not strictly needed here yet
):
    """
    Finds tool calls dependent on the completed tool call, updates their parameters
    by replacing the '__ref__' structure with the actual result, and marks the 
    associated cells as stale.

    Args:
        completed_record: The ToolCallRecord that has successfully completed.
        notebook: The notebook instance containing the state.
        step_id_to_record_id: Mapping from planner step_id to ToolCallRecord UUID (passed for consistency, may be used later).
    """
    if completed_record.status != "success" or not completed_record.named_outputs:
        # Only propagate successful results with outputs
        return

    dependency_graph = notebook.dependency_graph
    dependents = dependency_graph.get_tool_dependents(completed_record.id)

    execution_logger.info(
        f"Propagating results from tool {completed_record.id} to {len(dependents)} dependents.",
        extra={'notebook_id': str(notebook.id), 'source_tool_id': str(completed_record.id), 'dependent_count': len(dependents)}
    )

    cells_to_mark_stale: Set[UUID] = set()

    for dependent_id in dependents:
        if dependent_id not in notebook.tool_call_records:
            execution_logger.warning(
                f"Dependent tool call record {dependent_id} not found during propagation from {completed_record.id}",
                extra={'notebook_id': str(notebook.id), 'source_tool_id': str(completed_record.id), 'dependent_tool_id': str(dependent_id)}
            )
            continue
        
        dependent_record = notebook.tool_call_records[dependent_id]
        needs_update = False

        for param_name, param_value in list(dependent_record.parameters.items()): # Iterate over copy
            # Check for the reference structure pointing to the completed record
            if isinstance(param_value, dict) and '__ref__' in param_value:
                 ref_data = param_value['__ref__']
                 if isinstance(ref_data, dict) and 'step_id' in ref_data and 'output_name' in ref_data:
                    source_step_id = ref_data['step_id']
                    # Check if this reference points to the completed step's ID via the map
                    if source_step_id in step_id_to_record_id and step_id_to_record_id[source_step_id] == completed_record.id:
                        output_name = ref_data['output_name']
                        if output_name in completed_record.named_outputs:
                            actual_value = completed_record.named_outputs[output_name]
                            # Replace the reference dict with the actual value
                            dependent_record.parameters[param_name] = actual_value
                            needs_update = True
                            execution_logger.info(
                                f"Updated parameter '{param_name}' in dependent tool {dependent_id} with output '{output_name}' from {completed_record.id} (step: {source_step_id})",
                                extra={
                                    'notebook_id': str(notebook.id), 
                                    'source_tool_id': str(completed_record.id),
                                    'source_step_id': source_step_id,
                                    'dependent_tool_id': str(dependent_id),
                                    'parameter_name': param_name, 
                                    'output_name': output_name
                                }
                            )
                        else:
                            execution_logger.warning(
                                f"Output '{output_name}' not found in completed tool {completed_record.id} (step: {source_step_id}) for dependent {dependent_id}",
                                extra={
                                    'notebook_id': str(notebook.id),
                                    'source_tool_id': str(completed_record.id),
                                    'source_step_id': source_step_id,
                                    'dependent_tool_id': str(dependent_id),
                                    'output_name': output_name
                                }
                            )
                            # Decide how to handle missing output: error, leave reference, set to None? Mark stale anyway.
                            needs_update = True # Mark stale even if output name is missing
        
        if needs_update:
            # Mark the *cell* containing the dependent tool call as stale
            # The execution engine should then re-evaluate the cell, which might re-run the tool call
            if dependent_record.parent_cell_id in notebook.cells:
                cells_to_mark_stale.add(dependent_record.parent_cell_id)
            else:
                 execution_logger.warning(
                    f"Parent cell {dependent_record.parent_cell_id} for tool {dependent_id} not found in notebook.",
                    extra={'notebook_id': str(notebook.id), 'tool_id': str(dependent_id), 'cell_id': str(dependent_record.parent_cell_id)}
                 )

    # Mark affected cells as stale (and potentially their dependents)
    stale_cells = set()
    for cell_id in cells_to_mark_stale:
        stale_cells.update(notebook.mark_dependents_stale(cell_id)) # Use existing cell-level stale propagation
        # Also mark the cell itself stale if it wasn't already caught by dependents chain
        if cell_id in notebook.cells:
            cell = notebook.cells[cell_id]
            if cell.status != CellStatus.STALE:
                 cell.mark_stale()
                 stale_cells.add(cell_id)

    if stale_cells:
        execution_logger.info(
            f"Marked {len(stale_cells)} cells stale due to tool propagation from {completed_record.id}.",
            extra={'notebook_id': str(notebook.id), 'source_tool_id': str(completed_record.id), 'stale_cell_count': len(stale_cells)}
        )
        # Here, you would typically notify the frontend or trigger the execution engine
        # to re-run the stale cells based on the dependency graph order.
        # e.g., await notebook_manager.schedule_execution(notebook.id, list(stale_cells))


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
    def __init__(self, notebook_manager):
        self.notebook_manager = notebook_manager
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
            
            self.logger.debug(
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
            result = await self._execute_by_type(cell, context)
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
        Execute all cells in a notebook in dependency order
        
        Args:
            notebook_id: ID of the notebook to execute
        """
        notebook = self.notebook_manager.get_notebook(notebook_id)
        execution_order = notebook.get_execution_order()
        
        for cell_id in execution_order:
            await self.execute_cell(notebook_id, cell_id)
    
    async def _execute_by_type(self, cell: Cell, context: ExecutionContext) -> Any:
        """
        Execute a cell based on its type
        
        Args:
            cell: The cell to execute
            context: The execution context
            
        Returns:
            The result of the execution
        """
        if cell.type == CellType.MARKDOWN:
            return cell.content
        
        elif cell.type == CellType.GITHUB:
            if not isinstance(cell, GitHubCell):
                 self.logger.error(f"Cell {cell.id} has type GITHUB but is not a GitHubCell instance.")
                 raise TypeError("Incorrect cell type instance for GITHUB execution.")
                 
            github_cell: GitHubCell = cell # Type hint for clarity
            tool_name = github_cell.tool_name
            tool_args = github_cell.tool_arguments

            if not tool_name or tool_args is None: # Check for None specifically for args dict
                self.logger.warning(f"GitHub cell {github_cell.id} missing tool_name or tool_arguments for execution. Returning current content.")
                # Maybe return an error or the potentially outdated content?
                return github_cell.content # Fallback to content if execution info missing
                
            self.logger.info(f"Executing GitHub cell {github_cell.id}: tool='{tool_name}', args={tool_args}")
            stdio_params = await get_github_stdio_params()
            
            if not stdio_params:
                error_msg = "Failed to get GitHub MCP stdio parameters for execution."
                self.logger.error(error_msg, extra={'cell_id': github_cell.id})
                raise ConnectionError(error_msg)
                
            mcp_result = None
            try:
                async with stdio_client(stdio_params) as (read, write):
                    # Optional: Add sampling callback if needed
                    async with ClientSession(read, write) as session:
                        await session.initialize() 
                        # Ensure arguments are passed correctly
                        mcp_result = await session.call_tool(tool_name, arguments=tool_args)
                        self.logger.info(f"GitHub MCP call successful for cell {github_cell.id}. Result type: {type(mcp_result)}")
                return mcp_result # Return the direct result from MCP
            except McpError as e:
                error_msg = f"MCP Error executing GitHub cell {github_cell.id}: {e}"
                self.logger.error(error_msg, exc_info=True, extra={'cell_id': github_cell.id, 'tool_name': tool_name})
                raise # Re-raise McpError to be caught by the main execute_cell handler
            except Exception as e:
                error_msg = f"Unexpected Error executing GitHub cell {github_cell.id}: {e}"
                self.logger.error(error_msg, exc_info=True, extra={'cell_id': github_cell.id, 'tool_name': tool_name})
                raise # Re-raise other errors
            
        elif cell.type == CellType.SUMMARIZATION:
            return cell.content
        
        else:
            raise ValueError(f"Unknown cell type: {cell.type}")
    
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