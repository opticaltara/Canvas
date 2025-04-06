import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from backend.core.cell import Cell, CellStatus
from backend.core.notebook import Notebook
import logging

# Initialize loggers
execution_logger = logging.getLogger("execution")
cell_executor_logger = logging.getLogger("cell_executor")

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
        from backend.core.cell import CellType
        from backend.mcp.manager import get_mcp_server_manager
        from backend.services.connection_manager import get_connection_manager
        
        # Set up MCP connections if needed
        mcp_clients = {}
        if cell.type in [CellType.SQL, CellType.LOG, CellType.METRIC, CellType.S3]:
            # Get connection manager and MCP server manager
            connection_manager = get_connection_manager()
            mcp_manager = get_mcp_server_manager()
            
            # Get relevant connections based on cell type
            connection_type = None
            if cell.type == CellType.SQL:
                connection_type = "sql"
            elif cell.type == CellType.LOG or cell.type == CellType.METRIC:
                connection_type = "grafana"
            elif cell.type == CellType.S3:
                connection_type = "s3"
            
            if connection_type:
                # Use the cell's connection_id if available
                if cell.connection_id is not None:
                    # Get the specific connection
                    connection = connection_manager.get_connection(str(cell.connection_id))
                    if connection:
                        # Start MCP server for this specific connection
                        server_addresses = await mcp_manager.start_mcp_servers([connection])
                        
                        # Create client object for this connection
                        if connection.id in server_addresses:
                            mcp_clients[f"mcp_{connection.id}"] = {
                                "connection_id": connection.id,
                                "address": server_addresses[connection.id],
                                "query_db": lambda query, params: {"rows": [], "columns": []},
                                "query_logs": lambda query, params: {"data": []},
                                "query_metrics": lambda query, params: {"data": []},
                                "s3_list_objects": lambda bucket, prefix: {"data": []},
                                "s3_get_object": lambda bucket, key: {"data": ""},
                                "s3_select_object": lambda bucket, key, query: {"data": []},
                            }
                else:
                    # If no specific connection ID, get the default connection
                    connection = connection_manager.get_default_connection(connection_type)
                    if connection:
                        # Start MCP server for the default connection
                        server_addresses = await mcp_manager.start_mcp_servers([connection])
                        
                        # Create client object for this connection
                        if connection.id in server_addresses:
                            mcp_clients[f"mcp_{connection.id}"] = {
                                "connection_id": connection.id,
                                "address": server_addresses[connection.id],
                                "query_db": lambda query, params: {"rows": [], "columns": []},
                                "query_logs": lambda query, params: {"data": []},
                                "query_metrics": lambda query, params: {"data": []},
                                "s3_list_objects": lambda bucket, prefix: {"data": []},
                                "s3_get_object": lambda bucket, key: {"data": ""},
                                "s3_select_object": lambda bucket, key, query: {"data": []},
                            }
        
        # Dispatch to the appropriate executor based on cell type
        if cell.type == CellType.PYTHON:
            from backend.core.executors.python_executor import execute_python_cell
            return await execute_python_cell(cell, context)
        
        elif cell.type == CellType.SQL:
            from backend.core.executors.sql_executor import execute_sql_cell
            context.mcp_clients = mcp_clients
            return await execute_sql_cell(cell, context)
        
        elif cell.type == CellType.LOG:
            from backend.core.executors.log_executor import execute_log_cell
            context.mcp_clients = mcp_clients
            return await execute_log_cell(cell, context)
        
        elif cell.type == CellType.METRIC:
            from backend.core.executors.metric_executor import execute_metric_cell
            context.mcp_clients = mcp_clients
            return await execute_metric_cell(cell, context)
        
        elif cell.type == CellType.MARKDOWN:
            # Markdown cells don't need execution
            return cell.content
        
        elif cell.type == CellType.AI_QUERY:
            from backend.ai.agent import execute_ai_query_cell
            from backend.core.cell import AIQueryCell
            return await execute_ai_query_cell(AIQueryCell(**cell.model_dump()), context)
        
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