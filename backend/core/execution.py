import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from backend.core.cell import Cell, CellStatus
from backend.core.notebook import Notebook
import logging


@dataclass
class ExecutionContext:
    """Context for cell execution with access to notebook state"""
    notebook: Notebook
    cell_id: UUID
    variables: Dict[str, Any] = field(default_factory=dict)
    mcp_clients: Dict[str, Any] = field(default_factory=dict)


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
        self.logger = logging.getLogger("execution")
    
    async def add_cell(self, notebook_id: UUID, cell_id: UUID, executor: 'CellExecutor') -> None:
        """Add a cell to the execution queue"""
        if cell_id in self.active_tasks:
            self.logger.warning(f"Cell already in execution queue notebook_id={str(notebook_id)} cell_id={str(cell_id)}")
            return
        
        self.logger.info(f"Adding cell to execution queue notebook_id={str(notebook_id)} cell_id={str(cell_id)}")
        await self.queue.put((notebook_id, cell_id, executor))
    
    async def start(self) -> None:
        """Start processing the execution queue"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
        self.logger.info("Execution queue started")
    
    async def stop(self) -> None:
        """Stop processing the execution queue"""
        if not self._running:
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
        self.logger.info("Execution queue stopped")
    
    async def _process_queue(self) -> None:
        """Process items in the execution queue"""
        while self._running:
            try:
                # Get the next cell to execute
                notebook_id, cell_id, executor = await self.queue.get()
                
                # Skip if cell is already being processed
                if cell_id in self.active_tasks:
                    self.logger.warning(f"Cell already being processed, skipping notebook_id={str(notebook_id)} cell_id={str(cell_id)}")
                    self.queue.task_done()
                    continue
                
                # Track active task
                self.active_tasks.add(cell_id)
                
                try:
                    self.logger.info(f"Starting cell execution notebook_id={str(notebook_id)} cell_id={str(cell_id)}")
                    
                    # Execute the cell
                    await executor.execute_cell(notebook_id, cell_id)
                    
                    self.logger.info(f"Cell execution completed notebook_id={str(notebook_id)} cell_id={str(cell_id)}")
                except Exception as e:
                    self.logger.error(f"Cell execution failed notebook_id={str(notebook_id)} cell_id={str(cell_id)} error={str(e)}")
                finally:
                    # Remove from active tasks
                    self.active_tasks.remove(cell_id)
                    self.queue.task_done()
            
            except asyncio.CancelledError:
                # Queue processing has been cancelled
                break
            except Exception as e:
                # Unexpected error, log and continue
                self.logger.error(f"Error in execution queue processing error={str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop on error


class CellExecutor:
    """
    Executes cells and manages their execution state
    Different cell types will have different execution strategies
    """
    def __init__(self, notebook_manager):
        self.notebook_manager = notebook_manager
        self.logger = logging.getLogger("cell_executor")
    
    async def execute_cell(self, notebook_id: UUID, cell_id: UUID) -> Any:
        """
        Execute a cell and update its state
        
        Args:
            notebook_id: ID of the notebook containing the cell
            cell_id: ID of the cell to execute
            
        Returns:
            The result of the cell execution
        """
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
        
        execution_id = str(uuid4())
        self.logger.info(f"Executing cell notebook_id={str(notebook_id)} cell_id={str(cell_id)} cell_type={cell.type.value} execution_id={execution_id}")
        
        try:
            # Create execution context
            context = ExecutionContext(
                notebook=notebook,
                cell_id=cell_id,
                variables=self._get_dependency_results(notebook, cell_id)
            )
            
            # Execute the cell based on its type
            result = await self._execute_by_type(cell, context)
            
        except Exception as e:
            # Capture execution error
            error = str(e)
            print(f"Error executing cell {cell_id}: {error}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Update cell status and result
        cell.set_result(result, error, execution_time)
        
        # Notify that the cell has been updated
        await self.notebook_manager.notify_cell_update(notebook_id, cell_id)
        
        self.logger.info(f"Cell execution completed notebook_id={str(notebook_id)} cell_id={str(cell_id)} execution_id={execution_id} execution_time={execution_time}")
        
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
                connection_type = "postgres"
            elif cell.type == CellType.LOG:
                connection_type = "loki"
            elif cell.type == CellType.METRIC:
                connection_type = "prometheus"
            elif cell.type == CellType.S3:
                connection_type = "s3"
            
            if connection_type:
                # Get connections of this type
                connections = connection_manager.get_connections_by_type(connection_type)
                
                # Start any MCP servers needed for these connections
                server_addresses = await mcp_manager.start_mcp_servers(connections)
                
                # Create basic client objects for each connection/server
                for conn in connections:
                    if conn.id in server_addresses:
                        # Create a simple client object for the MCP server
                        # This is a placeholder - a real implementation would create actual client objects
                        mcp_clients[f"mcp_{conn.id}"] = {
                            "connection_id": conn.id,
                            "address": server_addresses[conn.id],
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
        
        elif cell.type == CellType.S3:
            from backend.core.executors.s3_executor import execute_s3_cell
            context.mcp_clients = mcp_clients
            return await execute_s3_cell(cell, context)
        
        elif cell.type == CellType.MARKDOWN:
            # Markdown cells don't need execution
            return cell.content
        
        elif cell.type == CellType.AI_QUERY:
            from backend.ai.agent import execute_ai_query_cell
            return await execute_ai_query_cell(cell, context)
        
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