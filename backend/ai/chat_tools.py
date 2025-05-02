"""
Chat Agent Tools

This module defines tools that allow the AI agent to manage notebook cells.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID

import logging
from pydantic import BaseModel, Field
from pydantic_ai.tools import Tool
from sqlalchemy.orm import Session

from backend.core.cell import CellType, CellResult, CellStatus
from backend.core.types import ToolCallID
from backend.services.notebook_manager import NotebookManager
from backend.db.database import get_db

# Initialize logger
tools_logger = logging.getLogger("ai.chat_tools")

# Add correlation ID filter to the logger
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

tools_logger.addFilter(CorrelationIdFilter())


class CreateCellParams(BaseModel):
    """Parameters for creating a cell"""
    notebook_id: str = Field(description="The UUID of the notebook")
    cell_type: str = Field(description="Type of cell to create: markdown, python, sql, log, metric, s3, ai_query")
    content: str = Field(description="Content for the cell (query, code, text)")
    position: Optional[int] = Field(description="Position to insert the cell (optional)", default=None)
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata for the cell", default=None)
    dependencies: Optional[List[UUID]] = Field(description="List of cell UUIDs this cell depends on", default=None)
    tool_call_id: ToolCallID = Field(description="ID of the associated tool call (Generated UUID)")
    tool_name: Optional[str] = Field(description="Name of the tool called", default=None)
    tool_arguments: Optional[Dict[str, Any]] = Field(description="Arguments passed to the tool", default=None)
    result: Optional[CellResult] = Field(description="Pre-computed result for the cell", default=None)
    status: Optional[CellStatus] = Field(description="Initial status for the cell", default=None)
    connection_id: Optional[str] = Field(description="ID of the data source connection to use (e.g., for GitHub)", default=None)


class UpdateCellParams(BaseModel):
    """Parameters for updating a cell"""
    notebook_id: str = Field(description="The UUID of the notebook")
    cell_id: str = Field(description="The UUID of the cell to update")
    content: Optional[str] = Field(description="New content for the cell", default=None)
    metadata: Optional[Dict[str, Any]] = Field(description="Updated metadata for the cell", default=None)


class ExecuteCellParams(BaseModel):
    """Parameters for executing a cell"""
    notebook_id: str = Field(description="The UUID of the notebook")
    cell_id: str = Field(description="The UUID of the cell to execute")


class ListCellsParams(BaseModel):
    """Parameters for listing cells"""
    notebook_id: str = Field(description="The UUID of the notebook")


class GetCellParams(BaseModel):
    """Parameters for getting a cell"""
    notebook_id: str = Field(description="The UUID of the notebook")
    cell_id: str = Field(description="The UUID of the cell to get")


class ExecuteAIQueryParams(BaseModel):
    """Parameters for executing an AI query"""
    notebook_id: str = Field(description="The UUID of the notebook")
    query: str = Field(description="The query to execute")
    position: Optional[int] = Field(description="Position to insert the cell (optional)", default=None)


class NotebookCellTools:
    """
    Tools for the AI agent to manage notebook cells.
    """
    def __init__(self, notebook_manager: NotebookManager):
        self.notebook_manager = notebook_manager
        
        # Create tool for creating cells
        create_cell_tool = Tool(name="create_cell", function=self.create_cell)
        create_cell_tool.description = "Create a new cell in a notebook"
        
        # Create tool for updating cells
        update_cell_tool = Tool(name="update_cell", function=self.update_cell)
        update_cell_tool.description = "Update an existing cell in a notebook"
        
        # Create tool for executing cells
        execute_cell_tool = Tool(name="execute_cell", function=self.execute_cell)
        execute_cell_tool.description = "Queue a cell for execution"
        
        # Create tool for listing cells
        list_cells_tool = Tool(name="list_cells", function=self.list_cells)
        list_cells_tool.description = "List all cells in a notebook"
        
        # Create tool for getting cell details
        get_cell_tool = Tool(name="get_cell", function=self.get_cell)
        get_cell_tool.description = "Get a specific cell by ID"
        
        # Store all tools in a list
        self.tools = [
            create_cell_tool,
            # Temporarily disable registration of other tools
            # update_cell_tool, 
            # execute_cell_tool, 
            # list_cells_tool,
            # get_cell_tool,
        ]
    
    def get_tools(self) -> List[Tool]:
        """Get all available tools"""
        return self.tools
    
    async def create_cell(self, params: CreateCellParams, **kwargs) -> Dict:
        """Create a new cell in a notebook"""
        db_gen = get_db()
        db: Session = next(db_gen)
        try:
            notebook_id = UUID(params.notebook_id)
            tools_logger.info(f"Creating cell in notebook: {notebook_id}")
            
            # Map cell type string to enum
            cell_type = getattr(CellType, params.cell_type.upper())
            
            # Merge provided metadata with tool info from kwargs
            cell_metadata = params.metadata or {}
            if 'tool_name' in kwargs:
                cell_metadata['tool_name'] = kwargs['tool_name']
            if 'tool_arguments' in kwargs:
                cell_metadata['tool_arguments'] = kwargs['tool_arguments']

            # Create the cell using the NotebookManager
            cell = self.notebook_manager.create_cell(
                db=db,
                notebook_id=notebook_id,
                cell_type=cell_type,
                content=params.content,
                position=params.position,
                metadata=cell_metadata,
                dependencies=params.dependencies,
                tool_call_id=params.tool_call_id,
                tool_name=params.tool_name,
                tool_arguments=params.tool_arguments,
                result=params.result,
                status=params.status,
                connection_id=params.connection_id
            )
            
            tools_logger.info(f"Created cell {cell.id} in notebook {notebook_id}")
            
            return {
                "success": True,
                "notebook_id": str(notebook_id),
                "cell_id": str(cell.id),
                "cell_type": params.cell_type,
                "position": params.position
            }
        except Exception as e:
            tools_logger.error(f"Error creating cell: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            try:
                next(db_gen) # Ensure session is closed
            except StopIteration:
                pass
            except Exception as close_exc:
                 tools_logger.error(f"Error closing DB session in create_cell: {close_exc}")
    
    # Keep method definitions but comment out registration above
   
    async def update_cell(self, params: UpdateCellParams) -> Dict:
        """Update an existing cell in a notebook"""
        # Temporarily disabled
        return {"success": False, "error": "Update cell tool temporarily disabled"}
        # db_gen = get_db()
        # db: Session = next(db_gen)
        # ... (original implementation commented out) ...
   
    async def execute_cell(self, params: ExecuteCellParams) -> Dict:
        """Queue a cell for execution"""
        # Temporarily disabled
        return {"success": False, "error": "Execute cell tool temporarily disabled"}
        # db_gen = get_db()
        # db: Session = next(db_gen)
        # ... (original implementation commented out) ...
   
    async def list_cells(self, params: ListCellsParams) -> Dict:
        """List all cells in a notebook"""
        # Temporarily disabled
        return {"success": False, "error": "List cells tool temporarily disabled"}
        # db_gen = get_db()
        # db: Session = next(db_gen)
        # ... (original implementation commented out) ...
   
    async def get_cell(self, params: GetCellParams) -> Dict:
        """Get a specific cell by ID"""
        # Temporarily disabled
        return {"success": False, "error": "Get cell tool temporarily disabled"}
        # db_gen = get_db()
        # db: Session = next(db_gen)
        # ... (original implementation commented out) ...

# Temporarily removed update_cell, execute_cell, list_cells, get_cell methods 