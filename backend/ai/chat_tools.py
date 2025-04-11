"""
Chat Agent Tools

This module defines tools that allow the AI agent to manage notebook cells.
"""

from typing import Dict, List, Optional, Union, Any
from uuid import UUID

import logging
from pydantic import BaseModel, Field
from pydantic_ai.tools import Tool

from backend.core.cell import CellType, create_cell
from backend.core.notebook import Notebook
from backend.services.notebook_manager import NotebookManager

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
        
        # Create tool for executing AI queries
        execute_ai_query_tool = Tool(name="execute_ai_query", function=self.execute_ai_query)
        execute_ai_query_tool.description = "Execute an AI query to investigate an issue"
        
        # Store all tools in a list
        self.tools = [
            create_cell_tool,
            update_cell_tool,
            execute_cell_tool,
            list_cells_tool,
            get_cell_tool,
            execute_ai_query_tool
        ]
    
    def get_tools(self) -> List[Tool]:
        """Get all available tools"""
        return self.tools
    
    async def create_cell(self, params: CreateCellParams) -> Dict:
        """Create a new cell in a notebook"""
        try:
            notebook_id = UUID(params.notebook_id)
            tools_logger.info(f"Creating cell in notebook: {notebook_id}")
            
            # Get the notebook
            notebook = self.notebook_manager.get_notebook(notebook_id)
            
            # Map cell type string to enum
            cell_type = getattr(CellType, params.cell_type.upper())
            
            # Create the cell
            cell = notebook.create_cell(
                cell_type=cell_type,
                content=params.content,
                position=params.position,
                metadata=params.metadata or {}
            )
            
            # Save the notebook
            self.notebook_manager.save_notebook(notebook_id)
            
            tools_logger.info(f"Created cell {cell.id} in notebook {notebook_id}")
            
            return {
                "success": True,
                "notebook_id": str(notebook_id),
                "cell_id": str(cell.id),
                "cell_type": params.cell_type,
                "position": params.position
            }
        except Exception as e:
            tools_logger.error(f"Error creating cell: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_cell(self, params: UpdateCellParams) -> Dict:
        """Update an existing cell in a notebook"""
        try:
            notebook_id = UUID(params.notebook_id)
            cell_id = UUID(params.cell_id)
            tools_logger.info(f"Updating cell {cell_id} in notebook {notebook_id}")
            
            # Get the notebook and cell
            notebook = self.notebook_manager.get_notebook(notebook_id)
            cell = notebook.get_cell(cell_id)
            
            # Update content if provided
            if params.content is not None:
                cell.update_content(params.content)
            
            # Update metadata if provided
            if params.metadata is not None:
                for key, value in params.metadata.items():
                    cell.metadata[key] = value
            
            # Save the notebook
            self.notebook_manager.save_notebook(notebook_id)
            
            tools_logger.info(f"Updated cell {cell_id} in notebook {notebook_id}")
            
            return {
                "success": True,
                "notebook_id": str(notebook_id),
                "cell_id": str(cell_id)
            }
        except Exception as e:
            tools_logger.error(f"Error updating cell: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_cell(self, params: ExecuteCellParams) -> Dict:
        """Queue a cell for execution"""
        try:
            notebook_id = UUID(params.notebook_id)
            cell_id = UUID(params.cell_id)
            tools_logger.info(f"Queueing cell {cell_id} for execution in notebook {notebook_id}")
            
            # Get the notebook and cell
            notebook = self.notebook_manager.get_notebook(notebook_id)
            cell = notebook.get_cell(cell_id)
            
            # Set cell status to queued
            from backend.core.cell import CellStatus
            cell.status = CellStatus.QUEUED
            
            # Save the notebook
            self.notebook_manager.save_notebook(notebook_id)
            
            tools_logger.info(f"Queued cell {cell_id} for execution in notebook {notebook_id}")
            
            return {
                "success": True,
                "notebook_id": str(notebook_id),
                "cell_id": str(cell_id),
                "status": "queued"
            }
        except Exception as e:
            tools_logger.error(f"Error executing cell: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_cells(self, params: ListCellsParams) -> Dict:
        """List all cells in a notebook"""
        try:
            notebook_id = UUID(params.notebook_id)
            tools_logger.info(f"Listing cells in notebook: {notebook_id}")
            
            # Get the notebook
            notebook = self.notebook_manager.get_notebook(notebook_id)
            
            # Get all cells
            cells = [
                {
                    "id": str(cell_id),
                    "type": notebook.cells[cell_id].type.value,
                    "content": notebook.cells[cell_id].content,
                    "status": notebook.cells[cell_id].status.value
                }
                for cell_id in notebook.cell_order
            ]
            
            return {
                "success": True,
                "notebook_id": str(notebook_id),
                "cells": cells
            }
        except Exception as e:
            tools_logger.error(f"Error listing cells: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_cell(self, params: GetCellParams) -> Dict:
        """Get a specific cell by ID"""
        try:
            notebook_id = UUID(params.notebook_id)
            cell_id = UUID(params.cell_id)
            tools_logger.info(f"Getting cell {cell_id} from notebook {notebook_id}")
            
            # Get the notebook and cell
            notebook = self.notebook_manager.get_notebook(notebook_id)
            cell = notebook.get_cell(cell_id)
            
            return {
                "success": True,
                "notebook_id": str(notebook_id),
                "cell_id": str(cell_id),
                "type": cell.type.value,
                "content": cell.content,
                "status": cell.status.value,
                "result": cell.result.model_dump() if cell.result else None,
                "metadata": cell.metadata
            }
        except Exception as e:
            tools_logger.error(f"Error getting cell: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_ai_query(self, params: ExecuteAIQueryParams) -> Dict:
        """Execute an AI query to investigate an issue"""
        try:
            notebook_id = UUID(params.notebook_id)
            tools_logger.info(f"Executing AI query in notebook {notebook_id}: {params.query}")
            
            # Get the notebook
            notebook = self.notebook_manager.get_notebook(notebook_id)
            
            # Create an AI query cell
            cell = notebook.create_cell(
                cell_type=CellType.AI_QUERY,
                content=params.query,
                position=params.position
            )
            
            # Set cell status to queued
            from backend.core.cell import CellStatus
            cell.status = CellStatus.QUEUED
            
            # Save the notebook
            self.notebook_manager.save_notebook(notebook_id)
            
            tools_logger.info(f"Created AI query cell {cell.id} in notebook {notebook_id}")
            
            return {
                "success": True,
                "notebook_id": str(notebook_id),
                "cell_id": str(cell.id),
                "query": params.query
            }
        except Exception as e:
            tools_logger.error(f"Error executing AI query: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 