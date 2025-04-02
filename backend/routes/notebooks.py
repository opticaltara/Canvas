"""
Notebook API Endpoints
"""

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.core.cell import CellType
from backend.core.notebook import Notebook
from backend.services.notebook_manager import NotebookManager, get_notebook_manager

router = APIRouter()


class NotebookCreate(BaseModel):
    """Parameters for creating a notebook"""
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict] = None


class NotebookUpdate(BaseModel):
    """Parameters for updating a notebook"""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict] = None


class CellCreate(BaseModel):
    """Parameters for creating a cell"""
    cell_type: CellType
    content: str = ""
    position: Optional[int] = None


@router.get("/")
async def list_notebooks(
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> List[Dict]:
    """
    Get a list of all notebooks
    
    Returns:
        List of notebook metadata
    """
    notebooks = notebook_manager.list_notebooks()
    return [
        {
            "id": str(notebook.id),
            "name": notebook.name,
            "description": notebook.description,
            "cell_count": len(notebook.cells),
            "created_at": notebook.created_at,
            "updated_at": notebook.updated_at
        }
        for notebook in notebooks
    ]


@router.post("/", status_code=201)
async def create_notebook(
    notebook_data: NotebookCreate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Create a new notebook
    
    Args:
        notebook_data: Parameters for creating the notebook
        
    Returns:
        The created notebook data
    """
    notebook = notebook_manager.create_notebook(
        name=notebook_data.name,
        description=notebook_data.description,
        metadata=notebook_data.metadata
    )
    
    return notebook.serialize()


@router.get("/{notebook_id}")
async def get_notebook(
    notebook_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Get a notebook by ID
    
    Args:
        notebook_id: The ID of the notebook
        
    Returns:
        The notebook data
    """
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        return notebook.serialize()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")


@router.put("/{notebook_id}")
async def update_notebook(
    notebook_id: UUID,
    notebook_data: NotebookUpdate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Update a notebook
    
    Args:
        notebook_id: The ID of the notebook
        notebook_data: Parameters for updating the notebook
        
    Returns:
        The updated notebook data
    """
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        
        # Update fields if provided
        if notebook_data.name is not None:
            notebook.name = notebook_data.name
        
        if notebook_data.description is not None:
            notebook.description = notebook_data.description
        
        if notebook_data.metadata is not None:
            notebook.metadata.update(notebook_data.metadata)
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        
        return notebook.serialize()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")


@router.delete("/{notebook_id}", status_code=204)
async def delete_notebook(
    notebook_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> None:
    """
    Delete a notebook
    
    Args:
        notebook_id: The ID of the notebook
    """
    try:
        notebook_manager.delete_notebook(notebook_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")


@router.post("/{notebook_id}/cells")
async def create_cell(
    notebook_id: UUID,
    cell_data: CellCreate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Create a new cell in a notebook
    
    Args:
        notebook_id: The ID of the notebook
        cell_data: Parameters for creating the cell
        
    Returns:
        The created cell data
    """
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        cell = notebook.create_cell(
            cell_type=cell_data.cell_type,
            content=cell_data.content,
            position=cell_data.position
        )
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        
        return cell.dict()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")