"""
Notebook API Endpoints
"""

import logging
import time
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.core.cell import CellType
from backend.core.notebook import Notebook
from backend.services.notebook_manager import NotebookManager, get_notebook_manager
from backend.core.cell import CellStatus

# Initialize logger
route_logger = logging.getLogger("routes.notebooks")

# Add correlation ID filter to the route logger
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

route_logger.addFilter(CorrelationIdFilter())

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
    route_logger.info("Listing all notebooks")
    notebooks = notebook_manager.list_notebooks()
    route_logger.info(f"Found {len(notebooks)} notebooks")
    return [
        {
            "id": str(notebook.id),
            "name": notebook.metadata.title,
            "description": notebook.metadata.description,
            "cell_count": len(notebook.cells),
            "created_at": notebook.metadata.created_at,
            "updated_at": notebook.metadata.updated_at
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
    route_logger.info(f"Creating new notebook with name: {notebook_data.name}")
    notebook = notebook_manager.create_notebook(
        name=notebook_data.name,
        description=notebook_data.description,
        metadata=notebook_data.metadata
    )
    route_logger.info(f"Created notebook with ID: {notebook.id}")
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
    route_logger.info(f"Fetching notebook with ID: {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        route_logger.info(f"Successfully retrieved notebook: {notebook_id}")
        return notebook.serialize()
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
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
    route_logger.info(f"Updating notebook: {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        
        # Update fields if provided
        if notebook_data.name is not None:
            route_logger.info(f"Updating notebook name to: {notebook_data.name}")
            notebook.metadata.title = notebook_data.name
        
        if notebook_data.description is not None:
            route_logger.info("Updating notebook description")
            notebook.metadata.description = notebook_data.description
        
        if notebook_data.metadata is not None:
            route_logger.info("Updating notebook metadata")
            for key, value in notebook_data.metadata.items():
                if hasattr(notebook.metadata, key):
                    setattr(notebook.metadata, key, value)
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        route_logger.info(f"Successfully updated notebook: {notebook_id}")
        
        return notebook.serialize()
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
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
    route_logger.info(f"Deleting notebook: {notebook_id}")
    try:
        notebook_manager.delete_notebook(notebook_id)
        route_logger.info(f"Successfully deleted notebook: {notebook_id}")
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
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
    route_logger.info(f"Creating new cell in notebook: {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        cell = notebook.create_cell(
            cell_type=cell_data.cell_type,
            content=cell_data.content,
            position=cell_data.position
        )
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        route_logger.info(f"Created cell {cell.id} in notebook {notebook_id}")
        
        return cell.model_dump()
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")


@router.get("/{notebook_id}/cells")
async def list_cells(
    notebook_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> List[Dict]:
    """
    List all cells in a notebook
    
    Args:
        notebook_id: The ID of the notebook
        
    Returns:
        List of cell data
    """
    route_logger.info(f"Listing cells for notebook: {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        cells = [notebook.cells[cell_id].model_dump() for cell_id in notebook.cell_order]
        route_logger.info(f"Found {len(cells)} cells in notebook {notebook_id}")
        return cells
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")


@router.get("/{notebook_id}/cells/{cell_id}")
async def get_cell(
    notebook_id: UUID,
    cell_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Get a specific cell from a notebook
    
    Args:
        notebook_id: The ID of the notebook
        cell_id: The ID of the cell
        
    Returns:
        The cell data
    """
    route_logger.info(f"Fetching cell {cell_id} from notebook {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        cell = notebook.get_cell(cell_id)
        route_logger.info(f"Successfully retrieved cell {cell_id}")
        return cell.model_dump()
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError:
        route_logger.error(f"Cell {cell_id} not found in notebook {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found in notebook {notebook_id}")


class CellUpdate(BaseModel):
    """Parameters for updating a cell"""
    content: Optional[str] = None
    metadata: Optional[Dict] = None
    settings: Optional[Dict] = None


@router.put("/{notebook_id}/cells/{cell_id}")
async def update_cell(
    notebook_id: UUID,
    cell_id: UUID,
    cell_data: CellUpdate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Update a cell in a notebook
    
    Args:
        notebook_id: The ID of the notebook
        cell_id: The ID of the cell
        cell_data: Parameters for updating the cell
        
    Returns:
        The updated cell data
    """
    route_logger.info(f"Updating cell {cell_id} in notebook {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        
        # Update content if provided
        if cell_data.content is not None:
            route_logger.info(f"Updating content for cell {cell_id}")
            notebook.update_cell_content(cell_id, cell_data.content)
        
        # Update metadata if provided
        if cell_data.metadata is not None:
            route_logger.info(f"Updating metadata for cell {cell_id}")
            notebook.update_cell_metadata(cell_id, cell_data.metadata)
            
        # Update settings if provided
        if cell_data.settings is not None:
            route_logger.info(f"Updating settings for cell {cell_id}")
            cell = notebook.get_cell(cell_id)
            cell.settings = cell_data.settings
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        route_logger.info(f"Successfully updated cell {cell_id}")
        
        return notebook.get_cell(cell_id).model_dump()
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError as e:
        route_logger.error(f"Error updating cell {cell_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{notebook_id}/cells/{cell_id}", status_code=204)
async def delete_cell(
    notebook_id: UUID,
    cell_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> None:
    """
    Delete a cell from a notebook
    
    Args:
        notebook_id: The ID of the notebook
        cell_id: The ID of the cell
    """
    route_logger.info(f"Deleting cell {cell_id} from notebook {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        notebook.remove_cell(cell_id)
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        route_logger.info(f"Successfully deleted cell {cell_id}")
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError as e:
        route_logger.error(f"Error deleting cell {cell_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


class DependencyCreate(BaseModel):
    """Parameters for creating a dependency between cells"""
    dependent_id: UUID
    dependency_id: UUID


@router.post("/{notebook_id}/dependencies")
async def add_dependency(
    notebook_id: UUID,
    dependency_data: DependencyCreate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Add a dependency relationship between cells
    
    Args:
        notebook_id: The ID of the notebook
        dependency_data: The dependency relationship to add
        
    Returns:
        Success message
    """
    route_logger.info(f"Adding dependency in notebook {notebook_id}: {dependency_data.dependent_id} -> {dependency_data.dependency_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        notebook.add_dependency(
            dependent_id=dependency_data.dependent_id,
            dependency_id=dependency_data.dependency_id
        )
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        route_logger.info("Successfully added dependency")
        
        return {
            "message": "Dependency added successfully",
            "dependent_id": str(dependency_data.dependent_id),
            "dependency_id": str(dependency_data.dependency_id)
        }
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError as e:
        route_logger.error(f"Error adding dependency: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{notebook_id}/dependencies")
async def remove_dependency(
    notebook_id: UUID,
    dependency_data: DependencyCreate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Remove a dependency relationship between cells
    
    Args:
        notebook_id: The ID of the notebook
        dependency_data: The dependency relationship to remove
        
    Returns:
        Success message
    """
    route_logger.info(f"Removing dependency in notebook {notebook_id}: {dependency_data.dependent_id} -> {dependency_data.dependency_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        notebook.remove_dependency(
            dependent_id=dependency_data.dependent_id,
            dependency_id=dependency_data.dependency_id
        )
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        route_logger.info("Successfully removed dependency")
        
        return {
            "message": "Dependency removed successfully",
            "dependent_id": str(dependency_data.dependent_id),
            "dependency_id": str(dependency_data.dependency_id)
        }
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError as e:
        route_logger.error(f"Error removing dependency: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{notebook_id}/cells/{cell_id}/execute")
async def execute_cell(
    notebook_id: UUID,
    cell_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> Dict:
    """
    Queue a cell for execution
    
    Args:
        notebook_id: The ID of the notebook
        cell_id: The ID of the cell to execute
        
    Returns:
        Status message
    """
    route_logger.info(f"Queueing cell {cell_id} for execution in notebook {notebook_id}")
    try:
        notebook = notebook_manager.get_notebook(notebook_id)
        cell = notebook.get_cell(cell_id)
        
        # Update cell status to queued
        cell.status = CellStatus.QUEUED
        route_logger.info(f"Updated cell {cell_id} status to QUEUED")
        
        # Save the notebook
        notebook_manager.save_notebook(notebook_id)
        route_logger.info(f"Successfully queued cell {cell_id} for execution")
        
        # Note: The actual execution would be handled by the execution service
        # which would pick up the queued cell from the execution queue
        
        return {
            "message": "Cell queued for execution",
            "cell_id": str(cell_id),
            "status": cell.status
        }
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError as e:
        route_logger.error(f"Error queueing cell {cell_id} for execution: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))