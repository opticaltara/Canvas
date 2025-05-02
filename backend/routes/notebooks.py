"""
Notebook API Endpoints
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.core.cell import CellType, CellStatus
from backend.core.types import ToolCallID
from backend.services.notebook_manager import NotebookManager, get_notebook_manager
from backend.db.database import get_db
from backend.websockets import WebSocketManager, get_ws_manager
from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.db.repositories import NotebookRepository

# Initialize logger
route_logger = logging.getLogger("api.routes.notebooks")

# Add correlation ID filter to the route logger
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

route_logger.addFilter(CorrelationIdFilter())

router = APIRouter()

# Dependency for WebSocket manager
# ws_dep: Callable[..., WebSocketManager] = Depends(get_ws_manager)


class NotebookCreate(BaseModel):
    """Parameters for creating a notebook"""
    title: str
    description: Optional[str] = None
    metadata: Optional[Dict] = None


class NotebookUpdate(BaseModel):
    """Parameters for updating a notebook"""
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict] = None


class CellCreate(BaseModel):
    """Parameters for creating a cell"""
    cell_type: CellType
    content: str = ""
    position: Optional[int] = None
    metadata: Optional[Dict] = None
    settings: Optional[Dict] = None
    connection_id: Optional[str] = None


class CellUpdate(BaseModel):
    """Parameters for updating a cell"""
    content: Optional[str] = None
    metadata: Optional[Dict] = None
    settings: Optional[Dict] = None


class CellReorder(BaseModel):
    """Parameters for reordering cells"""
    cell_order: List[str]  # List of cell IDs (as strings) in the new order


@router.get("/")
async def list_notebooks(
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
) -> List[Dict]:
    """
    Get a list of all notebooks
    
    Returns:
        List of notebook metadata
    """
    route_logger.info("Listing all notebooks")
    notebooks = notebook_manager.list_notebooks(db)
    route_logger.info(f"Found {len(notebooks)} notebooks")
    return [nb.model_dump() for nb in notebooks]


@router.post("/", status_code=201)
async def create_notebook(
    notebook_data: NotebookCreate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Create a new notebook
    
    Args:
        notebook_data: Parameters for creating the notebook
        
    Returns:
        The created notebook data
    """
    route_logger.info(f"Creating new notebook with title: {notebook_data.title}")
    try:
        notebook = notebook_manager.create_notebook(
            db=db,
            name=notebook_data.title,
            description=notebook_data.description,
            metadata=notebook_data.metadata
        )
        route_logger.info(f"Created notebook with ID: {notebook.id}")
        return notebook.model_dump()
    except Exception as e:
        route_logger.error(f"Error creating notebook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create notebook")


@router.get("/{notebook_id}")
async def get_notebook(
    notebook_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
        notebook = notebook_manager.get_notebook(db, notebook_id)
        route_logger.info(f"Successfully retrieved notebook: {notebook_id}")
        return notebook.model_dump()
    except KeyError:
        route_logger.error(f"Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except Exception as e:
        route_logger.error(f"Error fetching notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch notebook")


@router.put("/{notebook_id}")
async def update_notebook(
    notebook_id: UUID,
    notebook_data: NotebookUpdate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
        notebook = notebook_manager.get_notebook(db, notebook_id)
        
        # Update fields if provided
        if notebook_data.title is not None:
            route_logger.info(f"Updating notebook title to: {notebook_data.title}")
            notebook.metadata.title = notebook_data.title
        
        if notebook_data.description is not None:
            route_logger.info("Updating notebook description")
            notebook.metadata.description = notebook_data.description
        
        if notebook_data.metadata is not None:
            route_logger.info("Updating notebook metadata")
            for key, value in notebook_data.metadata.items():
                if hasattr(notebook.metadata, key):
                    setattr(notebook.metadata, key, value)
        
        # Save the notebook
        notebook_manager.save_notebook(db=db, notebook_id=notebook_id, notebook=notebook)
        route_logger.info(f"Successfully updated notebook: {notebook_id}")
        
        return notebook.model_dump()
    except KeyError:
        route_logger.error(f"Notebook not found for update: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except Exception as e:
        route_logger.error(f"Error updating notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update notebook")


@router.delete("/{notebook_id}", status_code=204)
async def delete_notebook(
    notebook_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
) -> None:
    """
    Delete a notebook
    
    Args:
        notebook_id: The ID of the notebook
    """
    route_logger.info(f"Deleting notebook: {notebook_id}")
    try:
        notebook_manager.delete_notebook(db, notebook_id)
        route_logger.info(f"Successfully deleted notebook: {notebook_id}")
    except KeyError:
        route_logger.error(f"Notebook not found for deletion: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except Exception as e:
        route_logger.error(f"Error deleting notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete notebook")


@router.post("/{notebook_id}/cells", response_model=Dict)
async def create_cell(
    notebook_id: UUID,
    cell_data: CellCreate,
    tool_call_id: ToolCallID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Create a new cell in a notebook
    
    Args:
        notebook_id: The ID of the notebook
        cell_data: Parameters for creating the cell
        
    Returns:
        The created cell data
    """
    route_logger.info(f"Creating new cell in notebook: {notebook_id} with type {cell_data.cell_type}")
    
    connection_id_to_use: Optional[str] = cell_data.connection_id
    
    if cell_data.cell_type == CellType.GITHUB:
        if connection_id_to_use is None:
            route_logger.info(f"GitHub cell type requires connection_id, fetching default.")
            try:
                default_connection = await connection_manager.get_default_connection('github')
                if default_connection:
                    connection_id_to_use = default_connection.id
                    route_logger.info(f"Using default GitHub connection: {connection_id_to_use}")
                else:
                    route_logger.warning("No default GitHub connection found. Cell will be created without one.")
            except Exception as e:
                route_logger.error(f"Error fetching default GitHub connection: {e}", exc_info=True)
                pass 
    
    try:
        cell = notebook_manager.create_cell(
            db=db,
            notebook_id=notebook_id,
            cell_type=cell_data.cell_type,
            content=cell_data.content,
            tool_call_id=tool_call_id,
            position=cell_data.position,
            connection_id=connection_id_to_use,
            metadata=cell_data.metadata,
            settings=cell_data.settings
        )
        
        # Fetch the notebook state after cell creation to pass to save_notebook
        notebook = notebook_manager.get_notebook(db, notebook_id)
        # Save the notebook
        notebook_manager.save_notebook(db=db, notebook_id=notebook_id, notebook=notebook)
        route_logger.info(f"Created cell {cell.id} in notebook {notebook_id}")
        
        return cell.model_dump()
    except KeyError:
        route_logger.error(f"Notebook not found for cell creation: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError as e:
        route_logger.error(f"Invalid data for cell creation in notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        route_logger.error(f"Error creating cell in notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create cell")


@router.get("/{notebook_id}/cells", response_model=List[Dict])
async def list_cells(
    notebook_id: UUID,
    # Remove notebook_manager dependency as we'll use the repository directly
    # notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
        # Instantiate repository directly
        repository = NotebookRepository(db)
        
        # Use the new repository method to get cells
        db_cells = repository.list_cells_by_notebook(str(notebook_id))
        
        # Convert db cell objects to dictionaries
        # Assuming Cell model has a to_dict() method
        cells_data = [cell.to_dict() for cell in db_cells]
        
        route_logger.info(f"Found {len(cells_data)} cells in notebook {notebook_id}")
        return cells_data
    # Update the exception handling if necessary (KeyError might not be raised here anymore)
    # except KeyError:
    #     route_logger.error(f"Notebook not found: {notebook_id}")
    #     raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except Exception as e:
        route_logger.error(f"Error listing cells for notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list cells")


@router.get("/{notebook_id}/cells/{cell_id}", response_model=Dict)
async def get_cell(
    notebook_id: UUID,
    cell_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
        cell = notebook_manager.get_cell(db, notebook_id, cell_id)
        route_logger.info(f"Cell fetched successfully: {cell_id}")
        return cell.model_dump()
    except KeyError:
        route_logger.error(f"Cell {cell_id} not found in notebook {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found in notebook {notebook_id}")
    except Exception as e:
        route_logger.error(f"Error fetching cell {cell_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch cell")


@router.put("/{notebook_id}/cells/{cell_id}")
async def update_cell(
    notebook_id: UUID,
    cell_id: UUID,
    cell_data: CellUpdate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
    updated_cell = None
    try:
        cell = notebook_manager.get_cell(db, notebook_id, cell_id)

        route_logger.info(f"Cell: {cell}")
        
        # Update content if provided
        if cell_data.content is not None:
            route_logger.info(f"Updating content for cell {cell_id}")
            updated_cell = notebook_manager.update_cell_content(db, notebook_id, cell_id, cell_data.content)
        
        # If no updates were performed, fetch the current cell state
        if updated_cell is None:
            updated_cell = notebook_manager.get_cell(db, notebook_id, cell_id)

        route_logger.info(f"Successfully updated cell {cell_id}")
        return updated_cell.model_dump()
    except KeyError:
        route_logger.error(f"Cell {cell_id} not found for update in notebook {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found in notebook {notebook_id}")
    except Exception as e:
        route_logger.error(f"Error updating cell {cell_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update cell")


@router.delete("/{notebook_id}/cells/{cell_id}", status_code=204)
async def delete_cell(
    notebook_id: UUID,
    cell_id: UUID,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
) -> None:
    """
    Delete a cell from a notebook
    
    Args:
        notebook_id: The ID of the notebook
        cell_id: The ID of the cell
    """
    route_logger.info(f"Deleting cell {cell_id} from notebook {notebook_id}")
    try:
        notebook_manager.delete_cell(db, notebook_id, cell_id)
        route_logger.info(f"Successfully deleted cell {cell_id}")
    except KeyError:
        route_logger.error(f"Cell {cell_id} not found for deletion in notebook {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found in notebook {notebook_id}")
    except Exception as e:
        route_logger.error(f"Error deleting cell {cell_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete cell")


class DependencyCreate(BaseModel):
    """Parameters for creating a dependency between cells"""
    dependent_id: UUID
    dependency_id: UUID


@router.post("/{notebook_id}/dependencies")
async def add_dependency(
    notebook_id: UUID,
    dependency_data: DependencyCreate,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
        notebook = notebook_manager.get_notebook(db, notebook_id)
        notebook.add_dependency(
            dependent_id=dependency_data.dependent_id,
            dependency_id=dependency_data.dependency_id
        )
        
        # Save the notebook
        notebook_manager.save_notebook(db=db, notebook_id=notebook_id, notebook=notebook)
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
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
        notebook = notebook_manager.get_notebook(db, notebook_id)
        notebook.remove_dependency(
            dependent_id=dependency_data.dependent_id,
            dependency_id=dependency_data.dependency_id
        )
        
        # Save the notebook
        notebook_manager.save_notebook(db=db, notebook_id=notebook_id, notebook=notebook)
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
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
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
        notebook = notebook_manager.get_notebook(db, notebook_id)
        cell = notebook.get_cell(cell_id)
        
        # Update cell status to queued
        cell.status = CellStatus.QUEUED
        route_logger.info(f"Updated cell {cell_id} status to QUEUED")
        
        # Save the notebook
        notebook_manager.save_notebook(db=db, notebook_id=notebook_id, notebook=notebook)
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


@router.post("/{notebook_id}/reorder", status_code=200, response_model=Dict)
async def reorder_cells(
    notebook_id: UUID,
    reorder_data: CellReorder,
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Reorder cells in a notebook
    
    Args:
        notebook_id: The ID of the notebook
        reorder_data: The new order of cells
        
    Returns:
        Success message
    """
    route_logger.info(f"Reordering cells in notebook: {notebook_id}")
    try:
        cell_order_uuids = [UUID(cell_id) for cell_id in reorder_data.cell_order]
        notebook_manager.reorder_cells(db, notebook_id, cell_order_uuids)
        route_logger.info(f"Cells reordered successfully for notebook: {notebook_id}")
        return {"message": "Cells reordered successfully"}
    except KeyError:
        route_logger.error(f"Notebook {notebook_id} not found for reordering")
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    except ValueError as e:
        route_logger.error(f"Error reordering cells for notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        route_logger.error(f"Error reordering cells for notebook {notebook_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reorder cells")


@router.websocket("/ws/notebook/{notebook_id}")
async def notebook_websocket(
    websocket: WebSocket, 
    notebook_id: str,
    ws_manager: WebSocketManager = Depends(get_ws_manager)
):
    notebook_uuid = UUID(notebook_id)
    await ws_manager.connect(websocket, notebook_uuid)
    route_logger.info(f"WebSocket connected for notebook: {notebook_id}")
    try:
        while True:
            await websocket.receive_text() 
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, notebook_uuid)
        route_logger.info(f"WebSocket disconnected for notebook: {notebook_id}")
    except Exception as e:
         route_logger.error(f"WebSocket error for notebook {notebook_id}: {str(e)}", exc_info=True)
         ws_manager.disconnect(websocket, notebook_uuid)