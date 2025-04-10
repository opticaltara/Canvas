"""
Notebook Manager Service
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.core.notebook import Notebook, NotebookMetadata
from backend.core.cell import Cell, CellType, CellStatus, CellResult
from backend.db.repositories import NotebookRepository
from backend.db.database import get_db

# Configure logging
logger = logging.getLogger(__name__)

# Make sure correlation_id is set for all logs
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

# Add the filter to our logger
logger.addFilter(CorrelationIdFilter())

# Singleton instance
_notebook_manager_instance = None


def get_notebook_manager():
    """Get the singleton NotebookManager instance"""
    global _notebook_manager_instance
    if _notebook_manager_instance is None:
        _notebook_manager_instance = NotebookManager()
    return _notebook_manager_instance


class NotebookManager:
    """
    Manages notebook instances and their persistence
    """
    def __init__(self):
        correlation_id = str(uuid4())
        self.settings = get_settings()
        self.notify_callback: Optional[Callable] = None
        
        # Get a session from the generator and initialize repository
        session_generator = get_db()
        self.db: Session = next(session_generator)
        self.repository = NotebookRepository(self.db)
        
        logger.info("NotebookManager initialized", extra={'correlation_id': correlation_id})
    
    def set_notify_callback(self, callback: Callable) -> None:
        """
        Set the callback function for notifying clients of changes
        
        Args:
            callback: The callback function
        """
        self.notify_callback = callback
    
    def create_notebook(self, name: str, description: Optional[str] = None, metadata: Optional[Dict] = None) -> Notebook:
        """
        Create a new notebook
        
        Args:
            name: The name of the notebook
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            The created notebook
        """
        correlation_id = str(uuid4())
        logger.info(f"Creating new notebook with name '{name}'", extra={'correlation_id': correlation_id})
        
        # Create notebook in database
        db_notebook = self.repository.create_notebook(
            title=name,
            description=description,
            created_by=metadata.get("created_by") if metadata else None,
            tags=metadata.get("tags", []) if metadata else []
        )
        
        # Get data as dictionary from to_dict
        notebook_data = db_notebook.to_dict()
        
        # Create notebook model with values from dictionary and default values for datetimes if needed
        notebook = Notebook(
            id=UUID(notebook_data["id"]),
            metadata=NotebookMetadata(
                title=notebook_data["metadata"]["title"],
                description=notebook_data["metadata"]["description"],
                created_by=notebook_data["metadata"]["created_by"],
                created_at=datetime.fromisoformat(notebook_data["metadata"]["created_at"]) if notebook_data["metadata"]["created_at"] else datetime.now(timezone.utc),
                updated_at=datetime.fromisoformat(notebook_data["metadata"]["updated_at"]) if notebook_data["metadata"]["updated_at"] else datetime.now(timezone.utc),
                tags=notebook_data["metadata"]["tags"]
            )
        )
        
        logger.info(f"Successfully created notebook {notebook.id}", extra={'correlation_id': correlation_id})
        
        return notebook
    
    def get_notebook(self, notebook_id: UUID) -> Notebook:
        """
        Get a notebook by ID
        
        Args:
            notebook_id: The ID of the notebook
            
        Returns:
            The notebook
            
        Raises:
            KeyError: If the notebook is not found
        """
        db_notebook = self.repository.get_notebook(str(notebook_id))
        if not db_notebook:
            logger.error(f"Notebook {notebook_id} not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        
        # Convert to model
        notebook = self._db_notebook_to_model(db_notebook)
        logger.debug(f"Retrieved notebook {notebook_id}")
        return notebook
    
    def list_notebooks(self) -> List[Notebook]:
        """
        Get a list of all notebooks
        
        Returns:
            List of notebooks
        """
        db_notebooks = self.repository.list_notebooks()
        return [self._db_notebook_to_model(db_notebook) for db_notebook in db_notebooks]
    
    def save_notebook(self, notebook_id: UUID) -> None:
        """
        Save a notebook to storage
        
        Args:
            notebook_id: The ID of the notebook
            
        Raises:
            KeyError: If the notebook is not found
        """
        # This method is now a no-op since changes are saved immediately
        # to the database in the repository methods
        pass
    
    def delete_notebook(self, notebook_id: UUID) -> None:
        """
        Delete a notebook
        
        Args:
            notebook_id: The ID of the notebook
            
        Raises:
            KeyError: If the notebook is not found
        """
        success = self.repository.delete_notebook(str(notebook_id))
        if not success:
            logger.error(f"Cannot delete notebook {notebook_id}: not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        
        logger.info(f"Successfully deleted notebook {notebook_id}")
    
    def create_cell(self, notebook_id: UUID, cell_type: CellType, content: str, 
                    position: Optional[int] = None, connection_id: Optional[str] = None,
                    metadata: Optional[Dict] = None, settings: Optional[Dict] = None) -> Cell:
        """
        Create a new cell in a notebook
        
        Args:
            notebook_id: ID of the notebook
            cell_type: Type of cell to create
            content: Content for the cell
            position: Optional position in the cell order
            connection_id: Optional connection ID
            metadata: Optional metadata
            settings: Optional settings
            
        Returns:
            The created cell
            
        Raises:
            KeyError: If the notebook is not found
        """
        # Get notebook to verify it exists
        notebook = self.get_notebook(notebook_id)
        
        # Determine position
        if position is None:
            position = len(notebook.cells)
        
        # Create cell in database
        db_cell = self.repository.create_cell(
            notebook_id=str(notebook_id),
            cell_type=cell_type.value,
            content=content,
            position=position,
            connection_id=connection_id,
            metadata=metadata,
            settings=settings
        )
        
        if not db_cell:
            raise KeyError(f"Failed to create cell in notebook {notebook_id}")
        
        # Convert to model using to_dict
        cell_dict = db_cell.to_dict()
        
        # Create cell model
        cell = Cell(
            id=UUID(cell_dict["id"]),
            type=cell_type,
            content=content,
            status=CellStatus(cell_dict["status"]),
            created_at=datetime.fromisoformat(cell_dict["created_at"]) if cell_dict["created_at"] else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(cell_dict["updated_at"]) if cell_dict["updated_at"] else datetime.now(timezone.utc),
            connection_id=cell_dict["connection_id"],
            metadata=cell_dict["metadata"],
            settings=cell_dict["settings"]
        )
        
        # Add to notebook model
        notebook.cells[cell.id] = cell
        if position is not None and 0 <= position <= len(notebook.cell_order):
            notebook.cell_order.insert(position, cell.id)
        else:
            notebook.cell_order.append(cell.id)
        
        logger.info(f"Created cell {cell.id} in notebook {notebook_id}")
        return cell
    
    def get_cell(self, cell_id: UUID) -> Cell:
        """
        Get a cell by ID
        
        Args:
            cell_id: The ID of the cell
            
        Returns:
            The cell
            
        Raises:
            KeyError: If the cell is not found
        """
        db_cell = self.repository.get_cell(str(cell_id))
        if not db_cell:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        # Convert to model
        cell = self._db_cell_to_model(db_cell)
        logger.debug(f"Retrieved cell {cell_id}")
        return cell
    
    def update_cell_content(self, cell_id: UUID, content: str) -> Cell:
        """
        Update a cell's content
        
        Args:
            cell_id: ID of the cell to update
            content: New content for the cell
            
        Returns:
            The updated cell
            
        Raises:
            KeyError: If the cell is not found
        """
        # Update in database
        db_cell = self.repository.update_cell(
            str(cell_id),
            content=content
        )
        
        if not db_cell:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        # Mark as stale
        self.repository.mark_cell_stale(str(cell_id))
        
        # Convert to model
        cell = self._db_cell_to_model(db_cell)
        logger.info(f"Updated content for cell {cell_id}")
        return cell
    
    def delete_cell(self, cell_id: UUID) -> None:
        """
        Delete a cell
        
        Args:
            cell_id: ID of the cell to delete
            
        Raises:
            KeyError: If the cell is not found
        """
        success = self.repository.delete_cell(str(cell_id))
        if not success:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        logger.info(f"Deleted cell {cell_id}")
    
    def add_dependency(self, dependent_id: UUID, dependency_id: UUID) -> None:
        """
        Add a dependency between cells
        
        Args:
            dependent_id: ID of the cell that depends on another
            dependency_id: ID of the cell that is depended upon
            
        Raises:
            KeyError: If either cell is not found
        """
        success = self.repository.add_dependency(str(dependent_id), str(dependency_id))
        if not success:
            logger.error(f"Failed to add dependency between cells {dependent_id} and {dependency_id}")
            raise KeyError(f"Failed to add dependency between cells {dependent_id} and {dependency_id}")
        
        logger.info(f"Added dependency: {dependent_id} depends on {dependency_id}")
    
    def remove_dependency(self, dependent_id: UUID, dependency_id: UUID) -> None:
        """
        Remove a dependency between cells
        
        Args:
            dependent_id: ID of the cell that depends on another
            dependency_id: ID of the cell that is depended upon
            
        Raises:
            KeyError: If either cell is not found
        """
        success = self.repository.remove_dependency(str(dependent_id), str(dependency_id))
        if not success:
            logger.error(f"Failed to remove dependency between cells {dependent_id} and {dependency_id}")
            raise KeyError(f"Failed to remove dependency between cells {dependent_id} and {dependency_id}")
        
        logger.info(f"Removed dependency: {dependent_id} no longer depends on {dependency_id}")
    
    def set_cell_result(self, cell_id: UUID, result: Any, error: Optional[str] = None, 
                        execution_time: float = 0.0) -> None:
        """
        Set the execution result for a cell
        
        Args:
            cell_id: ID of the cell
            result: The result data
            error: Optional error message
            execution_time: Time taken to execute the cell
            
        Raises:
            KeyError: If the cell is not found
        """
        db_cell = self.repository.set_cell_result(
            str(cell_id),
            content=result,
            error=error,
            execution_time=execution_time
        )
        
        if not db_cell:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        logger.info(f"Set result for cell {cell_id}")
    
    def mark_dependents_stale(self, cell_id: UUID) -> Set[UUID]:
        """
        Mark all cells that depend on this cell as stale
        
        Args:
            cell_id: ID of the cell whose dependents should be marked stale
            
        Returns:
            Set of cell IDs that were marked stale
            
        Raises:
            KeyError: If the cell is not found
        """
        # Get dependents from database
        db_dependents = self.repository.get_dependents(str(cell_id))
        if not db_dependents:
            return set()
        
        # Mark each dependent as stale
        stale_cells = set()
        for db_dependent in db_dependents:
            # Convert to dict first
            dependent_dict = db_dependent.to_dict()
            self.repository.mark_cell_stale(dependent_dict["id"])
            stale_cells.add(UUID(dependent_dict["id"]))
        
        logger.info(f"Marked {len(stale_cells)} dependents of cell {cell_id} as stale")
        return stale_cells
    
    def reorder_cells(self, notebook_id: UUID, cell_order: List[UUID]) -> None:
        """
        Reorder cells in a notebook
        
        Args:
            notebook_id: ID of the notebook
            cell_order: New order of cell IDs
            
        Raises:
            KeyError: If the notebook is not found
        """
        success = self.repository.reorder_cells(
            str(notebook_id),
            [str(cell_id) for cell_id in cell_order]
        )
        
        if not success:
            logger.error(f"Notebook {notebook_id} not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        
        logger.info(f"Reordered cells in notebook {notebook_id}")
    
    def _db_notebook_to_model(self, db_notebook) -> Notebook:
        """Convert database notebook to model"""
        # Get cells
        cells = {}
        cell_order = []
        
        for db_cell in db_notebook.cells:
            cell = self._db_cell_to_model(db_cell)
            cells[cell.id] = cell
            cell_order.append(cell.id)
        
        # Sort cell order by position
        cell_order.sort(key=lambda cell_id: cells[cell_id].metadata.get("position", 0))
        
        # Get data as dict
        notebook_dict = db_notebook.to_dict()
        
        # Create notebook model
        notebook = Notebook(
            id=UUID(notebook_dict["id"]),
            metadata=NotebookMetadata(
                title=notebook_dict["metadata"]["title"],
                description=notebook_dict["metadata"]["description"],
                created_by=notebook_dict["metadata"]["created_by"],
                created_at=datetime.fromisoformat(notebook_dict["metadata"]["created_at"]) if notebook_dict["metadata"]["created_at"] else datetime.now(timezone.utc),
                updated_at=datetime.fromisoformat(notebook_dict["metadata"]["updated_at"]) if notebook_dict["metadata"]["updated_at"] else datetime.now(timezone.utc),
                tags=notebook_dict["metadata"]["tags"]
            ),
            cells=cells,
            cell_order=cell_order
        )
        
        return notebook
    
    def _db_cell_to_model(self, db_cell) -> Cell:
        """Convert database cell to model"""
        # Get data as dict
        cell_dict = db_cell.to_dict()
        
        # Create result if exists
        result = None
        if cell_dict["result"] is not None:
            result = CellResult(
                content=cell_dict["result"]["content"],
                error=cell_dict["result"]["error"],
                execution_time=cell_dict["result"]["execution_time"],
                timestamp=datetime.fromisoformat(cell_dict["result"]["timestamp"]) if cell_dict["result"]["timestamp"] else datetime.now(timezone.utc)
            )
        
        # Create cell model
        cell = Cell(
            id=UUID(cell_dict["id"]),
            type=CellType(cell_dict["type"]),
            content=cell_dict["content"],
            result=result,
            status=CellStatus(cell_dict["status"]),
            created_at=datetime.fromisoformat(cell_dict["created_at"]) if cell_dict["created_at"] else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(cell_dict["updated_at"]) if cell_dict["updated_at"] else datetime.now(timezone.utc),
            connection_id=cell_dict["connection_id"],
            metadata=cell_dict["metadata"],
            settings=cell_dict["settings"]
        )
        
        # Add dependencies and dependents
        for dep_id in cell_dict["dependencies"]:
            cell.dependencies.add(UUID(dep_id))
        
        for dep_id in cell_dict["dependents"]:
            cell.dependents.add(UUID(dep_id))
        
        return cell