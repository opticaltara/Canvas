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

# Type hint for session dependency
DbSession = Session

def get_notebook_manager():
    """Factory function to get a NotebookManager instance."""
    # global _notebook_manager_instance
    # if _notebook_manager_instance is None:
    #     _notebook_manager_instance = NotebookManager()
    # return _notebook_manager_instance
    # Return a new instance each time
    return NotebookManager()


class NotebookManager:
    """
    Service layer for managing notebooks.
    Handles interactions between API/agents and the database repository.
    Database session is injected per method call.
    """
    def __init__(self):
        correlation_id = str(uuid4())
        self.settings = get_settings()
        self.notify_callback: Optional[Callable] = None
        
        # REMOVED: Session and Repository initialization moved to methods
        # session_generator = get_db()
        # self.db: Session = next(session_generator)
        # self.repository = NotebookRepository(self.db)
        
        logger.info("NotebookManager instance created", extra={'correlation_id': correlation_id})
    
    def set_notify_callback(self, callback: Callable) -> None:
        """
        Set the callback function for notifying clients of changes
        
        Args:
            callback: The callback function
        """
        self.notify_callback = callback
    
    def create_notebook(self, db: DbSession, name: str, description: Optional[str] = None, metadata: Optional[Dict] = None) -> Notebook:
        """Create a new notebook"""
        correlation_id = str(uuid4())
        logger.info(f"Creating new notebook with name '{name}'", extra={'correlation_id': correlation_id})
        
        # Instantiate repository locally
        repository = NotebookRepository(db)

        # Create notebook in database
        db_notebook = repository.create_notebook(
            title=name,
            description=description,
            created_by=metadata.get("created_by") if metadata else None,
            tags=metadata.get("tags", []) if metadata else []
        )
        
        # Get data as dictionary from to_dict
        notebook_data = db_notebook.to_dict()
        
        # Create notebook model 
        notebook = self._db_notebook_to_model(db_notebook) # Use helper

        logger.info(f"Successfully created notebook {notebook.id}", extra={'correlation_id': correlation_id})
        return notebook
    
    def get_notebook(self, db: DbSession, notebook_id: UUID) -> Notebook:
        """Get a notebook by ID"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        db_notebook = repository.get_notebook(str(notebook_id))
        if not db_notebook:
            logger.error(f"Notebook {notebook_id} not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        
        # Convert to model
        notebook = self._db_notebook_to_model(db_notebook)
        logger.info(f"Retrieved notebook {notebook_id}")
        return notebook
    
    def list_notebooks(self, db: DbSession) -> List[Notebook]:
        """Get a list of all notebooks"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        db_notebooks = repository.list_notebooks()
        return [self._db_notebook_to_model(db_notebook) for db_notebook in db_notebooks]
    
    def save_notebook(self, db: DbSession, notebook_id: UUID, notebook: Notebook) -> None:
        """
        Save the complete state of a Notebook object to the database.
        This includes tool call records, cell links, cell order, and timestamps.

        Args:
            db: The database session.
            notebook_id: The ID of the notebook to save.
            notebook: The Notebook object containing the state to save.
            
        Raises:
            KeyError: If the notebook is not found (potentially by repository methods).
            Exception: If repository methods fail.
        """
        correlation_id = str(uuid4())
        logger.info(f"Saving full state for notebook {notebook_id}", extra={'correlation_id': correlation_id})
        
        try:
            # Instantiate repository locally
            repository = NotebookRepository(db)

            # 1. Save Tool Call Records
            if notebook.tool_call_records:
                # Assuming repository has a method like bulk_upsert_tool_call_records
                # This method needs to be implemented in NotebookRepository
                logger.debug(f"[{correlation_id}] Saving {len(notebook.tool_call_records)} tool call records...")
                # repository.bulk_upsert_tool_call_records(list(notebook.tool_call_records.values())) 
                # Placeholder log until repository method is implemented:
                logger.info(f"[{correlation_id}] Placeholder: Would save {len(notebook.tool_call_records)} tool call records.")
            else:
                logger.debug(f"[{correlation_id}] No tool call records to save.")

            # 2. Save Cell Links (tool_call_ids)
            cell_updates: Dict[str, List[UUID]] = {}
            for cell in notebook.cells.values():
                if cell.tool_call_ids: # Only update if there are links
                     cell_updates[str(cell.id)] = cell.tool_call_ids
            
            if cell_updates:
                # Assuming repository has a method like bulk_update_cell_tool_call_ids
                # This method needs to be implemented in NotebookRepository
                logger.debug(f"[{correlation_id}] Saving tool call links for {len(cell_updates)} cells...")
                # repository.bulk_update_cell_tool_call_ids(cell_updates)
                # Placeholder log until repository method is implemented:
                logger.info(f"[{correlation_id}] Placeholder: Would update tool links for cells: {list(cell_updates.keys())}")
            else:
                 logger.debug(f"[{correlation_id}] No cell tool links to update.")

            # 3. Save Cell Order
            if notebook.cell_order:
                # Assuming repository has a method like update_cell_order
                # This method needs to be implemented in NotebookRepository
                str_cell_order = [str(cell_id) for cell_id in notebook.cell_order]
                logger.debug(f"[{correlation_id}] Saving cell order: {str_cell_order}")
                # repository.update_cell_order(str(notebook_id), str_cell_order)
                # Placeholder log until repository method is implemented:
                logger.info(f"[{correlation_id}] Placeholder: Would save cell order.")
            else:
                 logger.debug(f"[{correlation_id}] No cell order provided to save.")
                 
            # 4. Update Notebook Timestamp
            # Assuming repository has update_notebook_timestamp or similar
            logger.debug(f"[{correlation_id}] Updating notebook timestamp...")
            # repository.update_notebook_timestamp(str(notebook_id))
            # Placeholder log until repository method is implemented:
            logger.info(f"[{correlation_id}] Placeholder: Would update notebook timestamp.")

            logger.info(f"Successfully completed save operations for notebook {notebook_id}", extra={'correlation_id': correlation_id})

        except Exception as e:
            logger.error(f"Error saving full state for notebook {notebook_id}: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
            # Re-raise the exception to be handled by the caller
            raise
    
    def delete_notebook(self, db: DbSession, notebook_id: UUID) -> None:
        """Delete a notebook"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        success = repository.delete_notebook(str(notebook_id))
        if not success:
            logger.error(f"Cannot delete notebook {notebook_id}: not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        logger.info(f"Successfully deleted notebook {notebook_id}")
    
    def create_cell(self, db: DbSession, notebook_id: UUID, cell_type: CellType, content: str, 
                    position: Optional[int] = None, connection_id: Optional[str] = None,
                    metadata: Optional[Dict] = None, settings: Optional[Dict] = None) -> Cell:
        """Create a new cell in a notebook"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        
        # Get notebook to determine position if needed (uses its own repository instance)
        # Note: This might fetch the notebook twice if called right after get_notebook.
        # Could optimize later if performance becomes an issue.
        notebook_for_pos = self.get_notebook(db, notebook_id)
        
        if position is None:
            position = len(notebook_for_pos.cells)
        
        # Create cell in database
        db_cell = repository.create_cell(
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
        
        # Convert to model
        cell = self._db_cell_to_model(db_cell)
        
        logger.info(f"Created cell {cell.id} in notebook {notebook_id}")
        return cell
    
    def get_cell(self, db: DbSession, notebook_id: UUID, cell_id: UUID) -> Cell:
        """Get a cell by ID"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        db_cell = repository.get_cell(str(cell_id))
        if not db_cell:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        # Optional: Verify cell belongs to notebook_id
        if str(db_cell.notebook_id) != str(notebook_id):
             logger.error(f"Cell {cell_id} does not belong to notebook {notebook_id}")
             raise KeyError(f"Cell {cell_id} not found in notebook {notebook_id}")

        # Convert to model
        cell = self._db_cell_to_model(db_cell)
        logger.info(f"Retrieved cell {cell_id}")
        return cell
    
    def update_cell_content(self, db: DbSession, notebook_id: UUID, cell_id: UUID, content: str) -> Cell:
        """Update a cell's content"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        
        # Optional: Verify cell belongs to notebook before updating
        # db_cell_check = repository.get_cell(str(cell_id))
        # if not db_cell_check or str(db_cell_check.notebook_id) != str(notebook_id):
        #     logger.error(f"Cell {cell_id} not found in notebook {notebook_id} for update.")
        #     raise KeyError(f"Cell {cell_id} not found in notebook {notebook_id}")

        # Update in database
        db_cell = repository.update_cell(
            str(cell_id),
            content=content
        )
        
        if not db_cell:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        # Mark as stale (also uses the repository)
        repository.mark_cell_stale(str(cell_id))
        
        # Fetch the updated cell again to reflect changes
        updated_db_cell = repository.get_cell(str(cell_id))
        if not updated_db_cell:
             # Should not happen if update succeeded, but handle defensively
             logger.error(f"Could not re-fetch cell {cell_id} after update.")
             raise KeyError(f"Could not re-fetch cell {cell_id} after update.")

        cell = self._db_cell_to_model(updated_db_cell)
        logger.info(f"Updated content for cell {cell_id}")
        return cell
    
    def delete_cell(self, db: DbSession, notebook_id: UUID, cell_id: UUID) -> None:
        """Delete a cell"""
        # Instantiate repository locally
        repository = NotebookRepository(db)

        # Optional: Verify cell belongs to notebook before deleting
        # db_cell_check = repository.get_cell(str(cell_id))
        # if not db_cell_check or str(db_cell_check.notebook_id) != str(notebook_id):
        #     logger.error(f"Cell {cell_id} not found in notebook {notebook_id} for deletion.")
        #     raise KeyError(f"Cell {cell_id} not found in notebook {notebook_id}")

        success = repository.delete_cell(str(cell_id))
        if not success:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        logger.info(f"Deleted cell {cell_id}")
    
    def add_dependency(self, db: DbSession, notebook_id: UUID, dependent_id: UUID, dependency_id: UUID) -> None:
        """Add a dependency between cells"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        # Optional: Validate cells belong to the notebook?
        success = repository.add_dependency(str(dependent_id), str(dependency_id))
        if not success:
            logger.error(f"Failed to add dependency between cells {dependent_id} and {dependency_id}")
            raise KeyError(f"Failed to add dependency between cells {dependent_id} and {dependency_id}")
        
        logger.info(f"Added dependency: {dependent_id} depends on {dependency_id}")
    
    def remove_dependency(self, db: DbSession, notebook_id: UUID, dependent_id: UUID, dependency_id: UUID) -> None:
        """Remove a dependency between cells"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        # Optional: Validate cells belong to the notebook?
        success = repository.remove_dependency(str(dependent_id), str(dependency_id))
        if not success:
            logger.error(f"Failed to remove dependency between cells {dependent_id} and {dependency_id}")
            raise KeyError(f"Failed to remove dependency between cells {dependent_id} and {dependency_id}")
        
        logger.info(f"Removed dependency: {dependent_id} no longer depends on {dependency_id}")
    
    def set_cell_result(self, db: DbSession, notebook_id: UUID, cell_id: UUID, result: Any, error: Optional[str] = None, 
                        execution_time: float = 0.0) -> None:
        """Set the execution result for a cell"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        # Optional: Validate cell belongs to the notebook?
        db_cell = repository.set_cell_result(
            str(cell_id),
            content=result,
            error=error,
            execution_time=execution_time
        )
        
        if not db_cell:
            logger.error(f"Cell {cell_id} not found")
            raise KeyError(f"Cell {cell_id} not found")
        
        logger.info(f"Set result for cell {cell_id}")
    
    def mark_dependents_stale(self, db: DbSession, notebook_id: UUID, cell_id: UUID) -> Set[UUID]:
        """Mark all cells that depend on this cell as stale"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        # Optional: Validate cell belongs to the notebook?
        # Get dependents from database
        db_dependents = repository.get_dependents(str(cell_id))
        if not db_dependents:
            return set()
        
        # Mark each dependent as stale
        stale_cells = set()
        for db_dependent in db_dependents:
            dependent_id = repository._ensure_str_id(db_dependent.id) # Use repo helper
            if dependent_id:
                # Use the same repository instance for marking stale
                repository.mark_cell_stale(dependent_id) 
                stale_cells.add(UUID(dependent_id))
        
        logger.info(f"Marked {len(stale_cells)} dependents of cell {cell_id} as stale")
        return stale_cells

    def reorder_cells(self, db: DbSession, notebook_id: UUID, cell_order: List[UUID]) -> None:
        """Reorder cells in a notebook"""
        # Instantiate repository locally
        repository = NotebookRepository(db)
        success = repository.reorder_cells(str(notebook_id), [str(c_id) for c_id in cell_order])
        if not success:
             logger.error(f"Failed to reorder cells in notebook {notebook_id}")
             raise ValueError(f"Failed to reorder cells in notebook {notebook_id}") # Or specific error

        logger.info(f"Reordered cells in notebook {notebook_id}")
    
    # Helper methods _db_notebook_to_model and _db_cell_to_model do not need DB access directly
    # They operate on data already fetched from the DB
    def _db_notebook_to_model(self, db_notebook) -> Notebook:
        notebook_data = db_notebook.to_dict()
        cells = {
             UUID(cell_id): self._db_cell_to_model(cell_obj) 
             for cell_id, cell_obj in db_notebook.cells.items()
        } if hasattr(db_notebook, 'cells') and isinstance(db_notebook.cells, dict) else {
             UUID(cell.id): self._db_cell_to_model(cell) 
             for cell in getattr(db_notebook, 'cells', []) # Handle relationship list case
        }
        
        return Notebook(
            id=UUID(notebook_data["id"]),
            metadata=NotebookMetadata(
                title=notebook_data["metadata"]["title"],
                description=notebook_data["metadata"]["description"],
                created_by=notebook_data["metadata"]["created_by"],
                created_at=datetime.fromisoformat(notebook_data["metadata"]["created_at"]) if notebook_data["metadata"]["created_at"] else datetime.now(timezone.utc),
                updated_at=datetime.fromisoformat(notebook_data["metadata"]["updated_at"]) if notebook_data["metadata"]["updated_at"] else datetime.now(timezone.utc),
                tags=notebook_data["metadata"]["tags"]
            ),
             cells=cells,
             cell_order=[UUID(cell_id) for cell_id in notebook_data["cell_order"]],
             # tool_call_records might need loading here if not handled by relationship/to_dict
        )

    def _db_cell_to_model(self, db_cell) -> Cell:
        cell_dict = db_cell.to_dict()
        
        result = None
        if cell_dict["result"] is not None:
            result = CellResult(
                content=cell_dict["result"]["content"],
                error=cell_dict["result"]["error"],
                execution_time=cell_dict["result"]["execution_time"],
                timestamp=datetime.fromisoformat(cell_dict["result"]["timestamp"]) if cell_dict["result"]["timestamp"] else datetime.now(timezone.utc)
            )
        
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
            
        # tool_call_ids might need loading here if not handled by relationship/to_dict
        
        return cell