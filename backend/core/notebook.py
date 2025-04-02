from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from backend.core.cell import Cell, CellStatus, CellType, create_cell
from backend.core.dependency import DependencyGraph


class NotebookMetadata(BaseModel):
    """Metadata for a notebook"""
    title: str = "Untitled Investigation"
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Notebook(BaseModel):
    """
    A notebook containing a collection of cells with dependencies.
    Manages cell creation, updates, and execution.
    """
    id: UUID = Field(default_factory=uuid4)
    metadata: NotebookMetadata = Field(default_factory=NotebookMetadata)
    cells: Dict[UUID, Cell] = Field(default_factory=dict)
    cell_order: List[UUID] = Field(default_factory=list)
    dependency_graph: DependencyGraph = Field(default_factory=DependencyGraph)
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_cell(self, cell: Cell, position: Optional[int] = None) -> Cell:
        """
        Add a cell to the notebook
        
        Args:
            cell: The cell to add
            position: Optional position in the cell order (default: append to end)
        
        Returns:
            The added cell
        """
        self.cells[cell.id] = cell
        self.dependency_graph.add_cell(cell)
        
        # Update cell order
        if position is not None and 0 <= position <= len(self.cell_order):
            self.cell_order.insert(position, cell.id)
        else:
            self.cell_order.append(cell.id)
        
        # Update notebook metadata
        self.metadata.updated_at = datetime.utcnow()
        
        return cell
    
    def create_cell(
        self, 
        cell_type: CellType,
        content: str,
        dependencies: Optional[List[UUID]] = None,
        position: Optional[int] = None,
        **kwargs
    ) -> Cell:
        """
        Create a new cell and add it to the notebook
        
        Args:
            cell_type: Type of cell to create
            content: Content for the cell
            dependencies: Optional list of cell IDs this cell depends on
            position: Optional position in the cell order
            **kwargs: Additional arguments for the specific cell type
        
        Returns:
            The created cell
        """
        cell = create_cell(cell_type, content, **kwargs)
        
        # Add dependencies if provided
        if dependencies:
            for dep_id in dependencies:
                self.add_dependency(cell.id, dep_id)
        
        return self.add_cell(cell, position)
    
    def get_cell(self, cell_id: UUID) -> Cell:
        """Get a cell by ID"""
        if cell_id not in self.cells:
            raise ValueError(f"Cell not found: {cell_id}")
        return self.cells[cell_id]
    
    def remove_cell(self, cell_id: UUID) -> None:
        """Remove a cell from the notebook"""
        if cell_id not in self.cells:
            raise ValueError(f"Cell not found: {cell_id}")
        
        # Remove from cells dict
        cell = self.cells.pop(cell_id)
        
        # Remove from cell order
        if cell_id in self.cell_order:
            self.cell_order.remove(cell_id)
        
        # Remove from dependency graph
        self.dependency_graph.remove_cell(cell_id)
        
        # Update notebook metadata
        self.metadata.updated_at = datetime.utcnow()
    
    def update_cell_content(self, cell_id: UUID, content: str) -> Cell:
        """
        Update a cell's content and mark it and its dependents as stale
        
        Args:
            cell_id: ID of the cell to update
            content: New content for the cell
        
        Returns:
            The updated cell
        """
        cell = self.get_cell(cell_id)
        cell.update_content(content)
        
        # Mark all dependent cells as stale
        self.mark_dependents_stale(cell_id)
        
        # Update notebook metadata
        self.metadata.updated_at = datetime.utcnow()
        
        return cell
    
    def update_cell_metadata(self, cell_id: UUID, metadata: Dict) -> Cell:
        """Update a cell's metadata"""
        cell = self.get_cell(cell_id)
        cell.metadata.update(metadata)
        cell.updated_at = datetime.utcnow()
        
        # Update notebook metadata
        self.metadata.updated_at = datetime.utcnow()
        
        return cell
    
    def add_dependency(self, dependent_id: UUID, dependency_id: UUID) -> None:
        """
        Add a dependency relationship between cells
        
        Args:
            dependent_id: ID of the cell that depends on another
            dependency_id: ID of the cell that is depended upon
        """
        # Ensure both cells exist
        if dependent_id not in self.cells:
            raise ValueError(f"Dependent cell not found: {dependent_id}")
        if dependency_id not in self.cells:
            raise ValueError(f"Dependency cell not found: {dependency_id}")
        
        # Check for circular dependency
        transitive_deps = self.dependency_graph.get_transitive_dependencies(dependency_id)
        if dependent_id in transitive_deps:
            raise ValueError(f"Adding this dependency would create a cycle")
        
        # Update the cells
        dependent_cell = self.cells[dependent_id]
        dependency_cell = self.cells[dependency_id]
        
        dependent_cell.add_dependency(dependency_id)
        dependency_cell.add_dependent(dependent_id)
        
        # Update the dependency graph
        self.dependency_graph.add_dependency(dependent_id, dependency_id)
        
        # Mark the dependent cell as stale
        dependent_cell.mark_stale()
        
        # Update notebook metadata
        self.metadata.updated_at = datetime.utcnow()
    
    def remove_dependency(self, dependent_id: UUID, dependency_id: UUID) -> None:
        """Remove a dependency relationship between cells"""
        # Ensure both cells exist
        if dependent_id not in self.cells:
            raise ValueError(f"Dependent cell not found: {dependent_id}")
        if dependency_id not in self.cells:
            raise ValueError(f"Dependency cell not found: {dependency_id}")
        
        # Update the cells
        dependent_cell = self.cells[dependent_id]
        dependency_cell = self.cells[dependency_id]
        
        dependent_cell.remove_dependency(dependency_id)
        dependency_cell.remove_dependent(dependent_id)
        
        # Update the dependency graph
        self.dependency_graph.remove_dependency(dependent_id, dependency_id)
        
        # Mark the dependent cell as stale
        dependent_cell.mark_stale()
        
        # Update notebook metadata
        self.metadata.updated_at = datetime.utcnow()
    
    def mark_dependents_stale(self, cell_id: UUID) -> Set[UUID]:
        """
        Mark all cells that depend on this cell as stale
        
        Args:
            cell_id: The ID of the cell whose dependents should be marked stale
            
        Returns:
            Set of cell IDs that were marked stale
        """
        dependent_ids = self.dependency_graph.get_all_dependents(cell_id)
        
        for dep_id in dependent_ids:
            if dep_id in self.cells:
                self.cells[dep_id].mark_stale()
        
        return dependent_ids
    
    def get_execution_order(self, cell_ids: Optional[List[UUID]] = None) -> List[UUID]:
        """
        Get a valid execution order for the specified cells (or all cells if none specified)
        
        Args:
            cell_ids: Optional list of cell IDs to include (and their dependencies)
            
        Returns:
            List of cell IDs in execution order
        """
        return self.dependency_graph.get_execution_order(cell_ids)
    
    def move_cell(self, cell_id: UUID, position: int) -> None:
        """
        Move a cell to a new position in the cell order
        
        Args:
            cell_id: ID of the cell to move
            position: New position (index) in the cell order
        """
        if cell_id not in self.cells:
            raise ValueError(f"Cell not found: {cell_id}")
        
        if cell_id in self.cell_order:
            self.cell_order.remove(cell_id)
        
        if 0 <= position <= len(self.cell_order):
            self.cell_order.insert(position, cell_id)
        else:
            self.cell_order.append(cell_id)
        
        # Update notebook metadata
        self.metadata.updated_at = datetime.utcnow()
    
    def update_metadata(self, metadata: Dict) -> None:
        """Update the notebook metadata"""
        for key, value in metadata.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        
        self.metadata.updated_at = datetime.utcnow()
    
    def serialize(self) -> Dict:
        """Serialize the notebook to a dictionary for storage"""
        return {
            "id": str(self.id),
            "metadata": self.metadata.dict(),
            "cells": {str(cell_id): cell.dict() for cell_id, cell in self.cells.items()},
            "cell_order": [str(cell_id) for cell_id in self.cell_order],
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> Notebook:
        """Create a notebook from serialized data"""
        from backend.core.cell import CellType  # Import here to avoid circular imports
        
        notebook = cls(
            id=UUID(data["id"]),
            metadata=NotebookMetadata(**data["metadata"]),
            cell_order=[UUID(cell_id) for cell_id in data["cell_order"]],
        )
        
        # Recreate cells
        for cell_id_str, cell_data in data["cells"].items():
            cell_id = UUID(cell_id_str)
            cell_type = cell_data["type"]
            notebook.cells[cell_id] = create_cell(CellType(cell_type), **cell_data)
        
        # Rebuild dependency graph
        for cell_id, cell in notebook.cells.items():
            notebook.dependency_graph.add_cell(cell)
            for dep_id in cell.dependencies:
                notebook.dependency_graph.add_dependency(cell_id, dep_id)
        
        return notebook