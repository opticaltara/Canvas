"""
Dependency Graph Module for Sherlog Canvas

This module provides the core functionality for tracking and managing dependencies
between cells in a notebook. It implements a directed graph structure to maintain
relationships between cells and supports operations like finding execution order,
detecting cycles, and propagating changes.

Key components:
- DependencyGraph: Main class that tracks cell dependencies
- Helper functions for traversing the dependency graph
- Cycle detection to prevent invalid dependencies
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
from uuid import UUID

from pydantic import BaseModel

from backend.core.cell import Cell


class DependencyGraph(BaseModel):
    """
    Manages the dependency relationships between cells.
    
    This class tracks which cells depend on others and provides methods
    to find execution order and propagate stale states. It maintains a
    directed graph representation of cell dependencies.
    
    Attributes:
        dependents: Maps cell ID to the set of cell IDs that depend on it
        dependencies: Maps cell ID to the set of cell IDs it depends on
    """
    # Maps cell ID to the set of cell IDs that depend on it
    dependents: Dict[UUID, Set[UUID]] = defaultdict(set)
    
    # Maps cell ID to the set of cell IDs it depends on
    dependencies: Dict[UUID, Set[UUID]] = defaultdict(set)
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_cell(self, cell: Cell) -> None:
        """
        Add a cell to the dependency graph
        
        Args:
            cell: The cell to add
        """
        # Initialize empty sets if this is a new cell
        if cell.id not in self.dependents:
            self.dependents[cell.id] = set()
        if cell.id not in self.dependencies:
            self.dependencies[cell.id] = set()
    
    def add_dependency(self, dependent_id: UUID, dependency_id: UUID) -> None:
        """
        Add a dependency relationship: dependent_id depends on dependency_id
        
        Args:
            dependent_id: ID of the cell that depends on another
            dependency_id: ID of the cell that is depended upon
        """
        # Initialize sets if they don't exist
        if dependent_id not in self.dependencies:
            self.dependencies[dependent_id] = set()
        if dependency_id not in self.dependents:
            self.dependents[dependency_id] = set()
        
        # Add the relationship
        self.dependencies[dependent_id].add(dependency_id)
        self.dependents[dependency_id].add(dependent_id)
    
    def remove_dependency(self, dependent_id: UUID, dependency_id: UUID) -> None:
        """
        Remove a dependency relationship
        
        Args:
            dependent_id: ID of the dependent cell
            dependency_id: ID of the dependency cell
        """
        if dependent_id in self.dependencies:
            self.dependencies[dependent_id].discard(dependency_id)
        
        if dependency_id in self.dependents:
            self.dependents[dependency_id].discard(dependent_id)
    
    def remove_cell(self, cell_id: UUID) -> None:
        """
        Remove a cell and all its dependency relationships
        
        Args:
            cell_id: ID of the cell to remove
        """
        # Remove this cell as a dependency of other cells
        for dependent_id in list(self.dependents.get(cell_id, set())):
            self.dependencies[dependent_id].discard(cell_id)
        
        # Remove this cell's dependencies
        for dependency_id in list(self.dependencies.get(cell_id, set())):
            self.dependents[dependency_id].discard(cell_id)
        
        # Remove the cell from our maps
        self.dependents.pop(cell_id, None)
        self.dependencies.pop(cell_id, None)
    
    def get_dependents(self, cell_id: UUID) -> Set[UUID]:
        """
        Get all cells that directly depend on the given cell
        
        Args:
            cell_id: The cell ID to get dependents for
            
        Returns:
            Set of cell IDs that directly depend on the given cell
        """
        return self.dependents.get(cell_id, set())
    
    def get_dependencies(self, cell_id: UUID) -> Set[UUID]:
        """
        Get all cells that the given cell directly depends on
        
        Args:
            cell_id: The cell ID to get dependencies for
            
        Returns:
            Set of cell IDs that the given cell directly depends on
        """
        return self.dependencies.get(cell_id, set())
    
    def get_all_dependents(self, cell_id: UUID) -> Set[UUID]:
        """
        Get all cells that directly or indirectly depend on the given cell
        
        This performs a breadth-first traversal of the dependency graph
        to find all cells that would need to be re-executed if the given
        cell changes.
        
        Args:
            cell_id: The cell ID to get all dependents for
            
        Returns:
            Set of all cell IDs that directly or indirectly depend on the given cell
        """
        result = set()
        queue = deque([cell_id])
        
        while queue:
            current_id = queue.popleft()
            direct_dependents = self.dependents.get(current_id, set())
            
            for dependent_id in direct_dependents:
                if dependent_id not in result:
                    result.add(dependent_id)
                    queue.append(dependent_id)
        
        return result
    
    def get_execution_order(self, cell_ids: List[UUID] = None) -> List[UUID]:
        """
        Get a valid execution order for cells using topological sort
        
        If cell_ids is provided, only include those cells and their dependencies.
        This implementation uses a depth-first topological sort algorithm.
        
        Args:
            cell_ids: Optional list of cell IDs to include (and their dependencies)
            
        Returns:
            List of cell IDs in execution order (dependencies first)
            
        Raises:
            ValueError: If the dependency graph contains a cycle
        """
        if not cell_ids:
            # Include all cells
            cell_ids = list(set(list(self.dependencies.keys()) + list(self.dependents.keys())))
        
        # Find all cells needed (the provided cells and their dependencies)
        needed_cells = set(cell_ids)
        for cell_id in cell_ids:
            needed_cells.update(self.get_transitive_dependencies(cell_id))
        
        # Topological sort
        result = []
        temp_marks = set()
        perm_marks = set()
        
        def visit(node: UUID):
            if node in perm_marks:
                return
            if node in temp_marks:
                # This indicates a cycle, which shouldn't happen in our DAG
                raise ValueError(f"Dependency cycle detected involving cell {node}")
            
            temp_marks.add(node)
            
            for dep in self.dependencies.get(node, set()):
                if dep in needed_cells:  # Only visit needed cells
                    visit(dep)
            
            temp_marks.remove(node)
            perm_marks.add(node)
            result.append(node)
        
        for cell_id in needed_cells:
            if cell_id not in perm_marks:
                visit(cell_id)
        
        # Reverse to get correct execution order (dependencies first)
        return list(reversed(result))
    
    def get_transitive_dependencies(self, cell_id: UUID) -> Set[UUID]:
        """
        Get all cells that the given cell directly or indirectly depends on
        
        This performs a breadth-first traversal of the dependency graph
        to find all cells that would need to be executed before the given cell.
        
        Args:
            cell_id: The cell ID to get all dependencies for
            
        Returns:
            Set of all cell IDs that the given cell directly or indirectly depends on
        """
        result = set()
        queue = deque([cell_id])
        
        while queue:
            current_id = queue.popleft()
            direct_deps = self.dependencies.get(current_id, set())
            
            for dep_id in direct_deps:
                if dep_id not in result:
                    result.add(dep_id)
                    queue.append(dep_id)
        
        return result
    
    def check_for_cycles(self) -> Tuple[bool, List[UUID]]:
        """
        Check if the dependency graph has cycles
        
        Uses a depth-first search to detect cycles in the graph.
        
        Returns:
            Tuple of (has_cycle, cycle_nodes) where:
            - has_cycle: True if a cycle was found, False otherwise
            - cycle_nodes: List of cell IDs in the cycle, if one was found
        """
        # All nodes in the graph
        all_nodes = set(list(self.dependencies.keys()) + list(self.dependents.keys()))
        
        # DFS-based cycle detection
        unvisited = set(all_nodes)
        visiting = set()
        visited = set()
        cycle_nodes = []
        
        def dfs(node: UUID) -> bool:
            nonlocal cycle_nodes
            
            unvisited.discard(node)
            visiting.add(node)
            
            for dep in self.dependents.get(node, set()):
                if dep in visited:
                    continue
                if dep in visiting:
                    # Cycle detected
                    cycle_nodes = [node, dep]
                    return True
                if dfs(dep):
                    if not cycle_nodes[-1] == node:
                        cycle_nodes.append(node)
                    return True
            
            visiting.discard(node)
            visited.add(node)
            return False
        
        while unvisited:
            node = next(iter(unvisited))
            if dfs(node):
                return True, cycle_nodes
        
        return False, []