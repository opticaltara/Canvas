"""
Repository module for database operations
"""

import logging
from typing import Dict, List, Optional, Any, TypeVar, Generic, Union
from datetime import datetime, timezone

from sqlalchemy import select, delete, update, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from contextlib import contextmanager

from backend.db.models import Connection, DefaultConnection, Notebook, Cell, CellDependency
from backend.core.notebook import Notebook as NotebookModel, NotebookMetadata
from backend.core.cell import Cell as CellModel, CellType, CellStatus

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic repository functions
T = TypeVar('T')


class ConnectionRepository:
    """Repository for connection database operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        logger.info("ConnectionRepository initialized with session: %s", session, extra={'correlation_id': 'N/A'})
    
    async def get_all(self) -> List[Connection]:
        """Get all connections"""
        logger.info("Fetching all connections", extra={'correlation_id': 'N/A'})
        try:
            query = select(Connection)
            result = await self.session.execute(query)
            connections = list(result.scalars().all())
            logger.info("Retrieved %d connections", len(connections), extra={'correlation_id': 'N/A'})
            return connections
        except SQLAlchemyError as e:
            logger.error("Error fetching all connections: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def get_by_id(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID"""
        logger.info("Fetching connection with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
        try:
            query = select(Connection).where(Connection.id == connection_id)
            result = await self.session.execute(query)
            connection = result.scalar_one_or_none()
            if connection:
                logger.info("Found connection: %s", connection.name, extra={'correlation_id': 'N/A'})
            else:
                logger.info("No connection found with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
            return connection
        except SQLAlchemyError as e:
            logger.error("Error fetching connection with ID %s: %s", connection_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def get_by_type(self, connection_type: str) -> List[Connection]:
        """Get connections by type"""
        logger.info("Fetching connections of type: %s", connection_type, extra={'correlation_id': 'N/A'})
        try:
            query = select(Connection).where(Connection.type == connection_type)
            result = await self.session.execute(query)
            connections = list(result.scalars().all())
            logger.info("Retrieved %d connections of type %s", len(connections), connection_type, extra={'correlation_id': 'N/A'})
            return connections
        except SQLAlchemyError as e:
            logger.error("Error fetching connections of type %s: %s", connection_type, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def get_by_name_and_type(self, name: str, connection_type: str) -> Optional[Connection]:
        """Get connection by name and type"""
        logger.info("Getting connection with name %s and type %s", name, connection_type, extra={'correlation_id': 'N/A'})
        try:
            query = select(Connection).where(
                and_(
                    Connection.name == name,
                    Connection.type == connection_type
                )
            )
            result = await self.session.execute(query)
            connection = result.scalars().first()
            if connection:
                logger.info("Found connection: %s", connection.name, extra={'correlation_id': 'N/A'})
            else:
                logger.info("No connection found with name %s and type %s", name, connection_type, extra={'correlation_id': 'N/A'})
            return connection
        except SQLAlchemyError as e:
            logger.error("Error fetching connection with name %s and type %s: %s", name, connection_type, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def create(self, name: str, connection_type: str, config: Dict[str, Any]) -> Connection:
        """Create a new connection"""
        logger.info("Creating new connection: %s (type: %s)", name, connection_type, extra={'correlation_id': 'N/A'})
        try:
            connection = Connection(
                name=name,
                type=connection_type,
                config=config
            )
            self.session.add(connection)
            await self.session.commit()
            await self.session.refresh(connection)
            logger.info("Connection created successfully with ID: %s", connection.id, extra={'correlation_id': 'N/A'})
            return connection
        except IntegrityError as e:
            await self.session.rollback()
            logger.error("Integrity error creating connection: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Error creating connection: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def update(self, connection_id: str, name: str, config: Dict[str, Any]) -> Optional[Connection]:
        """Update an existing connection"""
        logger.info("Updating connection with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
        try:
            connection = await self.get_by_id(connection_id)
            if not connection:
                logger.info("Cannot update connection: ID %s not found", connection_id, extra={'correlation_id': 'N/A'})
                return None
            
            # Use SQLAlchemy update statement instead of direct attribute assignment
            stmt = update(Connection).where(Connection.id == connection_id).values(name=name, config=config)
            await self.session.execute(stmt)
            await self.session.commit()
            
            # Refresh and return the updated connection
            updated_connection = await self.get_by_id(connection_id)
            logger.info("Connection %s updated successfully", connection_id, extra={'correlation_id': 'N/A'})
            return updated_connection
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Error updating connection %s: %s", connection_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def delete(self, connection_id: str) -> bool:
        """Delete a connection"""
        logger.info("Deleting connection with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
        try:
            query = delete(Connection).where(Connection.id == connection_id)
            result = await self.session.execute(query)
            await self.session.commit()
            success = result.rowcount > 0
            if success:
                logger.info("Connection %s deleted successfully", connection_id, extra={'correlation_id': 'N/A'})
            else:
                logger.info("Connection %s not found for deletion", connection_id, extra={'correlation_id': 'N/A'})
            return success
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Error deleting connection %s: %s", connection_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def get_default_connection(self, connection_type: str) -> Optional[Connection]:
        """Get the default connection for a type"""
        logger.info("Fetching default connection for type: %s", connection_type, extra={'correlation_id': 'N/A'})
        try:
            query = select(DefaultConnection).where(DefaultConnection.connection_type == connection_type)
            result = await self.session.execute(query)
            default_conn = result.scalar_one_or_none()
            
            if default_conn:
                # Safely extract the connection ID
                conn_id = default_conn.connection_id
                conn_id_str = str(conn_id) if conn_id is not None else None
                if conn_id_str:
                    logger.info("Found default connection ID: %s", conn_id_str, extra={'correlation_id': 'N/A'})
                    return await self.get_by_id(conn_id_str)
            logger.info("No default connection found for type: %s", connection_type, extra={'correlation_id': 'N/A'})
            return None
        except SQLAlchemyError as e:
            logger.error("Error fetching default connection for type %s: %s", connection_type, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    async def set_default_connection(self, connection_type: str, connection_id: str) -> DefaultConnection:
        """Set a connection as default for its type"""
        logger.info("Setting connection %s as default for type: %s", connection_id, connection_type, extra={'correlation_id': 'N/A'})
        try:
            # First validate the connection exists
            connection = await self.get_by_id(connection_id)
            if not connection:
                raise ValueError(f"Connection with ID {connection_id} does not exist")
                
            # Check if a default connection already exists for this type
            query = select(DefaultConnection).where(DefaultConnection.connection_type == connection_type)
            result = await self.session.execute(query)
            default_conn = result.scalar_one_or_none()
            
            if default_conn:
                logger.info("Updating existing default connection for type: %s", connection_type, extra={'correlation_id': 'N/A'})
                # Update existing default using SQLAlchemy update statement
                stmt = update(DefaultConnection).where(
                    DefaultConnection.connection_type == connection_type
                ).values(connection_id=connection_id)
                await self.session.execute(stmt)
            else:
                logger.info("Creating new default connection entry for type: %s", connection_type, extra={'correlation_id': 'N/A'})
                # Create new default
                default_conn = DefaultConnection(
                    connection_type=connection_type,
                    connection_id=connection_id
                )
                self.session.add(default_conn)
            
            await self.session.commit()
            
            # Re-fetch the default connection to return
            query = select(DefaultConnection).where(DefaultConnection.connection_type == connection_type)
            result = await self.session.execute(query)
            default_conn = result.scalar_one()
            logger.info("Successfully set connection %s as default for type: %s", connection_id, connection_type, extra={'correlation_id': 'N/A'})
            return default_conn
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Error setting default connection: %s", str(e), extra={'correlation_id': 'N/A'})
            raise


class NotebookRepository:
    """Repository for notebook operations"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        logger.info("NotebookRepository initialized with session: %s", db_session, extra={'correlation_id': 'N/A'})
    
    @contextmanager
    def _transaction(self):
        """Context manager for transaction handling"""
        try:
            yield
            self.db.commit()
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error("Transaction error: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def _ensure_str_id(self, id_value: Union[str, Any]) -> Optional[str]:
        """Ensure ID is a string"""
        if id_value is None:
            return None
        return str(id_value)
    
    def _update_notebook_timestamp(self, notebook_id: Optional[str]):
        """Update notebook's updated_at timestamp"""
        if not notebook_id:
            return
        
        try:
            stmt = update(Notebook).where(Notebook.id == notebook_id).values(
                updated_at=datetime.now(timezone.utc)
            )
            self.db.execute(stmt)
        except SQLAlchemyError as e:
            logger.error("Error updating notebook timestamp: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def create_notebook(self, title: str, description: Optional[str] = None, 
                        created_by: Optional[str] = None, tags: Optional[List[str]] = None) -> Notebook:
        """Create a new notebook in the database"""
        logger.info("Creating new notebook: %s", title, extra={'correlation_id': 'N/A'})
        try:
            notebook = Notebook(
                title=title,
                description=description,
                created_by=created_by,
                tags=tags or []
            )
            
            with self._transaction():
                self.db.add(notebook)
                self.db.flush()  # Flush to get the ID before commit
                self.db.refresh(notebook)
                logger.info("Notebook created with ID: %s", notebook.id, extra={'correlation_id': 'N/A'})
            
            return notebook
        except SQLAlchemyError as e:
            logger.error("Error creating notebook: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def get_notebook(self, notebook_id: str) -> Optional[Notebook]:
        """Get a notebook by ID"""
        logger.info("Fetching notebook with ID: %s", notebook_id, extra={'correlation_id': 'N/A'})
        try:
            notebook = self.db.query(Notebook).filter(Notebook.id == notebook_id).first()
            if notebook:
                logger.info("Found notebook: %s", notebook.title, extra={'correlation_id': 'N/A'})
            else:
                logger.info("No notebook found with ID: %s", notebook_id, extra={'correlation_id': 'N/A'})
            return notebook
        except SQLAlchemyError as e:
            logger.error("Error fetching notebook %s: %s", notebook_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def list_notebooks(self) -> List[Notebook]:
        """Get all notebooks"""
        logger.info("Fetching all notebooks", extra={'correlation_id': 'N/A'})
        try:
            notebooks = self.db.query(Notebook).all()
            logger.info("Retrieved %d notebooks", len(notebooks), extra={'correlation_id': 'N/A'})
            return notebooks
        except SQLAlchemyError as e:
            logger.error("Error fetching all notebooks: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def update_notebook(self, notebook_id: str, **kwargs) -> Optional[Notebook]:
        """Update notebook attributes"""
        logger.info("Updating notebook with ID: %s", notebook_id, extra={'correlation_id': 'N/A'})
        try:
            notebook = self.get_notebook(notebook_id)
            if not notebook:
                logger.info("Cannot update notebook: ID %s not found", notebook_id, extra={'correlation_id': 'N/A'})
                return None
            
            # Filter kwargs to only include valid attributes
            valid_attrs = {k: v for k, v in kwargs.items() if hasattr(Notebook, k)}
            valid_attrs['updated_at'] = datetime.now(timezone.utc)
            
            # Use SQLAlchemy update statement
            with self._transaction():
                stmt = update(Notebook).where(Notebook.id == notebook_id).values(**valid_attrs)
                self.db.execute(stmt)
                logger.info("Notebook %s updated successfully", notebook_id, extra={'correlation_id': 'N/A'})
            
            return self.get_notebook(notebook_id)
        except SQLAlchemyError as e:
            logger.error("Error updating notebook %s: %s", notebook_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook"""
        logger.info("Deleting notebook with ID: %s", notebook_id, extra={'correlation_id': 'N/A'})
        try:
            notebook = self.get_notebook(notebook_id)
            if not notebook:
                logger.info("Cannot delete notebook: ID %s not found", notebook_id, extra={'correlation_id': 'N/A'})
                return False
            
            with self._transaction():
                # Delete all cells and dependencies associated with this notebook
                cells = self.db.query(Cell).filter(Cell.notebook_id == notebook_id).all()
                for cell in cells:
                    # Delete dependencies
                    self.db.query(CellDependency).filter(
                        (CellDependency.dependent_id == cell.id) | 
                        (CellDependency.dependency_id == cell.id)
                    ).delete()
                    # Delete cell
                    self.db.delete(cell)
                
                # Delete notebook
                self.db.delete(notebook)
                logger.info("Notebook %s deleted successfully", notebook_id, extra={'correlation_id': 'N/A'})
            
            return True
        except SQLAlchemyError as e:
            logger.error("Error deleting notebook %s: %s", notebook_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def create_cell(self, notebook_id: str, cell_type: str, content: str, 
                    position: int, connection_id: Optional[str] = None,
                    metadata: Optional[Dict] = None, settings: Optional[Dict] = None) -> Optional[Cell]:
        """Create a new cell in a notebook"""
        logger.info("Creating new cell in notebook %s", notebook_id, extra={'correlation_id': 'N/A'})
        try:
            notebook = self.get_notebook(notebook_id)
            if not notebook:
                logger.info("Cannot create cell: Notebook ID %s not found", notebook_id, extra={'correlation_id': 'N/A'})
                return None
            
            cell = Cell(
                notebook_id=notebook_id,
                type=cell_type,
                content=content,
                position=position,
                connection_id=connection_id,
                metadata=metadata or {},
                settings=settings or {}
            )
            
            with self._transaction():
                self.db.add(cell)
                self.db.flush()  # Flush to get the ID before commit
                self.db.refresh(cell)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Cell created with ID: %s in notebook %s", cell.id, notebook_id, extra={'correlation_id': 'N/A'})
            
            return cell
        except SQLAlchemyError as e:
            logger.error("Error creating cell in notebook %s: %s", notebook_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """Get a cell by ID"""
        logger.info("Fetching cell with ID: %s", cell_id, extra={'correlation_id': 'N/A'})
        try:
            cell = self.db.query(Cell).filter(Cell.id == cell_id).first()
            if cell:
                logger.info("Found cell with ID: %s", cell_id, extra={'correlation_id': 'N/A'})
            else:
                logger.info("No cell found with ID: %s", cell_id, extra={'correlation_id': 'N/A'})
            return cell
        except SQLAlchemyError as e:
            logger.error("Error fetching cell %s: %s", cell_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def update_cell(self, cell_id: str, **kwargs) -> Optional[Cell]:
        """Update cell attributes"""
        logger.info("Updating cell with ID: %s", cell_id, extra={'correlation_id': 'N/A'})
        try:
            cell = self.get_cell(cell_id)
            if not cell:
                logger.info("Cannot update cell: ID %s not found", cell_id, extra={'correlation_id': 'N/A'})
                return None
            
            # Extract notebook_id for timestamp update
            notebook_id = self._ensure_str_id(cell.notebook_id)
            
            # Filter kwargs to only include valid attributes
            valid_attrs = {k: v for k, v in kwargs.items() if hasattr(Cell, k)}
            valid_attrs['updated_at'] = datetime.now(timezone.utc)
            
            with self._transaction():
                # Use SQLAlchemy update statement
                stmt = update(Cell).where(Cell.id == cell_id).values(**valid_attrs)
                self.db.execute(stmt)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Cell %s updated successfully", cell_id, extra={'correlation_id': 'N/A'})
            
            return self.get_cell(cell_id)
        except SQLAlchemyError as e:
            logger.error("Error updating cell %s: %s", cell_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def delete_cell(self, cell_id: str) -> bool:
        """Delete a cell"""
        logger.info("Deleting cell with ID: %s", cell_id, extra={'correlation_id': 'N/A'})
        try:
            cell = self.get_cell(cell_id)
            if not cell:
                logger.info("Cannot delete cell: ID %s not found", cell_id, extra={'correlation_id': 'N/A'})
                return False
            
            # Extract notebook_id for timestamp update
            notebook_id = self._ensure_str_id(cell.notebook_id)
            
            with self._transaction():
                # Delete any dependencies
                self.db.query(CellDependency).filter(
                    (CellDependency.dependent_id == cell_id) | 
                    (CellDependency.dependency_id == cell_id)
                ).delete()
                
                # Delete the cell
                self.db.delete(cell)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Cell %s deleted successfully", cell_id, extra={'correlation_id': 'N/A'})
            
            return True
        except SQLAlchemyError as e:
            logger.error("Error deleting cell %s: %s", cell_id, str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def add_dependency(self, dependent_id: str, dependency_id: str) -> bool:
        """Add a dependency between cells"""
        logger.info("Adding dependency from cell %s to cell %s", dependent_id, dependency_id, extra={'correlation_id': 'N/A'})
        try:
            dependent = self.get_cell(dependent_id)
            dependency = self.get_cell(dependency_id)
            
            if not dependent or not dependency:
                logger.info("Cannot add dependency: Cells not found", extra={'correlation_id': 'N/A'})
                return False
            
            # Check if dependent and dependency are in the same notebook
            if self._ensure_str_id(dependent.notebook_id) != self._ensure_str_id(dependency.notebook_id):
                logger.error("Cannot add dependency: Cells belong to different notebooks", extra={'correlation_id': 'N/A'})
                return False
            
            # Extract notebook_id for timestamp update
            notebook_id = self._ensure_str_id(dependent.notebook_id)
            
            # Check if dependency already exists
            existing = self.db.query(CellDependency).filter(
                CellDependency.dependent_id == dependent_id,
                CellDependency.dependency_id == dependency_id
            ).first()
            
            if existing:
                logger.info("Dependency already exists", extra={'correlation_id': 'N/A'})
                return True
            
            with self._transaction():
                # Add dependency
                cell_dependency = CellDependency(
                    dependent_id=dependent_id,
                    dependency_id=dependency_id
                )
                
                self.db.add(cell_dependency)
                
                # Update cell status using update statement
                stmt = update(Cell).where(Cell.id == dependent_id).values(status="stale")
                self.db.execute(stmt)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Dependency added successfully", extra={'correlation_id': 'N/A'})
            
            return True
        except SQLAlchemyError as e:
            logger.error("Error adding dependency: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def remove_dependency(self, dependent_id: str, dependency_id: str) -> bool:
        """Remove a dependency between cells"""
        logger.info("Removing dependency from cell %s to cell %s", dependent_id, dependency_id, extra={'correlation_id': 'N/A'})
        try:
            # Get cell for notebook_id
            dependent = self.get_cell(dependent_id)
            if not dependent:
                logger.info("Dependent cell not found: %s", dependent_id, extra={'correlation_id': 'N/A'})
                return False
                
            # Extract notebook_id for timestamp update
            notebook_id = self._ensure_str_id(dependent.notebook_id)
            
            with self._transaction():
                # Find dependency
                dependency = self.db.query(CellDependency).filter(
                    CellDependency.dependent_id == dependent_id,
                    CellDependency.dependency_id == dependency_id
                ).first()
                
                if not dependency:
                    logger.info("Dependency not found", extra={'correlation_id': 'N/A'})
                    return False
                
                # Remove dependency
                self.db.delete(dependency)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Dependency removed successfully", extra={'correlation_id': 'N/A'})
            
            return True
        except SQLAlchemyError as e:
            logger.error("Error removing dependency: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def set_cell_result(self, cell_id: str, content: Any, error: Optional[str] = None, 
                        execution_time: float = 0.0) -> Optional[Cell]:
        """Set the execution result for a cell"""
        logger.info("Setting result for cell %s", cell_id, extra={'correlation_id': 'N/A'})
        try:
            cell = self.get_cell(cell_id)
            if not cell:
                logger.info("Cannot set result: Cell ID %s not found", cell_id, extra={'correlation_id': 'N/A'})
                return None
            
            # Extract notebook_id for timestamp update
            notebook_id = self._ensure_str_id(cell.notebook_id)
            
            with self._transaction():
                # Use SQLAlchemy update statement
                stmt = update(Cell).where(Cell.id == cell_id).values(
                    result_content=content,
                    result_error=error,
                    result_execution_time=execution_time,
                    result_timestamp=datetime.now(timezone.utc),
                    status="error" if error else "success",
                    updated_at=datetime.now(timezone.utc)
                )
                self.db.execute(stmt)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Cell result set successfully for cell %s", cell_id, extra={'correlation_id': 'N/A'})
            
            return self.get_cell(cell_id)
        except SQLAlchemyError as e:
            logger.error("Error setting cell result: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def mark_cell_stale(self, cell_id: str) -> Optional[Cell]:
        """Mark a cell as stale"""
        logger.info("Marking cell %s as stale", cell_id, extra={'correlation_id': 'N/A'})
        try:
            cell = self.get_cell(cell_id)
            if not cell:
                logger.info("Cannot mark cell stale: ID %s not found", cell_id, extra={'correlation_id': 'N/A'})
                return None
            
            # Extract notebook_id for timestamp update
            notebook_id = self._ensure_str_id(cell.notebook_id)
            
            with self._transaction():
                # Use SQLAlchemy update statement
                stmt = update(Cell).where(Cell.id == cell_id).values(
                    status="stale",
                    updated_at=datetime.now(timezone.utc)
                )
                self.db.execute(stmt)
                
                # Also mark any dependent cells as stale (recursive marking)
                dependents = self.get_dependents(cell_id)
                for dependent in dependents:
                    dependent_id = self._ensure_str_id(dependent.id)
                    if dependent_id and dependent_id != cell_id:  # Avoid circular dependencies
                        self.mark_cell_stale(dependent_id)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Cell %s marked as stale", cell_id, extra={'correlation_id': 'N/A'})
            
            return self.get_cell(cell_id)
        except SQLAlchemyError as e:
            logger.error("Error marking cell as stale: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def get_dependents(self, cell_id: str) -> List[Cell]:
        """Get all cells that depend on this cell"""
        logger.info("Getting dependents of cell %s", cell_id, extra={'correlation_id': 'N/A'})
        try:
            dependents = self.db.query(Cell).join(
                CellDependency, Cell.id == CellDependency.dependent_id
            ).filter(CellDependency.dependency_id == cell_id).all()
            logger.info("Found %d dependent cells for cell %s", len(dependents), cell_id, extra={'correlation_id': 'N/A'})
            return dependents
        except SQLAlchemyError as e:
            logger.error("Error getting cell dependents: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def get_dependencies(self, cell_id: str) -> List[Cell]:
        """Get all cells this cell depends on"""
        logger.info("Getting dependencies of cell %s", cell_id, extra={'correlation_id': 'N/A'})
        try:
            dependencies = self.db.query(Cell).join(
                CellDependency, Cell.id == CellDependency.dependency_id
            ).filter(CellDependency.dependent_id == cell_id).all()
            logger.info("Found %d dependencies for cell %s", len(dependencies), cell_id, extra={'correlation_id': 'N/A'})
            return dependencies
        except SQLAlchemyError as e:
            logger.error("Error getting cell dependencies: %s", str(e), extra={'correlation_id': 'N/A'})
            raise
    
    def reorder_cells(self, notebook_id: str, cell_order: List[str]) -> bool:
        """Update the order of cells in a notebook"""
        logger.info("Reordering cells in notebook %s", notebook_id, extra={'correlation_id': 'N/A'})
        try:
            notebook = self.get_notebook(notebook_id)
            if not notebook:
                logger.info("Cannot reorder cells: Notebook ID %s not found", notebook_id, extra={'correlation_id': 'N/A'})
                return False
            
            with self._transaction():
                # Validate that all cells exist and belong to this notebook
                existing_cells = self.db.query(Cell).filter(Cell.notebook_id == notebook_id).all()
                existing_ids = {self._ensure_str_id(cell.id) for cell in existing_cells}
                
                # Check if all cells in cell_order exist in the notebook
                for cell_id in cell_order:
                    if cell_id not in existing_ids:
                        logger.warning("Cell %s not found in notebook %s", cell_id, notebook_id, extra={'correlation_id': 'N/A'})
                        return False
                
                # Update position for each cell using update statement
                for position, cell_id in enumerate(cell_order):
                    stmt = update(Cell).where(Cell.id == cell_id).values(position=position)
                    self.db.execute(stmt)
                
                # Update notebook timestamp
                self._update_notebook_timestamp(notebook_id)
                logger.info("Cells reordered successfully in notebook %s", notebook_id, extra={'correlation_id': 'N/A'})
            
            return True
        except SQLAlchemyError as e:
            logger.error("Error reordering cells: %s", str(e), extra={'correlation_id': 'N/A'})
            raise