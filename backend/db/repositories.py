"""
Repository module for database operations
"""

import logging
from typing import Dict, List, Optional, Any, cast

from sqlalchemy import select, delete, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Connection, DefaultConnection

# Configure logging
logger = logging.getLogger(__name__)


class ConnectionRepository:
    """Repository for connection database operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        logger.info("ConnectionRepository initialized with session: %s", session, extra={'correlation_id': 'N/A'})
    
    async def get_all(self) -> List[Connection]:
        """Get all connections"""
        logger.info("Fetching all connections", extra={'correlation_id': 'N/A'})
        query = select(Connection)
        result = await self.session.execute(query)
        connections = list(result.scalars().all())
        logger.info("Retrieved %d connections", len(connections), extra={'correlation_id': 'N/A'})
        return connections
    
    async def get_by_id(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID"""
        logger.info("Fetching connection with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
        query = select(Connection).where(Connection.id == connection_id)
        result = await self.session.execute(query)
        connection = result.scalar_one_or_none()
        if connection:
            logger.info("Found connection: %s", connection.name, extra={'correlation_id': 'N/A'})
        else:
            logger.info("No connection found with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
        return connection
    
    async def get_by_type(self, connection_type: str) -> List[Connection]:
        """Get connections by type"""
        logger.info("Fetching connections of type: %s", connection_type, extra={'correlation_id': 'N/A'})
        query = select(Connection).where(Connection.type == connection_type)
        result = await self.session.execute(query)
        connections = list(result.scalars().all())
        logger.info("Retrieved %d connections of type %s", len(connections), connection_type, extra={'correlation_id': 'N/A'})
        return connections
    
    async def get_by_name_and_type(self, name: str, connection_type: str) -> Optional[Connection]:
        """Get connection by name and type"""
        logger.info(f"Getting connection with name {name} and type {connection_type}", extra={'correlation_id': 'N/A'})
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
    
    async def create(self, name: str, connection_type: str, config: Dict[str, Any]) -> Connection:
        """Create a new connection"""
        logger.info("Creating new connection: %s (type: %s)", name, connection_type, extra={'correlation_id': 'N/A'})
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
    
    async def update(self, connection_id: str, name: str, config: Dict[str, Any]) -> Optional[Connection]:
        """Update an existing connection"""
        logger.info("Updating connection with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
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
    
    async def delete(self, connection_id: str) -> bool:
        """Delete a connection"""
        logger.info("Deleting connection with ID: %s", connection_id, extra={'correlation_id': 'N/A'})
        query = delete(Connection).where(Connection.id == connection_id)
        result = await self.session.execute(query)
        await self.session.commit()
        success = result.rowcount > 0
        if success:
            logger.info("Connection %s deleted successfully", connection_id, extra={'correlation_id': 'N/A'})
        else:
            logger.info("Connection %s not found for deletion", connection_id, extra={'correlation_id': 'N/A'})
        return success
    
    async def get_default_connection(self, connection_type: str) -> Optional[Connection]:
        """Get the default connection for a type"""
        logger.info("Fetching default connection for type: %s", connection_type, extra={'correlation_id': 'N/A'})
        query = select(DefaultConnection).where(DefaultConnection.connection_type == connection_type)
        result = await self.session.execute(query)
        default_conn = result.scalar_one_or_none()
        
        if default_conn:
            # Extract the string value from the DefaultConnection object
            conn_id = default_conn.connection_id
            if isinstance(conn_id, str):
                logger.info("Found default connection ID: %s", conn_id, extra={'correlation_id': 'N/A'})
                return await self.get_by_id(conn_id)
        logger.info("No default connection found for type: %s", connection_type, extra={'correlation_id': 'N/A'})
        return None
    
    async def set_default_connection(self, connection_type: str, connection_id: str) -> DefaultConnection:
        """Set a connection as default for its type"""
        logger.info("Setting connection %s as default for type: %s", connection_id, connection_type, extra={'correlation_id': 'N/A'})
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