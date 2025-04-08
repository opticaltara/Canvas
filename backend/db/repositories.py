"""
Repository module for database operations
"""

import logging
from typing import Dict, List, Optional, Any, cast

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Connection, DefaultConnection

# Configure logging
logger = logging.getLogger(__name__)


class ConnectionRepository:
    """Repository for connection database operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_all(self) -> List[Connection]:
        """Get all connections"""
        query = select(Connection)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_id(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID"""
        query = select(Connection).where(Connection.id == connection_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_type(self, connection_type: str) -> List[Connection]:
        """Get connections by type"""
        query = select(Connection).where(Connection.type == connection_type)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def create(self, name: str, connection_type: str, config: Dict[str, Any]) -> Connection:
        """Create a new connection"""
        connection = Connection(
            name=name,
            type=connection_type,
            config=config
        )
        self.session.add(connection)
        await self.session.commit()
        await self.session.refresh(connection)
        return connection
    
    async def update(self, connection_id: str, name: str, config: Dict[str, Any]) -> Optional[Connection]:
        """Update an existing connection"""
        connection = await self.get_by_id(connection_id)
        if not connection:
            return None
        
        # Use SQLAlchemy update statement instead of direct attribute assignment
        stmt = update(Connection).where(Connection.id == connection_id).values(name=name, config=config)
        await self.session.execute(stmt)
        await self.session.commit()
        
        # Refresh and return the updated connection
        return await self.get_by_id(connection_id)
    
    async def delete(self, connection_id: str) -> bool:
        """Delete a connection"""
        query = delete(Connection).where(Connection.id == connection_id)
        result = await self.session.execute(query)
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_default_connection(self, connection_type: str) -> Optional[Connection]:
        """Get the default connection for a type"""
        query = select(DefaultConnection).where(DefaultConnection.connection_type == connection_type)
        result = await self.session.execute(query)
        default_conn = result.scalar_one_or_none()
        
        if default_conn:
            # Extract the string value from the DefaultConnection object
            conn_id = default_conn.connection_id
            if isinstance(conn_id, str):
                return await self.get_by_id(conn_id)
        return None
    
    async def set_default_connection(self, connection_type: str, connection_id: str) -> DefaultConnection:
        """Set a connection as default for its type"""
        # Check if a default connection already exists for this type
        query = select(DefaultConnection).where(DefaultConnection.connection_type == connection_type)
        result = await self.session.execute(query)
        default_conn = result.scalar_one_or_none()
        
        if default_conn:
            # Update existing default using SQLAlchemy update statement
            stmt = update(DefaultConnection).where(
                DefaultConnection.connection_type == connection_type
            ).values(connection_id=connection_id)
            await self.session.execute(stmt)
        else:
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
        return result.scalar_one() 