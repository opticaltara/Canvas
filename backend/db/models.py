"""
Database models for connections
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, JSON, func, Text
from sqlalchemy.orm import relationship

from backend.db.database import Base


class Connection(Base):
    """Connection model for storing connection configurations"""
    __tablename__ = "connections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # grafana, kubernetes, s3, postgres, etc.
    config = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "config": self.config,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else None,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else None,
        }


class DefaultConnection(Base):
    """Default connection mapping for connection types"""
    __tablename__ = "default_connections"

    connection_type = Column(String, primary_key=True)
    connection_id = Column(String, ForeignKey("connections.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to the connection
    connection = relationship("Connection") 