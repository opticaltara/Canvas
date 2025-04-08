"""
Database connection setup
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

# Create SQLAlchemy Base
Base = declarative_base()

# Session factory to be configured at runtime
SessionLocal = None

@asynccontextmanager
async def get_db_session():
    """Get a database session that automatically closes when done"""
    if SessionLocal is None:
        raise RuntimeError("Database session not initialized")
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close() 