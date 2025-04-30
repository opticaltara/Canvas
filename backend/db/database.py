"""
Database connection setup
"""

import contextlib
import logging
from typing import AsyncIterator, Iterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from backend.config import get_settings

# Set up logger
logger = logging.getLogger(__name__)

# Create SQLAlchemy Base
Base = declarative_base()

# Get database URL from settings
settings = get_settings()
DATABASE_URL = settings.database_url
logger.info("Database URL: %s", DATABASE_URL, extra={'correlation_id': 'N/A'})

# Convert standard URLs to async URLs
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    SYNC_DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    logger.info("Converted to async PostgreSQL URL", extra={'correlation_id': 'N/A'})
elif DATABASE_URL.startswith("sqlite:///"):
    DATABASE_URL = DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
    SYNC_DATABASE_URL = DATABASE_URL.replace("sqlite+aiosqlite:///", "sqlite:///")
    logger.info("Converted to async SQLite URL", extra={'correlation_id': 'N/A'})
else:
    SYNC_DATABASE_URL = DATABASE_URL

# Create async engine and sessionmaker
logger.info("Creating async database engine", extra={'correlation_id': 'N/A'})
engine = create_async_engine(DATABASE_URL)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Create synchronous engine and sessionmaker for compatibility
from sqlalchemy import create_engine
sync_engine = create_engine(SYNC_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# Initialize database
async def init_db():
    """Initialize database by creating all tables"""
    logger.info("Initializing database tables", extra={'correlation_id': 'N/A'})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialization complete", extra={'correlation_id': 'N/A'})

@contextlib.asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Get a database session as an async context manager"""
    session = async_session_factory()
    logger.info("Database session created", extra={'correlation_id': 'N/A'})
    try:
        yield session
        await session.commit()
        logger.info("Session committed successfully", extra={'correlation_id': 'N/A'})
    except Exception as e:
        await session.rollback()
        logger.error("Session rollback due to error: %s", str(e), extra={'correlation_id': 'N/A'})
        raise
    finally:
        await session.close()
        logger.info("Session closed", extra={'correlation_id': 'N/A'})

def get_db() -> Iterator[Session]:
    """Get a synchronous database session"""
    db = SessionLocal()
    logger.info("Synchronous database session created", extra={'correlation_id': 'N/A'})
    try:
        yield db
    finally:
        db.close()
        logger.info("Synchronous session closed", extra={'correlation_id': 'N/A'}) 