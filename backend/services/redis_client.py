"""
Redis Client Dependency

Provides a dependency function to get an initialized aioredis client.
"""

import aioredis
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from backend.config import get_settings

# Global variable to hold the client, managed by lifespan
_redis_client: aioredis.Redis | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage Redis connection pool during app lifespan."""
    global _redis_client
    settings = get_settings()
    try:
        _redis_client = await aioredis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
            password=settings.redis_password,
            encoding="utf-8",
            decode_responses=True
        )
        await _redis_client.ping() # Verify connection on startup
        app.state.redis = _redis_client # Store client in app state
        print("Redis connection pool created and attached to app state.") # Use logger in real app
        yield # Application runs here
    finally:
        if _redis_client:
            await _redis_client.close()
            print("Redis connection pool closed.") # Use logger in real app
        _redis_client = None
        app.state.redis = None

async def get_redis_client(request: Request) -> aioredis.Redis:
    """
    Dependency function to get the Redis client stored in app state.
    """
    redis_client = request.app.state.redis
    if redis_client is None:
        # This should theoretically not happen if lifespan is configured correctly
        print("ERROR: Redis client not found in app state. Lifespan manager might not be configured.")
        raise HTTPException(status_code=500, detail="Redis service not available.")
    # Optional: Add a quick ping check here if needed, but generally rely on lifespan startup check
    # try:
    #     await redis_client.ping()
    # except aioredis.RedisError:
    #     raise HTTPException(status_code=503, detail="Redis connection lost.")
    return redis_client

# close_redis_client might not be needed externally anymore as lifespan handles it
async def close_redis_client(redis: aioredis.Redis):
     """
     Cleanly close the redis connection. (Mainly for lifespan now)
     """
     if redis:
         await redis.close()