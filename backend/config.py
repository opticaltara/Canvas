import os
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    
    # Authentication settings (for future use)
    auth_enabled: bool = False
    
    # AI settings
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    
    # Storage settings
    notebook_storage_type: str = "file"  # Options: file, s3, none
    notebook_file_storage_dir: str = "./data/notebooks"
    notebook_s3_bucket: str = ""
    notebook_s3_prefix: str = "notebooks"
    
    # Connection storage settings
    connection_storage_type: str = "file"  # Options: file, env, none
    connection_file_path: str = "./data/connections.json"
    
    # AWS settings (for S3 storage)
    aws_access_key_id: str = Field("", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field("", env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field("us-east-1", env="AWS_REGION")
    
    # Cell execution settings
    python_cell_timeout: int = 30  # seconds
    python_cell_max_memory: int = 1024  # MB
    
    # Plugin settings
    default_query_timeout: int = 30  # seconds
    
    # Qdrant settings (for vector database)
    qdrant_url: str = Field("localhost", env="QDRANT_URL")
    qdrant_port: int = Field(6333, env="QDRANT_PORT") 
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    
    class Config:
        env_prefix = "SHERLOG_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()