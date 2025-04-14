from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


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
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    ai_model: str = Field(default="anthropic/claude-3.7-sonnet", validation_alias="SHERLOG_AI_MODEL")
    
    # Logging settings
    logfire_token: str = Field(default="", validation_alias="LOGFIRE_TOKEN")
    environment: str = Field(default="development", validation_alias="SHERLOG_ENV")
    
    # Connection storage settings
    connection_storage_type: str = "file"  # Options: file, env, db, none
    connection_file_path: str = "./data/connections.json"
    
    # Database settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sherlog"
    db_user: str = "sherlog"
    db_password: str = "sherlog"
    db_type: str = "sqlite"  # Options: sqlite, postgresql
    db_file: str = "./data/sherlog.db"
    
    # AWS settings (for S3 storage)
    aws_access_key_id: str = Field(default="", validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", validation_alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    
    # Cell execution settings
    python_cell_timeout: int = 30  # seconds
    python_cell_max_memory: int = 1024  # MB
    
    # Query execution settings
    default_query_timeout: int = 30  # seconds
    
    class Config:
        env_prefix = "SHERLOG_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    @property
    def database_url(self) -> str:
        """Get the database connection URL"""
        if self.db_type == "sqlite":
            return f"sqlite:///{self.db_file}"
        else:
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()