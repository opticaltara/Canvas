from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    openrouter_api_key: str = ""
    ai_model: str = "anthropic/claude-3.7-sonnet"
    # Logging settings
    logfire_token: str = ""
    environment: str = "development"
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
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None # Optional password
    chat_session_ttl: int = 3600 # Default TTL for chat sessions in Redis (1 hour)
    # AWS settings (for S3 storage)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    # Cell execution settings
    python_cell_timeout: int = 30  # seconds
    python_cell_max_memory: int = 1024  # MB
    # Query execution settings
    default_query_timeout: int = 30  # seconds

    model_config = SettingsConfigDict(
        env_prefix="SHERLOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )

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
    settings = Settings()
    print(f"OpenRouter API key: {settings.openrouter_api_key}")
    print(f"AI model: {settings.ai_model}")
    return settings