from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from backend.core.cell import Cell
from backend.core.execution import ExecutionContext


class PluginConfig(BaseModel):
    """Base configuration for a data source plugin"""
    name: str
    description: str
    version: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class ConnectionConfig(BaseModel):
    """Configuration for a data source connection"""
    id: str
    name: str
    plugin_name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    

class PluginBase(ABC):
    """
    Base class for all data source plugins
    Defines the interface that all plugins must implement
    """
    def __init__(self, config: PluginConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        """Get the plugin name"""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Get the plugin description"""
        return self.config.description
    
    @property
    def version(self) -> str:
        """Get the plugin version"""
        return self.config.version
    
    @property
    def is_enabled(self) -> bool:
        """Check if the plugin is enabled"""
        return self.config.enabled
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.config.get(key, default)
    
    @abstractmethod
    async def validate_connection(self, connection_config: ConnectionConfig) -> bool:
        """
        Validate a connection configuration
        
        Args:
            connection_config: The connection configuration to validate
            
        Returns:
            True if the connection is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        connection_config: ConnectionConfig,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a query against the data source
        
        Args:
            query: The query to execute
            connection_config: The connection configuration
            parameters: Optional parameters for the query
            
        Returns:
            The query result
        """
        pass
    
    @abstractmethod
    async def get_schema(self, connection_config: ConnectionConfig) -> Dict[str, Any]:
        """
        Get the schema for the data source
        
        Args:
            connection_config: The connection configuration
            
        Returns:
            The schema information
        """
        pass


class CellExecutorPlugin(ABC):
    """
    A plugin that can execute a specific type of cell
    """
    @abstractmethod
    async def execute_cell(self, cell: Cell, context: ExecutionContext) -> Any:
        """
        Execute a cell
        
        Args:
            cell: The cell to execute
            context: The execution context
            
        Returns:
            The result of the execution
        """
        pass
    
    @abstractmethod
    async def validate_cell(self, cell: Cell) -> bool:
        """
        Validate a cell's content
        
        Args:
            cell: The cell to validate
            
        Returns:
            True if the cell is valid, False otherwise
        """
        pass