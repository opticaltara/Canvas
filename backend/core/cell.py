"""
Cell Module for Sherlog Canvas

This module defines the core cell types and their functionality for the Sherlog Canvas
notebook system. It provides a flexible cell model that supports different data sources
and query types.

Key components:
- Cell class: Base class for all cell types with common functionality
- Specialized cell classes for different sources (SQL, Log, Metric, etc.)
- CellResult class to represent execution results
- Utility functions for cell management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class CellStatus(str, Enum):
    """
    Enumeration of possible cell execution states
    
    Attributes:
        IDLE: Cell has not been executed yet
        QUEUED: Cell is queued for execution
        RUNNING: Cell is currently executing
        SUCCESS: Cell executed successfully
        ERROR: Cell execution resulted in an error
        STALE: Cell needs to be re-executed due to dependency changes
    """
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    STALE = "stale"


class CellType(str, Enum):
    """
    Enumeration of supported cell types
    
    Attributes:
        AI_QUERY: Cell containing a query to the AI system
        SQL: Cell containing SQL queries
        PYTHON: Cell containing Python code
        MARKDOWN: Cell containing markdown text
        LOG: Cell for log analysis (e.g., Loki)
        METRIC: Cell for metric queries (e.g., PromQL)
        S3: Cell for S3 object storage queries
    """
    AI_QUERY = "ai_query"
    SQL = "sql"
    PYTHON = "python"
    MARKDOWN = "markdown"
    LOG = "log"
    METRIC = "metric"
    S3 = "s3"


class CellResult(BaseModel):
    """
    The result of a cell execution
    
    Attributes:
        content: The primary result data (can be any type)
        error: Optional error message if execution failed
        execution_time: Time taken to execute the cell in seconds
        timestamp: When the result was generated
    """
    content: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Cell(BaseModel):
    """
    Base class for all notebook cells
    
    Attributes:
        id: Unique identifier for the cell
        type: The type of cell (SQL, Python, etc.)
        content: The cell's content (code, query, markdown, etc.)
        result: Optional result from cell execution
        status: Current execution status
        created_at: When the cell was created
        updated_at: When the cell was last updated
        dependencies: Set of cell IDs this cell depends on
        dependents: Set of cell IDs that depend on this cell
        metadata: Additional cell metadata
        connection_id: ID of the connection to use (0 or 1)
    """
    id: UUID = Field(default_factory=uuid4)
    type: CellType
    content: str
    result: Optional[CellResult] = None
    status: CellStatus = CellStatus.IDLE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    connection_id: Optional[int] = None
    
    # Dependency handling
    dependencies: Set[UUID] = Field(default_factory=set)
    dependents: Set[UUID] = Field(default_factory=set)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Settings for cell execution
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('connection_id')
    def validate_connection_id(cls, v):
        if v is not None and v not in [0, 1]:
            raise ValueError('connection_id must be either 0 or 1')
        return v
    
    def mark_stale(self) -> None:
        """Mark this cell as stale (needs re-execution)"""
        if self.status != CellStatus.STALE:
            self.status = CellStatus.STALE
            self.updated_at = datetime.utcnow()
    
    def update_content(self, content: str) -> None:
        """
        Update the cell content and mark as stale
        
        Args:
            content: The new content for the cell
        """
        self.content = content
        self.updated_at = datetime.utcnow()
        self.mark_stale()
    
    def set_result(self, result: Any, error: Optional[str] = None, execution_time: float = 0.0) -> None:
        """
        Set the execution result for this cell
        
        Args:
            result: The result data
            error: Optional error message
            execution_time: Time taken to execute the cell
        """
        self.result = CellResult(
            content=result,
            error=error,
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )
        self.status = CellStatus.ERROR if error else CellStatus.SUCCESS
        self.updated_at = datetime.utcnow()
    
    def add_dependency(self, cell_id: UUID) -> None:
        """
        Add a dependency on another cell
        
        Args:
            cell_id: ID of the cell this cell depends on
        """
        self.dependencies.add(cell_id)
    
    def add_dependent(self, cell_id: UUID) -> None:
        """
        Add a cell that depends on this cell
        
        Args:
            cell_id: ID of the dependent cell
        """
        self.dependents.add(cell_id)
    
    def remove_dependency(self, cell_id: UUID) -> None:
        """
        Remove a dependency on another cell
        
        Args:
            cell_id: ID of the dependency to remove
        """
        if cell_id in self.dependencies:
            self.dependencies.remove(cell_id)
    
    def remove_dependent(self, cell_id: UUID) -> None:
        """
        Remove a cell that depends on this cell
        
        Args:
            cell_id: ID of the dependent to remove
        """
        if cell_id in self.dependents:
            self.dependents.remove(cell_id)
    
    def clear_dependencies(self) -> None:
        """Clear all dependencies"""
        self.dependencies.clear()


class AIQueryCell(Cell):
    """
    A cell containing a query to the AI system
    
    This cell type is used to ask the AI to perform investigations or analyses.
    The AI can generate a plan and create other cells in response.
    
    Attributes:
        thinking: Optional text showing the AI's thinking process
        generated_cells: List of cell IDs generated from this query
    """
    type: CellType = CellType.AI_QUERY
    thinking: Optional[str] = None
    generated_cells: List[UUID] = Field(default_factory=list)


class SQLCell(Cell):
    """
    A cell containing an SQL query
    
    This cell type executes SQL against connected databases.
    """
    type: CellType = CellType.SQL
    
    class Config:
        schema_extra = {
            "example": {
                "content": "SELECT * FROM users WHERE created_at > NOW() - INTERVAL '1 day';"
            }
        }


class PythonCell(Cell):
    """
    A cell containing Python code
    
    This cell type executes Python code for data analysis and visualization.
    
    Attributes:
        type: Always set to PYTHON
        use_sandbox: Whether to use the sandboxed execution environment
        dependencies: Python package dependencies for the sandboxed environment
    """
    type: CellType = CellType.PYTHON
    
    class Config:
        schema_extra = {
            "example": {
                "content": "import pandas as pd\ndf = pd.DataFrame(result)\ndf.groupby('status').count()",
                "settings": {
                    "use_sandbox": True,
                    "dependencies": ["pandas", "numpy"]
                }
            }
        }


class MarkdownCell(Cell):
    """
    A cell containing markdown content
    
    This cell type is used for documentation and explanations.
    
    Attributes:
        type: Always set to MARKDOWN
    """
    type: CellType = CellType.MARKDOWN
    
    class Config:
        schema_extra = {
            "example": {
                "content": "# Analysis Results\nThe query found **15 users** who registered in the last 24 hours."
            }
        }


class LogCell(Cell):
    """
    A cell for log analysis queries
    
    This cell type is used to query log data from sources like Loki.
    
    Attributes:
        source: The log source (e.g., "loki", "grafana")
        time_range: Optional time range for the query
    """
    type: CellType = CellType.LOG
    source: str = "loki"  # Default to Loki, could be other log sources
    time_range: Optional[Dict[str, str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "content": '{app="backend"} |= "error" | logfmt | rate[5m]',
                "source": "loki",
                "time_range": {"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"}
            }
        }


class MetricCell(Cell):
    """
    A cell for metric queries (e.g., PromQL)
    
    This cell type is used to query time series metrics from sources like Prometheus.
    
    Attributes:
        source: The metric source (e.g., "prometheus", "grafana")
        time_range: Optional time range for the query
    """
    type: CellType = CellType.METRIC
    source: str = "prometheus"
    time_range: Optional[Dict[str, str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "content": 'rate(http_requests_total{status="500"}[5m])',
                "source": "prometheus",
                "time_range": {"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"}
            }
        }


class S3Cell(Cell):
    """
    A cell for S3 object queries
    
    This cell type is used to query or list objects in S3 buckets.
    
    Attributes:
        bucket: The S3 bucket name
        prefix: Optional object key prefix to filter results
    """
    type: CellType = CellType.S3
    bucket: str
    prefix: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "content": "SELECT * FROM s3object s WHERE s._1 LIKE '%ERROR%'",
                "bucket": "logs-bucket",
                "prefix": "app/logs/2023/01/01/"
            }
        }


def create_cell(cell_type: CellType, content: str, **kwargs) -> Cell:
    """
    Factory function to create cells of different types
    
    Args:
        cell_type: The type of cell to create
        content: The content for the cell
        **kwargs: Additional arguments for the specific cell type
        
    Returns:
        A cell instance of the specified type
        
    Raises:
        ValueError: If an unknown cell type is specified
    """
    cell_classes = {
        CellType.AI_QUERY: AIQueryCell,
        CellType.SQL: SQLCell,
        CellType.PYTHON: PythonCell,
        CellType.MARKDOWN: MarkdownCell,
        CellType.LOG: LogCell,
        CellType.METRIC: MetricCell,
        CellType.S3: S3Cell,
    }
    
    cell_class = cell_classes.get(cell_type)
    if not cell_class:
        raise ValueError(f"Unknown cell type: {cell_type}")
    
    # Extract connection_id from kwargs if present
    connection_id = kwargs.pop('connection_id', None)
    
    # Create the cell
    cell = cell_class(content=content, **kwargs)
    
    # Set connection_id if provided
    if connection_id is not None:
        cell.connection_id = connection_id
    
    return cell