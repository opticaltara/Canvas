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

from datetime import datetime
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Import ToolCallID
from backend.core.dependency import ToolCallID

# Initialize logger
cell_logger = logging.getLogger("cell")

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
        MARKDOWN: Cell containing markdown text
        LOG: Cell for log analysis (e.g., Loki)
        METRIC: Cell for metric queries (e.g., PromQL)
        GITHUB: Cell containing a GitHub query or operation result
        SUMMARIZATION: Cell for summarization results
    """
    MARKDOWN = "markdown"
    LOG = "log"
    METRIC = "metric"
    GITHUB = "github"
    SUMMARIZATION = "summarization"


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
        tool_call_ids: List of IDs for tool calls associated with this cell
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
    
    # Tool calls associated with this cell
    tool_call_ids: List[ToolCallID] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Settings for cell execution
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        cell_logger.info(
            "Cell initialized",
            extra={
                'cell_id': str(self.id),
                'cell_type': str(self.type),
                'connection_id': self.connection_id
            }
        )
    
    @field_validator('connection_id')
    def validate_connection_id(cls, v):
        if v is not None and v not in [0, 1]:
            cell_logger.error(
                "Invalid connection_id",
                extra={
                    'connection_id': v,
                    'error': 'connection_id must be either 0 or 1'
                }
            )
            raise ValueError('connection_id must be either 0 or 1')
        return v
    
    def mark_stale(self) -> None:
        """Mark this cell as stale (needs re-execution)"""
        start_time = time.time()
        if self.status != CellStatus.STALE:
            self.status = CellStatus.STALE
            self.updated_at = datetime.utcnow()
            process_time = time.time() - start_time
            cell_logger.info(
                "Cell marked as stale",
                extra={
                    'cell_id': str(self.id),
                    'cell_type': str(self.type),
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
    
    def update_content(self, content: str) -> None:
        """
        Update the cell content and mark as stale
        
        Args:
            content: The new content for the cell
        """
        start_time = time.time()
        self.content = content
        self.updated_at = datetime.utcnow()
        self.mark_stale()
        process_time = time.time() - start_time
        cell_logger.info(
            "Cell content updated",
            extra={
                'cell_id': str(self.id),
                'cell_type': str(self.type),
                'content_length': len(content),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
    
    def set_result(self, result: Any, error: Optional[str] = None, execution_time: float = 0.0) -> None:
        """
        Set the execution result for this cell
        
        Args:
            result: The result data
            error: Optional error message
            execution_time: Time taken to execute the cell
        """
        start_time = time.time()
        self.result = CellResult(
            content=result,
            error=error,
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )
        self.status = CellStatus.ERROR if error else CellStatus.SUCCESS
        self.updated_at = datetime.utcnow()
        
        process_time = time.time() - start_time
        cell_logger.info(
            "Cell result set",
            extra={
                'cell_id': str(self.id),
                'cell_type': str(self.type),
                'status': str(self.status),
                'execution_time': execution_time,
                'has_error': error is not None,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        if error:
            cell_logger.error(
                "Cell execution error",
                extra={
                    'cell_id': str(self.id),
                    'cell_type': str(self.type),
                    'error': error
                }
            )
    
    def add_dependency(self, cell_id: UUID) -> None:
        """
        Add a dependency on another cell
        
        Args:
            cell_id: ID of the cell this cell depends on
        """
        start_time = time.time()
        self.dependencies.add(cell_id)
        process_time = time.time() - start_time
        cell_logger.info(
            "Added cell dependency",
            extra={
                'cell_id': str(self.id),
                'cell_type': str(self.type),
                'dependency_id': str(cell_id),
                'total_dependencies': len(self.dependencies),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
    
    def add_dependent(self, cell_id: UUID) -> None:
        """
        Add a cell that depends on this cell
        
        Args:
            cell_id: ID of the dependent cell
        """
        start_time = time.time()
        self.dependents.add(cell_id)
        process_time = time.time() - start_time
        cell_logger.info(
            "Added cell dependent",
            extra={
                'cell_id': str(self.id),
                'cell_type': str(self.type),
                'dependent_id': str(cell_id),
                'total_dependents': len(self.dependents),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
    
    def remove_dependency(self, cell_id: UUID) -> None:
        """
        Remove a dependency on another cell
        
        Args:
            cell_id: ID of the dependency to remove
        """
        start_time = time.time()
        if cell_id in self.dependencies:
            self.dependencies.remove(cell_id)
            process_time = time.time() - start_time
            cell_logger.info(
                "Removed cell dependency",
                extra={
                    'cell_id': str(self.id),
                    'cell_type': str(self.type),
                    'dependency_id': str(cell_id),
                    'total_dependencies': len(self.dependencies),
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
        else:
            process_time = time.time() - start_time
            cell_logger.warning(
                "Attempted to remove non-existent dependency",
                extra={
                    'cell_id': str(self.id),
                    'cell_type': str(self.type),
                    'dependency_id': str(cell_id),
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
    
    def remove_dependent(self, cell_id: UUID) -> None:
        """
        Remove a cell that depends on this cell
        
        Args:
            cell_id: ID of the dependent to remove
        """
        start_time = time.time()
        if cell_id in self.dependents:
            self.dependents.remove(cell_id)
            process_time = time.time() - start_time
            cell_logger.info(
                "Removed cell dependent",
                extra={
                    'cell_id': str(self.id),
                    'cell_type': str(self.type),
                    'dependent_id': str(cell_id),
                    'total_dependents': len(self.dependents),
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
        else:
            process_time = time.time() - start_time
            cell_logger.warning(
                "Attempted to remove non-existent dependent",
                extra={
                    'cell_id': str(self.id),
                    'cell_type': str(self.type),
                    'dependent_id': str(cell_id),
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
    
    def clear_dependencies(self) -> None:
        """Clear all dependencies"""
        start_time = time.time()
        num_dependencies = len(self.dependencies)
        self.dependencies.clear()
        process_time = time.time() - start_time
        cell_logger.info(
            "Cleared all dependencies",
            extra={
                'cell_id': str(self.id),
                'cell_type': str(self.type),
                'dependencies_cleared': num_dependencies,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )


class MarkdownCell(Cell):
    """
    A cell containing markdown content
    
    This cell type is used for documentation and explanations.
    
    Attributes:
        type: Always set to MARKDOWN
    """
    type: CellType = CellType.MARKDOWN
    
    def __init__(self, **data):
        super().__init__(**data)
        cell_logger.info(
            "Markdown cell initialized",
            extra={
                'cell_id': str(self.id),
                'content_length': len(self.content)
            }
        )
    
    class Config:
        schema_extra = {
            "example": {
                "content": "# Analysis Results\nThe query found **15 users** who registered in the last 24 hours."
            }
        }


class GitHubCell(Cell):
    """
    A cell containing a GitHub query/action intent.
    Stores the tool name and arguments needed for execution.
    """
    type: CellType = CellType.GITHUB
    source: str = "github"
    # Store the specific tool and its arguments for execution
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        cell_logger.info(
            "Github cell initialized",
            extra={
                'cell_id': str(self.id),
                'source': self.source
            }
        )


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
    
    def __init__(self, **data):
        super().__init__(**data)
        cell_logger.info(
            "Log query cell initialized",
            extra={
                'cell_id': str(self.id),
                'source': self.source,
                'has_time_range': self.time_range is not None
            }
        )
    
    def set_time_range(self, start: str, end: str) -> None:
        """Set the time range for the log query"""
        start_time = time.time()
        self.time_range = {"start": start, "end": end}
        process_time = time.time() - start_time
        cell_logger.info(
            "Log query time range set",
            extra={
                'cell_id': str(self.id),
                'start_time': start,
                'end_time': end,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )


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
    
    def __init__(self, **data):
        super().__init__(**data)
        cell_logger.info(
            "Metric query cell initialized",
            extra={
                'cell_id': str(self.id),
                'source': self.source,
                'has_time_range': self.time_range is not None
            }
        )
    
    def set_time_range(self, start: str, end: str) -> None:
        """Set the time range for the metric query"""
        start_time = time.time()
        self.time_range = {"start": start, "end": end}
        process_time = time.time() - start_time
        cell_logger.info(
            "Metric query time range set",
            extra={
                'cell_id': str(self.id),
                'start_time': start,
                'end_time': end,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )


class SummarizationCell(Cell):
    """
    A cell containing text to be summarized or the result of a summarization.
    """
    type: CellType = CellType.SUMMARIZATION
    source: str = "summarization"
    # Potentially add fields later if needed, e.g., original_text_ref

    def __init__(self, **data):
        super().__init__(**data)
        cell_logger.info(
            "Summarization cell initialized",
            extra={
                'cell_id': str(self.id),
                'source': self.source
            }
        )


def create_cell(cell_type: CellType, content: str, **kwargs) -> Cell:
    """
    Create a new cell of the specified type
    
    Args:
        cell_type: The type of cell to create
        content: Initial content for the cell
        **kwargs: Additional arguments to pass to the cell constructor
        
    Returns:
        A new cell instance of the appropriate type
    """
    start_time = time.time()
    
    try:
        # Map cell types to their classes
        cell_classes = {
            CellType.MARKDOWN: MarkdownCell,
            CellType.LOG: LogCell,
            CellType.METRIC: MetricCell,
            CellType.GITHUB: GitHubCell,
            CellType.SUMMARIZATION: SummarizationCell
        }
        
        # Create the cell
        if cell_type not in cell_classes:
            raise ValueError(f"Unsupported cell type: {cell_type}")
            
        cell_class = cell_classes[cell_type]
        
        # Prepare arguments, extracting GitHub specific ones if needed
        constructor_args = {"content": content, **kwargs}
        if cell_type == CellType.GITHUB:
            # Clean potential None values passed if agent didn't find tool calls
            if constructor_args.get('tool_name') is None:
                constructor_args.pop('tool_name', None)
            if constructor_args.get('tool_arguments') is None:
                constructor_args.pop('tool_arguments', None)
                
        cell = cell_class(**constructor_args)
        
        process_time = time.time() - start_time
        cell_logger.info(
            "Cell created",
            extra={
                'cell_id': str(cell.id),
                'cell_type': str(cell_type),
                'content_length': len(content),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return cell
    except Exception as e:
        process_time = time.time() - start_time
        cell_logger.error(
            "Error creating cell",
            extra={
                'cell_type': str(cell_type),
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise