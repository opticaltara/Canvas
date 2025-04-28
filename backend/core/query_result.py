"""
Query Result Module

This module defines the base QueryResult class and its specialized subclasses for different types of queries.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class QueryResult(BaseModel):
    """
    Base class for all query results
    
    Attributes:
        data: The query result data
        query: The executed query
        error: Optional error message if query failed
        metadata: Additional metadata about the query result
    """
    data: Any
    query: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LogQueryResult(QueryResult):
    """
    Specialized query result for log queries
    
    Attributes:
        data: List of log entries
        query: The executed log query
        error: Optional error message if query failed
        metadata: Additional metadata about the log query result
    """
    data: List[Dict[str, Any]]


class MetricQueryResult(QueryResult):
    """
    Specialized query result for metric queries
    
    Attributes:
        data: List of metric data points
        query: The executed metric query
        error: Optional error message if query failed
        metadata: Additional metadata about the metric query result
    """
    data: List[Dict[str, Any]]


class MarkdownQueryResult(QueryResult):
    """
    Specialized query result for markdown content
    
    Attributes:
        data: The markdown content as a string
        query: The original query that generated the markdown
        error: Optional error message if generation failed
        metadata: Additional metadata about the markdown content
    """
    data: str


class ToolCallRecord(BaseModel):
    """Represents a single recorded tool call with its result."""
    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any # Result can be diverse


class GithubQueryResult(QueryResult):
    """
    Specialized query result for GitHub queries
    
    Attributes:
        data: Data returned from the GitHub query (can be diverse)
        query: The executed GitHub operation description
        error: Optional error message if query failed
        metadata: Additional metadata about the GitHub query result
        tool_calls: Sequence of successful tool calls (with results) made within the agent attempt
                  that produced this result. Each item is a ToolCallRecord.
    """
    data: Any  # GitHub queries can return various structures 
    tool_calls: Optional[List[ToolCallRecord]] = None 


class SummarizationQueryResult(QueryResult):
    """
    Specialized query result for text summarization.
    
    Attributes:
        data: The generated summary as a Markdown string.
        query: The original request or description leading to the summarization.
        error: Optional error message if summarization failed.
        metadata: Additional metadata about the summarization process (optional).
    """
    data: str # Markdown summary
    # Inherits query, error, metadata from QueryResult