"""
Query Result Module

This module defines the base QueryResult class and its specialized subclasses for different types of queries.
"""

from typing import Any, Dict, List, Optional
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