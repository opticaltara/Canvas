"""
Query Result Module

This module defines the base QueryResult class and its specialized subclasses for different types of queries.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class QueryResult(BaseModel):
    """
    Base class for query results
    
    Attributes:
        data: The primary result data
        query: The original query string
        error: Optional error message if query failed
        metadata: Additional metadata about the query result
    """
    data: Any = None
    query: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MarkdownQueryResult(QueryResult):
    """Result containing markdown content."""
    query: str = "markdown generation"
    data: str = "" # Default empty string


class ToolCallRecord(BaseModel):
    """Represents a single recorded tool call with its result."""
    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any # Result can be diverse


class GithubQueryResult(QueryResult):
    """
    Result from a GitHub agent tool execution.
    Stores the original tool call info and the result.
    """
    query: str = "github tool execution" # Placeholder query
    data: Any = None # Tool results can be varied
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None


class SummarizationQueryResult(QueryResult):
    """Result containing summarized text."""
    query: str = "summarization request" # Placeholder query
    data: str = "" # Default empty string