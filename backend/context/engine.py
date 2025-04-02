"""
Context Engine Implementation

This module provides the core functionality for:
1. Generating embeddings for data source metadata
2. Storing and retrieving embeddings from Qdrant
3. Building context from retrieved information
4. Integrating with AI agents via RAG
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from pydantic import BaseModel, Field

from backend.config import get_settings


class ContextMetadata(BaseModel):
    """Metadata for a context item in the vector store"""
    source_id: str = Field(description="ID of the data source")
    source_type: str = Field(description="Type of data source (sql, prometheus, loki, etc.)")
    source_name: str = Field(description="Name of the data source")
    content_type: str = Field(description="Type of content (schema, log_format, metric, etc.)")
    item_name: Optional[str] = Field(description="Name of the specific item", default=None)
    item_path: Optional[str] = Field(description="Path or location of the item", default=None)
    timestamp: str = Field(description="When this context was indexed")


class EmbeddingService:
    """Service for generating embeddings using Claude API"""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.anthropic_api_key
        self.dimensions = 1536  # Claude embedding dimensions
        self.model = "claude-3-7-sonnet-20250219-embedding"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a text string
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding
        """
        url = "https://api.anthropic.com/v1/embeddings"
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "input": text,
            "dimensions": self.dimensions
        }
        
        response = await self.client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        embedding = result.get("embedding", [])
        
        if not embedding:
            raise ValueError("Empty embedding returned from API")
        
        return embedding


class ContextEngine:
    """
    Main context engine for managing vector embeddings and retrieval
    for data source metadata and schema information
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            port=self.settings.qdrant_port,
            api_key=self.settings.qdrant_api_key,
        )
        
        # Collection names for different data source types
        self.collections = {
            "sql": "sql_context",
            "prometheus": "prometheus_context",
            "loki": "loki_context",
            "grafana": "grafana_context",
            "s3": "s3_context",
        }
        
        # Ensure collections exist
        self._ensure_collections()
    
    async def close(self):
        """Close connections"""
        await self.embedding_service.close()
    
    def _ensure_collections(self):
        """Ensure all required collections exist in Qdrant"""
        for collection_name in self.collections.values():
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_service.dimensions,
                        distance=models.Distance.COSINE,
                    )
                )
    
    async def index_sql_schema(self, connection_id: str, connection_name: str, schema_data: Dict) -> None:
        """
        Index a SQL database schema
        
        Args:
            connection_id: ID of the connection
            connection_name: Name of the connection
            schema_data: Schema information for the database
        """
        collection_name = self.collections["sql"]
        timestamp = datetime.utcnow().isoformat()
        
        # Get tables from schema
        tables = schema_data.get("tables", {})
        
        # Index each table separately
        for table_name, table_info in tables.items():
            await self._index_table_schema(
                collection_name, connection_id, connection_name, table_name, table_info, timestamp
            )
        
        # Index the overall database schema (summary)
        await self._index_database_schema(
            collection_name, connection_id, connection_name, schema_data, timestamp
        )
    
    async def _index_table_schema(
        self, collection_name: str, connection_id: str, connection_name: str, 
        table_name: str, table_info: Dict, timestamp: str
    ) -> None:
        """
        Index a single table schema
        
        Args:
            collection_name: Qdrant collection name
            connection_id: ID of the connection
            connection_name: Name of the connection
            table_name: Name of the table
            table_info: Table schema information
            timestamp: Timestamp string
        """
        # Create detailed text representation of table schema
        columns = table_info.get("columns", [])
        
        # Format text description
        text = f"Table: {table_name}\n"
        text += "Columns:\n"
        
        for column in columns:
            col_name = column.get("name", "")
            col_type = column.get("type", "")
            nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
            pk = "PRIMARY KEY" if column.get("primary_key", False) else ""
            
            text += f"  - {col_name}: {col_type} {nullable} {pk}\n"
        
        # Additional metadata
        text += "\n"
        if "primary_key" in table_info:
            text += f"Primary Key: {table_info['primary_key']}\n"
        
        # Create metadata
        metadata = ContextMetadata(
            source_id=connection_id,
            source_type="sql",
            source_name=connection_name,
            content_type="table_schema",
            item_name=table_name,
            timestamp=timestamp
        )
        
        # Generate embedding and store
        embedding = await self.embedding_service.generate_embedding(text)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=f"sql_table_{connection_id}_{table_name}",
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata.dict(),
                        "table_name": table_name,
                        "connection_id": connection_id,
                    }
                )
            ]
        )
    
    async def _index_database_schema(
        self, collection_name: str, connection_id: str, connection_name: str, 
        schema_data: Dict, timestamp: str
    ) -> None:
        """
        Index the overall database schema summary
        
        Args:
            collection_name: Qdrant collection name
            connection_id: ID of the connection
            connection_name: Name of the connection
            schema_data: Complete schema information
            timestamp: Timestamp string
        """
        # Create a summary of the entire database
        tables = schema_data.get("tables", {})
        dialect = schema_data.get("dialect", "unknown")
        
        text = f"Database: {connection_name}\n"
        text += f"Dialect: {dialect}\n\n"
        text += f"Tables ({len(tables)}):\n"
        
        for table_name, table_info in tables.items():
            primary_key = table_info.get("primary_key", "None")
            col_count = len(table_info.get("columns", []))
            text += f"  - {table_name} ({col_count} columns, PK: {primary_key})\n"
        
        # Create metadata
        metadata = ContextMetadata(
            source_id=connection_id,
            source_type="sql",
            source_name=connection_name,
            content_type="database_schema",
            timestamp=timestamp
        )
        
        # Generate embedding and store
        embedding = await self.embedding_service.generate_embedding(text)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=f"sql_db_{connection_id}",
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata.dict(),
                        "connection_id": connection_id,
                    }
                )
            ]
        )
    
    async def index_prometheus_metrics(
        self, connection_id: str, connection_name: str, metrics_data: Dict
    ) -> None:
        """
        Index Prometheus metrics information
        
        Args:
            connection_id: ID of the connection
            connection_name: Name of the connection
            metrics_data: Metrics information from Prometheus
        """
        collection_name = self.collections["prometheus"]
        timestamp = datetime.utcnow().isoformat()
        
        # Get metrics list
        metrics = metrics_data.get("metrics", [])
        
        # Create a summary of all available metrics
        text = f"Prometheus Metrics for {connection_name}\n\n"
        text += f"Available Metrics ({len(metrics)}):\n"
        
        for metric in metrics:
            text += f"  - {metric}\n"
        
        # Create metadata
        metadata = ContextMetadata(
            source_id=connection_id,
            source_type="prometheus",
            source_name=connection_name,
            content_type="metrics_list",
            timestamp=timestamp
        )
        
        # Generate embedding and store
        embedding = await self.embedding_service.generate_embedding(text)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=f"prometheus_{connection_id}",
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata.dict(),
                        "connection_id": connection_id,
                    }
                )
            ]
        )
    
    async def index_loki_logs(
        self, connection_id: str, connection_name: str, labels_data: Dict
    ) -> None:
        """
        Index Loki log labels and information
        
        Args:
            connection_id: ID of the connection
            connection_name: Name of the connection
            labels_data: Labels information from Loki
        """
        collection_name = self.collections["loki"]
        timestamp = datetime.utcnow().isoformat()
        
        # Get labels and their values
        labels = labels_data.get("labels", [])
        label_values = labels_data.get("label_values", {})
        
        # Create text representation
        text = f"Loki Logs for {connection_name}\n\n"
        text += f"Available Labels ({len(labels)}):\n"
        
        for label in labels:
            values = label_values.get(label, [])
            value_count = len(values)
            value_examples = ", ".join(values[:5]) + ("..." if value_count > 5 else "")
            
            text += f"  - {label}: {value_count} unique values\n"
            if value_examples:
                text += f"    Example values: {value_examples}\n"
        
        # Create metadata
        metadata = ContextMetadata(
            source_id=connection_id,
            source_type="loki",
            source_name=connection_name,
            content_type="log_labels",
            timestamp=timestamp
        )
        
        # Generate embedding and store
        embedding = await self.embedding_service.generate_embedding(text)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=f"loki_{connection_id}",
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata.dict(),
                        "connection_id": connection_id,
                    }
                )
            ]
        )
    
    async def index_s3_buckets(
        self, connection_id: str, connection_name: str, buckets_data: Dict
    ) -> None:
        """
        Index S3 bucket information
        
        Args:
            connection_id: ID of the connection
            connection_name: Name of the connection
            buckets_data: Bucket information from S3
        """
        collection_name = self.collections["s3"]
        timestamp = datetime.utcnow().isoformat()
        
        # Get buckets
        buckets = buckets_data.get("buckets", [])
        
        # Create text representation
        text = f"S3 Storage for {connection_name}\n\n"
        text += f"Available Buckets ({len(buckets)}):\n"
        
        for bucket in buckets:
            bucket_name = bucket.get("name", "")
            creation_date = bucket.get("creation_date", "")
            
            text += f"  - {bucket_name} (Created: {creation_date})\n"
        
        # Create metadata
        metadata = ContextMetadata(
            source_id=connection_id,
            source_type="s3",
            source_name=connection_name,
            content_type="bucket_list",
            timestamp=timestamp
        )
        
        # Generate embedding and store
        embedding = await self.embedding_service.generate_embedding(text)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=f"s3_{connection_id}",
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata.dict(),
                        "connection_id": connection_id,
                    }
                )
            ]
        )
    
    async def retrieve_context(self, query: str, source_type: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: The query to find context for
            source_type: Optional filter for specific source type
            limit: Maximum number of results to return
            
        Returns:
            List of context items with relevance scores
        """
        # Generate embedding for the query
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Determine which collections to search
        if source_type and source_type in self.collections:
            collections = [self.collections[source_type]]
        else:
            collections = list(self.collections.values())
        
        # Search across all relevant collections
        all_results = []
        
        for collection_name in collections:
            # Skip if collection doesn't exist yet
            if not self.client.collection_exists(collection_name):
                continue
            
            # Search the collection
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Process results
            for result in search_result:
                all_results.append({
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {})
                })
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return all_results[:limit]
    
    def build_context_for_ai(self, context_items: List[Dict]) -> str:
        """
        Build a formatted context string for the AI agent
        
        Args:
            context_items: List of context items from retrieve_context
            
        Returns:
            Formatted context string
        """
        if not context_items:
            return ""
        
        context = "AVAILABLE DATA SOURCE CONTEXT:\n\n"
        
        for idx, item in enumerate(context_items, 1):
            metadata = item.get("metadata", {})
            source_type = metadata.get("source_type", "unknown")
            source_name = metadata.get("source_name", "unknown")
            content_type = metadata.get("content_type", "unknown")
            
            context += f"[Context {idx}: {source_type.upper()} - {source_name} - {content_type}]\n"
            context += item.get("text", "No content available")
            context += "\n\n"
        
        return context


# Singleton instance
_context_engine_instance = None


def get_context_engine() -> ContextEngine:
    """Get or create the singleton ContextEngine instance"""
    global _context_engine_instance
    if _context_engine_instance is None:
        _context_engine_instance = ContextEngine()
    return _context_engine_instance