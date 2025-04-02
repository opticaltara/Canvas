import asyncio
import io
import json
from typing import Any, Dict, List, Optional

import aioboto3
import pandas as pd

from backend.core.cell import Cell, S3Cell
from backend.core.execution import ExecutionContext
from backend.plugins.base import ConnectionConfig, PluginBase, PluginConfig


class S3Plugin(PluginBase):
    """Plugin for Amazon S3 queries"""
    
    def __init__(self):
        super().__init__(
            config=PluginConfig(
                name="s3",
                description="Amazon S3 connector",
                version="0.1.0",
                config={
                    "default_timeout": 60,  # Default query timeout in seconds
                }
            )
        )
        self.session = None
    
    async def validate_connection(self, connection_config: ConnectionConfig) -> bool:
        """Validate an S3 connection configuration"""
        try:
            # Create a session
            session = aioboto3.Session(
                aws_access_key_id=connection_config.config.get("aws_access_key_id"),
                aws_secret_access_key=connection_config.config.get("aws_secret_access_key"),
                region_name=connection_config.config.get("region_name", "us-east-1"),
            )
            
            # Try to list buckets
            async with session.client("s3") as s3:
                await s3.list_buckets()
                return True
        
        except Exception as e:
            print(f"Error validating S3 connection: {e}")
            return False
    
    async def execute_query(
        self,
        query: str,
        connection_config: ConnectionConfig,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute an S3 query (list objects or read object)"""
        try:
            operation = parameters.get("operation", "list_objects") if parameters else "list_objects"
            bucket = parameters.get("bucket") if parameters else connection_config.config.get("default_bucket")
            prefix = parameters.get("prefix", "") if parameters else ""
            
            if not bucket:
                raise ValueError("No bucket specified")
            
            # Create a session
            session = aioboto3.Session(
                aws_access_key_id=connection_config.config.get("aws_access_key_id"),
                aws_secret_access_key=connection_config.config.get("aws_secret_access_key"),
                region_name=connection_config.config.get("region_name", "us-east-1"),
            )
            
            # Execute the operation
            if operation == "list_objects":
                result = await self._list_objects(session, bucket, prefix)
            elif operation == "get_object":
                result = await self._get_object(session, bucket, query, parameters)
            elif operation == "select_object":
                result = await self._select_object(session, bucket, query, parameters)
            else:
                raise ValueError(f"Unsupported S3 operation: {operation}")
            
            return result
        
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
    
    async def _list_objects(self, session, bucket: str, prefix: str) -> Dict[str, Any]:
        """List objects in an S3 bucket"""
        objects = []
        
        async with session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects.append({
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                            "storage_class": obj["StorageClass"],
                        })
        
        return {
            "data": objects,
            "bucket": bucket,
            "prefix": prefix,
            "count": len(objects),
            "dataframe": pd.DataFrame(objects)
        }
    
    async def _get_object(self, session, bucket: str, key: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get an object from S3"""
        async with session.client("s3") as s3:
            response = await s3.get_object(Bucket=bucket, Key=key)
            
            # Read the object content
            content_bytes = await response["Body"].read()
            
            # Try to determine the content type
            content_type = response.get("ContentType", "")
            
            # Process the content based on its type
            if "json" in content_type:
                # JSON file
                try:
                    content = json.loads(content_bytes.decode("utf-8"))
                    df = pd.DataFrame(content) if isinstance(content, list) else pd.DataFrame([content])
                except:
                    content = content_bytes.decode("utf-8")
                    df = pd.DataFrame([{"content": content}])
            
            elif "csv" in content_type:
                # CSV file
                try:
                    df = pd.read_csv(io.BytesIO(content_bytes))
                    content = df.to_dict(orient="records")
                except:
                    content = content_bytes.decode("utf-8")
                    df = pd.DataFrame([{"content": content}])
            
            else:
                # Default to text
                try:
                    content = content_bytes.decode("utf-8")
                    df = pd.DataFrame([{"content": content}])
                except:
                    content = "Binary content (unable to decode as text)"
                    df = pd.DataFrame([{"content": content}])
            
            return {
                "data": content,
                "bucket": bucket,
                "key": key,
                "metadata": dict(response.get("Metadata", {})),
                "content_type": content_type,
                "dataframe": df
            }
    
    async def _select_object(self, session, bucket: str, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute an S3 Select query"""
        if not parameters:
            parameters = {}
        
        key = parameters.get("key")
        if not key:
            raise ValueError("No object key specified for S3 Select query")
        
        input_format = parameters.get("input_format", "CSV")
        output_format = parameters.get("output_format", "JSON")
        
        # Configure input serialization based on the format
        input_serialization = {}
        
        if input_format == "CSV":
            input_serialization = {
                "CompressionType": "NONE",
                "CSV": {
                    "FileHeaderInfo": parameters.get("file_header_info", "USE"),
                    "RecordDelimiter": parameters.get("record_delimiter", "\n"),
                    "FieldDelimiter": parameters.get("field_delimiter", ","),
                }
            }
        elif input_format == "JSON":
            input_serialization = {
                "CompressionType": "NONE",
                "JSON": {
                    "Type": parameters.get("json_type", "DOCUMENT"),
                }
            }
        elif input_format == "Parquet":
            input_serialization = {
                "CompressionType": "NONE",
                "Parquet": {}
            }
        else:
            raise ValueError(f"Unsupported input format for S3 Select: {input_format}")
        
        # Configure output serialization
        output_serialization = {}
        
        if output_format == "CSV":
            output_serialization["CSV"] = {
                "RecordDelimiter": parameters.get("record_delimiter", "\n"),
                "FieldDelimiter": parameters.get("field_delimiter", ","),
            }
        elif output_format == "JSON":
            output_serialization["JSON"] = {
                "RecordDelimiter": parameters.get("record_delimiter", "\n"),
            }
        else:
            raise ValueError(f"Unsupported output format for S3 Select: {output_format}")
        
        async with session.client("s3") as s3:
            response = await s3.select_object_content(
                Bucket=bucket,
                Key=key,
                Expression=query,
                ExpressionType="SQL",
                InputSerialization=input_serialization,
                OutputSerialization=output_serialization,
            )
            
            # Process the response
            result = ""
            records = []
            
            async for event in response["Payload"]:
                if "Records" in event:
                    records_data = event["Records"]["Payload"].decode("utf-8")
                    result += records_data
            
            # Parse the output based on the format
            if output_format == "JSON":
                # Split by record delimiter and parse each JSON record
                for line in result.split("\n"):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            records.append({"raw": line})
            else:
                # For CSV, just return the raw result
                records = [{"result": result}]
            
            return {
                "data": records,
                "bucket": bucket,
                "key": key,
                "query": query,
                "dataframe": pd.DataFrame(records) if records else pd.DataFrame()
            }
    
    async def get_schema(self, connection_config: ConnectionConfig) -> Dict[str, Any]:
        """Get schema information (list of buckets)"""
        try:
            # Create a session
            session = aioboto3.Session(
                aws_access_key_id=connection_config.config.get("aws_access_key_id"),
                aws_secret_access_key=connection_config.config.get("aws_secret_access_key"),
                region_name=connection_config.config.get("region_name", "us-east-1"),
            )
            
            # List buckets
            buckets = []
            
            async with session.client("s3") as s3:
                response = await s3.list_buckets()
                
                if "Buckets" in response:
                    for bucket in response["Buckets"]:
                        buckets.append({
                            "name": bucket["Name"],
                            "creation_date": bucket["CreationDate"].isoformat(),
                        })
            
            return {
                "buckets": buckets
            }
        
        except Exception as e:
            return {"error": str(e)}


async def execute_s3_cell(cell: S3Cell, context: ExecutionContext) -> Dict[str, Any]:
    """
    Execute an S3 query cell
    
    Args:
        cell: The S3 cell to execute
        context: The execution context
        
    Returns:
        The results of the S3 query
    """
    # This would normally be injected or retrieved from a registry
    from backend.services.connection_manager import get_connection_manager
    connection_manager = get_connection_manager()
    
    # Create parameters
    parameters = {
        "bucket": cell.bucket,
        "prefix": cell.prefix,
    }
    
    # Determine if this is a listing query or a SELECT query
    if cell.content.strip().upper().startswith("SELECT"):
        parameters["operation"] = "select_object"
        
        # If content refers to specific key, extract it
        if "FROM s3object" in cell.content:
            # This is an S3 Select query, needs a key
            key = cell.metadata.get("key")
            if key:
                parameters["key"] = key
            else:
                return {
                    "error": "S3 Select query requires a key in metadata",
                    "query": cell.content,
                }
    else:
        # Assume it's a key to fetch or a listing prefix
        parameters["operation"] = "get_object" if cell.content else "list_objects"
        if cell.content:
            parameters["key"] = cell.content
    
    # Add any variables from context
    if context.variables:
        parameters.update(context.variables)
    
    # Get the default connection for S3
    connection_config = connection_manager.get_default_connection("s3")
    if not connection_config:
        return {
            "error": "No default S3 connection found",
            "query": cell.content,
        }
    
    # Get the plugin
    plugin = connection_manager.get_plugin("s3")
    if not plugin:
        return {
            "error": "S3 plugin not found",
            "query": cell.content,
        }
    
    # Execute the query
    result = await plugin.execute_query(cell.content, connection_config, parameters)
    
    return result