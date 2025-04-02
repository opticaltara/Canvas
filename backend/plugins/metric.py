import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import numpy as np

from backend.core.cell import Cell, MetricCell
from backend.core.execution import ExecutionContext
from backend.plugins.base import ConnectionConfig, PluginBase, PluginConfig


class PrometheusPlugin(PluginBase):
    """Plugin for Prometheus metric queries"""
    
    def __init__(self):
        super().__init__(
            config=PluginConfig(
                name="prometheus",
                description="Prometheus metric query connector",
                version="0.1.0",
                config={
                    "default_timeout": 30,  # Default query timeout in seconds
                    "default_time_range": {"hours": 1},  # Default to last hour
                    "default_step": "15s",  # Default step for range queries
                }
            )
        )
    
    async def validate_connection(self, connection_config: ConnectionConfig) -> bool:
        """Validate a Prometheus connection configuration"""
        try:
            url = connection_config.config.get("url")
            
            # Query the Prometheus API status endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/-/ready") as response:
                    return response.status == 200
        
        except Exception as e:
            print(f"Error validating Prometheus connection: {e}")
            return False
    
    async def execute_query(
        self,
        query: str,
        connection_config: ConnectionConfig,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Prometheus query"""
        try:
            url = connection_config.config.get("url")
            timeout = connection_config.config.get("timeout", self.get_config("default_timeout"))
            
            # Get time range from parameters or use defaults
            time_range = None
            if parameters and "time_range" in parameters:
                time_range = parameters["time_range"]
            
            if not time_range:
                # Use default time range (last hour)
                default_range = self.get_config("default_time_range")
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(**default_range)
                time_range = {
                    "start": start_time.isoformat() + "Z",
                    "end": end_time.isoformat() + "Z"
                }
            
            # Determine if this is a range query or instant query
            is_instant = parameters.get("instant", False) if parameters else False
            
            if is_instant:
                # Instant query
                query_url = f"{url}/api/v1/query"
                query_params = {
                    "query": query,
                    "time": time_range["end"],
                }
            else:
                # Range query
                query_url = f"{url}/api/v1/query_range"
                step = parameters.get("step", self.get_config("default_step")) if parameters else self.get_config("default_step")
                query_params = {
                    "query": query,
                    "start": time_range["start"],
                    "end": time_range["end"],
                    "step": step,
                }
            
            # Execute the query
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    query_url,
                    params=query_params,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Prometheus query failed: {error_text}")
                    
                    result_json = await response.json()
                    
                    if result_json.get("status") != "success":
                        error_message = result_json.get("error", "Unknown error")
                        raise ValueError(f"Prometheus query failed: {error_message}")
                    
                    # Process the result
                    if is_instant:
                        return self._process_instant_result(result_json, query)
                    else:
                        return self._process_range_result(result_json, query)
        
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
    
    def _process_instant_result(self, result_json: Dict, query: str) -> Dict[str, Any]:
        """Process an instant query result from Prometheus"""
        result_type = result_json.get("data", {}).get("resultType", "")
        result_data = result_json.get("data", {}).get("result", [])
        
        processed_data = []
        
        if result_type == "vector":
            # Vector result (instant query)
            for series in result_data:
                metric = series.get("metric", {})
                value = series.get("value", [None, None])
                
                if len(value) >= 2:
                    timestamp, value_str = value
                    try:
                        value_float = float(value_str)
                    except (ValueError, TypeError):
                        value_float = None
                    
                    data_point = {
                        "timestamp": timestamp,
                        "value": value_float,
                        **metric
                    }
                    processed_data.append(data_point)
        
        # Create a pandas DataFrame
        df = pd.DataFrame(processed_data)
        
        return {
            "data": processed_data,
            "query": query,
            "result_type": result_type,
            "dataframe": df
        }
    
    def _process_range_result(self, result_json: Dict, query: str) -> Dict[str, Any]:
        """Process a range query result from Prometheus"""
        result_type = result_json.get("data", {}).get("resultType", "")
        result_data = result_json.get("data", {}).get("result", [])
        
        processed_data = []
        
        if result_type == "matrix":
            # Matrix result (range query)
            for series in result_data:
                metric = series.get("metric", {})
                values = series.get("values", [])
                
                # Create a unique identifier for this time series
                metric_name = "_".join([f"{k}={v}" for k, v in metric.items()])
                
                for value_pair in values:
                    if len(value_pair) >= 2:
                        timestamp, value_str = value_pair
                        try:
                            value_float = float(value_str)
                        except (ValueError, TypeError):
                            value_float = None
                        
                        data_point = {
                            "timestamp": timestamp,
                            "value": value_float,
                            "metric_name": metric_name,
                            **metric
                        }
                        processed_data.append(data_point)
        
        # Create a pandas DataFrame
        df = pd.DataFrame(processed_data)
        
        # If we have data and it has timestamp and value columns, create a pivot table
        pivot_df = None
        if not df.empty and "timestamp" in df.columns and "value" in df.columns and "metric_name" in df.columns:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Create a pivot table with timestamps as index and metrics as columns
            pivot_df = df.pivot_table(
                index="timestamp",
                columns="metric_name",
                values="value",
                aggfunc="first"
            )
        
        return {
            "data": processed_data,
            "query": query,
            "result_type": result_type,
            "dataframe": df,
            "pivot_dataframe": pivot_df
        }
    
    async def get_schema(self, connection_config: ConnectionConfig) -> Dict[str, Any]:
        """Get information about available metrics"""
        try:
            url = connection_config.config.get("url")
            
            async with aiohttp.ClientSession() as session:
                # Get all metric names
                async with session.get(f"{url}/api/v1/label/__name__/values") as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Failed to get Prometheus metrics: {error_text}")
                    
                    metrics_json = await response.json()
                    metrics = metrics_json.get("data", [])
                    
                    return {
                        "metrics": metrics
                    }
        
        except Exception as e:
            return {"error": str(e)}


class GrafanaPrometheusPlugin(PluginBase):
    """Plugin for Prometheus queries via Grafana API"""
    
    def __init__(self):
        super().__init__(
            config=PluginConfig(
                name="grafana_prometheus",
                description="Grafana Prometheus connector",
                version="0.1.0",
                config={
                    "default_timeout": 30,
                    "default_time_range": {"hours": 1},
                    "default_step": "15s",
                }
            )
        )
    
    async def validate_connection(self, connection_config: ConnectionConfig) -> bool:
        """Validate a Grafana connection for Prometheus"""
        try:
            url = connection_config.config.get("url")
            api_key = connection_config.config.get("api_key")
            
            # Check Grafana health
            headers = {"Authorization": f"Bearer {api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/api/health", headers=headers) as response:
                    return response.status == 200
        
        except Exception as e:
            print(f"Error validating Grafana Prometheus connection: {e}")
            return False
    
    async def execute_query(
        self,
        query: str,
        connection_config: ConnectionConfig,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Prometheus query via Grafana API"""
        try:
            url = connection_config.config.get("url")
            api_key = connection_config.config.get("api_key")
            datasource_uid = connection_config.config.get("prometheus_datasource_uid")
            timeout = connection_config.config.get("timeout", self.get_config("default_timeout"))
            
            # Get time range from parameters or use defaults
            time_range = None
            if parameters and "time_range" in parameters:
                time_range = parameters["time_range"]
            
            if not time_range:
                # Use default time range (last hour)
                default_range = self.get_config("default_time_range")
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(**default_range)
                time_range = {
                    "start": start_time.isoformat() + "Z",
                    "end": end_time.isoformat() + "Z"
                }
            
            # Determine if this is a range query or instant query
            is_instant = parameters.get("instant", False) if parameters else False
            
            # Build query payload
            payload = {
                "queries": [
                    {
                        "refId": "A",
                        "datasourceId": datasource_uid,
                        "expr": query,
                        "instant": is_instant,
                        "range": not is_instant,
                        "intervalMs": 15000,  # 15s default interval
                    }
                ],
                "range": {
                    "from": time_range["start"],
                    "to": time_range["end"],
                }
            }
            
            # Execute the query
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            query_url = f"{url}/api/ds/query"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    query_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Grafana Prometheus query failed: {error_text}")
                    
                    result_json = await response.json()
                    
                    # Process the result
                    if "results" in result_json and "A" in result_json["results"]:
                        prom_data = result_json["results"]["A"]
                        return self._process_grafana_result(prom_data, query, is_instant)
                    else:
                        return {"data": [], "query": query}
        
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
    
    def _process_grafana_result(self, result_json: Dict, query: str, is_instant: bool) -> Dict[str, Any]:
        """Process a Grafana Prometheus query result"""
        processed_data = []
        
        if "frames" in result_json:
            for frame in result_json["frames"]:
                if "data" not in frame or "fields" not in frame["data"]:
                    continue
                
                fields = frame["data"]["fields"]
                if not fields or len(fields) < 2:
                    continue
                
                # Extract labels if available
                labels = {}
                for field in fields:
                    if "labels" in field:
                        labels.update(field["labels"] or {})
                
                # Find time and value fields
                time_field = None
                value_fields = []
                
                for field in fields:
                    if field.get("type") == "time":
                        time_field = field
                    elif field.get("type") in ["number", "float", "integer"]:
                        value_fields.append(field)
                
                if not time_field or not value_fields:
                    continue
                
                # Process the time series data
                times = time_field.get("values", [])
                
                for value_field in value_fields:
                    values = value_field.get("values", [])
                    field_name = value_field.get("name", "value")
                    
                    for i in range(min(len(times), len(values))):
                        data_point = {
                            "timestamp": times[i],
                            "value": values[i],
                            "metric_name": field_name,
                            **labels
                        }
                        processed_data.append(data_point)
        
        # Create a pandas DataFrame
        df = pd.DataFrame(processed_data)
        
        # If we have data and it has timestamp and value columns, create a pivot table
        pivot_df = None
        if not df.empty and "timestamp" in df.columns and "value" in df.columns and "metric_name" in df.columns:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Create a pivot table with timestamps as index and metrics as columns
            pivot_df = df.pivot_table(
                index="timestamp",
                columns="metric_name",
                values="value",
                aggfunc="first"
            )
        
        return {
            "data": processed_data,
            "query": query,
            "dataframe": df,
            "pivot_dataframe": pivot_df
        }
    
    async def get_schema(self, connection_config: ConnectionConfig) -> Dict[str, Any]:
        """Get available metrics via Grafana API"""
        try:
            url = connection_config.config.get("url")
            api_key = connection_config.config.get("api_key")
            datasource_uid = connection_config.config.get("prometheus_datasource_uid")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Query Grafana for metric names (using label_values function)
            payload = {
                "queries": [
                    {
                        "refId": "metrics",
                        "datasourceId": datasource_uid,
                        "expr": "label_values(__name__)",
                        "instant": True,
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/api/ds/query",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Failed to get Grafana Prometheus metrics: {error_text}")
                    
                    result_json = await response.json()
                    metrics = []
                    
                    if "results" in result_json and "metrics" in result_json["results"]:
                        metrics_data = result_json["results"]["metrics"]
                        if "frames" in metrics_data and metrics_data["frames"]:
                            for frame in metrics_data["frames"]:
                                if "data" in frame and "values" in frame["data"]:
                                    values = frame["data"]["values"]
                                    if values and len(values) > 0:
                                        metrics.extend(values[0])
                    
                    return {
                        "metrics": metrics
                    }
        
        except Exception as e:
            return {"error": str(e)}


async def execute_metric_cell(cell: MetricCell, context: ExecutionContext) -> Dict[str, Any]:
    """
    Execute a metric query cell
    
    Args:
        cell: The metric cell to execute
        context: The execution context
        
    Returns:
        The results of the metric query
    """
    # This would normally be injected or retrieved from a registry
    from backend.services.connection_manager import get_connection_manager
    connection_manager = get_connection_manager()
    
    # Get the source and create parameters
    source = cell.source
    parameters = {}
    
    # Add time range if specified in the cell
    if cell.time_range:
        parameters["time_range"] = cell.time_range
    
    # Add any variables from context
    if context.variables:
        parameters.update(context.variables)
    
    # Use the appropriate plugin based on the source
    if source == "prometheus":
        plugin_name = "prometheus"
    elif source == "grafana":
        plugin_name = "grafana_prometheus"
    else:
        return {
            "error": f"Unsupported metric source: {source}",
            "query": cell.content,
        }
    
    # Get the default connection for this plugin
    connection_config = connection_manager.get_default_connection(plugin_name)
    if not connection_config:
        return {
            "error": f"No default connection found for {plugin_name}",
            "query": cell.content,
        }
    
    # Get the plugin
    plugin = connection_manager.get_plugin(plugin_name)
    if not plugin:
        return {
            "error": f"Plugin not found: {plugin_name}",
            "query": cell.content,
        }
    
    # Execute the query
    result = await plugin.execute_query(cell.content, connection_config, parameters)
    
    return result