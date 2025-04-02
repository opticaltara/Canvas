import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from backend.core.cell import Cell, LogCell
from backend.core.execution import ExecutionContext
from backend.plugins.base import ConnectionConfig, PluginBase, PluginConfig


class LokiPlugin(PluginBase):
    """Plugin for Loki log queries"""
    
    def __init__(self):
        super().__init__(
            config=PluginConfig(
                name="loki",
                description="Loki log query connector",
                version="0.1.0",
                config={
                    "default_timeout": 30,  # Default query timeout in seconds
                    "default_time_range": {"hours": 1},  # Default to last hour
                }
            )
        )
    
    async def validate_connection(self, connection_config: ConnectionConfig) -> bool:
        """Validate a Loki connection configuration"""
        try:
            # Simple status check
            url = f"{connection_config.config.get('url')}/ready"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Error validating Loki connection: {e}")
            return False
    
    async def execute_query(
        self,
        query: str,
        connection_config: ConnectionConfig,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Loki query"""
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
            is_instant = parameters is not None and "instant" in parameters and parameters["instant"]
            query_type = "query_range" if not is_instant else "query"
            
            # Build query URL and parameters
            query_url = f"{url}/loki/api/v1/{query_type}"
            query_params = {
                "query": query,
            }
            
            if not is_instant:
                query_params.update({
                    "start": time_range["start"],
                    "end": time_range["end"],
                    "step": parameters.get("step", "15s") if parameters else "15s"
                })
            else:
                query_params["time"] = time_range["end"]
            
            # Execute the query
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    query_url,
                    params=query_params,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Loki query failed: {error_text}")
                    
                    result_json = await response.json()
                    
                    # Process the result based on the query type
                    if query_type == "query_range":
                        return self._process_range_result(result_json, query)
                    else:
                        return self._process_instant_result(result_json, query)
        
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
    
    def _process_range_result(self, result_json: Dict, query: str) -> Dict[str, Any]:
        """Process a range query result from Loki"""
        if "data" not in result_json or "result" not in result_json["data"]:
            return {"data": [], "query": query}
        
        results = result_json["data"]["result"]
        processed_data = []
        
        for stream in results:
            # Extract stream labels
            labels = stream.get("stream", {})
            
            # Extract log entries
            for entry in stream.get("values", []):
                timestamp, log_line = entry
                processed_data.append({
                    "timestamp": timestamp,
                    "log": log_line,
                    **labels
                })
        
        return {
            "data": processed_data,
            "query": query,
            "dataframe": pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
        }
    
    def _process_instant_result(self, result_json: Dict, query: str) -> Dict[str, Any]:
        """Process an instant query result from Loki"""
        if "data" not in result_json or "result" not in result_json["data"]:
            return {"data": [], "query": query}
        
        results = result_json["data"]["result"]
        processed_data = []
        
        for stream in results:
            # Extract stream labels
            labels = stream.get("stream", {})
            
            # Extract log entry (should be just one for instant query)
            if "value" in stream:
                timestamp, log_line = stream["value"]
                processed_data.append({
                    "timestamp": timestamp,
                    "log": log_line,
                    **labels
                })
        
        return {
            "data": processed_data,
            "query": query,
            "dataframe": pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
        }
    
    async def get_schema(self, connection_config: ConnectionConfig) -> Dict[str, Any]:
        """Get the schema information (available labels and values)"""
        try:
            url = connection_config.config.get("url")
            labels_url = f"{url}/loki/api/v1/labels"
            
            async with aiohttp.ClientSession() as session:
                # Get all label names
                async with session.get(labels_url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Failed to get Loki labels: {error_text}")
                    
                    labels_json = await response.json()
                    labels = labels_json.get("data", [])
                    
                    # Get values for each label
                    label_values = {}
                    for label in labels:
                        label_values_url = f"{url}/loki/api/v1/label/{label}/values"
                        async with session.get(label_values_url) as values_response:
                            if values_response.status == 200:
                                values_json = await values_response.json()
                                label_values[label] = values_json.get("data", [])
                    
                    return {
                        "labels": labels,
                        "label_values": label_values
                    }
        
        except Exception as e:
            return {"error": str(e)}


class GrafanaLokiPlugin(PluginBase):
    """Plugin for Loki queries via Grafana API"""
    
    def __init__(self):
        super().__init__(
            config=PluginConfig(
                name="grafana_loki",
                description="Grafana Loki connector",
                version="0.1.0",
                config={
                    "default_timeout": 30,
                    "default_time_range": {"hours": 1},
                }
            )
        )
    
    async def validate_connection(self, connection_config: ConnectionConfig) -> bool:
        """Validate a Grafana connection for Loki"""
        try:
            url = connection_config.config.get("url")
            api_key = connection_config.config.get("api_key")
            
            # Check Grafana health
            headers = {"Authorization": f"Bearer {api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/api/health", headers=headers) as response:
                    return response.status == 200
        
        except Exception as e:
            print(f"Error validating Grafana Loki connection: {e}")
            return False
    
    async def execute_query(
        self,
        query: str,
        connection_config: ConnectionConfig,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Loki query via Grafana API"""
        try:
            url = connection_config.config.get("url")
            api_key = connection_config.config.get("api_key")
            datasource_uid = connection_config.config.get("loki_datasource_uid")
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
            
            # Build query payload
            payload = {
                "queries": [
                    {
                        "refId": "A",
                        "datasourceId": datasource_uid,
                        "expr": query,
                        "queryType": "range",
                        "instant": parameters.get("instant", False) if parameters else False,
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
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/api/datasources/proxy/{datasource_uid}/loki/api/v1/query_range",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Grafana Loki query failed: {error_text}")
                    
                    result_json = await response.json()
                    
                    # Process the result
                    if "results" in result_json and "A" in result_json["results"]:
                        loki_data = result_json["results"]["A"]
                        return self._process_grafana_result(loki_data, query)
                    else:
                        return {"data": [], "query": query}
        
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
    
    def _process_grafana_result(self, result_json: Dict, query: str) -> Dict[str, Any]:
        """Process a Grafana Loki query result"""
        processed_data = []
        
        if "frames" in result_json:
            for frame in result_json["frames"]:
                if "data" in frame and "values" in frame["data"]:
                    # Extract labels if available
                    labels = {}
                    if "fields" in frame["data"]:
                        for field in frame["data"]["fields"]:
                            if "labels" in field:
                                labels.update(field["labels"])
                    
                    # Extract log entries
                    timestamps = frame["data"]["values"][0]
                    log_lines = frame["data"]["values"][1]
                    
                    for i in range(len(timestamps)):
                        processed_data.append({
                            "timestamp": timestamps[i],
                            "log": log_lines[i],
                            **labels
                        })
        
        return {
            "data": processed_data,
            "query": query,
            "dataframe": pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
        }
    
    async def get_schema(self, connection_config: ConnectionConfig) -> Dict[str, Any]:
        """Get available labels via Grafana API"""
        try:
            url = connection_config.config.get("url")
            api_key = connection_config.config.get("api_key")
            datasource_uid = connection_config.config.get("loki_datasource_uid")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                # Get all label names
                labels_url = f"{url}/api/datasources/proxy/{datasource_uid}/loki/api/v1/labels"
                async with session.get(labels_url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Failed to get Grafana Loki labels: {error_text}")
                    
                    labels_json = await response.json()
                    labels = labels_json.get("data", [])
                    
                    return {
                        "labels": labels,
                        # Omit individual label values to avoid a large number of requests
                    }
        
        except Exception as e:
            return {"error": str(e)}


async def execute_log_cell(cell: LogCell, context: ExecutionContext) -> Dict[str, Any]:
    """
    Execute a log query cell
    
    Args:
        cell: The log cell to execute
        context: The execution context
        
    Returns:
        The results of the log query
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
    if source == "loki":
        plugin_name = "loki"
    elif source == "grafana":
        plugin_name = "grafana_loki"
    else:
        return {
            "error": f"Unsupported log source: {source}",
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