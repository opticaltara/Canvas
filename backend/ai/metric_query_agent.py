import logging
import json
from typing import Any, Dict, List, Optional, AsyncGenerator, cast, Union

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from mcp import StdioServerParameters

from backend.ai import prometheus_system_prompt
from backend.config import get_settings
from backend.core.query_result import MetricQueryResult
from backend.services.connection_manager import get_connection_manager

metric_query_agent_logger = logging.getLogger("ai.metric_query_agent")

class MetricQueryAgent:
    """Agent for querying metrics via Grafana Prometheus MCP stdio."""
    def __init__(self, source: str, notebook_id: str):
        metric_query_agent_logger.info(f"Initializing MetricQueryAgent for source: {source}, notebook_id: {notebook_id}")
        self.settings = get_settings()
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.notebook_id = notebook_id
    
        if source == "prometheus":
            system_prompt = prometheus_system_prompt.system_prompt
            metric_query_agent_logger.info(f"Using Prometheus system prompt.")
        else:
            system_prompt = ""
            metric_query_agent_logger.warning(f"No specific system prompt for source: {source}")

        self.system_prompt_str = system_prompt
        metric_query_agent_logger.info(f"MetricQueryAgent initialized successfully.")
        
    async def _get_stdio_server(self) -> Optional[MCPServerStdio]:
        """Fetches default Grafana connection and creates the MCPServerStdio instance."""
        correlation_id = self.notebook_id
        try:
            connection_manager = get_connection_manager()
            connection = await connection_manager.get_default_connection("grafana")
            
            if not connection:
                metric_query_agent_logger.error("No default Grafana connection found.", extra={'correlation_id': correlation_id})
                return None
            
            config = connection.config
            command_list = config.get("mcp_command")
            args_list = config.get("mcp_args")
            env_dict = config.get("env")
            
            if not command_list or args_list is None or env_dict is None:
                metric_query_agent_logger.error(f"Incomplete stdio configuration in Grafana connection {connection.id}", extra={'correlation_id': correlation_id})
                return None
                
            command = command_list[0]
            args = command_list[1:] + args_list
            
            metric_query_agent_logger.info(f"Creating MCPServerStdio for connection {connection.id}", extra={'correlation_id': correlation_id})
            metric_query_agent_logger.info(f"Command: {command}, Args: {args}, Env keys: {list(env_dict.keys())}", extra={'correlation_id': correlation_id})
            
            return MCPServerStdio(command=command, args=args, env=env_dict)
            
        except Exception as e:
            metric_query_agent_logger.error(f"Error getting/configuring Grafana stdio server: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
            return None

    async def run_query(self, query: str, time_range: Optional[Dict[str, str]] = None) -> AsyncGenerator[Union[Dict[str, Any], MetricQueryResult], None]:
        """Run a metric query using the Grafana MCP stdio server, yielding status updates."""
        metric_query_agent_logger.info(f"Running metric query: '{query}', Time range: {time_range}")
        correlation_id = self.notebook_id
        yield {"status": "starting", "message": "Initializing Grafana connection...", "agent_type": "metric"}

        stdio_server = await self._get_stdio_server()
        
        if not stdio_server:
            error_msg = "Failed to configure Grafana MCP stdio server. Check default connection."
            metric_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id})
            yield {"status": "error", "message": error_msg, "agent_type": "metric"}
            yield MetricQueryResult(query=query, data=[], error=error_msg)
            return
        
        yield {"status": "connection_ready", "message": "Grafana connection established.", "agent_type": "metric"}

        system_prompt_value = self.system_prompt_str

        local_agent = Agent(
            self.model,
            result_type=MetricQueryResult,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt_value,
        )
        metric_query_agent_logger.info(f"Local Agent instance created with MCPServerStdio for Grafana.", extra={'correlation_id': correlation_id})
        yield {"status": "agent_created", "message": "Metric agent ready.", "agent_type": "metric"}

        query_with_context = query
        if time_range:
             query_with_context += f" (Time range: {time_range})"
        metric_query_agent_logger.info(f"Enhanced query for agent: '{query_with_context}'")

        try:
            async with local_agent.run_mcp_servers():
                metric_query_agent_logger.info(f"Calling local_agent.run...", extra={'correlation_id': correlation_id})
                yield {"status": "agent_running", "message": "Agent is processing the metric query...", "agent_type": "metric"}
                
                result = await local_agent.run(query_with_context)
                
                metric_query_agent_logger.info(f"Agent run completed. Raw result: {result}", extra={'correlation_id': correlation_id})
                yield {"status": "agent_run_complete", "agent_type": "metric"}
                
                if result and isinstance(result.data, MetricQueryResult):
                    metric_query_agent_logger.info(f"Successfully parsed result data.", extra={'correlation_id': correlation_id})
                    yield {"status": "parsing_success", "agent_type": "metric"}
                    result.data.query = query
                    yield result.data
                    return
                elif result and result.data:
                     metric_query_agent_logger.warning(f"Agent returned data but not MetricQueryResult type: {type(result.data)}", extra={'correlation_id': correlation_id})
                     error_msg = "Agent returned unexpected data type"
                     yield {"status": "parsing_error", "error": error_msg, "agent_type": "metric"}
                     yield MetricQueryResult(query=query, data=[], error=error_msg)
                     return
                else:
                    error_detail = f"Raw result: {result}" if result else "Result was None"
                    metric_query_agent_logger.error(f"Agent did not return valid data. {error_detail}", extra={'correlation_id': correlation_id})
                    error_msg = "Agent failed to return valid data"
                    yield {"status": "no_valid_data", "error": error_msg, "agent_type": "metric"}
                    yield MetricQueryResult(query=query, data=[], error=error_msg) 
                    return
        
        except FileNotFoundError as fnf_err:
            error_msg = f"MCP command not found: {fnf_err}. Ensure mcp-grafana is installed."
            metric_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id}, exc_info=True)
            yield {"status": "mcp_command_error", "error": error_msg, "agent_type": "metric"}
            yield MetricQueryResult(query=query, data=[], error=error_msg)
            return
        except ConnectionRefusedError as conn_err:
             error_msg = f"MCP connection refused: {conn_err}"
             metric_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id}, exc_info=True)
             yield {"status": "mcp_connection_error", "error": error_msg, "agent_type": "metric"}
             yield MetricQueryResult(query=query, data=[], error=error_msg)
             return
        except Exception as e:
            error_msg = f"Agent run failed: {str(e)}"
            metric_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id}, exc_info=True)
            yield {"status": "general_error", "error": error_msg, "agent_type": "metric"}
            yield MetricQueryResult(query=query, data=[], error=error_msg)
            return