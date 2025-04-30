import logging
from typing import Dict, Optional, AsyncGenerator, Union, Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

from backend.ai import loki_system_prompt
from backend.config import get_settings
from backend.core.query_result import LogQueryResult
from backend.services.connection_manager import get_connection_manager

log_query_agent_logger = logging.getLogger("ai.log_query_agent")

class LogQueryAgent:
    """Agent for querying logs via Grafana Loki MCP stdio."""
    def __init__(self, source: str, notebook_id: str):
        log_query_agent_logger.info(f"Initializing LogQueryAgent for source: {source}, notebook_id: {notebook_id}")
        self.settings = get_settings()
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.notebook_id = notebook_id
    
        if source == "loki":
            system_prompt = loki_system_prompt.system_prompt
            log_query_agent_logger.info(f"Using Loki system prompt - {system_prompt}.")
        else:
            system_prompt = ""
            log_query_agent_logger.warning(f"No specific system prompt for source: {source}")

        # Store the system prompt string for later use
        self.system_prompt_str = system_prompt
        log_query_agent_logger.info(f"LogQueryAgent initialized successfully.")

    async def _get_stdio_server(self) -> Optional[MCPServerStdio]:
        """Fetches default Grafana connection and creates the MCPServerStdio instance."""
        correlation_id = self.notebook_id
        try:
            connection_manager = get_connection_manager()
            connection = await connection_manager.get_default_connection("grafana")
            
            if not connection:
                log_query_agent_logger.error("No default Grafana connection found.", extra={'correlation_id': correlation_id})
                return None
            
            config = connection.config
            command_list = config.get("mcp_command")
            args_list = config.get("mcp_args")
            env_dict = config.get("env")
            
            if not command_list or args_list is None or env_dict is None:
                log_query_agent_logger.error(f"Incomplete stdio configuration in Grafana connection {connection.id}", extra={'correlation_id': correlation_id})
                return None
                
            command = command_list[0]
            args = command_list[1:] + args_list
            
            log_query_agent_logger.info(f"Creating MCPServerStdio for connection {connection.id}", extra={'correlation_id': correlation_id})
            log_query_agent_logger.info(f"Command: {command}, Args: {args}, Env keys: {list(env_dict.keys())}", extra={'correlation_id': correlation_id})
            
            return MCPServerStdio(command=command, args=args, env=env_dict)
            
        except Exception as e:
            log_query_agent_logger.error(f"Error getting/configuring Grafana stdio server: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
            return None

    async def run_query(self, query: str, time_range: Optional[Dict[str, str]] = None) -> AsyncGenerator[Union[Dict[str, Any], LogQueryResult], None]:
        """Run a log query using the Grafana MCP stdio server, yielding status updates."""
        log_query_agent_logger.info(f"Running log query: '{query}', Time range: {time_range}")
        correlation_id = self.notebook_id
        yield {"status": "starting", "message": "Initializing Grafana connection...", "agent_type": "log"}
        
        stdio_server = await self._get_stdio_server()
        
        if not stdio_server:
            error_msg = "Failed to configure Grafana MCP stdio server. Check default connection."
            log_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id})
            yield {"status": "error", "message": error_msg, "agent_type": "log"}
            yield LogQueryResult(query=query, data=[], error=error_msg) # Yield final error result
            return
        
        yield {"status": "connection_ready", "message": "Grafana connection established.", "agent_type": "log"}

        # Use the system prompt string stored during initialization
        system_prompt_value = self.system_prompt_str

        # Create a local agent instance configured with the stdio server
        local_agent = Agent(
            self.model,
            result_type=LogQueryResult,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt_value, # Pass the resolved string
        )
        log_query_agent_logger.info(f"Local Agent instance created with MCPServerStdio for Grafana.", extra={'correlation_id': correlation_id})
        yield {"status": "agent_created", "message": "Log agent ready.", "agent_type": "log"}
            
        query_with_context = query
        if time_range:
             query_with_context += f" (Time range: {time_range})"
        log_query_agent_logger.info(f"Enhanced query for agent: '{query_with_context}'")

        try:
            async with local_agent.run_mcp_servers():
                log_query_agent_logger.info(f"Calling local_agent.run...", extra={'correlation_id': correlation_id})
                yield {"status": "agent_running", "message": "Agent is processing the log query...", "agent_type": "log"}
                
                result = await local_agent.run(query_with_context)
                
                log_query_agent_logger.info(f"Agent run completed. Raw result: {result}", extra={'correlation_id': correlation_id})
                yield {"status": "agent_run_complete", "agent_type": "log"}
                
                if result and isinstance(result.data, LogQueryResult):
                    log_query_agent_logger.info(f"Successfully parsed result data.", extra={'correlation_id': correlation_id})
                    yield {"status": "parsing_success", "agent_type": "log"}
                    result.data.query = query
                    yield result.data # Yield final success result
                    return 
                elif result and result.data: # Handle case where result.data is not LogQueryResult but has data
                    log_query_agent_logger.warning(f"Agent returned data but not LogQueryResult type: {type(result.data)}", extra={'correlation_id': correlation_id})
                    error_msg = "Agent returned unexpected data type"
                    yield {"status": "parsing_error", "error": error_msg, "agent_type": "log"}
                    yield LogQueryResult(query=query, data=[], error=error_msg)
                    return
                else:
                    error_detail = f"Raw result: {result}" if result else "Result was None"
                    log_query_agent_logger.error(f"Agent did not return valid data. {error_detail}", extra={'correlation_id': correlation_id})
                    error_msg = "Agent failed to return valid data"
                    yield {"status": "no_valid_data", "error": error_msg, "agent_type": "log"}
                    yield LogQueryResult(query=query, data=[], error=error_msg)
                    return
        
        except FileNotFoundError as fnf_err:
            error_msg = f"MCP command not found: {fnf_err}. Ensure mcp-grafana is installed."
            log_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id}, exc_info=True)
            yield {"status": "mcp_command_error", "error": error_msg, "agent_type": "log"}
            yield LogQueryResult(query=query, data=[], error=error_msg)
            return
        except ConnectionRefusedError as conn_err:
             error_msg = f"MCP connection refused: {conn_err}"
             log_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id}, exc_info=True)
             yield {"status": "mcp_connection_error", "error": error_msg, "agent_type": "log"}
             yield LogQueryResult(query=query, data=[], error=error_msg)
             return
        except Exception as e:
            error_msg = f"Agent run failed: {str(e)}"
            log_query_agent_logger.error(error_msg, extra={'correlation_id': correlation_id}, exc_info=True)
            yield {"status": "general_error", "error": error_msg, "agent_type": "log"}
            yield LogQueryResult(query=query, data=[], error=error_msg)
            return
