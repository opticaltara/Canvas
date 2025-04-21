import logging
from typing import Any, Dict, List, Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP

from backend.ai import loki_system_prompt
from backend.config import get_settings
from backend.core.query_result import LogQueryResult

log_query_agent_logger = logging.getLogger("ai.log_query_agent")

class LogQueryAgent:
    """Agent for querying logs"""
    def __init__(self, source: str, notebook_id: str, mcp_servers: Optional[List[MCPServerHTTP]] = None):
        log_query_agent_logger.info(f"Initializing LogQueryAgent for source: {source}, notebook_id: {notebook_id}")
        self.settings = get_settings()
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.mcp_servers = mcp_servers or []
        self.notebook_id = notebook_id
    
        if source == "loki":
            system_prompt = loki_system_prompt.system_prompt
            log_query_agent_logger.debug(f"Using Loki system prompt.")
        else:
            system_prompt = ""
            log_query_agent_logger.warning(f"No specific system prompt for source: {source}")

        self.agent = Agent(
            self.model,
            result_type=LogQueryResult,
            mcp_servers=self.mcp_servers,
            system_prompt=system_prompt
        )
        log_query_agent_logger.info(f"LogQueryAgent initialized successfully.")

    async def run_query(self, query: str, time_range: Optional[Dict[str, str]] = None) -> LogQueryResult:
        """Run a log query"""
        log_query_agent_logger.info(f"Running log query. Original query: '{query}', Time range: {time_range}")
        processed_query = f"Query: {query} in the Time range: {time_range}"
        log_query_agent_logger.debug(f"Formatted query for agent: '{processed_query}'")
        
        try:
            async with self.agent.run_mcp_servers():
                log_query_agent_logger.debug(f"Calling self.agent.run...")
                result = await self.agent.run(processed_query)
                log_query_agent_logger.debug(f"Agent run completed. Raw result: {result}")
                if result and result.data:
                    log_query_agent_logger.info(f"Successfully parsed result data of type: {type(result.data)}")
                    return result.data
                else:
                    log_query_agent_logger.error(f"Agent did not return valid data. Raw result: {result}")
                    return LogQueryResult(query=processed_query, data=[], error="Agent failed to return valid data")
        except Exception as e:
            log_query_agent_logger.error(f"Error during agent run: {e}", exc_info=True)
            return LogQueryResult(query=processed_query, data=[], error=str(e))
