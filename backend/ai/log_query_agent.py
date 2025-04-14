from typing import Any, Dict, List, Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP

from backend.ai import loki_system_prompt
from backend.config import get_settings
from backend.core.query_result import LogQueryResult


class LogQueryAgent:
    """Agent for querying logs"""
    def __init__(self, source: str, notebook_id: str, mcp_servers: Optional[List[MCPServerHTTP]] = None):
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
        else:
            system_prompt = ""

        self.agent = Agent(
            self.model,
            result_type=LogQueryResult,
            mcp_servers=self.mcp_servers,
            system_prompt=system_prompt
        )

    async def run_query(self, query: str, time_range: Optional[Dict[str, str]] = None) -> LogQueryResult:
        """Run a log query"""
        query = f"Query: {query} in the Time range: {time_range}"
        async with self.agent.run_mcp_servers():
            result = await self.agent.run(query)
            return result.data
