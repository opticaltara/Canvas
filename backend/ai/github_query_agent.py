"""
GitHub Query Agent Service

Agent for handling GitHub interactions via a locally running MCP server.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP

from backend.config import get_settings
from backend.ai.agent import DataQueryResult # For consistent return type

github_query_agent_logger = logging.getLogger("ai.github_query_agent")

# Placeholder for a specific result type if needed later
# from backend.core.query_result import GitHubQueryResult 
GitHubQueryResult = Dict # Use Dict as result_type for now

class GitHubQueryAgent:
    """Agent for interacting with GitHub MCP server."""
    def __init__(self, notebook_id: str, mcp_servers: Optional[List[MCPServerHTTP]] = None):
        # Note: Unlike LogQueryAgent, we don't have different "sources", only the GitHub MCP.
        github_query_agent_logger.info(f"Initializing GitHubQueryAgent for notebook_id: {notebook_id}")
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
    
        # Define system prompt for GitHub interactions
        system_prompt = (
            "You are a specialized agent interacting with a GitHub MCP server. "
            "Use the available tools provided by the MCP server to answer the user's request about GitHub resources "
            "(like repositories, issues, pull requests, users, etc.). "
            "Be precise in your tool usage based on the request."
        )
        github_query_agent_logger.debug(f"Using GitHub system prompt.")

        self.agent = Agent(
            self.model,
            result_type=GitHubQueryResult, # Use Dict for now
            mcp_servers=self.mcp_servers,
            system_prompt=system_prompt,
            # Consider adding specific GitHub tools here if needed for structured output,
            # but relying on MCP discovery is the primary mechanism.
            tools=[] 
        )
        github_query_agent_logger.info(f"GitHubQueryAgent initialized successfully.")

    async def run_query(self, description: str) -> DataQueryResult:
        """Run a query (natural language request) against the GitHub MCP server."""
        github_query_agent_logger.info(f"Running GitHub query. Description: '{description}'")
        
        try:
            async with self.agent.run_mcp_servers():
                github_query_agent_logger.debug(f"Calling self.agent.run...")
                result = await self.agent.run(description)
                github_query_agent_logger.debug(f"Agent run completed. Raw result: {result}")
                if result and result.data:
                    github_query_agent_logger.info(f"Successfully parsed result data of type: {type(result.data)}")
                    return DataQueryResult(query=description, data=result.data)
                else:
                    github_query_agent_logger.error(f"Agent did not return valid data. Raw result: {result}")
                    return DataQueryResult(query=description, data=[], error="Agent failed to return valid data")
        except Exception as e:
            github_query_agent_logger.error(f"Error during agent run: {e}", exc_info=True)
            return DataQueryResult(query=description, data=[], error=str(e))
