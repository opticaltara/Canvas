"""
GitHub Query Agent Service

Agent for handling GitHub interactions via a locally running MCP server.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP

from backend.config import get_settings
from backend.core.query_result import GithubQueryResult 

github_query_agent_logger = logging.getLogger("ai.github_query_agent")

class GitHubQueryAgent:
    """Agent for interacting with GitHub MCP server."""
    def __init__(self, notebook_id: str, mcp_servers: Optional[List[MCPServerHTTP]] = None):
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
            "Pay attention to the current date and time provided at the beginning of the user's query "
            "to accurately interpret time-related requests (e.g., 'recent', 'last week'). "
            "Be precise in your tool usage based on the request."
        )
        github_query_agent_logger.info(f"Using GitHub system prompt - {system_prompt}.")

        self.agent = Agent(
            self.model,
            result_type=GithubQueryResult,
            mcp_servers=self.mcp_servers,
            system_prompt=system_prompt,
            tools=[] 
        )
        github_query_agent_logger.info(f"GitHubQueryAgent initialized successfully.")

    async def run_query(self, description: str) -> GithubQueryResult:
        """Run a query (natural language request) against the GitHub MCP server."""
        github_query_agent_logger.info(f"Running GitHub query. Original description: '{description}'")
        
        # Check if MCP servers are configured
        if not self.mcp_servers:
            error_msg = "GitHub MCP server (MCPServerHTTP) not configured for this agent."
            github_query_agent_logger.error(error_msg)
            return GithubQueryResult(query=description, data=None, error=error_msg)
            
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        enhanced_description = f"Current time is {current_time_utc}. User query: {description}"
        github_query_agent_logger.info(f"Enhanced description with time: '{enhanced_description}'")

        try:
            async with self.agent.run_mcp_servers():
                github_query_agent_logger.info(f"Calling self.agent.run...")
                result = await self.agent.run(enhanced_description)
                github_query_agent_logger.info(f"Agent run completed. Raw result: {result}")
                if result and hasattr(result, 'data') and result.data:
                    github_query_agent_logger.info(f"Successfully parsed result data of type: {type(result.data)}")
                    if isinstance(result.data, GithubQueryResult):
                         result.data.query = description # Ensure original query is set
                         return result.data
                    else:
                         return GithubQueryResult(query=description, data=result.data) 
                else:
                    error_detail = f"Raw result: {result}" if result else "Result was None"
                    github_query_agent_logger.error(f"Agent did not return valid data. {error_detail}")
                    return GithubQueryResult(query=description, data=None, error="Agent failed to return valid data") 
        except Exception as e:
            github_query_agent_logger.error(f"Error during agent run: {e}", exc_info=True)
            return GithubQueryResult(query=description, data=None, error=str(e))
