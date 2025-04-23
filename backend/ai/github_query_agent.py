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
from pydantic_ai.mcp import MCPServerStdio
from mcp import StdioServerParameters

from backend.config import get_settings
from backend.core.query_result import GithubQueryResult
from backend.services.connection_manager import ConnectionManager, get_connection_manager

github_query_agent_logger = logging.getLogger("ai.github_query_agent")

class GitHubQueryAgent:
    """Agent for interacting with GitHub MCP server using stdio."""
    def __init__(self, notebook_id: str):
        github_query_agent_logger.info(f"Initializing GitHubQueryAgent for notebook_id: {notebook_id}")
        self.settings = get_settings()
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.notebook_id = notebook_id
        self.agent = None
        github_query_agent_logger.info(f"GitHubQueryAgent initialized successfully (Agent instance created later).")

    async def _get_stdio_server(self) -> Optional[MCPServerStdio]:
        """Fetches default GitHub connection and creates the MCPServerStdio instance."""
        try:
            connection_manager = get_connection_manager()
            github_conn = await connection_manager.get_default_connection("github")
            
            if not github_conn:
                github_query_agent_logger.error("No default GitHub connection found.")
                return None
            
            # Extract config
            config = github_conn.config
            command_list = config.get("mcp_command")
            args_template = config.get("mcp_args_template")
            pat = config.get("github_pat") # Get the stored PAT
            
            if not command_list or not args_template or not pat:
                github_query_agent_logger.error(f"Incomplete stdio configuration in GitHub connection {github_conn.id}: {config}")
                return None
                
            # Construct dynamic args with PAT
            # SECURITY: Ensure PAT is handled securely (e.g., not logged excessively)
            dynamic_args = [
                 f"-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={pat}",
                 *args_template # Append the rest of the template args
            ]
            
            # Combine command parts and dynamic args
            full_args = command_list[1:] + dynamic_args
            command = command_list[0]
            
            server_params = StdioServerParameters(
                command=command,
                args=full_args
            )
            
            github_query_agent_logger.info(f"Created StdioServerParameters for connection {github_conn.id}")
            # Note: We log the command/args template, but avoid logging dynamic_args containing the PAT
            github_query_agent_logger.debug(f"Command: {command}, Args Template: {args_template}")
            
            # Pass command and args directly, not wrapped in server_params
            return MCPServerStdio(command=command, args=full_args)
            
        except Exception as e:
            github_query_agent_logger.error(f"Error getting/configuring GitHub stdio server: {e}", exc_info=True)
            return None

    async def run_query(self, description: str) -> GithubQueryResult:
        """Run a query (natural language request) against the GitHub MCP server via stdio."""
        github_query_agent_logger.info(f"Running GitHub query. Original description: '{description}'")
        
        # Get the stdio server configuration
        stdio_server = await self._get_stdio_server()
        
        if not stdio_server:
            error_msg = "Failed to configure GitHub MCP stdio server. Check default connection and configuration."
            github_query_agent_logger.error(error_msg)
            return GithubQueryResult(query=description, data=None, error=error_msg)
            
        # Define system prompt for GitHub interactions
        system_prompt = (
            "You are a specialized agent interacting with a GitHub MCP server via stdio. "
            "Use the available tools provided by the MCP server to answer the user's request about GitHub resources "
            "(like repositories, issues, pull requests, users, etc.). "
            "Pay attention to the current date and time provided at the beginning of the user's query "
            "to accurately interpret time-related requests (e.g., 'recent', 'last week'). "
            "Be precise in your tool usage based on the request."
        )
        github_query_agent_logger.info(f"Using GitHub system prompt - {system_prompt}.")

        agent = Agent(
            self.model,
            result_type=GithubQueryResult,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt,
        )
        github_query_agent_logger.info(f"Agent instance created with MCPServerStdio.")
            
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        enhanced_description = f"Current time is {current_time_utc}. User query: {description}"
        github_query_agent_logger.info(f"Enhanced description with time: '{enhanced_description}'")

        try:
            # Run the stdio server within the context manager
            async with agent.run_mcp_servers():
                github_query_agent_logger.info(f"Calling agent.run...")
                result = await agent.run(enhanced_description)
                github_query_agent_logger.info(f"Agent run completed. Raw result: {result}")
                if result and hasattr(result, 'data') and result.data:
                    github_query_agent_logger.info(f"Successfully parsed result data of type: {type(result.data)}")
                    if isinstance(result.data, GithubQueryResult):
                         result.data.query = description
                         return result.data
                    else:
                         return GithubQueryResult(query=description, data=result.data)
                else:
                    error_detail = f"Raw result: {result}" if result else "Result was None"
                    github_query_agent_logger.error(f"Agent did not return valid data. {error_detail}")
                    return GithubQueryResult(query=description, data=None, error="Agent failed to return valid data") 
        except Exception as e:
            github_query_agent_logger.error(f"Error during agent run: {e}", exc_info=True)
            error_details = str(e)
            return GithubQueryResult(query=description, data=None, error=f"Agent run failed: {error_details}")
