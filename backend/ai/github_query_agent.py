"""
GitHub Query Agent Service

Agent for handling GitHub interactions via a locally running MCP server.
"""

import logging
from typing import  Optional, AsyncGenerator, Union, Dict, Any, List
from datetime import datetime, timezone

from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import CallToolsNode
from mcp import StdioServerParameters
from mcp.shared.exceptions import McpError 

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
                
            dynamic_args = [
                 f"-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={pat}",
                 *args_template
            ]
            
            full_args = command_list[1:] + dynamic_args
            command = command_list[0]
            
            server_params = StdioServerParameters(
                command=command,
                args=full_args
            )
            
            github_query_agent_logger.info(f"Created StdioServerParameters for connection {github_conn.id}")
            github_query_agent_logger.info(f"Command: {command}, Args Template: {args_template}")
            
            return MCPServerStdio(command=command, args=full_args)
            
        except Exception as e:
            github_query_agent_logger.error(f"Error getting/configuring GitHub stdio server: {e}", exc_info=True)
            return None

    async def run_query(
        self,
        description: str,
    ) -> AsyncGenerator[Union[Dict[str, Any], GithubQueryResult], None]:
        """Run a query (natural language request) against the GitHub MCP server via stdio, yielding status updates."""
        github_query_agent_logger.info(f"Running GitHub query. Original description: '{description}'")
        yield {"status": "starting", "message": "Initializing GitHub connection..."}
        
        stdio_server = await self._get_stdio_server()
        if not stdio_server:
            error_msg = "Failed to configure GitHub MCP stdio server. Check default connection and configuration."
            github_query_agent_logger.error(error_msg)
            yield {"status": "error", "message": error_msg}
            yield GithubQueryResult(query=description, data=None, error=error_msg) # Yield final error result
            return
            
        yield {"status": "connection_ready", "message": "GitHub connection established."}
            
        # Define system prompt for GitHub interactions
        system_prompt = (
            "You are a specialized agent interacting with a GitHub MCP server via stdio. "
            "Use the available tools provided by the MCP server to answer the user's request about GitHub resources "
            "(like repositories, issues, pull requests, users, etc.).\n"
            "Pay attention to the current date and time provided at the beginning of the user's query "
            "to accurately interpret time-related requests (e.g., 'recent', 'last week').\n"
            "Be precise in your tool usage based on the request.\n\n"
            "IMPORTANT: For complex requests like finding the 'most recent' or 'largest' item across multiple repositories: \n"
            "1. First, use tools to list the relevant repositories for the user context.\n"
            "2. Then, iterate through these repositories, using tools to gather the necessary information (e.g., commit dates, sizes).\n"
            "3. If a repository cannot provide the needed info (e.g., an empty repository has no commits), handle this gracefully: skip that repository for ranking/sorting based on that specific criteria.\n"
            "4. Finally, aggregate or sort the gathered information to answer the original request."
            "Structure the 'data' field of your final response (which should conform to the GithubQueryResult model) to contain ONLY the direct answer to the user's query. For example, if asked for the most recent repository, the 'data' field should contain only the details of that single repository, not the list of all repositories examined unless the query specifically asked for the full list."
        )

        agent = Agent(
            self.model,
            result_type=GithubQueryResult,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt,
        )
        github_query_agent_logger.info(f"Agent instance created with MCPServerStdio.")
        yield {"status": "agent_created", "message": "GitHub agent ready."}
            
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        base_description = f"Current time is {current_time_utc}. User query: {description}"
        current_description = base_description
        
        max_attempts = 5
        last_error = None
        final_result_data: Optional[GithubQueryResult] = None

        yield {"status": "starting_attempts", "message": f"Attempting GitHub query (max {max_attempts} attempts)..."}

        try:
            # Establish MCP connection outside the retry loop
            async with agent.run_mcp_servers():
                github_query_agent_logger.info("MCP Server connection active.")
                yield {"status": "mcp_connection_active", "message": "MCP server connection established."}

                for attempt in range(max_attempts):
                    github_query_agent_logger.info(f"GitHub Query Attempt {attempt + 1}/{max_attempts}")
                    yield {"status": "attempt_start", "attempt": attempt + 1, "max_attempts": max_attempts}

                    # Store sequence of tool calls for this attempt
                    successful_tool_calls_in_attempt: List[Dict[str, Any]] = []
                    
                    try:
                        # Run the agent query using iteration
                        github_query_agent_logger.info(f"Starting agent iteration with description (length {len(current_description)})...")
                        yield {"status": "agent_iterating", "attempt": attempt + 1, "message": "Agent is processing the request step-by-step..."}

                        run_result = None
                        async with agent.iter(current_description) as agent_run:
                             async for node in agent_run:
                                github_query_agent_logger.info(f"Agent step: {type(node).__name__}")
                                # Check for tool calls within CallToolsNode
                                if isinstance(node, CallToolsNode):
                                     tool_node = node
                                     if hasattr(tool_node, 'model_response') and tool_node.model_response and hasattr(tool_node.model_response, 'parts'):
                                         for part in tool_node.model_response.parts:
                                             if isinstance(part, ToolCallPart):
                                                 # Log and store the detected tool call
                                                 tool_name = getattr(part, 'name', 'UnknownTool')
                                                 tool_args = getattr(part, 'args', {})
                                                 github_query_agent_logger.info(f"Tool call detected: Name='{tool_name}', Args={tool_args}")
                                                 # Append to the sequence for this attempt
                                                 successful_tool_calls_in_attempt.append({"tool_name": tool_name, "tool_args": tool_args})
                                                 yield {"status": "tool_call", "attempt": attempt + 1, "tool_name": tool_name, "tool_args": tool_args}
                             run_result = agent_run.result
                             github_query_agent_logger.info(f"Agent iteration finished. Raw result: {run_result}")

                        github_query_agent_logger.info(f"Agent iteration attempt {attempt + 1} completed.")
                        yield {"status": "agent_run_complete", "attempt": attempt + 1}

                        # --- Success Case ---
                        # Access the actual data from AgentRunResult
                        result_data_obj = None
                        if run_result and hasattr(run_result, 'data'):
                           result_data_obj = run_result.data
                        elif run_result:
                           github_query_agent_logger.warning(f"AgentRunResult structure unexpected: {run_result}. Attempting to use directly.")
                           result_data_obj = run_result

                        if result_data_obj and isinstance(result_data_obj, GithubQueryResult) and result_data_obj.data is not None:
                             github_query_agent_logger.info(f"Successfully parsed GithubQueryResult data on attempt {attempt + 1}. Data type: {type(result_data_obj.data)}")
                             yield {"status": "parsing_success", "attempt": attempt + 1}
                             final_result_data = result_data_obj
                             final_result_data.query = description
                             final_result_data.tool_calls = successful_tool_calls_in_attempt # Assign the collected sequence
                             yield final_result_data
                             return
                        elif result_data_obj is not None:
                             github_query_agent_logger.info(f"Agent returned data, but not GithubQueryResult: {type(result_data_obj)}. Wrapping it. Attempt {attempt + 1}.")
                             yield {"status": "parsing_success", "attempt": attempt + 1}
                             # Wrap the data and add tool call sequence
                             final_result_data = GithubQueryResult(
                                 query=description, 
                                 data=result_data_obj,
                                 tool_calls=successful_tool_calls_in_attempt # Assign the collected sequence
                             )
                             yield final_result_data
                             return
                        else:
                             error_detail = f"Final data object from agent_run.result was None or empty. Raw result: {run_result}"
                             github_query_agent_logger.error(f"Agent did not return valid data object after iteration on attempt {attempt + 1}. {error_detail}")
                             yield {"status": "no_valid_data", "attempt": attempt + 1, "message": "Agent iteration completed but returned no usable data object."}
                             last_error = "Agent failed to return valid data after iteration."
                             error_context = f"\\n\\nINFO: Attempt {attempt + 1} completed but returned no valid data. Agent will retry."
                             current_description += error_context
                             yield {"status": "retrying", "attempt": attempt + 1, "reason": "no valid data"}
                             continue # Go to next attempt

                    except McpError as mcp_err:
                        error_str = str(mcp_err)
                        github_query_agent_logger.warning(f"MCPError during agent run attempt {attempt + 1}: {error_str}")
                        yield {"status": "mcp_error", "attempt": attempt + 1, "error": error_str}
                        last_error = error_str
                        if attempt < max_attempts - 1:
                            error_context = f"\\n\\nINFO: Attempt {attempt + 1} failed with tool error: {error_str}. This might be expected for certain operations (e.g., empty repo). Agent will retry."
                            current_description += error_context
                            yield {"status": "retrying", "attempt": attempt + 1, "reason": "mcp_error"}
                            continue # Go to next attempt in the inner loop
                        else:
                            github_query_agent_logger.error(f"MCPError on final attempt {max_attempts}: {error_str}", exc_info=True)
                            break # Exit inner loop to handle final error

                    except UnexpectedModelBehavior as e:
                        github_query_agent_logger.error(f"UnexpectedModelBehavior during agent run attempt {attempt + 1}: {e}", exc_info=True)
                        yield {"status": "model_error", "attempt": attempt + 1, "error": str(e)}
                        last_error = f"Agent run failed due to unexpected model behavior: {str(e)}"
                        break # Exit inner loop to handle final error

                    except Exception as e: # Catch other potential errors during agent iteration or result handling
                        github_query_agent_logger.error(f"General error during agent iteration/processing attempt {attempt + 1}: {e}", exc_info=True)
                        yield {"status": "general_error_run", "attempt": attempt + 1, "error": str(e)}
                        last_error = f"Agent iteration/processing failed unexpectedly: {str(e)}"
                        break # Exit inner loop to handle final error

                # If the loop finished without returning (i.e., max attempts reached or broke due to error)
                if final_result_data is None:
                    final_error_msg = f"GitHub query failed after {attempt + 1} attempts within active MCP connection. Last error: {last_error or 'Max attempts reached without success'}"
                    github_query_agent_logger.error(final_error_msg)
                    yield {"status": "failed_max_attempts", "attempts": attempt + 1, "error": final_error_msg}
                    # Yield final error result *before* exiting the 'async with' block
                    yield GithubQueryResult(query=description, data=None, error=final_error_msg)
                    # Let the 'async with' block clean up naturally

        except Exception as e: # Catch errors during MCP server setup/teardown (outside the loop)
            github_query_agent_logger.error(f"Fatal error during MCP server management or unhandled error in loop: {e}", exc_info=True)
            yield {"status": "fatal_mcp_error", "error": str(e)}
            last_error = f"Fatal error managing MCP connection: {str(e)}"
            # Ensure a final error result is yielded if setup/teardown failed
            if final_result_data is None: # Check if we somehow succeeded before this fatal error
                 yield GithubQueryResult(query=description, data=None, error=last_error)

        # Final check: If we somehow exit the 'try' block without final_result_data being set
        # (e.g., due to an error caught by the outer except) and haven't yielded an error result yet.
        # This is a fallback, the logic above should handle most cases.
        if final_result_data is None and last_error: # Check if an error was recorded
             # Check if an error result has already been yielded
             # This is tricky without tracking yield status, assume if last_error exists, we probably yielded one.
             # If not, yield one last time. This might be redundant but safer.
             # Note: The current structure yields the error within the except blocks or after the loop.
             # Let's remove this potentially redundant final yield as the blocks above should cover it.
             pass # Error handling should be complete within the try/except blocks
