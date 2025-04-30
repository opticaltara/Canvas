"""
GitHub Query Agent Service

Agent for handling GitHub interactions via a locally running MCP server.
"""

import logging
from typing import  Optional, AsyncGenerator, Union, Dict, Any, List
from datetime import datetime, timezone
import json
import os # Added import

from pydantic_ai import Agent, UnexpectedModelBehavior, CallToolsNode
from pydantic_ai.messages import (
    FunctionToolCallEvent, 
    FunctionToolResultEvent, 
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from mcp.shared.exceptions import McpError 

from backend.config import get_settings
from backend.core.query_result import GithubQueryResult, ToolCallRecord
from backend.services.connection_manager import get_github_stdio_params

github_query_agent_logger = logging.getLogger("ai.github_query_agent")

class GitHubQueryAgent:
    """Agent for interacting with GitHub MCP server using stdio."""
    def __init__(self, notebook_id: str):
        github_query_agent_logger.info(f"Initializing GitHubQueryAgent for notebook_id: {notebook_id}")
        self.settings = get_settings()
        # Revert to using standard OpenAIModel
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.notebook_id = notebook_id
        self.agent = None
        # Add instance variables to store results from the last attempt
        self._last_run_result: Any = None
        self._last_successful_calls: List[ToolCallRecord] = []
        github_query_agent_logger.info(f"GitHubQueryAgent initialized successfully (Agent instance created later).")

    async def _get_stdio_server(self) -> Optional[MCPServerStdio]:
        """Fetches default GitHub connection and creates the MCPServerStdio instance."""
        try:
            stdio_params = await get_github_stdio_params()
            if not stdio_params:
                github_query_agent_logger.error("Failed to get StdioServerParameters for GitHub.")
                return None

            github_query_agent_logger.info(f"Retrieved StdioServerParameters. Command: {stdio_params.command} {stdio_params.args}")
            return MCPServerStdio(command=stdio_params.command, args=stdio_params.args)

        except Exception as e:
            github_query_agent_logger.error(f"Error creating MCPServerStdio instance from params: {e}", exc_info=True)
            return None

    def _read_system_prompt(self) -> str:
        """Reads the system prompt from the dedicated file."""
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "github_agent_system_prompt.txt")
        try:
            with open(prompt_file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            github_query_agent_logger.error(f"System prompt file not found at: {prompt_file_path}")
            # Return a default minimal prompt as fallback
            return "You are a helpful AI assistant interacting with a GitHub MCP server."
        except Exception as e:
            github_query_agent_logger.error(f"Error reading system prompt file {prompt_file_path}: {e}", exc_info=True)
            return "You are a helpful AI assistant interacting with a GitHub MCP server." # Fallback

    def _initialize_agent(self, stdio_server: MCPServerStdio) -> Agent:
        """Initializes the Pydantic AI Agent with GitHub specific configuration."""
        # Read system prompt from file
        system_prompt = self._read_system_prompt()

        # Revert Agent instantiation type hint if necessary (or keep previous simple form)
        agent = Agent(
            self.model,
            output_type=GithubQueryResult,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt,
        )
        github_query_agent_logger.info(f"Agent instance created with MCPServerStdio.")
        # Add back type: ignore if needed by linter/type checker
        return agent # type: ignore

    async def _run_single_attempt(self, agent: Agent, current_description: str, attempt_num: int) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs a single attempt of the agent iteration, yielding status updates."""
        github_query_agent_logger.info(f"Starting agent iteration attempt {attempt_num}...")
        yield {"status": "agent_iterating", "attempt": attempt_num, "message": "Agent is processing..."}

        successful_tool_calls_in_attempt: List[ToolCallRecord] = []
        run_result = None
        self._last_run_result = None
        self._last_successful_calls = []
        
        async with agent.iter(current_description) as agent_run:
            pending_tool_calls: Dict[str, Dict[str, Any]] = {}
            async for node in agent_run:
                if isinstance(node, CallToolsNode):
                    github_query_agent_logger.info(f"Attempt {attempt_num}: Processing CallToolsNode, streaming events...")
                    try:
                        async with node.stream(agent_run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    tool_call_id = event.part.tool_call_id
                                    tool_name = getattr(event.part, 'tool_name', None)
                                    tool_args = getattr(event.part, 'args', {})
                                    if not tool_name:
                                        github_query_agent_logger.warning(f"Attempt {attempt_num}: ToolCallPart missing 'tool_name'. Part: {event.part!r}")
                                        tool_name = "UnknownTool"
                                    pending_tool_calls[tool_call_id] = {"tool_name": tool_name, "tool_args": tool_args}
                                    yield {"status": "tool_call_requested", "attempt": attempt_num, "tool_call_id": tool_call_id, "tool_name": tool_name, "tool_args": tool_args}
                                elif isinstance(event, FunctionToolResultEvent):
                                    tool_call_id = event.tool_call_id
                                    tool_result = event.result.content
                                    if tool_call_id in pending_tool_calls:
                                        call_info = pending_tool_calls.pop(tool_call_id)
                                        args_data = call_info["tool_args"]
                                        if isinstance(args_data, str):
                                            try:
                                                args_data = json.loads(args_data)
                                            except json.JSONDecodeError as json_err:
                                                github_query_agent_logger.error(f"Attempt {attempt_num}: Failed to parse tool_args JSON string: {args_data}. Error: {json_err}")
                                                args_data = {}

                                        record = ToolCallRecord(tool_call_id=tool_call_id, tool_name=call_info["tool_name"], tool_args=args_data, tool_result=tool_result)
                                        successful_tool_calls_in_attempt.append(record)
                                        github_query_agent_logger.info(f"Attempt {attempt_num}: Stored tool record: {record}")
                                    else:
                                        github_query_agent_logger.warning(f"Attempt {attempt_num}: Received result for unknown/processed ID: {tool_call_id}")
                    except Exception as stream_err:
                        github_query_agent_logger.error(f"Attempt {attempt_num}: Error during CallToolsNode stream: {stream_err}", exc_info=True)
                        raise stream_err 
            
            run_result = agent_run.result
            github_query_agent_logger.info(f"Attempt {attempt_num}: Agent iteration finished. Raw result: {run_result}")

        self._last_run_result = run_result
        self._last_successful_calls = successful_tool_calls_in_attempt

        yield {"status": "agent_run_complete", "attempt": attempt_num}

    def _process_final_result(self, run_result: Any, successful_tool_calls_in_attempt: List[ToolCallRecord], original_description: str) -> Optional[GithubQueryResult]:
        """Processes the raw agent result, filters tool calls, and constructs the final GithubQueryResult."""
        # Access the final output using .output
        result_data_obj = None
        if run_result and hasattr(run_result, 'output'):
            result_data_obj = run_result.output
        elif run_result:
            github_query_agent_logger.warning(f"AgentRunResult structure unexpected: {run_result}. Attempting to use directly.")
            result_data_obj = run_result

        if result_data_obj and isinstance(result_data_obj, GithubQueryResult) and result_data_obj.data is not None:
            github_query_agent_logger.info(f"Processing GithubQueryResult directly.")
            final_result_data = result_data_obj
            final_result_data.query = original_description # Ensure original query is set
            final_result_data.tool_calls = successful_tool_calls_in_attempt
            return final_result_data
        elif result_data_obj is not None:
            github_query_agent_logger.info(f"Wrapping non-GithubQueryResult data: {type(result_data_obj)}.")
            final_result_data = GithubQueryResult(
                query=original_description, 
                data=result_data_obj,
                tool_calls=successful_tool_calls_in_attempt
            )
            return final_result_data
        else:
            github_query_agent_logger.error(f"Final data object from agent run was None or invalid. Raw run_result: {run_result}")
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
        
        try:
            agent = self._initialize_agent(stdio_server)
            yield {"status": "agent_created", "message": "GitHub agent ready."}
        except Exception as init_err:
            error_msg = f"Failed to initialize GitHub Agent: {init_err}"
            github_query_agent_logger.error(error_msg, exc_info=True)
            yield {"status": "error", "message": error_msg}
            yield GithubQueryResult(query=description, data=None, error=error_msg)
            return
            
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        base_description = f"Current time is {current_time_utc}. User query: {description}"
        current_description = base_description
        
        max_attempts = 5
        last_error = None
        final_result_data: Optional[GithubQueryResult] = None
        attempt_completed = 0 # Track last completed attempt number for logging

        yield {"status": "starting_attempts", "message": f"Attempting GitHub query (max {max_attempts} attempts)..."}

        # Wrap the MCP server management and loop logic in a try block
        try:
            for attempt in range(max_attempts):
                attempt_completed = attempt + 1
                github_query_agent_logger.info(f"GitHub Query Attempt {attempt_completed}/{max_attempts}")
                yield {"status": "attempt_start", "attempt": attempt_completed, "max_attempts": max_attempts}

                try: 
                    # MCP Server context manager is now INSIDE the loop for each attempt
                    async with agent.run_mcp_servers():
                        github_query_agent_logger.info(f"Attempt {attempt_completed}: MCP Server connection active.")
                        yield {"status": "mcp_connection_active", "attempt": attempt_completed, "message": "MCP server connection established."}

                        # --- Inner try block for the actual agent run within the MCP context --- 
                        try:
                            # Consume the async generator to execute the attempt
                            async for status_update in self._run_single_attempt(agent, current_description, attempt_completed):
                                yield status_update # Forward the status updates

                            # Attempt finished, access results stored in instance variables
                            run_result = self._last_run_result
                            successful_calls = self._last_successful_calls

                            # --- Process the successful result ---
                            final_result_data = self._process_final_result(run_result, successful_calls, description)

                            # --- If successful, exit the outer loop ---
                            if final_result_data:
                                github_query_agent_logger.info(f"Successfully processed result on attempt {attempt_completed}.")
                                yield {"status": "parsing_success", "attempt": attempt_completed} # Use a distinct status
                                break # Break inner try block; outer loop will break due to final_result_data check
                            else:
                                # Handle case where processing the result failed (returned None)
                                github_query_agent_logger.error(f"_process_final_result returned None on attempt {attempt_completed}. Raw run_result: {run_result}")
                                last_error = "Failed to process successful agent result."

                                # Non-retryable for now, break inner try block
                                yield {"status": "error", "attempt": attempt_completed, "reason": "result processing failed"}
                                break 

                        except McpError as mcp_err:
                            error_str = str(mcp_err)
                            github_query_agent_logger.warning(f"MCPError during agent run attempt {attempt_completed}: {error_str}")
                            yield {"status": "mcp_error", "attempt": attempt_completed, "error": error_str}
                            last_error = error_str
                            # Check if retry is possible (move continue outside inner try)
                            if attempt < max_attempts - 1:
                                error_context = f"\\n\\nINFO: Attempt {attempt_completed} failed with MCP tool error: {error_str}. Retrying..."
                                current_description += error_context
                                yield {"status": "retrying", "attempt": attempt_completed, "reason": "mcp_error"}
                                # No break here, let the outer loop continue after this inner try block finishes
                            else:
                                github_query_agent_logger.error(f"MCPError on final attempt {max_attempts}: {error_str}", exc_info=True)
                                break # Break inner try block

                        except UnexpectedModelBehavior as e:
                            github_query_agent_logger.error(f"UnexpectedModelBehavior during agent run attempt {attempt_completed}: {e}", exc_info=True)
                            yield {"status": "model_error", "attempt": attempt_completed, "error": str(e)}
                            last_error = f"Agent run failed due to unexpected model behavior: {str(e)}"
                            break # Break inner try block

                        except Exception as e: # Catch other potential errors during agent iteration or processing
                            if isinstance(e, TypeError) and "'NoneType' object cannot be interpreted as an integer" in str(e):
                                error_msg = "TypeError: OpenAI API response likely missing 'created' timestamp."
                                github_query_agent_logger.error(f"Timestamp Error during agent run attempt {attempt_completed}: {error_msg}", exc_info=True)
                                yield {"status": "timestamp_error", "attempt": attempt_completed, "error": error_msg}
                                last_error = f"Agent iteration failed: {error_msg}"
                                # Check if retry is possible (move continue outside inner try)
                                if attempt < max_attempts - 1:
                                    error_context = f"\\n\\nINFO: Attempt {attempt_completed} failed due to a likely missing timestamp in the API response. Retrying..."
                                    current_description += error_context
                                    yield {"status": "retrying", "attempt": attempt_completed, "reason": "timestamp_error"}
                                    # No break here, let the outer loop continue
                                else:
                                    github_query_agent_logger.error(f"Timestamp Error on final attempt {max_attempts}: {error_msg}")
                                    break # Break inner try block
                            else:
                                # Handle other general errors
                                github_query_agent_logger.error(f"General error during agent iteration/processing attempt {attempt_completed}: {e}", exc_info=True)
                                yield {"status": "general_error_run", "attempt": attempt_completed, "error": str(e)}
                                last_error = f"Agent iteration/processing failed unexpectedly: {str(e)}"
                                break # Break inner try block

                except Exception as mcp_mgmt_err: # Catch errors specifically from `async with agent.run_mcp_servers()`
                    github_query_agent_logger.error(f"Fatal error managing MCP server for attempt {attempt_completed}: {mcp_mgmt_err}", exc_info=True)
                    yield {"status": "fatal_mcp_error", "attempt": attempt_completed, "error": str(mcp_mgmt_err)}
                    last_error = f"Fatal error managing MCP connection: {str(mcp_mgmt_err)}"
                    # If MCP mgmt fails, we cannot continue the loop
                    break

                # --- Check if outer loop should break or continue ---
                if final_result_data: # Success occurred in the inner block
                    break
                elif last_error and attempt >= max_attempts - 1: # Failure on the last attempt
                    break
                elif last_error: # Failure on a retryable attempt
                    # The 'retrying' status was already yielded inside the inner except blocks
                    continue # Proceed to the next iteration of the outer loop
                # Add a safety break if something unexpected happened
                elif attempt >= max_attempts - 1:
                    github_query_agent_logger.warning(f"Reached max attempts ({max_attempts}) without explicit success or failure decision.")
                    last_error = last_error or "Max attempts reached without explicit success or failure."
                    break

        except Exception as e: # Catch errors managing MCP connection specifically
            github_query_agent_logger.error(f"Fatal error during MCP server management or unhandled error in loop: {e}", exc_info=True)
            yield {"status": "fatal_mcp_error", "error": str(e)}
            last_error = f"Fatal error managing MCP connection: {str(e)}"
            # Proceed to final yield block below

        # --- Yield final result or error *after* the loop ---
        if final_result_data:
            github_query_agent_logger.info(f"Yielding final successful result after MCP context exit.")
            yield final_result_data
        elif last_error: # If loop broke due to error or max attempts reached with an error, or MCP error
            final_error_msg = f"GitHub query finished after {attempt_completed} attempts. Last error: {last_error}"
            github_query_agent_logger.error(final_error_msg)
            yield {"status": "error_final", "error": final_error_msg}
            yield GithubQueryResult(query=description, data=None, error=final_error_msg)
        else: # Should not happen if logic is correct, but catch if loop finished without success/error
            final_error_msg = f"GitHub query finished after {attempt_completed} attempts without success or explicit error."
            github_query_agent_logger.warning(final_error_msg)
            yield {"status": "finished_no_result", "attempts": attempt_completed}
            yield GithubQueryResult(query=description, data=None, error=final_error_msg)
