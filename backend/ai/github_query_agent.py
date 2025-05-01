"""
GitHub Query Agent Service

Agent for handling GitHub interactions via a locally running MCP server.
"""

import logging
from typing import  Optional, AsyncGenerator, Dict, Any
from datetime import datetime, timezone
import json
import os

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
from backend.services.connection_manager import get_github_stdio_params

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
            return "You are a helpful AI assistant interacting with a GitHub MCP server."
        except Exception as e:
            github_query_agent_logger.error(f"Error reading system prompt file {prompt_file_path}: {e}", exc_info=True)
            return "You are a helpful AI assistant interacting with a GitHub MCP server."

    def _initialize_agent(self, stdio_server: MCPServerStdio) -> Agent:
        """Initializes the Pydantic AI Agent with GitHub specific configuration."""
        system_prompt = self._read_system_prompt()

        agent = Agent(
            self.model,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt,
        )
        github_query_agent_logger.info(f"Agent instance created with MCPServerStdio.")
        return agent

    async def _run_single_attempt(self, agent: Agent, current_description: str, attempt_num: int) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs a single attempt of the agent iteration, yielding status updates and individual tool success/error dictionaries."""
        github_query_agent_logger.info(f"Starting agent iteration attempt {attempt_num}...")
        yield {"type": "status_update", "status": "agent_iterating", "attempt": attempt_num, "message": "Agent is processing..."}

        run_result = None
        
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
                                    
                                    if isinstance(tool_args, str):
                                        try:
                                            tool_args = json.loads(tool_args)
                                        except json.JSONDecodeError as json_err:
                                            github_query_agent_logger.error(f"Attempt {attempt_num}: Failed to parse tool_args JSON string: {tool_args}. Error: {json_err}")
                                            tool_args = {}
                                    pending_tool_calls[tool_call_id] = {"tool_name": tool_name, "tool_args": tool_args}
                                    yield {"type": "status_update", "status": "tool_call_requested", "agent_type": "github_agent", "attempt": attempt_num, "tool_call_id": tool_call_id, "tool_name": tool_name, "tool_args": tool_args}
                                
                                elif isinstance(event, FunctionToolResultEvent):
                                    tool_call_id = event.tool_call_id
                                    tool_result = event.result.content
                                    if tool_call_id in pending_tool_calls:
                                        call_info = pending_tool_calls.pop(tool_call_id)
                                        args_data = call_info["tool_args"]
                                        
                                        yield {
                                            "type": "status_update",
                                            "status": "tool_success",
                                            "agent_type": "github_agent",
                                            "tool_call_id": tool_call_id,
                                            "tool_name": call_info["tool_name"],
                                            "tool_args": args_data,
                                            "tool_result": tool_result
                                        }
                                        github_query_agent_logger.info(f"Attempt {attempt_num}: Yielded success for tool: {call_info['tool_name']} (ID: {tool_call_id})")
                                    else:
                                        github_query_agent_logger.warning(f"Attempt {attempt_num}: Received result for unknown/processed ID: {tool_call_id}")
                    
                    except McpError as mcp_err:
                        error_str = str(mcp_err)
                        github_query_agent_logger.warning(f"MCPError during CallToolsNode stream in attempt {attempt_num}: {error_str}")
                        yield {
                             "type": "tool_error",
                             "status": "error",
                             "agent_type": "github_agent",
                             "tool_call_id": None,
                             "tool_name": None,
                             "tool_args": None,
                             "error": f"MCP Tool Error: {error_str}"
                         }
                        raise mcp_err

                    except Exception as stream_err:
                        github_query_agent_logger.error(f"Attempt {attempt_num}: Error during CallToolsNode stream: {stream_err}", exc_info=True)
                        yield {
                            "type": "tool_error",
                            "status": "error",
                            "agent_type": "github_agent",
                            "tool_call_id": None, 
                            "tool_name": None,
                            "tool_args": None,
                            "error": f"Stream Processing Error: {stream_err}"
                        }
                        raise stream_err
            
            run_result = agent_run.result 
            github_query_agent_logger.info(f"Attempt {attempt_num}: Agent iteration finished. Raw final result (less relevant now): {run_result}")

        yield {"type": "status_update", "status": "agent_run_complete", "agent_type": "github_agent", "attempt": attempt_num}

    async def run_query(
        self,
        description: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run a query (natural language request) against the GitHub MCP server via stdio, yielding status updates and individual tool results/errors."""
        github_query_agent_logger.info(f"Running GitHub query. Original description: '{description}'")
        yield {"type": "status_update", "status": "starting", "agent_type": "github_agent", "message": "Initializing GitHub connection..."}
        
        stdio_server = await self._get_stdio_server()
        if not stdio_server:
            error_msg = "Failed to configure GitHub MCP stdio server. Check default connection and configuration."
            github_query_agent_logger.error(error_msg)
            yield {"type": "tool_error", "status": "error", "agent_type": "github_agent", "error": error_msg, "message": error_msg}
            return
            
        yield {"type": "status_update", "status": "connection_ready", "agent_type": "github_agent", "message": "GitHub connection established."}
        
        try:
            agent = self._initialize_agent(stdio_server)
            yield {"type": "status_update", "status": "agent_created", "agent_type": "github_agent", "message": "GitHub agent ready."}
        except Exception as init_err:
            error_msg = f"Failed to initialize GitHub Agent: {init_err}"
            github_query_agent_logger.error(error_msg, exc_info=True)
            yield {"type": "tool_error", "status": "error", "agent_type": "github_agent", "error": error_msg, "message": error_msg}
            return
            
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        base_description = f"Current time is {current_time_utc}. User query: {description}"
        current_description = base_description
        
        max_attempts = 5
        last_error = None
        attempt_completed = 0
        success_occurred = False

        yield {"type": "status_update", "status": "starting_attempts", "agent_type": "github_agent", "message": f"Attempting GitHub query (max {max_attempts} attempts)..."}

        try:
            for attempt in range(max_attempts):
                attempt_completed = attempt + 1
                github_query_agent_logger.info(f"GitHub Query Attempt {attempt_completed}/{max_attempts}")
                yield {"type": "status_update", "status": "attempt_start", "agent_type": "github_agent", "attempt": attempt_completed, "max_attempts": max_attempts}
                attempt_failed = False

                try: 
                    async with agent.run_mcp_servers():
                        github_query_agent_logger.info(f"Attempt {attempt_completed}: MCP Server connection active.")
                        yield {"type": "status_update", "status": "mcp_connection_active", "agent_type": "github_agent", "attempt": attempt_completed, "message": "MCP server connection established."}

                        try:
                            async for event_data in self._run_single_attempt(agent, current_description, attempt_completed):
                                yield event_data

                                if event_data.get("type") == "status_update" and event_data.get("status") == "tool_success":
                                    success_occurred = True
                                if event_data.get("type") == "tool_error":
                                    last_error = event_data.get("error", "Unknown tool error")

                            github_query_agent_logger.info(f"Agent iteration attempt {attempt_completed} completed without raising exceptions.")

                        except McpError as mcp_err:
                            error_str = str(mcp_err)
                            github_query_agent_logger.warning(f"Attempt {attempt_completed} caught MCPError: {error_str}")
                            last_error = f"MCP Tool Error: {error_str}"
                            attempt_failed = True
                            if attempt < max_attempts - 1:
                                error_context = f"\n\nINFO: Attempt {attempt_completed} failed with MCP tool error: {error_str}. Retrying..."
                                current_description += error_context
                                yield {"type": "status_update", "status": "retrying", "agent_type": "github_agent", "attempt": attempt_completed, "reason": "mcp_error"}
                                continue
                            else:
                                github_query_agent_logger.error(f"MCPError on final attempt {max_attempts}: {error_str}", exc_info=True)
                                break

                        except UnexpectedModelBehavior as e:
                            github_query_agent_logger.error(f"Attempt {attempt_completed} caught UnexpectedModelBehavior: {e}", exc_info=True)
                            yield {"type": "error", "status": "model_error", "agent_type": "github_agent", "attempt": attempt_completed, "error": str(e)}
                            last_error = f"Agent run failed due to unexpected model behavior: {str(e)}"
                            attempt_failed = True
                            break

                        except Exception as e:
                            if isinstance(e, TypeError) and "'NoneType' object cannot be interpreted as an integer" in str(e):
                                error_msg = "TypeError: OpenAI API response likely missing 'created' timestamp."
                                github_query_agent_logger.error(f"Attempt {attempt_completed} caught Timestamp Error: {error_msg}", exc_info=True)
                                yield {"type": "error", "status": "timestamp_error", "agent_type": "github_agent", "attempt": attempt_completed, "error": error_msg}
                                last_error = f"Agent iteration failed: {error_msg}"
                                attempt_failed = True
                                if attempt < max_attempts - 1:
                                    error_context = f"\n\nINFO: Attempt {attempt_completed} failed due to a likely missing timestamp in the API response. Retrying..."
                                    current_description += error_context
                                    yield {"type": "status_update", "status": "retrying", "agent_type": "github_agent", "attempt": attempt_completed, "reason": "timestamp_error"}
                                    continue
                                else:
                                    github_query_agent_logger.error(f"Timestamp Error on final attempt {max_attempts}: {error_msg}")
                                    break
                            else:
                                github_query_agent_logger.error(f"Attempt {attempt_completed} caught general error: {e}", exc_info=True)
                                yield {"type": "error", "status": "general_error_run", "agent_type": "github_agent", "attempt": attempt_completed, "error": str(e)}
                                last_error = f"Agent iteration failed unexpectedly: {str(e)}"
                                attempt_failed = True
                                break

                except Exception as mcp_mgmt_err:
                    github_query_agent_logger.error(f"Fatal error managing MCP server for attempt {attempt_completed}: {mcp_mgmt_err}", exc_info=True)
                    yield {"type": "tool_error", "status": "fatal_mcp_error", "agent_type": "github_agent", "error": f"Fatal MCP management error: {mcp_mgmt_err}", "attempt": attempt_completed}
                    last_error = f"Fatal error managing MCP connection: {str(mcp_mgmt_err)}"
                    break

                if not attempt_failed: 
                    if success_occurred:
                         github_query_agent_logger.info(f"Query processing completed successfully on attempt {attempt_completed} (at least one tool success).")
                         break
                    else:
                         github_query_agent_logger.warning(f"Attempt {attempt_completed} finished without tool successes or critical errors. Last error: {last_error}")
                         if attempt >= max_attempts - 1:
                             last_error = last_error or "Agent finished without calling successful tools."
                             break

                if attempt >= max_attempts - 1:
                    github_query_agent_logger.warning(f"Reached max attempts ({max_attempts}) safety break.")
                    last_error = last_error or "Max attempts reached."
                    break

        except Exception as e:
            github_query_agent_logger.error(f"Fatal error during GitHub query processing: {e}", exc_info=True)
            yield {"type": "tool_error", "status": "fatal_error", "agent_type": "github_agent", "error": f"Fatal query processing error: {str(e)}"}
            last_error = f"Fatal query processing error: {str(e)}"

        if success_occurred:
             github_query_agent_logger.info(f"GitHub query finished after {attempt_completed} attempts with at least one tool success.")
             yield {"type": "final_status", "status": "finished_success", "agent_type": "github_agent", "attempts": attempt_completed}
        elif last_error:
            final_error_msg = f"GitHub query finished after {attempt_completed} attempts. Last error: {last_error}"
            github_query_agent_logger.error(final_error_msg)
            yield {"type": "tool_error", "status": "finished_error", "agent_type": "github_agent", "error": final_error_msg, "attempts": attempt_completed}
        else:
            final_error_msg = f"GitHub query finished after {attempt_completed} attempts without success or explicit error."
            github_query_agent_logger.warning(final_error_msg)
            yield {"type": "tool_error", "status": "finished_no_result", "agent_type": "github_agent", "error": final_error_msg, "attempts": attempt_completed}
