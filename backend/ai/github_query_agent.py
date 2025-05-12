"""
GitHub Query Agent Service

Agent for handling GitHub interactions via a locally running MCP server.
"""

import logging
from typing import  Optional, AsyncGenerator, Dict, Any, Union
from datetime import datetime, timezone
import json
import os
import asyncio  # For implementing retry backoff

from pydantic_ai import Agent, UnexpectedModelBehavior, CallToolsNode
from pydantic_ai.messages import (
    FunctionToolCallEvent, 
    FunctionToolResultEvent,
    ModelMessage, # For message_history
)
# from pydantic_ai.models.openai import OpenAIModel # Replaced with SafeOpenAIModel
from backend.ai.models import SafeOpenAIModel # Added for safer timestamp handling
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from mcp.shared.exceptions import McpError 
from mcp import StdioServerParameters

from backend.config import get_settings
from backend.ai.events import (
    EventType,
    AgentType,
    StatusType,
    StatusUpdateEvent,
    ToolCallRequestedEvent,
    ToolSuccessEvent,
    ToolErrorEvent,
    FinalStatusEvent,
    FatalErrorEvent,
)
from backend.services.connection_manager import get_connection_manager
from backend.services.connection_handlers.registry import get_handler

github_query_agent_logger = logging.getLogger("ai.github_query_agent")

class GitHubQueryAgent:
    """Agent for interacting with GitHub MCP server using stdio."""
    def __init__(self, notebook_id: str):
        github_query_agent_logger.info(f"Initializing GitHubQueryAgent for notebook_id: {notebook_id}")
        self.settings = get_settings()
        self.model = SafeOpenAIModel( # Changed from OpenAIModel to SafeOpenAIModel
                "openai/gpt-4.1",
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.notebook_id = notebook_id
        self.agent = None
        github_query_agent_logger.info(f"GitHubQueryAgent initialized successfully (Agent instance created later).")

    async def _get_stdio_server(self) -> Optional[MCPServerStdio]:
        """Fetches default GitHub connection, gets handler, and creates MCPServerStdio instance."""
        try:
            connection_manager = get_connection_manager()
            # Get the default github connection
            default_conn = await connection_manager.get_default_connection("github")
            if not default_conn:
                github_query_agent_logger.error("No default GitHub connection found.")
                return None
            if not default_conn.config:
                 github_query_agent_logger.error(f"Default GitHub connection {default_conn.id} has no config.")
                 return None

            # Get the handler and stdio parameters
            handler = get_handler("github")
            stdio_params: StdioServerParameters = handler.get_stdio_params(default_conn.config)

            github_query_agent_logger.info(f"Retrieved StdioServerParameters. Command: {stdio_params.command} Args: {stdio_params.args}")
            # Create the MCPServerStdio instance needed by the agent
            return MCPServerStdio(command=stdio_params.command, args=stdio_params.args, env=stdio_params.env)

        except ValueError as ve: # Catch errors from get_handler or get_stdio_params
             github_query_agent_logger.error(f"Configuration error getting GitHub stdio params: {ve}", exc_info=True)
             return None
        except Exception as e:
            github_query_agent_logger.error(f"Error creating MCPServerStdio instance: {e}", exc_info=True)
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

    # Removed _run_single_attempt as its logic is now directly in run_query's attempt loop.

    async def run_query(
        self,
        description: str,
        session_id: str,
        notebook_id: str
    ) -> AsyncGenerator[Any, None]:
        """Run a query (natural language request) against the GitHub MCP server via stdio, yielding status updates and individual tool results/errors."""
        github_query_agent_logger.info(f"Running GitHub query. Original description: '{description}'")
        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.STARTING,
            agent_type=AgentType.GITHUB,
            message="Initializing GitHub connection...",
            attempt=None,
            max_attempts=None,
            reason=None,
            step_id=None,
            original_plan_step_id=None,
            session_id=session_id,
            notebook_id=notebook_id
        )
        
        stdio_server = await self._get_stdio_server()
        if not stdio_server:
            error_msg = "Failed to configure GitHub MCP stdio server. Check default connection and configuration."
            github_query_agent_logger.error(error_msg)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR,
                status=StatusType.FATAL_ERROR,
                agent_type=AgentType.GITHUB,
                error=error_msg,
                session_id=session_id,
                notebook_id=notebook_id
            )
            return
            
        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.CONNECTION_READY, 
            agent_type=AgentType.GITHUB, 
            message="GitHub connection established.",
            attempt=None,
            max_attempts=None,
            reason=None,
            step_id=None,
            original_plan_step_id=None,
            session_id=session_id,
            notebook_id=notebook_id
        )
        
        try:
            agent = self._initialize_agent(stdio_server)
            yield StatusUpdateEvent(
                type=EventType.STATUS_UPDATE,
                status=StatusType.AGENT_CREATED, 
                agent_type=AgentType.GITHUB, 
                message="GitHub agent ready.",
                attempt=None,
                max_attempts=None,
                reason=None,
                step_id=None,
                original_plan_step_id=None,
                session_id=session_id,
                notebook_id=notebook_id
            )
        except Exception as init_err:
            error_msg = f"Failed to initialize GitHub Agent: {init_err}"
            github_query_agent_logger.error(error_msg, exc_info=True)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR,
                status=StatusType.FATAL_ERROR,
                agent_type=AgentType.GITHUB,
                error=error_msg,
                session_id=session_id,
                notebook_id=notebook_id
            )
            return
            
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        base_description = f"Current time is {current_time_utc}. User query: {description}"
        current_description = base_description
        
        max_attempts = 5
        last_error = None
        attempt_completed = 0
        success_occurred = False
        accumulated_message_history: list[ModelMessage] = [] # Initialize message history

        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.STARTING_ATTEMPTS, 
            agent_type=AgentType.GITHUB, 
            message=f"Attempting GitHub query (max {max_attempts} attempts)...",
            max_attempts=max_attempts,
            attempt=None,
            reason=None,
            step_id=None,
            original_plan_step_id=None,
            session_id=session_id,
            notebook_id=notebook_id
        )

        try:
            for attempt in range(max_attempts):
                attempt_completed = attempt + 1
                github_query_agent_logger.info(f"GitHub Query Attempt {attempt_completed}/{max_attempts}")
                yield StatusUpdateEvent(
                    type=EventType.STATUS_UPDATE, status=StatusType.ATTEMPT_START, agent_type=AgentType.GITHUB, 
                    attempt=attempt_completed, max_attempts=max_attempts, 
                    message=None, reason=None, step_id=None, original_plan_step_id=None,
                    session_id=session_id,
                    notebook_id=notebook_id
                )
                attempt_failed = False
                current_attempt_success = False # Track success within this attempt - success_occurred is global
                last_error_for_attempt = None # Specific to this attempt
                agent_produced_final_result = False # Flag if agent completed cleanly

                try:
                    async with agent.run_mcp_servers(): # MCP server management should wrap agent.iter
                        github_query_agent_logger.info(f"Attempt {attempt_completed}: MCP Server context entered.")
                        # Pass accumulated_message_history to agent.iter
                        async with agent.iter(current_description, message_history=accumulated_message_history) as agent_run:
                            github_query_agent_logger.info(f"Attempt {attempt_completed}: agent.iter called. History length: {len(accumulated_message_history)}")
                            yield StatusUpdateEvent(
                                type=EventType.STATUS_UPDATE, status=StatusType.MCP_CONNECTION_ACTIVE,
                                agent_type=AgentType.GITHUB,
                                attempt=attempt_completed,
                                message="MCP server connection established and agent.iter active.",
                                max_attempts=None, reason=None, step_id=None, original_plan_step_id=None,
                                session_id=session_id,
                                notebook_id=notebook_id
                            )

                        # --- Core agent interaction logic ---
                        pending_tool_calls: Dict[str, Dict[str, Any]] = {}
                        yield StatusUpdateEvent(
                            type=EventType.STATUS_UPDATE,
                            status=StatusType.AGENT_ITERATING,
                            agent_type=AgentType.GITHUB,
                            attempt=attempt_completed,
                            message="Agent is processing...",
                            max_attempts=max_attempts,
                            reason=None,
                            step_id=None,
                            original_plan_step_id=None,
                            session_id=session_id,
                            notebook_id=notebook_id
                        )
                        
                        run_result_for_attempt = None # To store agent_run.result

                        try: # This try block is for errors within the agent.iter() and node.stream()
                            async for node in agent_run:
                                if isinstance(node, CallToolsNode):
                                    github_query_agent_logger.info(f"Attempt {attempt_completed}: Processing CallToolsNode, streaming events...")
                                    try:
                                        async with node.stream(agent_run.ctx) as handle_stream:
                                            async for event in handle_stream:
                                                if isinstance(event, FunctionToolCallEvent):
                                                    tool_call_id = event.part.tool_call_id
                                                    tool_name = getattr(event.part, 'tool_name', None)
                                                    tool_args = getattr(event.part, 'args', {})
                                                    if not tool_name:
                                                        github_query_agent_logger.warning(f"Attempt {attempt_completed}: ToolCallPart missing 'tool_name'. Part: {event.part!r}")
                                                        tool_name = "UnknownTool"
                                                    
                                                    if isinstance(tool_args, str):
                                                        try:
                                                            tool_args = json.loads(tool_args)
                                                        except json.JSONDecodeError as json_err:
                                                            github_query_agent_logger.error(f"Attempt {attempt_completed}: Failed to parse tool_args JSON string: {tool_args}. Error: {json_err}")
                                                            tool_args = {}
                                                    elif not isinstance(tool_args, dict):
                                                        github_query_agent_logger.warning(f"Attempt {attempt_completed}: tool_args is not a dict or valid JSON string: {type(tool_args)}. Using empty dict.")
                                                        tool_args = {}
                                                    
                                                    pending_tool_calls[tool_call_id] = {"tool_name": tool_name, "tool_args": tool_args}
                                                    yield ToolCallRequestedEvent(
                                                        type=EventType.TOOL_CALL_REQUESTED, status=StatusType.TOOL_CALL_REQUESTED,
                                                        agent_type=AgentType.GITHUB, attempt=attempt_completed,
                                                        tool_call_id=tool_call_id, tool_name=tool_name, tool_args=tool_args,
                                                        original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                                                    )
                                                
                                                elif isinstance(event, FunctionToolResultEvent):
                                                    tool_call_id = event.tool_call_id
                                                    tool_result = event.result.content
                                                    if tool_call_id in pending_tool_calls:
                                                        call_info = pending_tool_calls.pop(tool_call_id)
                                                        args_data = call_info["tool_args"]
                                                        yield ToolSuccessEvent(
                                                            type=EventType.TOOL_SUCCESS, status=StatusType.TOOL_SUCCESS,
                                                            agent_type=AgentType.GITHUB, attempt=attempt_completed,
                                                            tool_call_id=tool_call_id, tool_name=call_info["tool_name"],
                                                            tool_args=args_data, tool_result=tool_result,
                                                            original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                                                        )
                                                        current_attempt_success = True # Mark success for this attempt
                                                        success_occurred = True # Mark overall success
                                                        github_query_agent_logger.info(f"Attempt {attempt_completed}: Yielded success for tool: {call_info['tool_name']} (ID: {tool_call_id})")
                                                    else:
                                                        github_query_agent_logger.warning(f"Attempt {attempt_completed}: Received result for unknown/processed ID: {tool_call_id}")
                                    
                                    except McpError as mcp_err_stream:
                                        error_str = str(mcp_err_stream)
                                        github_query_agent_logger.warning(f"MCPError during CallToolsNode stream in attempt {attempt_completed}: {error_str}")
                                        mcp_tool_call_id = getattr(mcp_err_stream, 'tool_call_id', None)
                                        failed_tool_info = pending_tool_calls.get(mcp_tool_call_id, {}) if mcp_tool_call_id else {}
                                        yield ToolErrorEvent(
                                            type=EventType.TOOL_ERROR, status=StatusType.MCP_ERROR,
                                            agent_type=AgentType.GITHUB, attempt=attempt_completed,
                                            tool_call_id=mcp_tool_call_id, tool_name=failed_tool_info.get("tool_name"),
                                            tool_args=failed_tool_info.get("tool_args"), error=f"MCP Tool Error: {error_str}",
                                            message=None, original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                                        )
                                        raise mcp_err_stream 

                                    except Exception as stream_err_inner: 
                                        github_query_agent_logger.error(f"Attempt {attempt_completed}: Error during CallToolsNode stream: {stream_err_inner}", exc_info=True)
                                        yield ToolErrorEvent(
                                            type=EventType.TOOL_ERROR, status=StatusType.ERROR,
                                            agent_type=AgentType.GITHUB, attempt=attempt_completed,
                                            tool_call_id=None, tool_name=None, tool_args=None,
                                            error=f"Stream Processing Error: {stream_err_inner}", message=None,
                                            original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                                        )
                                        raise stream_err_inner 
                            
                            run_result_for_attempt = agent_run.result 
                            github_query_agent_logger.info(f"Attempt {attempt_completed}: Agent iteration finished. Raw final result: {run_result_for_attempt}")
                            if not attempt_failed:
                                agent_produced_final_result = True
                        
                        # --- Handle exceptions from the agent.iter() or node.stream() logic ---
                        except McpError as mcp_err: 
                            error_str = str(mcp_err)
                            github_query_agent_logger.warning(f"Attempt {attempt_completed} caught MCPError: {error_str}")
                            last_error_for_attempt = f"MCP Tool Error: {error_str}"
                            attempt_failed = True
                        except UnexpectedModelBehavior as e:
                            github_query_agent_logger.error(f"Attempt {attempt_completed} caught UnexpectedModelBehavior: {e}", exc_info=True)
                            last_error_for_attempt = f"Agent run failed due to unexpected model behavior: {str(e)}"
                            attempt_failed = True
                            yield ToolErrorEvent( 
                                type=EventType.TOOL_ERROR, status=StatusType.MODEL_ERROR,
                                agent_type=AgentType.GITHUB, attempt=attempt_completed,
                                error=last_error_for_attempt, 
                                tool_call_id=None, tool_name=None, tool_args=None, message=None,
                                original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                            )
                        except Exception as e:
                            is_timestamp_error = isinstance(e, TypeError) and "'NoneType' object cannot be interpreted as an integer" in str(e)
                            error_msg = "TypeError: OpenAI API response likely missing 'created' timestamp." if is_timestamp_error else f"Agent iteration failed unexpectedly: {str(e)}"
                            error_status = StatusType.TIMESTAMP_ERROR if is_timestamp_error else StatusType.GENERAL_ERROR_RUN
                            github_query_agent_logger.error(f"Attempt {attempt_completed} caught general error: {error_msg}", exc_info=True)
                            last_error_for_attempt = error_msg
                            attempt_failed = True
                            yield ToolErrorEvent( 
                                type=EventType.TOOL_ERROR, status=error_status,
                                agent_type=AgentType.GITHUB, attempt=attempt_completed,
                                error=error_msg, tool_call_id=None, tool_name=None, tool_args=None, message=None,
                                original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                            )
                        finally:
                            # Capture history after the agent_run block finishes or if an error occurs within it.
                            if hasattr(agent_run, 'ctx') and hasattr(agent_run.ctx, 'state') and hasattr(agent_run.ctx.state, 'message_history'):
                                accumulated_message_history = agent_run.ctx.state.message_history[:]
                                github_query_agent_logger.info(f"Attempt {attempt_completed} (finally): Captured message history of length {len(accumulated_message_history)}.")
                            else: # This case should ideally not happen if agent_run is always valid here
                                github_query_agent_logger.warning(f"Attempt {attempt_completed} (finally): Could not capture message history from agent_run.ctx.state.")
                            
                            yield StatusUpdateEvent( # Moved from _run_single_attempt
                                type=EventType.STATUS_UPDATE,
                                status=StatusType.AGENT_RUN_COMPLETE, 
                                agent_type=AgentType.GITHUB, 
                                attempt=attempt_completed,
                                message=None, max_attempts=None, reason=None, step_id=None,
                                original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                            )
                            github_query_agent_logger.info(f"Attempt {attempt_completed}: Exiting core agent interaction block.")
                        # --- End core agent interaction wrap ---

                        if attempt_failed:  # Check if the attempt itself failed
                            # Update last_error so that outer scope has latest info
                            last_error = last_error_for_attempt or last_error

                            # Detect provider-level errors (OpenRouter often returns "Provider returned error" with code 4)
                            provider_error_detected = False
                            if last_error_for_attempt:
                                lowered = last_error_for_attempt.lower()
                                provider_error_detected = (
                                    "provider returned error" in lowered
                                    or "'code': 4" in lowered
                                    or '"code": 4' in lowered
                                )

                            # Determine back-off delay – exponential capped at 60 s if provider error, otherwise 0
                            backoff_seconds = 0
                            if provider_error_detected:
                                backoff_seconds = min(2 ** (attempt_completed), 60)

                            if attempt < max_attempts - 1:
                                # Build retry context for the next prompt
                                error_context = (
                                    f"\\n\\nINFO: Attempt {attempt_completed} failed with error: {last_error}. "
                                    "Retrying..."
                                )
                                current_description += error_context

                                # Inform frontend about retry/backoff
                                retry_message = (
                                    f"Backing off {backoff_seconds}s before retry due to provider error." if backoff_seconds else None
                                )
                                yield StatusUpdateEvent(
                                    type=EventType.STATUS_UPDATE,
                                    status=StatusType.RETRYING,
                                    agent_type=AgentType.GITHUB,
                                    attempt=attempt_completed,
                                    reason=f"error: {str(last_error)[:100]}...",
                                    message=retry_message,
                                    max_attempts=max_attempts,
                                    step_id=None,
                                    original_plan_step_id=None,
                                    session_id=session_id,
                                    notebook_id=notebook_id,
                                )

                                # Apply backoff delay if needed
                                if backoff_seconds:
                                    github_query_agent_logger.info(
                                        f"Attempt {attempt_completed}: Backing off for {backoff_seconds}s before next retry due to provider error."
                                    )
                                    await asyncio.sleep(backoff_seconds)

                                continue  # Move to next attempt
                            else:
                                github_query_agent_logger.error(
                                    f"Error on final attempt {max_attempts}: {last_error}"
                                )
                                break  # Exit attempt loop

                except Exception as mcp_mgmt_err:
                    # This catches errors in managing the MCP server itself (e.g., startup/shutdown)
                    github_query_agent_logger.error(f"Fatal error managing MCP server for attempt {attempt_completed}: {mcp_mgmt_err}", exc_info=True)
                    yield FatalErrorEvent(
                        type=EventType.FATAL_ERROR,
                        status=StatusType.FATAL_MCP_ERROR,
                        agent_type=AgentType.GITHUB,
                        error=f"Fatal MCP management error: {str(mcp_mgmt_err)}",
                        session_id=session_id,
                        notebook_id=notebook_id
                    )
                    last_error = f"Fatal error managing MCP connection: {str(mcp_mgmt_err)}"
                    break # Exit attempt loop due to fatal MCP issue

                # --- End of single attempt logic ---
                if not attempt_failed: 
                    if current_attempt_success: # If this specific attempt had success
                         github_query_agent_logger.info(f"Query processing completed successfully on attempt {attempt_completed}.")
                         break # Exit loop on first successful attempt
                    else: # Attempt finished without error, but also no success
                         github_query_agent_logger.warning(f"Attempt {attempt_completed} finished without tool successes or critical errors. Last error note: {last_error}")
                         if attempt >= max_attempts - 1:
                             last_error = last_error or "Agent finished without calling successful tools after max attempts."
                             break # Exit loop after max attempts with no success
                         else:
                            # Optionally add context for retry? Or just let it try again.
                            error_context = f"\\n\\nINFO: Attempt {attempt_completed} completed without making progress. Retrying..."
                            current_description += error_context
                            yield StatusUpdateEvent( type=EventType.STATUS_UPDATE, status=StatusType.RETRYING, agent_type=AgentType.GITHUB, attempt=attempt_completed, reason="no_progress", message=None, max_attempts=max_attempts, step_id=None, original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id)
                            continue # Continue to next attempt if no success but also no failure

                # Safety break if loop didn't exit cleanly (shouldn't be needed with logic above)
                if attempt >= max_attempts - 1:
                    github_query_agent_logger.warning(f"Reached max attempts ({max_attempts}) safety break without explicit success.")
                    last_error = last_error or "Max attempts reached without success."
                    break

        except Exception as e:
            # Catch errors outside the main attempt loop (e.g., during initialization before loop)
            github_query_agent_logger.error(f"Fatal error during GitHub query processing (outside attempt loop): {e}", exc_info=True)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR,
                status=StatusType.FATAL_ERROR,
                agent_type=AgentType.GITHUB,
                error=f"Fatal query processing error: {str(e)}",
                session_id=session_id,
                notebook_id=notebook_id
            )
            last_error = f"Fatal query processing error: {str(e)}"
        finally:
            github_query_agent_logger.info("Exiting GitHubQueryAgent.run_query method.")

        # --- Final status reporting based on overall success or last error ---
        if success_occurred:
             github_query_agent_logger.info(f"GitHub query finished after {attempt_completed} attempts with at least one tool success.")
             yield FinalStatusEvent(
                 type=EventType.FINAL_STATUS,
                 status=StatusType.FINISHED_SUCCESS,
                 agent_type=AgentType.GITHUB,
                 attempts=attempt_completed,
                 message=None,
                 session_id=session_id,
                 notebook_id=notebook_id
             )
        elif last_error:
            final_error_msg = f"GitHub query finished after {attempt_completed} attempts. Last error: {last_error}"
            github_query_agent_logger.error(final_error_msg)
            yield ToolErrorEvent(
                type=EventType.TOOL_ERROR,
                status=StatusType.FINISHED_ERROR,
                agent_type=AgentType.GITHUB,
                error=final_error_msg,
                attempt=attempt_completed,
                tool_call_id=None,
                tool_name=None,
                tool_args=None,
                message=final_error_msg,
                original_plan_step_id=None,
                session_id=session_id,
                notebook_id=notebook_id
            )
        else:
            final_error_msg = f"GitHub query finished after {attempt_completed} attempts without success or explicit error."
            github_query_agent_logger.warning(final_error_msg)
            yield ToolErrorEvent(
                type=EventType.TOOL_ERROR,
                status=StatusType.FINISHED_NO_RESULT,
                agent_type=AgentType.GITHUB,
                error=final_error_msg,
                attempt=attempt_completed,
                tool_call_id=None,
                tool_name=None,
                tool_args=None,
                message=final_error_msg,
                original_plan_step_id=None,
                session_id=session_id,
                notebook_id=notebook_id
            )
