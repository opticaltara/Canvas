"""
Python Agent Service

Agent for handling Python code execution via a locally running MCP server.
"""

import logging
from typing import  Optional, AsyncGenerator, Dict, Any, Union
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
from mcp import StdioServerParameters

from backend.config import get_settings
from backend.ai.events import (
    EventType,
    AgentType, # Changed from FILESYSTEM
    StatusType,
    StatusUpdateEvent,
    ToolCallRequestedEvent,
    ToolSuccessEvent,
    ToolErrorEvent,
    FinalStatusEvent,
    FatalErrorEvent,
)
# Removed Connection Manager/Handler imports as we'll use direct command

# Updated logger name
python_agent_logger = logging.getLogger("ai.python_agent")

class PythonAgent:
    """Agent for interacting with Python MCP server using stdio."""
    def __init__(self, notebook_id: str):
        python_agent_logger.info(f"Initializing PythonAgent for notebook_id: {notebook_id}")
        self.settings = get_settings()
        # Using the same AI model settings as other agents
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.notebook_id = notebook_id
        self.agent = None
        python_agent_logger.info(f"PythonAgent initialized successfully (Agent instance created later).")

    def _get_stdio_server_params(self) -> StdioServerParameters:
        """Returns the hardcoded StdioServerParameters for the Python MCP server."""
        # Directly define the parameters based on the user's reference
        server_params = StdioServerParameters(
            command='deno',
            args=[
                'run',
                '-A', # Use -A for simplicity, grants all permissions
                # '-N', # These seem less relevant if using -A
                # '-R=node_modules',
                # '-W=node_modules',
                # '--node-modules-dir=auto',
                'jsr:@pydantic/mcp-run-python',
                'stdio',
            ],
            env=os.environ.copy() # Pass current environment
        )
        python_agent_logger.info(f"Using StdioServerParameters. Command: {server_params.command} Args: {server_params.args}")
        return server_params

    async def _get_stdio_server(self) -> Optional[MCPServerStdio]:
        """Creates the MCPServerStdio instance using hardcoded parameters."""
        try:
            stdio_params = self._get_stdio_server_params()
            return MCPServerStdio(command=stdio_params.command, args=stdio_params.args, env=stdio_params.env)
        except Exception as e:
            python_agent_logger.error(f"Error creating MCPServerStdio instance: {e}", exc_info=True)
            return None

    def _read_system_prompt(self) -> str:
        """Reads the system prompt from the dedicated file."""
        # Updated prompt file path
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "python_agent_system_prompt.txt")
        try:
            with open(prompt_file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            python_agent_logger.error(f"System prompt file not found at: {prompt_file_path}")
            return "You are a helpful AI assistant interacting with a Python execution MCP server. Use the 'run_python_code' tool to execute Python code."
        except Exception as e:
            python_agent_logger.error(f"Error reading system prompt file {prompt_file_path}: {e}", exc_info=True)
            return "You are a helpful AI assistant interacting with a Python execution MCP server."

    def _initialize_agent(self, stdio_server: MCPServerStdio) -> Agent:
        """Initializes the Pydantic AI Agent with Python specific configuration."""
        system_prompt = self._read_system_prompt()

        agent = Agent(
            self.model,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt,
        )
        python_agent_logger.info(f"Agent instance created with MCPServerStdio.")
        return agent

    async def _run_single_attempt(
        self, 
        agent: Agent, 
        current_description: str, 
        attempt_num: int,
        session_id: str,
        notebook_id: str
    ) -> AsyncGenerator[
        Union[StatusUpdateEvent, ToolCallRequestedEvent, ToolSuccessEvent, ToolErrorEvent],
        None,
    ]:
        """Runs a single attempt of the agent iteration, yielding status updates and individual tool success/error dictionaries."""
        python_agent_logger.info(f"Starting agent iteration attempt {attempt_num}...")
        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.AGENT_ITERATING,
            agent_type=AgentType.PYTHON, # <-- Changed agent type
            attempt=attempt_num,
            message="Agent is processing Python request...",
            max_attempts=None,
            reason=None,
            step_id=None,
            original_plan_step_id=None,
            session_id=session_id,
            notebook_id=notebook_id
        )

        run_result = None
        
        async with agent.iter(current_description) as agent_run:
            pending_tool_calls: Dict[str, Dict[str, Any]] = {}
            async for node in agent_run:
                if isinstance(node, CallToolsNode):
                    python_agent_logger.info(f"Attempt {attempt_num}: Processing CallToolsNode, streaming events...")
                    try:
                        async with node.stream(agent_run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    tool_call_id = event.part.tool_call_id
                                    tool_name = getattr(event.part, 'tool_name', None)
                                    tool_args = getattr(event.part, 'args', {})
                                    if not tool_name:
                                        python_agent_logger.warning(f"Attempt {attempt_num}: ToolCallPart missing 'tool_name'. Part: {event.part!r}")
                                        tool_name = "UnknownTool"
                                    
                                    # Expect 'run_python_code' tool, args should contain 'python_code' string
                                    if tool_name == 'run_python_code':
                                        if isinstance(tool_args, str):
                                            try:
                                                # The argument itself might be a JSON string containing the python_code key
                                                parsed_args = json.loads(tool_args)
                                                if 'python_code' in parsed_args and isinstance(parsed_args['python_code'], str):
                                                    tool_args = parsed_args # Use the parsed dict
                                                else:
                                                    python_agent_logger.error(f"Attempt {attempt_num}: Parsed tool_args JSON missing 'python_code' string for run_python_code. Args: {tool_args}")
                                                    tool_args = {} # Reset on error
                                            except json.JSONDecodeError:
                                                # Assume the string IS the python code itself (less ideal)
                                                python_agent_logger.warning(f"Attempt {attempt_num}: tool_args for run_python_code is a string but not JSON. Assuming it's the code directly.")
                                                tool_args = {'python_code': tool_args}
                                        elif not isinstance(tool_args, dict) or 'python_code' not in tool_args or not isinstance(tool_args.get('python_code'), str):
                                            python_agent_logger.error(f"Attempt {attempt_num}: tool_args for run_python_code is not a dict with a 'python_code' string. Args: {tool_args}")
                                            tool_args = {} # Reset on error
                                    else:
                                        python_agent_logger.warning(f"Attempt {attempt_num}: Received unexpected tool call: {tool_name}. Args: {tool_args}")
                                        # Attempt to parse args anyway, just in case
                                        if isinstance(tool_args, str):
                                            try: tool_args = json.loads(tool_args)
                                            except json.JSONDecodeError: pass # Keep as string if not JSON
                                        if not isinstance(tool_args, dict): tool_args = {} # Default to empty dict
                                    
                                    pending_tool_calls[tool_call_id] = {
                                        "tool_name": tool_name, 
                                        "tool_args": tool_args
                                    }
                                    yield ToolCallRequestedEvent(
                                        type=EventType.TOOL_CALL_REQUESTED,
                                        status=StatusType.TOOL_CALL_REQUESTED,
                                        agent_type=AgentType.PYTHON, # <-- Changed agent type
                                        attempt=attempt_num,
                                        tool_call_id=tool_call_id,
                                        tool_name=tool_name,
                                        tool_args=tool_args,
                                        original_plan_step_id=None,
                                        session_id=session_id,
                                        notebook_id=notebook_id
                                    )
                                
                                elif isinstance(event, FunctionToolResultEvent):
                                    tool_call_id = event.tool_call_id
                                    # The result content from run_python_code is structured text/xml
                                    tool_result = event.result.content 
                                    if tool_call_id in pending_tool_calls:
                                        call_info = pending_tool_calls.pop(tool_call_id)
                                        args_data = call_info["tool_args"]
                                        
                                        yield ToolSuccessEvent(
                                            type=EventType.TOOL_SUCCESS,
                                            status=StatusType.TOOL_SUCCESS,
                                            agent_type=AgentType.PYTHON, # <-- Changed agent type
                                            attempt=attempt_num,
                                            tool_call_id=tool_call_id,
                                            tool_name=call_info["tool_name"],
                                            tool_args=args_data,
                                            tool_result=tool_result, # Pass the structured text result
                                            original_plan_step_id=None,
                                            session_id=session_id,
                                            notebook_id=notebook_id
                                        )
                                        python_agent_logger.info(f"Attempt {attempt_num}: Yielded success for tool: {call_info['tool_name']} (ID: {tool_call_id})")
                                    else:
                                        python_agent_logger.warning(f"Attempt {attempt_num}: Received result for unknown/processed ID: {tool_call_id}")
                    
                    except McpError as mcp_err:
                        error_str = str(mcp_err)
                        python_agent_logger.warning(f"MCPError during CallToolsNode stream in attempt {attempt_num}: {error_str}")
                        mcp_tool_call_id = getattr(mcp_err, 'tool_call_id', None)
                        failed_tool_info = pending_tool_calls.get(mcp_tool_call_id, {}) if mcp_tool_call_id else {}
                        yield ToolErrorEvent(
                            type=EventType.TOOL_ERROR,
                            status=StatusType.MCP_ERROR,
                            agent_type=AgentType.PYTHON, # <-- Changed agent type
                            attempt=attempt_num,
                            tool_call_id=mcp_tool_call_id,
                            tool_name=failed_tool_info.get("tool_name"),
                            tool_args=failed_tool_info.get("tool_args"),
                            error=f"MCP Tool Error: {error_str}",
                            message=None,
                            original_plan_step_id=None,
                            session_id=session_id,
                            notebook_id=notebook_id
                        )
                        raise mcp_err # Re-raise to potentially trigger retry

                    except Exception as stream_err:
                        python_agent_logger.error(f"Attempt {attempt_num}: Error during CallToolsNode stream: {stream_err}", exc_info=True)
                        yield ToolErrorEvent(
                            type=EventType.TOOL_ERROR,
                            status=StatusType.ERROR, # Use generic ERROR status
                            agent_type=AgentType.PYTHON, # <-- Changed agent type
                            attempt=attempt_num,
                            tool_call_id=None,
                            tool_name=None,
                            tool_args=None,
                            error=f"Stream Processing Error: {stream_err}",
                            message=None,
                            original_plan_step_id=None,
                            session_id=session_id,
                            notebook_id=notebook_id
                        )
                        raise stream_err # Re-raise to potentially trigger retry
            
            run_result = agent_run.result 
            python_agent_logger.info(f"Attempt {attempt_num}: Agent iteration finished. Raw final result (less relevant now): {run_result}")

        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.AGENT_RUN_COMPLETE, 
            agent_type=AgentType.PYTHON, # <-- Changed agent type
            attempt=attempt_num,
            message=None,
            max_attempts=None,
            reason=None,
            step_id=None,
            original_plan_step_id=None,
            session_id=session_id,
            notebook_id=notebook_id
        )

    async def run_query(
        self,
        description: str,
        session_id: str,
        notebook_id: str
    ) -> AsyncGenerator[Any, None]:
        """Run a query (natural language request) against the Python MCP server via stdio, yielding status updates and individual tool results/errors."""
        python_agent_logger.info(f"Running Python query. Original description: '{description}'")
        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.STARTING,
            agent_type=AgentType.PYTHON, # <-- Changed agent type
            message="Initializing Python MCP connection...",
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
            error_msg = "Failed to configure Python MCP stdio server."
            python_agent_logger.error(error_msg)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR,
                status=StatusType.FATAL_ERROR,
                agent_type=AgentType.PYTHON, # <-- Changed agent type
                error=error_msg,
                session_id=session_id,
                notebook_id=notebook_id
            )
            return
            
        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.CONNECTION_READY, 
            agent_type=AgentType.PYTHON, # <-- Changed agent type
            message="Python MCP connection ready.",
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
                agent_type=AgentType.PYTHON, # <-- Changed agent type
                message="Python agent ready.",
                attempt=None,
                max_attempts=None,
                reason=None,
                step_id=None,
                original_plan_step_id=None,
                session_id=session_id,
                notebook_id=notebook_id
            )
        except Exception as init_err:
            error_msg = f"Failed to initialize Python Agent: {init_err}"
            python_agent_logger.error(error_msg, exc_info=True)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR,
                status=StatusType.FATAL_ERROR,
                agent_type=AgentType.PYTHON, # <-- Changed agent type
                error=error_msg,
                session_id=session_id,
                notebook_id=notebook_id
            )
            return
            
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        # Include instructions about the tool in the base description
        base_description = f"Current time is {current_time_utc}. You have a tool 'run_python_code' that takes a single argument 'python_code' (a string of Python code). Use this tool to answer the user's query. Ensure the Python code prints results clearly to stdout or returns a value. User query: {description}"
        current_description = base_description
        
        max_attempts = 3 # Python execution might be less prone to transient errors? Set lower initially.
        last_error = None
        attempt_completed = 0
        success_occurred = False

        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.STARTING_ATTEMPTS, 
            agent_type=AgentType.PYTHON, # <-- Changed agent type
            message=f"Attempting Python execution (max {max_attempts} attempts)...",
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
                python_agent_logger.info(f"Python Query Attempt {attempt_completed}/{max_attempts}")
                yield StatusUpdateEvent(
                    type=EventType.STATUS_UPDATE, status=StatusType.ATTEMPT_START, 
                    agent_type=AgentType.PYTHON, # <-- Changed agent type
                    attempt=attempt_completed, max_attempts=max_attempts, 
                    message=None, reason=None, step_id=None, original_plan_step_id=None,
                    session_id=session_id,
                    notebook_id=notebook_id
                )
                attempt_failed = False
                current_attempt_success = False # Track success within this attempt

                try: 
                    # Add timeout for MCP server context manager if needed
                    # async with asyncio.timeout(60): # Example 60-second timeout
                    async with agent.run_mcp_servers():
                        python_agent_logger.info(f"Attempt {attempt_completed}: MCP Server connection active.")
                        yield StatusUpdateEvent(
                            type=EventType.STATUS_UPDATE, status=StatusType.MCP_CONNECTION_ACTIVE, 
                            agent_type=AgentType.PYTHON, # <-- Changed agent type
                            attempt=attempt_completed, 
                            message="Python MCP server connection established.",
                            max_attempts=None, reason=None, step_id=None, original_plan_step_id=None,
                            session_id=session_id,
                            notebook_id=notebook_id
                        )

                        try:
                            async for event_data in self._run_single_attempt(agent, current_description, attempt_completed, session_id, notebook_id):
                                yield event_data

                                if isinstance(event_data, ToolSuccessEvent): 
                                    current_attempt_success = True
                                    success_occurred = True 
                                elif isinstance(event_data, ToolErrorEvent):
                                    last_error = event_data.error
                                    # Check if the error suggests a code issue vs. system issue
                                    if "Traceback" in str(last_error) or "SyntaxError" in str(last_error) or "NameError" in str(last_error):
                                         python_agent_logger.warning(f"Attempt {attempt_completed} failed due to likely Python code error: {last_error}")
                                         # Don't necessarily fail the whole attempt immediately if it's a code error, let retry happen.
                                         attempt_failed = True # Mark as failed, but allow retry logic below
                                    else:
                                        attempt_failed = True # Mark general errors as failure
                        
                        except McpError as mcp_err:
                            error_str = str(mcp_err)
                            python_agent_logger.warning(f"Attempt {attempt_completed} caught MCPError within _run_single_attempt loop: {error_str}")
                            last_error = f"MCP Tool Error: {error_str}"
                            attempt_failed = True
                        except UnexpectedModelBehavior as e:
                            python_agent_logger.error(f"Attempt {attempt_completed} caught UnexpectedModelBehavior within _run_single_attempt loop: {e}", exc_info=True)
                            last_error = f"Agent run failed due to unexpected model behavior: {str(e)}"
                            attempt_failed = True
                        except Exception as e:
                            python_agent_logger.error(f"Attempt {attempt_completed} caught general error within _run_single_attempt loop: {e}", exc_info=True)
                            yield ToolErrorEvent( 
                                type=EventType.TOOL_ERROR,
                                status=StatusType.GENERAL_ERROR_RUN,
                                agent_type=AgentType.PYTHON, # <-- Changed agent type
                                attempt=attempt_completed,
                                error=f"Agent iteration failed unexpectedly: {str(e)}",
                                tool_call_id=None, tool_name=None, tool_args=None, message=None,
                                original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                            )
                            last_error = f"Agent iteration failed unexpectedly: {str(e)}"
                            attempt_failed = True
                        finally:
                             python_agent_logger.info(f"Attempt {attempt_completed}: Exiting core _run_single_attempt processing block (inside run_mcp_servers).")
                        
                        if attempt_failed:
                             if attempt < max_attempts - 1:
                                 error_context = f"\n\nINFO: Attempt {attempt_completed} failed with error: {last_error}. Please check the python code for errors and try again, or try a different approach."
                                 current_description += error_context
                                 yield StatusUpdateEvent(
                                     type=EventType.STATUS_UPDATE, status=StatusType.RETRYING, 
                                     agent_type=AgentType.PYTHON, # <-- Changed agent type
                                     attempt=attempt_completed, 
                                     reason=f"error: {str(last_error)[:100]}...", 
                                     message=None, max_attempts=max_attempts, step_id=None, original_plan_step_id=None,
                                     session_id=session_id,
                                     notebook_id=notebook_id
                                 )
                                 continue 
                             else:
                                 python_agent_logger.error(f"Error on final attempt {max_attempts}: {last_error}")
                                 break 

                except Exception as mcp_mgmt_err:
                    python_agent_logger.error(f"Fatal error managing MCP server for attempt {attempt_completed}: {mcp_mgmt_err}", exc_info=True)
                    yield FatalErrorEvent(
                        type=EventType.FATAL_ERROR,
                        status=StatusType.FATAL_MCP_ERROR,
                        agent_type=AgentType.PYTHON, # <-- Changed agent type
                        error=f"Fatal MCP management error: {str(mcp_mgmt_err)}",
                        session_id=session_id,
                        notebook_id=notebook_id
                    )
                    last_error = f"Fatal error managing MCP connection: {str(mcp_mgmt_err)}"
                    break 

                if not attempt_failed: 
                    if current_attempt_success: 
                         python_agent_logger.info(f"Python execution completed successfully on attempt {attempt_completed}.")
                         break 
                    else: 
                         python_agent_logger.warning(f"Attempt {attempt_completed} finished without tool successes or critical errors. Last error note: {last_error}")
                         if attempt >= max_attempts - 1:
                             last_error = last_error or "Agent finished without calling successful tools after max attempts."
                             break 
                         else:
                            error_context = f"\n\nINFO: Attempt {attempt_completed} completed without making progress. Please ensure you are using the 'run_python_code' tool correctly. Retrying..."
                            current_description += error_context
                            yield StatusUpdateEvent( 
                                type=EventType.STATUS_UPDATE, status=StatusType.RETRYING, 
                                agent_type=AgentType.PYTHON, # <-- Changed agent type
                                attempt=attempt_completed, reason="no_progress", 
                                message=None, max_attempts=max_attempts, step_id=None, 
                                original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                            )
                            continue 

                if attempt >= max_attempts - 1:
                    python_agent_logger.warning(f"Reached max attempts ({max_attempts}) safety break without explicit success.")
                    last_error = last_error or "Max attempts reached without success."
                    break

        except Exception as e:
            python_agent_logger.error(f"Fatal error during Python query processing (outside attempt loop): {e}", exc_info=True)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR,
                status=StatusType.FATAL_ERROR,
                agent_type=AgentType.PYTHON, # <-- Changed agent type
                error=f"Fatal query processing error: {str(e)}",
                session_id=session_id,
                notebook_id=notebook_id
            )
            last_error = f"Fatal query processing error: {str(e)}"
        finally:
            python_agent_logger.info("Exiting PythonAgent.run_query method.")

        if success_occurred:
             python_agent_logger.info(f"Python query finished after {attempt_completed} attempts with at least one tool success.")
             yield FinalStatusEvent(
                 type=EventType.FINAL_STATUS,
                 status=StatusType.FINISHED_SUCCESS,
                 agent_type=AgentType.PYTHON, # <-- Changed agent type
                 attempts=attempt_completed,
                 message=None,
                 session_id=session_id,
                 notebook_id=notebook_id
             )
        elif last_error:
            final_error_msg = f"Python query finished after {attempt_completed} attempts. Last error: {last_error}"
            python_agent_logger.error(final_error_msg)
            yield ToolErrorEvent(
                type=EventType.TOOL_ERROR,
                status=StatusType.FINISHED_ERROR,
                agent_type=AgentType.PYTHON, # <-- Changed agent type
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
            final_error_msg = f"Python query finished after {attempt_completed} attempts without success or explicit error."
            python_agent_logger.warning(final_error_msg)
            yield ToolErrorEvent(
                type=EventType.TOOL_ERROR,
                status=StatusType.FINISHED_NO_RESULT,
                agent_type=AgentType.PYTHON, # <-- Changed agent type
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