"""
Python Agent Service

Agent for handling Python code execution via a locally running MCP server.
"""

import logging
from typing import  Optional, AsyncGenerator, Dict, Any, Union, List
from datetime import datetime, timezone
import json
import os
import re # Added for parsing
import uuid # For unique filenames
from uuid import UUID # Import UUID

from pydantic import UUID4 # To convert notebook_id string to UUID

from pydantic_ai import Agent, UnexpectedModelBehavior, CallToolsNode
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage, # For message_history
)
# from pydantic_ai.models.openai import OpenAIModel # Replaced with SafeOpenAIModel
from backend.ai.models import SafeOpenAIModel, FileDataRef, PythonAgentInput # Added for safer timestamp handling and new input models
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
# from mcp.shared.exceptions import McpError # Not directly used by PythonAgent for calling other MCPs now
from mcp import StdioServerParameters
from backend.core.cell import CellStatus # Added import
# Potential import if NotebookManager is injected - for type hinting
from backend.services.notebook_manager import NotebookManager # For type hinting
from sqlalchemy.ext.asyncio import AsyncSession # Added import

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
python_agent_logger = logging.getLogger("ai.python_agent")

# Define a base path for storing imported datasets. This could come from settings.
# IMPORTANT: Ensure this path is writable by the application.
# For Docker, this path is relative to the container's filesystem.
IMPORTED_DATA_BASE_PATH = "data/imported_datasets"


class PythonAgent:
    """Agent for interacting with Python MCP server using stdio."""
    def __init__(self, notebook_id: str, notebook_manager: Optional[NotebookManager] = None): # Added notebook_manager
        python_agent_logger.info(f"Initializing PythonAgent (notebook_id will be from input)")
        self.settings = get_settings()
        self.model = SafeOpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.agent = None # Will be initialized in run_query
        self.notebook_manager: Optional[NotebookManager] = notebook_manager # Assign passed manager
        python_agent_logger.info(f"PythonAgent class initialized (model configured, agent instance created per run).")

    async def _fetch_all_cells_data(self, notebook_id: str, db: AsyncSession) -> Optional[List[Dict[str, Any]]]: # db type AsyncSession
        """Fetches all cell data for a given notebook_id."""
        # This method assumes self.notebook_manager is available.
        # It also assumes 'db' (AsyncSession) is passed appropriately or available.
        if not self.notebook_manager: # Direct check now that it's an attribute
            python_agent_logger.error("NotebookManager not available in PythonAgent. Cannot fetch cell data.")
            return None
        try:
            notebook_uuid = UUID(notebook_id)
            python_agent_logger.info(f"Fetching notebook {notebook_uuid} to get all cells data.")
            # The 'db' session needs to be an AsyncSession instance here
            notebook = await self.notebook_manager.get_notebook(db, notebook_uuid)
            if notebook and notebook.cells:
                cells_data = [cell.model_dump(mode='json') for cell in notebook.cells.values()]
                python_agent_logger.info(f"Successfully fetched {len(cells_data)} cells for notebook {notebook_id}.")
                return cells_data
            else:
                python_agent_logger.warning(f"Notebook {notebook_id} not found or has no cells.")
                return None
        except ValueError: # For invalid UUID string
            python_agent_logger.error(f"Invalid notebook_id format: {notebook_id}. Cannot fetch cells.", exc_info=True)
            return None
        except Exception as e:
            python_agent_logger.error(f"Error fetching all cells data for notebook {notebook_id}: {e}", exc_info=True)
            return None

    async def _prepare_local_file(
        self, 
        data_ref: FileDataRef, 
        notebook_id: str, 
        session_id: str # Added session_id for better namespacing
    ) -> str:
        """
        Ensures the CSV data is available as a local file and returns its path.
        Currently assumes data_ref.type is 'content_string'.
        If data_ref.type is 'fsmcp_path', it's the responsibility of the
        StepProcessor to resolve it to content before calling PythonAgent, or
        if type is 'local_staged_path', the value is the already prepared path.
        """
        if data_ref.type == "local_staged_path":
            # Value is the path to the already staged file. Validate and return.
            local_path = data_ref.value
            if not os.path.exists(local_path):
                err_msg = f"Local staged file path provided does not exist: {local_path}"
                python_agent_logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            python_agent_logger.info(f"Using already staged local file: {local_path}")
            return local_path
        elif data_ref.type == "content_string":
            # Save the provided content to a new permanent file
            python_agent_logger.info(f"Received content_string. Staging data to permanent local file...")
            # Sanitize original_filename if provided, or use a generic name
            base_filename = "dataset"
        if data_ref.original_filename:
            # Basic sanitization: replace non-alphanumeric with underscore
            sanitized_name = re.sub(r'[^\w\.-]', '_', data_ref.original_filename)
            # Ensure it ends with .csv if it's a CSV, or add if no extension
            name_part, ext_part = os.path.splitext(sanitized_name)
            if not ext_part: # if no extension
                sanitized_name += ".csv" # Assume CSV for now
            elif ext_part.lower() != ".csv": # if extension is not csv
                 # This might be too aggressive if other file types are supported later
                python_agent_logger.warning(f"Original filename '{data_ref.original_filename}' does not end with .csv. Appending .csv to sanitized name '{name_part}'.")
                sanitized_name = name_part + ".csv"
            base_filename = sanitized_name

        # Create a unique filename to avoid collisions, incorporating notebook_id and session_id
        unique_id = str(uuid.uuid4().hex[:8])
        filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.csv"
        
        # Construct path: IMPORTED_DATA_BASE_PATH / notebook_id / session_id / filename
        # Using session_id for better isolation if multiple sessions run for the same notebook
        # Or use source_cell_id if available and makes sense for organization
        dir_path = os.path.join(IMPORTED_DATA_BASE_PATH, notebook_id, session_id)
        
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            python_agent_logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
            raise # Re-raise to be caught by run_query

        local_permanent_path = os.path.join(dir_path, filename)

        try:
            with open(local_permanent_path, 'w', encoding='utf-8') as f:
                f.write(data_ref.value) # data_ref.value is the CSV content string
            python_agent_logger.info(f"Successfully wrote data to permanent local file: {local_permanent_path} (source: {data_ref.original_filename or 'N/A'}, cell: {data_ref.source_cell_id or 'N/A'})")
            return local_permanent_path
        except IOError as e:
            python_agent_logger.error(f"Failed to write to local file {local_permanent_path}: {e}", exc_info=True)
            raise # Re-raise to be caught by run_query
        else:
            # Handle unsupported types
            err_msg = f"PythonAgent._prepare_local_file received unsupported data_ref type '{data_ref.type}'. Expected 'content_string' or 'local_staged_path'."
            python_agent_logger.error(err_msg)
            raise ValueError(err_msg)

    def _parse_python_mcp_output(self, raw_output: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parses the output from the Python MCP server (mcp-server-data-exploration)
        into a structured dictionary. Expects JSON or a dict.
        """
        if isinstance(raw_output, dict):
            # If it's already a dict, assume it's in the desired format or close to it.
            # Ensure basic keys are present.
            parsed_result = {
                "status": raw_output.get("status", "success" if raw_output.get("stdout") is not None else "error"),
                "stdout": raw_output.get("stdout"),
                "stderr": raw_output.get("stderr"),
                "return_value": raw_output.get("return_value"), # Or other relevant fields
                "dependencies": raw_output.get("dependencies"), # If provided by new server
                "error_details": raw_output.get("error_details")
            }
            if parsed_result["status"] == "error" and not parsed_result["error_details"] and parsed_result["stderr"]:
                 parsed_result["error_details"] = {
                    "type": "PythonExecutionError",
                    "message": str(parsed_result["stderr"]).splitlines()[-1] if parsed_result["stderr"] else "Unknown Python error",
                    "traceback": str(parsed_result["stderr"])
                }
            python_agent_logger.info(f"MCP output was already a dict: {parsed_result}")
            return parsed_result
        elif isinstance(raw_output, str):
            try:
                # Attempt to parse as JSON if it's a string
                data = json.loads(raw_output)
                if isinstance(data, dict):
                    # Recursively call to handle if the JSON string contained a dict
                    return self._parse_python_mcp_output(data)
                else:
                    # JSON, but not a dict - treat as stdout
                    return {"status": "success", "stdout": raw_output, "stderr": None, "return_value": None, "dependencies": None, "error_details": None}
            except json.JSONDecodeError:
                # Not JSON, check for known error patterns before treating as plain stdout
                python_agent_logger.warning(f"MCP output is a string but not JSON. Raw: {raw_output[:500]}")
                # Regex to capture (-XXXXX, "Error message")
                mcp_error_match = re.match(r"\s*\(\s*(-\d+)\s*,\s*\"(.*)\"\s*\)\s*", raw_output)
                if mcp_error_match:
                    error_code = mcp_error_match.group(1)
                    error_message = mcp_error_match.group(2)
                    python_agent_logger.error(f"Identified MCP-style error in string output: Code {error_code}, Message: {error_message}")
                    return {
                        "status": "error", 
                        "stdout": None, 
                        "stderr": error_message, 
                        "return_value": None, 
                        "dependencies": None, 
                        "error_details": {"type": "MCPExecutionError", "code": error_code, "message": error_message, "traceback": raw_output}
                    }
                else:
                    # No known error pattern, treat as plain string output (likely stdout)
                    python_agent_logger.info(f"MCP output is a non-JSON string without known error patterns. Treating as direct stdout: {raw_output[:200]}")
                    return {"status": "success", "stdout": raw_output, "stderr": None, "return_value": None, "dependencies": None, "error_details": None}
        else:
            python_agent_logger.warning(f"Unexpected MCP output type: {type(raw_output)}. Returning error structure.")
            return {"status": "error", "stderr": f"Invalid output type: {type(raw_output)}", "stdout": None, "return_value": None, "dependencies": None, "error_details": {"type": "ParsingError", "message": "Invalid/unexpected output type from MCP server"}}

    def _get_stdio_server_params(self) -> StdioServerParameters:
        """Returns the StdioServerParameters for the mcp-server-data-exploration."""
        env = os.environ.copy()
        server_params = StdioServerParameters(
            command='/root/.local/bin/uv',
            args=[
                '--directory',
                '/opt/mcp-server-data-exploration', # Path where we cloned it in Dockerfile.backend
                'run',
                'mcp-server-ds' # Changed to 'mcp-server-ds' based on documentation
            ],
            env=env
        )
        python_agent_logger.info(f"Using StdioServerParameters for mcp-server-data-exploration. Command: {server_params.command} Args: {server_params.args}")
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

    def _initialize_agent(self, stdio_server: MCPServerStdio, extra_tools: Optional[list] = None) -> Agent:
        """Initializes the Pydantic AI Agent with Python specific configuration.

        The caller can supply *extra_tools* (e.g. `list_cells` / `get_cell`) which
        will be registered with the underlying Agent instance.
        """
        system_prompt = self._read_system_prompt()
        agent = Agent(
            self.model,
            mcp_servers=[stdio_server],
            system_prompt=system_prompt,
            tools=extra_tools or [],  # type: ignore[arg-type]
        )
        python_agent_logger.info("Agent instance created with MCPServerStdio and %d extra tools.", len(extra_tools or []))
        return agent

    async def run_query(
        self,
        agent_input: PythonAgentInput,
        db: Optional[AsyncSession] = None  # Added db parameter, make it optional for now
    ) -> AsyncGenerator[Any, None]:
        # Extract necessary fields from agent_input
        notebook_id = agent_input.notebook_id
        session_id = agent_input.session_id
        user_query = agent_input.user_query
        # all_cells_data, max_context_cells, and context_summary_truncate_limit 
        # are no longer passed via agent_input.

        python_agent_logger.info(f"Running Python query for notebook {notebook_id}, session {session_id}. User query: '{user_query}'")
        
        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE, status=StatusType.STARTING, agent_type=AgentType.PYTHON,
            message="Initializing Python Agent and preparing data...", notebook_id=notebook_id, session_id=session_id,
            attempt=None, max_attempts=None, reason=None, step_id=None, original_plan_step_id=None
        )

        # Prepare local files from data_references
        local_file_paths_map: Dict[str, str] = {}
        if agent_input.data_references:
            python_agent_logger.info(f"Preparing {len(agent_input.data_references)} local files from data_references.")
            for i, data_ref in enumerate(agent_input.data_references):
                try:
                    # Use a generic name if original_filename is not available or to ensure uniqueness in map
                    map_key = data_ref.original_filename if data_ref.original_filename else f"dataset_{i+1}"
                    # Basic collision avoidance for map_key
                    temp_key = map_key
                    collision_count = 1
                    while temp_key in local_file_paths_map:
                        temp_key = f"{map_key}_{collision_count}"
                        collision_count += 1
                    map_key = temp_key

                    local_path = await self._prepare_local_file(data_ref, notebook_id, session_id)
                    local_file_paths_map[map_key] = local_path
                except Exception as prep_err:
                    error_msg = f"Failed to prepare local file for data_ref (original: {data_ref.original_filename}, type: {data_ref.type}): {prep_err}"
                    python_agent_logger.error(error_msg, exc_info=True)
                    yield FatalErrorEvent(
                        type=EventType.FATAL_ERROR, status=StatusType.FILE_PREPARATION_ERROR, agent_type=AgentType.PYTHON,
                        error=error_msg, notebook_id=notebook_id, session_id=session_id
                    )
                    return
        else:
            python_agent_logger.info("No data_references provided. Proceeding without pre-staged local files.")

        stdio_server = await self._get_stdio_server()
        if not stdio_server:
            error_msg = "Failed to configure Python MCP stdio server."
            python_agent_logger.error(error_msg)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR, status=StatusType.FATAL_ERROR, agent_type=AgentType.PYTHON,
                error=error_msg, notebook_id=notebook_id, session_id=session_id
            )
            return

        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE, status=StatusType.CONNECTION_READY, agent_type=AgentType.PYTHON,
            message="Python MCP server connection configured.", notebook_id=notebook_id, session_id=session_id,
            attempt=None, max_attempts=None, reason=None, step_id=None, original_plan_step_id=None
        )

        try:
            # Build notebook-aware retrieval tools (list_cells / get_cell)
            try:
                from backend.ai.notebook_context_tools import create_notebook_context_tools
                notebook_tools = create_notebook_context_tools(notebook_id, self.notebook_manager)
            except Exception as tool_err:
                python_agent_logger.error("Failed to create notebook context tools: %s", tool_err, exc_info=True)
                notebook_tools = []

            # Initialize the Pydantic AI Agent here, as it's per-run with MCP server and extra tools
            self.agent = self._initialize_agent(stdio_server, extra_tools=notebook_tools)
            yield StatusUpdateEvent(
                type=EventType.STATUS_UPDATE, status=StatusType.AGENT_CREATED, agent_type=AgentType.PYTHON,
                message="Pydantic AI agent instance created.", notebook_id=notebook_id, session_id=session_id,
                attempt=None, max_attempts=None, reason=None, step_id=None, original_plan_step_id=None
            )
        except Exception as init_err:
            error_msg = f"Failed to initialize Pydantic AI Agent: {init_err}"
            python_agent_logger.error(error_msg, exc_info=True)
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR, status=StatusType.FATAL_ERROR, agent_type=AgentType.PYTHON,
                error=error_msg, notebook_id=notebook_id, session_id=session_id
            )
            return

        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')

        data_info_parts = []
        if local_file_paths_map:
            data_info_parts.append("The following datasets have been prepared from previous steps and are available for you to load using their EXACT ABSOLUTE local paths with the `load-csv` tool:")
            for name, path in local_file_paths_map.items():
                # Ensure path is absolute for clarity in prompt
                abs_path = os.path.abspath(path) 
                data_info_parts.append(f"- Dataset '{name}': {abs_path}")
        else:
            data_info_parts.append("No datasets were explicitly passed from previous steps.")
        
        data_info_str = "\n".join(data_info_parts)

        # Enhanced Prompt Instructions
        current_description = (
            f"Current time is {current_time_utc}.\n"
            "You are a data analysis assistant. You have access to a Python execution environment via the `mcp-server-data-exploration` server with the following tools:\n"
            "1. `load_csv`: Loads a CSV file into a pandas DataFrame.\n"
            "   - `csv_path` (string, required): The **ABSOLUTE** path to the CSV file within the server's environment (e.g., '/app/data/imported_datasets/...').\n"
            "   - `df_name` (string, optional): Variable name for the loaded DataFrame (e.g., 'df1'). Defaults to df_1, df_2, etc.\n"
            "2. `run_script`: Executes a Python script.\n"
            "   - `script` (string, required): The Python script to execute. This script can operate on DataFrames loaded by `load_csv` using their assigned `df_name`.\n"
            
            f"\n--- Available Datasets ---\n{data_info_str}\n-------------------------\n\n"
            
            f"User Query for this Python cell: {user_query}\n\n"
            
            "Instructions:\\n"\
            "1. **Load Data:** If the User Query requires data from the 'Available Datasets' list, use the `load_csv` tool with the EXACT absolute path provided for that dataset. Assign a meaningful `df_name`.\\n"\
            "2. **Handle Other Paths:** If the User Query mentions loading a file path *not* listed in 'Available Datasets', **DO NOT GUESS THE PATH**. First, respond asking the user to provide the correct absolute path or to use a filesystem tool step to locate the file.\\n"\
            "3. **Execute Analysis:** Once data is loaded, use the `run_script` tool to write and execute Python code (using pandas, numpy, etc.) to address the User Query, operating on the loaded DataFrame(s).\\n"\
            "4. **Output Formatting:** Structure the output printed by your Python script within `run_script`:\\n"\
            "   - **DataFrames:** If showing a DataFrame `df`, print it using `print('--- DataFrame ---')`, then `print(df.to_markdown(index=False, numalign='left', stralign='left'))`, then `print('--- End DataFrame ---')`.\\n"\
            "   - **Plots:** If generating a plot (e.g., with matplotlib), save it to '/tmp/plot.png'. Then read the file, encode it to base64, import the 'base64' library, and print the result EXACTLY as: `print(f'<PLOT_BASE64>{base64.b64encode(plot_file.read()).decode()}</PLOT_BASE64>')`.\\n"\
            "   - **JSON:** If the result is a dict or list `d`, import 'json' and print it EXACTLY as: `print(f'<JSON_OUTPUT>{json.dumps(d)}</JSON_OUTPUT>')`.\\n"\
            "   - **Other Text:** Print any other textual results or summaries directly.\\n"\
            "5. **Clarity:** Ensure all necessary results are printed to stdout within the `run_script` tool."
        )
        
        max_attempts = 5
        last_error = None
        attempt_completed = 0
        success_occurred = False
        accumulated_message_history: list[ModelMessage] = []

        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.STARTING_ATTEMPTS,
            agent_type=AgentType.PYTHON,
            message=f"Attempting Python query (max {max_attempts} attempts)...",
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
                    agent_type=AgentType.PYTHON,
                    attempt=attempt_completed, max_attempts=max_attempts,
                    message=None, reason=None, step_id=None, original_plan_step_id=None,
                    session_id=session_id,
                    notebook_id=notebook_id
                )
                attempt_failed = False
                last_error_for_attempt = None
                agent_produced_final_result = False

                try:
                    # Ensure self.agent is used
                    async with self.agent.run_mcp_servers():
                        python_agent_logger.info(f"Attempt {attempt_completed}: MCP Server context entered.")
                        
                        python_agent_logger.info(f"Attempt {attempt_completed}: Calling agent.iter with history length {len(accumulated_message_history)} and description:\n{current_description[:500]}...")
                        async with self.agent.iter(current_description, message_history=accumulated_message_history) as agent_run:
                            yield StatusUpdateEvent(
                                type=EventType.STATUS_UPDATE, status=StatusType.MCP_CONNECTION_ACTIVE,
                                agent_type=AgentType.PYTHON, attempt=attempt_completed,
                            message="MCP server connection established and agent.iter active.",
                            notebook_id=notebook_id, session_id=session_id,
                            max_attempts=None, reason=None, step_id=None, original_plan_step_id=None
                        )

                        pending_tool_calls: Dict[str, Dict[str, Any]] = {}
                        yield StatusUpdateEvent(
                            type=EventType.STATUS_UPDATE, status=StatusType.AGENT_ITERATING,
                            agent_type=AgentType.PYTHON, attempt=attempt_completed,
                            message="Agent is processing Python request...",
                            notebook_id=notebook_id, session_id=session_id,
                            max_attempts=None, reason=None, step_id=None, original_plan_step_id=None
                        )

                        # This try-except-finally block is for errors during the agent's iteration and tool handling
                        try:
                            async for node in agent_run: # type: ignore
                                if isinstance(node, CallToolsNode):
                                    python_agent_logger.info(f"Attempt {attempt_completed}: Processing CallToolsNode, streaming events...")
                                    # Inner try for node.stream
                                    try:
                                        async with node.stream(agent_run.ctx) as handle_stream: # type: ignore
                                            async for event in handle_stream:
                                                if isinstance(event, FunctionToolCallEvent):
                                                    tool_call_id = event.part.tool_call_id
                                                    tool_name = getattr(event.part, 'tool_name', "UnknownTool")
                                                    tool_args = getattr(event.part, 'args', {})

                                                    if isinstance(tool_args, str):
                                                        try:
                                                            tool_args = json.loads(tool_args)
                                                        except json.JSONDecodeError:
                                                            if tool_name == 'run-script':
                                                                tool_args = {'script': tool_args}
                                                            else:
                                                                tool_args = {}
                                                    elif not isinstance(tool_args, dict):
                                                        tool_args = {}
                                                    
                                                    if tool_name == 'load-csv' and ('csv_path' not in tool_args or not isinstance(tool_args.get('csv_path'), str)):
                                                        python_agent_logger.error(f"Attempt {attempt_completed}: Invalid 'csv_path' for load-csv. Args: {tool_args}")
                                                    elif tool_name == 'run-script' and ('script' not in tool_args or not isinstance(tool_args.get('script'), str)):
                                                        python_agent_logger.error(f"Attempt {attempt_completed}: Invalid 'script' for run-script. Args: {tool_args}")

                                                    pending_tool_calls[tool_call_id] = {"tool_name": tool_name, "tool_args": tool_args}
                                                    yield ToolCallRequestedEvent(
                                                        type=EventType.TOOL_CALL_REQUESTED, status=StatusType.TOOL_CALL_REQUESTED,
                                                        agent_type=AgentType.PYTHON, attempt=attempt_completed,
                                                        tool_call_id=tool_call_id, tool_name=tool_name, tool_args=tool_args,
                                                        notebook_id=notebook_id, session_id=session_id, original_plan_step_id=None
                                                    )
                                                elif isinstance(event, FunctionToolResultEvent):
                                                    tool_call_id = event.tool_call_id
                                                    raw_tool_result = event.result.content
                                                    
                                                    if tool_call_id in pending_tool_calls:
                                                        call_info = pending_tool_calls.pop(tool_call_id)
                                                        python_agent_logger.info(f"Attempt {attempt_completed}: Parsing output for {call_info['tool_name']}. Raw: '{str(raw_tool_result)[:200]}...'")
                                                        
                                                        raw_tool_result_for_parsing: Union[str, Dict[str, Any]]
                                                        if not isinstance(raw_tool_result, (str, dict)):
                                                            python_agent_logger.warning(f"Tool result for {call_info['tool_name']} is not a string or dict ({type(raw_tool_result)}), converting to string.")
                                                            raw_tool_result_for_parsing = str(raw_tool_result)
                                                        else:
                                                            raw_tool_result_for_parsing = raw_tool_result
                                                        parsed_tool_result = self._parse_python_mcp_output(raw_tool_result_for_parsing) # Corrected variable
                                                        
                                                        yield ToolSuccessEvent(
                                                            type=EventType.TOOL_SUCCESS, status=StatusType.TOOL_SUCCESS,
                                                            agent_type=AgentType.PYTHON, attempt=attempt_completed,
                                                            tool_call_id=tool_call_id, tool_name=call_info["tool_name"],
                                                            tool_args=call_info["tool_args"], tool_result=parsed_tool_result,
                                                            notebook_id=notebook_id, session_id=session_id, original_plan_step_id=None
                                                        )
                                                        success_occurred = True
                                                    else:
                                                        python_agent_logger.warning(f"Attempt {attempt_completed}: Received result for unknown/processed tool_call_id: {tool_call_id}")
                                    except Exception as stream_err_inner: 
                                        python_agent_logger.error(f"Attempt {attempt_completed}: Error during CallToolsNode stream: {stream_err_inner}", exc_info=True)
                                        # Yield a general tool error, specific details might be lost from McpError if it was one
                                        yield ToolErrorEvent(
                                            type=EventType.TOOL_ERROR, status=StatusType.ERROR, agent_type=AgentType.PYTHON,
                                            attempt=attempt_completed, error=f"Stream Processing Error: {stream_err_inner}",
                                            notebook_id=notebook_id, session_id=session_id,
                                            tool_call_id=getattr(stream_err_inner, 'tool_call_id', None), # Try to get tool_call_id if McpError
                                            tool_name=None, tool_args=None, message=None, original_plan_step_id=None
                                        )
                                        # Re-raise to ensure the attempt is marked as failed if a stream error occurs
                                        # This will be caught by the outer except blocks of this `try` for agent_run
                                        raise stream_err_inner 

                            run_result = agent_run.result # type: ignore
                            python_agent_logger.info(f"Attempt {attempt_completed}: Agent iteration finished. Raw final result: {run_result}")
                            if not attempt_failed: 
                                agent_produced_final_result = True
                        
                        except UnexpectedModelBehavior as e_model_behavior: 
                            python_agent_logger.error(f"Attempt {attempt_completed} caught UnexpectedModelBehavior: {e_model_behavior}", exc_info=True)
                            last_error_for_attempt = f"Agent run failed due to unexpected model behavior: {str(e_model_behavior)}"
                            attempt_failed = True
                            yield ToolErrorEvent( # Corrected indentation
                                type=EventType.TOOL_ERROR, status=StatusType.MODEL_ERROR,
                                agent_type=AgentType.PYTHON, attempt=attempt_completed,
                                error=last_error_for_attempt,
                                tool_call_id=None, tool_name=None, tool_args=None, message=None,
                                original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                            )
                        except Exception as e_general_iter: # Specific exception variable
                            is_timestamp_error = isinstance(e_general_iter, TypeError) and "'NoneType' object cannot be interpreted as an integer" in str(e_general_iter)
                            error_msg = "TypeError: OpenAI API response likely missing 'created' timestamp." if is_timestamp_error else f"Agent iteration failed unexpectedly: {str(e_general_iter)}"
                            error_status = StatusType.TIMESTAMP_ERROR if is_timestamp_error else StatusType.GENERAL_ERROR_RUN
                            python_agent_logger.error(f"Attempt {attempt_completed} caught general error during agent.iter: {error_msg}", exc_info=True)
                            last_error_for_attempt = error_msg
                            attempt_failed = True
                            yield ToolErrorEvent( # Corrected indentation
                                type=EventType.TOOL_ERROR,
                                status=error_status,
                                agent_type=AgentType.PYTHON,
                                attempt=attempt_completed,
                                error=error_msg,
                                tool_call_id=None, tool_name=None, tool_args=None, message=None,
                                original_plan_step_id=None, session_id=session_id, notebook_id=notebook_id
                            )
                        finally: # This finally is for the inner try block for agent_run
                            if hasattr(agent_run, 'ctx') and hasattr(agent_run.ctx, 'state') and hasattr(agent_run.ctx.state, 'message_history'): # type: ignore
                                accumulated_message_history = agent_run.ctx.state.message_history[:] # type: ignore
                                python_agent_logger.info(f"Attempt {attempt_completed} (inner finally): Captured message history of length {len(accumulated_message_history)}.")
                            else:
                                python_agent_logger.warning(f"Attempt {attempt_completed} (inner finally): Could not capture message history from agent_run.ctx.state.")
                            python_agent_logger.info(f"Attempt {attempt_completed}: Finished processing agent.iter() try-except-finally block.")
                
                # This except is for errors with `async with self.agent.run_mcp_servers():` or `async with self.agent.iter(...)` context managers themselves
                except Exception as mcp_context_err:
                    python_agent_logger.error(f"Error with MCP server context or agent.iter context for attempt {attempt_completed}: {mcp_context_err}", exc_info=True)
                    last_error_for_attempt = f"MCP/Agent Context Error: {str(mcp_context_err)}" # This is a string
                    attempt_failed = True
                    # Yield FatalErrorEvent with the specific error from this context
                    yield FatalErrorEvent(
                        type=EventType.FATAL_ERROR,
                        status=StatusType.FATAL_MCP_ERROR, # Ensure this exists in StatusType
                        agent_type=AgentType.PYTHON,
                        error=last_error_for_attempt, # This is now guaranteed to be a string
                        session_id=session_id,
                        notebook_id=notebook_id
                    )
                    break # Exit attempt loop

                if agent_produced_final_result and not attempt_failed:
                    python_agent_logger.info(f"Agent completed successfully on attempt {attempt_completed}. Exiting loop.")
                    break
                elif attempt_failed:
                    last_error = last_error_for_attempt
                    if last_error_for_attempt and "TypeError: OpenAI API response likely missing 'created' timestamp." in last_error_for_attempt:
                        python_agent_logger.error(f"Persistent timestamp error encountered on attempt {attempt_completed}. Not retrying further. Error: {last_error_for_attempt}")
                        break
                    if attempt < max_attempts - 1:
                        error_context = f"\\n\\nINFO: Attempt {attempt_completed} failed with error: {last_error_for_attempt}. Retrying..."
                        current_description += error_context
                        yield StatusUpdateEvent(
                            type=EventType.STATUS_UPDATE, status=StatusType.RETRYING,
                            agent_type=AgentType.PYTHON,
                            attempt=attempt_completed,
                            reason=f"error: {str(last_error_for_attempt)[:100]}...",
                            message=None, max_attempts=max_attempts, step_id=None, original_plan_step_id=None,
                            session_id=session_id,
                            notebook_id=notebook_id
                        )
                        continue
                    else:
                        python_agent_logger.error(f"Error on final attempt {max_attempts}: {last_error_for_attempt}")
                        break
                elif attempt >= max_attempts - 1:
                    python_agent_logger.warning(f"Reached max attempts ({max_attempts}) without the agent explicitly completing successfully in the final attempt.")
                    last_error = last_error or "Max attempts reached without successful agent completion."
                    break
                else:
                    python_agent_logger.error(f"Attempt {attempt_completed}: Reached unexpected state in loop control. Breaking.")
                    last_error = last_error or "Unexpected loop state."
                    break

        except Exception as e:
            python_agent_logger.error(f"Fatal error during Python query processing (outside attempt loop): {e}", exc_info=True)
            last_error = f"Fatal query processing error: {str(e)}"
            yield FatalErrorEvent(
                type=EventType.FATAL_ERROR,
                status=StatusType.FATAL_ERROR,
                agent_type=AgentType.PYTHON,
                error=last_error,
                session_id=session_id,
                notebook_id=notebook_id
            )
        finally:
            python_agent_logger.info("Exiting PythonAgent.run_query method's main try/except/finally block.")

        if success_occurred and not last_error:
            python_agent_logger.info(f"Python query finished after {attempt_completed} attempts with tool success and no final error.")
            yield FinalStatusEvent(
                type=EventType.FINAL_STATUS,
                status=StatusType.FINISHED_SUCCESS,
                agent_type=AgentType.PYTHON,
                attempts=attempt_completed,
                message=None,
                session_id=session_id,
                notebook_id=notebook_id
            )
        elif last_error:
            final_status = StatusType.FINISHED_ERROR
            if "Fatal" in last_error:
                final_status = StatusType.FINISHED_ERROR # Or keep specific fatal status if needed
            final_error_msg = f"Python query finished after {attempt_completed} attempts. Last error: {last_error}"
            python_agent_logger.error(final_error_msg)
            yield FinalStatusEvent(
                type=EventType.FINAL_STATUS,
                status=final_status,
                agent_type=AgentType.PYTHON,
                attempts=attempt_completed,
                message=final_error_msg,
                session_id=session_id,
                notebook_id=notebook_id
            )
        else:
            final_error_msg = f"Python query finished after {attempt_completed} attempts without reported success or error."
            python_agent_logger.warning(final_error_msg)
            yield FinalStatusEvent(
                type=EventType.FINAL_STATUS,
                status=StatusType.FINISHED_NO_RESULT,
                agent_type=AgentType.PYTHON,
                attempts=attempt_completed,
                message=final_error_msg,
                session_id=session_id,
                notebook_id=notebook_id
            )
