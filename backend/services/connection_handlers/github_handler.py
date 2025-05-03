"""
Handler for GitHub MCP Connections
"""

import logging
import asyncio
from typing import Type, Dict, Tuple, List, Optional, Any, Literal

from pydantic import BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from .base import MCPConnectionHandler
from .registry import register_handler

logger = logging.getLogger(__name__)

# Constants moved from ConnectionManager
GITHUB_MCP_COMMAND = ["docker", "run", "-i", "--rm"]
GITHUB_MCP_ARGS_TEMPLATE = [
    # PAT environment variable will be added dynamically during validation/agent execution
    "ghcr.io/github/github-mcp-server"
]
GITHUB_PAT_ENV_VAR = "GITHUB_PERSONAL_ACCESS_TOKEN"

# --- Pydantic Models --- 
# Mirroring structures from routes/connections.py for clarity

class GithubConnectionCreate(BaseModel):
    """GitHub connection creation parameters"""
    type: Literal["github"] = "github" # Added type discriminator with default
    # name: str # Name is handled by ConnectionManager
    github_personal_access_token: str = Field(..., description="GitHub Personal Access Token")

class GithubConnectionUpdate(BaseModel):
    """GitHub connection update parameters"""
    type: Literal["github"] = "github" # Added type discriminator with default
    # name: Optional[str] = None # Name is handled by ConnectionManager
    github_personal_access_token: Optional[str] = Field(None, description="GitHub Personal Access Token (optional)")

class GithubConnectionTest(BaseModel):
    """GitHub connection test parameters"""
    type: Literal["github"] = "github" # Added type discriminator with default
    github_personal_access_token: str = Field(..., description="GitHub Personal Access Token")


# --- Handler Implementation --- 

class GithubHandler(MCPConnectionHandler):
    """Handles GitHub MCP connections via Docker stdio."""

    def get_connection_type(self) -> str:
        return "github"

    def get_create_model(self) -> Type[BaseModel]:
        return GithubConnectionCreate

    def get_update_model(self) -> Type[BaseModel]:
        return GithubConnectionUpdate
    
    def get_test_model(self) -> Type[BaseModel]:
        return GithubConnectionTest

    async def prepare_config(self, connection_id: Optional[str], input_data: BaseModel, existing_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Prepares the GitHub connection config for DB storage.
        Stores the command, args template, and the PAT.
        Validates that a PAT is provided if creating or explicitly updating it.
        """
        github_pat = None

        if isinstance(input_data, (GithubConnectionCreate, GithubConnectionTest)):
            github_pat = input_data.github_personal_access_token
        elif isinstance(input_data, GithubConnectionUpdate):
            if input_data.github_personal_access_token:
                github_pat = input_data.github_personal_access_token
            elif existing_config:
                # If PAT is not provided in update, reuse the existing one
                github_pat = existing_config.get("github_pat")
        
        if not github_pat:
            # This condition should generally be caught by Pydantic validation or earlier checks
            # in ConnectionManager, but serves as a safeguard.
            raise ValueError("GitHub Personal Access Token is required for this operation.")

        if not isinstance(github_pat, str):
            raise ValueError("Invalid GitHub Personal Access Token format.")
        
        # Perform a quick validation test before saving config
        # We use the PAT directly provided or inferred
        is_valid, msg = await self._run_mcp_validation(github_pat)
        if not is_valid:
             raise ValueError(f"GitHub connection validation failed: {msg}")

        # Store PAT directly (NEEDS ENCRYPTION IN PRODUCTION!)
        logger.warning("Storing GitHub PAT directly in config. Needs encryption!")
        prepared_config = {
            "mcp_command": GITHUB_MCP_COMMAND,
            "mcp_args_template": GITHUB_MCP_ARGS_TEMPLATE,
            "github_pat": github_pat 
        }
        return prepared_config

    async def test_connection(self, config_to_test: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Tests a GitHub connection configuration by running a test MCP session.
        Expects config_to_test to contain 'github_personal_access_token' key for the test.
        """
        github_pat = config_to_test.get("github_personal_access_token")
        if not github_pat or not isinstance(github_pat, str):
            return False, "Missing or invalid GitHub Personal Access Token for testing."
        
        return await self._run_mcp_validation(github_pat)

    def get_stdio_params(self, db_config: Dict[str, Any]) -> StdioServerParameters:
        """
        Constructs StdioServerParameters for launching the GitHub MCP server.
        Injects the stored PAT via environment variables for Docker.
        """
        command_list = db_config.get("mcp_command")
        args_template = db_config.get("mcp_args_template", [])
        github_pat = db_config.get("github_pat")

        if not command_list or not isinstance(command_list, list) or not command_list:
            raise ValueError("Invalid or missing 'mcp_command' in GitHub config.")
        if not github_pat or not isinstance(github_pat, str):
             raise ValueError("Missing or invalid 'github_pat' in GitHub config.")

        command = command_list[0]
        base_args = command_list[1:]
        
        # Prepare args for Docker: docker run -i --rm -e GITHUB_PAT=... <image>
        dynamic_env_args = ["-e", f"{GITHUB_PAT_ENV_VAR}={github_pat}"]
        full_args = base_args + dynamic_env_args + args_template
        
        # Env dictionary is not used here as Docker injects via -e
        return StdioServerParameters(command=command, args=full_args, env={})

    def get_sensitive_fields(self) -> List[str]:
        """Returns keys containing sensitive information in the stored config."""
        return ["github_pat", "github_personal_access_token"]

    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any], db_config: Dict[str, Any], correlation_id: Optional[str] = None) -> Any:
        """
        Executes a GitHub tool call via the Docker stdio MCP server.
        """
        log_extra = {'correlation_id': correlation_id, 'connection_type': 'github', 'tool_name': tool_name}
        logger.info(f"Executing GitHub tool call: {tool_name}", extra=log_extra)

        try:
            stdio_params = self.get_stdio_params(db_config)
            logger.info(f"Using Stdio params: Command='{stdio_params.command}', Args='{stdio_params.args}'", extra=log_extra)
        except ValueError as e:
            logger.error(f"Failed to get stdio params for GitHub tool call: {e}", extra=log_extra)
            raise # Re-raise config error
        
        mcp_result = None
        try:
            async with stdio_client(stdio_params) as (read, write):
                async with ClientSession(read, write) as session:
                    init_timeout = 60.0 
                    try:
                        logger.info("Initializing MCP session...", extra=log_extra)
                        await asyncio.wait_for(session.initialize(), timeout=init_timeout)
                        logger.info("MCP session initialized.", extra=log_extra)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout ({init_timeout}s) initializing MCP session.", extra=log_extra)
                        raise ConnectionError(f"Timeout initializing MCP session for GitHub tool: {tool_name}")
                    except Exception as init_err:
                        logger.error(f"Error initializing MCP session: {init_err}", exc_info=True, extra=log_extra)
                        raise ConnectionError(f"Failed to initialize MCP session for GitHub tool {tool_name}: {init_err}") from init_err

                    call_timeout = 120.0 
                    try:
                        logger.info(f"Calling tool '{tool_name}' with args: {tool_args}", extra=log_extra)
                        mcp_result = await asyncio.wait_for(session.call_tool(tool_name, arguments=tool_args), timeout=call_timeout)
                        logger.info(f"Tool '{tool_name}' call successful. Result type: {type(mcp_result)}", extra=log_extra)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout ({call_timeout}s) calling tool '{tool_name}'.", extra=log_extra)
                        # Raise McpError with ErrorData structure
                        raise McpError(ErrorData(code=500, message=f"Timeout calling GitHub tool: {tool_name}"))
                    except McpError as tool_err: # Catch McpError from session.call_tool explicitly
                        logger.error(f"MCPError calling tool '{tool_name}': {tool_err}", exc_info=True, extra=log_extra)
                        raise # Re-raise McpError as is (assuming it contains ErrorData)
                    except Exception as call_err: # Catch other unexpected errors during call_tool
                        logger.error(f"Unexpected error calling tool '{tool_name}': {call_err}", exc_info=True, extra=log_extra)
                        # Raise McpError with ErrorData structure
                        raise McpError(ErrorData(code=500, message=f"Failed to call GitHub tool {tool_name}: {call_err}")) from call_err
            return mcp_result
        except McpError as e: # Catch McpErrors raised within this method (e.g., timeout) or re-raised from session
            logger.error(f"MCPError during stdio_client execution for tool '{tool_name}': {e}", extra=log_extra)
            raise # Re-raise the original McpError
        except ConnectionError as e: # Catch connection errors raised during init
            logger.error(f"ConnectionError during stdio_client execution for tool '{tool_name}': {e}", extra=log_extra)
            raise # Re-raise the ConnectionError
        except Exception as e: # Catch other unexpected errors (e.g., stdio_client failure)
            error_msg = f"Unexpected Error executing tool '{tool_name}' via stdio: {e}"
            logger.error(error_msg, exc_info=True, extra=log_extra)
            # Wrap unexpected errors in a standard Exception or a custom BackendError if available
            raise Exception(error_msg) from e 

    async def _run_mcp_validation(self, github_pat: str) -> Tuple[bool, str]:
        """Internal helper to run the actual MCP stdio validation logic."""
        correlation_id = 'github-validation' # Simple ID for validation logs
        logger.info("Performing GitHub stdio validation", extra={'correlation_id': correlation_id})

        # Construct arguments for docker run, including the -e flag for the PAT
        dynamic_env_arg = ["-e", f"{GITHUB_PAT_ENV_VAR}={github_pat}"]
        full_args = GITHUB_MCP_COMMAND[1:] + dynamic_env_arg + GITHUB_MCP_ARGS_TEMPLATE
        command = GITHUB_MCP_COMMAND[0]

        params = StdioServerParameters(command=command, args=full_args, env={})
        
        logger.info(f"Constructed StdioServerParameters for GitHub validation: Command='{params.command}', Args={' '.join(arg if GITHUB_PAT_ENV_VAR not in arg else arg.split('=')[0]+'=****' for arg in params.args)}", extra={'correlation_id': correlation_id})

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.info("Initializing MCP session for GitHub validation...", extra={'correlation_id': correlation_id})
                    init_task = asyncio.create_task(session.initialize())
                    try:
                        await asyncio.wait_for(init_task, timeout=45.0) # Increased timeout for docker potentially
                    except asyncio.TimeoutError:
                         logger.error("Timeout during GitHub MCP session initialization", extra={'correlation_id': correlation_id})
                         return False, "Validation failed: Timeout initializing GitHub MCP session. Check Docker, network, and PAT."
                    except Exception as init_err:
                        # Catch generic Exception and check message for MCP hints
                        err_str = str(init_err)
                        if "mcp" in err_str.lower(): # Basic check for MCP mention
                            err_msg = f"Validation failed: MCP-related Error initializing GitHub session: {err_str}" 
                        else:
                            err_msg = f"Validation failed: Error initializing GitHub MCP session: {err_str}"
                        logger.error(f"Error during GitHub MCP session initialization: {init_err}", extra={'correlation_id': correlation_id})
                        return False, err_msg

                    logger.info("Listing tools via MCP session for GitHub validation...", extra={'correlation_id': correlation_id})
                    list_tools_task = asyncio.create_task(session.list_tools())
                    try:
                         tools = await asyncio.wait_for(list_tools_task, timeout=20.0) 
                    except asyncio.TimeoutError:
                         logger.error("Timeout during GitHub MCP list_tools call", extra={'correlation_id': correlation_id})
                         return False, "Validation failed: Timeout listing tools via GitHub MCP. Server connected but unresponsive."
                    except Exception as list_err:
                        # Catch generic Exception and check message for MCP hints
                        err_str = str(list_err)
                        if "mcp" in err_str.lower(): # Basic check for MCP mention
                            err_msg = f"Validation failed: MCP-related Error listing tools: {err_str}"
                        else:
                            err_msg = f"Validation failed: Error listing tools via GitHub MCP: {err_str}"
                        logger.error(f"Error during GitHub MCP list_tools call: {list_err}", extra={'correlation_id': correlation_id})
                        return False, err_msg

                    # Simple check: Did we get a response (even if tools list is empty)?
                    if tools is not None: # Check for successful communication
                        num_tools = len(tools.tools) if hasattr(tools, 'tools') and isinstance(tools.tools, list) else 0
                        logger.info(f"GitHub stdio validation successful. Found {num_tools} tools.", extra={'correlation_id': correlation_id})
                        return True, "GitHub stdio connection validated successfully."
                    else:
                        # Should ideally not happen if list_tools didn't raise exception
                        logger.warning("GitHub stdio validation potentially failed: list_tools returned None.", extra={'correlation_id': correlation_id})
                        return False, "Validation failed: MCP server connected but list_tools returned an unexpected response." 

        except FileNotFoundError:
             logger.error(f"Validation failed: Command '{command}' not found. Is Docker installed and in PATH?", extra={'correlation_id': correlation_id})
             return False, f"Validation failed: Required command '{command}' not found. Ensure Docker is installed and in PATH."
        except Exception as e:
            logger.error(f"Error during GitHub stdio validation process startup or communication: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
            return False, f"Validation failed due to system error starting Docker: {str(e)}"
        
        # Fallback should ideally not be reached
        return False, "Unknown validation error for GitHub stdio."


# --- Registration --- 

# Register an instance of the handler
register_handler(GithubHandler()) 