"""
Filesystem Connection Handler
"""

import logging
import os
import shlex
from typing import List, Optional, Any, Dict, Literal, Tuple
import asyncio

from pydantic import BaseModel, Field, validator, ValidationError

from .base import MCPConnectionHandler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from .registry import register_handler

logger = logging.getLogger(__name__)

# --- Pydantic Models for Filesystem Configuration --- 

class FileSystemConnectionBase(BaseModel):
    """Base model for Filesystem connection configuration."""
    allowed_directories: List[str] = Field(
        ..., 
        description="List of local directories the MCP server is allowed to access."
    )
    type: Literal['filesystem'] = "filesystem"

    @validator('allowed_directories')
    def validate_directories(cls, v):
        if not v:
            raise ValueError("At least one allowed directory must be provided.")
        # Basic syntactic check - further validation might be needed
        for directory in v:
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError("Allowed directories must be non-empty strings.")
            # Avoid validating existence/permissions here on the backend server
        return v

class FileSystemConnectionCreate(FileSystemConnectionBase):
    """Model for creating a Filesystem connection."""
    # Inherits all fields from Base (allowed_directories, type)
    pass

class FileSystemConnectionUpdate(BaseModel):
    """Model for updating a Filesystem connection. All fields optional."""
    allowed_directories: Optional[List[str]] = Field(
        None, 
        description="New list of local directories the MCP server is allowed to access."
    )
    type: Literal['filesystem'] = "filesystem" # Required for discrimination

    @validator('allowed_directories')
    def validate_optional_directories(cls, v):
        if v is not None:
            if not v: # If list is provided but empty
                raise ValueError("Allowed directories list cannot be empty if provided.")
            for directory in v:
                 if not isinstance(directory, str) or not directory.strip():
                    raise ValueError("Allowed directories must be non-empty strings.")
        return v

# We are skipping the Test model for now as requested
# Use the Base model for testing structure for now
# class FileSystemConnectionTest(FileSystemConnectionBase):
#     """Model for testing a Filesystem connection configuration."""
#     pass


# --- Filesystem Connection Handler --- 

class FileSystemConnectionHandler(MCPConnectionHandler):
    """Handles Filesystem MCP connections."""

    def __init__(self):
        # Base class likely doesn't take connection_type in __init__
        super().__init__() 
        self.connection_type = "filesystem"

    def get_connection_type(self) -> str:
        # Return the type string directly
        return self.connection_type

    def get_create_model(self) -> type[BaseModel]:
        """Return the Pydantic model for creating a Filesystem connection."""
        return FileSystemConnectionCreate

    def get_update_model(self) -> type[BaseModel]:
        """Return the Pydantic model for updating a Filesystem connection."""
        return FileSystemConnectionUpdate

    def get_test_model(self) -> Optional[type[BaseModel]]:
        """Return the Pydantic model for testing a Filesystem connection."""
        # Use the Base model for validation during testing
        return FileSystemConnectionBase

    def get_sensitive_fields(self) -> List[str]:
        """Return a list of sensitive fields to be redacted."""
        # Directory paths can be sensitive, but aren't typically redacted like secrets.
        # Consider implications if paths contain sensitive info in their names.
        return []

    async def prepare_config(self,
                               connection_id: Optional[str],
                               input_data: BaseModel,
                               existing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate and prepare the configuration for saving."""
        logger.debug(f"Preparing config for Filesystem connection (ID: {connection_id})")
        
        if existing_config is None:
            existing_config = {}

        # Validate input_data based on whether it's Create or Update
        if isinstance(input_data, (FileSystemConnectionCreate, FileSystemConnectionBase)): # Also handle FileSystemConnectionBase
            prepared_config = input_data.model_dump()
        elif isinstance(input_data, FileSystemConnectionUpdate):
            update_dict = input_data.model_dump(exclude_unset=True) 
            prepared_config = {**existing_config, **update_dict} 
            try:
                FileSystemConnectionBase(**prepared_config)
            except ValidationError as e:
                 logger.error(f"Validation error merging update for connection {connection_id}: {e}")
                 raise ValueError(f"Invalid configuration update: {e}")
        else:
            raise TypeError(f"Unsupported input data type: {type(input_data)}")

        # Remove execution_mode if it exists from old configs
        prepared_config.pop('execution_mode', None)

        logger.debug(f"Prepared config: {prepared_config}")
        return prepared_config

    async def test_connection(self, config_to_test: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Test the filesystem connection configuration.
        Checks if required fields are present. A more robust test could try to list tools.
        """
        # Remove execution_mode if present for validation
        config_to_validate = config_to_test.copy()
        config_to_validate.pop('execution_mode', None)
        
        logger.info(f"Testing Filesystem connection with config: {config_to_validate}")
        try:
            validated_config = FileSystemConnectionBase(**config_to_validate)

            if not validated_config.allowed_directories:
                 return False, "Allowed directories list cannot be empty."

            logger.info(f"Filesystem connection test successful (structural validation only) for config: {config_to_validate}")
            return True, "Configuration structure is valid. (Note: Does not test runtime execution)."

        except ValidationError as e:
            logger.error(f"Filesystem connection test failed validation: {e}")
            error_messages = '; '.join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            return False, f"Configuration validation failed: {error_messages}"
        except Exception as e:
            logger.error(f"Unexpected error during Filesystem connection test: {e}", exc_info=True)
            return False, f"An unexpected error occurred: {str(e)}"

    def get_stdio_params(self, config: Dict[str, Any]) -> StdioServerParameters:
        """Generate MCP server command and arguments based on config (NPX mode)."""
        logger.debug(f"Generating MCP stdio params for config (NPX mode): {config}")
        
        config_for_validation = config.copy()
        config_for_validation.pop('execution_mode', None)
        
        try:
            validated_config = FileSystemConnectionBase(**config_for_validation)
            if not validated_config.allowed_directories:
                raise ValueError("No allowed directories configured for Filesystem MCP.")
        except ValidationError as e:
            logger.error(f"Invalid configuration provided to get_stdio_params: {e}")
            raise ValueError(f"Invalid configuration for starting MCP: {e}")
        except KeyError as e:
            logger.error(f"Missing key '{e}' in configuration for get_stdio_params")
            raise ValueError(f"Missing required configuration key: {e}")

        command = "npx"
        server_script = "@modelcontextprotocol/server-filesystem"
        env: Dict[str, str] = os.environ.copy() 

        # For NPX mode, the server is often initialized with a single root path.
        # We'll use the first allowed_directory.
        # The server, when executing tools with absolute paths (e.g., read_file('/abs/path')),
        # should ideally validate those paths against all configured allowed_directories.
        server_root_path_config_value = validated_config.allowed_directories[0]
        
        if len(validated_config.allowed_directories) > 1:
            logger.warning(
                f"Multiple allowed_directories configured: {validated_config.allowed_directories}. "
                f"Using the first one ('{server_root_path_config_value}') as the primary path argument for the NPX server. "
                f"The NPX server itself must handle/validate operations against all configured allowed_directories for security."
            )

        actual_server_root_path = server_root_path_config_value
        logger.info(f"Using NPX server root path directly from configuration: '{actual_server_root_path}'")

        if not os.path.isabs(actual_server_root_path):
             logger.error(f"Configured root path '{actual_server_root_path}' for NPX server is not absolute. Original configured path: '{server_root_path_config_value}'.")
             raise ValueError(f"Root path for NPX server must be absolute. Configured path was '{actual_server_root_path}'.")
        
        # Ensure the resolved path actually exists for the server to start correctly.
        if not os.path.isdir(actual_server_root_path):
            logger.error(f"Resolved NPX server root path '{actual_server_root_path}' does not exist or is not a directory.")
            raise ValueError(f"NPX server root path '{actual_server_root_path}' must exist and be a directory.")

        args = [
            "-y",          # Assume 'yes' to any prompts from npx
            server_script,
            actual_server_root_path # The primary path argument for the server
            # If the server needs to know all allowed_directories, this would need to be passed differently,
            # e.g., via env vars or a temporary config file, if the server supports it.
            # For now, we pass only one main path, relying on the server's ability to handle
            # absolute paths in tool calls and validate them against its own full configuration.
        ]
        

        logger.info(f"Filesystem MCP NPX stdio params: command='{command}', args={args}")
        return StdioServerParameters(command=command, args=args, env=env) 

    # --- Abstract Method Implementations (or Placeholders) ---

    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any], db_config: Dict[str, Any], correlation_id: Optional[str] = None) -> Any:
        """Execute a filesystem tool call via the configured MCP server."""
        log_extra = {'correlation_id': correlation_id, 'connection_type': 'filesystem', 'tool_name': tool_name}
        logger.info(f"Executing Filesystem tool call: {tool_name}", extra=log_extra)

        try:
            stdio_params = self.get_stdio_params(db_config)
            # Avoid logging potentially sensitive args here if tool_args could contain paths etc.
            logger.info(f"Using Stdio params: Command='{stdio_params.command}', Args='{' '.join(stdio_params.args)}'", extra=log_extra)
        except ValueError as e:
            logger.error(f"Failed to get stdio params for Filesystem tool call: {e}", extra=log_extra)
            raise # Re-raise config error

        mcp_result = None
        try:
            # Connect to the MCP server (npx or docker)
            async with stdio_client(stdio_params) as (read, write):
                # Start an MCP client session
                async with ClientSession(read, write) as session:
                    init_timeout = 60.0 # Timeout for server process to start and initialize
                    try:
                        logger.info("Initializing MCP session for Filesystem...", extra=log_extra)
                        await asyncio.wait_for(session.initialize(), timeout=init_timeout)
                        logger.info("MCP session for Filesystem initialized.", extra=log_extra)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout ({init_timeout}s) initializing Filesystem MCP session.", extra=log_extra)
                        # Use ConnectionError for init failures
                        raise ConnectionError(f"Timeout initializing MCP session for Filesystem tool: {tool_name}")
                    except Exception as init_err:
                        logger.error(f"Error initializing Filesystem MCP session: {init_err}", exc_info=True, extra=log_extra)
                         # Use ConnectionError for init failures
                        raise ConnectionError(f"Failed to initialize MCP session for Filesystem tool {tool_name}: {init_err}") from init_err

                    call_timeout = 120.0 # Timeout for the actual tool call
                    try:
                        logger.info(f"Calling tool '{tool_name}' with args: {tool_args}", extra=log_extra)
                        mcp_result = await asyncio.wait_for(session.call_tool(tool_name, arguments=tool_args), timeout=call_timeout)
                        logger.info(f"Filesystem tool '{tool_name}' call successful. Result type: {type(mcp_result)}", extra=log_extra)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout ({call_timeout}s) calling Filesystem tool '{tool_name}'.", extra=log_extra)
                        # Raise McpError for tool call failures
                        raise McpError(ErrorData(code=500, message=f"Timeout calling Filesystem tool: {tool_name}"))
                    except McpError as tool_err: # Catch McpError from session.call_tool explicitly
                        logger.error(f"MCPError calling Filesystem tool '{tool_name}': {tool_err}", exc_info=True, extra=log_extra)
                        raise # Re-raise McpError as is
                    except Exception as call_err: # Catch other unexpected errors during call_tool
                        logger.error(f"Unexpected error calling Filesystem tool '{tool_name}': {call_err}", exc_info=True, extra=log_extra)
                        # Raise McpError for tool call failures
                        raise McpError(ErrorData(code=500, message=f"Failed to call Filesystem tool {tool_name}: {call_err}")) from call_err
            
            # Return the result obtained from the MCP server
            return mcp_result
            
        except McpError as e: # Catch McpErrors raised within this method or re-raised from session
            logger.error(f"MCPError during stdio_client execution for Filesystem tool '{tool_name}': {e}", extra=log_extra)
            raise # Re-raise the original McpError
        except ConnectionError as e: # Catch connection errors raised during init
            logger.error(f"ConnectionError during stdio_client execution for Filesystem tool '{tool_name}': {e}", extra=log_extra)
            raise # Re-raise the ConnectionError
        except FileNotFoundError as e:
             # Catch if the command (npx/docker) itself is not found
             logger.error(f"Command '{stdio_params.command}' not found when executing Filesystem tool '{tool_name}': {e}", extra=log_extra)
             raise McpError(ErrorData(code=500, message=f"Execution command '{stdio_params.command}' not found. Ensure it's installed and in PATH."))
        except Exception as e: # Catch other unexpected errors (e.g., stdio_client failure)
            error_msg = f"Unexpected Error executing Filesystem tool '{tool_name}' via stdio: {e}"
            logger.error(error_msg, exc_info=True, extra=log_extra)
            # Wrap unexpected errors in a standard Exception or McpError
            raise McpError(ErrorData(code=500, message=error_msg)) from e

# Register the handler instance
register_handler(FileSystemConnectionHandler())
