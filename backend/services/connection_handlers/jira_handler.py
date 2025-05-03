"""
Handler for Jira MCP Connections (via mcp-atlassian)
"""

import logging
import asyncio
from typing import Type, Dict, Tuple, List, Optional, Any, Literal

from pydantic import BaseModel, Field, model_validator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from .base import MCPConnectionHandler
from .registry import register_handler

logger = logging.getLogger(__name__)

# Constants moved from ConnectionManager
JIRA_MCP_IMAGE = "ghcr.io/sooperset/mcp-atlassian:latest"
JIRA_MCP_COMMAND = ["docker", "run", "-i", "--rm"]
JIRA_MCP_ARGS_TEMPLATE = [JIRA_MCP_IMAGE]

# --- Standalone Validator Function --- 
def _check_jira_auth_fields(data: Any) -> Any:
    """Shared validation logic for Jira create/test models."""
    # Check type for safety, assumes Pydantic passes the model instance
    # We need to check against the expected model types
    if isinstance(data, (JiraConnectionCreate, JiraConnectionTest)): 
        if data.jira_auth_type == "cloud" and (not data.jira_username or not data.jira_api_token):
            raise ValueError("Jira Cloud requires jira_username and jira_api_token")
        if data.jira_auth_type == "server" and not data.jira_personal_token:
            raise ValueError("Jira Server/DC requires jira_personal_token")
    # It might be better practice to let Pydantic handle the type and raise if validation fails
    # on an unexpected type, rather than returning potentially invalid `data`.
    # However, this matches the original structure.
    return data


# --- Pydantic Models --- 

class JiraConnectionCreate(BaseModel):
    """Jira connection creation parameters"""
    type: Literal["jira"] = "jira"
    jira_url: str
    jira_auth_type: Literal["cloud", "server"]
    jira_ssl_verify: bool = True
    
    # Optional fields based on auth_type
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    jira_personal_token: Optional[str] = None

    # Optional MCP server config
    enabled_tools: Optional[str] = None
    read_only_mode: Optional[bool] = None
    jira_projects_filter: Optional[str] = None

    # Apply the standalone validator
    _validate_auth = model_validator(mode='after')(_check_jira_auth_fields)

class JiraConnectionUpdate(BaseModel):
    """Jira connection update parameters"""
    type: Literal["jira"] = "jira"
    # name: Optional[str] = None # Handled by ConnectionManager
    jira_url: Optional[str] = None
    jira_auth_type: Optional[Literal["cloud", "server"]] = None
    jira_ssl_verify: Optional[bool] = None
    
    # Optional fields - allow clearing/updating
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    jira_personal_token: Optional[str] = None

    # Optional MCP server config - allow clearing/updating
    enabled_tools: Optional[str] = None
    read_only_mode: Optional[bool] = None
    jira_projects_filter: Optional[str] = None

    # Use the standalone validator
    _validate_auth = model_validator(mode='after')(_check_jira_auth_fields)

class JiraConnectionTest(BaseModel):
    """Jira connection test parameters"""
    type: Literal["jira"] = "jira"
    jira_url: str
    jira_auth_type: Literal["cloud", "server"]
    jira_ssl_verify: bool = True
    
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    jira_personal_token: Optional[str] = None

    enabled_tools: Optional[str] = None
    read_only_mode: Optional[bool] = None
    jira_projects_filter: Optional[str] = None

    _validate_auth = model_validator(mode='after')(_check_jira_auth_fields)


# --- Handler Implementation --- 

class JiraHandler(MCPConnectionHandler):
    """Handles Jira MCP connections via mcp-atlassian Docker image."""

    def get_connection_type(self) -> str:
        return "jira"

    def get_create_model(self) -> Type[BaseModel]:
        return JiraConnectionCreate

    def get_update_model(self) -> Type[BaseModel]:
        return JiraConnectionUpdate
    
    def get_test_model(self) -> Type[BaseModel]:
        return JiraConnectionTest

    async def prepare_config(self, connection_id: Optional[str], input_data: BaseModel, existing_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Prepares the Jira connection config for DB storage.
        Merges input data with existing config for updates.
        Validates the resulting configuration.
        """
        if not isinstance(input_data, (JiraConnectionCreate, JiraConnectionUpdate)):
             raise TypeError("Input data must be JiraConnectionCreate or JiraConnectionUpdate")

        # Start with existing config if updating, otherwise empty dict
        current_config = existing_config.copy() if existing_config else {}

        # Update current_config with non-None values from input_data
        update_values = input_data.model_dump(exclude_unset=True)
        current_config.update(update_values)
        
        # --- Extract final values after merge ---
        jira_url = current_config.get("jira_url")
        jira_auth_type = current_config.get("jira_auth_type")
        jira_ssl_verify = current_config.get("jira_ssl_verify", True) # Default if missing
        jira_username = current_config.get("jira_username")
        jira_api_token = current_config.get("jira_api_token")
        jira_personal_token = current_config.get("jira_personal_token")
        enabled_tools = current_config.get("enabled_tools")
        read_only_mode = current_config.get("read_only_mode")
        jira_projects_filter = current_config.get("jira_projects_filter")
        
        # --- Validation of the final merged configuration --- 
        if not jira_url:
            raise ValueError("Jira URL is required.")
        if not jira_auth_type:
             raise ValueError("Jira Authentication Type is required.")
        if jira_auth_type not in ["cloud", "server"]:
             raise ValueError("Invalid Jira Authentication Type.")
        
        # Check required fields based on final auth_type
        if jira_auth_type == "cloud":
            if not jira_username or not jira_api_token:
                 raise ValueError("Jira Cloud requires Username and API Token.")
            # Clear server token if switching to cloud
            current_config["jira_personal_token"] = None 
        elif jira_auth_type == "server":
             if not jira_personal_token:
                 raise ValueError("Jira Server/DC requires Personal Access Token.")
             # Clear cloud credentials if switching to server
             current_config["jira_username"] = None
             current_config["jira_api_token"] = None

        # Run MCP validation using the final merged & validated parameters
        is_valid, msg = await self._run_mcp_validation(
            jira_url=jira_url,
            jira_auth_type=jira_auth_type,
            jira_ssl_verify=jira_ssl_verify,
            jira_username=jira_username,
            jira_api_token=jira_api_token,
            jira_personal_token=jira_personal_token,
            enabled_tools=enabled_tools,
            read_only_mode=read_only_mode,
            jira_projects_filter=jira_projects_filter
        )
        if not is_valid:
            raise ValueError(f"Jira connection validation failed: {msg}")

        # Prepare the final structure for DB storage
        logger.warning("Storing Jira tokens directly in config. Needs encryption!")
        prepared_config = {
            "mcp_command": JIRA_MCP_COMMAND,
            "mcp_args_template": JIRA_MCP_ARGS_TEMPLATE,
            # Store all relevant fields extracted above
            "jira_url": jira_url,
            "jira_auth_type": jira_auth_type,
            "jira_ssl_verify": jira_ssl_verify,
            "jira_username": jira_username,
            "jira_api_token": jira_api_token, # Sensitive
            "jira_personal_token": jira_personal_token, # Sensitive
            "enabled_tools": enabled_tools,
            "read_only_mode": read_only_mode,
            "jira_projects_filter": jira_projects_filter
        }
        return prepared_config

    async def test_connection(self, config_to_test: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Tests a Jira connection configuration by running a test MCP session.
        Expects config_to_test to contain the necessary Jira fields.
        """
        # Extract fields from the input dict
        try:
            return await self._run_mcp_validation(
                jira_url=config_to_test["jira_url"],
                jira_auth_type=config_to_test["jira_auth_type"],
                jira_ssl_verify=config_to_test.get("jira_ssl_verify", True),
                jira_username=config_to_test.get("jira_username"),
                jira_api_token=config_to_test.get("jira_api_token"),
                jira_personal_token=config_to_test.get("jira_personal_token"),
                enabled_tools=config_to_test.get("enabled_tools"),
                read_only_mode=config_to_test.get("read_only_mode"),
                jira_projects_filter=config_to_test.get("jira_projects_filter")
            )
        except KeyError as e:
            return False, f"Missing required field for testing: {e}"
        except Exception as e:
            logger.error(f"Unexpected error during Jira test setup: {e}", exc_info=True)
            return False, f"Unexpected error preparing test: {e}"

    def get_stdio_params(self, db_config: Dict[str, Any]) -> StdioServerParameters:
        """
        Constructs StdioServerParameters for launching the Jira MCP server.
        Injects stored config via environment variables for Docker.
        """
        command_list = db_config.get("mcp_command")
        args_template = db_config.get("mcp_args_template", [])
        
        if not command_list or not isinstance(command_list, list) or not command_list:
            raise ValueError("Invalid or missing 'mcp_command' in Jira config.")

        command = command_list[0]
        base_args = command_list[1:]
        
        # Prepare environment variables for Docker -e flags
        dynamic_env_args = []
        required_fields = ["jira_url", "jira_auth_type", "jira_ssl_verify"]
        optional_fields = ["jira_username", "jira_api_token", "jira_personal_token", 
                           "enabled_tools", "read_only_mode", "jira_projects_filter"]

        for field in required_fields:
            value = db_config.get(field)
            if value is None: # Should not happen if prepare_config worked
                raise ValueError(f"Missing required field '{field}' in Jira DB config.")
            env_var = field.upper()
            env_value = str(value).lower() if isinstance(value, bool) else str(value)
            dynamic_env_args.extend(["-e", f"{env_var}={env_value}"])

        for field in optional_fields:
             value = db_config.get(field)
             if value is not None:
                 env_var = field.upper()
                 env_value = str(value).lower() if isinstance(value, bool) else str(value)
                 dynamic_env_args.extend(["-e", f"{env_var}={env_value}"])
                 
        # Check auth consistency (should be guaranteed by prepare_config)
        auth_type = db_config.get("jira_auth_type")
        if auth_type == "cloud" and not (db_config.get("jira_username") and db_config.get("jira_api_token")):
            raise ValueError("Inconsistent cloud auth state in Jira DB config.")
        if auth_type == "server" and not db_config.get("jira_personal_token"):
             raise ValueError("Inconsistent server auth state in Jira DB config.")

        full_args = base_args + dynamic_env_args + args_template
        
        return StdioServerParameters(command=command, args=full_args, env={})

    def get_sensitive_fields(self) -> List[str]:
        """Returns keys containing sensitive information in the stored config."""
        return ["jira_api_token", "jira_personal_token"]

    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any], db_config: Dict[str, Any], correlation_id: Optional[str] = None) -> Any:
        """
        Executes a Jira tool call via the mcp-atlassian Docker stdio MCP server.
        """
        log_extra = {'correlation_id': correlation_id, 'connection_type': 'jira', 'tool_name': tool_name}
        logger.info(f"Executing Jira tool call: {tool_name}", extra=log_extra)

        try:
            stdio_params = self.get_stdio_params(db_config)
            # Redact args for logging sensitivity
            redacted_args = ' '.join(arg if 'TOKEN' not in arg.upper() and 'USERNAME' not in arg.upper() else arg.split('=')[0]+'=****' for arg in stdio_params.args)
            logger.info(f"Using Stdio params: Command='{stdio_params.command}', Args='{redacted_args}'", extra=log_extra)
        except ValueError as e:
            logger.error(f"Failed to get stdio params for Jira tool call: {e}", extra=log_extra)
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
                        raise ConnectionError(f"Timeout initializing MCP session for Jira tool: {tool_name}")
                    except Exception as init_err:
                        logger.error(f"Error initializing MCP session: {init_err}", exc_info=True, extra=log_extra)
                        raise ConnectionError(f"Failed to initialize MCP session for Jira tool {tool_name}: {init_err}") from init_err

                    call_timeout = 120.0 
                    try:
                        logger.info(f"Calling tool '{tool_name}' with args: {tool_args}", extra=log_extra)
                        mcp_result = await asyncio.wait_for(session.call_tool(tool_name, arguments=tool_args), timeout=call_timeout)
                        logger.info(f"Tool '{tool_name}' call successful. Result type: {type(mcp_result)}", extra=log_extra)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout ({call_timeout}s) calling tool '{tool_name}'.", extra=log_extra)
                        raise McpError(ErrorData(code=500, message=f"Timeout calling Jira tool: {tool_name}"))
                    except McpError as tool_err:
                        logger.error(f"MCPError calling tool '{tool_name}': {tool_err}", exc_info=True, extra=log_extra)
                        raise
                    except Exception as call_err:
                        logger.error(f"Unexpected error calling tool '{tool_name}': {call_err}", exc_info=True, extra=log_extra)
                        raise McpError(ErrorData(code=500, message=f"Failed to call Jira tool {tool_name}: {call_err}")) from call_err
            return mcp_result
        except McpError as e:
            logger.error(f"MCPError during stdio_client execution for tool '{tool_name}': {e}", extra=log_extra)
            raise
        except ConnectionError as e:
            logger.error(f"ConnectionError during stdio_client execution for tool '{tool_name}': {e}", extra=log_extra)
            raise
        except Exception as e:
            error_msg = f"Unexpected Error executing tool '{tool_name}' via stdio: {e}"
            logger.error(error_msg, exc_info=True, extra=log_extra)
            raise Exception(error_msg) from e

    async def _run_mcp_validation(
        self,
        jira_url: str,
        jira_auth_type: str,
        jira_ssl_verify: bool,
        jira_username: Optional[str] = None,
        jira_api_token: Optional[str] = None,
        jira_personal_token: Optional[str] = None,
        enabled_tools: Optional[str] = None,
        read_only_mode: Optional[bool] = None,
        jira_projects_filter: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Internal helper to run the actual MCP stdio validation logic for Jira."""
        correlation_id = 'jira-validation' # Simple ID for validation logs
        logger.info(f"Performing Jira stdio validation for URL: {jira_url}", extra={'correlation_id': correlation_id})

        # Construct Docker command with environment variables
        dynamic_env_args = []
        dynamic_env_args.extend(["-e", f"JIRA_URL={jira_url}"])
        dynamic_env_args.extend(["-e", f"JIRA_SSL_VERIFY={'true' if jira_ssl_verify else 'false'}"])

        # Authentication args
        if jira_auth_type == "cloud":
            if not jira_username or not jira_api_token:
                return False, "Internal validation error: Missing username or API token for cloud."
            dynamic_env_args.extend(["-e", f"JIRA_USERNAME={jira_username}"])
            dynamic_env_args.extend(["-e", f"JIRA_API_TOKEN={jira_api_token}"]) # Pass actual token
        elif jira_auth_type == "server":
            if not jira_personal_token:
                 return False, "Internal validation error: Missing personal token for server."
            dynamic_env_args.extend(["-e", f"JIRA_PERSONAL_TOKEN={jira_personal_token}"]) # Pass actual token
        else:
             return False, f"Internal validation error: Invalid auth type '{jira_auth_type}'"
        
        # Optional args
        if enabled_tools: dynamic_env_args.extend(["-e", f"ENABLED_TOOLS={enabled_tools}"])
        if read_only_mode is not None: dynamic_env_args.extend(["-e", f"READ_ONLY_MODE={'true' if read_only_mode else 'false'}"])
        if jira_projects_filter: dynamic_env_args.extend(["-e", f"JIRA_PROJECTS_FILTER={jira_projects_filter}"])
        
        full_args = JIRA_MCP_COMMAND[1:] + dynamic_env_args + JIRA_MCP_ARGS_TEMPLATE
        command = JIRA_MCP_COMMAND[0]
        params = StdioServerParameters(command=command, args=full_args, env={})

        logger.info(f"Constructed StdioServerParameters for Jira validation: Command='{params.command}', Args={' '.join(arg if 'TOKEN' not in arg.upper() and 'USERNAME' not in arg.upper() else arg.split('=')[0]+'=****' for arg in params.args)}", extra={'correlation_id': correlation_id})

        # Attempt connection and list_tools
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.info("Initializing MCP session for Jira validation...", extra={'correlation_id': correlation_id})
                    init_task = asyncio.create_task(session.initialize())
                    try:
                        await asyncio.wait_for(init_task, timeout=45.0)
                    except asyncio.TimeoutError:
                         logger.error("Timeout during Jira MCP session initialization", extra={'correlation_id': correlation_id})
                         return False, "Validation failed: Timeout initializing Jira MCP session. Check Docker, network, URL, and credentials."
                    except Exception as init_err:
                        err_str = str(init_err)
                        if "mcp" in err_str.lower():
                             err_msg = f"Validation failed: MCP-related Error initializing Jira session: {err_str}"
                        else:
                            err_msg = f"Validation failed: Error initializing Jira MCP session: {err_str}"
                        logger.error(f"Error during Jira MCP session initialization: {init_err}", extra={'correlation_id': correlation_id})
                        return False, err_msg

                    logger.info("Listing tools via MCP session for Jira validation...", extra={'correlation_id': correlation_id})
                    list_tools_task = asyncio.create_task(session.list_tools())
                    try:
                         tools = await asyncio.wait_for(list_tools_task, timeout=20.0)
                    except asyncio.TimeoutError:
                         logger.error("Timeout during Jira MCP list_tools call", extra={'correlation_id': correlation_id})
                         return False, "Validation failed: Timeout listing tools via Jira MCP. Server connected but unresponsive."
                    except Exception as list_err:
                        err_str = str(list_err)
                        if "mcp" in err_str.lower():
                            err_msg = f"Validation failed: MCP-related Error listing tools: {err_str}"
                        else:
                            err_msg = f"Validation failed: Error listing tools via Jira MCP: {err_str}"
                        logger.error(f"Error during Jira MCP list_tools call: {list_err}", extra={'correlation_id': correlation_id})
                        return False, err_msg

                    if tools is not None and hasattr(tools, 'tools'):
                        num_tools = len(tools.tools) if isinstance(tools.tools, list) else 0
                        logger.info(f"Jira stdio validation successful. Found {num_tools} tools.", extra={'correlation_id': correlation_id})
                        # Check if tools list is empty, which might indicate auth success but permission issues
                        if num_tools == 0:
                            logger.warning("Jira validation succeeded, but MCP reported 0 tools. Check Jira permissions or MCP config (filters/enabled_tools).", extra={'correlation_id': correlation_id})
                            return True, "Validation succeeded, but MCP reported 0 tools. Check Jira permissions or connection configuration (e.g., project filters)." # Still valid, but with warning
                        return True, "Jira stdio connection validated successfully."
                    else:
                        logger.warning("Jira stdio validation potentially failed: MCP server connected but reported no tools or invalid response.", extra={'correlation_id': correlation_id})
                        return False, "Validation failed: Jira MCP server connected but reported no tools or an invalid response. Check credentials, permissions, or server logs."

        except FileNotFoundError:
             logger.error(f"Validation failed: Command '{command}' not found. Is Docker installed and in PATH?", extra={'correlation_id': correlation_id})
             return False, f"Validation failed: Required command '{command}' not found. Ensure Docker is installed and in PATH."
        except Exception as e:
            logger.error(f"Error during Jira stdio validation process startup or communication: {e}", extra={'correlation_id': correlation_id}, exc_info=True)
            return False, f"Validation failed due to system error starting Docker: {str(e)}"

        return False, "Unknown validation error for Jira stdio."


# --- Registration --- 

# Register an instance of the handler
register_handler(JiraHandler()) 