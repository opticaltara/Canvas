"""
Chat Agent Service

This module implements a chat agent service using Pydantic AI,
focused on interactive conversations with the AI that can create
and manage notebook cells.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator, Union
from datetime import datetime, timezone
from pathlib import Path
import redis.asyncio as redis
from redis.asyncio.client import Redis as AsyncRedis
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent
# OpenAIModel is no longer directly used, SafeOpenAIModel is used instead
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import (
    ModelMessage, 
    ModelRequest, 
    ModelResponse, 
    TextPart, 
    UserPromptPart
)

from backend.services.connection_handlers.registry import get_handler
from pydantic_ai.mcp import MCPServerStdio

from backend.config import get_settings
from backend.ai.chat_tools import NotebookCellTools
from backend.services.notebook_manager import NotebookManager
from backend.ai.agent import AIAgent
from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.ai.events import ( 
    AgentType, 
    ClarificationNeededEvent,
    PlanCreatedEvent,
    PlanCellCreatedEvent,
    PlanRevisedEvent,
    StepStartedEvent,
    StepCompletedEvent,
    StepErrorEvent,
    GitHubToolCellCreatedEvent,
    GitHubToolErrorEvent,
    SummaryStartedEvent,
    SummaryUpdateEvent,
    SummaryCellCreatedEvent,
    SummaryCellErrorEvent,
    InvestigationCompleteEvent,
    StatusUpdateEvent,
    BaseEvent,
    FileSystemToolCellCreatedEvent,
    PythonToolCellCreatedEvent,
    LogAIToolCellCreatedEvent,
    LogAIToolErrorEvent,
    StatusType,
    StepExecutionCompleteEvent # Added import
)
from backend.ai.models import SafeOpenAIModel # Import the custom model

# Initialize logger
chat_agent_logger = logging.getLogger("ai.chat_agent")

# Add correlation ID filter to the logger
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

chat_agent_logger.addFilter(CorrelationIdFilter())

class CellResponsePart(TextPart):
    """A specialized response part for cell data"""
    def __init__(self, cell_id: str, cell_params: Dict[str, Any], status_type: str, agent_type: str, result: Optional[Dict[str, Any]] = None):
        super().__init__(content="")  # Empty content since we're using custom fields
        self.cell_params = cell_params
        self.status_type = status_type
        self.agent_type = agent_type
        self.cell_id = cell_id
        self.result = result

    def model_dump(self) -> Dict[str, Any]:
        # Helper to recursively convert non-serializable types
        def _serialize_value(value):
            if isinstance(value, UUID):
                return str(value)
            elif isinstance(value, datetime):
                 # Example: Add handling for datetime if needed
                 return value.isoformat()
            elif isinstance(value, dict):
                # Recursively process dictionaries
                return {k: _serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                # Recursively process lists
                return [_serialize_value(item) for item in value]
            elif isinstance(value, BaseModel):
                 # Handle nested Pydantic models by dumping and then serializing
                 return _serialize_value(value.model_dump())
            # Add other type checks as necessary
            try:
                # Attempt to serialize other types, fail gracefully if needed
                json.dumps(value) 
                return value
            except TypeError:
                # Fallback for unserializable types
                chat_agent_logger.warning(f"Unserializable type {type(value)} encountered in model_dump, converting to string.")
                return str(value)

        # Apply the recursive serializer to relevant fields
        dump = {
            "type": "cell_response",
            "cell_id": _serialize_value(self.cell_id),
            "cell_params": _serialize_value(self.cell_params), 
            "status_type": _serialize_value(self.status_type), # Apply defensively
            "agent_type": _serialize_value(self.agent_type)    # Apply defensively
        }
        if self.result:
            # Apply to the result dictionary as well
            dump["result"] = _serialize_value(self.result)
            
        return dump

class StatusResponsePart(TextPart):
    """A specialized response part for status messages"""
    def __init__(self, content: str, agent_type: str):
        super().__init__(content=content)
        self.agent_type = agent_type

    def model_dump(self) -> Dict[str, Any]:
        # Similar helper for StatusResponsePart if needed, or reuse/import
        def _serialize_value(value):
            if isinstance(value, UUID):
                return str(value)
            elif isinstance(value, datetime):
                 return value.isoformat()
            # Simpler version for status: only needs basic types + UUID
            return value 

        return {
            "type": "status_response",
            "content": _serialize_value(self.content), # Apply defensively
            "agent_type": _serialize_value(self.agent_type)
        }

class ChatMessage(BaseModel):
    """Format of messages used in the API"""
    role: str = Field(description="Role: 'user' or 'model'")
    content: str = Field(description="Message content")
    timestamp: str = Field(description="Timestamp of the message")
    agent: Optional[str] = Field(description="Agent that generated the message: 'chat_agent' or 'ai_agent'", default=None)


def to_chat_message(message: ModelMessage, agent: Optional[str] = None) -> ChatMessage:
    """Convert a ModelMessage to a ChatMessage"""
    # Check if message.parts is not empty before accessing the first element
    if not message.parts:
        chat_agent_logger.warning("Received ModelMessage with empty parts list.")
        return ChatMessage(
            role="unknown",
            content=f"Empty message parts: {str(message)}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent
        )
        
    first_part = message.parts[0]
    
    if isinstance(message, ModelRequest) and isinstance(first_part, UserPromptPart):
        if isinstance(first_part.content, str):
            return ChatMessage(
                role="user",
                content=first_part.content,
                timestamp=first_part.timestamp.isoformat(),
                agent=agent
            )
    elif isinstance(message, ModelResponse):
        if isinstance(first_part, CellResponsePart):
            # For cell responses, create a structured message
            return ChatMessage(
                role="model",
                content=json.dumps(first_part.model_dump()),
                timestamp=message.timestamp.isoformat(),
                agent=first_part.agent_type
            )
        elif isinstance(first_part, StatusResponsePart):
            # For status responses, create a structured message
            return ChatMessage(
                role="model",
                content=json.dumps(first_part.model_dump()),
                timestamp=message.timestamp.isoformat(),
                agent=first_part.agent_type
            )
        elif isinstance(first_part, TextPart):
            return ChatMessage(
                role="model",
                content=first_part.content,
                timestamp=message.timestamp.isoformat(),
                agent=agent
            )
    
    # If we can't convert, use a default representation
    chat_agent_logger.info(f"Unable to convert message type: {type(message)}, using default representation.")
    return ChatMessage(
        role="unknown",
        content=str(message),
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent=agent
    )

class ChatRequest(BaseModel):
    """Request to send a message to the chat agent"""
    prompt: str = Field(description="User's message")
    session_id: Optional[str] = Field(description="Chat session ID", default=None)
    notebook_id: str = Field(description="Notebook ID to associate with chat")


class ChatAgentService:
    """
    Chat agent service that handles interactive conversations
    and coordinates with the AI agent for investigations.
    Instances are cached per session_id.
    """
    def __init__(
        self,
        notebook_manager: NotebookManager,
        connection_manager: Optional[ConnectionManager] = None, # Allow None initially
        redis_client: Optional[AsyncRedis] = None # Use imported class alias
    ):
        self.settings = get_settings()
        chat_agent_logger.info(f"Instantiating ChatAgentService (initialization pending)...")
        self.notebook_manager = notebook_manager
        # Store connection manager instance, get default if not provided
        self._connection_manager = connection_manager or get_connection_manager()
        # Store the redis client
        self.redis_client = redis_client

        # Initialize state variables to None, will be set in initialize()
        self.notebook_id: Optional[str] = None
        self.available_tools_info: Optional[str] = None
        self._available_data_source_types: Optional[List[str]] = None
        # self.chat_agent: Optional[Agent[None, ClarificationResult]] = None # Keep chat_agent commented out or remove if clarification stays removed
        self.ai_agent: Optional[AIAgent] = None
        self.cell_tools: Optional[NotebookCellTools] = None
        self.github_mcp_server = None
        self.filesystem_mcp_server = None

    async def initialize(self, notebook_id: str):
        """Asynchronously initializes the agent instance after creation."""
        if self.notebook_id is not None:
             chat_agent_logger.warning(f"Agent for notebook {self.notebook_id} already initialized. Ignoring re-initialization attempt.")
             return

        chat_agent_logger.info(f"Initializing ChatAgentService for notebook: {notebook_id}")
        self.notebook_id = notebook_id

        # Fetch tool/connection info
        await self._fetch_and_set_available_tools_info()

        # Setup Cell Tools
        if not self.cell_tools: # Initialize only if not already set (e.g. by constructor)
            self.cell_tools = NotebookCellTools(notebook_manager=self.notebook_manager)

        # Setup Clarification Agent (chat_agent) - Keep commented/removed
        # system_prompt = self._generate_system_prompt() 
        # chat_agent_logger.info(f"System prompt generated for chat agent (notebook: {self.notebook_id}).")

        # -----------------------------------------------------------
        # Attach notebook retrieval tools (list_cells / get_cell)
        # -----------------------------------------------------------
        try:
            from backend.ai.notebook_context_tools import create_notebook_context_tools
            notebook_tools = create_notebook_context_tools(self.notebook_id, self.notebook_manager)
            chat_agent_logger.info(f"ChatAgentService: Successfully created {len(notebook_tools)} notebook context tools for notebook {self.notebook_id}.")
        except Exception as tool_err:
            chat_agent_logger.error("Failed to create notebook context tools for chat_agent: %s", tool_err, exc_info=True)
            notebook_tools = []

        # chat_agent_logger.info(f"ChatAgentService: Clarification agent removed.")

        # Setup Investigation Agent (ai_agent)
        chat_agent_logger.info(f"Initializing AIAgent for notebook {self.notebook_id} with sources: {self._available_data_source_types}")
        self.ai_agent = await AIAgent.create( # Use the async factory method
            notebook_manager=self.notebook_manager,
            notebook_id=self.notebook_id,
            available_data_sources=self._available_data_source_types or [],
            connection_manager_instance=self._connection_manager # Pass the existing connection manager
        )

        chat_agent_logger.info(f"ChatAgentService for notebook {self.notebook_id} initialized successfully.")

    async def _fetch_and_set_available_tools_info(self):
        """Fetches connection types and sets the available_tools_info string."""
        if self.available_tools_info is None:
            try:
                connections = await self._connection_manager.get_all_connections()
                # Explicitly filter for dictionaries and string types, then sort
                valid_types = [
                    str(conn.get("type")) 
                    for conn in connections 
                    if isinstance(conn, dict) and isinstance(conn.get("type"), str)
                ]
                connection_based_types = sorted(list(set(valid_types)))
                
                # Always include built-in types
                builtin_types = ["markdown", "python", "log_ai"]
                all_available_types = sorted(list(set(connection_based_types + builtin_types)))
                
                self._available_data_source_types = all_available_types
                
                display_types = [t.title() for t in all_available_types]
                if display_types:
                    self.available_tools_info = f"You have access to the following data sources/capabilities: {', '.join(display_types)}."
                else:
                    # This case should ideally not happen if markdown and python are always included
                    self.available_tools_info = "No specific external data sources are currently connected, but built-in capabilities like Markdown and Python are available."
                
                chat_agent_logger.info(f"Determined available sources/capabilities. For prompt: {self.available_tools_info}. For AIAgent: {self._available_data_source_types}")
            except Exception as e:
                chat_agent_logger.error(f"Failed to fetch connection types: {e}", exc_info=True)
                self.available_tools_info = "Error fetching available data sources. Built-in capabilities like Markdown and Python should still be available."
                # Fallback to built-in types if connection fetching fails
                self._available_data_source_types = ["markdown", "python"]

    def _load_system_prompt_template(self) -> str:
        """Loads the system prompt template from the dedicated file."""
        try:
            prompt_path = Path(__file__).parent / "prompts" / "chat_agent_system_prompt.txt"
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            chat_agent_logger.error("System prompt file not found! Using default empty prompt.")
            return ""
        except Exception as e:
            chat_agent_logger.error(f"Error loading system prompt: {e}")
            return ""

    def _generate_system_prompt(self) -> str:
        """Generates the system prompt by formatting the loaded template."""
        tools_info = self.available_tools_info or "(Data source information is loading...)"
        try:
            return self._load_system_prompt_template().format(tools_info=tools_info)
        except KeyError as e:
            chat_agent_logger.error(f"Error formatting system prompt - missing key: {e}")
            # Fallback or modified prompt if formatting fails
            return self._load_system_prompt_template() # Or return a default prompt string
        except Exception as e:
            chat_agent_logger.error(f"Unexpected error formatting system prompt: {e}")
            return self._load_system_prompt_template() # Fallback

    # Moved _get_timestamp helper method into the class
    def _get_timestamp(self, cell_data: Dict[str, Any], field: str) -> datetime:
        ts_str = cell_data.get(field)
        if ts_str:
            try:
                # Handle 'Z' suffix for UTC explicitly
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(ts_str)
                # If datetime is naive, assume UTC
                if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                chat_agent_logger.warning(f"Could not parse timestamp: '{ts_str}' in _get_timestamp. Using min datetime.")
                return datetime.min.replace(tzinfo=timezone.utc) # Fallback for bad format
        return datetime.min.replace(tzinfo=timezone.utc) # Fallback for missing

    async def handle_message(
        self,
        prompt: str,
        session_id: str, # Still needed for logging/context maybe, but notebook_id is internal
        message_history: List[ModelMessage] = [],
    ) -> AsyncGenerator[Tuple[str, ModelResponse], None]:
        """
        Process a user message and stream status updates
        
        Args:
            prompt: The user's message
            session_id: The chat session ID (primarily for logging)
            message_history: Previous messages in the session
            
        Yields:
            Tuples of (event_type_str, response) as updates occur, where response
            contains appropriate CellResponsePart or StatusResponsePart.
            The first element of the tuple corresponds to the EventType enum value.
        """
        # Ensure agent is initialized before handling messages
        if not self.notebook_id or not self.ai_agent or not self.cell_tools:
             chat_agent_logger.error(f"Agent for session {session_id} accessed handle_message before initialization.")
             raise RuntimeError("ChatAgentService not initialized. Call initialize() first.")

        chat_agent_logger.info(f"Handling message in session {session_id} (Notebook: {self.notebook_id})")
        chat_agent_logger.info(f"Received prompt: '{prompt}'")
        chat_agent_logger.info(f"Received message_history length: {len(message_history)} for session {session_id}")
        if message_history:
            chat_agent_logger.info(f"Last message in history: {str(message_history[-1])}") 
        start_time = time.time()
        
        try:
            # --- Removed Clarification Check and Redis Logic ---

            # ---> Immediately yield a status update message <---
            initial_status_message = "Starting investigation..."
            chat_agent_logger.info(f"Sending initial status update for session {session_id}: '{initial_status_message}'")
            # Use StatusUpdateEvent or similar appropriate event type if available, otherwise use a generic status update
            # Using StatusUpdateEvent seems reasonable here
            initial_status_event = StatusUpdateEvent(
                message=initial_status_message,
                status=StatusType.STARTING, # Use StatusType.STARTING
                session_id=session_id,
                notebook_id=self.notebook_id,
                # Provide default values for other optional fields
                step_id=None,
                attempt=None,
                max_attempts=None,
                # Provide the missing required fields
                agent_type=AgentType.CHAT, # Use AgentType.CHAT
                reason=None,
                original_plan_step_id=None 
            )
            # Ensure message is a string
            content_str = initial_status_event.message if initial_status_event.message is not None else ""
            # Agent type is now guaranteed to be set in the event
            agent_type_val = initial_status_event.agent_type.value if initial_status_event.agent_type else AgentType.UNKNOWN.value # Fallback just in case
            response_part = StatusResponsePart(
                content=content_str, # Use guaranteed string
                agent_type=agent_type_val
            )
            response = ModelResponse(parts=[response_part], timestamp=datetime.now(timezone.utc))
            yield initial_status_event.type.value, response

            # ---> Proceed directly to investigation <---
            log_msg = f"Proceeding directly to investigation for session {session_id}."
            chat_agent_logger.info(log_msg)

            # Notebook context summary no longer pre-computed; agents can call
            # list_cells/get_cell tools on demand. Pass None to investigate.
            notebook_context_summary = None

            async for event in self.ai_agent.investigate(
                prompt, # Pass the potentially context-enhanced prompt
                session_id,
                notebook_id=self.notebook_id,
                message_history=message_history, # Pass history including the user's latest message
                cell_tools=self.cell_tools,
                notebook_context_summary=notebook_context_summary  # remains None
            ):
                # Now expect event model instances directly
                event_type_enum = getattr(event, 'type', None)
                if not event_type_enum or not hasattr(event_type_enum, 'value'):
                    chat_agent_logger.warning(f"Event object missing 'type' attribute or type enum has no value: {event!r}. Skipping.")
                    continue # Skip if we can't get a valid type string

                event_type_str = event_type_enum.value # Guaranteed to be a string here
                agent_type_attr = getattr(event, 'agent_type', None)
                agent_type_str = agent_type_attr.value if agent_type_attr else AgentType.UNKNOWN.value
                chat_agent_logger.info(f"Yielding event update for session {session_id}. Type: {event_type_str}")
                chat_agent_logger.debug(f"Received event object: {event!r} of type {type(event)}") # Added type logging

                # Detailed logging for StepExecutionCompleteEvent before match
                if isinstance(event, StepExecutionCompleteEvent):
                    chat_agent_logger.info(f"Pre-match check: Event IS StepExecutionCompleteEvent. step_id: {getattr(event, 'step_id', 'N/A')}")

                response_part: Optional[Union[CellResponsePart, StatusResponsePart]] = None # Initialize to None
                
                match event:
                    case PlanCellCreatedEvent(cell_id=cid, cell_params=cp, status=st):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str}")
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value, 
                            agent_type=agent_type_str,
                            result=None 
                        )
                    case StepCompletedEvent(cell_id=cid, cell_params=cp, status=st, result=r):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str}")
                        result_dict = r.model_dump() if isinstance(r, BaseModel) else {"content": str(r)}
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value,
                            agent_type=agent_type_str,
                            result=result_dict
                        )
                    case StepErrorEvent(step_id=event_step_id, agent_type=event_agent_type, error=event_error, session_id=event_session_id, notebook_id=event_notebook_id):
                        # session_id and notebook_id are captured for potential logging or future use, though not directly in this StatusResponsePart
                        chat_agent_logger.info(f"Creating StatusResponsePart for StepErrorEvent (Step: {event_step_id}, Session: {event_session_id}, Notebook: {event_notebook_id})")
                        error_message = f"Error in step {event_step_id} ({event_agent_type.value if event_agent_type else 'Unknown Agent'}): {event_error}"
                        response_part = StatusResponsePart(
                            content=error_message, 
                            agent_type=event_agent_type.value if event_agent_type else AgentType.UNKNOWN.value
                        )
                    case GitHubToolCellCreatedEvent(cell_id=cid, cell_params=cp, status=st, result=r, tool_name=tn):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str}")
                        result_dict = {"content": r, "tool_name": tn}
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value,
                            agent_type=agent_type_str,
                            result=result_dict
                        )
                    case FileSystemToolCellCreatedEvent(cell_id=cid, cell_params=cp, status=st, result=r, tool_name=tn):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str}")
                        # Assuming result `r` might be complex, try model_dump or str()
                        result_content = None
                        if isinstance(r, BaseModel):
                            result_content = r.model_dump()
                        elif r is not None:
                            result_content = str(r)
                            
                        result_dict = {"content": result_content, "tool_name": tn}
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value, # Should be StatusType.SUCCESS
                            agent_type=agent_type_str, # Should be AgentType.FILESYSTEM
                            result=result_dict
                        )
                    case PythonToolCellCreatedEvent(cell_id=cid, cell_params=cp, status=st, result=r, tool_name=tn, agent_type=at):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str} (Python Tool)")
                        # r is likely None at this stage as this event signals cell creation, not tool completion.
                        result_dict = {"content": r, "tool_name": tn} 
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value,
                            agent_type=at.value, # Use agent_type from the event
                            result=result_dict
                        )
                    case LogAIToolCellCreatedEvent(
                        cell_id=cid, 
                        cell_params=cp, 
                        status=st, # This is StatusType.SUCCESS from the event
                        result=r, 
                        tool_name=tn, 
                        agent_type=at # This is AgentType.LOG_AI from the event
                    ):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str} (LogAI Tool)")
                        
                        result_content_for_frontend = None
                        if isinstance(r, BaseModel):
                            try:
                                result_content_for_frontend = r.model_dump(mode='json')
                            except Exception as e:
                                chat_agent_logger.warning(f"Could not model_dump LogAI result: {e}, falling back to str.")
                                result_content_for_frontend = str(r)
                        elif r is not None:
                            result_content_for_frontend = str(r)

                        result_dict = {"content": result_content_for_frontend, "tool_name": tn}
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value, # e.g., "success"
                            agent_type=at.value, # e.g., "log_ai"
                            result=result_dict
                        )
                    case SummaryCellCreatedEvent(cell_id=cid, cell_params=cp, status=st, error=err):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str} (Investigation Report)")
                        # Extract the report JSON string stored by AIAgent
                        report_json_string = None
                        if cp and 'result' in cp and isinstance(cp['result'], dict) and 'content' in cp['result']:
                            report_json_string = cp['result']['content']
                        
                        # Package the report JSON string into the result field for the frontend
                        result_dict_for_frontend = {"content": report_json_string, "error": err} if report_json_string else {"error": err or "Report content missing"}
                        
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value, # Use status from the event
                            agent_type=AgentType.INVESTIGATION_REPORTER.value, # Use specific reporter type
                            result=result_dict_for_frontend
                        )
                    case SummaryCellErrorEvent(status=st, error=err, cell_params=cp, session_id=sid, notebook_id=nid):
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        message = f"Error creating report cell: {err}"
                        # Include attempted params if available
                        # if cp: message += f"\nAttempted Params: {json.dumps(cp, indent=2)[:500]}..."
                        response_part = StatusResponsePart(content=message, agent_type=AgentType.INVESTIGATION_REPORTER.value)
                    
                    # --- Status/Progress Events --- 
                    case PlanCreatedEvent(thinking=t, status=st, session_id=sid, notebook_id=nid):
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        message = f"Plan created. Thinking: {t}" if t else "Investigation plan created."
                        response_part = StatusResponsePart(content=message, agent_type=agent_type_str)
                    case PlanRevisedEvent(message=msg, revised_steps=_, session_id=_, notebook_id=_): # Match revised_steps, session_id, notebook_id
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        response_part = StatusResponsePart(content=msg or "Investigation plan revised.", agent_type=agent_type_str)
                    case StepStartedEvent(step_id=sid, status=st, session_id=sess_id, notebook_id=nid):
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        message = f"Status: {st.value} (Step: {sid})"
                        response_part = StatusResponsePart(content=message, agent_type=agent_type_str)
                    case GitHubToolErrorEvent(original_plan_step_id=sid, tool_name=tn, error=err, status=st, session_id=sess_id, notebook_id=nid):
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        message = f"GitHub Tool Error (Step: {sid}, Tool: {tn}): {err}" 
                        response_part = StatusResponsePart(content=message, agent_type=agent_type_str)
                    case SummaryStartedEvent(status=st, session_id=sid, notebook_id=nid):
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        message = "Starting final summarization..."
                        response_part = StatusResponsePart(content=message, agent_type=agent_type_str)
                    case SummaryUpdateEvent(update_info=ui, status=st, session_id=sid, notebook_id=nid): 
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        if isinstance(ui, dict):
                            message = ui.get('message', ui.get('error', json.dumps(ui)))
                        else:
                            message = str(ui) # Fallback if not dict
                        response_part = StatusResponsePart(content=message, agent_type=agent_type_str)
                    case InvestigationCompleteEvent(status=st, session_id=sid, notebook_id=nid):
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        message = "Investigation completed."
                        response_part = StatusResponsePart(content=message, agent_type=agent_type_str)
                    case StatusUpdateEvent(message=msg, status=st, step_id=sid, attempt=att, max_attempts=ma, session_id=sess_id, notebook_id=nid): # Generic status update
                        chat_agent_logger.info(f"Creating StatusResponsePart for {event_type_str}")
                        content = msg or f"Status: {st.value}"
                        if sid: content += f" (Step: {sid})"
                        if att: content += f" (Attempt: {att}/{ma})"
                        response_part = StatusResponsePart(content=content, agent_type=agent_type_str)
                    
                    case StepExecutionCompleteEvent() as exec_event: # Simplified pattern
                        event_step_id = exec_event.step_id
                        event_step_error = getattr(exec_event, 'step_error', None)
                        chat_agent_logger.info(f"MATCHED StepExecutionCompleteEvent for step {event_step_id}. Error: {event_step_error}")

                        if event_step_error:
                            event_agent_type_attr = getattr(exec_event, 'agent_type', None)
                            current_agent_type_str = event_agent_type_attr.value if event_agent_type_attr and hasattr(event_agent_type_attr, 'value') else AgentType.UNKNOWN.value
                            error_message = f"Step {event_step_id} completed with error: {event_step_error}"
                            response_part = StatusResponsePart(
                                content=error_message,
                                agent_type=current_agent_type_str
                            )
                        else:
                            chat_agent_logger.info(f"Step {event_step_id} completed successfully (from StepExecutionCompleteEvent). No new status sent to frontend.")
                            response_part = None

                    # --- Catch-all for BaseEvent --- 
                    case BaseEvent(session_id=sid, notebook_id=nid):
                        # Handle any other event inheriting from BaseEvent that wasn't explicitly matched
                        chat_agent_logger.warning(f"Unhandled BaseEvent type: {type(event)}. Creating generic StatusResponsePart.")
                        status_val = getattr(event, 'status', None)
                        message = f"Status: {status_val.value if status_val and hasattr(status_val, 'value') else 'Unknown Status'}"
                        # Try to get agent_type from the event, default to UNKNOWN
                        actual_agent_type_attr = getattr(event, 'agent_type', AgentType.UNKNOWN)
                        actual_agent_type_str = actual_agent_type_attr.value if hasattr(actual_agent_type_attr, 'value') else AgentType.UNKNOWN.value
                        response_part = StatusResponsePart(content=message, agent_type=actual_agent_type_str)

                    # --- Fallback for non-BaseEvent types (should ideally not happen) ---
                    case _:
                        chat_agent_logger.error(f"Unhandled non-BaseEvent type: {type(event)}. Cannot create standard response.")
                        # Optionally yield a specific error message or skip
                        continue # Skip yielding for completely unknown types
                
                # Ensure response_part was assigned before creating ModelResponse
                if response_part is not None: # Check if it was assigned in the match block
                    response = ModelResponse(
                        parts=[response_part],
                        timestamp=datetime.now(timezone.utc)
                    )
                    yield event_type_str, response # Yield the guaranteed string
                else:
                    chat_agent_logger.error(f"Response part was not created or was None for event type {event_type_str}. Skipping yield.")
            
            response_time = time.time() - start_time
            chat_agent_logger.info(
                f"Message handled successfully in session {session_id}",
                extra={
                    'session_id': session_id,
                    'response_time_ms': int(response_time * 1000)
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            chat_agent_logger.error(
                f"Error handling message in session {session_id}",
                extra={
                    'session_id': session_id,
                    'error': str(e),
                    'response_time_ms': int(response_time * 1000)
                },
                exc_info=True
            )
            raise
