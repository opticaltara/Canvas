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
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import (
    ModelMessage, 
    ModelRequest, 
    ModelResponse, 
    TextPart, 
    UserPromptPart
)

from backend.config import get_settings
from backend.ai.chat_tools import NotebookCellTools
from backend.services.notebook_manager import NotebookManager
from backend.ai.agent import AIAgent
from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.ai.events import (
    EventType, 
    AgentType, 
    StatusType, 
    ClarificationNeededEvent,
    PlanCreatedEvent,
    PlanCellCreatedEvent,
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
    FileSystemToolCellCreatedEvent
)

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

class ClarificationResult(BaseModel):
    needs_clarification: bool = Field(description="Whether the query needs clarification")
    clarification_message: Optional[str] = Field(description="Message to ask for clarification if needed", default=None)


def to_chat_message(message: ModelMessage, agent: Optional[str] = None) -> ChatMessage:
    """Convert a ModelMessage to a ChatMessage"""
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
        self.chat_agent: Optional[Agent[None, ClarificationResult]] = None
        self.ai_agent: Optional[AIAgent] = None
        self.cell_tools: Optional[NotebookCellTools] = None

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
        self.cell_tools = NotebookCellTools(notebook_manager=self.notebook_manager)

        # Setup Clarification Agent (chat_agent)
        system_prompt = self._generate_system_prompt() # Uses fetched tool info
        chat_agent_logger.info(f"System prompt generated for chat agent (notebook: {self.notebook_id}).")
        self.chat_agent = Agent(
            model=OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
            ),
            system_prompt=system_prompt,
            result_type=ClarificationResult
        )

        # Setup Investigation Agent (ai_agent)
        chat_agent_logger.info(f"Initializing AIAgent for notebook {self.notebook_id} with sources: {self._available_data_source_types}")
        self.ai_agent = AIAgent(
            notebook_manager=self.notebook_manager,
            notebook_id=self.notebook_id,
            available_data_sources=self._available_data_source_types or []
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
                unique_types = sorted(list(set(valid_types)))
                display_types = [t.title() for t in unique_types]
                if display_types:
                    self.available_tools_info = f"You have access to the following data sources: {', '.join(display_types)}."
                else:
                    self.available_tools_info = "No specific external data sources are currently connected."
                self._available_data_source_types = unique_types
                chat_agent_logger.info(f"Fetched connection types. Available tools info: {self.available_tools_info}")
            except Exception as e:
                chat_agent_logger.error(f"Failed to fetch connection types: {e}", exc_info=True)
                self.available_tools_info = "Error fetching available data sources."
                self._available_data_source_types = [] # Set to empty list on error

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
        if not self.notebook_id or not self.chat_agent or not self.ai_agent or not self.cell_tools:
             chat_agent_logger.error(f"Agent for session {session_id} accessed handle_message before initialization.")
             raise RuntimeError("ChatAgentService not initialized. Call initialize() first.")

        chat_agent_logger.info(f"Handling message in session {session_id} (Notebook: {self.notebook_id})")
        chat_agent_logger.info(f"Received prompt: '{prompt}'")
        chat_agent_logger.info(f"Received message_history length: {len(message_history)} for session {session_id}")
        if message_history:
            chat_agent_logger.info(f"Last message in history: {str(message_history[-1])}") 
        start_time = time.time()
        
        try:
            # Define Redis key for the skip flag
            redis_skip_key = f"skip_clarification:{session_id}"
            CLARIFICATION_SKIP_TTL = 300 # 5 minutes
            
            # --- Check Redis FIRST --- 
            should_skip_clarification = False
            if self.redis_client:
                try:
                    skip_flag = await self.redis_client.get(redis_skip_key)
                    if skip_flag == "True":
                        chat_agent_logger.info(f"Redis flag found, skipping clarification check for session {session_id}.")
                        should_skip_clarification = True
                        # Consume the flag
                        await self.redis_client.delete(redis_skip_key)
                except redis.RedisError as redis_err:
                    chat_agent_logger.warning(f"Redis error checking skip flag for session {session_id}: {redis_err}. Proceeding without skip.")
            else:
                 chat_agent_logger.warning(f"Redis client not available in ChatAgentService for session {session_id}. Cannot check skip flag.")

            # --- End Optional History Check --- 

            # ---> Run Clarification Agent only if NOT skipping <--- 
            if not should_skip_clarification:
                chat_agent_logger.info(f"Checking if clarification is needed for session {session_id}...")
                # Construct the prompt for clarification assessment based on the full context
                clarification_assessment_prompt = (
                    f"Review the *entire conversation history* ending with the latest user message. "
                    f"Previously, I may have asked for clarification. The user's latest message might provide some or all of that information. "
                    f"Available data sources: {self.available_tools_info or '(loading...)'}. "
                    f"Assess if, *after considering the user's latest response*, there is *still* critical information missing to proceed with their *original* request. "
                    f"If the latest user message sufficiently answers the previous clarification or provides enough information to take the *next step*, then DO NOT ask for clarification again (needs_clarification: false). "
                    f"Only ask for clarification (needs_clarification: true) if the request *remains* genuinely ambiguous or is *still* missing essential details *despite* the user's last message."
                )

                # Log the history being passed to the clarification check
                chat_agent_logger.info(f"Message history for clarification check (session {session_id}): {len(message_history)} messages.")
                if message_history: # Log the last message if history is not empty
                    chat_agent_logger.info(f"Last message in history for clarification: {str(message_history[-1])}")

                try:
                    # Call the correctly typed agent
                    clarification_run_result = await self.chat_agent.run(
                        clarification_assessment_prompt,
                        message_history=message_history, # History includes user's LATEST reply here
                    )
                    # Access the actual output model from the result
                    clarification_output: ClarificationResult = clarification_run_result.output
                except Exception as clar_err:
                    chat_agent_logger.error(f"Error during clarification check API call: {clar_err}", exc_info=True)
                    # Handle error appropriately, maybe yield an error status
                    yield "error", ModelResponse(parts=[StatusResponsePart(content=f"Error checking clarification: {clar_err}", agent_type="chat_agent")], timestamp=datetime.now(timezone.utc))
                    return

                chat_agent_logger.info(f"Clarification check completed. Result: {clarification_output.needs_clarification}")

                if clarification_output.needs_clarification and clarification_output.clarification_message:
                    chat_agent_logger.info(f"Asking for clarification: {clarification_output.clarification_message}")
                    
                    # --- SET Redis Flag BEFORE yielding --- 
                    if self.redis_client:
                        try:
                             await self.redis_client.set(redis_skip_key, "True", ex=CLARIFICATION_SKIP_TTL)
                             chat_agent_logger.info(f"Set Redis skip flag for session {session_id} with TTL {CLARIFICATION_SKIP_TTL}s.")
                        except redis.RedisError as redis_err:
                             chat_agent_logger.warning(f"Redis error setting skip flag for session {session_id}: {redis_err}")
                    # --- End SET Redis Flag ---
                    
                    # Create ClarificationNeededEvent object
                    clarification_event = ClarificationNeededEvent(
                        message=clarification_output.clarification_message,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                    # Wrap in ModelResponse using StatusResponsePart for transport
                    response_part = StatusResponsePart(
                        content=clarification_event.message,
                        agent_type=clarification_event.agent_type.value
                    )
                    response = ModelResponse(parts=[response_part], timestamp=datetime.now(timezone.utc))
                    # Yield EventType string and the response object
                    yield clarification_event.type.value, response
                    return # Exit after asking for clarification

            # If clarification wasn't needed OR was skipped (by Redis or History check), proceed to investigation
            if should_skip_clarification:
                 log_msg = f"Proceeding to investigation for session {session_id} (Clarification skipped)."
            else:
                 log_msg = f"Proceeding to investigation for session {session_id} (Clarification not needed)."
            chat_agent_logger.info(log_msg)
        
        
            async for event in self.ai_agent.investigate(
                prompt, # Pass the potentially context-enhanced prompt
                session_id,
                notebook_id=self.notebook_id,
                message_history=message_history, # Pass history including the user's latest message
                cell_tools=self.cell_tools
            ):
                # Now expect event model instances directly
                event_type_enum = event.type
                event_type_str = event_type_enum.value
                agent_type_attr = getattr(event, 'agent_type', None)
                agent_type_str = agent_type_attr.value if agent_type_attr else AgentType.UNKNOWN.value
                chat_agent_logger.info(f"Yielding event update for session {session_id}. Type: {event_type_str}")
                chat_agent_logger.debug(f"Received event object: {event!r}")

                response_part: Union[CellResponsePart, StatusResponsePart]
                
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
                    case StepErrorEvent(cell_id=cid, cell_params=cp, status=st, result=r, error=err):
                        chat_agent_logger.info(f"Creating CellResponsePart for {event_type_str}")
                        
                        if r is not None:
                            result_dict = r.model_dump() if isinstance(r, BaseModel) else {"content": str(r)}
                        else:
                            result_dict = {}
 
                        if err is not None:
                            if isinstance(result_dict, dict): 
                                result_dict['error'] = err
                            else:
                                result_dict = {"original_result": result_dict, "error": err}
 
                        response_part = CellResponsePart(
                            cell_id=str(cid) if cid else "",
                            cell_params=cp or {},
                            status_type=st.value,
                            agent_type=agent_type_str,
                            result=result_dict
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
                    
                    # --- Catch-all for BaseEvent --- 
                    case BaseEvent(session_id=sid, notebook_id=nid):
                        # Handle any other event inheriting from BaseEvent that wasn't explicitly matched
                        chat_agent_logger.warning(f"Unhandled BaseEvent type: {type(event)}. Creating generic StatusResponsePart.")
                        status_val = getattr(event, 'status', None)
                        message = f"Status: {status_val.value if status_val and hasattr(status_val, 'value') else 'Unknown Status'}"
                        agent_type_val = getattr(event, 'agent_type', AgentType.UNKNOWN)
                        response_part = StatusResponsePart(content=message, agent_type=agent_type_val.value)

                    # --- Fallback for non-BaseEvent types (should ideally not happen) ---
                    case _:
                        chat_agent_logger.error(f"Unhandled non-BaseEvent type: {type(event)}. Cannot create standard response.")
                        # Optionally yield a specific error message or skip
                        continue # Skip yielding for completely unknown types
                
                # Ensure response_part was assigned before creating ModelResponse
                if 'response_part' in locals():
                    response = ModelResponse(
                        parts=[response_part],
                        timestamp=datetime.now(timezone.utc)
                    )
                    yield event_type_str, response
                else:
                    chat_agent_logger.error(f"Response part was not created for event type {event_type_str}. Skipping yield.")
            
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