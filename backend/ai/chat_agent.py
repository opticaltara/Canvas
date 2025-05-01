"""
Chat Agent Service

This module implements a chat agent service using Pydantic AI,
focused on interactive conversations with the AI that can create
and manage notebook cells.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import redis.asyncio as redis
from redis.asyncio.client import Redis as AsyncRedis

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
        dump = {
            "type": "cell_response",
            "cell_id": self.cell_id,
            "cell_params": self.cell_params,
            "status_type": self.status_type,
            "agent_type": self.agent_type
        }
        if self.result:
            serializable_result = {}
            for key, value in self.result.items():
                if isinstance(value, BaseModel):
                    # Handle nested Pydantic models (like QueryResult data)
                    serializable_result[key] = value.model_dump()
                elif isinstance(value, list) and value: 
                    # Check if it's a list and potentially contains Pydantic models
                    if all(isinstance(item, BaseModel) for item in value):
                        # Handle lists of Pydantic models (e.g., tool_calls)
                        serializable_result[key] = [item.model_dump() for item in value]
                    else:
                        # Assume list contains other JSON-serializable types
                        serializable_result[key] = value 
                else:
                    # Assume other types are JSON serializable
                    serializable_result[key] = value
            dump["result"] = serializable_result
            
        return dump

class StatusResponsePart(TextPart):
    """A specialized response part for status messages"""
    def __init__(self, content: str, agent_type: str):
        super().__init__(content=content)
        self.agent_type = agent_type

    def model_dump(self) -> Dict[str, Any]:
        return {
            "type": "status_response",
            "content": self.content,
            "agent_type": self.agent_type
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
            notebook_id=self.notebook_id,
            available_data_sources=self._available_data_source_types or [] # Pass the fetched list
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
            Tuples of (status_type, response) as updates occur
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

            # --- Original History Check (Optional Fallback - can be removed if Redis is reliable) --- 
            # If Redis didn't tell us to skip, we can still check history as a fallback
            # Remove this block if you want to rely purely on the Redis flag.
            if not should_skip_clarification:
                if message_history and len(message_history) > 1:
                    last_model_message = message_history[-2]
                    if (isinstance(last_model_message, ModelResponse) and
                        last_model_message.parts and
                        isinstance(last_model_message.parts[0], StatusResponsePart) and
                        last_model_message.parts[0].agent_type == 'chat_agent'):
                        chat_agent_logger.info(f"DB History indicates last message was clarification, setting skip flag (Redis flag was missing/expired). Session: {session_id}")
                        should_skip_clarification = True
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
                    
                    clarification_response = ModelResponse(
                        parts=[StatusResponsePart(
                            content=clarification_output.clarification_message,
                            agent_type="chat_agent" # Mark as chat_agent response
                        )],
                        timestamp=datetime.now(timezone.utc)
                    )
                    yield "clarification", clarification_response
                    return # Exit after asking for clarification

            # If clarification wasn't needed OR was skipped (by Redis or History check), proceed to investigation
            if should_skip_clarification:
                 log_msg = f"Proceeding to investigation for session {session_id} (Clarification skipped)."
            else:
                 log_msg = f"Proceeding to investigation for session {session_id} (Clarification not needed)."
            chat_agent_logger.info(log_msg)
            
            # ---> Investigation block starts here <---
            # Make sure to pass the correct prompt (prompt_to_use includes context)
            # Ensure message_history includes the user's latest reply which might be the clarification
            async for status_dict in self.ai_agent.investigate(
                prompt, # Pass the potentially context-enhanced prompt
                session_id,
                notebook_id=self.notebook_id,
                message_history=message_history, # Pass history including the user's latest message
                cell_tools=self.cell_tools
            ):
                # Extract event type from the dict itself
                event_type = status_dict.get('type', 'unknown_event') # Use 'type' key
                chat_agent_logger.info(f"Yielding status update for session {session_id}. Type: {event_type}")
                chat_agent_logger.info(f"Received status dictionary for {event_type}: {status_dict}")
                
                # Determine if it's a cell event based on the type key
                is_cell_event = (
                    event_type.startswith("step_") or 
                    event_type == "plan_cell_created" or 
                    event_type == "summary_cell_created" or
                    event_type == "github_tool_cell_created"
                )
                
                # Use status_dict instead of status
                if is_cell_event and 'cell_params' in status_dict:
                    chat_agent_logger.info(f"Creating CellResponsePart for {event_type}")
                    # Format as CellResponsePart if it's a cell event and has params
                    response = ModelResponse(
                        parts=[CellResponsePart(
                            cell_id=status_dict.get('cell_id', ''),
                            cell_params=status_dict.get('cell_params', {}),
                            # Use status from dict if available, else event_type
                            status_type=status_dict.get('status', event_type), 
                            agent_type=status_dict.get('agent_type', 'unknown'),
                            result=status_dict.get('result', None)
                        )],
                        timestamp=datetime.now(timezone.utc)
                    )
                else:
                    chat_agent_logger.info(f"Creating StatusResponsePart for {event_type} (is_cell_event={is_cell_event}, has_cell_params={'cell_params' in status_dict})")
                    # Otherwise, format as StatusResponsePart
                    update_info = status_dict.get('update_info')
                    # Default content from status_dict['status'] or the event_type itself
                    content_message = f"Status: {status_dict.get('status', event_type)}"
                    
                    # Try getting more specific message from update_info or top level
                    if isinstance(update_info, dict):
                        content_message = update_info.get('message', update_info.get('error', content_message))
                    elif isinstance(status_dict.get("message"), str):
                        content_message = status_dict["message"]
                    elif isinstance(status_dict.get("error"), str):
                        content_message = status_dict["error"]
                    
                    response = ModelResponse(
                        parts=[StatusResponsePart(
                            content=content_message,
                            agent_type=status_dict.get('agent_type', 'unknown')
                        )],
                        timestamp=datetime.now(timezone.utc)
                    )
                
                # Yield the original event type string and the response object
                yield event_type, response
            
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