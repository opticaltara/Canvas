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
    """
    def __init__(
        self, 
        notebook_manager: NotebookManager,
        connection_manager: Optional[ConnectionManager] = None # Allow None initially
    ):
        self.settings = get_settings()
        chat_agent_logger.info(f"Initializing ChatAgentService...")
        self.notebook_manager = notebook_manager
        self.sessions: Dict[str, str] = {}
        
        # Initialize cell tools
        self.cell_tools = NotebookCellTools(notebook_manager)

        chat_agent_logger.info(f"AI model: {self.settings.ai_model}")
        chat_agent_logger.info(f"OpenRouter API key: {self.settings.openrouter_api_key}")
        self.available_tools_info: Optional[str] = None # Initialize as None
        self._available_data_source_types: Optional[List[str]] = None # Store the actual list of types
        self._connection_manager = connection_manager or get_connection_manager() # Store connection manager instance
        system_prompt = self._generate_system_prompt()
        
        chat_agent_logger.info(f"System prompt - preparing for chat agent.")
        
        # Create the chat agent
        self.chat_agent = Agent(
            model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
            ),
            system_prompt=system_prompt,
            result_type=ClarificationResult
        )
        
        chat_agent_logger.info("Chat agent service initialized (will fetch connection types async).")

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

    def _generate_system_prompt(self) -> str:
        """Generates the system prompt, using placeholder if types not fetched yet."""
        tools_info = self.available_tools_info or "(Data source information is loading...)" 
        
        return f"""
            You are an AI assistant managing conversations and coordinating data investigations.
            {tools_info}

            Your primary responsibilities are:
            1. Understanding user queries.
            2. Managing the conversation flow.
            3. Coordinating investigations using available data sources.
            4. Presenting results clearly.

            When a user asks to investigate something:
            1. Assess if the query can be addressed with the available data sources ({tools_info}).
            2. **CRITICAL: Proceed DIRECTLY with the investigation if the query is reasonably understandable.** Do NOT ask for clarification unless the query is fundamentally ambiguous (e.g., completely unclear intent) or lacks ESSENTIAL information that prevents *any* meaningful action (e.g., the specific platform like GitHub is absolutely required but missing, and the query doesn't imply it).
            3. **DO NOT ask for usernames (like GitHub username) if the relevant data source (e.g., GitHub MCP) is listed as available.** Assume the connection provides the necessary user context.
            4. For standard requests (e.g., 'recent pull requests', 'active repositories', 'my recent commits'), assume common definitions (like 'recent' means the last few weeks/month, 'active' involves recent activity) and PROCEED. Do not ask for clarification on timeframes or precise definitions unless the user explicitly requests something non-standard or highly specific.
            5. If the query is clear and actionable according to these strict guidelines, pass it for investigation.
            6. Present investigation results clearly. Be ready for follow-up questions.

            Your goal is to be proactive and action-oriented. Avoid unnecessary conversational turns. Focus on executing the request based on the available tools and context.
            """

    async def create_session(self, session_id: str, notebook_id: str):
        """Create a new chat session and initialize necessary components."""
        chat_agent_logger.info(f"Creating new chat session: {session_id} for notebook: {notebook_id}")
        self.sessions[session_id] = notebook_id 
        await self._fetch_and_set_available_tools_info()
        chat_agent_logger.info(f"Initializing AIAgent for session {session_id} with sources: {self._available_data_source_types}")
        self.ai_agent = AIAgent(
            notebook_id=notebook_id,
            available_data_sources=self._available_data_source_types or [] # Pass the fetched list
        )
        self.cell_tools = NotebookCellTools(notebook_manager=self.notebook_manager)
        
        chat_agent_logger.info(f"Chat session {session_id} created successfully.")

    async def handle_message(
        self, 
        prompt: str, 
        session_id: str,
        message_history: List[ModelMessage] = [],
    ) -> AsyncGenerator[Tuple[str, ModelResponse], None]:
        """
        Process a user message and stream status updates
        
        Args:
            prompt: The user's message
            session_id: The chat session ID
            message_history: Previous messages in the session
            
        Yields:
            Tuples of (status_type, response) as updates occur
        """
        chat_agent_logger.info(f"Handling message in session {session_id}")
        chat_agent_logger.info(f"Received prompt: '{prompt}'")
        chat_agent_logger.info(f"Received message_history length: {len(message_history)} for session {session_id}")
        if message_history:
            chat_agent_logger.debug(f"Last message in history: {str(message_history[-1])}") 
        start_time = time.time()
        
        try:
            # Ensure available tools info is fetched before proceeding
            await self._fetch_and_set_available_tools_info()

            notebook_id = self.sessions.get(session_id)
            chat_agent_logger.info(f"Retrieved notebook_id: {notebook_id} for session {session_id}")
            if not notebook_id:
                chat_agent_logger.error(f"No notebook_id found for session {session_id}")
                raise ValueError(f"No notebook_id found for session {session_id}")
            
            chat_agent_logger.info(f"Checking if clarification is needed for session {session_id}...")
            clarification_result = await self.chat_agent.run(
                f"Assess if this user request needs clarification before proceeding: '{prompt}'. "
                f"Consider the available tools and context ({self.available_tools_info or '(loading...)'}). "
                f"Only ask for clarification if the request is genuinely ambiguous "
                f"or missing critical information needed to act.",
                message_history=message_history,
            )

            chat_agent_logger.info(f"Clarification check completed for session {session_id}. Result: {clarification_result.data.needs_clarification}")
            
            if clarification_result.data.needs_clarification and clarification_result.data.clarification_message:
                chat_agent_logger.info(f"Asking for clarification in session {session_id}: {clarification_result.data.clarification_message}")
                clarification_response = ModelResponse(
                    parts=[StatusResponsePart(
                        content=clarification_result.data.clarification_message,
                        agent_type="chat_agent"
                    )],
                    timestamp=datetime.now(timezone.utc)
                )
                yield "clarification", clarification_response
                return
            chat_agent_logger.info(f"Starting investigation for prompt: '{prompt}' in session {session_id}")
            async for status_type, status in self.ai_agent.investigate(
                prompt,
                session_id,
                notebook_id=notebook_id,
                message_history=message_history,
                cell_tools=self.cell_tools
            ):
                chat_agent_logger.info(f"Yielding status update for session {session_id}. Type: {status_type}")
                
                if 'cell_params' in status:
                    response = ModelResponse(
                        parts=[CellResponsePart(
                            cell_id=status.get('cell_id', ''),
                            cell_params=status.get('cell_params', {}),
                            status_type=status.get('status', ''),
                            agent_type=status.get('agent_type', 'unknown'),
                            result=status.get('result', None)
                        )],
                        timestamp=datetime.now(timezone.utc)
                    )
                else:
                    update_info = status.get('update_info')
                    content_message = f"Status: {status.get('status', '')}"
                    
                    if isinstance(update_info, dict):
                        content_message = update_info.get('message', update_info.get('error', content_message))
                    
                    response = ModelResponse(
                        parts=[StatusResponsePart(
                            content=content_message,
                            agent_type=status.get('agent_type', 'unknown')
                        )],
                        timestamp=datetime.now(timezone.utc)
                    )
                
                yield status_type, response
            
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