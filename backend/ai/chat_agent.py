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
from pydantic_ai.mcp import MCPServerHTTP

from backend.config import get_settings
from backend.ai.chat_tools import NotebookCellTools
from backend.services.notebook_manager import NotebookManager
from backend.ai.agent import AIAgent

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
            if 'data' in self.result and isinstance(self.result['data'], BaseModel):
                serializable_result = self.result.copy()
                serializable_result['data'] = self.result['data'].model_dump()
                dump["result"] = serializable_result
            else:
                dump["result"] = self.result
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
        mcp_server_info: Optional[List[Tuple[str, str, str]]] = None
    ):
        self.settings = get_settings()
        chat_agent_logger.info(f"Initializing ChatAgentService with mcp_server_info: {mcp_server_info}")
        self.notebook_manager = notebook_manager
        self.mcp_server_info = mcp_server_info or []
        self.sessions: Dict[str, str] = {}  # Store session_id -> notebook_id mapping
        
        # Initialize cell tools
        self.cell_tools = NotebookCellTools(notebook_manager)

        chat_agent_logger.info(f"AI model: {self.settings.ai_model}")
        chat_agent_logger.info(f"OpenRouter API key: {self.settings.openrouter_api_key}")
        
        # Determine available MCP server types from the provided info
        mcp_server_types = []
        chat_agent_logger.info("Detecting available MCP server types from mcp_server_info...")
        seen_types = set()
        for conn_id, conn_type, url in self.mcp_server_info:
             # Use title case for display, store unique types
             display_type = conn_type.title() 
             if display_type not in seen_types:
                 mcp_server_types.append(display_type)
                 seen_types.add(display_type)
             chat_agent_logger.info(f"Detected MCP server: {url} (Conn ID: {conn_id}) as type: {conn_type}")
        
        # Store available tools info for later use (using the detected types)
        self.available_tools_info = f"You have access to the following data sources: {', '.join(mcp_server_types)}." if mcp_server_types else "No specific external data sources are currently connected."
        chat_agent_logger.info(f"Available tools info constructed: {self.available_tools_info}")
        chat_agent_logger.info(f"Detected MCP server types for tools info: {list(seen_types)}")

        system_prompt = f"""
            You are an AI assistant with access to the following data sources:
            {self.available_tools_info}
            
            Your primary responsibilities are:
            1. Understanding user queries using the available tools.
            2. Managing the conversation flow and maintaining context.
            3. Coordinating with the investigation agent for complex queries that require data retrieval or analysis.
            4. Presenting investigation results in a clear, user-friendly way.
            
            When a user asks to investigate something:
            1. Assess if the query can be addressed with the available data sources ({', '.join(mcp_server_types) or 'none'}).
            2. **Your default behavior is to proceed directly with the investigation if the query is reasonably understandable.** Do NOT ask for clarification unless the query is genuinely ambiguous or lacks essential information that prevents *any* meaningful investigation (e.g., the platform like GitHub/GitLab is required but missing, and the query doesn't specify).
            3. For standard requests (e.g., 'recent pull requests', 'active repositories', 'recently contributed repositories on GitHub'), assume common definitions (like 'recent' meaning the last few weeks/months, 'active' or 'contributed' meaning commits/PRs) and proceed. Do not ask for clarification on timeframes or precise definitions unless the user explicitly asks for something non-standard.
            4. If the query is clear and actionable according to these guidelines, pass it to the investigation agent.
            5. Present the investigation results clearly.
            6. Be ready for follow-up questions.
            
            Always respond in a helpful, conversational manner while maintaining context. 
            Focus on action and providing results based on the available tools. Avoid unnecessary conversational turns asking for clarification.
            """
        
        chat_agent_logger.info(f"System prompt - {system_prompt} for chat agent constructed.")
        
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
        
        chat_agent_logger.info("Chat agent service initialized")

    async def create_session(self, session_id: str, notebook_id: str):
        """Create a new chat session and initialize necessary components."""
        chat_agent_logger.info(f"Creating new chat session: {session_id} for notebook: {notebook_id}")
        
        # Store notebook ID for the session
        self.sessions[session_id] = notebook_id 
        mcp_server_map: Dict[str, MCPServerHTTP] = {}
        chat_agent_logger.info(f"Mapping MCP servers for session {session_id} using mcp_server_info: {self.mcp_server_info}")
        
        # Iterate through the provided server info (conn_id, type, url)
        for conn_id, conn_type, url in self.mcp_server_info:
            server_type_key = conn_type.lower() # Use lower case for map keys

            # If this type isn't mapped yet, add it.
            # This implicitly prefers the first encountered server for a given type.
            # TODO: Consider logic to prefer default connections if multiple exist.
            if server_type_key not in mcp_server_map:
                mcp_server_map[server_type_key] = MCPServerHTTP(url=url)
                chat_agent_logger.info(f"Mapped MCP server {url} (Conn ID: {conn_id}) as type {server_type_key} for session {session_id}")
            else:
                 chat_agent_logger.info(f"Skipping additional MCP server {url} (Conn ID: {conn_id}) for already mapped type {server_type_key}")

        # Initialize AIAgent with the correctly constructed map
        chat_agent_logger.info(f"Initializing AIAgent with mcp_server_map: {mcp_server_map}")
        self.ai_agent = AIAgent(mcp_server_map=mcp_server_map, notebook_id=notebook_id)
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
        start_time = time.time()
        
        try:
            # Get notebook_id for this session
            notebook_id = self.sessions.get(session_id)
            chat_agent_logger.info(f"Retrieved notebook_id: {notebook_id} for session {session_id}")
            if not notebook_id:
                chat_agent_logger.error(f"No notebook_id found for session {session_id}")
                raise ValueError(f"No notebook_id found for session {session_id}")
            
            # First, check if we need clarification
            chat_agent_logger.info(f"Checking if clarification is needed for session {session_id}...")
            clarification_result = await self.chat_agent.run(
                f"Assess if this user request needs clarification before proceeding: '{prompt}'. "
                f"Consider the available tools and context ({self.available_tools_info}). "
                f"Only ask for clarification if the request is genuinely ambiguous "
                f"or missing critical information needed to act.",
                message_history=message_history,
            )

            chat_agent_logger.info(f"Clarification check completed for session {session_id}. Result: {clarification_result.data.needs_clarification}")
            
            if clarification_result.data.needs_clarification and clarification_result.data.clarification_message:
                # Ask for clarification
                chat_agent_logger.info(f"Asking for clarification in session {session_id}: {clarification_result.data.clarification_message}")
                clarification_response = ModelResponse(
                    parts=[StatusResponsePart(
                        content=clarification_result.data.clarification_message,
                        agent_type="chat_agent"
                    )],
                    timestamp=datetime.now(timezone.utc)
                )
                # Include chat_agent as the source
                yield "clarification", clarification_response
                return
            
            # If no clarification needed, start investigation
            chat_agent_logger.info(f"Starting investigation for prompt: '{prompt}' in session {session_id}")
            async for status_type, status in self.ai_agent.investigate(
                prompt,
                session_id,
                notebook_id=notebook_id,
                message_history=message_history,
                cell_tools=self.cell_tools
            ):
                chat_agent_logger.info(f"Yielding status update for session {session_id}. Type: {status_type}")
                
                # Create appropriate response based on status type
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
                    # For non-cell responses, create a StatusResponsePart
                    response = ModelResponse(
                        parts=[StatusResponsePart(
                            content=f"Status: {status.get('status', '')}",
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