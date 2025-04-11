"""
Chat Agent Service

This module implements a chat agent service using Pydantic AI,
focused on interactive conversations with the AI that can create
and manage notebook cells.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.messages import (
    ModelMessage, 
    ModelMessagesTypeAdapter, 
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


class ChatMessage(BaseModel):
    """Format of messages used in the API"""
    role: str = Field(description="Role: 'user' or 'model'")
    content: str = Field(description="Message content")
    timestamp: str = Field(description="Timestamp of the message")


def to_chat_message(message: ModelMessage) -> ChatMessage:
    """Convert a ModelMessage to a ChatMessage"""
    first_part = message.parts[0]
    
    if isinstance(message, ModelRequest) and isinstance(first_part, UserPromptPart):
        if isinstance(first_part.content, str):
            return ChatMessage(
                role="user",
                content=first_part.content,
                timestamp=first_part.timestamp.isoformat()
            )
    elif isinstance(message, ModelResponse) and isinstance(first_part, TextPart):
        return ChatMessage(
            role="model",
            content=first_part.content,
            timestamp=message.timestamp.isoformat()
        )
    
    # If we can't convert, use a default representation
    return ChatMessage(
        role="unknown",
        content=str(message),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


class ChatRequest(BaseModel):
    """Request to send a message to the chat agent"""
    prompt: str = Field(description="User's message")
    session_id: Optional[str] = Field(description="Chat session ID", default=None)
    notebook_id: Optional[str] = Field(description="Notebook ID to associate with chat", default=None)


class ChatAgentService:
    """
    Chat agent service that handles interactive conversations
    and coordinates with the AI agent for investigations.
    """
    def __init__(
        self, 
        notebook_manager: NotebookManager,
        mcp_servers: Optional[List[MCPServerHTTP]] = None
    ):
        self.settings = get_settings()
        self.notebook_manager = notebook_manager
        self.mcp_servers = mcp_servers or []
        self.sessions: Dict[str, str] = {}  # Store session_id -> notebook_id mapping
        
        # Initialize the AI agent for investigation
        self.ai_agent = AIAgent(mcp_servers=self.mcp_servers)
        
        # Initialize cell tools
        self.cell_tools = NotebookCellTools(notebook_manager)
        
        # Create the chat agent
        self.chat_agent = Agent(
            model=AnthropicModel(
                self.settings.anthropic_model,
                provider=AnthropicProvider(api_key=self.settings.anthropic_api_key)
            ),
            tools=self.cell_tools.get_tools(),
            system_prompt="""
            You are an AI assistant integrated with Sherlog Canvas, a reactive notebook for software engineering investigations.
            
            Your primary responsibilities are:
            1. Understanding user queries and asking clarifying questions when needed
            2. Managing the conversation flow and maintaining context
            3. Coordinating with the investigation agent for complex queries
            4. Presenting investigation results in a clear, user-friendly way
            
            When a user asks to investigate something:
            1. First, ensure you understand their request completely. Ask clarifying questions if needed
            2. Once clear, pass the query to the investigation agent
            3. Present the investigation results in a clear, organized way
            4. Be ready to answer follow-up questions about the investigation
            
            Always respond in a helpful, conversational manner while maintaining context of the investigation.
            """
        )
        
        chat_agent_logger.info("Chat agent service initialized")

    async def create_session(self, session_id: Optional[str] = None, notebook_id: Optional[str] = None) -> str:
        """Create a new chat session"""
        session_id = session_id or str(uuid4())
        if notebook_id:
            self.sessions[session_id] = notebook_id
        chat_agent_logger.info(f"Creating new chat session: {session_id} with notebook: {notebook_id}")
        return session_id
    
    async def handle_message(
        self, 
        prompt: str, 
        session_id: str,
        message_history: List[ModelMessage] = []
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
        start_time = time.time()
        
        try:
            # Get notebook_id for this session
            notebook_id = self.sessions.get(session_id)
            if not notebook_id:
                raise ValueError(f"No notebook_id found for session {session_id}")
            
            # First, check if we need clarification
            clarification_result = await self.chat_agent.run(
                f"Check if this query needs clarification: {prompt}",
                message_history=message_history
            )
            
            if "needs_clarification" in clarification_result.data.lower():
                # Ask for clarification
                clarification_response = ModelResponse(
                    parts=[TextPart(clarification_result.data)],
                    timestamp=datetime.now(timezone.utc)
                )
                yield "clarification", clarification_response
                return
            
            # If no clarification needed, start investigation
            async for status_type, status in self.ai_agent.investigate(
                prompt,
                session_id,
                notebook_id=notebook_id,
                message_history=message_history,
                cell_tools=self.cell_tools
            ):
                # Create response for this status update
                response = ModelResponse(
                    parts=[TextPart(f"Status: {status_type} - {status['status']}")],
                    timestamp=datetime.now(timezone.utc)
                )
                yield status_type, response
            
            response_time = time.time() - start_time
            chat_agent_logger.info(
                f"Message handled successfully",
                extra={
                    'session_id': session_id,
                    'response_time_ms': int(response_time * 1000)
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            chat_agent_logger.error(
                f"Error handling message",
                extra={
                    'session_id': session_id,
                    'error': str(e),
                    'response_time_ms': int(response_time * 1000)
                },
                exc_info=True
            )
            raise 