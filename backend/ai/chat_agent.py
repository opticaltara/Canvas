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
from uuid import uuid4
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
    def __init__(self, cell_params: Dict[str, Any], status_type: str, agent_type: str):
        super().__init__(content="")  # Empty content since we're using custom fields
        self.cell_params = cell_params
        self.status_type = status_type
        self.agent_type = agent_type

    def model_dump(self) -> Dict[str, Any]:
        return {
            "type": "cell_response",
            "cell_params": self.cell_params,
            "status_type": self.status_type,
            "agent_type": self.agent_type
        }

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
        mcp_servers: Optional[List[MCPServerHTTP]] = None
    ):
        self.settings = get_settings()
        self.notebook_manager = notebook_manager
        self.mcp_servers = mcp_servers or []
        self.sessions: Dict[str, str] = {}  # Store session_id -> notebook_id mapping
        
        # Initialize cell tools
        self.cell_tools = NotebookCellTools(notebook_manager)

        chat_agent_logger.info(f"AI model: {self.settings.ai_model}")
        chat_agent_logger.info(f"OpenRouter API key: {self.settings.openrouter_api_key}")
        
        # Create the chat agent
        self.chat_agent = Agent(
            model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
            ),
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
            # Initialize the AI agent with the notebook ID
            self.ai_agent = AIAgent(mcp_servers=self.mcp_servers, notebook_id=notebook_id)
        chat_agent_logger.info(f"Creating new chat session: {session_id} with notebook: {notebook_id}")
        return session_id
    
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
        start_time = time.time()
        
        try:
            # Get notebook_id for this session
            notebook_id = self.sessions.get(session_id)
            if not notebook_id:
                raise ValueError(f"No notebook_id found for session {session_id}")
            
            # First, check if we need clarification
            clarification_result = await self.chat_agent.run(
                f"Check if this query needs clarification: {prompt}. The current investigation notebook is {notebook_id}.",
                message_history=message_history,
                result_type=ClarificationResult
            )

            chat_agent_logger.info(f"Clarification result: {clarification_result}")
            
            if clarification_result.data.needs_clarification and clarification_result.data.clarification_message:
                # Ask for clarification
                chat_agent_logger.info(f"Asking for clarification: {clarification_result.data.clarification_message}")
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
            chat_agent_logger.info(f"Starting investigation for prompt: {prompt}")
            async for status_type, status in self.ai_agent.investigate(
                prompt,
                session_id,
                notebook_id=notebook_id,
                message_history=message_history,
                cell_tools=self.cell_tools
            ):
                chat_agent_logger.info(f"Status: {status}")
                
                # Create appropriate response based on status type
                if 'cell_params' in status:
                    response = ModelResponse(
                        parts=[CellResponsePart(
                            cell_params=status.get('cell_params', {}),
                            status_type=status.get('status', ''),
                            agent_type=status.get('agent_type', 'unknown')
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