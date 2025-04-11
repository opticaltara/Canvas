"""
Chat Agent Service

This module implements a chat agent service using Pydantic AI,
focused on interactive conversations with the AI that can create
and manage notebook cells.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple, AsyncGenerator
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent
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
from backend.core.cell import AIQueryCell, CellType
from backend.core.notebook import Notebook
from backend.ai.planning import InvestigationPlan
from backend.ai.chat_tools import NotebookCellTools
from backend.services.notebook_manager import NotebookManager

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
        timestamp=message.timestamp.isoformat()
    )


class ChatRequest(BaseModel):
    """Request to send a message to the chat agent"""
    prompt: str = Field(description="User's message")
    session_id: Optional[str] = Field(description="Chat session ID", default=None)
    notebook_id: Optional[str] = Field(description="Notebook ID to associate with chat", default=None)


class ChatAgentService:
    """
    Chat agent service that handles interactive conversations
    and can create/manage notebook cells.
    """
    def __init__(
        self, 
        notebook_manager: NotebookManager,
        mcp_servers: Optional[List[MCPServerHTTP]] = None
    ):
        self.settings = get_settings()
        self.notebook_manager = notebook_manager
        self.mcp_servers = mcp_servers or []
        
        # Initialize cell tools
        self.cell_tools = NotebookCellTools(notebook_manager)
        
        # Create the chat agent
        self.chat_agent = Agent(
            model=f"anthropic:{self.settings.anthropic_model}",
            tools=self.cell_tools.get_tools(),
            mcp_servers=self.mcp_servers,
            system_prompt="""
            You are an AI assistant integrated with Sherlog Canvas, a reactive notebook for software engineering investigations.
            
            You can help users:
            1. Understand complex software systems
            2. Investigate and diagnose issues
            3. Create and manage notebook cells for data analysis
            4. Interpret logs, metrics, and other data sources
            
            You have access to tools that can manage notebook cells directly:
            - create_cell: Create a new cell (markdown, python, sql, log, metric, s3, ai_query)
            - update_cell: Update an existing cell's content or metadata
            - execute_cell: Queue a cell for execution
            - list_cells: Get information about all cells in a notebook
            - get_cell: Get detailed information about a specific cell
            - execute_ai_query: Create and queue an AI query cell
            
            When asked to investigate an issue or analyze data, you should:
            1. Break down the problem into logical steps
            2. Create appropriate cells for each step (markdown for explanations, data queries, Python for analysis)
            3. Make sure to create cells in the right order with correct dependencies
            4. Provide clear explanations in markdown cells
            
            Respond in a helpful, conversational manner while using your tools to manage notebook cells as needed.
            """
        )
        
        chat_agent_logger.info("Chat agent service initialized")

    async def create_session(self, session_id: Optional[str] = None, notebook_id: Optional[str] = None) -> str:
        """Create a new chat session"""
        session_id = session_id or str(uuid4())
        chat_agent_logger.info(f"Creating new chat session: {session_id}")
        
        return session_id
    
    async def handle_message(
        self, 
        prompt: str, 
        session_id: str,
        message_history: List[ModelMessage] = None
    ) -> ModelResponse:
        """
        Process a user message and return the AI response
        
        This is a non-streaming version used for simple requests
        """
        chat_agent_logger.info(f"Handling message in session {session_id}")
        start_time = time.time()
        
        try:
            # Use the agent with message history if provided
            history = message_history or []
            result = await self.chat_agent.run(prompt, message_history=history)
            
            response_time = time.time() - start_time
            chat_agent_logger.info(
                f"Message handled successfully",
                extra={
                    'session_id': session_id,
                    'response_time_ms': int(response_time * 1000)
                }
            )
            
            return result.response
            
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
    
    async def stream_response(
        self, 
        prompt: str, 
        session_id: str,
        message_history: List[ModelMessage] = None
    ) -> AsyncGenerator[Tuple[str, ModelResponse], None]:
        """
        Stream the AI's response to a user message
        
        Yields tuples of (text_chunk, full_response)
        """
        chat_agent_logger.info(f"Streaming response for session {session_id}")
        start_time = time.time()
        
        try:
            # Use the agent with message history if provided
            history = message_history or []
            
            async with self.chat_agent.run_stream(prompt, message_history=history) as result:
                async for text in result.stream(debounce_by=0.01):
                    # Create a response object with the current text
                    current_response = ModelResponse(
                        parts=[TextPart(text)],
                        timestamp=result.timestamp()
                    )
                    
                    yield text, current_response
            
            response_time = time.time() - start_time
            chat_agent_logger.info(
                f"Response streaming completed",
                extra={
                    'session_id': session_id,
                    'response_time_ms': int(response_time * 1000)
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            chat_agent_logger.error(
                f"Error streaming response",
                extra={
                    'session_id': session_id,
                    'error': str(e),
                    'response_time_ms': int(response_time * 1000)
                },
                exc_info=True
            )
            raise 