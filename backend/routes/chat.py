"""
Chat API Endpoints

This module implements the chat API endpoints for interacting with the AI agent.
"""

import json
import logging
import time
from typing import Dict
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.ai.chat_agent import ChatAgentService, ChatRequest, ChatMessage, to_chat_message, CellResponsePart
from backend.db.chat_db import ChatDatabase
from backend.services.notebook_manager import NotebookManager, get_notebook_manager

# Initialize logger
chat_logger = logging.getLogger("routes.chat")

# Add correlation ID filter to the logger
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

chat_logger.addFilter(CorrelationIdFilter())

# Create router
router = APIRouter()


# Dependency to get chat agent service
async def get_chat_agent_service(
    notebook_manager: NotebookManager = Depends(get_notebook_manager)
) -> ChatAgentService:
    """Get the chat agent service singleton"""
    return ChatAgentService(notebook_manager)


# Dependency to get chat database
async def get_chat_db(request: Request) -> ChatDatabase:
    """Get the chat database from the app state"""
    return request.app.state.chat_db


class CreateSessionRequest(BaseModel):
    """Request to create a new chat session"""
    notebook_id: str = Field(description="The ID of the notebook to associate with the chat session")


class CreateSessionResponse(BaseModel):
    """Response for creating a new chat session"""
    session_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    chat_agent: ChatAgentService = Depends(get_chat_agent_service),
    chat_db: ChatDatabase = Depends(get_chat_db)
) -> Dict:
    """
    Create a new chat session
    
    Returns:
        The session ID
    """
    start_time = time.time()
    
    try:
        # Generate a new session ID
        session_id = str(uuid4())
        
        # Create the session in the database
        await chat_db.create_session(session_id, request.notebook_id)
        
        # Initialize the session with the agent
        await chat_agent.create_session(session_id, request.notebook_id)
        
        process_time = time.time() - start_time
        chat_logger.info(
            f"Created new chat session",
            extra={
                'correlation_id': str(uuid4()),
                'session_id': session_id,
                'notebook_id': request.notebook_id,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return {"session_id": session_id}
    
    except Exception as e:
        process_time = time.time() - start_time
        chat_logger.error(
            f"Error creating chat session",
            extra={
                'correlation_id': str(uuid4()),
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    chat_db: ChatDatabase = Depends(get_chat_db)
) -> Response:
    """
    Get all messages for a chat session
    
    Returns:
        Newline-delimited JSON messages
    """
    start_time = time.time()
    
    try:
        # Verify the session exists
        session = await chat_db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
        
        # Get messages from the database
        messages = await chat_db.get_messages(session_id)
        
        process_time = time.time() - start_time
        chat_logger.info(
            f"Retrieved chat session messages",
            extra={
                'correlation_id': str(uuid4()),
                'session_id': session_id,
                'message_count': len(messages),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        # Convert to ChatMessage format and serialize as newline-delimited JSON
        chat_messages = [json.dumps(ChatMessage.model_validate(msg).model_dump()) for msg in messages]
        return Response(
            content="\n".join(chat_messages),
            media_type="text/plain"
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        process_time = time.time() - start_time
        chat_logger.error(
            f"Error getting chat session messages",
            extra={
                'correlation_id': str(uuid4()),
                'session_id': session_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to get chat messages: {str(e)}")


@router.post("/sessions/{session_id}/messages")
async def post_message(
    session_id: str,
    prompt: str = Form(...),
    chat_agent: ChatAgentService = Depends(get_chat_agent_service),
    chat_db: ChatDatabase = Depends(get_chat_db)
) -> StreamingResponse:
    """
    Send a message to the chat agent and stream the response
    
    Args:
        session_id: The chat session ID
        prompt: The user's message
        
    Returns:
        Streaming response with newline-delimited JSON messages
    """
    correlation_id = str(uuid4())
    start_time = time.time()
    
    async def stream_response():
        try:
            # Verify the session exists and get notebook_id
            session = await chat_db.get_session(session_id)
            if not session:
                error_json = json.dumps({
                    "error": f"Chat session {session_id} not found",
                    "status_code": 404
                })
                yield error_json.encode('utf-8') + b'\n'
                return
            
            if session_id not in chat_agent.sessions:
                await chat_agent.create_session(session_id, session["notebook_id"])
            
            # First, stream the user prompt so it can be displayed immediately
            user_msg = ChatMessage(
                role="user",
                content=prompt,
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent="user"
            )
            yield json.dumps(user_msg.model_dump()).encode('utf-8') + b'\n'
            
            # Get message history from the database
            messages = await chat_db.get_messages(session_id)
            
            # Use handle_message flow which includes clarification and investigation
            async for status_type, response in chat_agent.handle_message(
                prompt=prompt,
                session_id=session_id,
                message_history=messages
            ):
                # Convert the response to ChatMessage format
                agent = status_type
                if isinstance(response.parts[0], CellResponsePart):
                    agent = response.parts[0].agent_type
                chat_msg = to_chat_message(response, agent=agent)
                yield json.dumps(chat_msg.model_dump()).encode('utf-8') + b'\n'
                
                # Save the message to database
                await chat_db.add_message(session_id, response)
            
            process_time = time.time() - start_time
            chat_logger.info(
                f"Completed chat message stream",
                extra={
                    'correlation_id': correlation_id,
                    'session_id': session_id,
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
            
        except Exception as e:
            process_time = time.time() - start_time
            chat_logger.error(
                f"Error streaming chat response",
                extra={
                    'correlation_id': correlation_id,
                    'session_id': session_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_time_ms': round(process_time * 1000, 2)
                },
                exc_info=True
            )
            
            # Return error as JSON
            error_json = json.dumps({
                "error": f"Error streaming response: {str(e)}",
                "status_code": 500
            })
            yield error_json.encode('utf-8') + b'\n'
    
    chat_logger.info(
        f"Starting chat message stream",
        extra={
            'correlation_id': correlation_id,
            'session_id': session_id
        }
    )
    
    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    chat_db: ChatDatabase = Depends(get_chat_db)
) -> None:
    """
    Delete a chat session and all its messages
    
    Args:
        session_id: The chat session ID to delete
    """
    start_time = time.time()
    
    try:
        # Verify the session exists
        session = await chat_db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
        
        # Clear all messages
        await chat_db.clear_session_messages(session_id)
        
        process_time = time.time() - start_time
        chat_logger.info(
            f"Deleted chat session",
            extra={
                'correlation_id': str(uuid4()),
                'session_id': session_id,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
    except HTTPException:
        raise
    
    except Exception as e:
        process_time = time.time() - start_time
        chat_logger.error(
            f"Error deleting chat session",
            extra={
                'correlation_id': str(uuid4()),
                'session_id': session_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to delete chat session: {str(e)}")


@router.post("/chat")
async def unified_chat(
    request: ChatRequest,
    chat_agent: ChatAgentService = Depends(get_chat_agent_service),
    chat_db: ChatDatabase = Depends(get_chat_db)
) -> StreamingResponse:
    """
    Unified endpoint for chat that creates a session if not provided
    
    This is a convenience endpoint that combines session creation and messaging
    """
    correlation_id = str(uuid4())
    start_time = time.time()
    
    async def process_chat():
        try:
            session_id = request.session_id
            
            # If no session ID provided, create a new session
            if not session_id:
                if not request.notebook_id:
                    error_json = json.dumps({
                        "error": "notebook_id is required when creating a new session",
                        "status_code": 400
                    })
                    yield error_json.encode('utf-8') + b'\n'
                    return
                    
                session_id = str(uuid4())
                await chat_db.create_session(session_id, request.notebook_id)
                await chat_agent.create_session(session_id, request.notebook_id)
                
                # Return the session ID in the first message
                session_info = json.dumps({
                    "session_id": session_id
                })
                yield session_info.encode('utf-8') + b'\n'
            else:
                # Verify the session exists
                session = await chat_db.get_session(session_id)
                if not session:
                    error_json = json.dumps({
                        "error": f"Chat session {session_id} not found",
                        "status_code": 404
                    })
                    yield error_json.encode('utf-8') + b'\n'
                    return
                
                # If a different notebook_id is provided, return an error
                if request.notebook_id and request.notebook_id != session["notebook_id"]:
                    error_json = json.dumps({
                        "error": f"Cannot change notebook_id for existing session {session_id}",
                        "status_code": 400
                    })
                    yield error_json.encode('utf-8') + b'\n'
                    return
            
            # Stream the user prompt
            user_msg = ChatMessage(
                role="user",
                content=request.prompt,
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent="user"
            )
            yield json.dumps(user_msg.model_dump()).encode('utf-8') + b'\n'
            
            # Get message history
            messages = await chat_db.get_messages(session_id)
            
            # Use handle_message flow which includes clarification and investigation
            async for status_type, response in chat_agent.handle_message(
                prompt=request.prompt,
                session_id=session_id,
                message_history=messages
            ):
                # Convert the response to ChatMessage format
                agent = status_type
                if isinstance(response.parts[0], CellResponsePart):
                    agent = response.parts[0].agent_type
                chat_msg = to_chat_message(response, agent=agent)
                yield json.dumps(chat_msg.model_dump()).encode('utf-8') + b'\n'
                
                await chat_db.add_message(session_id, response)
                   
            
            process_time = time.time() - start_time
            chat_logger.info(
                f"Completed unified chat",
                extra={
                    'correlation_id': correlation_id,
                    'session_id': session_id,
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
            
        except Exception as e:
            process_time = time.time() - start_time
            chat_logger.error(
                f"Error in unified chat",
                extra={
                    'correlation_id': correlation_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_time_ms': round(process_time * 1000, 2)
                },
                exc_info=True
            )
            
            # Return error as JSON
            error_json = json.dumps({
                "error": f"Error in chat: {str(e)}",
                "status_code": 500
            })
            yield error_json.encode('utf-8') + b'\n'
    
    chat_logger.info(
        f"Starting unified chat processing",
        extra={
            'correlation_id': correlation_id
        }
    )
    
    return StreamingResponse(
        process_chat(),
        media_type="text/plain"
    ) 