"""
Chat API Endpoints

This module implements the chat API endpoints for interacting with the AI agent.
"""

import json
import logging
import time
from typing import Dict, List, Tuple
from uuid import uuid4, UUID
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)

from backend.ai.chat_agent import (
    ChatAgentService,
    ChatMessage,
    to_chat_message,
)
from backend.db.chat_db import ChatDatabase
from backend.services.notebook_manager import NotebookManager, get_notebook_manager
from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.services.redis_client import get_redis_client
from backend.config import Settings, get_settings
import aioredis

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

# Updated Dependency: Get managers, not the service directly
async def get_managers(
    notebook_manager: NotebookManager = Depends(get_notebook_manager),
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> Tuple[NotebookManager, ConnectionManager]:
    return notebook_manager, connection_manager

# Dependency to get chat database
async def get_chat_db(request: Request) -> ChatDatabase:
    """Get the chat database from the app state"""
    return request.app.state.chat_db

MAX_CELL_HISTORY = 5
MAX_SUMMARY_LENGTH = 150 # Max chars for code/output summaries

def summarize_cell(cell_dict: Dict) -> str:
    """Creates a concise summary string for a notebook cell dictionary."""
    cell_id = cell_dict.get('id', 'N/A')
    cell_type = cell_dict.get('type', 'unknown')
    status = cell_dict.get('status', cell_dict.get('metadata', {}).get('status', 'unknown')) 
    summary = f"[Cell ID: {cell_id}, Type: {cell_type}, Status: {status}"

    content = cell_dict.get('content', '')
    if content:
        if len(content) > MAX_SUMMARY_LENGTH:
            content_summary = content[:MAX_SUMMARY_LENGTH] + "..."
        else:
            content_summary = content
        # Clean summary before adding to f-string
        cleaned_content_summary = content_summary.replace("\n", " ")
        summary += f', Content: \'{cleaned_content_summary}\''

    result = cell_dict.get('result')
    if isinstance(result, dict):
        error = result.get('error')
        output = result.get('output', result.get('content')) 
        if error:
             error_str = str(error)
             if len(error_str) > MAX_SUMMARY_LENGTH:
                 error_summary = error_str[:MAX_SUMMARY_LENGTH] + "..."
             else:
                 error_summary = error_str
             cleaned_error = error_summary.replace("\n", " ")
             summary += f', Error: {cleaned_error}'
        elif output:
            output_str = str(output)
            if len(output_str) > MAX_SUMMARY_LENGTH:
                 output_summary = output_str[:MAX_SUMMARY_LENGTH] + "..."
            else:
                 output_summary = output_str
            cleaned_output = output_summary.replace("\n", " ")
            summary += f', Result: {cleaned_output}'

    summary += "]"
    return summary

class CreateSessionRequest(BaseModel):
    """Request to create a new chat session"""
    notebook_id: str = Field(description="The ID of the notebook to associate with the chat session")


class CreateSessionResponse(BaseModel):
    """Response for creating a new chat session"""
    session_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    request_data: CreateSessionRequest, # Renamed from 'request' to avoid conflict
    chat_db: ChatDatabase = Depends(get_chat_db),
    redis: aioredis.Redis = Depends(get_redis_client), # Inject Redis client
    settings: Settings = Depends(get_settings) # Inject Settings for TTL
) -> Dict:
    """
    Create a new chat session record in DB and store essential info in Redis
    
    Returns:
        The session ID
    """
    start_time = time.time()
    
    try:
        # Generate a new session ID
        session_id = str(uuid4())
        notebook_id = request_data.notebook_id # Use request_data

        # Create the session in the database
        await chat_db.create_session(session_id, notebook_id) # Use notebook_id

        # Store essential info (notebook_id) in Redis with TTL
        redis_key = f"chat_session:{session_id}:notebook_id"
        await redis.set(redis_key, notebook_id, ex=settings.chat_session_ttl)
        chat_logger.info(f"Stored session info in Redis for session {session_id} with TTL {settings.chat_session_ttl}s")

        process_time = time.time() - start_time
        chat_logger.info(
            f"Created new chat session record in DB and Redis", # Updated log message
            extra={
                'correlation_id': str(uuid4()),
                'session_id': session_id,
                'notebook_id': notebook_id, 
                'redis_key': redis_key,
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
        # Consider closing Redis connection if managed manually
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
    chat_db: ChatDatabase = Depends(get_chat_db),
    managers: Tuple[NotebookManager, ConnectionManager] = Depends(get_managers), # Inject managers
    redis: aioredis.Redis = Depends(get_redis_client), # Inject Redis client
    settings: Settings = Depends(get_settings) # Inject Settings for TTL
) -> StreamingResponse:
    """
    Send a message to the chat agent and stream the response.
    Uses Redis for session state lookup.
    """
    start_time = time.time()
    request_id = str(uuid4())
    notebook_id_str = None
    notebook_id = None
    
    # --- Retrieve session state (notebook_id) ---
    redis_key = f"chat_session:{session_id}:notebook_id"
    try:
        notebook_id_str = await redis.get(redis_key)
        
        if notebook_id_str:
            chat_logger.info(f"Cache hit for session {session_id} notebook_id in Redis.")
            # Refresh TTL on cache hit
            await redis.expire(redis_key, settings.chat_session_ttl)
        else:
            chat_logger.info(f"Cache miss for session {session_id} notebook_id in Redis. Checking DB.")
            # Cache miss, try DB
            session_data = await chat_db.get_session(session_id)
            if not session_data:
                chat_logger.warning(f"Session {session_id} not found in database either.")
                raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
            
            # Extract notebook_id from DB data (adjust based on your DB model)
            if isinstance(session_data, dict):
                notebook_id_str = session_data.get('notebook_id')
            else: # Assume object
                notebook_id_str = getattr(session_data, 'notebook_id', None)

            if not notebook_id_str:
                 chat_logger.error(f"Session {session_id} found in DB, but notebook_id is missing.")
                 raise HTTPException(status_code=500, detail="Internal server error: Session data incomplete.")
                 
            # Store back in Redis
            await redis.set(redis_key, notebook_id_str, ex=settings.chat_session_ttl)
            chat_logger.info(f"Populated Redis cache for session {session_id} from DB.")

        # Convert notebook_id_str to UUID
        notebook_id = UUID(notebook_id_str)

    except ValueError: # Catch UUID conversion error
         chat_logger.error(f"Invalid notebook_id format '{notebook_id_str}' found for session {session_id} (Redis/DB).")
         raise HTTPException(status_code=500, detail="Internal server error: Invalid notebook ID in session data.")
    except aioredis.RedisError as e:
        chat_logger.error(f"Redis error retrieving session {session_id}: {e}", exc_info=True)
        # Potentially fall back to DB only, or return error
        raise HTTPException(status_code=503, detail="Could not connect to session cache.")
    except HTTPException: # Re-raise specific HTTP exceptions
        raise
    except Exception as e:
        chat_logger.error(f"Error retrieving session details for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve session details.")
    # --- End Retrieve session state ---


    # Get message history from DB (remains the same)
    try:
        history_from_db = await chat_db.get_messages(session_id)
    except Exception as e:
        chat_logger.error(f"Error retrieving history for session {session_id}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve message history.")

    # --- Instantiate ChatAgentService on-demand ---
    notebook_manager, connection_manager = managers # Unpack managers
    chat_agent = ChatAgentService(
        notebook_manager=notebook_manager,
        connection_manager=connection_manager
    )
    # Initialize the agent for this specific session context using the retrieved notebook_id_str
    try:
        await chat_agent.create_session(session_id, notebook_id_str) 
        chat_logger.info(f"Instantiated and initialized ChatAgentService for session {session_id} on demand using notebook_id {notebook_id_str}.")
    except Exception as agent_init_error:
        chat_logger.error(f"Failed to initialize ChatAgentService for session {session_id}: {agent_init_error}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize chat agent.")
    # --- End Instantiate ChatAgentService ---

    # Agent initialization now handles associating notebook_id internally, remove redundant checks here

    # --- Fetch and Summarize Cell History ---
    # Use the UUID notebook_id retrieved earlier
    cell_history_context = "No recent cell history found."
    try:
        notebook_manager = chat_agent.notebook_manager # Get manager from agent
        notebook = notebook_manager.get_notebook(notebook_id) # Fetch the notebook object
        if notebook and notebook.cell_order:
            recent_cell_ids = notebook.cell_order[-MAX_CELL_HISTORY:]
            # Use model_dump() which is standard in Pydantic v2
            recent_cells_dicts = [notebook.cells[cell_id].model_dump() 
                                  for cell_id in recent_cell_ids 
                                  if cell_id in notebook.cells] 
            summaries = [summarize_cell(cell_dict) for cell_dict in recent_cells_dicts]
            if summaries:
                 cell_history_context = "Recent Cell History:\n" + "\n".join(summaries)
            else:
                 cell_history_context = "Found notebook but no recent cells to summarize."
        elif notebook:
             cell_history_context = "Notebook found, but it has no cells."
             
        chat_logger.debug(f"Generated cell history context for session {session_id}:\n{cell_history_context}")
    except KeyError: 
         chat_logger.warning(f"Notebook {notebook_id} not found when fetching cell history.")
         cell_history_context = "Could not retrieve notebook for cell history."
    except Exception as e:
        chat_logger.error(f"Error fetching or summarizing cell history for notebook {notebook_id}: {e}", exc_info=True)
        cell_history_context = "Error retrieving recent cell history."
    
    # Prepend cell history context to the user's prompt
    prompt_with_context = f"{cell_history_context}\n\nUser Prompt:\n{prompt}"
    # --- End Fetch and Summarize Cell History ---

    # Construct the user message object using the combined prompt
    # Timestamp added automatically by UserPromptPart if not provided
    user_message = ModelRequest(parts=[UserPromptPart(content=prompt_with_context)]) 
    
    # The agent receives history *including* the user message with prepended context
    # We save the user_message (containing context) to DB
    message_history_for_agent = history_from_db # History *before* the current message

    # Save the user message (with context) to the database asynchronously
    async def save_user_message():
        try:
            await chat_db.add_message(session_id, user_message)
        except Exception as e:
            chat_logger.error(f"Error saving user message for session {session_id}", exc_info=True)

    await save_user_message() 

    chat_logger.info(
        f"Processing message for session {session_id}",
        extra={
            'correlation_id': request_id,
            'session_id': session_id,
            'prompt': prompt,
            'history_length': len(history_from_db), # Log length before adding current prompt
            'context_added': True # Indicate cell context was added
        }
    )

    async def stream_response():
        nonlocal start_time # Allow modification for response time logging
        message_buffer: List[Tuple[str, ModelResponse]] = [] # Buffer for DB saving

        try:
            # Pass history *before* the current user message
            async for status_type, response_part in chat_agent.handle_message(
                prompt=prompt_with_context, # Pass the prompt string WITH context
                session_id=session_id,
                message_history=message_history_for_agent 
            ):
                message_buffer.append((status_type, response_part))
                
                # Convert ModelResponse to ChatMessage for streaming
                chat_message = to_chat_message(response_part)
                
                # Ensure chat_message content is JSON serializable before streaming
                try:
                    stream_content = json.dumps(chat_message.model_dump()) + "\\n"
                    yield stream_content
                    chat_logger.debug(f"Streamed part: {stream_content.strip()} for session {session_id}")
                except TypeError as e:
                    chat_logger.error(f"Serialization error for chat message part: {e} - {chat_message}", exc_info=True)
                    # Yield an error message or skip? For now, skip.
                    yield json.dumps({"role": "error", "content": f"Serialization error: {e}", "timestamp": datetime.now(timezone.utc).isoformat()}) + "\\n"

            # After streaming finishes, save all buffered model responses to DB
            for _, response_to_save in message_buffer:
                 try:
                    await chat_db.add_message(session_id, response_to_save)
                 except Exception as e:
                     chat_logger.error(f"Error saving buffered model response for session {session_id}", exc_info=True)
                     # Log error but continue saving others

            process_time = time.time() - start_time
            chat_logger.info(
                f"Completed chat message stream",
                extra={
                    'correlation_id': request_id,
                    'session_id': session_id,
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )

        except Exception as e:
            process_time = time.time() - start_time
            chat_logger.error(
                f"Error during chat stream",
                extra={
                    'correlation_id': request_id,
                    'session_id': session_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_time_ms': round(process_time * 1000, 2)
                },
                exc_info=True
            )
            # Stream an error message to the client
            error_message = ChatMessage(
                role="error", # Custom role for errors
                content=f"An error occurred: {str(e)}",
                timestamp=datetime.now(timezone.utc).isoformat()
            ).model_dump()
            try:
                yield json.dumps(error_message) + "\\n"
            except Exception as serialization_error:
                 chat_logger.error(f"Failed to serialize error message: {serialization_error}", exc_info=True)


    return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    chat_db: ChatDatabase = Depends(get_chat_db),
    redis: aioredis.Redis = Depends(get_redis_client) # Inject Redis client
) -> None:
    """
    Delete a chat session from DB and Redis
    
    Args:
        session_id: The chat session ID to delete
    """
    start_time = time.time()
    
    try:
        # 1. Delete from Redis (best effort, ignore if key doesn't exist)
        redis_key = f"chat_session:{session_id}:notebook_id"
        try:
            deleted_count = await redis.delete(redis_key)
            if deleted_count > 0:
                 chat_logger.info(f"Deleted session info from Redis for session {session_id}")
            else:
                 chat_logger.info(f"Session info for {session_id} not found in Redis (already expired or deleted).")
        except aioredis.RedisError as e:
             chat_logger.error(f"Redis error deleting key {redis_key} for session {session_id}: {e}", exc_info=True)
             # Decide if this should halt the process or just be logged

        # 2. Verify the session exists in DB before cleaning messages
        session = await chat_db.get_session(session_id)
        if not session:
            # If not in DB, and wasn't in Redis (or Redis failed), it's effectively gone.
            # Still return 204 as the goal is deletion.
            chat_logger.warning(f"Chat session {session_id} not found in DB during delete operation.")
            # Optionally raise 404 if strict existence is required:
            # raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
            return # Exit early if not found in DB

        # 3. Clear messages from DB
        await chat_db.clear_session_messages(session_id)
        
        # 4. Delete session record from DB (assuming you have a method for this)
        # Example: await chat_db.delete_session(session_id)
        # If `clear_session_messages` implies deletion, this might not be needed.
        # Add DB session deletion logic here if necessary.
        chat_logger.info(f"Cleared messages for session {session_id} from DB.")

        process_time = time.time() - start_time
        chat_logger.info(
            f"Deleted chat session state from Redis and DB", # Updated log
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
            f"Error deleting chat session {session_id}", # Updated log
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


# Removed commented-out unified_chat endpoint for now

# Removed commented-out unified_chat endpoint for now 