"""
Media Timeline Agent

This agent processes media (video, images) to create a timeline of events,
correlate them with code, and provide a bug hypothesis.
It aims to produce a single, comprehensive cell output.
"""

from datetime import datetime, timezone
import logging
import asyncio
import json
import os
import tempfile
import aiohttp
from typing import Any, Dict, AsyncGenerator, Union, Optional, List, cast
from uuid import uuid4, UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, VideoUrl, ImageUrl

from backend.core.cell import CellType
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams
from backend.ai.step_agents import StepAgent
from backend.ai.models import InvestigationStepModel, PythonAgentInput, SafeOpenAIModel
from backend.ai.events import (
    EventType,
    AgentType,
    FinalStatusEvent,
    StatusType,
    StatusUpdateEvent,
    MediaTimelineCellCreatedEvent,
    FatalErrorEvent,
    BaseEvent
)
from sqlalchemy.ext.asyncio import AsyncSession
from enum import Enum
from pydantic_ai.providers.openai import OpenAIProvider

from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.services.connection_handlers.registry import get_handler
from pydantic_ai.mcp import MCPServerStdio
from backend.config import get_settings
from backend.services.notebook_manager import NotebookManager
from backend.ai.notebook_context_tools import create_notebook_context_tools
from backend.ai.prompts.correlator_prompts import CORRELATOR_SYSTEM_PROMPT

media_agent_logger = logging.getLogger("ai.media_agent")

# --- Data Models for Media Processing and Correlation ---

class UrlType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"

class MediaUrl(BaseModel):
    url: str
    type: UrlType

class MediaAgentInput(BaseModel):
    original_query: str
    notebook_id: str
    session_id: str
    urls: List[MediaUrl]

class MediaUrlExtractionFailed(Exception):
    """Exception raised when media URLs cannot be extracted from the original query."""

class FrameAnalysisData(BaseModel):
    image_identifier: str  # e.g., path or frame number or URL of the static image
    visible_text: List[str] = Field(default_factory=list)
    ui_elements: List[Dict[str, Any]] = Field(default_factory=list) # Generic for now
    potential_errors: List[str] = Field(default_factory=list)
    # Add other relevant fields like dominant colors, detected objects if needed

class MediaTimelineEvent(BaseModel):
    image_identifier: str  # Link back to the frame/image
    description: str  # What happened in this frame from AI analysis
    code_references: List[str] = Field(default_factory=list) # List of "file.py:line"

class MediaTimelinePayload(BaseModel):
    hypothesis: str
    timeline_events: List[MediaTimelineEvent]
    original_query: str

# --- End Data Models ---


class MediaUrlsExtractorAgent:
    def __init__(self, notebook_id: str, connection_manager: ConnectionManager, notebook_manager: Optional[NotebookManager] = None):
        self.notebook_id = notebook_id
        self.connection_manager = connection_manager
        self.settings = get_settings()
        self.model = SafeOpenAIModel(
                "openai/gpt-4.1-mini",
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.agent = None
        self.notebook_manager: Optional[NotebookManager] = notebook_manager
        media_agent_logger.info(f"MediaUrlsExtractorAgent class initialized (model configured, agent instance created per run).")

    def _read_system_prompt(self) -> str:
        system_prompt = """

        You are a helpful assistant whose primary
        task is to extract media URLs from either a user query or a notebook cell.
        You will be given a query and tools to help you retrieve notebook cells and their content.

        You can also use tools to get information from github or the filesystem.

        The urls you extract are primarily bug reports, so they will be video or image files.
        We only want urls that are media files, not text files.
        
        The following are examples of how you should behave and not what the actual scenario might be:
        Example:
        User query: "Original query: I am trying to fix a bug with the login page. I think it has to do with the password input field."
        Notebook cell: "I think the problem is that the password input field is not visible. Result: https://example.com/login_bug.mp4" 
        Output: ["https://example.com/login_bug.mp4"]
        
        User query: "Original query: I am trying to fix a bug with the login page. I think it has to do with the password input field."
        Github search: "I think the problem is that the password input field is not visible. Result: https://github.com/username/repo/blob/main/path/to/video.mp4"
        Output: ["https://github.com/username/repo/blob/main/path/to/video.mp4"]

        # No media urls found
        User query: "Original query: I am trying to fix a bug with the login page. I think it has to do with the password input field."
        Github search: "I think the problem is that the password input field is not visible. Result: https://github.com/username/repo/blob/main/path/to/file.py"
        Output: []

        Your job is to get media urls and their types, based on the query and the tools you have available. 
        If no media urls are found, return nothing.

        """

        return system_prompt
        
    async def _get_mcp_server(self, connection_type: str) -> Optional[MCPServerStdio]:
        """Helper to get an MCP server instance for a given connection type."""
        try:
            default_conn = await self.connection_manager.get_default_connection(connection_type)
            if not default_conn:
                media_agent_logger.error(f"No default {connection_type} connection found.")
                return None
            if not default_conn.config:
                media_agent_logger.error(f"Default {connection_type} connection {default_conn.id} has no config.")
                return None

            handler = get_handler(connection_type)
            stdio_params = handler.get_stdio_params(default_conn.config)
            media_agent_logger.info(f"Retrieved StdioServerParameters for {connection_type}. Command: {stdio_params.command}")
            return MCPServerStdio(command=stdio_params.command, args=stdio_params.args, env=stdio_params.env)
        except ValueError as ve:
            media_agent_logger.error(f"Configuration error getting {connection_type} stdio params: {ve}", exc_info=True)
        except Exception as e:
            media_agent_logger.error(f"Error creating MCPServerStdio instance for {connection_type}: {e}", exc_info=True)
        return None
        
    
    async def _initialize_agent(self, extra_tools: Optional[list] = None) -> Agent[None, Union[MediaAgentInput, MediaUrlExtractionFailed]]:
        system_prompt = self._read_system_prompt()
        if not self.github_mcp_server:
            self.github_mcp_server = await self._get_mcp_server("github")
        if not self.filesystem_mcp_server:
            self.filesystem_mcp_server = await self._get_mcp_server("filesystem")
        
        mcp_servers_for_agent = []
        if self.github_mcp_server: mcp_servers_for_agent.append(self.github_mcp_server)
        if self.filesystem_mcp_server: mcp_servers_for_agent.append(self.filesystem_mcp_server)

        notebook_tools = create_notebook_context_tools(self.notebook_id, self.notebook_manager)

        agent = Agent[None, Union[MediaAgentInput, MediaUrlExtractionFailed]](
            model=self.model,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers_for_agent,
            tools=notebook_tools,
            output_type=Union[MediaAgentInput, MediaUrlExtractionFailed] # type: ignore
        )
        media_agent_logger.info(f"MediaUrlsExtractorAgent initialized with {len(extra_tools) if extra_tools else 0} tools.")
        return agent

    async def run_query(self, query: str) -> Union[MediaAgentInput, MediaUrlExtractionFailed]:
        
        media_agent_logger.info(f"Running query: {query}")
        media_agent_logger.info(f"Agent: {self.agent}")

        self.agent = await self._initialize_agent()

        async with self.agent.run_mcp_servers():
            result = await self.agent.run(query)
        return result.output


class MediaCorrelatorAgent:
    def __init__(self, notebook_id: str, connection_manager: ConnectionManager, notebook_manager: Optional[NotebookManager] = None):
        self.notebook_id = notebook_id
        media_agent_logger.info(f"Initializing MediaCorrelatorAgent for notebook: {self.notebook_id}")
        self.connection_manager = connection_manager
        self.settings = get_settings()
        self.notebook_manager: Optional[NotebookManager] = notebook_manager

    async def _get_mcp_server(self, connection_type: str) -> Optional[MCPServerStdio]:
        """Helper to get an MCP server instance for a given connection type."""
        try:
            default_conn = await self.connection_manager.get_default_connection(connection_type)
            if not default_conn:
                media_agent_logger.error(f"No default {connection_type} connection found.")
                return None
            if not default_conn.config:
                media_agent_logger.error(f"Default {connection_type} connection {default_conn.id} has no config.")
                return None

            handler = get_handler(connection_type)
            stdio_params = handler.get_stdio_params(default_conn.config)
            media_agent_logger.info(f"Retrieved StdioServerParameters for {connection_type}. Command: {stdio_params.command}")
            return MCPServerStdio(command=stdio_params.command, args=stdio_params.args, env=stdio_params.env)
        except ValueError as ve:
            media_agent_logger.error(f"Configuration error getting {connection_type} stdio params: {ve}", exc_info=True)
        except Exception as e:
            media_agent_logger.error(f"Error creating MCPServerStdio instance for {connection_type}: {e}", exc_info=True)
        return None

    async def _initialize_correlator_agent(self) -> Optional[Agent[None, MediaTimelinePayload]]:
        media_agent_logger.info("Initializing MediaCorrelatorAgent with real MCPs and LLM...")

        notebook_tools = create_notebook_context_tools(self.notebook_id, self.notebook_manager)

        if not self.github_mcp_server:
            self.github_mcp_server = await self._get_mcp_server("github")
        if not self.filesystem_mcp_server:
            self.filesystem_mcp_server = await self._get_mcp_server("filesystem")
        
        mcp_servers_for_agent = []
        if self.github_mcp_server: mcp_servers_for_agent.append(self.github_mcp_server)
        if self.filesystem_mcp_server: mcp_servers_for_agent.append(self.filesystem_mcp_server)

        if not mcp_servers_for_agent:
            media_agent_logger.error("Cannot initialize correlator agent: No MCP servers (GitHub/Filesystem) could be configured.")
            return None
        
        self.model = SafeOpenAIModel(
            "google/gemini-2.5-pro-preview",
            provider=OpenAIProvider(
                base_url='https://openrouter.ai/api/v1',
                api_key=self.settings.openrouter_api_key,
            ),
            )
        self.correlator_agent = Agent[None, MediaTimelinePayload](
            self.model,
            system_prompt=CORRELATOR_SYSTEM_PROMPT,
            mcp_servers=mcp_servers_for_agent,
            output_type=MediaTimelinePayload,
            tools=notebook_tools
        )
        media_agent_logger.info(f"MediaCorrelatorAgent initialized successfully with {len(mcp_servers_for_agent)} MCPs.")


    async def run_correlator(self, original_query: str, urls: List[MediaUrl]) -> Optional[MediaTimelinePayload]:
        self.correlator_agent = await self._initialize_correlator_agent()
        if not self.correlator_agent:
            media_agent_logger.error("Correlator agent not initialized.")
            return None
        
        original_query_context = f"Original user query: {original_query}"

        media_urls = []

        for url in urls:
            if url.type == UrlType.VIDEO:
                media_urls.append(VideoUrl(url=url.url))
            elif url.type == UrlType.IMAGE:
                media_urls.append(ImageUrl(url=url.url))
            else:
                media_agent_logger.error(f"Unknown media type: {url.type}")
                continue

        async with self.correlator_agent.run_mcp_servers():
            result = await self.correlator_agent.run([original_query_context, *media_urls])
        return result.output
            
            

class MediaTimelineAgent:
    def __init__(self, notebook_id: str, session_id: str, connection_manager: ConnectionManager, notebook_manager: Optional[NotebookManager] = None):
        self.notebook_id = notebook_id
        self.connection_manager = connection_manager
        self.notebook_manager = notebook_manager
        self.session_id = session_id
        self.urls_extractor_agent = MediaUrlsExtractorAgent(notebook_id, connection_manager, notebook_manager)
        self.correlator_agent = MediaCorrelatorAgent(notebook_id, connection_manager, notebook_manager)

    async def run_query(
        self,
        description: str
    ) -> AsyncGenerator[Union[StatusUpdateEvent, FinalStatusEvent, MediaTimelinePayload], None]:
        media_agent_logger.info(f"Running MediaTimelineAgent for notebook: {self.notebook_id}")
        media_agent_logger.info(f"Description: {description}")

        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.STARTING,
            agent_type=AgentType.MEDIA_TIMELINE,
            message="Initializing Media timeline...",
            attempt=None,
            max_attempts=None,
            reason=None,
            step_id=None,
            original_plan_step_id=None,
            session_id=self.session_id,
            notebook_id=self.notebook_id
        )


        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        base_description = f"Current time is {current_time_utc}. User query: {description}"
    
        result = await self.urls_extractor_agent.run_query(base_description)
        if isinstance(result, MediaUrlExtractionFailed):
            media_agent_logger.error(f"Error extracting media URLs: {result}")
            yield FinalStatusEvent(
                 type=EventType.FINAL_STATUS,
                 status=StatusType.FINISHED_NO_RESULT,
                 agent_type=AgentType.MEDIA_TIMELINE,
                 attempts=None,
                 message=None,
                 session_id=self.session_id,
                 notebook_id=self.notebook_id
             )
            return
        
        urls = result.urls
        media_agent_logger.info(f"Extracted {len(urls)} media URLs: {urls}")

        correlator_result = await self.correlator_agent.run_correlator(description, urls)
        if correlator_result is None:
            media_agent_logger.error("Error running correlator.")
            yield StatusUpdateEvent(
                type=EventType.STATUS_UPDATE,
                status=StatusType.COMPLETE,
                agent_type=AgentType.MEDIA_TIMELINE,
                message="Error running correlator",
                session_id=self.session_id,
                notebook_id=self.notebook_id,
                attempt=None,
                max_attempts=None,
                reason="Error running correlator",
                step_id=None,
                original_plan_step_id=None
            )
            return
        
        media_agent_logger.info(f"Correlator result: {correlator_result}")

        yield correlator_result

        