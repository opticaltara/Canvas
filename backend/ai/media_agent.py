"""
Media Timeline Agent

This agent processes media (video, images) to create a timeline of events,
correlate them with code, and provide a bug hypothesis.
It aims to produce a single, comprehensive cell output.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, AsyncGenerator, Union, Optional, List
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider

from backend.services.connection_manager import ConnectionManager
from backend.services.connection_handlers.registry import get_handler
from pydantic_ai.mcp import MCPServerStdio
from backend.config import get_settings
from backend.services.notebook_manager import NotebookManager
from backend.ai.notebook_context_tools import create_notebook_context_tools
from backend.ai.prompts.correlator_prompts import CORRELATOR_SYSTEM_PROMPT
from backend.ai.models import SafeOpenAIModel
from backend.ai.events import (
    EventType,
    AgentType,
    FinalStatusEvent,
    StatusType,
    StatusUpdateEvent,
)

# No direct binary downloads in this module anymore – handled by MediaDescriberAgent.

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

class MediaUrlExtractionError(BaseModel):
    message: str = "Media URL extraction failed"

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

# -------------------------------------------------------------------
# NEW – Correlator output data models
# -------------------------------------------------------------------

class CorrelatorHypothesis(BaseModel):
    """Detailed hypothesis object returned by the correlator."""
    files_likely_containing_bug: List[str] = Field(
        default_factory=list,
        description="Files the correlator believes most likely contain the bug.",
    )
    reasoning: List[str] = Field(
        default_factory=list,
        description="Bullet-point reasoning supporting the hypothesis.",
    )
    specific_code_snippets_to_check: List[str] = Field(
        default_factory=list,
        description="Concrete code snippets or identifiers that should be inspected.",
    )

class CorrelatorResult(BaseModel):
    """Top-level structured payload produced by the correlator."""
    hypothesis: CorrelatorHypothesis
    timeline_events: List[MediaTimelineEvent]
    # The original_query is optional because legacy correlator prompts may omit it.
    original_query: Optional[str] = None

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
        self.github_mcp_server = None
        self.filesystem_mcp_server = None

    def _read_system_prompt(self) -> str:
        system_prompt = """

        You are a helpful assistant whose primary
        task is to extract media URLs from either a user query or a notebook cell.
        You will be given a query and tools to help you retrieve notebook cells and their content.

        You can also use tools to get information from github or the filesystem.

        The urls you extract are primarily bug reports, so they will be video or image files.
        We only want urls that are media files, not text files.

        **Important:** Your primary task is to *extract* URLs that appear to be media files (videos, images). You should identify these based on the URL structure or file extension (e.g., .mp4, .mov, .jpg, .png, .gif, etc., including URLs from services like `github.com/user-attachments/`). **DO NOT** attempt to use any tools (like GitHub file readers, filesystem readers, or web fetchers) to validate, access, or fetch the content of these media URLs. Your only job regarding these URLs is to extract the URL string itself. Focus on identifying and returning the strings.
        
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
        
    
    async def _initialize_agent(self, extra_tools: Optional[list] = None) -> Agent[None, Union[MediaAgentInput, MediaUrlExtractionError]]:
        system_prompt = self._read_system_prompt()
        if not self.github_mcp_server:
            self.github_mcp_server = await self._get_mcp_server("github")
        if not self.filesystem_mcp_server:
            self.filesystem_mcp_server = await self._get_mcp_server("filesystem")
        
        mcp_servers_for_agent = []
        if self.github_mcp_server: mcp_servers_for_agent.append(self.github_mcp_server)
        if self.filesystem_mcp_server: mcp_servers_for_agent.append(self.filesystem_mcp_server)

        notebook_tools = create_notebook_context_tools(self.notebook_id, self.notebook_manager)

        agent = Agent[None, Union[MediaAgentInput, MediaUrlExtractionError]](
            model=self.model,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers_for_agent,
            tools=notebook_tools,
            output_type=MediaAgentInput | MediaUrlExtractionError # type: ignore
        )
        media_agent_logger.info(f"MediaUrlsExtractorAgent initialized with {len(extra_tools) if extra_tools else 0} tools.")
        return agent

    async def run_query(self, query: str) -> Union[MediaAgentInput, MediaUrlExtractionError]:
        
        media_agent_logger.info(f"Running query for URL extraction: {query}")

        self.agent = await self._initialize_agent()
        if not self.agent: # Check if agent initialization failed
             media_agent_logger.error("Failed to initialize the MediaUrlsExtractorAgent.")
             return MediaUrlExtractionError(message="Agent initialization failed.")

        media_agent_logger.info(f"URL Extractor Agent: {self.agent}")

        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            media_agent_logger.info(f"URL extractor agent attempt {attempt}/{max_attempts} for query: {query}")
            try:
                async with self.agent.run_mcp_servers():
                    result = await self.agent.run(query)
                media_agent_logger.info(
                    f"URL extractor agent finished successfully on attempt {attempt}. Query: {query} | Output: {result.output}",
                )
                return result.output
            except Exception as e:
                last_exception = e
                media_agent_logger.warning(
                    f"URL extractor agent attempt {attempt} failed. Query: {query} | Error: {e}",
                    exc_info=True, # Log traceback for warnings too
                )
                if attempt < max_attempts:
                    await asyncio.sleep(1) # Wait 1 second before retrying

        # If all attempts failed
        media_agent_logger.exception(
            f"URL extractor agent failed after {max_attempts} attempts. Query: {query} | Last Error: {last_exception}",
            exc_info=last_exception, # Log the last exception with traceback
        )
        return MediaUrlExtractionError(message=str(last_exception))


class MediaCorrelatorAgent:
    def __init__(self, notebook_id: str, connection_manager: ConnectionManager, notebook_manager: Optional[NotebookManager] = None):
        self.notebook_id = notebook_id
        media_agent_logger.info(f"Initializing MediaCorrelatorAgent for notebook: {self.notebook_id}")
        self.connection_manager = connection_manager
        self.settings = get_settings()
        self.notebook_manager: Optional[NotebookManager] = notebook_manager
        self.github_mcp_server = None
        self.filesystem_mcp_server = None

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

    async def _initialize_correlator_agent(self) -> Optional[Agent[None, CorrelatorResult]]:
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
            "openai/gpt-4.1",
            provider=OpenAIProvider(
                base_url='https://openrouter.ai/api/v1',
                api_key=self.settings.openrouter_api_key,
            ),
            )
        self.correlator_agent = Agent[None, CorrelatorResult](
            self.model,
            system_prompt=CORRELATOR_SYSTEM_PROMPT,
            mcp_servers=mcp_servers_for_agent,
            output_type=CorrelatorResult,
            tools=notebook_tools
        )
        media_agent_logger.info(f"MediaCorrelatorAgent initialized successfully with {len(mcp_servers_for_agent)} MCPs.")

        # Return the fully configured correlator agent so callers can use it
        return self.correlator_agent

    async def run_correlator(self, timeline_payload: MediaTimelinePayload, context: Dict[str, Any]) -> Optional[CorrelatorResult]:
        """Run the correlator with a text-only timeline payload."""
        self.correlator_agent = await self._initialize_correlator_agent()
        if not self.correlator_agent:
            media_agent_logger.error("Correlator agent not initialized.")
            return None

        # Convert the payload to a string that can fit in a single LLM message.
        timeline_json = timeline_payload.model_dump_json(indent=2)
        
        # --- Incorporate context --- 
        context_items_xml_parts = []
        if context: # Check if context exists and is not empty
            for key, value in context.items():
                # Escape CDATA closing sequence in context values
                safe_value = str(value).replace("]]>'", "]] ]]> <![CDATA[") # Basic escaping
                context_items_xml_parts.append(f'        <item key="{key}"><![CDATA[{safe_value}]]></item>')
        context_items_xml = "\n".join(context_items_xml_parts)
        # --- End Incorporate context ---

        context_payload = (
            f"<task_with_context>\n"
            f"    <original_user_query><![CDATA[{timeline_payload.original_query}]]></original_user_query>\n"
        )
        if context_items_xml: # Add context block only if there's context
            context_payload += (
                f"    <shared_context>\n"
                f"{context_items_xml}\n"
                f"    </shared_context>\n"
            )
        context_payload += (
            f"    <vision_model_timeline_analysis>\n"
            f"        <![CDATA[{timeline_json}]]>\n"
            f"    </vision_model_timeline_analysis>\n"
            f"</task_with_context>"            
        )

        media_agent_logger.info(f"Context payload for correlator: {context_payload}")
        try:
            async with self.correlator_agent.run_mcp_servers():
                result = await self.correlator_agent.run(context_payload)
            # Safely log the size of the returned payload without assuming it's a sequence.
            output_obj = result.output
            try:
                if isinstance(output_obj, BaseModel):
                    size_hint = len(output_obj.model_dump_json())
                else:
                    size_hint = len(str(output_obj))
            except Exception:  # pragma: no cover
                size_hint = "unknown"

            media_agent_logger.debug(
                "Correlator agent finished successfully. Approx. output length: %s",
                size_hint,
            )

            # The agent should already return a CorrelatorResult, but be defensive
            if isinstance(output_obj, CorrelatorResult):
                return output_obj

            # Fallback: attempt to parse JSON embedded in string output
            parsed_output = self._parse_correlator_output(str(output_obj))
            return parsed_output
        except Exception as e:
            media_agent_logger.exception(
                "Correlator agent raised an exception. Error: %s",
                e,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_correlator_output(self, raw_output: str) -> Optional[CorrelatorResult]:
        """Attempt to extract JSON from the correlator's raw output and
        convert it into a CorrelatorResult object.

        The correlator sometimes wraps the JSON payload in additional markdown
        or commentary. We perform a simple heuristic extraction between the
        first '{' and the last '}' characters.
        """
        import json  # Local import to avoid adding to global import section unnecessarily

        try:
            start = raw_output.find('{')
            end = raw_output.rfind('}')
            if start == -1 or end == -1:
                media_agent_logger.error("Unable to locate JSON object in correlator output.")
                return None

            json_str = raw_output[start : end + 1]
            data = json.loads(json_str)
            return CorrelatorResult.model_validate(data)
        except Exception as parse_exc:
            media_agent_logger.error("Failed to parse correlator JSON output: %s", parse_exc, exc_info=True)
            return None


class MediaTimelineAgent:
    def __init__(self, notebook_id: str, session_id: str, connection_manager: ConnectionManager, notebook_manager: Optional[NotebookManager] = None):
        self.notebook_id = notebook_id
        self.connection_manager = connection_manager
        self.notebook_manager = notebook_manager
        self.session_id = session_id
        self.urls_extractor_agent = MediaUrlsExtractorAgent(notebook_id, connection_manager, notebook_manager)
        self.describer_agent = None
        self.correlator_agent = MediaCorrelatorAgent(notebook_id, connection_manager, notebook_manager)

    async def run_query(
        self,
        description: str,
        context: Dict[str, Any]  # Added context parameter
    ) -> AsyncGenerator[Union[StatusUpdateEvent, FinalStatusEvent, CorrelatorResult], None]:
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
        if isinstance(result, MediaUrlExtractionError):
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

        # ------------------------------------------------------------------
        # NEW: Describe media first via vision model
        # ------------------------------------------------------------------
        try:
            # Lazy import to avoid circular dependency
            from backend.ai.media_describer_agent import MediaDescriberAgent  # noqa: WPS433
            if self.describer_agent is None:
                self.describer_agent = MediaDescriberAgent()
            timeline_payload = await self.describer_agent.describe_media(description, urls)
        except Exception as e:
            media_agent_logger.error("Error generating media timeline description: %s", e, exc_info=True)
            yield StatusUpdateEvent(
                type=EventType.STATUS_UPDATE,
                status=StatusType.COMPLETE,
                agent_type=AgentType.MEDIA_TIMELINE,
                message="Error generating media timeline description",
                session_id=self.session_id,
                notebook_id=self.notebook_id,
                attempt=None,
                max_attempts=None,
                reason="Vision model error",
                step_id=None,
                original_plan_step_id=None,
            )
            return

        # Pass context to the correlator
        correlator_result = await self.correlator_agent.run_correlator(timeline_payload, context=context)
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

        