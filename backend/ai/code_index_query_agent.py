import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.logging import get_logger
from backend.ai.step_agents import StepAgent
from backend.ai.models import InvestigationStepModel, SafeOpenAIModel
from backend.ai.events import (
    AgentType,
    StatusType,
    EventType,
    ToolCallRequestedEvent,
    ToolSuccessEvent,
    ToolErrorEvent,
    StatusUpdateEvent,
    FatalErrorEvent,
    BaseEvent,
)
from backend.services.connection_handlers.registry import get_handler

# pydantic-ai imports
from pydantic_ai import Agent, CallToolsNode, UnexpectedModelBehavior
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage
from pydantic_ai.mcp import MCPServerStdio
from mcp.shared.exceptions import McpError

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# System prompt (inline for now – could be moved to a txt file in prompts/)
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    """
You are a helpful AI assistant specialised in searching large codebases that have
been indexed into Qdrant.  You have access to the `qdrant.qdrant-find` tool via
your MCP connection.  Given a natural-language description of what the user is
looking for, you should:

1. Convert the description into an appropriate semantic search query.
2. Call the `qdrant.qdrant-find` tool exactly once, providing the following
   arguments:
      • `query`:   the search text you generated.
      • `collection_name`: the Qdrant collection that was provided to you.
      • `limit`:   the maximum number of results you should return (default 5).
3. Do NOT attempt to post-process or re-call the tool.  Simply return the tool
   result to the caller.

Return format guidance:
• The raw list returned by `qdrant.qdrant-find` should be forwarded directly –
  do not wrap it in additional prose.
"""
)


class CodeIndexQueryAgent(StepAgent):
    """Step agent that leverages a pydantic-ai Agent + Qdrant MCP to perform a
    semantic search over an indexed codebase.
    """

    def __init__(
        self,
        notebook_id: str,
        connection_manager: Any,
        mcp_servers: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(notebook_id)
        self.connection_manager = connection_manager
        self.mcp_servers = mcp_servers or []

        # Initialize Qdrant server reference to None; will be fetched lazily.
        self.qdrant_mcp_server: Optional[MCPServerStdio] = None

        # pydantic-ai model & agent will be created lazily (once per instance)
        from backend.config import get_settings

        settings = get_settings()
        self._model = SafeOpenAIModel(
            "openai/gpt-4.1",
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
            ),
        )

        self._agent: Optional[Agent] = None  # lazy init in _ensure_agent

    # ---------------------------------------------------------------------
    # Helper - Get Qdrant MCP server using the git_repo connection type
    # ---------------------------------------------------------------------
    async def _get_qdrant_mcp_server(self) -> Optional[MCPServerStdio]:
        """Helper to get the Qdrant MCP server instance using the default 'git_repo' connection."""
        connection_type = "git_repo"
        try:
            default_conn = await self.connection_manager.get_default_connection(connection_type)
            if not default_conn:
                logger.error(f"No default {connection_type} connection found for Qdrant MCP.")
                return None
            if not default_conn.config:
                logger.error(f"Default {connection_type} connection {default_conn.id} has no config for Qdrant MCP.")
                return None

            handler = get_handler(connection_type)
            # Assuming the git_repo handler provides Qdrant server params
            stdio_params = handler.get_stdio_params(default_conn.config)
            logger.info(f"Retrieved StdioServerParameters for Qdrant MCP via {connection_type}. Command: {stdio_params.command}")
            return MCPServerStdio(command=stdio_params.command, args=stdio_params.args, env=stdio_params.env)
        except ValueError as ve:
            logger.error(f"Configuration error getting {connection_type} stdio params for Qdrant MCP: {ve}", exc_info=True)
        except Exception as e:
            logger.error(f"Error creating MCPServerStdio instance for Qdrant MCP via {connection_type}: {e}", exc_info=True)
        return None

    # ---------------------------------------------------------------------
    # Helper – create the pydantic-ai Agent if we don't have it yet.
    # ---------------------------------------------------------------------
    async def _ensure_agent(self) -> Optional[Agent]:
        if self._agent is not None:
            return self._agent

        # Fetch Qdrant MCP server if not already done
        if self.qdrant_mcp_server is None:
            self.qdrant_mcp_server = await self._get_qdrant_mcp_server()

        # If fetching failed, we can't proceed
        if self.qdrant_mcp_server is None:
            logger.error("Could not obtain Qdrant MCP server configuration.")
            return None

        self._agent = Agent(
            model=self._model,
            system_prompt=SYSTEM_PROMPT,
            mcp_servers=[self.qdrant_mcp_server], # Use the fetched server
        )
        logger.info("pydantic-ai Agent for CodeIndexQueryAgent initialized.")
        return self._agent

    # ------------------------------------------------------------------
    # StepAgent interface
    # ------------------------------------------------------------------
    def get_agent_type(self) -> AgentType:
        return AgentType.CODE_INDEX_QUERY

    async def execute(
        self,
        step: InvestigationStepModel,
        agent_input_data: Any,
        session_id: str,
        context: Dict[str, Any],
        db: Optional[AsyncSession] = None,
    ) -> AsyncGenerator[Union[BaseEvent, Dict[str, Any]], None]:
        """Run the code-index search and emit Sherlog events."""

        search_query = step.parameters.get("search_query", step.description)
        collection_name = step.parameters.get("collection_name")
        limit = step.parameters.get("limit", 5)

        if not collection_name:
            yield ToolErrorEvent(
                original_plan_step_id=step.step_id,
                tool_name="qdrant-find",
                tool_args={"search_query": search_query, "collection_name": None, "limit": limit},
                error="Missing collection_name parameter.",
                agent_type=self.get_agent_type(),
                session_id=session_id,
                notebook_id=self.notebook_id,
                status=StatusType.ERROR,
                tool_call_id=str(uuid4()),
                attempt=1,
                message="Missing collection_name parameter.",
            )
            return

        # Ensure we have a working pydantic-ai Agent (and therefore MCP server)
        agent = await self._ensure_agent() # Now async
        if agent is None:
            err_msg = "Qdrant MCP server is not available – cannot execute code-index search."
            yield ToolErrorEvent(
                original_plan_step_id=step.step_id,
                tool_name="qdrant-find",
                tool_args={"search_query": search_query, "collection_name": collection_name, "limit": limit},
                error=err_msg,
                agent_type=self.get_agent_type(),
                session_id=session_id,
                notebook_id=self.notebook_id,
                status=StatusType.FATAL_ERROR,
                tool_call_id=str(uuid4()),
                attempt=1,
                message=err_msg,
            )
            return

        # ------------------------------------------------------------------
        # Compose the user description for the agent
        # ------------------------------------------------------------------
        current_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        user_description = (
            f"Current time is {current_time_utc}. Search query: {search_query}. "
            f"Target collection: {collection_name}. Limit: {limit}."
        )

        # Yield starting status
        yield StatusUpdateEvent(
            type=EventType.STATUS_UPDATE,
            status=StatusType.STARTING,
            agent_type=self.get_agent_type(),
            message="Initialising code-index query agent…",
            attempt=None,
            max_attempts=None,
            reason=None,
            step_id=step.step_id,
            original_plan_step_id=step.step_id,
            session_id=session_id,
            notebook_id=self.notebook_id,
        )

        try:
            async with agent.run_mcp_servers():
                async with agent.iter(user_description) as agent_run:
                    pending_calls: Dict[str, Dict[str, Any]] = {}

                    async for node in agent_run:
                        if isinstance(node, CallToolsNode):
                            async with node.stream(agent_run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        call_id = event.part.tool_call_id
                                        tool_name = getattr(event.part, "tool_name", "qdrant-find")
                                        tool_args = getattr(event.part, "args", {})
                                        pending_calls[call_id] = {
                                            "tool_name": tool_name,
                                            "tool_args": tool_args,
                                        }
                                        yield ToolCallRequestedEvent(
                                            agent_type=self.get_agent_type(),
                                            tool_call_id=call_id,
                                            tool_name=tool_name,
                                            tool_args=tool_args,
                                            attempt=1,
                                            original_plan_step_id=step.step_id,
                                            session_id=session_id,
                                            notebook_id=self.notebook_id,
                                        )

                                    elif isinstance(event, FunctionToolResultEvent):
                                        call_id = event.tool_call_id
                                        call_info = pending_calls.pop(call_id, {})
                                        tool_name = call_info.get("tool_name", "qdrant-find")
                                        tool_args = call_info.get("tool_args", {})
                                        tool_result_content = event.result.content

                                        yield ToolSuccessEvent(
                                            original_plan_step_id=step.step_id,
                                            tool_name=tool_name,
                                            tool_args=tool_args,
                                            tool_result=tool_result_content,
                                            agent_type=self.get_agent_type(),
                                            session_id=session_id,
                                            notebook_id=self.notebook_id,
                                            tool_call_id=call_id,
                                            attempt=1,
                                        )

        except McpError as mcp_err:
            err_msg = f"MCP error: {mcp_err}"
            yield ToolErrorEvent(
                original_plan_step_id=step.step_id,
                tool_name="qdrant-find",
                tool_args={"query": search_query, "collection_name": collection_name, "limit": limit},
                error=err_msg,
                agent_type=self.get_agent_type(),
                session_id=session_id,
                notebook_id=self.notebook_id,
                status=StatusType.MCP_ERROR,
                tool_call_id=str(uuid4()),
                attempt=1,
                message=err_msg,
            )
        except UnexpectedModelBehavior as umb:
            err_msg = f"Model behaviour error: {umb}"
            yield ToolErrorEvent(
                original_plan_step_id=step.step_id,
                tool_name="qdrant-find",
                tool_args={"query": search_query, "collection_name": collection_name, "limit": limit},
                error=err_msg,
                agent_type=self.get_agent_type(),
                session_id=session_id,
                notebook_id=self.notebook_id,
                status=StatusType.MODEL_ERROR,
                tool_call_id=str(uuid4()),
                attempt=1,
                message=err_msg,
            )
        except Exception as exc:
            err_msg = f"Unexpected error: {exc}"
            logger.error(err_msg, exc_info=True)
            yield ToolErrorEvent(
                original_plan_step_id=step.step_id,
                tool_name="qdrant-find",
                tool_args={"query": search_query, "collection_name": collection_name, "limit": limit},
                error=err_msg,
                agent_type=self.get_agent_type(),
                session_id=session_id,
                notebook_id=self.notebook_id,
                status=StatusType.FATAL_ERROR,
                tool_call_id=str(uuid4()),
                attempt=1,
                message=err_msg,
            )
