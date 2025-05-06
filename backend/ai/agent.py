"""
AI Agent Module for Sherlog Canvas

This module implements the AI agent system using pydantic-ai for our notebook.
The agent is responsible for:
1. Interpreting user queries
2. Generating investigation plans
3. Creating cells with appropriate content
4. Analyzing data and providing insights

It uses Anthropic's Claude model for natural language understanding and code generation,
with tools that connect to various data sources like SQL, Prometheus, Loki, and S3.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from uuid import UUID, uuid4
from datetime import datetime, timezone

import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from sqlalchemy.orm import Session

from backend.core.cell import CellType, CellResult, CellStatus
from backend.ai.github_query_agent import GitHubQueryAgent
from backend.ai.summarization_agent import SummarizationAgent
from backend.ai.filesystem_agent import FileSystemAgent
from backend.config import get_settings
from backend.core.query_result import (
    QueryResult, 
    MarkdownQueryResult, 
    GithubQueryResult,
    SummarizationQueryResult,
    InvestigationReport,
    Finding
)
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams
from backend.core.notebook import Notebook
from backend.services.notebook_manager import NotebookManager
from backend.db.database import get_db
from backend.db.database import get_db_session
from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.ai.events import (
    EventType,
    AgentType,
    StatusType,
    PlanCreatedEvent,
    PlanCellCreatedEvent,
    StepStartedEvent,
    StepExecutionCompleteEvent,
    StepCompletedEvent,
    StepErrorEvent,
    GitHubToolCellCreatedEvent,
    GitHubToolErrorEvent,
    InvestigationCompleteEvent,
    FinalStatusEvent,
    ToolSuccessEvent,
    ToolErrorEvent,
    StatusUpdateEvent,
    BaseEvent,
    SummaryStartedEvent,
    SummaryUpdateEvent,
    SummaryCellCreatedEvent,
    SummaryCellErrorEvent,
    GitHubStepProcessingCompleteEvent,
    FileSystemStepProcessingCompleteEvent,
    FileSystemToolCellCreatedEvent,
    FileSystemToolErrorEvent,
    PythonToolCellCreatedEvent,
    PythonToolErrorEvent
)


import asyncio

from backend.ai.prompts.investigation_prompts import (
    INVESTIGATION_PLANNER_SYSTEM_PROMPT,
    PLAN_REVISER_SYSTEM_PROMPT,
    MARKDOWN_GENERATOR_SYSTEM_PROMPT
)

# Import the new agent and report type
from backend.ai.investigation_report_agent import InvestigationReportAgent

ai_logger = logging.getLogger("ai")

settings = get_settings()

class StepType(str, Enum):
    MARKDOWN = "markdown"
    GITHUB = "github"
    FILESYSTEM = "filesystem"
    PYTHON = "python"
    INVESTIGATION_REPORT = "investigation_report"


class StepCategory(str, Enum):
    PHASE = "phase"
    DECISION = "decision"


class InvestigationStepModel(BaseModel):
    """A step in the investigation plan"""
    step_id: str = Field(description="Unique identifier for this step")
    step_type: StepType = Field(description="Type of step (sql, python, markdown, etc.)")
    category: StepCategory = Field(
        description="PHASE for coarse step, DECISION for markdown decision cell",
        default=StepCategory.PHASE
    )
    description: str = Field(description="Description of what this step will do")
    tool_name: Optional[str] = Field(
        description="The specific tool name to be executed in this step, if applicable (e.g., 'get_file_contents')",
        default=None
    )
    dependencies: List[str] = Field(
        description="List of step IDs this step depends on",
        default_factory=list
    )
    parameters: Dict[str, Any] = Field(
        description="Parameters for the step",
        default_factory=dict
    )


class InvestigationPlanModel(BaseModel):
    """The complete investigation plan"""
    steps: List[InvestigationStepModel] = Field(
        description="Steps to execute in the investigation"
    )
    thinking: Optional[str] = Field(
        description="Reasoning behind the investigation plan",
        default=None
    )
    hypothesis: Optional[str] = Field(
        description="Current working hypothesis",
        default=None
    )

class InvestigationDependencies(BaseModel):
    """Dependencies for the investigation agent"""
    user_query: str = Field(description="The user's investigation query")
    notebook_id: UUID = Field(description="The ID of the notebook")
    available_data_sources: List[str] = Field(
        description="Available data sources for queries",
        default_factory=list
    )
    executed_steps: Dict[str, Any] = Field(
        description="Previously executed steps and their results",
        default_factory=dict
    )
    current_hypothesis: Optional[str] = Field(
        description="Current working hypothesis based on findings so far",
        default=None
    )
    message_history: List[ModelMessage] = Field(
        description="Message history for the investigation",
        default_factory=list
    )

class PlanRevisionRequest(BaseModel):
    """Request to revise an investigation plan"""
    original_plan: InvestigationPlanModel = Field(description="The original investigation plan")
    executed_steps: Dict[str, Any] = Field(description="Steps that have been executed so far")
    step_results: Dict[str, Any] = Field(description="Results from executed steps")
    unexpected_results: List[str] = Field(
        description="Step IDs with unexpected or interesting results",
        default_factory=list
    )
    current_hypothesis: Optional[str] = Field(
        description="Current working hypothesis based on findings so far",
        default=None
    )

class PlanRevisionResult(BaseModel):
    """Result of a plan revision"""
    continue_as_is: bool = Field(description="Whether to continue with the original plan")
    new_hypothesis: Optional[str] = Field(description="Updated hypothesis based on findings", default=None)
    new_steps: List[InvestigationStepModel] = Field(
        description="New steps to add to the plan",
        default_factory=list
    )
    steps_to_remove: List[str] = Field(
        description="Step IDs to remove from the plan",
        default_factory=list
    )
    explanation: str = Field(description="Explanation of the revision decision")


def format_message_history(history: List[ModelMessage]) -> str:
    """Formats message history into a simple string for the prompt."""
    formatted_lines = []
    for msg in history:
        role = "Unknown"
        content = ""
        if isinstance(msg, ModelRequest) and msg.parts:
            role = "User"
            content = getattr(msg.parts[0], 'content', '')
        elif isinstance(msg, ModelResponse) and msg.parts:
            role = "Assistant"
            # Handle different part types (StatusResponsePart, CellResponsePart etc.)
            if hasattr(msg.parts[0], 'content'):
                content = getattr(msg.parts[0], 'content', '')
            elif hasattr(msg.parts[0], 'cell_params'): # Example for CellResponsePart
                 content = f"[Assistant created cell: {getattr(msg.parts[0], 'cell_id', 'unknown')}]"
            else:
                 content = f"[Assistant action: {getattr(msg.parts[0], 'part_kind', 'unknown')}]"

        if content:
             # Basic cleaning/truncation might be needed here for long content
             cleaned_content = str(content).replace('\n', ' ').strip()
             formatted_lines.append(f"{role}: {cleaned_content}")
             
    return "\n".join(formatted_lines)

class AIAgent:
    """
    The AI agent that handles investigation planning and execution.
    Responsible for:
    1. Creating investigation plans
    2. Executing steps and streaming results
    3. Adjusting plans based on results
    4. Creating cells directly
    """
    def __init__(
        self,
        notebook_id: str,
        available_data_sources: List[str],
        notebook_manager: NotebookManager,
        mcp_server_map: Optional[Dict[str, MCPServerHTTP]] = None,
        connection_manager: Optional[ConnectionManager] = None
    ):
        self.settings = get_settings()
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.mcp_server_map = mcp_server_map or {}
        self.notebook_id = notebook_id
        self.available_data_sources = available_data_sources
        self.connection_manager = connection_manager or get_connection_manager()
        self.notebook_manager = notebook_manager
        
        ai_logger.info(f"AIAgent for notebook {self.notebook_id} initialized with available data sources: {self.available_data_sources}")
        
        available_data_sources_str = ', '.join(self.available_data_sources) if self.available_data_sources else 'None'
        formatted_planner_prompt = INVESTIGATION_PLANNER_SYSTEM_PROMPT.format(
            available_data_sources_str=available_data_sources_str
        )

        all_mcps_list = list(self.mcp_server_map.values())
        ai_logger.info(f"Initializing general agents with {len(all_mcps_list)} HTTP MCP servers.")
        
        self.investigation_planner = Agent(
            self.model,
            deps_type=InvestigationDependencies,
            output_type=InvestigationPlanModel,
            system_prompt=formatted_planner_prompt
        )

        self.plan_reviser: Optional[Agent[PlanRevisionRequest, PlanRevisionResult]] = None

        self.markdown_generator = Agent(
            self.model,
            output_type=MarkdownQueryResult,
            mcp_servers=all_mcps_list,
            system_prompt=MARKDOWN_GENERATOR_SYSTEM_PROMPT
        )

        # Initialize specialized agents without passing mcp_servers
        self.github_generator: Optional[GitHubQueryAgent] = None
        self.filesystem_generator: Optional[FileSystemAgent] = None
        self.investigation_report_generator: Optional[InvestigationReportAgent] = None

        ai_logger.info(f"AIAgent initialized for notebook {notebook_id}.")

    async def _handle_step_execution(
        self,
        current_step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, Any],
        cell_tools: NotebookCellTools,
        plan_step_id_to_cell_ids: Dict[str, List[UUID]], # Map plan step IDs to list of created cell IDs
        session_id: str,
        step_start_time: datetime
    ) -> AsyncGenerator[Union[
        StatusUpdateEvent, 
        GitHubToolCellCreatedEvent, GitHubToolErrorEvent, 
        FileSystemToolCellCreatedEvent, FileSystemToolErrorEvent,
        PythonToolCellCreatedEvent, PythonToolErrorEvent, 
        StepExecutionCompleteEvent
    ], None]:
        """Executes a single plan step by calling generate_content and handling its events.

        Yields status updates, github tool cell/error events, and a final 
        'step_execution_complete' event containing the outcome.

        Args:
            current_step: The plan step to execute.
            executed_steps: Dictionary of previously executed steps.
            step_results: Dictionary of results from previous steps.
            cell_tools: Tools for cell manipulation.
            plan_step_id_to_cell_ids: Map from plan step_id to list of created cell UUIDs
            session_id: The current chat session ID.
            step_start_time: Approximate start time of the step.

        Yields:
            Specific Event objects, culminating in a StepExecutionCompleteEvent.
        """
        final_result_data: Optional[QueryResult] = None
        step_error: Optional[str] = None
        github_step_has_tool_errors: bool = False
        filesystem_step_has_tool_errors: bool = False
        python_step_has_tool_errors: bool = False
        notebook_id_str = str(self.notebook_id)
        github_step_outputs: List[Any] = []
        filesystem_step_outputs: List[Any] = []
        python_step_outputs: List[Any] = []

        async for event_data in self.generate_content(
            current_step=current_step,
            step_type=current_step.step_type,
            description=current_step.description,
            executed_steps=executed_steps,
            step_results=step_results,
            cell_tools=cell_tools, 
            session_id=session_id
        ):
            # Now expect Pydantic models directly
            if isinstance(event_data, StatusUpdateEvent):
                 # Add step_id if missing and forward
                 if not event_data.step_id:
                     event_data.step_id = current_step.step_id
                 yield event_data

            # 1. Handle final results for NON-GitHub/Filesystem steps
            elif isinstance(event_data, dict) and event_data.get("type") == "final_step_result" and current_step.step_type not in [StepType.GITHUB, StepType.FILESYSTEM, StepType.PYTHON]:
                 final_result_data = event_data.get("result")
                 if not isinstance(final_result_data, QueryResult):
                      ai_logger.error(f"final_step_result dict did not contain QueryResult for step {current_step.step_id}")
                      final_result_data = None # Reset if type mismatch
                      step_error = step_error or "Internal error: Invalid final result format"
                 break # Stop processing for this step

            # 2. Handle individual GitHub tool success events
            elif isinstance(event_data, ToolSuccessEvent) and event_data.agent_type == AgentType.GITHUB:
                ai_logger.info(f"[_handle_step_execution] Received GitHub ToolSuccessEvent for plan step {current_step.step_id}: {event_data.tool_name}")
                 
                if not event_data.tool_call_id or not event_data.tool_name:
                    ai_logger.error(f"CRITICAL: GitHub ToolSuccessEvent missing tool_call_id or tool_name for step {current_step.step_id}. Event: {event_data!r}")
                    continue

                github_cell_content = f"GitHub: {event_data.tool_name}"
                 
                # Get dependencies based on the PLAN step's dependencies
                dependency_cell_ids: List[UUID] = []
                for dep_plan_id in current_step.dependencies:
                    dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_plan_id, []))

                # --- Fetch default GitHub connection ID (also for error cells) ---
                default_github_connection_id: Optional[str] = None # Renamed from _for_error
                try:
                    default_connection = await self.connection_manager.get_default_connection('github')
                    if default_connection:
                        default_github_connection_id = default_connection.id
                        ai_logger.info(f"Found default GitHub connection ID: {default_github_connection_id} for step {current_step.step_id}")
                    else:
                        ai_logger.warning(f"No default GitHub connection found. GitHub cell for step {current_step.step_id} will be created without a connection ID.")
                except Exception as conn_err:
                    ai_logger.error(f"Error fetching default GitHub connection for step {current_step.step_id}: {conn_err}", exc_info=True)
                # --- End Fetch default GitHub connection ID ---
                 
                # --- Generate UUID and update metadata ---
                cell_tool_call_id = uuid4() # Generate UUID for cell link
                github_cell_metadata = {
                    "session_id": session_id,
                    "original_plan_step_id": current_step.step_id,
                    "external_tool_call_id": event_data.tool_call_id # Store original event ID
                }
                # --- End UUID/Metadata ---
                 
                # --- Convert tool_result to JSON serializable format ---
                tool_result_content = event_data.tool_result
                serializable_result_content: Optional[Union[Dict, List, str, int, float, bool]] = None
                 
                # Try standard JSON types first
                if isinstance(tool_result_content, (dict, list, str, int, float, bool, type(None))):
                    serializable_result_content = tool_result_content
                # Try Pydantic model dump next
                elif isinstance(tool_result_content, BaseModel):
                      try:
                          serializable_result_content = tool_result_content.model_dump(mode='json')
                          ai_logger.info(f"Serialized Pydantic model result for {event_data.tool_name}")
                      except Exception as dump_err:
                          ai_logger.warning(f"Failed to dump Pydantic model result for {event_data.tool_name}, falling back to str: {dump_err}")
                          serializable_result_content = str(tool_result_content)
                # Fallback to string representation
                else:
                    try:
                        serializable_result_content = str(tool_result_content)
                        ai_logger.warning(f"Tool result for {event_data.tool_name} was not JSON serializable or Pydantic, converting to string: {type(tool_result_content)}")
                    except Exception as str_err:
                        # Very unlikely fallback
                        serializable_result_content = f"Error converting result to string: {str_err}"
                        ai_logger.warning(f"Failed even to convert tool result for {event_data.tool_name} to string: {str_err}", exc_info=True)
                # --- End Conversion ---
                 
                # --> Append successful result to outputs
                # Ensure output is captured even if subsequent steps fail slightly
                if serializable_result_content is not None:
                    github_step_outputs.append(serializable_result_content)
                else:
                    # Log if content was None unexpectedly
                    ai_logger.warning(f"Tool {event_data.tool_name} succeeded but produced None content after serialization.")
                  
                # Prepare parameters for creating the cell, including tool info
                create_cell_kwargs = {
                    "notebook_id": notebook_id_str,
                    "cell_type": CellType.GITHUB,
                    "content": github_cell_content,
                    "metadata": github_cell_metadata, # Use updated metadata
                    "dependencies": dependency_cell_ids, # Set actual cell dependencies
                    "connection_id": default_github_connection_id, # Use fetched ID
                    "tool_call_id": cell_tool_call_id, # Pass NEWLY generated UUID
                    "tool_name": event_data.tool_name,
                    "tool_arguments": event_data.tool_args or {},
                    "result": CellResult(content=serializable_result_content, error=None, execution_time=0.0) # Use serializable content
                }
                github_cell_params = CreateCellParams(**create_cell_kwargs)
                 
                # Directly create the cell for this tool call
                try:
                    cell_creation_result = await cell_tools.create_cell(params=github_cell_params)
                    individual_cell_id = cell_creation_result.get("cell_id")
                    if not individual_cell_id:
                         raise ValueError("create_cell tool did not return a cell_id for GitHub tool")
                    individual_cell_uuid = UUID(individual_cell_id)

                    # Update the map tracking created cells for this plan step
                    if current_step.step_id not in plan_step_id_to_cell_ids:
                        plan_step_id_to_cell_ids[current_step.step_id] = []
                    plan_step_id_to_cell_ids[current_step.step_id].append(individual_cell_uuid)
                     
                    # Yield event to notify frontend
                    yield GitHubToolCellCreatedEvent(
                        original_plan_step_id=current_step.step_id,
                        cell_id=individual_cell_id,
                        tool_name=event_data.tool_name,
                        tool_args=event_data.tool_args or {},
                        result=event_data.tool_result,
                        cell_params=github_cell_params.model_dump(),
                        session_id=session_id,
                        notebook_id=notebook_id_str
                    )

                except Exception as cell_creation_err:
                    error_msg = f"Failed creating cell for GitHub tool {event_data.tool_name} (plan step {current_step.step_id}): {cell_creation_err}"
                    # Log as warning, don't mark step as error, don't set step_error
                    ai_logger.warning(error_msg, exc_info=True) 
                    yield GitHubToolErrorEvent(
                        original_plan_step_id=current_step.step_id,
                        tool_call_id=event_data.tool_call_id, # Include ID if available
                        tool_name=event_data.tool_name,
                        tool_args=event_data.tool_args,
                        error=error_msg,
                        session_id=session_id,
                        notebook_id=notebook_id_str
                    )

            # 3. Handle individual Filesystem tool success events
            elif isinstance(event_data, ToolSuccessEvent) and event_data.agent_type == AgentType.FILESYSTEM:
                ai_logger.info(f"[_handle_step_execution] Received Filesystem ToolSuccessEvent for plan step {current_step.step_id}: {event_data.tool_name}")
                
                if not event_data.tool_call_id or not event_data.tool_name:
                    ai_logger.error(f"CRITICAL: Filesystem ToolSuccessEvent missing tool_call_id or tool_name for step {current_step.step_id}. Event: {event_data!r}")
                    continue

                fs_cell_content = f"Filesystem: {event_data.tool_name}"
                dependency_cell_ids: List[UUID] = []
                for dep_plan_id in current_step.dependencies:
                    dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_plan_id, []))
                
                # Fetch default Filesystem connection ID (if needed for cell metadata)
                default_fs_connection_id: Optional[str] = None
                try:
                    default_connection = await self.connection_manager.get_default_connection('filesystem')
                    if default_connection:
                        default_fs_connection_id = default_connection.id
                    else:
                        ai_logger.warning(f"No default Filesystem connection found for step {current_step.step_id}")
                except Exception as conn_err:
                    ai_logger.error(f"Error fetching default Filesystem connection for step {current_step.step_id}: {conn_err}", exc_info=True)

                cell_tool_call_id = uuid4()
                fs_cell_metadata = {
                    "session_id": session_id,
                    "original_plan_step_id": current_step.step_id,
                    "external_tool_call_id": event_data.tool_call_id
                }
                
                # Serialize result (similar to GitHub)
                tool_result_content = event_data.tool_result
                serializable_result_content: Optional[Union[Dict, List, str, int, float, bool]] = None
                if isinstance(tool_result_content, (dict, list, str, int, float, bool, type(None))):
                    serializable_result_content = tool_result_content
                elif isinstance(tool_result_content, BaseModel):
                    try: serializable_result_content = tool_result_content.model_dump(mode='json')
                    except Exception: serializable_result_content = str(tool_result_content)
                else:
                    try: serializable_result_content = str(tool_result_content)
                    except Exception: serializable_result_content = "Error converting result to string"

                if serializable_result_content is not None:
                    filesystem_step_outputs.append(serializable_result_content)
                else:
                    ai_logger.warning(f"Filesystem tool {event_data.tool_name} succeeded but produced None content after serialization.")
                
                create_cell_kwargs = {
                    "notebook_id": notebook_id_str,
                    "cell_type": CellType.FILESYSTEM, # Use the correct CellType
                    "content": fs_cell_content,
                    "metadata": fs_cell_metadata,
                    "dependencies": dependency_cell_ids,
                    "connection_id": default_fs_connection_id,
                    "tool_call_id": cell_tool_call_id,
                    "tool_name": event_data.tool_name,
                    "tool_arguments": event_data.tool_args or {},
                    "result": CellResult(content=serializable_result_content, error=None, execution_time=0.0)
                }
                fs_cell_params = CreateCellParams(**create_cell_kwargs)
                
                try:
                    cell_creation_result = await cell_tools.create_cell(params=fs_cell_params)
                    individual_cell_id = cell_creation_result.get("cell_id")
                    if not individual_cell_id: raise ValueError("create_cell tool did not return a cell_id for Filesystem tool")
                    individual_cell_uuid = UUID(individual_cell_id)

                    if current_step.step_id not in plan_step_id_to_cell_ids:
                        plan_step_id_to_cell_ids[current_step.step_id] = []
                    plan_step_id_to_cell_ids[current_step.step_id].append(individual_cell_uuid)
                    
                    # Yield the specific Filesystem event
                    yield FileSystemToolCellCreatedEvent(
                        original_plan_step_id=current_step.step_id,
                        cell_id=individual_cell_id,
                        tool_name=event_data.tool_name,
                        tool_args=event_data.tool_args,
                        result=event_data.tool_result,
                        cell_params=fs_cell_params.model_dump(),
                        session_id=session_id,
                        notebook_id=notebook_id_str
                    )
                except Exception as cell_creation_err:
                    error_msg = f"Failed creating cell for Filesystem tool {event_data.tool_name} (plan step {current_step.step_id}): {cell_creation_err}"
                    ai_logger.warning(error_msg, exc_info=True) 
                    # Yield specific Filesystem error event
                    yield FileSystemToolErrorEvent(
                        original_plan_step_id=current_step.step_id,
                        tool_call_id=event_data.tool_call_id,
                        tool_name=event_data.tool_name,
                        tool_args=event_data.tool_args,
                        error=error_msg,
                        session_id=session_id,
                        notebook_id=notebook_id_str
                    )

            # 4. Handle individual Python tool success events
            elif isinstance(event_data, ToolSuccessEvent) and event_data.agent_type == AgentType.PYTHON:
                ai_logger.info(f"[_handle_step_execution] Received Python ToolSuccessEvent for plan step {current_step.step_id}: {event_data.tool_name}")
                
                if not event_data.tool_call_id or not event_data.tool_name:
                    ai_logger.error(f"CRITICAL: Python ToolSuccessEvent missing tool_call_id or tool_name for step {current_step.step_id}. Event: {event_data!r}")
                    continue

                # Safely get python_code from tool_args, provide default content
                current_tool_args = event_data.tool_args or {}
                cell_content = current_tool_args.get('python_code', f"Python: {event_data.tool_name}") 
                dependency_cell_ids: List[UUID] = []
                for dep_plan_id in current_step.dependencies:
                    dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_plan_id, []))
                
                # Fetch default Python connection ID (if needed for cell metadata)
                default_python_connection_id: Optional[str] = None
                try:
                    default_connection = await self.connection_manager.get_default_connection('python')
                    if default_connection:
                        default_python_connection_id = default_connection.id
                    else:
                        ai_logger.warning(f"No default Python connection found for step {current_step.step_id}")
                except Exception as conn_err:
                    ai_logger.error(f"Error fetching default Python connection for step {current_step.step_id}: {conn_err}", exc_info=True)

                cell_tool_call_id = uuid4()
                python_cell_metadata = {
                    "session_id": session_id,
                    "original_plan_step_id": current_step.step_id,
                    "external_tool_call_id": event_data.tool_call_id
                }
                
                # Serialize result (similar to GitHub)
                tool_result_content = event_data.tool_result
                serializable_result_content: Optional[Union[Dict, List, str, int, float, bool]] = None
                if isinstance(tool_result_content, (dict, list, str, int, float, bool, type(None))):
                    serializable_result_content = tool_result_content
                elif isinstance(tool_result_content, BaseModel):
                    try: serializable_result_content = tool_result_content.model_dump(mode='json')
                    except Exception: serializable_result_content = str(tool_result_content)
                else:
                    try: serializable_result_content = str(tool_result_content)
                    except Exception: serializable_result_content = "Error converting result to string"

                if serializable_result_content is not None:
                    python_step_outputs.append(serializable_result_content)
                else:
                    ai_logger.warning(f"Tool {event_data.tool_name} succeeded but produced None content after serialization.")
                
                create_cell_kwargs = {
                    "notebook_id": notebook_id_str,
                    "cell_type": CellType.PYTHON, # Use the correct CellType
                    "content": cell_content,
                    "metadata": python_cell_metadata,
                    "dependencies": dependency_cell_ids,
                    "connection_id": default_python_connection_id,
                    "tool_call_id": cell_tool_call_id,
                    "tool_name": event_data.tool_name,
                    "tool_arguments": event_data.tool_args or {},
                    "result": CellResult(content=serializable_result_content, error=None, execution_time=0.0)
                }
                python_cell_params = CreateCellParams(**create_cell_kwargs)
                
                try:
                    cell_creation_result = await cell_tools.create_cell(params=python_cell_params)
                    individual_cell_id = cell_creation_result.get("cell_id")
                    if not individual_cell_id: raise ValueError("create_cell tool did not return a cell_id for Python tool")
                    individual_cell_uuid = UUID(individual_cell_id)

                    if current_step.step_id not in plan_step_id_to_cell_ids:
                        plan_step_id_to_cell_ids[current_step.step_id] = []
                    plan_step_id_to_cell_ids[current_step.step_id].append(individual_cell_uuid)
                    
                    # Yield the specific Python event
                    yield PythonToolCellCreatedEvent(
                        original_plan_step_id=current_step.step_id,
                        cell_id=individual_cell_id,
                        tool_name=event_data.tool_name,
                        tool_args=event_data.tool_args,
                        result=event_data.tool_result,
                        cell_params=python_cell_params.model_dump(),
                        session_id=session_id,
                        notebook_id=notebook_id_str
                    )
                except Exception as cell_creation_err:
                    error_msg = f"Failed creating cell for Python tool {event_data.tool_name} (plan step {current_step.step_id}): {cell_creation_err}"
                    ai_logger.warning(error_msg, exc_info=True) 
                    # Yield specific Python error event
                    yield PythonToolErrorEvent(
                        original_plan_step_id=current_step.step_id,
                        tool_call_id=event_data.tool_call_id,
                        tool_name=event_data.tool_name,
                        tool_args=event_data.tool_args,
                        error=error_msg,
                        session_id=session_id,
                        notebook_id=notebook_id_str
                    )

            # 5. Handle individual GitHub tool error events
            elif isinstance(event_data, ToolErrorEvent) and event_data.agent_type == AgentType.GITHUB:
                ai_logger.warning(f"[_handle_step_execution] Received GitHub ToolErrorEvent for plan step {current_step.step_id}: {event_data.tool_name} - {event_data.error}")
                # Yield specific GitHubToolErrorEvent
                yield GitHubToolErrorEvent(
                    original_plan_step_id=current_step.step_id,
                    tool_call_id=event_data.tool_call_id,
                    tool_name=event_data.tool_name,
                    tool_args=event_data.tool_args,
                    error=event_data.error,
                    session_id=session_id,
                    notebook_id=notebook_id_str
                )
                # Do not mark step as failed, but record the error
                # github_step_has_tool_errors = True # No longer marking step as failed for individual tool errors
                step_error = step_error or f"Error in tool {event_data.tool_name}: {event_data.error}" # Collect errors

            # 6. Handle individual Filesystem tool error events
            elif isinstance(event_data, ToolErrorEvent) and event_data.agent_type == AgentType.FILESYSTEM:
                ai_logger.warning(f"[_handle_step_execution] Received Filesystem ToolErrorEvent for plan step {current_step.step_id}: {event_data.tool_name} - {event_data.error}")
                # Yield specific FileSystemToolErrorEvent
                yield FileSystemToolErrorEvent(
                    original_plan_step_id=current_step.step_id,
                    tool_call_id=event_data.tool_call_id,
                    tool_name=event_data.tool_name,
                    tool_args=event_data.tool_args,
                    error=event_data.error,
                    session_id=session_id,
                    notebook_id=notebook_id_str
                )
                # Do not mark step as failed, but record the error
                filesystem_step_has_tool_errors = True # Track if any tool failed
                step_error = step_error or f"Error in tool {event_data.tool_name}: {event_data.error}" # Collect errors

            # 7. Handle Python tool error events
            elif isinstance(event_data, ToolErrorEvent) and event_data.agent_type == AgentType.PYTHON:
                ai_logger.warning(f"[_handle_step_execution] Received Python ToolErrorEvent for plan step {current_step.step_id}: {event_data.tool_name} - {event_data.error}")
                # Yield specific PythonToolErrorEvent
                yield PythonToolErrorEvent(
                    original_plan_step_id=current_step.step_id,
                    tool_call_id=event_data.tool_call_id,
                    tool_name=event_data.tool_name,
                    tool_args=event_data.tool_args,
                    error=event_data.error,
                    session_id=session_id,
                    notebook_id=notebook_id_str
                )
                # Collect error only if we recognized the agent type
                step_error = step_error or f"Error in Python tool {event_data.tool_name}: {event_data.error}" 

            # 8. Handle GitHub step completion marker
            elif isinstance(event_data, GitHubStepProcessingCompleteEvent):
                 ai_logger.info(f"[_handle_step_execution] Received GitHubStepProcessingCompleteEvent for plan step {current_step.step_id}")
                 break # Stop processing event loop for this step
                 
            # 9. Handle Filesystem step completion marker
            elif isinstance(event_data, FileSystemStepProcessingCompleteEvent):
                 ai_logger.info(f"[_handle_step_execution] Received FileSystemStepProcessingCompleteEvent for plan step {current_step.step_id}")
                 break # Stop processing event loop for this step
                 
            # 10. Handle other unexpected dict events (should be less common now)
            elif isinstance(event_data, dict):
                 ai_logger.warning(f"[_handle_step_execution] Unexpected dict event type '{event_data.get('type')}' while processing step {current_step.step_id}. Data: {event_data}")
            else:
                 ai_logger.error(f"[_handle_step_execution] FATAL: generate_content yielded non-event/dict type {type(event_data)} for step {current_step.step_id}")
                 step_error = step_error or "Internal error: Invalid generator output"
                 break 
        
        # Determine final outputs
        final_step_outputs = None
        if current_step.step_type in [StepType.GITHUB, StepType.FILESYSTEM, StepType.PYTHON]:
            if current_step.step_type == StepType.GITHUB: final_step_outputs = github_step_outputs
            elif current_step.step_type == StepType.FILESYSTEM: final_step_outputs = filesystem_step_outputs
            elif current_step.step_type == StepType.PYTHON: final_step_outputs = python_step_outputs
            
        # Yield final completion event - **Ensure all args are passed**
        yield StepExecutionCompleteEvent(
            step_id=current_step.step_id,
            final_result_data=final_result_data, 
            step_outputs=final_step_outputs,
            step_error=step_error,
            github_step_has_tool_errors=github_step_has_tool_errors, 
            filesystem_step_has_tool_errors=filesystem_step_has_tool_errors, 
            python_step_has_tool_errors=python_step_has_tool_errors, 
            session_id=session_id,
            notebook_id=notebook_id_str
        )

    async def investigate(
        self, 
        query: str, 
        session_id: str, 
        notebook_id: Optional[str] = None,
        message_history: List[ModelMessage] = [],
        cell_tools: Optional[NotebookCellTools] = None
    ) -> AsyncGenerator[Union[
        PlanCreatedEvent, PlanCellCreatedEvent, StepStartedEvent, 
        StepCompletedEvent, StepErrorEvent, 
        GitHubToolCellCreatedEvent, GitHubToolErrorEvent, 
        FileSystemToolCellCreatedEvent, FileSystemToolErrorEvent, 
        PythonToolCellCreatedEvent, PythonToolErrorEvent, 
        StatusUpdateEvent, 
        SummaryStartedEvent, SummaryUpdateEvent, SummaryCellCreatedEvent, SummaryCellErrorEvent,
        InvestigationCompleteEvent,
        # StepExecutionCompleteEvent is internal, but ToolErrorEvent covers generic errors
        ToolErrorEvent # Add generic ToolErrorEvent for generate_content failures
    ], None]:
        """
        Investigate a query by creating and executing a plan.
        Streams status updates as steps are completed.
        
        Args:
            query: The user's query
            session_id: The chat session ID
            notebook_id: The notebook ID to create cells in
            message_history: Previous messages in the session
            cell_tools: Tools for creating and managing cells
            
        Yields:
            Event model instances representing status updates
        """
        async with get_db_session() as db:
            if not cell_tools:
                raise ValueError("cell_tools is required for creating cells")
                
            notebook_id_str = notebook_id or self.notebook_id
            if not notebook_id_str:
                raise ValueError("notebook_id is required for creating cells")
            
            # Fetch the actual Notebook object using the stored manager
            try:
                notebook_uuid = UUID(notebook_id_str)
                notebook: Notebook = await self.notebook_manager.get_notebook(db=db, notebook_id=notebook_uuid)
            except ValueError:
                ai_logger.error(f"Invalid notebook_id format: {notebook_id_str}")
                raise ValueError(f"Invalid notebook_id format: {notebook_id_str}")
            except Exception as e:
                 ai_logger.error(f"Failed to fetch notebook {notebook_id_str}: {e}", exc_info=True)
                 raise ValueError(f"Failed to fetch notebook {notebook_id_str}")

            plan = await self.create_investigation_plan(query, notebook_id_str, message_history)
            # Pass session_id and notebook_id to event
            yield PlanCreatedEvent(thinking=plan.thinking, session_id=session_id, notebook_id=notebook_id_str)
            
            plan_step_id_to_cell_ids: Dict[str, List[UUID]] = {}
            executed_steps: Dict[str, Any] = {}
            step_results: Dict[str, Any] = {}

            # --- Plan Explanation Cell (only if multiple steps) ---
            if len(plan.steps) > 1:
                plan_cell_params = CreateCellParams(
                    notebook_id=notebook_id_str,
                    cell_type="markdown",
                    content=f"# Investigation Plan\\n\\n## Steps:\\n" +
                        "\\n".join(f"- `{step.step_id}`: {step.step_type.value}" for step in plan.steps),
                    metadata={
                        "session_id": session_id,
                        "step_id": "plan",
                        "dependencies": []
                    },
                    tool_call_id=uuid4()
                )
                try:
                    plan_cell_result = await cell_tools.create_cell(params=plan_cell_params)
                    # Pass session_id and notebook_id to event
                    yield PlanCellCreatedEvent(
                        cell_params=plan_cell_params.model_dump(mode='json'),
                        cell_id=plan_cell_result.get("cell_id", ""),
                        session_id=session_id,
                        notebook_id=notebook_id_str
                    )
                except Exception as plan_cell_err:
                     ai_logger.error(f"Failed to create plan explanation cell: {plan_cell_err}", exc_info=True)
                     # Potentially yield an error event here if needed for frontend awareness
                     # Example: yield BaseEvent(type=EventType.ERROR, data={"message": "Failed to create plan cell", "details": str(plan_cell_err)}, session_id=session_id, notebook_id=notebook_id_str)

            else:
                ai_logger.info("Skipping plan explanation cell creation for single-step plan.")

            remaining_steps = plan.steps.copy()

            while remaining_steps:
                executable_steps = [
                    step for step in remaining_steps
                    if all(dep in executed_steps for dep in step.dependencies)
                ]

                if not executable_steps:
                     # Check if remaining steps have unresolved dependencies
                     unresolved_deps = False
                     for step in remaining_steps:
                         if not all(dep in executed_steps for dep in step.dependencies):
                             unresolved_deps = True
                             ai_logger.warning(f"Step {step.step_id} blocked, missing dependencies: {[dep for dep in step.dependencies if dep not in executed_steps]}")
                             break # Found an unresolved dep, no need to check further

                     if not unresolved_deps and remaining_steps:
                         ai_logger.error("No executable steps but plan not complete and no unresolved dependencies found.")
                         # Pass session_id and notebook_id to event
                         yield StepErrorEvent(step_id="PLAN_STALLED", error="Plan execution stalled unexpectedly.", agent_type=AgentType.INVESTIGATION_PLANNER, step_type=None, cell_id=None, cell_params=None, result=None, session_id=session_id, notebook_id=notebook_id_str)
                     elif not remaining_steps:
                         ai_logger.info("Plan execution complete.")
                     else:
                        # Normal case: waiting for dependencies
                        ai_logger.info(f"Plan execution waiting for dependencies. Remaining steps: {[s.step_id for s in remaining_steps]}")
                        # Optional: Yield a status indicating waiting state (using StatusUpdateEvent)
                        # yield StatusUpdateEvent(status=StatusType.WAITING_FOR_DEPENDENCIES, message=f"Waiting for dependencies for steps: {[s.step_id for s in remaining_steps]}", session_id=session_id, notebook_id=notebook_id_str)
                     break # Exit the while loop if no steps are immediately executable

                current_step = executable_steps[0]
                remaining_steps.remove(current_step)

                # Determine AgentType based on StepType
                agent_type_enum: AgentType
                step_type_val = current_step.step_type
                if not step_type_val:
                     ai_logger.error(f"Step {current_step.step_id} has missing step_type. Skipping.")
                     continue # Skip this invalid step
                     
                if step_type_val == StepType.GITHUB: agent_type_enum = AgentType.GITHUB
                elif step_type_val == StepType.FILESYSTEM: agent_type_enum = AgentType.FILESYSTEM
                elif step_type_val == StepType.PYTHON: agent_type_enum = AgentType.PYTHON
                elif step_type_val == StepType.MARKDOWN: agent_type_enum = AgentType.MARKDOWN
                else: agent_type_enum = AgentType.UNKNOWN

                # Add specific agent types for new steps
                if step_type_val == StepType.INVESTIGATION_REPORT: agent_type_enum = AgentType.INVESTIGATION_REPORT_GENERATOR

                yield StepStartedEvent(step_id=current_step.step_id, agent_type=agent_type_enum, session_id=session_id, notebook_id=notebook_id_str)
                step_start_time = datetime.now(timezone.utc)

                # --- Special Handling for Investigation Report Step --- 
                if current_step.step_type == StepType.INVESTIGATION_REPORT:
                    ai_logger.info(f"Executing dedicated Investigation Report step: {current_step.step_id}")
                    # Pass session_id and notebook_id to event
                    yield SummaryStartedEvent(session_id=session_id, notebook_id=notebook_id_str)

                    # Format context for the report generator (copied from original end block)
                    findings_text_lines = []
                    # Iterate over ALL previous steps in the original plan for findings
                    for step in plan.steps:
                        if step.step_id != current_step.step_id and step.step_id in executed_steps: # Ensure we don't include the report step itself
                            step_info = executed_steps[step.step_id]
                            step_obj = step_info.get("step")
                            step_result_data = step_results.get(step.step_id)
                            step_error_msg = step_info.get("error")
                            
                            desc = step_obj.get("description", "") if step_obj else ""
                            type_val = step_obj.get("step_type", "unknown") if step_obj else "unknown"
                            
                            result_str = ""
                            if step_error_msg:
                                result_str = f"Error: {step_error_msg}"
                            elif step_result_data:
                                if isinstance(step_result_data, QueryResult):
                                    if step_result_data.error:
                                        result_str = f"Error: {step_result_data.error}"
                                    elif step_result_data.data:
                                        result_str = f"Data: {str(step_result_data.data)[:500]}..."
                                    else:
                                        result_str = "Step completed, but no data returned."
                                elif isinstance(step_result_data, list):
                                    formatted_outputs = []
                                    for i, output in enumerate(step_result_data):
                                        formatted_outputs.append(f"  - Output {i+1}: {str(output)[:200]}...")
                                    if formatted_outputs:
                                        result_str = "Data (multiple outputs):\\n" + "\\n".join(formatted_outputs)
                                    else:
                                        result_str = "Step completed, but no specific outputs captured."
                                else:
                                    result_str = f"Data: {str(step_result_data)[:500]}..."
                            else:
                                result_str = "Step completed without explicit data/error capture."
                            
                            findings_text_lines.append(f"Step {step.step_id} ({type_val}): {desc}\\nResult:\\n{result_str}")

                    findings_text = "\\n---\\n".join(findings_text_lines)
                    if not findings_text:
                        findings_text = "No preceding investigation steps were successfully executed or produced results."
                        
                    # Input for the new agent
                    report_generation_input = findings_text
                    ai_logger.info(f"Findings text prepared for report generation (length: {len(report_generation_input)}):\\n{report_generation_input[:1000]}...") # Log input summary

                    report_error = None
                    final_report_object: Optional[InvestigationReport] = None
                    try:
                        # Lazy initialization of the new agent
                        if self.investigation_report_generator is None:
                            ai_logger.info("Lazily initializing InvestigationReportAgent.")
                            self.investigation_report_generator = InvestigationReportAgent(notebook_id=self.notebook_id)
                            
                        async for result_part in self.investigation_report_generator.run_report_generation(
                            original_query=query, 
                            findings_summary=report_generation_input,
                            session_id=session_id,
                            notebook_id=notebook_id_str
                        ):
                            if isinstance(result_part, InvestigationReport):
                                final_report_object = result_part
                                report_error = final_report_object.error # Check if the report itself contains an error
                                # Pass session_id and notebook_id to event
                                yield SummaryUpdateEvent(update_info={"message": "Report object generated"}, session_id=session_id, notebook_id=notebook_id_str) # Use new event
                                ai_logger.info("Received InvestigationReport object from generator.") # Log object received
                                break # Got the final report object
                            elif isinstance(result_part, StatusUpdateEvent):
                                # Forward status updates, ensuring session/notebook IDs are present
                                if result_part.session_id is None:
                                     result_part.session_id = session_id
                                if result_part.notebook_id is None:
                                     result_part.notebook_id = notebook_id_str
                                yield result_part # Yield the updated event
                            else:
                                ai_logger.warning(f"Unexpected item yielded from investigation report generator: {type(result_part)}")
                        
                        if final_report_object is None:
                            report_error = report_error or "Investigation report agent did not return a final report object."
                            ai_logger.error(report_error)
                            # Create a placeholder error report if none was yielded
                            final_report_object = InvestigationReport(
                                query=query, 
                                title="Report Generation Failed", 
                                issue_summary=Finding(summary="Generation Error", details=None, code_reference=None, supporting_quotes=None), 
                                root_cause=Finding(summary="Generation Error", details=None, code_reference=None, supporting_quotes=None), 
                                error=report_error,
                                status=None, status_reason="Generation Error", estimated_severity=None, root_cause_confidence=None,
                                proposed_fix=None, proposed_fix_confidence=None, affected_context=None,
                                key_components=[], related_items=[], suggested_next_steps=[], tags=[]
                            )

                    except Exception as e:
                        report_error = f"Error during final report generation step '{current_step.step_id}': {e}"
                        ai_logger.error(report_error, exc_info=True)
                        # Create a placeholder error report
                        final_report_object = InvestigationReport(
                            query=query, title="Report Generation Failed", 
                            issue_summary=Finding(summary="Exception during generation", details=None, code_reference=None, supporting_quotes=None), 
                            root_cause=Finding(summary="Exception during generation", details=None, code_reference=None, supporting_quotes=None), 
                            error=report_error,
                            status=None, status_reason="Exception during generation", estimated_severity=None, root_cause_confidence=None,
                            proposed_fix=None, proposed_fix_confidence=None, affected_context=None,
                            key_components=[], related_items=[], suggested_next_steps=[], tags=[]
                        )

                    # --- Create Final Report Cell (moved inside handler) --- 
                    report_cell_content = f"# Investigation Report: {final_report_object.title}\\n\\n_Structured report data generated._"
                    if report_error and not final_report_object.error:
                         final_report_object.error = report_error
                    report_data_dict = final_report_object.model_dump(mode='json')
                    
                    report_dependency_cell_ids: List[UUID] = []
                    for dep_id in current_step.dependencies:
                        report_dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_id, []))

                    report_cell_metadata = {"session_id": session_id, "step_id": current_step.step_id, "is_investigation_report": True}
                    report_cell_params = CreateCellParams(
                        notebook_id=notebook_id_str, cell_type=CellType.INVESTIGATION_REPORT.value,
                        content=report_cell_content, metadata=report_cell_metadata, dependencies=report_dependency_cell_ids,
                        tool_call_id=uuid4(),
                        result=CellResult(content=report_data_dict, error=report_error, execution_time=0.0),
                        status=CellStatus.ERROR if report_error else CellStatus.SUCCESS
                    )
                    
                    created_cell_id_report: Optional[UUID] = None
                    try:
                        report_cell_result = await cell_tools.create_cell(params=report_cell_params)
                        report_cell_id_str = report_cell_result.get("cell_id")
                        if report_cell_id_str: created_cell_id_report = UUID(report_cell_id_str)
                        # Pass session_id and notebook_id to event
                        yield SummaryCellCreatedEvent( # Use summary event for consistency 
                            cell_params=report_cell_params.model_dump(), cell_id=report_cell_id_str or None,
                            error=report_error, session_id=session_id, notebook_id=notebook_id_str 
                        )
                        ai_logger.info(f"Investigation report cell created ({report_cell_id_str}) for step {current_step.step_id}")
                        if created_cell_id_report and current_step.step_id not in plan_step_id_to_cell_ids:
                             plan_step_id_to_cell_ids[current_step.step_id] = []
                        if created_cell_id_report: plan_step_id_to_cell_ids[current_step.step_id].append(created_cell_id_report)
                    except Exception as cell_err:
                        ai_logger.error(f"Failed to create investigation report cell for step {current_step.step_id}: {cell_err}", exc_info=True)
                        # Pass session_id and notebook_id to event
                        yield SummaryCellErrorEvent(
                            error=str(cell_err), cell_params=report_cell_params.model_dump(),
                            session_id=session_id, notebook_id=notebook_id_str
                        )

                    # Update executed_steps/step_results for the report step itself
                    executed_steps[current_step.step_id] = {
                        "step": current_step.model_dump(),
                        "content": report_cell_content, # Store the markdown placeholder content
                        "error": report_error
                    }
                    step_results[current_step.step_id] = final_report_object # Store the actual report object

                    # Yield StepCompleted/StepError for the *report* step
                    if report_error:
                        yield StepErrorEvent(
                            step_id=current_step.step_id, step_type=StepType.INVESTIGATION_REPORT.value,
                            cell_id=str(created_cell_id_report) if created_cell_id_report else None,
                            cell_params=report_cell_params.model_dump() if report_cell_params else None, 
                            result=final_report_object.model_dump(mode='json') if final_report_object else None,
                            error=report_error, agent_type=agent_type_enum, 
                            session_id=session_id, notebook_id=notebook_id_str
                        )
                    else:
                        yield StepCompletedEvent(
                            step_id=current_step.step_id, step_type=StepType.INVESTIGATION_REPORT.value,
                            cell_id=str(created_cell_id_report) if created_cell_id_report else None,
                            cell_params=report_cell_params.model_dump() if report_cell_params else None, 
                            result=final_report_object.model_dump(mode='json') if final_report_object else None,
                            agent_type=agent_type_enum, 
                            session_id=session_id, notebook_id=notebook_id_str
                        )

                    # Skip the generic _handle_step_execution as we handled it here
                    continue # Move to the next step in the while loop

                async for execution_event in self._handle_step_execution(
                     current_step=current_step,
                     executed_steps=executed_steps,
                     step_results=step_results,
                     cell_tools=cell_tools, 
                     plan_step_id_to_cell_ids=plan_step_id_to_cell_ids, 
                     session_id=session_id,
                     step_start_time=step_start_time
                 ):
                    if isinstance(execution_event, (
                        StatusUpdateEvent, GitHubToolCellCreatedEvent, GitHubToolErrorEvent,
                        FileSystemToolCellCreatedEvent, FileSystemToolErrorEvent, PythonToolCellCreatedEvent, PythonToolErrorEvent
                    )):
                        yield execution_event
                    
                    elif isinstance(execution_event, StepExecutionCompleteEvent):
                        final_result_data = execution_event.final_result_data
                        step_error = execution_event.step_error
                        step_outputs = execution_event.step_outputs 
                        
                        created_cell_id = None
                        cell_params_for_step = None 
                        plan_step_has_error = False 
                        
                        current_step_type_val = current_step.step_type
                        if not current_step_type_val:
                            ai_logger.error(f"Step {current_step.step_id} completion processing skipped due to missing step_type.")
                            plan_step_has_error = True
                            step_error = step_error or "Missing step type"
                            # Update executed_steps/step_results for error even if type is missing
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": None,
                                "error": step_error
                            }
                            step_results[current_step.step_id] = []
                        
                        elif current_step_type_val == StepType.MARKDOWN:
                            # ... (Markdown cell creation/error handling logic)
                            pass
                        elif current_step_type_val in [StepType.GITHUB, StepType.FILESYSTEM, StepType.PYTHON]:
                            plan_step_has_error = bool(step_error)
                            step_completion_content = f"Completed {current_step_type_val.value} step (see individual tool cells)"
                            if plan_step_has_error:
                                step_completion_content += f" - Last Error: {step_error}"
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": step_completion_content,
                                "error": step_error
                            }
                            step_results[current_step.step_id] = step_outputs if step_outputs else []
                        else:
                             step_type_log_val_err = current_step_type_val.value if current_step_type_val else "Unknown"
                             ai_logger.error(f"Investigate loop encountered unknown step type: {step_type_log_val_err}")
                             plan_step_has_error = True
                             step_error = step_error or f"Unknown step type {step_type_log_val_err}"
                             executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": None,
                                "error": step_error
                             }
                             step_results[current_step.step_id] = []

                        # --- Yield StepCompleted/StepError ONLY for MARKDOWN steps --- 
                        if current_step_type_val == StepType.MARKDOWN:
                            step_type_md_val = current_step_type_val.value # Safe access guaranteed by outer check
                            if plan_step_has_error:
                                yield StepErrorEvent( 
                                    step_id=current_step.step_id,
                                    step_type=step_type_md_val,
                                    cell_id=str(created_cell_id) if created_cell_id else None, 
                                    cell_params=cell_params_for_step.model_dump() if cell_params_for_step else None, 
                                    result=getattr(final_result_data, 'model_dump', lambda: None)() if final_result_data else None,
                                    error=step_error or getattr(final_result_data, 'error', None),
                                    agent_type=agent_type_enum, 
                                    session_id=session_id,
                                    notebook_id=notebook_id_str
                                )
                            else:
                                yield StepCompletedEvent( 
                                    step_id=current_step.step_id,
                                    step_type=step_type_md_val,
                                    cell_id=str(created_cell_id) if created_cell_id else None,
                                    cell_params=cell_params_for_step.model_dump() if cell_params_for_step else None,
                                    result=getattr(final_result_data, 'model_dump', lambda: None)() if final_result_data else None,
                                    agent_type=agent_type_enum, 
                                    session_id=session_id,
                                    notebook_id=notebook_id_str
                                )
                            # Update executed_steps/step_results for Markdown
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": getattr(final_result_data, 'data', None),
                                "error": step_error or getattr(final_result_data, 'error', None)
                            }
                            step_results[current_step.step_id] = final_result_data
                        
                        final_plan_step_status = "error" if plan_step_has_error else "success"
                        step_type_log_val = current_step_type_val.value if current_step_type_val else "Unknown"
                        ai_logger.info(f"Processed plan step: {current_step.step_id} ({step_type_log_val}) with final status: {final_plan_step_status}")
                        
                        break 
                    else:
                        ai_logger.warning(f"Investigate loop received unexpected event type from _handle_step_execution: {type(execution_event)}")


            ai_logger.info(f"Investigation plan execution finished for notebook {notebook_id_str}")

            # Pass session_id and notebook_id to event
            yield InvestigationCompleteEvent(session_id=session_id, notebook_id=notebook_id_str) 

            if 'notebook' in locals() and notebook is not None:
                try:
                    await self.notebook_manager.save_notebook(db=db, notebook_id=notebook.id, notebook=notebook)
                    ai_logger.info(f"Successfully initiated save for notebook {notebook.id} state after investigation.")
                except Exception as save_err:
                    ai_logger.error(f"Failed to initiate save for notebook {notebook.id} state after investigation: {save_err}", exc_info=True)
        

    async def create_investigation_plan(self, query: str, notebook_id: Optional[str] = None, message_history: List[ModelMessage] = []) -> InvestigationPlanModel:
        """Create an investigation plan for the given query, incorporating history into the prompt."""
        notebook_id = notebook_id or self.notebook_id
        if not notebook_id:
            raise ValueError("notebook_id is required for creating investigation plan")

        # Format the history
        formatted_history = format_message_history(message_history)

        # Combine history and the latest query into a single prompt
        combined_prompt = f"Conversation History:\n---\n{formatted_history}\n---\n\nLatest User Query: {query}"
        ai_logger.info(f"Planner combined prompt:\n{combined_prompt}")

        # Create dependencies, but exclude message_history as it's now in the prompt
        deps = InvestigationDependencies(
            user_query=query, # Keep the original query here for potential separate use if needed
            notebook_id=UUID(notebook_id),
            available_data_sources=self.available_data_sources,
            executed_steps={}, # Assuming fresh plan creation
            current_hypothesis=None,
            message_history=[] # Pass empty list or None if model allows
        )

        # Generate the plan using the combined prompt
        result = await self.investigation_planner.run(
            user_prompt=combined_prompt, # Use the combined history + query
            deps=deps # Pass deps object (without history)
        )
        # Use .output instead of .data
        plan_data = result.output

        # Convert to InvestigationPlan
        steps = []
        for step_data in plan_data.steps:
            step = InvestigationStepModel(
                step_id=step_data.step_id,
                step_type=step_data.step_type,
                description=step_data.description,
                dependencies=step_data.dependencies,
                parameters=step_data.parameters,
                category=step_data.category
            )
            steps.append(step)

        return InvestigationPlanModel(
            steps=steps,
            thinking=plan_data.thinking,
            hypothesis=plan_data.hypothesis
        )

    async def revise_plan(
        self,
        original_plan: InvestigationPlanModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, Any],
        current_hypothesis: Optional[str] = None,
        unexpected_results: List[str] = []
    ) -> PlanRevisionResult:
        """
        Revise the investigation plan based on executed steps
        
        Args:
            original_plan: The original investigation plan
            executed_steps: Steps that have been executed so far
            step_results: Results from executed steps
            current_hypothesis: Current working hypothesis based on findings
            unexpected_results: Step IDs with unexpected or interesting results
            
        Returns:
            Revised plan with potential new steps or steps to remove
        """
        # Lazy initialization
        if self.plan_reviser is None:
            ai_logger.info("Lazily initializing PlanReviser Agent.")
            # Ensure self.model is available (it's initialized in AIAgent.__init__)
            if self.model is None:
                 raise RuntimeError("AIAgent model is not initialized, cannot create PlanReviser.")
            self.plan_reviser = Agent(
                self.model,
                deps_type=PlanRevisionRequest,
                output_type=PlanRevisionResult,
                system_prompt=PLAN_REVISER_SYSTEM_PROMPT
            )
            
        revision_request = PlanRevisionRequest(
            original_plan=original_plan,
            executed_steps=executed_steps,
            step_results=step_results,
            unexpected_results=unexpected_results,
            current_hypothesis=current_hypothesis
        )
        
        result = await self.plan_reviser.run(
            user_prompt="Revise the investigation plan based on executed steps and their results.",
            deps=revision_request
        )
        return result.output

    async def _yield_result_async(self, result):
        """Helper to wrap a non-async-generator result."""
        yield result
        await asyncio.sleep(0)

    async def generate_content(
        self,
        current_step: InvestigationStepModel,
        step_type: StepType,
        description: str,
        executed_steps: Dict[str, Any] | None = None,
        step_results: Dict[str, Any] = {},
        cell_tools: Optional[NotebookCellTools] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Union[
        Dict[str, Any], StatusUpdateEvent, ToolSuccessEvent, ToolErrorEvent, StepErrorEvent,
        GitHubStepProcessingCompleteEvent, FileSystemStepProcessingCompleteEvent, PythonToolCellCreatedEvent, PythonToolErrorEvent
    ], None]:
        """
        Generate content for a specific step type.

        Args:
            current_step: The current step in the investigation plan
            step_type: Type of step (sql, python, markdown, etc.)
            description: Description of what the step will do
            executed_steps: Previously executed steps (for context)
            step_results: Results from previously executed steps (for context)
            cell_tools: Tools for creating and managing cells
            session_id: The chat session ID

        Yields:
            Event models (StatusUpdateEvent, ToolSuccessEvent, ToolErrorEvent) or 
            a dictionary containing the final QueryResult for non-GitHub steps.
        """

        context = ""
        if executed_steps:
            dependency_context = []
            for dep_id in current_step.dependencies:
                if dep_id in executed_steps:
                    dep_step_info = executed_steps[dep_id]
                    dep_result_data = step_results.get(dep_id) # Get results for this dependency

                    dep_desc = dep_step_info.get('step', {}).get('description', 'No description')
                    dep_error = dep_step_info.get('error')
                    status = "Error" if dep_error else "Success"
                    
                    context_line = f"- Dependency '{dep_id}' ({status}): {dep_desc[:100]}..."
                    
                    if dep_error:
                        context_line += f" (Error: {str(dep_error)[:150]}...)"
                    elif dep_result_data:
                         # Add a snippet of the result data if available and not an error
                         result_snippet = ""
                         if isinstance(dep_result_data, QueryResult) and dep_result_data.data:
                             result_snippet = str(dep_result_data.data)[:150]
                         elif isinstance(dep_result_data, list) and dep_result_data:
                             # Show first item snippet if it's a list of results (like from GitHub/FS tools)
                             result_snippet = str(dep_result_data[0])[:150]
                         elif dep_result_data: # Handle other non-QueryResult, non-list types
                             result_snippet = str(dep_result_data)[:150]
                         
                         if result_snippet:
                             context_line += f" (Result Snippet: {result_snippet}...)"
                         else:
                              context_line += " (Completed with no specific data output)"
                    else:
                        context_line += " (Completed without explicit results)"

                    dependency_context.append(context_line)
            if dependency_context:
                 context = f"\\n\\nContext from Dependencies:\\n" + "\\n".join(dependency_context)

        agent_prompt = description

        if current_step.parameters:
            params_str = "\n".join([f"- {k}: {v}" for k, v in current_step.parameters.items()])
            agent_prompt += f"\n\nParameters for this step:\n{params_str}"

        agent_prompt += context

        full_description = agent_prompt
        step_type_val = step_type.value if step_type else "Unknown"
        ai_logger.info(f"[generate_content] Passing to sub-agent ({step_type_val}): {full_description[:500]}...")

        final_result: Optional[QueryResult] = None

        if step_type == StepType.GITHUB:
            if not cell_tools or not session_id:
                error_msg = "Internal Error: Missing required parameters (cell_tools, session_id) for GitHub step processing."
                ai_logger.error(error_msg)
                # Pass session_id and notebook_id if available
                yield ToolErrorEvent(error=error_msg, agent_type=AgentType.GITHUB, status=StatusType.FATAL_ERROR, attempt=None, tool_call_id=None, tool_name=None, tool_args=None, message=None, original_plan_step_id=current_step.step_id, session_id=session_id, notebook_id=self.notebook_id)
                return

            assert cell_tools is not None
            assert session_id is not None

        try:
            if step_type == StepType.MARKDOWN:
                 # Pass session_id and notebook_id to event
                 yield StatusUpdateEvent(status=StatusType.AGENT_ITERATING, agent_type=AgentType.MARKDOWN, message="Generating Markdown...", attempt=None, max_attempts=None, reason=None, step_id=current_step.step_id, original_plan_step_id=None, session_id=session_id, notebook_id=self.notebook_id)
                 result = await self.markdown_generator.run(full_description)
                 if isinstance(result.output, MarkdownQueryResult):
                     final_result = result.output
                 else:
                     ai_logger.warning(f"Markdown generator did not return MarkdownQueryResult in output. Got: {type(result.output)}. Falling back.")
                     fallback_content = str(result.output) if result.output else ''
                     final_result = MarkdownQueryResult(query=description, data=fallback_content)
                 yield {"type": "final_step_result", "result": final_result}

            elif step_type == StepType.GITHUB:
                # Lazy initialization
                if self.github_generator is None:
                    ai_logger.info("Lazily initializing GitHubQueryAgent.")
                    self.github_generator = GitHubQueryAgent(notebook_id=self.notebook_id)
                    
                ai_logger.info(f"Processing GitHub step {current_step.step_id} using run_query generator...")
                
                # Explicit check to satisfy type checker, though assertion should prevent None
                if session_id is None:
                    # This should ideally not be reached due to the assertion earlier
                    raise ValueError("session_id cannot be None when calling github_generator.run_query")

                async for event_data in self.github_generator.run_query(
                    full_description,
                    session_id=session_id, # Now type checker knows it's str
                    notebook_id=self.notebook_id
                ):
                    # Forward specific event types
                    match event_data:
                        case StatusUpdateEvent() | ToolSuccessEvent() | ToolErrorEvent():
                            # These events should have necessary IDs set by the generator if defined in their model
                            yield event_data
                        case FinalStatusEvent():
                            ai_logger.info(f"GitHub query finished for step {current_step.step_id} with status: {event_data.status}")
                        case _: # Ensure this matches BaseEvent too
                            if isinstance(event_data, BaseEvent):
                                ai_logger.warning(f"Received unexpected BaseEvent type {type(event_data)} from GitHub agent during step {current_step.step_id}: {event_data!r}")
                            else:
                                ai_logger.warning(f"Received unexpected non-BaseEvent type {type(event_data)} from GitHub agent during step {current_step.step_id}: {event_data!r}")
                # Yield the specific event class, providing required fields from the current definition
                yield GitHubStepProcessingCompleteEvent(
                    original_plan_step_id=current_step.step_id,
                    session_id=session_id,
                    notebook_id=self.notebook_id
                )
            
            elif step_type == StepType.FILESYSTEM:
                # Similar handling as GitHub, but using FileSystemAgent
                if not cell_tools or not session_id:
                    error_msg = "Internal Error: Missing required parameters (cell_tools, session_id) for Filesystem step processing."
                    ai_logger.error(error_msg)
                    yield ToolErrorEvent(error=error_msg, agent_type=AgentType.FILESYSTEM, status=StatusType.FATAL_ERROR, attempt=None, tool_call_id=None, tool_name=None, tool_args=None, message=None, original_plan_step_id=current_step.step_id, session_id=session_id, notebook_id=self.notebook_id)
                    return
                
                assert cell_tools is not None
                assert session_id is not None
                
                # Lazy initialization
                if self.filesystem_generator is None:
                    ai_logger.info("Lazily initializing FileSystemAgent.")
                    self.filesystem_generator = FileSystemAgent(notebook_id=self.notebook_id)
                    
                ai_logger.info(f"Processing Filesystem step {current_step.step_id} using run_query generator...")
                
                async for event_data in self.filesystem_generator.run_query(
                    full_description,
                    session_id=session_id,
                    notebook_id=self.notebook_id
                ):
                    # Forward specific event types directly
                    # The FileSystemAgent should yield ToolSuccessEvent, ToolErrorEvent, StatusUpdateEvent, etc.
                    match event_data:
                        case ToolSuccessEvent() | ToolErrorEvent() | StatusUpdateEvent():
                             # Add original_plan_step_id to events before yielding if missing?
                             # Or expect AIAgent._handle_step_execution to handle this association?
                             # Let's forward for now, handle in _handle_step_execution.
                             yield event_data 
                        case FinalStatusEvent():
                            ai_logger.info(f"FileSystem query finished for step {current_step.step_id} with status: {event_data.status}")
                        case _:
                            if isinstance(event_data, BaseEvent):
                                ai_logger.warning(f"Received unexpected BaseEvent type {type(event_data)} from Filesystem agent during step {current_step.step_id}: {event_data!r}")
                            else:
                                ai_logger.warning(f"Received unexpected non-BaseEvent type {type(event_data)} from Filesystem agent during step {current_step.step_id}: {event_data!r}")
                
                # Yield the specific Step Processing Complete event for Filesystem
                yield FileSystemStepProcessingCompleteEvent(
                    original_plan_step_id=current_step.step_id,
                    session_id=session_id,
                    notebook_id=self.notebook_id
                )

            elif step_type == StepType.PYTHON:
                # Similar handling as GitHub, but using FileSystemAgent
                if not cell_tools or not session_id:
                    error_msg = "Internal Error: Missing required parameters (cell_tools, session_id) for Python step processing."
                    ai_logger.error(error_msg)
                    yield ToolErrorEvent(error=error_msg, agent_type=AgentType.PYTHON, status=StatusType.FATAL_ERROR, attempt=None, tool_call_id=None, tool_name=None, tool_args=None, message=None, original_plan_step_id=current_step.step_id, session_id=session_id, notebook_id=self.notebook_id)
                    return
                
                assert cell_tools is not None
                assert session_id is not None
                
                # Lazy initialization
                if self.filesystem_generator is None:
                    ai_logger.info("Lazily initializing FileSystemAgent.")
                    self.filesystem_generator = FileSystemAgent(notebook_id=self.notebook_id)
                    
                ai_logger.info(f"Processing Python step {current_step.step_id} using run_query generator...")
                
                async for event_data in self.filesystem_generator.run_query(
                    full_description,
                    session_id=session_id,
                    notebook_id=self.notebook_id
                ):
                    # Forward specific event types directly
                    # The FileSystemAgent should yield ToolSuccessEvent, ToolErrorEvent, StatusUpdateEvent, etc.
                    match event_data:
                        case ToolSuccessEvent() | ToolErrorEvent() | StatusUpdateEvent():
                             # Add original_plan_step_id to events before yielding if missing?
                             # Or expect AIAgent._handle_step_execution to handle this association?
                             # Let's forward for now, handle in _handle_step_execution.
                             yield event_data 
                        case FinalStatusEvent():
                            ai_logger.info(f"FileSystem query finished for step {current_step.step_id} with status: {event_data.status}")
                        case _:
                            if isinstance(event_data, BaseEvent):
                                ai_logger.warning(f"Received unexpected BaseEvent type {type(event_data)} from Filesystem agent during step {current_step.step_id}: {event_data!r}")
                            else:
                                ai_logger.warning(f"Received unexpected non-BaseEvent type {type(event_data)} from Filesystem agent during step {current_step.step_id}: {event_data!r}")
                
                # Yield the specific Step Processing Complete event for Filesystem
                yield FileSystemStepProcessingCompleteEvent(
                    original_plan_step_id=current_step.step_id,
                    session_id=session_id,
                    notebook_id=self.notebook_id
                )

            else:
                error_msg = f"Unsupported step type encountered in generate_content: {step_type_val}"
                ai_logger.error(error_msg)
                yield ToolErrorEvent(
                    error=error_msg, 
                    agent_type=AgentType.UNKNOWN, 
                    status=StatusType.ERROR, 
                    original_plan_step_id=current_step.step_id, 
                    session_id=session_id, 
                    notebook_id=self.notebook_id,
                    # Explicitly provide None for optional fields
                    attempt=None,
                    tool_call_id=None,
                    tool_name=None,
                    tool_args=None,
                    message=None
                )
        except Exception as e:
            error_msg = f"Error during generate_content for step type {step_type_val}: {e}"
            ai_logger.error(error_msg, exc_info=True)
            yield ToolErrorEvent(
                error=str(e), 
                agent_type=AgentType.UNKNOWN, 
                status=StatusType.FATAL_ERROR, 
                original_plan_step_id=current_step.step_id, 
                session_id=session_id, 
                notebook_id=self.notebook_id,
                # Explicitly provide None for optional fields
                attempt=None,
                tool_call_id=None,
                tool_name=None,
                tool_args=None,
                message=None
            )