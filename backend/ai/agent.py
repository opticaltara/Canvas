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
from backend.config import get_settings
from backend.core.query_result import (
    QueryResult, 
    MarkdownQueryResult, 
    GithubQueryResult,
    SummarizationQueryResult
)
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams
from backend.core.notebook import Notebook
from backend.services.notebook_manager import get_notebook_manager
from backend.db.database import get_db
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
    SummaryStartedEvent,
    SummaryUpdateEvent,
    SummaryCellCreatedEvent,
    SummaryCellErrorEvent,
    InvestigationCompleteEvent,
    FinalStatusEvent,
    ToolSuccessEvent,
    ToolErrorEvent,
    StatusUpdateEvent
)


import asyncio

from backend.ai.prompts.investigation_prompts import (
    INVESTIGATION_PLANNER_SYSTEM_PROMPT,
    PLAN_REVISER_SYSTEM_PROMPT,
    MARKDOWN_GENERATOR_SYSTEM_PROMPT
)

ai_logger = logging.getLogger("ai")

settings = get_settings()

class StepType(str, Enum):
    MARKDOWN = "markdown"
    GITHUB = "github"


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
        
        # Log the received MCP server map and available sources
        ai_logger.info(f"AIAgent for notebook {self.notebook_id} initialized with available data sources: {self.available_data_sources}")
        
        # Format the planner prompt with the available data sources
        available_data_sources_str = ', '.join(self.available_data_sources) if self.available_data_sources else 'None'
        formatted_planner_prompt = INVESTIGATION_PLANNER_SYSTEM_PROMPT.format(
            available_data_sources_str=available_data_sources_str
        )

        # Create the full list of MCPServerHTTP for general agents
        all_mcps_list = list(self.mcp_server_map.values())
        ai_logger.info(f"Initializing general agents with {len(all_mcps_list)} HTTP MCP servers.")
        
        # Initialize investigation planner using the formatted prompt from the constants file
        self.investigation_planner = Agent(
            self.model,
            deps_type=InvestigationDependencies,
            output_type=InvestigationPlanModel,
            system_prompt=formatted_planner_prompt
        )

        # Initialize plan reviser using the constant
        self.plan_reviser: Optional[Agent[PlanRevisionRequest, PlanRevisionResult]] = None

        # Initialize markdown generator using the constant
        self.markdown_generator = Agent(
            self.model,
            output_type=MarkdownQueryResult,
            mcp_servers=all_mcps_list,
            system_prompt=MARKDOWN_GENERATOR_SYSTEM_PROMPT
        )

        # Initialize specialized agents without passing mcp_servers
        self.github_generator: Optional[GitHubQueryAgent] = None
        self.summarization_generator: Optional[SummarizationAgent] = None

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
        StatusUpdateEvent, GitHubToolCellCreatedEvent, GitHubToolErrorEvent, StepExecutionCompleteEvent
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
        notebook_id_str = str(self.notebook_id) # Get notebook ID as string

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

            # 1. Handle final results for NON-GitHub steps
            elif isinstance(event_data, dict) and event_data.get("type") == "final_step_result" and current_step.step_type != StepType.GITHUB:
                 final_result_data = event_data.get("result")
                 if not isinstance(final_result_data, QueryResult):
                      ai_logger.error(f"final_step_result dict did not contain QueryResult for step {current_step.step_id}")
                      final_result_data = None # Reset if type mismatch
                      step_error = step_error or "Internal error: Invalid final result format"
                 break # Stop processing for this step

            # 2. Handle individual GitHub tool success events
            elif isinstance(event_data, ToolSuccessEvent) and current_step.step_type == StepType.GITHUB:
                 ai_logger.info(f"[_handle_step_execution] Received ToolSuccessEvent for plan step {current_step.step_id}: {event_data.tool_name}")
                 
                 if not event_data.tool_call_id or not event_data.tool_name:
                     ai_logger.error(f"CRITICAL: ToolSuccessEvent missing tool_call_id or tool_name for step {current_step.step_id}. Event: {event_data!r}")
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
                          ai_logger.debug(f"Serialized Pydantic model result for {event_data.tool_name}")
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
                         ai_logger.error(f"Failed even to convert tool result for {event_data.tool_name} to string: {str_err}", exc_info=True)
                 # --- End Conversion ---
                 
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
                         tool_args=event_data.tool_args,
                         result=event_data.tool_result,
                         cell_params=github_cell_params.model_dump()
                     )

                 except Exception as cell_creation_err:
                     error_msg = f"Failed creating cell for GitHub tool {event_data.tool_name} (plan step {current_step.step_id}): {cell_creation_err}"
                     ai_logger.error(error_msg, exc_info=True) # Log with traceback
                     github_step_has_tool_errors = True
                     step_error = step_error or error_msg
                     yield GitHubToolErrorEvent(
                         original_plan_step_id=current_step.step_id,
                         tool_call_id=event_data.tool_call_id, # Include ID if available
                         tool_name=event_data.tool_name,
                         tool_args=event_data.tool_args,
                         error=error_msg
                     )

            # 3. Handle individual GitHub tool error events
            elif isinstance(event_data, ToolErrorEvent) and current_step.step_type == StepType.GITHUB:
                 ai_logger.error(f"[_handle_step_execution] Received ToolErrorEvent for plan step {current_step.step_id}: {event_data.tool_name} - {event_data.error}")
                 github_step_has_tool_errors = True
                 step_error = step_error or event_data.error
                 tool_name = event_data.tool_name or 'UnknownTool'
                 tool_args = event_data.tool_args or {}
                 tool_error_content = event_data.error
                 pydantic_ai_tool_call_id = event_data.tool_call_id

                 # Attempt to create an error cell anyway
                 error_cell_content = f"GitHub Error: {tool_name}\n```\n{tool_error_content}\n```"
                 
                 # Get dependencies based on the PLAN step's dependencies
                 dependency_cell_ids: List[UUID] = []
                 for dep_plan_id in current_step.dependencies:
                     dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_plan_id, []))

                 # --- Generate UUID and update metadata ---
                 error_cell_tool_call_id = uuid4() # Generate UUID for error cell link
                 error_cell_metadata = {
                     "session_id": session_id,
                     "original_plan_step_id": current_step.step_id,
                     "is_error_cell": True,
                     "external_tool_call_id": pydantic_ai_tool_call_id # Store original event ID
                 }
                 # --- End UUID/Metadata ---

                 error_cell_kwargs = {
                     "notebook_id": notebook_id_str,
                     "cell_type": CellType.GITHUB, # Still a github step origin
                     "content": error_cell_content,
                     "metadata": error_cell_metadata, # Use updated metadata
                     "dependencies": dependency_cell_ids,
                     "connection_id": default_github_connection_id, # Use fetched ID
                     "tool_call_id": error_cell_tool_call_id, # Pass NEWLY generated UUID
                     "tool_name": tool_name,
                     "tool_arguments": tool_args,
                     "result": CellResult(content=None, error=tool_error_content, execution_time=0.0), # Store error in result
                     "status": CellStatus.ERROR # Mark cell status as error
                 }
                 error_cell_params = CreateCellParams(**error_cell_kwargs)

                 try:
                     error_cell_creation_result = await cell_tools.create_cell(params=error_cell_params)
                     error_cell_id = error_cell_creation_result.get("cell_id")
                     if error_cell_id:
                         error_cell_uuid = UUID(error_cell_id)
                         if current_step.step_id not in plan_step_id_to_cell_ids:
                             plan_step_id_to_cell_ids[current_step.step_id] = []
                         plan_step_id_to_cell_ids[current_step.step_id].append(error_cell_uuid)
                         ai_logger.info(f"Created error cell {error_cell_id} for failed GitHub tool {tool_name}")
                     else:
                         ai_logger.error("Failed to create error cell for GitHub tool error, no cell_id returned.")
                 except Exception as error_cell_err:
                     ai_logger.error(f"Failed creating error cell for GitHub tool error: {error_cell_err}", exc_info=True)

                 # Yield the original error event regardless of error cell creation success
                 yield GitHubToolErrorEvent(
                     original_plan_step_id=current_step.step_id,
                     tool_call_id=pydantic_ai_tool_call_id,
                     tool_name=event_data.tool_name,
                     tool_args=event_data.tool_args,
                     error=event_data.error
                 )

            # 4. Handle GitHub step completion marker
            elif isinstance(event_data, dict) and event_data.get("type") == EventType.GITHUB_STEP_PROCESSING_COMPLETE.value:
                 ai_logger.info(f"[_handle_step_execution] Received github_step_processing_complete for plan step {current_step.step_id}")
                 break # Stop processing for this step
                 
            # 5. Handle other unexpected dict events (should be less common now)
            elif isinstance(event_data, dict):
                 ai_logger.warning(f"[_handle_step_execution] Unexpected dict event type '{event_data.get('type')}' while processing step {current_step.step_id}. Data: {event_data}")
            else:
                 ai_logger.error(f"[_handle_step_execution] FATAL: generate_content yielded non-event/dict type {type(event_data)} for step {current_step.step_id}")
                 step_error = step_error or "Internal error: Invalid generator output"
                 break 
        
        # Yield final completion event using the Pydantic model
        yield StepExecutionCompleteEvent(
            step_id=current_step.step_id,
            final_result_data=final_result_data, # Will be None for GitHub steps
            step_error=step_error,
            github_step_has_tool_errors=github_step_has_tool_errors
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
        StepCompletedEvent, StepErrorEvent, GitHubToolCellCreatedEvent, 
        GitHubToolErrorEvent, StatusUpdateEvent, SummaryStartedEvent, 
        SummaryUpdateEvent, SummaryCellCreatedEvent, SummaryCellErrorEvent, 
        InvestigationCompleteEvent
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
        db_gen = get_db()
        db: Session = next(db_gen)
        try:
            if not cell_tools:
                raise ValueError("cell_tools is required for creating cells")
                
            notebook_id_str = notebook_id or self.notebook_id
            if not notebook_id_str:
                raise ValueError("notebook_id is required for creating cells")
            
            # Fetch the actual Notebook object - needed for dependency graph and records
            notebook_manager = get_notebook_manager() 
            try:
                notebook_uuid = UUID(notebook_id_str)
                notebook: Notebook = notebook_manager.get_notebook(db=db, notebook_id=notebook_uuid) # Pass db and use keyword arg
            except ValueError:
                ai_logger.error(f"Invalid notebook_id format: {notebook_id_str}")
                raise ValueError(f"Invalid notebook_id format: {notebook_id_str}")
            except Exception as e:
                 ai_logger.error(f"Failed to fetch notebook {notebook_id_str}: {e}", exc_info=True)
                 raise ValueError(f"Failed to fetch notebook {notebook_id_str}")

            plan = await self.create_investigation_plan(query, notebook_id_str, message_history)
            yield PlanCreatedEvent(thinking=plan.thinking) # Use model
            
            plan_step_id_to_cell_ids: Dict[str, List[UUID]] = {} # Map plan step to list of created cell UUIDs
            executed_steps: Dict[str, Any] = {}
            step_results: Dict[str, Any] = {}

            # --- Plan Explanation Cell (only if multiple steps) --- 
            if len(plan.steps) > 1:
                plan_cell_params = CreateCellParams(
                    notebook_id=notebook_id_str,
                    cell_type="markdown",
                    content=f"# Investigation Plan\n\n## Steps:\n" +
                        "\n".join(f"- `{step.step_id}`: {step.step_type.value}" for step in plan.steps),
                    metadata={
                        "session_id": session_id,
                        "step_id": "plan",
                        "dependencies": []
                    },
                    tool_call_id=uuid4() # Add generated UUID
                )
                plan_cell_result = await cell_tools.create_cell(params=plan_cell_params)
                yield PlanCellCreatedEvent(
                    cell_params=plan_cell_params.model_dump(mode='json'), 
                    cell_id=plan_cell_result.get("cell_id", "")
                )
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
                     if not unresolved_deps and remaining_steps:
                         # Should not happen if logic is correct, but indicates an issue.
                         ai_logger.error("No executable steps but plan not complete and no unresolved dependencies found.")
                         yield StepErrorEvent(step_id="PLAN_STALLED", error="Plan execution stalled unexpectedly.", agent_type=AgentType.INVESTIGATION_PLANNER, step_type=None, cell_id=None, cell_params=None, result=None) # Explicitly pass None for optional args to satisfy linter
                     elif not remaining_steps:
                         # Plan is actually complete
                         ai_logger.info("Plan execution complete.")
                     else:
                        # Normal case: waiting for dependencies
                        ai_logger.info(f"Plan execution waiting for dependencies. Remaining steps: {[s.step_id for s in remaining_steps]}")
                        # Optional: Yield a status indicating waiting state (using StatusUpdateEvent)
                        # yield StatusUpdateEvent(status=StatusType.WAITING_FOR_DEPENDENCIES, message=f"Waiting for dependencies for steps: {[s.step_id for s in remaining_steps]}")
                     break # Exit the loop if no steps are immediately executable
                    
                current_step = executable_steps[0]
                remaining_steps.remove(current_step)
                
                # Determine agent type string based on step type
                agent_type_enum = AgentType.GITHUB if current_step.step_type == StepType.GITHUB else AgentType.MARKDOWN # Map step type to AgentType
                # Map other step types if they exist later...

                yield StepStartedEvent(step_id=current_step.step_id, agent_type=agent_type_enum) # Use model
                step_start_time = datetime.now(timezone.utc)

                # --- Call helper to execute step and handle its events --- 
                async for execution_event in self._handle_step_execution(
                     current_step=current_step,
                     executed_steps=executed_steps,
                     step_results=step_results,
                     cell_tools=cell_tools, # Pass cell_tools
                     plan_step_id_to_cell_ids=plan_step_id_to_cell_ids, # Pass the map
                     session_id=session_id,
                     step_start_time=step_start_time
                 ):
                    # Forward events yielded by the helper (StatusUpdate, GitHubToolCell/Error Events)
                    if isinstance(execution_event, (StatusUpdateEvent, GitHubToolCellCreatedEvent, GitHubToolErrorEvent)):
                        yield execution_event
                    
                    # --- Process the final completion event from the helper --- 
                    elif isinstance(execution_event, StepExecutionCompleteEvent):
                        final_result_data = execution_event.final_result_data
                        step_error = execution_event.step_error
                        github_step_has_tool_errors = execution_event.github_step_has_tool_errors
                        
                        # --- Process NON-GitHub Step Result (Now happens *after* execution) --- 
                        created_cell_id = None
                        cell_params_for_step = None # Initialize
                        
                        if current_step.step_type != StepType.GITHUB:
                            if final_result_data:
                                # Prepare cell params
                                query_content = final_result_data.query
                                if current_step.step_type == StepType.MARKDOWN:
                                    query_content = str(final_result_data.data)
                                
                                # --- Generate UUID and update metadata ---
                                cell_tool_call_id = uuid4() # Generate UUID for non-github cell link
                                cell_metadata_multi = {
                                    "session_id": session_id,
                                    "original_plan_step_id": current_step.step_id # Store original step ID
                                }
                                # --- End UUID/Metadata ---

                                # Get actual cell dependencies from the map
                                dependency_cell_ids: List[UUID] = []
                                for dep_plan_id in current_step.dependencies:
                                    dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_plan_id, []))

                                # Prepare kwargs for creating the cell
                                create_cell_kwargs = {
                                    "notebook_id": notebook_id_str,
                                    "cell_type": current_step.step_type.value, # Use step type string
                                    "content": query_content,
                                    "metadata": cell_metadata_multi, # Use updated metadata
                                    "dependencies": dependency_cell_ids, # Set actual cell dependencies
                                    "tool_call_id": cell_tool_call_id, # Pass NEWLY generated UUID
                                    "tool_name": f"step_{current_step.step_type.value}", # Synthesize tool name
                                    "tool_arguments": current_step.parameters, # Use plan step params as args
                                    "result": CellResult( # Set initial result from QueryResult
                                        content=final_result_data.data,
                                        error=final_result_data.error,
                                        execution_time=0.0 # Execution time might need calculation
                                    ),
                                    "status": CellStatus.ERROR if final_result_data.error else CellStatus.SUCCESS
                                }
                                
                                cell_params_for_step = CreateCellParams(**create_cell_kwargs)
                                
                                # Directly create the cell
                                try:
                                    cell_creation_result = await cell_tools.create_cell(params=cell_params_for_step)
                                    created_cell_id = cell_creation_result.get("cell_id")
                                    if not created_cell_id:
                                        raise ValueError(f"create_cell tool did not return a cell_id for step {current_step.step_id}")
                                    created_cell_uuid = UUID(created_cell_id)

                                    # Update the map tracking created cells for this plan step
                                    if current_step.step_id not in plan_step_id_to_cell_ids:
                                        plan_step_id_to_cell_ids[current_step.step_id] = []
                                    plan_step_id_to_cell_ids[current_step.step_id].append(created_cell_uuid)

                                except Exception as cell_creation_err:
                                    # Handle creation failure
                                    step_error = step_error or f"Failed to create cell for step {current_step.step_id}: {cell_creation_err}"
                                    if final_result_data:
                                        final_result_data.error = step_error 
                                    else:
                                        final_result_data = QueryResult(query=current_step.description, data=None, error=step_error)
                                    
                            elif not final_result_data: # Handle missing final result
                                step_error = step_error or f"Failed to get final result data for step {current_step.step_id}."
                                ai_logger.error(f"final_result_data is None for step {current_step.step_id} after execution. Error: {step_error}")
                                final_result_data = QueryResult(query=current_step.description, data=None, error=step_error)
                        
                        # --- Update Execution State for the PLAN step --- 
                        plan_step_has_error = False
                        if current_step.step_type == StepType.GITHUB:
                            plan_step_has_error = github_step_has_tool_errors
                            executed_steps[current_step.step_id] = {
                                 "step": current_step.model_dump(),
                                 "content": f"Completed (See individual tool cells generated for this step)", 
                                 "error": step_error 
                            }
                            step_results[current_step.step_id] = None 
                        elif final_result_data: 
                            plan_step_has_error = bool(final_result_data.error) or bool(step_error)
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": getattr(final_result_data, 'data', None),
                                "error": step_error or getattr(final_result_data, 'error', None)
                            }
                            step_results[current_step.step_id] = final_result_data
                        else:
                            plan_step_has_error = True
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": None,
                                "error": step_error or f"Missing final_result_data for step {current_step.step_id}"
                            }
                        
                        # No separate record propagation needed, dependencies handled via cell creation

                        # --- Yield Completion/Error Status for the PLAN STEP --- 
                        if current_step.step_type != StepType.GITHUB:
                            if plan_step_has_error:
                                yield StepErrorEvent( # Use model
                                    step_id=current_step.step_id,
                                    step_type=current_step.step_type.value,
                                    cell_id=str(created_cell_id) if created_cell_id else None, 
                                    cell_params=cell_params_for_step.model_dump() if cell_params_for_step else None, 
                                    result=getattr(final_result_data, 'model_dump', lambda: None)() if final_result_data else None,
                                    error=step_error or getattr(final_result_data, 'error', None),
                                    agent_type=agent_type_enum # Use the mapped enum
                                )
                            else:
                                yield StepCompletedEvent( # Use model
                                    step_id=current_step.step_id,
                                    step_type=current_step.step_type.value,
                                    cell_id=str(created_cell_id) if created_cell_id else None,
                                    cell_params=cell_params_for_step.model_dump() if cell_params_for_step else None,
                                    result=getattr(final_result_data, 'model_dump', lambda: None)() if final_result_data else None,
                                    agent_type=agent_type_enum # Use the mapped enum
                                )
                                
                        # Log overall status for the plan step
                        final_plan_step_status = "error" if plan_step_has_error else "success"
                        ai_logger.info(f"Processed plan step: {current_step.step_id} with final status: {final_plan_step_status}")
                        
                        # Break from the inner loop after handling completion
                        break 
                    else:
                        ai_logger.warning(f"Investigate loop received unexpected event type from _handle_step_execution: {type(execution_event)}")


            ai_logger.info(f"Investigation plan execution finished for notebook {notebook_id_str}")

            # --- Final Summarization Step --- 
            ai_logger.info("Starting final summarization step.")
            yield SummaryStartedEvent() # Use model
            
            # Format context for summarizer
            findings_text_lines = []
            # Iterate through executed_steps in the order they appear in the original plan for better flow
            for step in plan.steps:
                if step.step_id in executed_steps:
                    step_info = executed_steps[step.step_id]
                    step_obj = step_info.get("step")
                    content = step_info.get("content") # Content for GitHub steps is now just a status message
                    error = step_info.get("error")
                    desc = step_obj.get("description", "") if step_obj else ""
                    type_val = step_obj.get("step_type", "unknown") if step_obj else "unknown"
                    
                    result_str = f"Data: {str(content)[:500]}..." if content else "No data/cells returned."
                    if error:
                        result_str = f"Error: {error}"
                        
                    # Add note if GitHub step indicates individual cells were generated
                    if type_val == StepType.GITHUB.value and content and "individual tool cells" in str(content):
                        result_str += " (See individual GitHub cells above for details)"

                    findings_text_lines.append(f"Step {step.step_id} ({type_val}): {desc}\nResult:\n{result_str}")

            findings_text = "\n---\n".join(findings_text_lines)
            if not findings_text:
                findings_text = "No investigation steps were successfully executed or produced results."
                
            summarizer_input = f"""Original User Query:
{query}

Investigation Findings:
---
{findings_text}
"""

            summary_content = ""
            summary_error = None
            summary_result: Optional[SummarizationQueryResult] = None
            try:
                # Lazy initialization
                if self.summarization_generator is None:
                    ai_logger.info("Lazily initializing SummarizationAgent.")
                    self.summarization_generator = SummarizationAgent(notebook_id=self.notebook_id)
                    
                async for result_part in self.summarization_generator.run_summarization(
                    text_to_summarize=summarizer_input, # Pass the structured input
                    original_request=query # Keep the original query separate if needed
                ):
                    # Expect SummarizationQueryResult or Dict for status
                    if isinstance(result_part, SummarizationQueryResult):
                        summary_result = result_part
                        summary_content = summary_result.data
                        summary_error = summary_result.error
                        # Yield SummaryUpdateEvent with message
                        yield SummaryUpdateEvent(update_info={"message": "Summary generated"})
                        break # Got the final summary object
                    elif isinstance(result_part, dict) and "status" in result_part:
                        # Yield SummaryUpdateEvent with the dictionary
                        yield SummaryUpdateEvent(update_info=result_part)
                    else:
                        ai_logger.warning(f"Unexpected item yielded from summarization generator: {type(result_part)}")
                
                if summary_result is None:
                    summary_error = "Summarization agent did not return a final result object."
                    ai_logger.error(summary_error)
                    
            except Exception as e:
                summary_error = f"Error during final summarization: {e}"
                ai_logger.error(summary_error, exc_info=True)
                summary_content = f"# Final Summary Error\n\nAn error occurred while generating the final summary:\n```\n{summary_error}\n```"

            # --- Create Final Summary Cell ---
            if not summary_error and not summary_content:
                 summary_content = "# Final Summary\n\nThe investigation completed, but no summary content was generated."
                 ai_logger.warning("Summarization finished without error but produced empty content.")
            elif summary_error and not summary_content:
                 # Ensure content reflects the error if summary_content wasn't set in except block
                 summary_content = f"# Final Summary Error\n\nAn error occurred during summarization: {summary_error}"

            # Determine dependencies for the summary cell (all cells created during the investigation)
            summary_dependency_cell_ids: List[UUID] = []
            for step_id in executed_steps.keys(): # Iterate executed plan steps
                summary_dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(step_id, []))

            summary_cell_params = CreateCellParams(
                notebook_id=notebook_id_str,
                cell_type="markdown",
                content=summary_content, 
                metadata={
                    "session_id": session_id,
                    "step_id": "final_summary",
                    # "dependencies": list(executed_steps.keys()) # Old way, use actual cell IDs now
                },
                dependencies=summary_dependency_cell_ids, # Set actual cell dependencies
                tool_call_id=uuid4() # Add generated UUID
            )
            try:
                summary_cell_result = await cell_tools.create_cell(params=summary_cell_params)
                summary_cell_id = summary_cell_result.get("cell_id")
                yield SummaryCellCreatedEvent( # Use model
                    cell_params=summary_cell_params.model_dump(),
                    cell_id=summary_cell_id or None,
                    error=summary_error # Include any summary generation error here
                )
                ai_logger.info(f"Final summary cell created ({summary_cell_id}) for notebook {notebook_id_str}")
            except Exception as cell_err:
                ai_logger.error(f"Failed to create final summary cell: {cell_err}", exc_info=True)
                yield SummaryCellErrorEvent(error=str(cell_err)) # Use model

            # --- Investigation Complete ---
            # Now yield the final completion status after attempting summarization
            yield InvestigationCompleteEvent() # Use model

        finally:
            # --- IMPORTANT: Save the notebook state --- 
            if 'notebook' in locals() and notebook is not None:
                try:
                    notebook_manager = get_notebook_manager() # Ensure manager is available
                    # Call save_notebook assuming its signature will be updated to accept these args
                    # The linter will likely complain based on the current no-op implementation
                    notebook_manager.save_notebook(db=db, notebook_id=notebook.id, notebook=notebook)
                    ai_logger.info(f"Successfully initiated save for notebook {notebook.id} state after investigation.")
                except Exception as save_err:
                    ai_logger.error(f"Failed to initiate save for notebook {notebook.id} state after investigation: {save_err}", exc_info=True)
            # --- End Save Notebook --- 
            
            try:
                next(db_gen) # Close the session
            except StopIteration:
                pass # Expected
            except Exception as e:
                ai_logger.error(f"Error closing database session in AIAgent.investigate: {e}", exc_info=True)
        

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
    ) -> AsyncGenerator[Union[Dict[str, Any], StatusUpdateEvent, ToolSuccessEvent, ToolErrorEvent, StepErrorEvent], None]:
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
                    dep_desc = dep_step_info.get('step', {}).get('description', '')
                    dep_error = dep_step_info.get('error')
                    status = "Error" if dep_error else "Success"
                    context_line = f"- Dependency {dep_id} ({status}): {dep_desc[:100]}..."
                    if dep_error:
                        context_line += f" (Error: {str(dep_error)[:100]}...)"
                    dependency_context.append(context_line)
            if dependency_context:
                 context = f"\n\nContext from Dependencies:\n" + "\n".join(dependency_context)

        agent_prompt = description

        if current_step.parameters:
            params_str = "\n".join([f"- {k}: {v}" for k, v in current_step.parameters.items()])
            agent_prompt += f"\n\nParameters for this step:\n{params_str}"

        agent_prompt += context

        full_description = agent_prompt
        ai_logger.info(f"[generate_content] Passing to sub-agent ({step_type.value}): {full_description}")

        final_result: Optional[QueryResult] = None

        if step_type == StepType.GITHUB:
            if not cell_tools or not session_id:
                error_msg = "Internal Error: Missing required parameters (cell_tools, session_id) for GitHub step processing."
                ai_logger.error(error_msg)
                yield ToolErrorEvent(error=error_msg, agent_type=AgentType.GITHUB, status=StatusType.FATAL_ERROR, attempt=None, tool_call_id=None, tool_name=None, tool_args=None, message=None, original_plan_step_id=current_step.step_id) # Use model, provide None for missing
                return
             
            assert cell_tools is not None
            assert session_id is not None

        try:
            if step_type == StepType.MARKDOWN:
                 yield StatusUpdateEvent(status=StatusType.AGENT_ITERATING, agent_type=AgentType.MARKDOWN, message="Generating Markdown...", attempt=None, max_attempts=None, reason=None, step_id=current_step.step_id, original_plan_step_id=None) # Use model, provide None for missing
                 result = await self.markdown_generator.run(full_description)
                 if isinstance(result.output, MarkdownQueryResult):
                     final_result = result.output
                 else:
                     ai_logger.warning(f"Markdown generator did not return MarkdownQueryResult in output. Got: {type(result.output)}. Falling back.")
                     fallback_content = str(result.output) if result.output else ''
                     final_result = MarkdownQueryResult(query=description, data=fallback_content)
                 # Yield final result in a dictionary wrapper (as before)
                 yield {"type": "final_step_result", "result": final_result} # Use string literal
            elif step_type == StepType.GITHUB:
                # Lazy initialization
                if self.github_generator is None:
                    ai_logger.info("Lazily initializing GitHubQueryAgent.")
                    self.github_generator = GitHubQueryAgent(notebook_id=self.notebook_id)
                    
                # --- MODIFIED: Stream GitHub agent events directly --- 
                ai_logger.info(f"Processing GitHub step {current_step.step_id} using run_query generator...")
                
                async for event_data in self.github_generator.run_query(full_description):
                    # Now expect Pydantic models directly from github_generator
                    match event_data:
                        case StatusUpdateEvent() | ToolSuccessEvent() | ToolErrorEvent():
                            ai_logger.debug(f"Yielding event from GitHub agent: {event_data.type} for step {current_step.step_id}")
                            # Add original plan step ID if missing (safer approach)
                            if getattr(event_data, 'original_plan_step_id', None) is None:
                                # Re-yield with the ID added if possible, or just yield as is
                                try:
                                    # Attempt creation with updated ID - might need specific event type handling
                                    # For simplicity, yield as is for now, assuming github_agent sets it.
                                    pass 
                                except Exception as copy_err:
                                    ai_logger.warning(f"Could not add original_plan_step_id to event {event_data.type}: {copy_err}")
                            yield event_data
                        
                        case FinalStatusEvent(): 
                            ai_logger.info(f"GitHub query finished for step {current_step.step_id} with status: {event_data.status}")
                        case _:
                            ai_logger.warning(f"Received unexpected event type {type(event_data)} from GitHub agent during step {current_step.step_id}: {event_data!r}")

                yield {"type": EventType.GITHUB_STEP_PROCESSING_COMPLETE.value, "original_plan_step_id": current_step.step_id, "agent_type": AgentType.GITHUB}
            
            else:
                error_msg = f"Unsupported step type encountered in generate_content: {step_type}"
                ai_logger.error(error_msg)
                # Yield StepErrorEvent
                yield StepErrorEvent(step_id=current_step.step_id, error=error_msg, agent_type=AgentType.UNKNOWN, step_type=step_type.value, cell_id=None, cell_params=None, result=None) # Use model

        except Exception as e:
            error_msg = f"Error during generate_content for step type {step_type}: {e}"
            ai_logger.error(error_msg, exc_info=True)
            # Yield StepErrorEvent for general errors
            yield StepErrorEvent(step_id=current_step.step_id, error=str(e), agent_type=AgentType.UNKNOWN, step_type=step_type.value, cell_id=None, cell_params=None, result=None) # Use model