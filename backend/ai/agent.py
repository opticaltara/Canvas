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
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple, Union
from uuid import UUID
from datetime import datetime, timezone

import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from sqlalchemy.orm import Session

from backend.core.cell import CellType
from backend.core.types import ToolCallID, ToolCallRecord
from backend.core.execution import register_tool_dependencies, propagate_tool_results
from backend.ai.log_query_agent import LogQueryAgent
from backend.ai.metric_query_agent import MetricQueryAgent
from backend.ai.github_query_agent import GitHubQueryAgent
from backend.ai.summarization_agent import SummarizationAgent
from backend.config import get_settings
from backend.core.query_result import (
    QueryResult, 
    LogQueryResult, 
    MetricQueryResult, 
    MarkdownQueryResult, 
    GithubQueryResult,
    SummarizationQueryResult
)
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams
from backend.core.notebook import Notebook
from backend.services.notebook_manager import get_notebook_manager
from backend.db.database import get_db

import asyncio # Add this import at the top of the file

# Import the prompts
from backend.ai.prompts.investigation_prompts import (
    INVESTIGATION_PLANNER_SYSTEM_PROMPT,
    PLAN_REVISER_SYSTEM_PROMPT,
    MARKDOWN_GENERATOR_SYSTEM_PROMPT
)

# Get logger for AI operations
ai_logger = logging.getLogger("ai")

# Get settings for API keys
settings = get_settings()

class StepType(str, Enum):
    MARKDOWN = "markdown"
    LOG = "log"
    METRIC = "metric"
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
        mcp_server_map: Optional[Dict[str, MCPServerHTTP]] = None
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
        self.log_generator: Optional[LogQueryAgent] = None
        self.metric_generator: Optional[MetricQueryAgent] = None
        self.github_generator: Optional[GitHubQueryAgent] = None
        self.summarization_generator: Optional[SummarizationAgent] = None

        ai_logger.info(f"AIAgent initialized for notebook {notebook_id}.")

    async def _create_cell_and_record(
        self,
        cell_params: CreateCellParams,
        tool_call_id: str, # Use the specific pydantic_ai_tool_call_id or the plan step_id
        tool_name: str,
        tool_args: Dict[str, Any],
        step_start_time: datetime,
        notebook: Notebook,
        step_id_to_record_id: Dict[str, UUID], # Map from plan step_id to record UUID
        plan_step_id: str, # The ID of the original plan step this cell belongs to
        cell_tools: NotebookCellTools # Pass cell_tools explicitly
    ) -> Tuple[Optional[str], Optional[ToolCallRecord]]:
        """Creates a notebook cell and a corresponding ToolCallRecord, links them, and registers dependencies.

        Args:
            cell_params: Parameters for creating the cell.
            tool_call_id: The specific ID for this tool execution (e.g., from pydantic-ai or plan step ID).
            tool_name: The name of the tool or step type.
            tool_args: Arguments passed to the tool/step.
            step_start_time: Approximate start time of the step execution.
            notebook: The Notebook object.
            step_id_to_record_id: Map from plan step_id to the UUID of its synthetic record.
            plan_step_id: The ID of the plan step this execution belongs to.
            cell_tools: The tools object for creating cells.

        Returns:
            A tuple containing the created cell ID (str) and the ToolCallRecord object, or (None, None) on failure.
        """
        created_cell_id: Optional[str] = None
        synthetic_record: Optional[ToolCallRecord] = None
        
        try:
            ai_logger.info(f"[_create_cell_and_record] Attempting cell creation for plan step {plan_step_id} / tool {tool_name}")
            cell_result = await cell_tools.create_cell(params=cell_params)
            created_cell_id = cell_result.get("cell_id")
            if not created_cell_id:
                raise ValueError("create_cell tool did not return a cell_id")
            ai_logger.info(f"[_create_cell_and_record] Cell {created_cell_id} created for {tool_name}")

            # Create Synthetic ToolCallRecord
            synthetic_record = ToolCallRecord(
                parent_cell_id=UUID(created_cell_id),
                pydantic_ai_tool_call_id=tool_call_id, # Use the provided ID
                tool_name=tool_name, # Use specific tool name or step type
                parameters=tool_args, 
                status="pending", # Will be updated later
                started_at=step_start_time,
            )
            
            # Store record in notebook (using its own UUID as key)
            notebook.tool_call_records[synthetic_record.id] = synthetic_record
            # Update step_id_to_record_id map *only if* this record represents the main plan step
            # (i.e., for non-GitHub steps where tool_call_id == plan_step_id)
            if tool_call_id == plan_step_id:
                 step_id_to_record_id[plan_step_id] = synthetic_record.id
                 
            ai_logger.info(f"[_create_cell_and_record] Record {synthetic_record.id} created for tool {tool_name} (plan step {plan_step_id})")

            # Link cell to record
            cell_uuid = UUID(created_cell_id)
            if cell_uuid in notebook.cells:
                notebook.cells[cell_uuid].tool_call_ids.append(synthetic_record.id)
            else:
                ai_logger.warning(f"[_create_cell_and_record] Could not find cell {created_cell_id} to link record {synthetic_record.id}")

            # Register Dependencies (using the main plan step dependencies)
            ai_logger.info(f"[_create_cell_and_record] Registering dependencies for record {synthetic_record.id} (plan step {plan_step_id})")
            # We register dependencies based on the *plan* step's dependencies
            register_tool_dependencies(synthetic_record, notebook, step_id_to_record_id)
            ai_logger.info(f"[_create_cell_and_record] Dependencies registered for record {synthetic_record.id}")
            
            return created_cell_id, synthetic_record

        except Exception as creation_error:
            ai_logger.error(f"[_create_cell_and_record] Failed for tool {tool_name} (plan step {plan_step_id}): {creation_error}", exc_info=True)
            # Clean up partial state if necessary (e.g., if cell created but record failed?)
            # For now, just return None
            return None, None

    async def _handle_step_execution(
        self,
        current_step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, Any],
        cell_tools: NotebookCellTools,
        notebook: Notebook,
        step_id_to_record_id: Dict[str, UUID],
        session_id: str,
        step_start_time: datetime
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Executes a single plan step by calling generate_content and handling its events.

        Yields status updates, github tool cell/error events, and a final 
        'step_execution_complete' event containing the outcome.

        Args:
            current_step: The plan step to execute.
            executed_steps: Dictionary of previously executed steps.
            step_results: Dictionary of results from previous steps.
            cell_tools: Tools for cell manipulation.
            notebook: The Notebook object.
            step_id_to_record_id: Map from plan step_id to record UUID.
            session_id: The current chat session ID.
            step_start_time: Approximate start time of the step.

        Yields:
            Dictionaries representing various events during step execution, culminating in
            a 'step_execution_complete' event.
        """
        final_result_data: Optional[QueryResult] = None
        step_error: Optional[str] = None
        github_step_has_tool_errors: bool = False
        notebook_id_str = str(notebook.id) # Get notebook ID as string

        async for event_data in self.generate_content(
            current_step=current_step,
            step_type=current_step.step_type,
            description=current_step.description,
            executed_steps=executed_steps,
            step_results=step_results,
            cell_tools=cell_tools, 
            notebook=notebook, 
            step_id_to_record_id=step_id_to_record_id, 
            session_id=session_id 
        ):
            if isinstance(event_data, dict):
                event_type = event_data.get("type")

                # 1. Handle final results for NON-GitHub steps
                if event_type == "final_step_result" and current_step.step_type != StepType.GITHUB:
                     final_result_data = event_data.get("result")
                     if not isinstance(final_result_data, QueryResult):
                          ai_logger.error(f"final_step_result dict did not contain QueryResult for step {current_step.step_id}")
                          final_result_data = None # Reset if type mismatch
                          step_error = step_error or "Internal error: Invalid final result format"
                     break # Stop processing for this step

                # 2. Handle individual GitHub tool success events
                elif event_type == "tool_success" and current_step.step_type == StepType.GITHUB:
                     ai_logger.info(f"[_handle_step_execution] Received tool_success for plan step {current_step.step_id}: {event_data.get('tool_name')}")
                     tool_name = event_data.get('tool_name', 'UnknownTool')
                     tool_args = event_data.get('tool_args', {})
                     tool_result_content = event_data.get('tool_result')
                     try:
                         pydantic_ai_tool_call_id = event_data['tool_call_id']
                     except KeyError:
                         ai_logger.error(f"CRITICAL: tool_success event missing 'tool_call_id' for step {current_step.step_id}. Event: {event_data}")
                         continue # Skip this malformed event
                         
                     github_cell_content = f"GitHub: {tool_name}"
                     github_cell_metadata = {
                         "session_id": session_id,
                         "original_plan_step_id": current_step.step_id, 
                         "tool_name": tool_name,
                         "tool_args": tool_args,
                         "pydantic_ai_tool_call_id": pydantic_ai_tool_call_id,
                     }
                     github_cell_params = CreateCellParams(
                         notebook_id=notebook_id_str,
                         cell_type=CellType.GITHUB,
                         content=github_cell_content,
                         metadata=github_cell_metadata
                     )
                     
                     # Call helper to create cell and record
                     individual_cell_id, individual_record = await self._create_cell_and_record(
                         cell_params=github_cell_params,
                         tool_call_id=pydantic_ai_tool_call_id,
                         tool_name=tool_name,
                         tool_args=tool_args,
                         step_start_time=step_start_time,
                         notebook=notebook,
                         step_id_to_record_id=step_id_to_record_id,
                         plan_step_id=current_step.step_id,
                         cell_tools=cell_tools # Pass cell_tools
                     )

                     if individual_cell_id and individual_record:
                         # Update the record status and result
                         individual_record.status = "success"
                         individual_record.completed_at = datetime.now(timezone.utc)
                         individual_record.result = tool_result_content
                         individual_record.error = None
                         # Note: Propagation is handled by the main investigate loop if needed
                         
                         # Yield event to notify frontend
                         yield {
                             "type": "github_tool_cell_created",
                             "status": "success",
                             "original_plan_step_id": current_step.step_id,
                             "cell_id": individual_cell_id,
                             "tool_call_record_id": str(individual_record.id),
                             "tool_name": tool_name,
                             "tool_args": tool_args,
                             "result": tool_result_content,
                             "agent_type": "github_agent",
                             "cell_params": github_cell_params.model_dump()
                         }
                     else:
                         # Error during cell/record creation
                         error_msg = f"Failed creating cell/record for GitHub tool {tool_name} (plan step {current_step.step_id})"
                         ai_logger.error(error_msg)
                         github_step_has_tool_errors = True
                         step_error = step_error or error_msg
                         yield {
                             "type": "github_tool_error",
                             "status": "error",
                             "original_plan_step_id": current_step.step_id,
                             "tool_name": tool_name,
                             "tool_args": tool_args,
                             "error": error_msg,
                             "agent_type": "github_agent"
                         }

                # 3. Handle individual GitHub tool error events
                elif event_type == "tool_error" and current_step.step_type == StepType.GITHUB:
                     ai_logger.error(f"[_handle_step_execution] Received tool_error for plan step {current_step.step_id}: {event_data.get('tool_name')} - {event_data.get('error')}")
                     github_step_has_tool_errors = True
                     step_error = step_error or event_data.get('error')
                     yield {
                         "type": "github_tool_error", 
                         "status": "error",
                         "original_plan_step_id": current_step.step_id,
                         "tool_name": event_data.get("tool_name"),
                         "tool_args": event_data.get("tool_args"),
                         "error": event_data.get("error"),
                         "agent_type": "github_agent"
                     }

                # 4. Handle GitHub step completion marker
                elif event_type == "github_step_processing_complete":
                     ai_logger.info(f"[_handle_step_execution] Received github_step_processing_complete for plan step {current_step.step_id}")
                     break # Stop processing for this step
                     
                # 5. Forward other status updates
                elif event_type == "status_update":
                     if 'step_id' not in event_data:
                         event_data['step_id'] = current_step.step_id
                     yield event_data 

                # 6. Handle other unexpected dict events
                else:
                     ai_logger.warning(f"[_handle_step_execution] Unexpected dict event type '{event_type}' while processing step {current_step.step_id}. Data: {event_data}")
            else:
                 ai_logger.error(f"[_handle_step_execution] FATAL: generate_content yielded non-dict type {type(event_data)} for step {current_step.step_id}")
                 step_error = step_error or "Internal error: Invalid generator output"
                 break 
        
        # Yield final completion event instead of returning
        yield {
            "type": "step_execution_complete",
            "step_id": current_step.step_id,
            "final_result_data": final_result_data, # Will be None for GitHub steps
            "step_error": step_error,
            "github_step_has_tool_errors": github_step_has_tool_errors
        }

    async def investigate(
        self, 
        query: str, 
        session_id: str, 
        notebook_id: Optional[str] = None,
        message_history: List[ModelMessage] = [],
        cell_tools: Optional[NotebookCellTools] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
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
            Dictionaries with status updates
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
            yield {"type": "plan_created", "status": "plan_created", "thinking": plan.thinking, "agent_type": "investigation_planner"}
            
            step_id_to_record_id: Dict[str, UUID] = {}
            executed_steps: Dict[str, Any] = {}
            step_results: Dict[str, Any] = {}

            # --- Plan Explanation Cell --- 
            plan_cell_params = CreateCellParams(
                notebook_id=notebook_id_str,
                cell_type="markdown",
                content=f"# Investigation Plan\\n\\n## Steps:\\n" +
                    "\\n".join(f"- `{step.step_id}`: {step.step_type.value}" for step in plan.steps),
                metadata={
                    "session_id": session_id,
                    "step_id": "plan",
                    "dependencies": []
                }
            )
            plan_cell_result = await cell_tools.create_cell(params=plan_cell_params)
            yield {
                "type": "plan_cell_created", 
                "status": "plan_cell_created",
                "agent_type": "investigation_planner",
                "cell_params": plan_cell_params.model_dump(),
                "cell_id": plan_cell_result.get("cell_id", "")
            }

            current_hypothesis = plan.hypothesis
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
                         # MODIFIED: Yield single dict
                         yield {"type": "error", "status": "error", "message": "Plan execution stalled unexpectedly.", "agent_type": "investigation_planner"}
                     elif not remaining_steps:
                         # Plan is actually complete
                         ai_logger.info("Plan execution complete.")
                     else:
                        # Normal case: waiting for dependencies
                        ai_logger.info(f"Plan execution waiting for dependencies. Remaining steps: {[s.step_id for s in remaining_steps]}")
                        # Optional: Yield a status indicating waiting state
                     break # Exit the loop if no steps are immediately executable
                    
                current_step = executable_steps[0]
                remaining_steps.remove(current_step)
                
                # Determine agent type string based on step type
                agent_type_str = "github_agent" if current_step.step_type == StepType.GITHUB else current_step.step_type.value

                yield {
                    "type": "step_started", 
                    "status": "step_started", 
                    "step_id": current_step.step_id, 
                    "agent_type": agent_type_str # Use the determined string
                }
                step_start_time = datetime.now(timezone.utc)

                # --- Call helper to execute step and handle its events --- 
                async for execution_event in self._handle_step_execution(
                     current_step=current_step,
                     executed_steps=executed_steps,
                     step_results=step_results,
                     cell_tools=cell_tools, # Pass cell_tools
                     notebook=notebook,
                     step_id_to_record_id=step_id_to_record_id,
                     session_id=session_id,
                     step_start_time=step_start_time
                 ):
                    # Forward events yielded by the helper (status updates, github cell/error events)
                    yield execution_event
                    
                    # --- Process the final completion event from the helper --- 
                    if execution_event.get("type") == "step_execution_complete":
                        final_result_data = execution_event.get("final_result_data")
                        step_error = execution_event.get("step_error")
                        github_step_has_tool_errors = execution_event.get("github_step_has_tool_errors", False)
                        
                        # --- Process NON-GitHub Step Result (Now happens *after* execution) --- 
                        created_cell_id = None
                        synthetic_record = None
                        cell_params_for_step = None # Initialize
                        
                        if current_step.step_type != StepType.GITHUB:
                            if final_result_data:
                                # Prepare cell params
                                query_content = final_result_data.query
                                if current_step.step_type == StepType.MARKDOWN:
                                    query_content = str(final_result_data.data)
                                cell_metadata_multi = {
                                    "session_id": session_id,
                                    "step_id": current_step.step_id,
                                    "dependencies": current_step.dependencies
                                }
                                cell_params_for_step = CreateCellParams(
                                    notebook_id=notebook_id_str,
                                    cell_type=current_step.step_type,
                                    content=query_content,
                                    metadata=cell_metadata_multi
                                )
                                
                                # Create cell and record using helper
                                created_cell_id, synthetic_record = await self._create_cell_and_record(
                                    cell_params=cell_params_for_step,
                                    tool_call_id=current_step.step_id, # Use plan step ID
                                    tool_name=f"step_{current_step.step_type.value}",
                                    tool_args=current_step.parameters,
                                    step_start_time=step_start_time,
                                    notebook=notebook,
                                    step_id_to_record_id=step_id_to_record_id,
                                    plan_step_id=current_step.step_id,
                                    cell_tools=cell_tools
                                )
                                
                                if not created_cell_id or not synthetic_record:
                                     # Handle creation failure
                                     step_error = step_error or f"Failed to create cell/record for step {current_step.step_id}"
                                     if final_result_data:
                                         final_result_data.error = step_error 
                                     else:
                                         final_result_data = QueryResult(query=current_step.description, data=None, error=step_error)
                                     
                            elif not final_result_data: # Handle missing final result
                                step_error = step_error or f"Failed to get final result data for step {current_step.step_id}."
                                ai_logger.error(f"final_result_data is None for step {current_step.step_id} after execution. Error: {step_error}")
                                final_result_data = QueryResult(query=current_step.description, data=None, error=step_error)
                        
                        # --- Update Execution State for the PLAN step --- 
                        plan_step_status = "success"
                        if current_step.step_type == StepType.GITHUB:
                            plan_step_status = "error" if github_step_has_tool_errors else "success"
                            executed_steps[current_step.step_id] = {
                                 "step": current_step.model_dump(),
                                 "content": f"Completed (See individual tool cells generated for this step)", 
                                 "error": step_error 
                            }
                            step_results[current_step.step_id] = None 
                        elif final_result_data: 
                            plan_step_status = "error" if final_result_data.error else "success"
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": getattr(final_result_data, 'data', None),
                                "error": step_error or getattr(final_result_data, 'error', None)
                            }
                            step_results[current_step.step_id] = final_result_data
                        else:
                            plan_step_status = "error"
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": None,
                                "error": step_error or f"Missing final_result_data for step {current_step.step_id}"
                            }
                        
                        # Update the synthetic record for NON-GitHub steps
                        if synthetic_record: # Check if record was created
                            synthetic_record.status = plan_step_status 
                            synthetic_record.completed_at = datetime.now(timezone.utc)
                            synthetic_record.result = getattr(final_result_data, 'data', None) 
                            synthetic_record.error = step_error or getattr(final_result_data, 'error', None)
                            
                            if synthetic_record.status == "success" and final_result_data is not None:
                                synthetic_record.named_outputs["result"] = getattr(final_result_data, 'data', None)
                                if hasattr(final_result_data, 'query'):
                                     synthetic_record.named_outputs["_query"] = getattr(final_result_data, 'query', None)
                                if hasattr(final_result_data, 'metadata'):
                                     synthetic_record.named_outputs["_metadata"] = getattr(final_result_data, 'metadata', None)
                            
                            ai_logger.info(f"[Agent] Propagating results for record {synthetic_record.id} (step {current_step.step_id})")
                            propagate_tool_results(synthetic_record, notebook, step_id_to_record_id)
                            ai_logger.info(f"[Agent] Results propagated for record {synthetic_record.id}")

                        # --- Yield Completion/Error Status for the PLAN STEP --- 
                        plan_step_yield_type = "step_completed" if plan_step_status == "success" else "step_error"
                        if current_step.step_type != StepType.GITHUB:
                            yield {
                                "type": plan_step_yield_type, 
                                "status": plan_step_status, 
                                "step_id": current_step.step_id,
                                "step_type": current_step.step_type.value,
                                "cell_id": str(created_cell_id) if created_cell_id else "", 
                                "tool_call_record_id": str(synthetic_record.id) if synthetic_record else "", 
                                "cell_params": cell_params_for_step.model_dump() if cell_params_for_step else {}, 
                                "result": getattr(final_result_data, 'model_dump', lambda: None)() if final_result_data else None,
                                "agent_type": current_step.step_type.value
                            }
                        ai_logger.info(f"Processed plan step: {current_step.step_id} with final status: {plan_step_status}")
                        
                        # Break from the inner loop after handling completion
                        break 

            ai_logger.info(f"Investigation plan execution finished for notebook {notebook_id_str}")

            # --- Final Summarization Step --- 
            ai_logger.info("Starting final summarization step.")
            yield {"type": "summary_started", "status": "summary_started", "agent_type": "summarization_generator"}
            
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
                    
                    # --- REVERTED GitHub result handling --- 
                    # Use the generic content/error status for GitHub plan steps
                    result_str = f"Data: {str(content)[:500]}..." if content else "No data/cells returned."
                    if error:
                        result_str = f"Error: {error}"
                        
                    # Add note if GitHub step indicates individual cells were generated
                    if type_val == StepType.GITHUB.value and content and "individual tool cells" in str(content):
                        result_str += " (See individual GitHub cells above for details)"

                    findings_text_lines.append(f"Step {step.step_id} ({type_val}): {desc}\\nResult:\\n{result_str}")

            findings_text = "\\n---\\n".join(findings_text_lines)
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
                    if isinstance(result_part, SummarizationQueryResult):
                        summary_result = result_part
                        summary_content = summary_result.data
                        summary_error = summary_result.error
                        # MODIFIED: Yield single dict
                        yield {"type": "summary_update", "status": "summary_update", "update_info": {"message": "Summary generated"}, "agent_type": "summarization_generator"}
                        break # Got the final summary object
                    elif isinstance(result_part, dict) and "status" in result_part:
                        # MODIFIED: Yield single dict
                        yield {"type": "summary_update", "status": "summary_update", "update_info": result_part, "agent_type": "summarization_generator"}
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

            summary_cell_params = CreateCellParams(
                notebook_id=notebook_id_str,
                cell_type="markdown",
                content=summary_content, 
                metadata={
                    "session_id": session_id,
                    "step_id": "final_summary",
                    "dependencies": list(executed_steps.keys()) # Depend on all executed steps
                }
            )
            try:
                summary_cell_result = await cell_tools.create_cell(params=summary_cell_params)
                summary_cell_id = summary_cell_result.get("cell_id")
                # MODIFIED: Yield single dict
                yield {
                    "type": "summary_cell_created", # Use type matching status
                    "status": "summary_cell_created",
                    "agent_type": "summarization_generator",
                    "cell_params": summary_cell_params.model_dump(),
                    "cell_id": summary_cell_id or "",
                    "error": summary_error # Include any summary generation error here
                }
                ai_logger.info(f"Final summary cell created ({summary_cell_id}) for notebook {notebook_id_str}")
            except Exception as cell_err:
                ai_logger.error(f"Failed to create final summary cell: {cell_err}", exc_info=True)
                # MODIFIED: Yield single dict
                yield {
                    "type": "summary_cell_error", # Use type matching status
                    "status": "summary_cell_error", 
                    "agent_type": "summarization_generator",
                    "error": str(cell_err)
                }

            # --- Investigation Complete ---
            # Now yield the final completion status after attempting summarization
            # MODIFIED: Yield single dict
            yield {"type": "investigation_complete", "status": "complete", "agent_type": "investigation_planner"}

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
        notebook: Optional[Notebook] = None,
        step_id_to_record_id: Optional[Dict[str, UUID]] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate content for a specific step type.

        Args:
            current_step: The current step in the investigation plan
            step_type: Type of step (sql, python, markdown, etc.)
            description: Description of what the step will do
            executed_steps: Previously executed steps (for context)
            step_results: Results from previously executed steps (for context)
            cell_tools: Tools for creating and managing cells
            notebook: The notebook object
            step_id_to_record_id: Mapping from step_id to synthetic ToolCallRecord ID
            session_id: The chat session ID

        Yields:
            Dictionaries with status updates, and finally the QueryResult.
        """

        context = ""
        if executed_steps:
            # Simplify context - passing full dicts can be very large and may hit token limits
            # Pass only the descriptions and outcome status/error of direct dependencies
            dependency_context = []
            for dep_id in current_step.dependencies:
                if dep_id in executed_steps:
                    dep_step_info = executed_steps[dep_id]
                    dep_desc = dep_step_info.get('step', {}).get('description', '')
                    dep_error = dep_step_info.get('error')
                    status = "Error" if dep_error else "Success"
                    context_line = f"- Dependency {dep_id} ({status}): {dep_desc[:100]}..." # Truncate description
                    if dep_error:
                        context_line += f" (Error: {str(dep_error)[:100]}...)" # Truncate error
                    dependency_context.append(context_line)
            if dependency_context:
                 context = f"\n\nContext from Dependencies:\n" + "\n".join(dependency_context)

        # --- Prepare the description for the sub-agent ---
        # Start with the original step description
        agent_prompt = description

        # Append parameters clearly if they exist
        if current_step.parameters:
            params_str = "\n".join([f"- {k}: {v}" for k, v in current_step.parameters.items()])
            agent_prompt += f"\n\nParameters for this step:\n{params_str}"

        # Append the dependency context
        agent_prompt += context

        # Assign to full_description for use by agents
        full_description = agent_prompt
        ai_logger.info(f"[generate_content] Passing to sub-agent ({step_type.value}): {full_description}")
        # -------------------------------------------------

        final_result: Optional[QueryResult] = None

        if step_type == StepType.GITHUB:
            # --- Check for required params --- 
            if not cell_tools or not notebook or step_id_to_record_id is None or not session_id:
                 error_msg = "Internal Error: Missing required parameters (cell_tools, notebook, map, session_id) for GitHub step processing."
                 ai_logger.error(error_msg)
                 yield {"type": "github_tool_error", "status": "error", "error": error_msg, "agent_type": "github_agent"} # Use new error type
                 return
            
            # --- Add assertions to help type checker --- 
            assert cell_tools is not None
            assert notebook is not None
            assert step_id_to_record_id is not None
            assert session_id is not None
            # --- End assertions ---

        try:
            if step_type == StepType.LOG:
                # Lazy initialization
                if self.log_generator is None:
                    ai_logger.info("Lazily initializing LogQueryAgent.")
                    self.log_generator = LogQueryAgent(source="loki", notebook_id=self.notebook_id)
                
                async for result_part in self.log_generator.run_query(full_description):
                    # MODIFIED: Yield dicts or wrap QueryResult
                    if isinstance(result_part, LogQueryResult):
                        final_result = result_part
                        # Wrap final result in a dict
                        yield {"type": "final_step_result", "result": final_result}
                        break # Stop after final result
                    elif isinstance(result_part, dict):
                         yield result_part # Forward status dicts
                    else:
                         ai_logger.warning(f"Unexpected item type {type(result_part)} yielded from log_generator")
            elif step_type == StepType.METRIC:
                # Lazy initialization
                if self.metric_generator is None:
                    ai_logger.info("Lazily initializing MetricQueryAgent.")
                    self.metric_generator = MetricQueryAgent(source="prometheus", notebook_id=self.notebook_id)
                    
                async for result_part in self.metric_generator.run_query(full_description):
                    if isinstance(result_part, MetricQueryResult):
                        final_result = result_part
                        # Wrap final result in a dict
                        yield {"type": "final_step_result", "result": final_result}
                        break # Stop after final result
                    elif isinstance(result_part, dict):
                         yield result_part # Forward status dicts
                    else:
                         ai_logger.warning(f"Unexpected item type {type(result_part)} yielded from metric_generator")
            elif step_type == StepType.MARKDOWN:
                 # Markdown generator yields status dicts, then we get final result
                 # Kept eagerly initialized for now
                 yield {"type": "status_update", "status": "generating_markdown", "agent": "markdown_generator"}
                 result = await self.markdown_generator.run(full_description)
                 if isinstance(result.output, MarkdownQueryResult):
                     final_result = result.output
                 else:
                     ai_logger.warning(f"Markdown generator did not return MarkdownQueryResult in output. Got: {type(result.output)}. Falling back.")
                     fallback_content = str(result.output) if result.output else ''
                     final_result = MarkdownQueryResult(query=description, data=fallback_content)
                 # MODIFIED: Wrap final result in a dict
                 yield {"type": "final_step_result", "result": final_result}
            elif step_type == StepType.GITHUB:
                # Lazy initialization
                if self.github_generator is None:
                    ai_logger.info("Lazily initializing GitHubQueryAgent.")
                    self.github_generator = GitHubQueryAgent(notebook_id=self.notebook_id)
                    
                # --- MODIFIED: Stream individual GitHub tool events upward --- 
                ai_logger.info(f"Processing GitHub step {current_step.step_id} using run_query generator...")
                
                async for event_data in self.github_generator.run_query(full_description):
                    # Check if event_data is a dictionary before accessing keys
                    if isinstance(event_data, dict):
                        event_type = event_data.get("type")
                        
                        # Forward relevant events (status updates, tool results/errors)
                        if event_type in ["status_update", "tool_success", "tool_error"]:
                            ai_logger.debug(f"Yielding event from GitHub agent: {event_type} for step {current_step.step_id}")
                            # Add original plan step ID for context
                            event_data['original_plan_step_id'] = current_step.step_id 
                            yield event_data
                        
                        # Log final status but don't yield it here (investigate loop handles step completion)
                        elif event_type == "final_status":
                            ai_logger.info(f"GitHub query finished for step {current_step.step_id} with status: {event_data.get('status')}")
                        
                        # Log other unexpected dictionary events
                        else:
                            ai_logger.warning(f"Unexpected dict event type '{event_type}' received during GitHub step {current_step.step_id}. Data: {event_data}")
                    else:
                        # Log non-dictionary events
                        ai_logger.warning(f"Received non-dict event from GitHub agent during step {current_step.step_id}: {event_data}")

                # After the github_generator finishes, yield a specific marker
                # The investigate loop will use this to mark the *original plan step* as processed.
                yield {"type": "github_step_processing_complete", "original_plan_step_id": current_step.step_id, "agent_type": "github_agent"}
                # --- END MODIFIED GitHub logic --- 
            
            else:
                ai_logger.error(f"Unsupported step type encountered in generate_content: {step_type}")

        except Exception as e:
            ai_logger.error(f"Error during generate_content for step type {step_type}: {e}", exc_info=True)
            # Yield an error status update (already a dict)
            yield {"type": "error", "status": "error", "step_type": step_type.value, "error": str(e)}