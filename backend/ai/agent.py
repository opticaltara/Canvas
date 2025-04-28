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
from pydantic_ai.messages import ModelMessage

from backend.ai.log_query_agent import LogQueryAgent
from backend.ai.metric_query_agent import MetricQueryAgent
from backend.ai.github_query_agent import GitHubQueryAgent
from backend.ai.summarization_agent import SummarizationAgent
from backend.config import get_settings
from backend.core.execution import ExecutionContext, ToolCallRecord, register_tool_dependencies, propagate_tool_results
from backend.core.query_result import (
    QueryResult, 
    LogQueryResult, 
    MetricQueryResult, 
    MarkdownQueryResult, 
    GithubQueryResult,
    SummarizationQueryResult
)
from backend.services.connection_manager import get_connection_manager
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams
from backend.core.dependency import ToolOutputReference
from backend.core.notebook import Notebook
from backend.services.notebook_manager import get_notebook_manager

import asyncio # Add this import at the top of the file

# Get logger for AI operations
ai_logger = logging.getLogger("ai")

# Get settings for API keys
settings = get_settings()


class StepType(str, Enum):
    MARKDOWN = "markdown"
    LOG = "log"
    METRIC = "metric"
    GITHUB = "github"
    SUMMARIZATION = "summarization"


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
        
        # Create the full list of MCPServerHTTP for general agents
        all_mcps_list = list(self.mcp_server_map.values())
        ai_logger.info(f"Initializing general agents with {len(all_mcps_list)} HTTP MCP servers.")
        
        # Initialize investigation planner
        self.investigation_planner = Agent(
            self.model,
            deps_type=InvestigationDependencies,
            result_type=InvestigationPlanModel,
            system_prompt="""
            You are a Senior Software Engineer and the Lead Investigator. Your purpose is to 
            coordinate a team of specialized agents by creating and adapting investigation plans.
            
            You will be given a user query and a list of available data sources.
            You will need to create a plan for an investigation team to address the query.
            Sometimes the query will be a simple ask such as "What is my most starred github repo?"
            If this question, can be answered with a simple tool call, you should do that.
            Otherwise, you should create a plan that uses the github agent to answer the query.

            When analyzing a user query about a software incident:

            **Complex Query Handling (Multi-Step Plan):**
            If the query requires multiple steps or analysis across different data sources:

            1. CONTEXT ASSESSMENT
              • Extract the key incident characteristics (affected services, error patterns, timing)
              • Consider the available data sources provided in the investigation dependencies (you have access to: {', '.join(self.available_data_sources) if self.available_data_sources else 'None'})
              • Frame the investigation as a structured debugging process

            2. INVESTIGATION DESIGN
              Produce **1–5 PHASES**, each a coarse investigative goal (e.g. "Identify error fingerprint",
              "Correlate spikes with deploys").  Do **NOT** emit fine‑grained LogQL/PromQL/code;
              micro‑agents will handle that.

            3. STEP SPECIFICATION
              For each step in your plan, define:
              
              • step_id: A unique identifier (use S1, S2, etc.)
              • step_type: Choose the *primary* data source this phase will use ("log", "metric", "github", "summarization", or "markdown" for analysis/decision steps)
              • category: Choose the *primary* category for this step ("PHASE" or "DECISION"). Always "phase" here.
              • description: Instructions for the specialized agent that will:
                - State precisely what question this step answers
                - Provide all context needed for the specialized agent (including relevant prior results)
                - Explain how to interpret the results
                - Reference specific artifacts from previous steps when needed
                - **If the step_type is 'summarization', ensure the description clearly indicates what text needs summarizing (e.g., referencing results from a previous step).**
              • dependencies: Array of step IDs required before this step can execute
              • parameters: Configuration details relevant to this step type. 
                - For "github" type, should contain 'connection_id' (string) referencing the relevant GitHub connection.
                - **For "summarization", parameters are generally not needed unless specifying constraints (e.g., max length), but the text to summarize should come from the description or dependencies.**
                - **REFERENCING OUTPUTS:** If a parameter needs to use the output of a previous step (listed in `dependencies`), use the following structure within the `parameters` dict: `"<parameter_name>": {"__ref__": {"step_id": "<ID_of_dependency_step>", "output_name": "result"}}`. For now, always use `"result"` as the `output_name`. For example, if step S2 needs the result of S1 as its 'input_data' parameter, its parameters might look like: `{"input_data": {"__ref__": {"step_id": "S1", "output_name": "result"}}}`.
              • is_decision_point: Set to true for markdown steps that evaluate previous results
              
            4. DECISION POINTS
              Include explicit markdown steps that will:
              
              • Evaluate results from previous steps
              • Determine if hypothesis needs revision
              • Decide whether to continue with planned steps or pivot
              • Document the reasoning for continuing or changing direction

            5. COMPLETION CRITERIA
              Define specific technical indicators that will confirm:
              
              • Root cause has been identified
              • Impact has been quantified
              • Contributing factors are understood
              • Potential remediation approaches

            IMPORTANT CONSTRAINTS (for multi-step plans):
            • Keep the investigation plan concise. Aim for the minimum number of steps required to address the user's core query. 
            Do not generate more than 5 steps unless absolutely necessary for a complex investigation.
            • Stick strictly to the scope of the user's request. Do not add steps for tangential inquiries 
            or explorations not explicitly requested.
            • Keep the `thinking` explanation brief (2-3 sentences maximum). Focus on the core strategy and rationale, omitting excessive detail.

            Remember that you are creating instructions for specialized agents, not executing the investigation yourself.
            Your instructions must be detailed and self-contained, as each specialized agent only sees its specific task.
            """
        )


        # Initialize plan reviser
        self.plan_reviser = Agent(
            self.model,
            deps_type=PlanRevisionRequest,
            result_type=PlanRevisionResult,
            system_prompt="""
            You are an Investigation Plan Reviser responsible for analyzing the results of executed steps and determining if the investigation plan should be adjusted.

            When reviewing executed steps and their results:

            1. ANALYZE EXECUTED STEPS
              • Review the data and insights gained from steps executed so far
              • Compare actual results against expected outputs
              • Identify any unexpected patterns or anomalies
              • Evaluate how the results support or contradict the current hypothesis

            2. REVISION DECISION
              Decide whether to:
              • Continue with the existing plan (if results align with expectations)
              • Modify the plan by adding new steps or removing planned steps
              • Update the working hypothesis based on new evidence
              
            3. NEW STEP SPECIFICATION
              If adding new steps, define each one with:
              • step_id: A unique identifier not conflicting with existing steps
              • step_type: The appropriate type for this step
              • description: Detailed instructions for the specialized agent
              • dependencies: Steps that must complete before this one
              • parameters: Configuration for this step
              • is_decision_point: Whether this is another evaluation step

            4. EXPLANATION
              Provide clear reasoning for your decision, including:
              • How executed results influenced your decision
              • Why the current plan is sufficient or needs changes
              • How any new steps will address gaps in the investigation
              • How updated hypothesis better explains the observed behavior

            Your role is critical for adaptive investigation - don't hesitate to recommend significant changes if the evidence warrants it, but also maintain investigation focus and avoid unnecessary steps.
            """
        )
        
        self.markdown_generator = Agent(
            self.model,
            result_type=MarkdownQueryResult,
            mcp_servers=all_mcps_list,
            system_prompt="""
            You are an expert at technical documentation and result analysis. Create clear and **concise** markdown to address the user's request. 
            Your primary goal is brevity and clarity. Avoid unnecessary jargon or overly detailed explanations.
            
            When analyzing investigation results:
            1. Summarize key findings **briefly** and objectively
            2. Identify patterns and anomalies in the data
            3. Draw connections between different data sources
            4. Evaluate how findings support or contradict hypotheses
            5. Recommend next steps based on the evidence
            
            Focus on being succinct. Return ONLY the markdown with no meta-commentary.
            """
        )

        # Initialize specialized agents without passing mcp_servers
        self.log_generator = LogQueryAgent(source="loki", notebook_id=self.notebook_id) 
        self.metric_generator = MetricQueryAgent(source="prometheus", notebook_id=self.notebook_id)
        self.github_generator = GitHubQueryAgent(notebook_id=self.notebook_id)
        self.summarization_generator = SummarizationAgent(notebook_id=self.notebook_id)

        ai_logger.info(f"AIAgent initialized for notebook {notebook_id}.")

    async def investigate(
        self, 
        query: str, 
        session_id: str, 
        notebook_id: Optional[str] = None,
        message_history: List[ModelMessage] = [],
        cell_tools: Optional[NotebookCellTools] = None
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
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
            Tuples of (step_id, status) as steps are completed
        """
        if not cell_tools:
            raise ValueError("cell_tools is required for creating cells")
            
        notebook_id_str = notebook_id or self.notebook_id
        if not notebook_id_str:
            raise ValueError("notebook_id is required for creating cells")
        
        # Fetch the actual Notebook object - needed for dependency graph and records
        notebook_manager = get_notebook_manager() 
        try:
            notebook_uuid = UUID(notebook_id_str)
            notebook: Notebook = notebook_manager.get_notebook(notebook_uuid) 
        except ValueError:
            ai_logger.error(f"Invalid notebook_id format: {notebook_id_str}")
            raise ValueError(f"Invalid notebook_id format: {notebook_id_str}")
        except Exception as e:
             ai_logger.error(f"Failed to fetch notebook {notebook_id_str}: {e}", exc_info=True)
             raise ValueError(f"Failed to fetch notebook {notebook_id_str}")

            
        plan = await self.create_investigation_plan(query, notebook_id_str, message_history)
        yield "plan_created", {"status": "plan_created", "thinking": plan.thinking, "agent_type": "investigation_planner"}
        
        # Store mapping from step_id to synthetic ToolCallRecord ID for propagation
        step_id_to_record_id: Dict[str, UUID] = {}

        if len(plan.steps) == 1:
            ai_logger.info(f"Detected single-step plan for query: {query}")
            single_step = plan.steps[0]
            
            if single_step.step_type == StepType.GITHUB:
                 yield "status_update", {"message": "GitHub Agent: Starting query...", "agent_type": "github"}

            # --- Handle Async Generator for single step ---
            final_result_data: Optional[QueryResult] = None
            all_yielded_items = [] # Store all yielded items for potential debugging/logging

            async for result_part in self.generate_content(
                current_step=single_step,
                step_type=single_step.step_type,
                description=single_step.description,
                executed_steps={},
                step_results={},
            ):
                all_yielded_items.append(result_part)
                if isinstance(result_part, QueryResult):
                    final_result_data = result_part
                    # Assuming the QueryResult is the last item yielded
                    break # Exit loop once we have the final result
                elif isinstance(result_part, dict) and "status" in result_part:
                     # Forward status updates from generate_content
                     yield f"step_{single_step.step_id}_update", {
                         "status": "step_update",
                         "step_id": single_step.step_id,
                         "update_info": result_part,
                         "agent_type": result_part.get("agent", single_step.step_type.value) # Get agent from update if available
                     }
                else:
                    # Log unexpected yield type
                    ai_logger.warning(f"Unexpected item yielded from generate_content for single step {single_step.step_id}: {type(result_part)}")


            if not final_result_data:
                # Handle case where generate_content didn't yield a QueryResult
                ai_logger.error(f"generate_content did not yield a final QueryResult for single step {single_step.step_id}. Yielded items: {all_yielded_items}")
                yield f"step_{single_step.step_id}_error", {
                    "status": "error",
                    "step_id": single_step.step_id,
                    "message": "Failed to get final result from content generation.",
                    "agent_type": single_step.step_type.value # Use enum value
                }
                return # Stop processing this step

            query_content = final_result_data.query
            if single_step.step_type == StepType.MARKDOWN:
                query_content = str(final_result_data.data) 

            # --- Create Cell ---
            cell_params_for_step = CreateCellParams(
                notebook_id=notebook_id_str,
                cell_type=single_step.step_type,
                content=query_content, 
                metadata={
                    "session_id": session_id,
                    "step_id": single_step.step_id,
                    "dependencies": single_step.dependencies # Keep original step dependencies here for info
                }
            )
            # --- Add tool info to cell params if it's a GitHub cell ---
            tool_info_kwargs = {}
            if single_step.step_type == StepType.GITHUB and isinstance(final_result_data, GithubQueryResult) and final_result_data.tool_calls:
                last_tool_call = final_result_data.tool_calls[-1] # Get the last successful call
                tool_info_kwargs['tool_name'] = last_tool_call.tool_name
                tool_info_kwargs['tool_arguments'] = last_tool_call.tool_args
            
            cell_result = await cell_tools.create_cell(
                params=cell_params_for_step,
                # Pass tool info as additional kwargs if available
                **tool_info_kwargs 
            )
            created_cell_id = cell_result.get("cell_id")
            if not created_cell_id:
                ai_logger.error(f"Failed to get cell_id after creating cell for step {single_step.step_id}")
                # Handle error appropriately, maybe yield error status
                yield f"step_{single_step.step_id}_error", {"status": "error", "message": "Failed to create notebook cell.", "step_id": single_step.step_id, "agent_type": single_step.step_type.value}
                return

            # --- Create Synthetic ToolCallRecord ---
            # ASSUMPTION: Planner puts ToolOutputReference in parameters if needed
            synthetic_record = ToolCallRecord(
                parent_cell_id=UUID(created_cell_id),
                pydantic_ai_tool_call_id=single_step.step_id, # Link loosely via step_id
                tool_name=f"step_{single_step.step_type.value}",
                parameters=single_step.parameters, # Assumes planner might add ToolOutputReferences here
                status="pending", # Will be updated shortly
                started_at=datetime.now(timezone.utc) # Approximate start
            )
            step_id_to_record_id[single_step.step_id] = synthetic_record.id
            notebook.tool_call_records[synthetic_record.id] = synthetic_record
            # Update cell to link to this record
            # TODO: Need a way to update cell.tool_call_ids, potentially via notebook_manager/cell_tools
            if created_cell_id:
                cell_uuid = UUID(created_cell_id)
                if cell_uuid in notebook.cells:
                    notebook.cells[cell_uuid].tool_call_ids.append(synthetic_record.id)
                else:
                    ai_logger.warning(f"Could not find cell {created_cell_id} in notebook to link tool record {synthetic_record.id}")
            
            ai_logger.info(f"Added synthetic tool record {synthetic_record.id} for step {single_step.step_id}")

            # --- Register Dependencies ---
            register_tool_dependencies(synthetic_record, notebook, step_id_to_record_id)
            
            # --- Update Record with Result & Propagate ---
            synthetic_record.status = "success" if final_result_data.error is None else "error"
            synthetic_record.completed_at = datetime.now(timezone.utc)
            synthetic_record.result = final_result_data.data
            synthetic_record.error = final_result_data.error
            # Populate named_outputs (using a default name 'result' for now)
            if synthetic_record.status == "success":
                synthetic_record.named_outputs["result"] = final_result_data.data
                # Store query/metadata if needed for dependencies
                synthetic_record.named_outputs["_query"] = final_result_data.query
                synthetic_record.named_outputs["_metadata"] = final_result_data.metadata
                
            propagate_tool_results(synthetic_record, notebook, step_id_to_record_id)
            # Persist notebook changes (including updated records/cells)
            notebook_manager.save_notebook(notebook.id)

            # --- Yield Completion ---
            yield f"step_{single_step.step_id}_completed", {
                "status": "step_completed",
                "step_id": single_step.step_id,
                "step_type": single_step.step_type,
                "cell_id": created_cell_id,
                "tool_call_record_id": str(synthetic_record.id), # Include record ID
                "cell_params": cell_params_for_step.model_dump(),
                "result": { # Use the final_result_data
                    "data": final_result_data.data,
                    "query": final_result_data.query,
                    "error": final_result_data.error,
                    "metadata": final_result_data.metadata,
                    "tool_calls": getattr(final_result_data, 'tool_calls', None)
                },
                "agent_type": single_step.step_type.value,
                "is_single_step_plan": True 
            }
            ai_logger.info(f"Completed single-step plan execution for step: {single_step.step_id}")
            return 
        
        # --- Proceed with multi-step plan execution ---
        else:
            # Create plan explanation cell (no change here)
            # ... (existing plan cell creation code) ...
            cell_params = CreateCellParams(
                notebook_id=notebook_id_str,
                cell_type="markdown",
                content=f"# Investigation Plan\n\n## Steps:\n" +
                    "\n".join(f"- `{step.step_id}`: {step.step_type.value}" for step in plan.steps),
                metadata={
                    "session_id": session_id,
                    "step_id": "plan",
                    "dependencies": []
                }
            )
            cell_result = await cell_tools.create_cell(params=cell_params)
            yield "plan_cell_created", {
                "status": "plan_cell_created",
                "agent_type": "investigation_planner",
                "cell_params": cell_params.model_dump(),
                "cell_id": cell_result.get("cell_id", "")
            }


        executed_steps = {}
        step_results = {}
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
                     yield "error", {"status": "error", "message": "Plan execution stalled unexpectedly.", "agent_type": "investigation_planner"}
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
            
            yield f"step_{current_step.step_id}_started", {"status": "step_started", "step_id": current_step.step_id, "agent_type": current_step.step_type.value}
            step_start_time = datetime.now(timezone.utc)

            # --- Handle Async Generator for multi-step ---
            final_result_data: Optional[QueryResult] = None
            all_yielded_items_multi = []
            step_error: Optional[str] = None

            try:
                if current_step.category == StepCategory.PHASE:
                    async for result_part in self.generate_content(
                        current_step=current_step,
                        step_type=current_step.step_type,
                        description=current_step.description,
                        executed_steps=executed_steps,
                        step_results=step_results,
                    ):
                        all_yielded_items_multi.append(result_part)
                        if isinstance(result_part, QueryResult):
                            final_result_data = result_part
                            break # Assume QueryResult is the final item
                        elif isinstance(result_part, dict) and "status" in result_part:
                            # Forward status updates
                            yield f"step_{current_step.step_id}_update", {
                                 "status": "step_update",
                                 "step_id": current_step.step_id,
                                 "update_info": result_part,
                                 "agent_type": result_part.get("agent", current_step.step_type.value)
                            }
                        else:
                             ai_logger.warning(f"Unexpected item yielded from generate_content for multi-step {current_step.step_id}: {type(result_part)}")

                elif current_step.category == StepCategory.DECISION:
                     # ... (existing decision logic generating final_result_data) ...
                     yield f"step_{current_step.step_id}_update", {
                         "status": "step_update",
                         "step_id": current_step.step_id,
                         "update_info": {"status": "generating_decision_markdown"},
                         "agent_type": "markdown_generator"
                     }
                     # Assuming markdown_generator.run is awaitable
                     markdown_result = await self.markdown_generator.run(
                         current_step.description
                     )
                     # Adapt based on actual return type of markdown_generator.run
                     # If it returns a structured object like other agents:
                     if markdown_result and hasattr(markdown_result, 'output'):
                         markdown_string = str(markdown_result.output) # Adjust attribute access if needed
                     else:
                         ai_logger.warning(f"Could not extract markdown data from result for decision step {current_step.step_id}: {markdown_result}")
                         markdown_string = f"Error generating decision markdown for step {current_step.step_id}."
                     final_result_data = MarkdownQueryResult(data=markdown_string, query=current_step.description)

            except Exception as e:
                step_error = f"Error during execution of step {current_step.step_id}: {e}"
                ai_logger.error(step_error, exc_info=True)
                final_result_data = QueryResult(query=current_step.description, data=None, error=step_error)


            # --- Process Step Result ---
            created_cell_id = None
            synthetic_record = None
            if final_result_data:
                query = final_result_data.query
                if current_step.step_type == StepType.MARKDOWN:
                    query = str(final_result_data.data) # Use data for markdown cell content

                # --- Create Cell ---
                cell_params_for_step = CreateCellParams(
                    notebook_id=notebook_id_str,
                    cell_type=current_step.step_type,
                    content=query,
                    metadata={
                        "session_id": session_id,
                        "step_id": current_step.step_id,
                        "dependencies": current_step.dependencies # Keep original step dependencies here for info
                    }
                )
                # --- Add tool info to cell params if it's a GitHub cell ---
                tool_info_kwargs_multi = {}
                if current_step.step_type == StepType.GITHUB and isinstance(final_result_data, GithubQueryResult) and final_result_data.tool_calls:
                    last_tool_call = final_result_data.tool_calls[-1] # Get the last successful call
                    tool_info_kwargs_multi['tool_name'] = last_tool_call.tool_name
                    tool_info_kwargs_multi['tool_arguments'] = last_tool_call.tool_args
                    
                try:
                    ai_logger.info(f"[Agent] Attempting to create cell for step {current_step.step_id} with params: {cell_params_for_step.model_dump()}")
                    cell_result = await cell_tools.create_cell(
                        params=cell_params_for_step,
                        # Pass tool info as additional kwargs if available
                        **tool_info_kwargs_multi 
                    )
                    created_cell_id = cell_result.get("cell_id")
                    ai_logger.info(f"[Agent] Cell creation result for step {current_step.step_id}: {cell_result}")
                    if not created_cell_id:
                         raise ValueError("create_cell tool did not return a cell_id")
                except Exception as cell_creation_error:
                    ai_logger.error(f"[Agent] Failed to create cell for step {current_step.step_id}: {cell_creation_error}", exc_info=True)
                    step_error = step_error or f"Failed to create notebook cell: {cell_creation_error}"
                    # Update final result data if it exists
                    if final_result_data:
                        final_result_data.error = step_error
                    else:
                        final_result_data = QueryResult(query=current_step.description, data=None, error=step_error)
                
                if created_cell_id:
                    # --- Create Synthetic ToolCallRecord ---
                    # ASSUMPTION: Planner puts ToolOutputReference in parameters if needed
                    synthetic_record = ToolCallRecord(
                        parent_cell_id=UUID(created_cell_id),
                        pydantic_ai_tool_call_id=current_step.step_id, # Link loosely via step_id
                        tool_name=f"step_{current_step.step_type.value}",
                        parameters=current_step.parameters, # Assumes planner might add ToolOutputReferences here
                        status="pending", # Will be updated shortly
                        started_at=step_start_time 
                    )
                    step_id_to_record_id[current_step.step_id] = synthetic_record.id
                    notebook.tool_call_records[synthetic_record.id] = synthetic_record
                    # Update cell to link to this record
                    # TODO: Need a way to update cell.tool_call_ids, potentially via notebook_manager/cell_tools
                    if created_cell_id:
                        cell_uuid = UUID(created_cell_id)
                        if cell_uuid in notebook.cells:
                            notebook.cells[cell_uuid].tool_call_ids.append(synthetic_record.id)
                        else:
                            ai_logger.warning(f"Could not find cell {created_cell_id} in notebook to link tool record {synthetic_record.id}")
                            
                    ai_logger.info(f"[Agent] Added synthetic tool record {synthetic_record.id} for step {current_step.step_id}")

                    # --- Register Dependencies ---
                    # Pass the step_id map for resolving references
                    ai_logger.info(f"[Agent] Registering dependencies for record {synthetic_record.id} (step {current_step.step_id})")
                    register_tool_dependencies(synthetic_record, notebook, step_id_to_record_id)
                    ai_logger.info(f"[Agent] Dependencies registered for record {synthetic_record.id}")

            else: # Handle case where final_result_data is None even after try/except
                step_error = step_error or f"Failed to get final result data for step {current_step.step_id}."
                ai_logger.error(f"final_result_data is None for step {current_step.step_id} after execution attempt. Error: {step_error}")
                final_result_data = QueryResult(query=current_step.description, data=None, error=step_error)
            
            # --- Update Execution State and Record ---
            step_status = "step_completed" if step_error is None else "error"
            executed_steps[current_step.step_id] = {
                "step": current_step.model_dump(),
                "content": final_result_data.data,
                "error": step_error
            }
            step_results[current_step.step_id] = final_result_data
            
            if synthetic_record:
                synthetic_record.status = "success" if step_error is None else "error"
                synthetic_record.completed_at = datetime.now(timezone.utc)
                synthetic_record.result = final_result_data.data
                synthetic_record.error = step_error
                # Populate named_outputs 
                if synthetic_record.status == "success":
                    synthetic_record.named_outputs["result"] = final_result_data.data
                    synthetic_record.named_outputs["_query"] = final_result_data.query
                    synthetic_record.named_outputs["_metadata"] = final_result_data.metadata
                
                # --- Propagate Results ---
                # Pass the step_id map for resolving references during propagation
                ai_logger.info(f"[Agent] Propagating results for record {synthetic_record.id} (step {current_step.step_id})")
                propagate_tool_results(synthetic_record, notebook, step_id_to_record_id)
                ai_logger.info(f"[Agent] Results propagated for record {synthetic_record.id}")

            # Persist notebook changes (including updated records/cells)
            ai_logger.info(f"[Agent] Saving notebook {notebook.id} after step {current_step.step_id}")
            notebook_manager.save_notebook(notebook.id)
            ai_logger.info(f"[Agent] Notebook {notebook.id} saved after step {current_step.step_id}")

            # --- Yield Completion/Error ---
            yield f"step_{current_step.step_id}_{'completed' if step_error is None else 'error'}", {
                "status": step_status,
                "step_id": current_step.step_id,
                "step_type": current_step.step_type,
                "cell_id": created_cell_id or "",
                "tool_call_record_id": str(synthetic_record.id) if synthetic_record else "",
                "cell_params": cell_params_for_step.model_dump() if 'cell_params_for_step' in locals() else {},
                "result": final_result_data.model_dump(), # Send full result object
                "agent_type": current_step.step_type.value
            }
            ai_logger.info(f"Processed multi-step plan step: {current_step.step_id} with status: {step_status}")

            # Plan Revision Logic (Optional - could be added here) 
            # if some condition based on final_result_data.error or content:
            #    revision_result = await self.revise_plan(...)
            #    update remaining_steps, current_hypothesis etc.

        ai_logger.info(f"Investigation plan execution finished for notebook {notebook_id_str}")
        yield "investigation_complete", {"status": "complete", "agent_type": "investigation_planner"}
        

    async def create_investigation_plan(self, query: str, notebook_id: Optional[str] = None, message_history: List[ModelMessage] = []) -> InvestigationPlanModel:
        """Create an investigation plan for the given query"""
        # Use stored notebook_id if none provided
        notebook_id = notebook_id or self.notebook_id
        if not notebook_id:
            raise ValueError("notebook_id is required for creating investigation plan")
            
        # Create dependencies for the investigation planner
        deps = InvestigationDependencies(
            user_query=query,
            notebook_id=UUID(notebook_id),
            available_data_sources=self.available_data_sources,
            executed_steps={},
            current_hypothesis=None,
            message_history=message_history
        )

        # Generate the plan
        result = await self.investigation_planner.run(
            user_prompt=deps.user_query,
            deps=deps
        )
        plan_data = result.data

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
        
        return result.data

    async def _yield_result_async(self, result):
        """Helper to wrap a non-async-generator result."""
        yield result
        await asyncio.sleep(0) # Yield control briefly

    async def generate_content(
        self,
        current_step: InvestigationStepModel,
        step_type: StepType,
        description: str,
        executed_steps: Dict[str, Any] | None = None,
        step_results: Dict[str, Any] = {},
    ) -> AsyncGenerator[Union[Dict[str, Any], QueryResult], None]:
        """
        Generate content for a specific step type, yielding status updates.

        Args:
            current_step: The current step in the investigation plan
            step_type: Type of step (sql, python, markdown, etc.)
            description: Description of what the step will do
            executed_steps: Previously executed steps (for context)
            step_results: Results from previously executed steps (for context)

        Yields:
            Dictionaries with status updates, and finally the QueryResult.
        """
        if description in step_results:
            # Ensure we return the correct type if cached
            cached_result = step_results[description]
            if isinstance(cached_result, QueryResult):
                 # Wrap the cached result in an async generator
                 async for item in self._yield_result_async(cached_result):
                     yield item
                 return # Important: exit after yielding cached result
            else:
                 # If it's not a QueryResult subclass, log a warning and potentially wrap it
                 ai_logger.warning(f"Cached result for '{description}' is not a QueryResult type: {type(cached_result)}. Attempting to use as data.")
                 # Fallback: yield a generic QueryResult - adjust if needed
                 fallback_result = QueryResult(query=description, data=cached_result)
                 async for item in self._yield_result_async(fallback_result):
                     yield item
                 return # Important: exit after yielding cached result

        context = ""
        if executed_steps and step_results:
            # TODO: Consider how large context might get. Maybe pass relevant parts only?
            context = f"\n\nContext from previous steps:\n{executed_steps}\n\nResults from previous steps:\n{step_results}"

        full_description = description + context
        final_result: Optional[QueryResult] = None

        try:
            if step_type == StepType.LOG:
                async for result_part in self.log_generator.run_query(full_description):
                    yield result_part
                    if isinstance(result_part, LogQueryResult):
                        final_result = result_part
            elif step_type == StepType.METRIC:
                async for result_part in self.metric_generator.run_query(full_description):
                    yield result_part
                    if isinstance(result_part, MetricQueryResult):
                        final_result = result_part
            elif step_type == StepType.MARKDOWN:
                 # Markdown generator's run might not be async generator
                 yield {"status": "generating_markdown", "agent": "markdown_generator"}
                 result = await self.markdown_generator.run(full_description)
                 # Ensure we get the actual MarkdownQueryResult
                 if isinstance(result, MarkdownQueryResult):
                     final_result = result
                 elif result and hasattr(result, 'data') and isinstance(result.data, MarkdownQueryResult):
                     final_result = result.data
                 else:
                     ai_logger.warning(f"Markdown generator did not return MarkdownQueryResult. Got: {type(result)}. Falling back.")
                     fallback_content = str(getattr(result, 'data', getattr(result, 'output', ''))) if result else ''
                     final_result = MarkdownQueryResult(query=description, data=fallback_content)
                 yield final_result # Yield the final result
            elif step_type == StepType.GITHUB:
                async for result_part in self.github_generator.run_query(full_description):
                    yield result_part
                    if isinstance(result_part, GithubQueryResult):
                        final_result = result_part
            elif step_type == StepType.SUMMARIZATION:
                # Extract the text to summarize. Assume it's passed within the description 
                # or implicitly from a dependency referenced in the description.
                # For now, assume `full_description` contains necessary context *and* the text.
                # A more robust implementation might parse `description` or use `step_results`.
                # Let's assume the planner puts the text directly in the description for now.
                text_to_summarize = full_description # Simplified assumption
                original_request = description # Pass the step description as the original request context
                yield {"status": "generating_summary", "agent": "summarization_generator"}
                async for result_part in self.summarization_generator.run_summarization(
                    text_to_summarize=text_to_summarize,
                    original_request=original_request
                ):
                    if isinstance(result_part, SummarizationQueryResult):
                        final_result = result_part
                        yield final_result
                    elif isinstance(result_part, dict):
                        yield result_part
            else:
                ai_logger.error(f"Unsupported step type encountered in generate_content: {step_type}")
                raise ValueError(f"Unsupported step type: {step_type}")

        except Exception as e:
            ai_logger.error(f"Error during generate_content for step type {step_type}: {e}", exc_info=True)
            # Yield an error status update
            yield {"status": "error", "step_type": step_type.value, "error": str(e)}
            # Optionally, yield a QueryResult with the error state if needed downstream
            # final_result = QueryResult(query=description, error=str(e))
            # yield final_result
            # Depending on desired error handling, you might re-raise or just yield error status

        # Ensure a final result is yielded if an error occurred mid-stream
        # and was caught but didn't yield a final QueryResult object.
        # This part might need adjustment based on how errors should propagate.
        # If an error status is yielded above, maybe we shouldn't yield a final_result?
        # Or maybe we yield a QueryResult with an error field populated.
        # For now, let's assume if final_result exists, it was the last thing yielded
        # by the sub-generator or markdown block. If it doesn't exist after the try-except
        # (and no exception was raised to exit), it implies an issue.

        if final_result is None and step_type != StepType.MARKDOWN: # Markdown handles its own final yield
             ai_logger.warning(f"No final QueryResult object obtained for step type {step_type} description '{description}'. This might indicate an issue in the sub-generator.")
             # Yield a generic error result if needed by downstream logic
             yield QueryResult(query=description, data='', error=f"Failed to generate final result for {step_type}")