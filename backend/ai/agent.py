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
from backend.config import get_settings
from backend.core.cell import AIQueryCell
from backend.core.execution import ExecutionContext
from backend.core.query_result import (
    QueryResult, 
    LogQueryResult, 
    MetricQueryResult, 
    MarkdownQueryResult, 
    GithubQueryResult
)
from backend.services.connection_manager import get_connection_manager
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams

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
              • step_type: Choose the *primary* data source this phase will use ("log", "metric", or "github")
              • category: Choose the *primary* category for this step ("PHASE" or "DECISION"). Always "phase" here.
              • description: Instructions for the specialized agent that will:
                - State precisely what question this step answers
                - Provide all context needed for the specialized agent
                - Explain how to interpret the results
                - Reference specific artifacts from previous steps when needed
              • dependencies: Array of step IDs required before this step can execute
              • parameters: Configuration details relevant to this step type. 
                - For "github" type, should contain 'connection_id' (string) referencing the relevant GitHub connection.
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
        # GitHubQueryAgent already correctly initialized
        self.github_generator = GitHubQueryAgent(notebook_id=self.notebook_id)

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
            notebook_id: The notebook ID to create cells in (optional, will use stored notebook_id if not provided)
            message_history: Previous messages in the session
            cell_tools: Tools for creating and managing cells
            
        Yields:
            Tuples of (step_id, status) as steps are completed
        """
        if not cell_tools:
            raise ValueError("cell_tools is required for creating cells")
            
        notebook_id = notebook_id or self.notebook_id
        if not notebook_id:
            raise ValueError("notebook_id is required for creating cells")
            
        plan = await self.create_investigation_plan(query, notebook_id, message_history)
        yield "plan_created", {"status": "plan_created", "thinking": plan.thinking, "agent_type": "investigation_planner"}
        
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
                    "agent_type": single_step.step_type
                }
                return # Stop processing this step

            query_content = final_result_data.query
            if single_step.step_type == StepType.MARKDOWN:
                query_content = str(final_result_data.data) 

            cell_params_for_step = CreateCellParams(
                notebook_id=notebook_id,
                cell_type=single_step.step_type,
                content=query_content, 
                metadata={
                    "session_id": session_id,
                    "step_id": single_step.step_id,
                    "dependencies": single_step.dependencies
                }
            )
            cell_result = await cell_tools.create_cell(
                params=cell_params_for_step
            )
            
            yield f"step_{single_step.step_id}_completed", {
                "status": "step_completed",
                "step_id": single_step.step_id,
                "step_type": single_step.step_type,
                "cell_id": cell_result.get("cell_id", ""),
                "cell_params": cell_params_for_step.model_dump(),
                "result": { # Use the final_result_data
                    "data": final_result_data.data,
                    "query": final_result_data.query,
                    "error": final_result_data.error,
                    "metadata": final_result_data.metadata
                },
                "agent_type": single_step.step_type,
                "is_single_step_plan": True 
            }
            ai_logger.info(f"Completed single-step plan execution for step: {single_step.step_id}")
            return 
        
        # --- Proceed with multi-step plan execution ---
        else:
            # Create plan explanation cell (no change here)
            cell_params = CreateCellParams(
                notebook_id=notebook_id,
                cell_type="markdown",
                content=f"# Investigation Plan\n\n{plan.thinking}\n\n## Steps:\n" +
                    "\n".join(f"- {step.description}" for step in plan.steps),
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
            # ... (keep existing logic for finding executable_steps) ...
            executable_steps = [
                step for step in remaining_steps
                if all(dep in executed_steps for dep in step.dependencies)
            ]

            if not executable_steps:
                yield "error", {"status": "error", "message": "No executable steps but plan not complete", "agent_type": "investigation_planner"}
                break
                
            current_step = executable_steps[0]
            remaining_steps.remove(current_step)
            
            if current_step.step_type == StepType.GITHUB:
                yield "status_update", {"message": "GitHub Agent: Starting query...", "agent_type": "github"}

            # --- Handle Async Generator for multi-step ---
            final_result_data: Optional[QueryResult] = None
            all_yielded_items_multi = [] # Store yields for debugging

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
                 # Decision steps use markdown_generator directly, might not yield updates yet
                 # For now, assume it returns MarkdownQueryResult directly as before generate_content refactor
                 # TODO: Potentially refactor markdown_generator usage if needed
                 yield f"step_{current_step.step_id}_update", {
                     "status": "step_update",
                     "step_id": current_step.step_id,
                     "update_info": {"status": "generating_decision_markdown"},
                     "agent_type": "markdown_generator"
                 }
                 markdown = await self.markdown_generator.run(
                     current_step.description
                 )
                 if markdown and markdown.data and hasattr(markdown.data, 'data'):
                     markdown_string = markdown.data.data
                 else:
                     ai_logger.warning(f"Could not extract markdown data from result for decision step {current_step.step_id}: {markdown}")
                     markdown_string = f"Error generating decision markdown for step {current_step.step_id}."
                 # Create a MarkdownQueryResult for consistency
                 final_result_data = MarkdownQueryResult(data=markdown_string, query=current_step.description)


            if not final_result_data:
                ai_logger.error(f"generate_content/decision logic did not yield/produce a final QueryResult for multi-step {current_step.step_id}. Yielded items: {all_yielded_items_multi}")
                yield f"step_{current_step.step_id}_error", {
                    "status": "error",
                    "step_id": current_step.step_id,
                    "message": f"Failed to get final result for step {current_step.step_id}.",
                    "agent_type": current_step.step_type
                }
                # Decide if we should skip this step or halt the entire plan
                # For now, let's skip and continue if possible, but mark as failed in executed_steps
                executed_steps[current_step.step_id] = {
                     "step": current_step.model_dump(),
                     "content": None, # Indicate failure
                     "error": f"Failed to get final result for step {current_step.step_id}."
                }
                step_results[current_step.step_id] = QueryResult(query=current_step.description, data=None, error=f"Failed to get final result for step {current_step.step_id}.")
                continue # Move to the next potential step


            query = final_result_data.query
            if current_step.step_type == StepType.MARKDOWN:
                query = str(final_result_data.data) # Use data for markdown cell content

            # Create cell for this step (using final_result_data)
            cell_params_for_step = CreateCellParams(
                notebook_id=notebook_id,
                cell_type=current_step.step_type,
                content=query,
                metadata={
                    "session_id": session_id,
                    "step_id": current_step.step_id,
                    "dependencies": current_step.dependencies
                }
            )
            cell_result = await cell_tools.create_cell(
                params=cell_params_for_step
            )
            
            # Store completed step and its result (using final_result_data)
            executed_steps[current_step.step_id] = {
                "step": current_step.model_dump(),
                "content": final_result_data # Store the QueryResult object
            }
            step_results[current_step.step_id] = final_result_data
            
            # Stream status update with cell results (using final_result_data)
            yield f"step_{current_step.step_id}_completed", {
                "status": "step_completed",
                "step_id": current_step.step_id,
                "step_type": current_step.step_type,
                "cell_id": cell_result.get("cell_id", ""), # Use .get for safety
                "cell_params": cell_params_for_step.model_dump(),
                "result": {
                    "data": final_result_data.data,
                    "query": final_result_data.query,
                    "error": final_result_data.error,
                    "metadata": final_result_data.metadata
                },
                "agent_type": current_step.step_type
            }
            
            # --- Plan revision logic (no change here) ---
            if current_step.category == StepCategory.DECISION:
                 # ... (keep existing plan revision logic) ...
                 plan_revision = await self.revise_plan(
                     original_plan=InvestigationPlanModel(
                         steps=plan.steps,
                         thinking=plan.thinking,
                         hypothesis=current_hypothesis
                     ),
                     executed_steps=executed_steps,
                     step_results=step_results,
                     current_hypothesis=current_hypothesis
                 )
                 # ... (keep rest of plan revision logic) ...


        # --- Final summary cell (no change here) ---
        # ... (keep existing summary cell creation code) ...
        # Create final summary cell
        summary_content = await self.markdown_generator.run(
            f"""
            Create a **brief and concise** investigation summary based on these steps and results:
            
            Original Query: {query}
            
            Final Hypothesis: {current_hypothesis}
            
            Step Results: {step_results}
            
            **Focus only on the most critical points:**
            1. Key findings (1-2 sentences)
            2. Root cause(s) identified (if any, briefly stated)
            3. Recommendations for resolution (if any, concise)
            
            Keep the entire summary very short.
            """
        )
        
        # Extract the markdown string from the result
        summary_markdown = ""
        if summary_content and summary_content.data and hasattr(summary_content.data, 'data'):
            summary_markdown = summary_content.data.data
        else:
            ai_logger.warning(f"Could not extract summary markdown from result: {summary_content}")

        summary_cell_params = CreateCellParams(
            notebook_id=notebook_id,
            cell_type="markdown",
            content=f"# Investigation Summary\n\n{summary_markdown}",
            metadata={
                "session_id": session_id,
                "step_id": "summary",
                "dependencies": list(executed_steps.keys())
            }
        )
        summary_cell_result = await cell_tools.create_cell(
            params=summary_cell_params
        )
        yield "summary_created", {
            "status": "summary_created", 
            "agent_type": "markdown_generator",
            "cell_params": summary_cell_params.model_dump(),
            "cell_id": summary_cell_result.get("cell_id", "")
        }

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


async def execute_ai_query_cell(cell: AIQueryCell, context: ExecutionContext) -> Dict:
    """
    Execute an AI query cell
    
    Args:
        cell: The AI query cell to execute
        context: The execution context
        
    Returns:
        The result of executing the cell
    """
    try:
        connection_manager = get_connection_manager()
        # NOTE: execute_ai_query_cell is likely deprecated or needs significant rework
        # as the AIAgent now handles the investigation loop directly via self.investigate
        # It no longer relies on creating MCPServerHTTP instances here.
        # The GitHub agent now uses stdio and fetches its own connection.
        # Grafana agents still rely on MCPServerHTTP passed during AIAgent init.
        ai_logger.warning("execute_ai_query_cell is likely outdated and may not function correctly with current agent architecture.")
        
        # Minimal setup to allow AIAgent init (though it might not be used effectively here)
        mcp_server_map = {} # Keep empty for now as GitHub is handled internally
        
        notebook_id = cell.metadata.get("notebook_id")
        if not notebook_id:
            raise ValueError("Cell metadata must contain notebook_id")
        
        # Initialize the AI agent with the map of MCPServerHTTP clients
        # NOTE: Passing empty list for available_data_sources as this function is likely outdated.
        agent = AIAgent(
            notebook_id=notebook_id, 
            available_data_sources=[], # Added empty list to satisfy constructor
            mcp_server_map=mcp_server_map
        )
        
        # Generate an investigation plan
        plan = await agent.create_investigation_plan(
            query=cell.content,
            notebook_id=notebook_id
        )
        
        # Store thinking output if available
        if plan.thinking:
            cell.thinking = plan.thinking
        
        # Return the plan for the ChatAgent to execute
        return {
            "plan": plan.model_dump(),
            "thinking": plan.thinking
        }
    
    except Exception as e:
        print(f"Error executing AI query cell: {e}")
        return {
            "error": str(e),
            "plan": None
        }