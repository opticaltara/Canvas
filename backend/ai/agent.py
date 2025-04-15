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
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple
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
from backend.config import get_settings
from backend.core.cell import AIQueryCell
from backend.core.execution import ExecutionContext
from backend.core.query_result import  MarkdownQueryResult
from backend.services.connection_manager import get_connection_manager, BaseConnectionConfig
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams

# Get logger for AI operations
ai_logger = logging.getLogger("ai")

# Get settings for API keys
settings = get_settings()


class DataQueryResult(BaseModel):
    """Result of a data source query"""
    data: Any = Field(description="The query result data")
    query: str = Field(description="The executed query")
    error: Optional[str] = Field(description="Error message if query failed", default=None)
    metadata: Dict[str, Any] = Field(description="Additional metadata", default_factory=dict)


class StepType(str, Enum):
    MARKDOWN = "markdown"
    LOG = "log"
    METRIC = "metric"


class InvestigationStepModel(BaseModel):
    """A step in the investigation plan"""
    step_id: str = Field(description="Unique identifier for this step")
    step_type: StepType = Field(description="Type of step (sql, python, markdown, etc.)")
    description: str = Field(description="Description of what this step will do")
    dependencies: List[str] = Field(
        description="List of step IDs this step depends on",
        default_factory=list
    )
    parameters: Dict[str, Any] = Field(
        description="Parameters for the step",
        default_factory=dict
    )
    is_decision_point: bool = Field(
        description="Whether this step evaluates previous results and may change the plan",
        default=False
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
        mcp_servers: Optional[List[MCPServerHTTP]] = None
    ):
        self.settings = get_settings()
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.mcp_servers = mcp_servers or []
        self.notebook_id = notebook_id
    
        # Initialize investigation planner
        self.investigation_planner = Agent(
            self.model,
            deps_type=InvestigationDependencies,
            result_type=InvestigationPlanModel,
            mcp_servers=self.mcp_servers,
            system_prompt="""
            You are the Lead Investigator in a distributed AI system designed to solve complex software incidents. 
            You coordinate a team of specialized agents by creating and adapting investigation plans.

            When analyzing a user query about a software incident:

            1. CONTEXT ASSESSMENT
              • Extract the key incident characteristics (affected services, error patterns, timing)
              • Consider the available data sources provided in the investigation dependencies
              • Frame the investigation as a structured debugging process

            2. INVESTIGATION DESIGN
              Create a structured, adaptable plan with:
              
              • Problem Definition: Clear statement of what's being investigated
              • Expected Impact: How this issue affects users/systems
              • Investigation Graph: Steps with clear dependencies and information flow
              • Adaptation Points: Where plan might need revision based on findings
              • Success Criteria: Specific conditions that indicate resolution

            3. STEP SPECIFICATION
              For each step in your plan, define:
              
              • step_id: A unique identifier (use S1, S2, etc.)
              • step_type: Must be one of ["log", "metric"]
              • description: Instructions for the specialized agent that will:
                - State precisely what question this step answers
                - Provide all context needed for the specialized agent
                - Explain how to interpret the results
                - Reference specific artifacts from previous steps when needed
              • dependencies: Array of step IDs required before this step can execute
              • parameters: Configuration details relevant to this step type
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

            IMPORTANT CONSTRAINTS:
            • Keep the investigation plan concise. Aim for the minimum number of steps required to address the user's core query. 
            Do not generate more than 5 steps unless absolutely necessary for a complex investigation.
            • Stick strictly to the scope of the user's request. Do not add steps for tangential inquiries 
            or explorations not explicitly requested.

            Remember that you are creating instructions for specialized agents, not executing the investigation yourself.
            Your instructions must be detailed and self-contained, as each specialized agent only sees its specific task.
            """
        )


        # Initialize plan reviser
        self.plan_reviser = Agent(
            self.model,
            deps_type=PlanRevisionRequest,
            result_type=PlanRevisionResult,
            mcp_servers=self.mcp_servers,
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
            mcp_servers=self.mcp_servers,
            system_prompt="""
            You are an expert at technical documentation and result analysis. Create clear markdown to address the user's request. When analyzing investigation results:
            
            1. Summarize key findings clearly and objectively
            2. Identify patterns and anomalies in the data
            3. Draw connections between different data sources
            4. Evaluate how findings support or contradict hypotheses
            5. Recommend next steps based on the evidence
            
            Return ONLY the markdown with no meta-commentary.
            """
        )

        self.log_generator = LogQueryAgent(source="loki", notebook_id=self.notebook_id, mcp_servers=self.mcp_servers)
        self.metric_generator = MetricQueryAgent(source="prometheus", notebook_id=self.notebook_id, mcp_servers=self.mcp_servers)

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
            
        # Use stored notebook_id if none provided
        notebook_id = notebook_id or self.notebook_id
        if not notebook_id:
            raise ValueError("notebook_id is required for creating cells")
            
        # Create initial investigation plan
        plan = await self.create_investigation_plan(query, notebook_id, message_history)
        yield "plan_created", {"status": "plan_created", "thinking": plan.thinking, "agent_type": "investigation_planner"}
        
        # Create a markdown cell explaining the plan
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

        # Execute steps in order with potential for plan revision
        executed_steps = {}
        step_results = {}
        current_hypothesis = plan.hypothesis
        remaining_steps = plan.steps.copy()
        
        while remaining_steps:
            # Find executable steps (all dependencies satisfied)
            executable_steps = [
                step for step in remaining_steps
                if all(dep in executed_steps for dep in step.dependencies)
            ]
            
            if not executable_steps:
                # This should not happen with a valid plan, but just in case
                yield "error", {"status": "error", "message": "No executable steps but plan not complete", "agent_type": "investigation_planner"}
                break
                
            # Execute the first available step
            current_step = executable_steps[0]
            remaining_steps.remove(current_step)
            
            # Generate content for this step
            result = await self.generate_content(
                current_step.step_type,
                current_step.description,
                executed_steps=executed_steps,
                step_results=step_results
            )

            query = result.query
            
            # Create cell for this step
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
            
            # Store completed step and its result
            executed_steps[current_step.step_id] = {
                "step": current_step.model_dump(),
                "content": result
            }
            step_results[current_step.description] = result
            
            # Stream status update with cell results
            yield f"step_{current_step.step_id}_completed", {
                "status": "step_completed",
                "step_id": current_step.step_id,
                "step_type": current_step.step_type,
                "cell_id": cell_result["cell_id"],
                "cell_params": cell_params_for_step.model_dump(),
                "result": {
                    "data": result.data,
                    "query": result.query,
                    "error": result.error,
                    "metadata": result.metadata
                },
                "agent_type": current_step.step_type
            }
            
            # Check if this is a decision point
            if current_step.is_decision_point and current_step.step_type == "markdown":
                # This is a decision point, so we should revise the plan
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
                
                # If the plan revision suggests changes
                if not plan_revision.continue_as_is:
                    # Update the hypothesis if provided
                    if plan_revision.new_hypothesis:
                        current_hypothesis = plan_revision.new_hypothesis
                        
                        # Create a markdown cell explaining the hypothesis update
                        hypothesis_cell_params = CreateCellParams(
                            notebook_id=notebook_id,
                            cell_type="markdown",
                            content=f"## Updated Hypothesis\n\n{current_hypothesis}",
                            metadata={
                                "session_id": session_id,
                                "step_id": f"hypothesis_update_after_{current_step.step_id}",
                                "dependencies": [current_step.step_id]
                            }
                        )
                        hypothesis_cell_result = await cell_tools.create_cell(
                            params=hypothesis_cell_params
                        )
                        
                        # Yield the hypothesis cell creation status
                        yield f"hypothesis_cell_created_after_{current_step.step_id}", {
                            "status": "hypothesis_cell_created",
                            "agent_type": "plan_reviser",
                            "cell_params": hypothesis_cell_params.model_dump(),
                            "cell_id": hypothesis_cell_result.get("cell_id", "")
                        }
                    
                    # Remove steps that should be skipped
                    if plan_revision.steps_to_remove:
                        # Create a set of steps to remove and their dependencies
                        steps_to_remove = set(plan_revision.steps_to_remove)
                        
                        # Find all steps that depend on steps being removed
                        dependent_steps = set()
                        for step in remaining_steps:
                            if any(dep in steps_to_remove for dep in step.dependencies):
                                dependent_steps.add(step.step_id)
                        
                        # Combine direct steps to remove and their dependents
                        all_steps_to_remove = steps_to_remove.union(dependent_steps)
                        
                        # Filter remaining steps
                        remaining_steps = [
                            step for step in remaining_steps
                            if step.step_id not in all_steps_to_remove
                        ]
                        
                        # Create a markdown cell explaining removed steps
                        removed_steps_cell_params = CreateCellParams(
                            notebook_id=notebook_id,
                            cell_type="markdown",
                            content=f"## Plan Adaptation: Removed Steps\n\n{plan_revision.explanation}\n\nRemoved steps: {', '.join(all_steps_to_remove)}",
                            metadata={
                                "session_id": session_id,
                                "step_id": f"plan_adaptation_removed_steps_after_{current_step.step_id}",
                                "dependencies": [current_step.step_id]
                            }
                        )
                        removed_steps_cell_result = await cell_tools.create_cell(
                            params=removed_steps_cell_params
                        )
                        
                        # Yield the removed steps cell creation status
                        yield f"removed_steps_cell_created_after_{current_step.step_id}", {
                            "status": "removed_steps_cell_created",
                            "agent_type": "plan_reviser",
                            "cell_params": removed_steps_cell_params.model_dump(),
                            "cell_id": removed_steps_cell_result.get("cell_id", "")
                        }
                    
                    # Add new steps
                    if plan_revision.new_steps:
                        new_step_ids = []
                        for new_step_data in plan_revision.new_steps:
                            new_step = InvestigationStepModel(**new_step_data.model_dump())
                            remaining_steps.append(new_step)
                            new_step_ids.append(new_step.step_id)
                        
                        # Create a markdown cell explaining new steps
                        new_steps_cell_params = CreateCellParams(
                            notebook_id=notebook_id,
                            cell_type="markdown",
                            content=f"## Plan Adaptation: New Steps\n\n{plan_revision.explanation}\n\nAdded steps: {', '.join(new_step_ids)}",
                            metadata={
                                "session_id": session_id,
                                "step_id": f"plan_adaptation_new_steps_after_{current_step.step_id}",
                                "dependencies": [current_step.step_id]
                            }
                        )
                        new_steps_cell_result = await cell_tools.create_cell(
                            params=new_steps_cell_params
                        )
                        
                        # Yield the new steps cell creation status
                        yield f"new_steps_cell_created_after_{current_step.step_id}", {
                            "status": "new_steps_cell_created",
                            "agent_type": "plan_reviser",
                            "cell_params": new_steps_cell_params.model_dump(),
                            "cell_id": new_steps_cell_result.get("cell_id", "")
                        }
                    
                    yield "plan_revised", {
                        "status": "plan_revised", 
                        "explanation": plan_revision.explanation,
                        "agent_type": "plan_reviser"
                    }
        
        # Create final summary cell
        summary_content = await self.markdown_generator.run(
            f"""
            Create a comprehensive investigation summary based on these steps and results:
            
            Original Query: {query}
            
            Final Hypothesis: {current_hypothesis}
            
            Step Results: {step_results}
            
            Create a summary with:
            1. Brief restatement of the original issue
            2. Key findings from the investigation
            3. Root cause(s) identified
            4. Impact assessment
            5. Recommendations for resolution
            6. Any open questions or further investigation needed
            """
        )
        
        summary_cell_params = CreateCellParams(
            notebook_id=notebook_id,
            cell_type="markdown",
            content=f"# Investigation Summary\n\n{summary_content}",
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
            
        # Get available data sources from connection manager
        connection_manager = get_connection_manager()
        connections = await connection_manager.get_all_connections()
        available_data_sources = [conn["type"] for conn in connections]

        # Create dependencies for the investigation planner
        deps = InvestigationDependencies(
            user_query=query,
            notebook_id=UUID(notebook_id),
            available_data_sources=available_data_sources,
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
                is_decision_point=step_data.is_decision_point
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

    async def generate_content(
        self,
        step_type: str,
        description: str,
        executed_steps: Dict[str, Any] | None = None,
        step_results: Dict[str, Any] = {}
    ):
        """
        Generate content for a specific step type
        
        Args:
            step_type: Type of step (sql, python, markdown, etc.)
            description: Description of what the step will do
            executed_steps: Previously executed steps (for context)
            step_results: Results from previously executed steps (for context)
            
        Returns:
            Generated content for the step
        """
        # If we already have results for this step, use them
        if description in step_results:
            return step_results[description]
            
        context = ""
        if executed_steps and step_results:
            # Add context from previous steps if available
            context = f"\n\nContext from previous steps:\n{executed_steps}\n\nResults from previous steps:\n{step_results}"
        
        # Generate content based on step type with context
        if step_type == "log":
            result = await self.log_generator.run_query(description + context)
            return result
        elif step_type == "metric":
            result = await self.metric_generator.run_query(description + context)
            return result
        elif step_type == "markdown":
            result = await self.markdown_generator.run(description + context)
            return result.data
        else:
            raise ValueError(f"Unsupported step type: {step_type}")


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
        # Get connection manager
        connection_manager = get_connection_manager()
        
        # Get all connections
        connections = await connection_manager.get_all_connections()
        
        # Start MCP servers for all connections
        server_addresses = {}
        for conn in connections:
            if isinstance(conn, dict) and "id" in conn:
                connection_config = BaseConnectionConfig(**conn)
                await connection_config.start_mcp_server()
                server_addresses[conn["id"]] = connection_config.mcp_status["address"]
        
        # Create MCPServerHTTP instances for each connection
        mcp_servers = []
        for conn in connections:
            if isinstance(conn, dict) and "id" in conn and conn["id"] in server_addresses:
                server_url = server_addresses[conn["id"]]
                if server_url.startswith("http"):
                    mcp_servers.append(MCPServerHTTP(url=f"{server_url}/sse"))
        
        # Get notebook_id from cell metadata
        notebook_id = cell.metadata.get("notebook_id")
        if not notebook_id:
            raise ValueError("Cell metadata must contain notebook_id")
        
        # Initialize the AI agent with MCP servers and notebook ID
        agent = AIAgent(mcp_servers=mcp_servers, notebook_id=notebook_id)
        
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