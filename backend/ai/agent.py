"""
Main AI Agent for investigation planning and execution.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
from uuid import UUID, uuid4
from datetime import datetime, timezone

from pydantic_ai import Agent
# OpenAIModel is no longer directly used, SafeOpenAIModel is used instead
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from backend.ai.models import (
    StepType, InvestigationStepModel, InvestigationPlanModel, 
    InvestigationDependencies, PlanRevisionRequest, PlanRevisionResult,
    SafeOpenAIModel # Import the custom model
)
from backend.ai.step_result import StepResult
from backend.ai.step_processor import StepProcessor

from backend.core.cell import CellType, CellStatus
from backend.ai.events import (
    AgentType, BaseEvent, StatusType, 
    PlanCreatedEvent, PlanCellCreatedEvent, 
    StepStartedEvent, StepExecutionCompleteEvent, 
    StepErrorEvent, InvestigationCompleteEvent,
    ToolSuccessEvent, ToolErrorEvent
)
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams
from backend.core.notebook import Notebook
from backend.services.notebook_manager import NotebookManager
from backend.db.database import get_db_session
from backend.services.connection_manager import ConnectionManager, get_connection_manager
from backend.ai.prompts.investigation_prompts import (
    INVESTIGATION_PLANNER_SYSTEM_PROMPT,
    PLAN_REVISER_SYSTEM_PROMPT,
    MARKDOWN_GENERATOR_SYSTEM_PROMPT
)
from backend.config import get_settings

ai_logger = logging.getLogger("ai")

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
            # Handle different part types
            if hasattr(msg.parts[0], 'content'):
                content = getattr(msg.parts[0], 'content', '')
            elif hasattr(msg.parts[0], 'cell_params'):
                content = f"[Assistant created cell: {getattr(msg.parts[0], 'cell_id', 'unknown')}]"
            else:
                content = f"[Assistant action: {getattr(msg.parts[0], 'part_kind', 'unknown')}]"

        if content:
            # Basic cleaning/truncation for long content
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
        self.model = SafeOpenAIModel( # Use the SafeOpenAIModel
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
        
        # Format the planner prompt with available data sources
        available_data_sources_str = ', '.join(self.available_data_sources) if self.available_data_sources else 'None'
        formatted_planner_prompt = INVESTIGATION_PLANNER_SYSTEM_PROMPT.format(
            available_data_sources_str=available_data_sources_str
        )

        # Build notebook context tools (list_cells & get_cell)
        try:
            from .notebook_context_tools import create_notebook_context_tools
            self._notebook_tools = create_notebook_context_tools(self.notebook_id, self.notebook_manager)
        except Exception as tool_err:
            ai_logger.error("Failed to create notebook context tools for AIAgent: %s", tool_err, exc_info=True)
            self._notebook_tools = []

        # Initialize the step processor
        self.step_processor = StepProcessor(
            notebook_id=self.notebook_id,
            model=self.model,
            connection_manager=self.connection_manager,
            notebook_manager=self.notebook_manager
        )

        # Initialize the investigation planner
        self.investigation_planner = Agent(
            self.model,
            deps_type=InvestigationDependencies,
            output_type=InvestigationPlanModel,
            system_prompt=formatted_planner_prompt,
            tools=self._notebook_tools,  # type: ignore[arg-type]
        )

        # Other agents will be initialized on demand
        self.plan_reviser = None
        self.markdown_generator = Agent(
            self.model,
            output_type=str,  # Just using str as output for simplicity
            mcp_servers=list(self.mcp_server_map.values()),
            system_prompt=MARKDOWN_GENERATOR_SYSTEM_PROMPT,
            tools=self._notebook_tools,  # type: ignore[arg-type]
        )
        
        ai_logger.info(f"AIAgent initialized for notebook {notebook_id}.")

    async def investigate(
        self, 
        query: str, 
        session_id: str, 
        notebook_id: Optional[str] = None,
        message_history: List[ModelMessage] = [],
        cell_tools: Optional[NotebookCellTools] = None,
        notebook_context_summary: Optional[str] = None
    ) -> AsyncGenerator[BaseEvent, None]:
        """
        Investigate a query by creating and executing a plan.
        Streams status updates as steps are completed.
        """
        async with get_db_session() as db:
            if not cell_tools:
                raise ValueError("cell_tools is required for creating cells")
                
            notebook_id_str = notebook_id or self.notebook_id
            if not notebook_id_str:
                raise ValueError("notebook_id is required for creating cells")
            
            # Fetch the actual Notebook object
            try:
                notebook_uuid = UUID(notebook_id_str)
                notebook: Notebook = await self.notebook_manager.get_notebook(db=db, notebook_id=notebook_uuid)
            except ValueError:
                ai_logger.error(f"Invalid notebook_id format: {notebook_id_str}")
                raise ValueError(f"Invalid notebook_id format: {notebook_id_str}")
            except Exception as e:
                ai_logger.error(f"Failed to fetch notebook {notebook_id_str}: {e}", exc_info=True)
                raise ValueError(f"Failed to fetch notebook {notebook_id_str}")

            # Create the investigation plan
            plan = await self.create_investigation_plan(query, notebook_id_str, message_history, notebook_context_summary)
            
            # Yield plan created event
            yield PlanCreatedEvent(thinking=plan.thinking, session_id=session_id, notebook_id=notebook_id_str)
            
            # Initialize tracking
            plan_step_id_to_cell_ids: Dict[str, List[UUID]] = {}
            executed_steps: Dict[str, Any] = {}
            step_results: Dict[str, Any] = {}

            # Create plan explanation cell if there are multiple steps
            if len(plan.steps) > 1:
                plan_cell_event = await self.create_plan_explanation_cell(plan, cell_tools, session_id, notebook_id_str)
                if plan_cell_event:
                    yield plan_cell_event

            # Execute the plan steps
            remaining_steps = plan.steps.copy()

            while remaining_steps:
                # Get executable steps (all dependencies satisfied)
                executable_steps = [
                    step for step in remaining_steps
                    if all(dep in executed_steps for dep in step.dependencies)
                ]

                if not executable_steps:
                    # Check if we're blocked on dependencies
                    is_stalled, stall_error = self.check_plan_stalled(remaining_steps, executed_steps, session_id, notebook_id_str)
                    if is_stalled:
                        if stall_error:
                            yield stall_error
                        break
                    break
                
                # Execute the first available step
                current_step = executable_steps[0]
                remaining_steps.remove(current_step)

                # Get the agent type for this step
                agent_type = self._get_agent_type_for_step(current_step.step_type)
                
                # Yield step started event
                yield StepStartedEvent(
                    step_id=current_step.step_id, 
                    agent_type=agent_type, 
                    session_id=session_id, 
                    notebook_id=notebook_id_str
                )
                
                # Process the step
                async for event in self.step_processor.process_step(
                    step=current_step,
                    executed_steps=executed_steps,
                    step_results=step_results,
                    plan_step_id_to_cell_ids=plan_step_id_to_cell_ids,
                    cell_tools=cell_tools,
                    session_id=session_id,
                    db=db
                ):
                    # Process specific events for AIAgent's internal state first
                    processed_internally = False
                    if isinstance(event, StepExecutionCompleteEvent):
                        # Update executed_steps based on the StepExecutionCompleteEvent
                        step_result = step_results.get(current_step.step_id)
                        
                        if step_result and isinstance(step_result, StepResult):
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": self._get_step_summary_content(step_result),
                                "error": step_result.get_combined_error()
                            }
                        else:
                            ai_logger.warning(f"StepResult not found or invalid for step {current_step.step_id} upon StepExecutionCompleteEvent. Fallback.")
                            executed_steps[current_step.step_id] = {
                                "step": current_step.model_dump(),
                                "content": None,
                                "error": event.step_error or "Step result missing after execution"
                            }
                        processed_internally = True
                        # This event will also be yielded below by the BaseEvent check

                    # Forward all BaseEvents (including StepExecutionCompleteEvent after its specific processing) to the client
                    if isinstance(event, BaseEvent):
                        yield event
                    elif not processed_internally: # Log if an event was not a BaseEvent and not processed internally
                        ai_logger.warning(f"Received unexpected event type from StepProcessor: {type(event)}. This event was not yielded.")
                
            # All steps completed
            ai_logger.info(f"Investigation plan execution finished for notebook {notebook_id_str}")
            
            # Yield investigation complete event
            yield InvestigationCompleteEvent(session_id=session_id, notebook_id=notebook_id_str)
            
            # Save the notebook state
            if 'notebook' in locals() and notebook is not None:
                try:
                    await self.notebook_manager.save_notebook(db=db, notebook_id=notebook.id, notebook=notebook)
                    ai_logger.info(f"Successfully saved notebook {notebook.id} state after investigation.")
                except Exception as save_err:
                    ai_logger.error(f"Failed to save notebook {notebook.id} state: {save_err}", exc_info=True)

    def _get_step_summary_content(self, step_result: StepResult) -> str:
        """Get a summary content string for the step result"""
        if step_result.step_type == StepType.MARKDOWN and step_result.outputs:
            # For markdown, use the data from the output
            output = step_result.outputs[0]
            if hasattr(output, "data"):
                return output.data or "No markdown content available"
            return str(output)
        elif step_result.step_type == StepType.INVESTIGATION_REPORT and step_result.outputs:
            # For report, use the title from the first output
            output = step_result.outputs[0]
            if hasattr(output, "title"):
                return f"Investigation Report: {output.title}"
            return "Investigation Report"
        else:
            # For other step types, create a summary
            if step_result.has_error():
                return f"Completed {step_result.step_type.value if step_result.step_type else 'unknown'} step with errors"
            else:
                return f"Completed {step_result.step_type.value if step_result.step_type else 'unknown'} step with {len(step_result.outputs)} outputs"

    async def create_plan_explanation_cell(self, plan: InvestigationPlanModel, cell_tools: NotebookCellTools, session_id: str, notebook_id_str: str) -> Optional[PlanCellCreatedEvent]:
        """Create a cell that explains the investigation plan"""
        plan_cell_params = CreateCellParams(
            notebook_id=notebook_id_str,
            cell_type=CellType.MARKDOWN,
            content=f"# Investigation Plan\n\n## Steps:\n" +
                    "\n".join(f"- `{step.step_id}` ({step.step_type.value}): {step.description}" for step in plan.steps),
            metadata={
                "session_id": session_id,
                "step_id": "plan",
                "dependencies": []
            },
            tool_call_id=uuid4()
        )
        
        try:
            plan_cell_result = await cell_tools.create_cell(params=plan_cell_params)
            return PlanCellCreatedEvent(
                cell_params=plan_cell_params.model_dump(mode='json'),
                cell_id=plan_cell_result.get("cell_id", ""),
                session_id=session_id,
                notebook_id=notebook_id_str
            )
        except Exception as plan_cell_err:
            ai_logger.error(f"Failed to create plan explanation cell: {plan_cell_err}", exc_info=True)
            # We continue even if this fails, as it's not critical
            return None
    
    def check_plan_stalled(self, remaining_steps: List[InvestigationStepModel], executed_steps: Dict[str, Any], session_id: str, notebook_id_str: str) -> Tuple[bool, Optional[StepErrorEvent]]:
        """Check if plan is stalled due to unresolvable dependencies and return error if needed"""
        # Check for unresolved dependencies
        unresolved_deps = False
        for step in remaining_steps:
            missing_deps = [dep for dep in step.dependencies if dep not in executed_steps]
            if missing_deps:
                unresolved_deps = True
                ai_logger.warning(f"Step {step.step_id} blocked, missing dependencies: {missing_deps}")
                break
        
        # If no unresolved deps but still have remaining steps, plan is stalled
        if not unresolved_deps and remaining_steps:
            ai_logger.error("No executable steps but plan not complete and no unresolved dependencies found.")
            error_event = StepErrorEvent(
                step_id="PLAN_STALLED",
                error="Plan execution stalled unexpectedly.",
                agent_type=AgentType.INVESTIGATION_PLANNER,
                step_type=None,
                cell_id=None,
                cell_params=None,
                result=None,
                session_id=session_id,
                notebook_id=notebook_id_str
            )
            return True, error_event
        
        return False, None
    
    def _get_agent_type_for_step(self, step_type: Optional[StepType]) -> AgentType:
        """Map step type to agent type"""
        if not step_type:
            return AgentType.UNKNOWN
        
        agent_type_map = {
            StepType.MARKDOWN: AgentType.MARKDOWN,
            StepType.GITHUB: AgentType.GITHUB,
            StepType.FILESYSTEM: AgentType.FILESYSTEM,
            StepType.PYTHON: AgentType.PYTHON,
            StepType.INVESTIGATION_REPORT: AgentType.INVESTIGATION_REPORT_GENERATOR
        }
        
        return agent_type_map.get(step_type, AgentType.UNKNOWN)

    async def create_investigation_plan(self, query: str, notebook_id: Optional[str] = None, message_history: List[ModelMessage] = [], notebook_context_summary: Optional[str] = None) -> InvestigationPlanModel:
        """Create an investigation plan for the given query, incorporating history and notebook context into the prompt."""
        notebook_id = notebook_id or self.notebook_id
        if not notebook_id:
            raise ValueError("notebook_id is required for creating investigation plan")

        # Format the history and combine with query
        formatted_history = format_message_history(message_history)
        combined_prompt = f"Conversation History:\n---\n{formatted_history}\n---\n\nLatest User Query: {query}"

        if notebook_context_summary:
            combined_prompt += f"\n\n--- Existing Notebook Context ---\n{notebook_context_summary}\n--- End of Existing Notebook Context ---"
        
        ai_logger.info(f"Planner combined prompt:\n{combined_prompt}")

        # Create dependencies
        deps = InvestigationDependencies(
            user_query=query,
            notebook_id=UUID(notebook_id),
            available_data_sources=self.available_data_sources,
            executed_steps={},
            current_hypothesis=None,
            message_history=[]  # Pass empty list as history is in the prompt
        )

        # Generate the plan
        result = await self.investigation_planner.run(
            user_prompt=combined_prompt,
            deps=deps
        )
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

    async def update_plan(
        self,
        original_plan: InvestigationPlanModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, Any],
        current_hypothesis: Optional[str] = None,
        unexpected_results: List[str] = []
    ) -> PlanRevisionResult:
        """
        Revise the investigation plan based on executed steps
        """
        # Lazy initialization of plan reviser
        if self.plan_reviser is None:
            ai_logger.info("Lazily initializing PlanReviser Agent.")
            self.plan_reviser = Agent(
                self.model,
                deps_type=PlanRevisionRequest,
                output_type=PlanRevisionResult,
                system_prompt=PLAN_REVISER_SYSTEM_PROMPT,
                tools=self._notebook_tools,  # type: ignore[arg-type]
            )
            
        # Create revision request
        revision_request = PlanRevisionRequest(
            original_plan=original_plan,
            executed_steps=executed_steps,
            step_results=step_results,
            unexpected_results=unexpected_results,
            current_hypothesis=current_hypothesis
        )
        
        # Generate revision
        result = await self.plan_reviser.run(
            user_prompt="Revise the investigation plan based on executed steps and their results.",
            deps=revision_request
        )
        
        return result.output

    async def execute_content(
        self,
        current_step: InvestigationStepModel,
        step_type: StepType,
        description: str,
        executed_steps: Dict[str, Any] | None = None,
        step_results: Dict[str, Any] = {},
        cell_tools: Optional[NotebookCellTools] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Union[Dict[str, Any], BaseEvent], None]:
        """
        Legacy method to maintain API compatibility.
        Delegates to the step processor.
        """
        # Build context for the step
        context = {}
        if executed_steps:
            context = {"executed_steps": executed_steps, "step_results": step_results}
        
        # Get the appropriate agent for this step
        try:
            agent = self.step_processor._get_agent_for_step_type(step_type)
            
            # Execute the step
            generator = await agent.execute(
                step=current_step,
                agent_input_data=description,
                session_id=session_id or "",
                context=context
            )
            async for event in generator:
                yield event
        except Exception as e:
            ai_logger.error(f"Error in execute_content: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": f"Failed to execute content: {str(e)}"
            }
