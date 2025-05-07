"""
Step processor for executing and processing individual investigation steps.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Type, AsyncGenerator
from uuid import UUID, uuid4

from backend.ai.models import StepType, InvestigationStepModel
from backend.ai.step_result import StepResult
from backend.ai.step_agents import StepAgent, MarkdownStepAgent, GitHubStepAgent, FileSystemStepAgent, PythonStepAgent, ReportStepAgent
from backend.ai.cell_creator import CellCreator

from pydantic_ai.models.openai import OpenAIModel
from backend.core.cell import CellType
from backend.core.query_result import QueryResult
from backend.ai.chat_tools import NotebookCellTools
from backend.services.connection_manager import ConnectionManager
from backend.ai.events import (
    AgentType, BaseEvent, StatusType, StatusUpdateEvent,
    ToolSuccessEvent, ToolErrorEvent,
    GitHubToolCellCreatedEvent, GitHubToolErrorEvent, 
    FileSystemToolCellCreatedEvent, FileSystemToolErrorEvent, 
    PythonToolCellCreatedEvent, PythonToolErrorEvent,
    StepCompletedEvent, StepErrorEvent, StepExecutionCompleteEvent,
    SummaryCellCreatedEvent, SummaryCellErrorEvent
)

ai_logger = logging.getLogger("ai")

class StepProcessor:
    """Handles the execution and processing of individual investigation steps"""
    def __init__(
        self,
        notebook_id: str, 
        model: OpenAIModel,
        connection_manager: ConnectionManager
    ):
        self.notebook_id = notebook_id
        self.model = model
        self.connection_manager = connection_manager
        self.cell_creator = CellCreator(notebook_id, connection_manager)
        
        # Initialize step agent registry
        self._step_agents: Dict[StepType, StepAgent] = {}
    
    def _get_agent_for_step_type(self, step_type: StepType) -> StepAgent:
        """Get or create the appropriate agent for a step type"""
        if step_type not in self._step_agents:
            if step_type == StepType.MARKDOWN:
                self._step_agents[step_type] = MarkdownStepAgent(self.notebook_id, self.model)
            elif step_type == StepType.GITHUB:
                self._step_agents[step_type] = GitHubStepAgent(self.notebook_id)
            elif step_type == StepType.FILESYSTEM:
                self._step_agents[step_type] = FileSystemStepAgent(self.notebook_id)
            elif step_type == StepType.PYTHON:
                self._step_agents[step_type] = PythonStepAgent(self.notebook_id)
            elif step_type == StepType.INVESTIGATION_REPORT:
                self._step_agents[step_type] = ReportStepAgent(self.notebook_id)
            else:
                raise ValueError(f"Unsupported step type: {step_type}")
                
        return self._step_agents[step_type]
    
    async def process_step(
        self,
        step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, Any],
        plan_step_id_to_cell_ids: Dict[str, List[UUID]],
        cell_tools: NotebookCellTools,
        session_id: str
    ) -> AsyncGenerator[Union[BaseEvent, StepExecutionCompleteEvent], None]:
        """Process a single investigation step and yield event updates"""
        ai_logger.info("<<<<< STEP_PROCESSOR_V_LATEST_MOUNTED >>>>>") 
        if not step.step_type:
            error_msg = f"Step {step.step_id} is missing step_type"
            ai_logger.error(error_msg)
            yield StepErrorEvent(
                step_id=step.step_id,
                error=error_msg,
                agent_type=AgentType.UNKNOWN,
                step_type=None,
                cell_id=None,
                cell_params=None,
                result=None,
                session_id=session_id,
                notebook_id=self.notebook_id
            )
            return
        
        # Initialize a StepResult to collect all outputs from this step
        step_result = StepResult(step.step_id, step.step_type)
        
        # Build context for the step from dependencies
        context = self._build_step_context(step, executed_steps, step_results)
        
        # Get the appropriate agent for this step type
        try:
            step_agent = self._get_agent_for_step_type(step.step_type)
        except ValueError as e:
            error_msg = f"Could not get agent for step {step.step_id}: {e}"
            ai_logger.error(error_msg)
            yield StepErrorEvent(
                step_id=step.step_id,
                error=error_msg,
                agent_type=AgentType.UNKNOWN,
                step_type=str(step.step_type),
                cell_id=None,
                cell_params=None,
                result=None,
                session_id=session_id,
                notebook_id=self.notebook_id
            )
            return
        
        # Track execution results
        final_result_data = None
        github_step_has_tool_errors = False
        filesystem_step_has_tool_errors = False
        python_step_has_tool_errors = False
        
        # Special handling for report steps
        if step.step_type == StepType.INVESTIGATION_REPORT:
            # Add findings for the report
            context["findings_text"] = self._format_findings_for_report(step, executed_steps, step_results)
            # Get original query
            for exec_step in executed_steps.values():
                if 'original_query' in exec_step:
                    context["original_query"] = exec_step["original_query"]
                    break
            
            # Process the report step
            report_error = None
            report_result = None
            
            # Execute report generation
            generator = step_agent.execute(step, context.get("prompt", ""), session_id, context)
            async for event in generator:
                if isinstance(event, BaseEvent):
                    # Forward status events
                    yield event
                elif isinstance(event, dict) and event.get("type") == "final_step_result":
                    report_result = event.get("result")
                    report_error = event.get("error")
                
            # Create report cell
            if report_result:
                dependency_cell_ids = []
                for dep_id in step.dependencies:
                    dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_id, []))
                
                created_cell_id, cell_params, cell_error = await self.cell_creator.create_report_cell(
                    cell_tools=cell_tools,
                    step=step,
                    report=report_result,
                    dependency_cell_ids=dependency_cell_ids,
                    session_id=session_id
                )
                
                if cell_error:
                    yield SummaryCellErrorEvent(
                        error=cell_error,
                        cell_params=cell_params.model_dump() if cell_params else None,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                    step_result.add_error(cell_error)
                else:
                    yield SummaryCellCreatedEvent(
                        cell_params=cell_params.model_dump() if cell_params else {},
                        cell_id=str(created_cell_id) if created_cell_id else None,
                        error=report_error,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                
                # Update tracking
                if created_cell_id:
                    if step.step_id not in plan_step_id_to_cell_ids:
                        plan_step_id_to_cell_ids[step.step_id] = []
                    plan_step_id_to_cell_ids[step.step_id].append(created_cell_id)
                    step_result.add_cell_id(created_cell_id)
                
                # Update step result
                final_result_data = report_result
                step_result.add_output(report_result)
                if report_error:
                    step_result.add_error(report_error)
            else:
                error_msg = "Report generation failed to produce a report"
                step_result.add_error(error_msg)
            
            # Store the comprehensive step result
            step_results[step.step_id] = step_result
            
            # Create final event for the step
            yield StepExecutionCompleteEvent(
                step_id=step.step_id,
                final_result_data=final_result_data,
                step_outputs=step_result.outputs,
                step_error=step_result.get_combined_error(),
                github_step_has_tool_errors=False,
                filesystem_step_has_tool_errors=False,
                python_step_has_tool_errors=False,
                session_id=session_id,
                notebook_id=self.notebook_id
            )
            return
        
        # --- Regular step processing for non-report steps ---
        
        # Extract the agent type based on step type
        agent_type = step_agent.get_agent_type()
        
        # Execute the step
        generator = step_agent.execute(step, context.get("prompt", ""), session_id, context)
        async for event in generator:
            # Handle each type of event
            if isinstance(event, StatusUpdateEvent):
                # Add step_id if missing
                if not event.step_id:
                    # Create a new StatusUpdateEvent with all required fields
                    yield StatusUpdateEvent(
                        status=event.status,
                        agent_type=event.agent_type,
                        message=event.message,
                        attempt=event.attempt,
                        max_attempts=event.max_attempts,
                        reason=event.reason,
                        step_id=step.step_id,
                        original_plan_step_id=event.original_plan_step_id,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                else:
                    yield event
            
            elif isinstance(event, dict) and event.get("type") == "final_step_result" and step.step_type not in [StepType.GITHUB, StepType.FILESYSTEM, StepType.PYTHON]:
                # Handle final results for MARKDOWN steps
                final_result_data = event.get("result")
                if not isinstance(final_result_data, QueryResult):
                    ai_logger.error(f"final_step_result did not contain QueryResult for step {step.step_id}")
                    final_result_data = None
                    step_result.add_error("Internal error: Invalid final result format")
                else:
                    step_result.add_output(final_result_data)
                    if final_result_data.error:
                        step_result.add_error(final_result_data.error)
                break
            
            elif isinstance(event, ToolSuccessEvent):
                # Handle tool success events
                if step.step_type == StepType.GITHUB:
                    # Process GitHub tool event
                    result_content, tool_event = await self._process_tool_success_event(
                        event=event, 
                        step=step, 
                        step_type=step.step_type, 
                        agent_type=agent_type, 
                        plan_step_id_to_cell_ids=plan_step_id_to_cell_ids,
                        cell_tools=cell_tools, 
                        session_id=session_id, 
                        created_event_class=GitHubToolCellCreatedEvent,
                        step_result=step_result
                    )
                    # Forward the event
                    yield tool_event
                    
                elif step.step_type == StepType.FILESYSTEM:
                    # Process Filesystem tool event
                    result_content, tool_event = await self._process_tool_success_event(
                        event=event, 
                        step=step, 
                        step_type=step.step_type, 
                        agent_type=agent_type, 
                        plan_step_id_to_cell_ids=plan_step_id_to_cell_ids,
                        cell_tools=cell_tools, 
                        session_id=session_id, 
                        created_event_class=FileSystemToolCellCreatedEvent,
                        step_result=step_result
                    )
                    # Forward the event
                    yield tool_event
                    
                elif step.step_type == StepType.PYTHON:
                    # Process Python tool event
                    result_content, tool_event = await self._process_tool_success_event(
                        event=event, 
                        step=step, 
                        step_type=step.step_type, 
                        agent_type=agent_type, 
                        plan_step_id_to_cell_ids=plan_step_id_to_cell_ids,
                        cell_tools=cell_tools, 
                        session_id=session_id, 
                        created_event_class=PythonToolCellCreatedEvent,
                        step_result=step_result
                    )
                    # Forward the event
                    yield tool_event
            
            elif isinstance(event, ToolErrorEvent):
                # Handle tool error events
                if step.step_type == StepType.GITHUB:
                    yield GitHubToolErrorEvent(
                        original_plan_step_id=step.step_id,
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        tool_args=event.tool_args,
                        error=event.error,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                    step_result.add_error(f"Error in tool {event.tool_name}: {event.error}")
                
                elif step.step_type == StepType.FILESYSTEM:
                    yield FileSystemToolErrorEvent(
                        original_plan_step_id=step.step_id,
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        tool_args=event.tool_args,
                        error=event.error,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                    filesystem_step_has_tool_errors = True
                    step_result.add_error(f"Error in tool {event.tool_name}: {event.error}")
                
                elif step.step_type == StepType.PYTHON:
                    yield PythonToolErrorEvent(
                        original_plan_step_id=step.step_id,
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        tool_args=event.tool_args,
                        error=event.error,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                    python_step_has_tool_errors = True
                    step_result.add_error(f"Error in Python tool {event.tool_name}: {event.error}")
            
            elif isinstance(event, BaseEvent):
                yield event
        
        # Handle Markdown cell creation
        if step.step_type == StepType.MARKDOWN and final_result_data:
            # Get dependencies
            dependency_cell_ids = []
            for dep_id in step.dependencies:
                dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_id, []))
            
            # Create the markdown cell
            created_cell_id, cell_params, cell_error = await self.cell_creator.create_markdown_cell(
                cell_tools=cell_tools,
                step=step,
                result=final_result_data,
                dependency_cell_ids=dependency_cell_ids,
                session_id=session_id
            )
            
            if cell_error:
                step_result.add_error(cell_error)
            
            # Update step tracking
            if created_cell_id:
                if step.step_id not in plan_step_id_to_cell_ids:
                    plan_step_id_to_cell_ids[step.step_id] = []
                plan_step_id_to_cell_ids[step.step_id].append(created_cell_id)
                step_result.add_cell_id(created_cell_id)
            
            # Yield step completion event
            if step.step_type == StepType.MARKDOWN:
                if step_result.has_error():
                    yield StepErrorEvent(
                        step_id=step.step_id,
                        step_type=step.step_type.value,
                        cell_id=str(created_cell_id) if created_cell_id else None,
                        cell_params=cell_params.model_dump() if cell_params else None,
                        result=final_result_data.model_dump() if final_result_data else None,
                        error=step_result.get_combined_error(),
                        agent_type=agent_type,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                else:
                    yield StepCompletedEvent(
                        step_id=step.step_id,
                        step_type=step.step_type.value,
                        cell_id=str(created_cell_id) if created_cell_id else None,
                        cell_params=cell_params.model_dump() if cell_params else None,
                        result=final_result_data.model_dump() if final_result_data else None,
                        agent_type=agent_type,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
        
        # Store the comprehensive step result
        step_results[step.step_id] = step_result
        
        # Create final event for the step
        yield StepExecutionCompleteEvent(
            step_id=step.step_id,
            final_result_data=final_result_data,
            step_outputs=step_result.outputs,
            step_error=step_result.get_combined_error(),
            github_step_has_tool_errors=github_step_has_tool_errors,
            filesystem_step_has_tool_errors=filesystem_step_has_tool_errors,
            python_step_has_tool_errors=python_step_has_tool_errors,
            session_id=session_id,
            notebook_id=self.notebook_id
        )
    
    async def _process_tool_success_event(
        self, 
        event: ToolSuccessEvent,
        step: InvestigationStepModel,
        step_type: StepType,
        agent_type: AgentType,
        plan_step_id_to_cell_ids: Dict[str, List[UUID]],
        cell_tools: NotebookCellTools,
        session_id: str,
        created_event_class: Type,
        step_result: StepResult
    ) -> Tuple[Any, BaseEvent]:
        """Process a tool success event and create a cell for it"""
        # Get dependencies
        dependency_cell_ids = []
        for dep_id in step.dependencies:
            dependency_cell_ids.extend(plan_step_id_to_cell_ids.get(dep_id, []))
        
        # Map step type to cell type
        # Correctly map based on the agent_type from the event, not the overall step_type
        agent_to_cell_type_map = {
            AgentType.GITHUB: CellType.GITHUB,
            AgentType.FILESYSTEM: CellType.FILESYSTEM,
            AgentType.PYTHON: CellType.PYTHON,
            AgentType.SQL: CellType.SQL,
            AgentType.S3: CellType.S3,
            AgentType.METRIC: CellType.METRIC
            # Add other direct mappings as needed
        }
        
        current_event_agent_type = event.agent_type if event.agent_type is not None else AgentType.UNKNOWN
        cell_type = agent_to_cell_type_map.get(current_event_agent_type)
        
        if cell_type is None:
            # Fallback or error if no direct mapping exists for the agent type that made the tool call
            ai_logger.warning(
                f"No specific CellType mapping for AgentType '{current_event_agent_type.value}' "
                f"that executed tool '{event.tool_name}'. Defaulting cell to AI_QUERY."
            )
            cell_type = CellType.AI_QUERY # A generic fallback
        
        # Create the cell
        created_cell_id, cell_params, cell_error = await self.cell_creator.create_tool_cell(
            cell_tools=cell_tools,
            step=step,
            cell_type=cell_type, # Use the correctly derived cell_type
            agent_type=agent_type, # This is the agent_type of the investigation step's agent (for metadata)
            tool_event=event,
            dependency_cell_ids=dependency_cell_ids,
            session_id=session_id
        )
        
        # Serialize result for storage
        tool_name_str = event.tool_name or "unknown_tool" # Provide default if None
        serialized_result = self.cell_creator._serialize_tool_result(event.tool_result, tool_name_str)
        
        # Add result to the step result
        step_result.add_output(serialized_result)
        
        # Handle cell creation error
        if cell_error:
            ai_logger.warning(f"Error creating cell for {step_type.value} tool {event.tool_name}: {cell_error}")
            step_result.add_error(cell_error)
            # Create a minimal event even when cell creation fails
            tool_event = created_event_class(
                original_plan_step_id=step.step_id,
                cell_id=None,
                tool_name=event.tool_name or "",  # Ensure tool_name is never None
                tool_args=event.tool_args or {},
                result=event.tool_result,
                cell_params=cell_params.model_dump() if cell_params else {},  # Use empty dict if None
                session_id=session_id,
                notebook_id=self.notebook_id
            )
            return serialized_result, tool_event
        
        # Add the cell ID to the step result
        if created_cell_id:
            step_result.add_cell_id(created_cell_id)
        
        # Create the appropriate event
        tool_event = created_event_class(
            original_plan_step_id=step.step_id,
            cell_id=str(created_cell_id) if created_cell_id else None,
            tool_name=event.tool_name,
            tool_args=event.tool_args or {},
            result=event.tool_result,
            cell_params=cell_params.model_dump() if cell_params else None,
            session_id=session_id,
            notebook_id=self.notebook_id
        )
        
        # Update plan_step_id_to_cell_ids
        if created_cell_id:
            if step.step_id not in plan_step_id_to_cell_ids:
                plan_step_id_to_cell_ids[step.step_id] = []
            plan_step_id_to_cell_ids[step.step_id].append(created_cell_id)
        
        # Return both the serialized result and the event
        return serialized_result, tool_event
    
    def _build_step_context(
        self,
        step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context from dependencies for a step"""
        context = {"prompt": step.description}
        
        if step.parameters:
            params_str = "\n".join([f"- {k}: {v}" for k, v in step.parameters.items()])
            context["prompt"] += f"\n\nParameters for this step:\n{params_str}"
        
        # Add dependency information
        dependency_context = []
        for dep_id in step.dependencies:
            if dep_id in executed_steps:
                dep_step_info = executed_steps[dep_id] # Contains {'step': InvestigationStepModel | dict, 'error': Optional[str]}
                raw_dep_step_model_data = dep_step_info.get('step')
                dep_overall_error = dep_step_info.get('error')
                
                dep_result_obj = step_results.get(dep_id) # This should be the StepResult object

                if not raw_dep_step_model_data:
                    ai_logger.warning(f"Could not find step model data for dependency ID {dep_id} in executed_steps.")
                    dependency_context.append(f"- Dependency '{dep_id}': Information unavailable.")
                    continue
                
                # Ensure dep_step_model is an InvestigationStepModel instance
                if isinstance(raw_dep_step_model_data, dict):
                    try:
                        dep_step_model = InvestigationStepModel(**raw_dep_step_model_data)
                    except Exception as e:
                        ai_logger.error(f"Failed to parse dep_step_model data for {dep_id}: {e}. Data: {raw_dep_step_model_data}")
                        dependency_context.append(f"- Dependency '{dep_id}': Step data parsing error.")
                        continue
                elif isinstance(raw_dep_step_model_data, InvestigationStepModel):
                    dep_step_model = raw_dep_step_model_data
                else:
                    ai_logger.error(f"Unexpected type for dep_step_model data for {dep_id}: {type(raw_dep_step_model_data)}")
                    dependency_context.append(f"- Dependency '{dep_id}': Invalid step data type.")
                    continue
                    
                context_line = f"- Dependency '{dep_id}' ({dep_step_model.step_type.value}): {dep_step_model.description[:100]}..."
                
                # Prioritize successful outputs from StepResult
                if isinstance(dep_result_obj, StepResult) and dep_result_obj.outputs:
                    context_line += " (Outputs Available)"
                    outputs_summary_lines = []
                    for i, out_data in enumerate(dep_result_obj.outputs):
                        # Summarize each output. out_data could be dict from JSON, string, or QueryResult.
                        # The Python agent's _parse_python_mcp_output now returns a dict.
                        # Filesystem agent's tool_result is often a string or list of strings.
                        # GitHub agent's tool_result can be a list of dicts or a dict.
                        summary_prefix = f"  - Output {i+1}:"
                        if isinstance(out_data, dict):
                            if out_data.get("status") == "success" and "stdout" in out_data and out_data.get("stdout") is not None:
                                outputs_summary_lines.append(f"{summary_prefix} (stdout): {str(out_data['stdout'])[:100].strip()}...")
                            elif "path" in out_data and "content" in out_data : # For read_file like results
                                outputs_summary_lines.append(f"{summary_prefix} (file: {out_data['path']}): {str(out_data['content'])[:100].strip()}...")
                            elif "search_results" in out_data: # For search_files like results
                                search_results_data = out_data.get("search_results")
                                if isinstance(search_results_data, list) and len(search_results_data) > 0:
                                    outputs_summary_lines.append(f"{summary_prefix} ({len(search_results_data)} search results found). First: {str(search_results_data[0])[:80]}...")
                                elif isinstance(search_results_data, list) and len(search_results_data) == 0:
                                    outputs_summary_lines.append(f"{summary_prefix} (No search results found).")
                                elif search_results_data is None:
                                     outputs_summary_lines.append(f"{summary_prefix} (No search results found).")
                                else: 
                                    outputs_summary_lines.append(f"{summary_prefix} (Search results in unexpected format: {str(search_results_data)[:80]}...).")
                            else: # Generic dict
                                outputs_summary_lines.append(f"{summary_prefix} {str(out_data)[:100].strip()}...")
                        elif isinstance(out_data, list):
                             outputs_summary_lines.append(f"{summary_prefix} (list with {len(out_data)} items). First: {str(out_data[0])[:80]}..." if out_data else f"{summary_prefix} (empty list).")
                        elif isinstance(out_data, QueryResult):
                             outputs_summary_lines.append(f"{summary_prefix} {str(out_data.data)[:100].strip()}...")
                        else: # Fallback for other types (e.g., string from list_files)
                            outputs_summary_lines.append(f"{summary_prefix} {str(out_data)[:100].strip()}...")
                    
                    if outputs_summary_lines:
                        context_line += "\n" + "\n".join(outputs_summary_lines)
                    else:
                        # This case should be rare if dep_result_obj.outputs was not empty
                        context_line += " (Step had outputs, but they could not be summarized for context.)"
                
                # If no successful outputs from StepResult, then consider overall step status
                elif dep_overall_error:
                    context_line += f" (Failed: {str(dep_overall_error)[:100]}... No usable output data for context.)"
                elif isinstance(dep_result_obj, StepResult) and dep_result_obj.has_error():
                     combined_error = dep_result_obj.get_combined_error()
                     error_snippet = str(combined_error)[:100] if combined_error else "Unknown error"
                     context_line += f" (Failed: {error_snippet}... No usable output data for context.)"
                else:
                    context_line += " (Completed with no specific output data for context.)"
                
                dependency_context.append(context_line)
        
        if dependency_context:
            context["prompt"] += f"\n\nContext from Dependencies:\n" + "\n".join(dependency_context)
        else:
            context["prompt"] += "\n\nContext from Dependencies:\nNo preceding steps or no relevant output from them.\n"
            
        return context
    
    def _format_findings_for_report(
        self,
        step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, Any]
    ) -> str:
        """Format findings from previous steps for the report"""
        findings_text_lines = []
        
        for step_id, step_info in executed_steps.items():
            if step_id != step.step_id:  # Exclude the report step itself
                step_obj = step_info.get("step")
                step_result = step_results.get(step_id)
                step_error_msg = step_info.get("error")
                
                desc = step_obj.get("description", "") if step_obj else ""
                type_val = step_obj.get("step_type", "unknown") if step_obj else "unknown"
                
                result_str = ""
                if step_error_msg:
                    result_str = f"Error: {step_error_msg}"
                elif step_result:
                    if isinstance(step_result, StepResult):
                        # New format - StepResult
                        if step_result.has_error():
                            result_str = f"Error: {step_result.get_combined_error()}"
                        elif step_result.outputs:
                            if step_result.step_type == StepType.MARKDOWN and step_result.outputs:
                                # For markdown, just include the data
                                output = step_result.outputs[0]
                                if isinstance(output, QueryResult) and output.data:
                                    result_str = f"Data: {str(output.data)[:500]}..."
                                else:
                                    result_str = f"Data: {str(output)[:500]}..."
                            else:
                                # For multi-output steps like GitHub
                                formatted_outputs = []
                                for i, output in enumerate(step_result.outputs):
                                    formatted_outputs.append(f"  - Output {i+1}: {str(output)[:200]}...")
                                if formatted_outputs:
                                    result_str = "Data (multiple outputs):\n" + "\n".join(formatted_outputs)
                                else:
                                    result_str = "Step completed, but no specific outputs captured."
                        else:
                            result_str = "Step completed, but no outputs available."
                    elif isinstance(step_result, QueryResult):
                        # Legacy format - QueryResult
                        if step_result.error:
                            result_str = f"Error: {step_result.error}"
                        elif step_result.data:
                            result_str = f"Data: {str(step_result.data)[:500]}..."
                        else:
                            result_str = "Step completed, but no data returned."
                    elif isinstance(step_result, list):
                        # Legacy format - list of outputs
                        formatted_outputs = []
                        for i, output in enumerate(step_result):
                            formatted_outputs.append(f"  - Output {i+1}: {str(output)[:200]}...")
                        if formatted_outputs:
                            result_str = "Data (multiple outputs):\n" + "\n".join(formatted_outputs)
                        else:
                            result_str = "Step completed, but no specific outputs captured."
                    else:
                        # Unknown format
                        result_str = f"Data: {str(step_result)[:500]}..."
                else:
                    result_str = "Step completed without explicit data/error capture."
                
                findings_text_lines.append(f"Step {step_id} ({type_val}): {desc}\nResult:\n{result_str}")
        
        findings_text = "\n---\n".join(findings_text_lines)
        if not findings_text:
            findings_text = "No preceding investigation steps were successfully executed or produced results."
        
        return findings_text
