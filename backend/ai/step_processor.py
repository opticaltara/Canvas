"""
Step processor for executing and processing individual investigation steps.
"""

import logging
import os
import hashlib
import traceback # Added
from typing import Any, Dict, List, Optional, Tuple, Union, Type, AsyncGenerator
from uuid import UUID, uuid4

# Model Imports
from backend.ai.models import StepType, InvestigationStepModel, PythonAgentInput, FileDataRef # Added PythonAgentInput, FileDataRef
from backend.ai.step_result import StepResult
from backend.ai.step_agents import StepAgent, MarkdownStepAgent, GitHubStepAgent, FileSystemStepAgent, PythonStepAgent, ReportStepAgent
from backend.ai.cell_creator import CellCreator
from backend.core.file_utils import stage_file_content # Added
from pydantic_ai.models.openai import OpenAIModel
from backend.core.cell import CellType, CellStatus # Added CellStatus
from backend.core.query_result import QueryResult, InvestigationReport # Added InvestigationReport
from backend.services.notebook_manager import NotebookManager # Added import
from sqlalchemy.ext.asyncio import AsyncSession # Added import

# Tool/Service Imports
from backend.ai.chat_tools import NotebookCellTools
from backend.services.connection_manager import ConnectionManager
from backend.services.connection_handlers.registry import get_handler # Added
from backend.services.connection_handlers.filesystem_handler import FileSystemConnectionHandler # Added
from mcp.shared.exceptions import McpError # Added

# Event Imports
from backend.ai.events import (
    AgentType, BaseEvent, StatusType, StatusUpdateEvent,
    ToolSuccessEvent, ToolErrorEvent,
    GitHubToolCellCreatedEvent, GitHubToolErrorEvent, 
    FileSystemToolCellCreatedEvent, FileSystemToolErrorEvent, 
    PythonToolCellCreatedEvent, PythonToolErrorEvent,
    StepCompletedEvent, StepErrorEvent, StepExecutionCompleteEvent,
    SummaryCellCreatedEvent, SummaryCellErrorEvent, FatalErrorEvent # Added FatalErrorEvent
)

ai_logger = logging.getLogger("ai")

class StepProcessor:
    """Handles the execution and processing of individual investigation steps"""
    def __init__(
        self,
        notebook_id: str, 
        model: OpenAIModel,
        connection_manager: ConnectionManager,
        notebook_manager: Optional[NotebookManager] = None # Added notebook_manager
    ):
        self.notebook_id = notebook_id
        self.model = model
        self.connection_manager = connection_manager
        self.notebook_manager = notebook_manager # Store notebook_manager
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
                # Pass notebook_manager to PythonStepAgent
                self._step_agents[step_type] = PythonStepAgent(self.notebook_id, notebook_manager=self.notebook_manager)
            elif step_type == StepType.INVESTIGATION_REPORT:
                self._step_agents[step_type] = ReportStepAgent(self.notebook_id)
            else:
                raise ValueError(f"Unsupported step type: {step_type}")
                
        return self._step_agents[step_type]

    async def _prepare_data_references(
        self, 
        step: InvestigationStepModel, 
        step_results: Dict[str, StepResult], 
        session_id: str
    ) -> List[FileDataRef]:
        """
        Prepares FileDataRef objects for PythonAgent, resolving paths from Filesystem MCP
        if necessary, and staging content locally.
        Raises exceptions on critical failures (IOError, ConnectionError, McpError).
        """
        resolved_refs: List[FileDataRef] = []
        staged_files_tracker: Dict[str, str] = {} # Track content hash -> staged_path

        for dep_id in step.dependencies:
            dep_result_obj = step_results.get(dep_id)
            if not isinstance(dep_result_obj, StepResult) or \
               dep_result_obj.step_type != StepType.FILESYSTEM or \
               not dep_result_obj.outputs or \
               dep_result_obj.has_error(): # Skip if dependency step itself had errors
                continue

            source_cell_id = str(dep_result_obj.cell_ids[0]) if dep_result_obj.cell_ids else None
            ai_logger.info(f"Processing outputs from FileSystem dependency step: {dep_id} for Python step {step.step_id}, session: {session_id}")

            for output_idx, output_item in enumerate(dep_result_obj.outputs):
                content: Optional[str] = None
                original_filename: Optional[str] = None
                fsmcp_path: Optional[str] = None
                
                default_original_filename = f"staged_from_content_{dep_id}_output_{output_idx}.dat"

                if isinstance(output_item, str) and len(output_item) > 10 and ('\n' in output_item or ',' in output_item or output_item.startswith("Traceback")):
                    # This case implies output_item is likely direct file content (or large text).
                    # If the FileSystemAgent.ToolSuccessEvent.tool_result for read_file is now a dict,
                    # this branch will be less likely hit for read_file results.
                    # If it *is* hit, and it's from a read_file that somehow didn't get structured,
                    # it would use this string content. This is the old problematic path if content is truncated.
                    content = output_item
                    original_filename = default_original_filename
                elif isinstance(output_item, list) and all(isinstance(p, str) and p.startswith('/') for p in output_item):
                     if output_item:
                         fsmcp_path = output_item[0] 
                         original_filename = os.path.basename(fsmcp_path) if fsmcp_path else default_original_filename
                     else:
                         ai_logger.info(f"Empty list from search_files for dep {dep_id}, output {output_idx}")
                         continue
                elif isinstance(output_item, dict):
                    # Check if this is a structured result from our modified FileSystemAgent for 'read_file'
                    if output_item.get("tool_name") == "read_file" and "path" in output_item:
                        fsmcp_path = output_item["path"]
                        original_filename = os.path.basename(fsmcp_path) if fsmcp_path else default_original_filename
                        # Crucially, ensure content is None to trigger the re-fetch logic below using fsmcp_path
                        content = None 
                        ai_logger.info(f"Identified structured 'read_file' result for {fsmcp_path}. Will re-fetch full content.")
                    elif 'content' in output_item and isinstance(output_item['content'], str):
                         # This handles other tools that might return content directly in a dict,
                         # or our FileSystemAgent's 'content_preview' if the above 'read_file' check wasn't met.
                         # If it's 'content_preview', it's truncated, but re-fetch should be preferred.
                         content = output_item['content'] # This might be a preview.
                         original_filename = output_item.get('original_filename') or \
                                           (os.path.basename(output_item['path']) if output_item.get('path') and isinstance(output_item.get('path'), str) else None) or \
                                           default_original_filename
                         # If we got here with a 'read_file' tool's preview, fsmcp_path might not be set yet.
                         # Try to get it if available, to allow re-fetch to take precedence if content is just a preview.
                         if output_item.get("tool_name") == "read_file" and output_item.get("path"):
                             fsmcp_path = output_item.get("path")
                             ai_logger.info(f"Found path {fsmcp_path} for read_file output that also had direct content/preview. Will prioritize re-fetching.")
                             content = None # Force re-fetch

                    elif 'path' in output_item and isinstance(output_item['path'], str) and output_item['path'].startswith('/'):
                         # This handles dicts that primarily provide a path (e.g., from search_files if it returned dicts)
                         fsmcp_path = output_item['path']
                         original_filename = output_item.get('original_filename') or \
                                           (os.path.basename(fsmcp_path) if fsmcp_path else None) or \
                                           default_original_filename
                    else:
                        ai_logger.warning(f"Unsupported dict structure in FS output for dep {dep_id}, output {output_idx}: {str(output_item)[:200]}")
                        continue
                elif isinstance(output_item, str) and output_item.startswith('/'):
                    # Handles if output_item is just a path string (e.g. from search_files)
                    fsmcp_path = output_item
                    original_filename = os.path.basename(fsmcp_path) if fsmcp_path else default_original_filename
                else:
                    ai_logger.warning(f"Skipping unrecognized output item type from FS dep {dep_id}, output {output_idx}: {type(output_item)}, data: {str(output_item)[:100]}")
                    continue
                         
                if fsmcp_path and content is None:
                    ai_logger.info(f"Attempting to read content from Filesystem MCP for path: {fsmcp_path}")
                    try:
                        fs_handler_instance = get_handler("filesystem")
                        if not isinstance(fs_handler_instance, FileSystemConnectionHandler):
                            raise ConnectionError("Filesystem handler not correctly configured or not found.")
                        
                        default_conn = await self.connection_manager.get_default_connection("filesystem")
                        if not default_conn or not default_conn.config:
                            raise ConnectionError("Default filesystem connection or its config not found.")
                        
                        read_result = await fs_handler_instance.execute_tool_call(
                            tool_name="read_file",
                            tool_args={"path": fsmcp_path},
                            db_config=default_conn.config, 
                            correlation_id=session_id
                        )
                        if isinstance(read_result, str):
                            content = read_result
                            ai_logger.info(f"Successfully read content for {fsmcp_path}")
                        else:
                            ai_logger.warning(f"Filesystem MCP read_file for {fsmcp_path} returned non-string: {type(read_result)}. Result: {str(read_result)[:100]}")
                            continue 
                    except (ConnectionError, McpError, Exception) as e:
                        ai_logger.error(f"Failed to read file from Filesystem MCP ({fsmcp_path}): {e}", exc_info=True)
                        raise IOError(f"Failed to resolve fsmcp_path {fsmcp_path} for notebook {self.notebook_id}: {e}") from e

                if content is not None:
                    effective_original_filename = original_filename or default_original_filename
                    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    if content_hash in staged_files_tracker:
                        staged_path = staged_files_tracker[content_hash]
                        ai_logger.info(f"Reusing already staged file for content hash {content_hash[:8]}: {staged_path} (original: {effective_original_filename})")
                    else:
                        try:
                            staged_path = await stage_file_content(
                                content=content,
                                original_filename=effective_original_filename,
                                notebook_id=self.notebook_id,
                                session_id=session_id,
                                source_cell_id=source_cell_id
                            )
                            staged_files_tracker[content_hash] = staged_path
                            ai_logger.info(f"Successfully staged content for {effective_original_filename} to {staged_path}")
                        except Exception as e:
                            ai_logger.error(f"Failed to stage file content (source: {effective_original_filename}): {e}", exc_info=True)
                            continue 
                    
                    resolved_refs.append(FileDataRef(
                        source_cell_id=source_cell_id,
                        type="local_staged_path", 
                        value=staged_path, 
                        original_filename=effective_original_filename
                    ))
                elif fsmcp_path and content is None:
                    ai_logger.warning(f"No content obtained for fsmcp_path {fsmcp_path}, cannot stage.")
        return resolved_refs
    
    async def process_step(
        self,
        step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, StepResult], # Changed Any to StepResult
        plan_step_id_to_cell_ids: Dict[str, List[UUID]],
        cell_tools: NotebookCellTools,
        session_id: str,
        db: AsyncSession # Added db parameter
    ) -> AsyncGenerator[Union[BaseEvent, StepExecutionCompleteEvent], None]:
        """Process a single investigation step and yield event updates"""
        step_type_value = step.step_type.value if step.step_type else "Unknown"
        ai_logger.info(f"<<<<< Processing step {step.step_id} ({step_type_value}) for session {session_id} >>>>>")
        ai_logger.info(f"Step details: {step.model_dump_json(indent=2)}")

        if not step.step_type:
            error_msg = f"Step {step.step_id} is missing step_type"
            ai_logger.error(error_msg)
            yield StepErrorEvent(
                step_id=step.step_id,
                error=error_msg,
                agent_type=AgentType.UNKNOWN,
                step_type=None, # Kept as None since step.step_type is None
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
        ai_logger.info(f"Step {step.step_id} context: {context}")
        
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
            context["findings_text"] = self._format_findings_for_report(step, executed_steps, step_results)
            for exec_step in executed_steps.values():
                if 'original_query' in exec_step:
                    context["original_query"] = exec_step["original_query"]
                    break
            
            report_error = None
            report_result = None
            
            generator = step_agent.execute(step, context.get("prompt", ""), session_id, context, db=db)
            async for event in generator:
                if isinstance(event, BaseEvent):
                    yield event
                elif isinstance(event, dict) and event.get("type") == "final_step_result":
                    report_result = event.get("result")
                    report_error = event.get("error")
                
            if report_result:
                dependency_cell_ids = [cid for dep_id in step.dependencies for cid in plan_step_id_to_cell_ids.get(dep_id, [])]
                created_cell_id, cell_params, cell_error = await self.cell_creator.create_report_cell(
                    cell_tools=cell_tools, step=step, report=report_result,
                    dependency_cell_ids=dependency_cell_ids, session_id=session_id
                )
                if cell_error:
                    yield SummaryCellErrorEvent(
                        error=cell_error, cell_params=cell_params.model_dump() if cell_params else None,
                        session_id=session_id, notebook_id=self.notebook_id
                    )
                    step_result.add_error(cell_error)
                else:
                    yield SummaryCellCreatedEvent(
                        cell_params=cell_params.model_dump() if cell_params else {},
                        cell_id=str(created_cell_id) if created_cell_id else None,
                        error=report_error, session_id=session_id, notebook_id=self.notebook_id
                    )
                if created_cell_id:
                    plan_step_id_to_cell_ids.setdefault(step.step_id, []).append(created_cell_id)
                    step_result.add_cell_id(created_cell_id)
                final_result_data = report_result
                step_result.add_output(report_result)
                if report_error:
                    step_result.add_error(report_error)
            else:
                error_msg = "Report generation failed to produce a report"
                step_result.add_error(error_msg)
            
            step_results[step.step_id] = step_result
            yield StepExecutionCompleteEvent(
                step_id=step.step_id, final_result_data=final_result_data,
                step_outputs=step_result.outputs, step_error=step_result.get_combined_error(),
                github_step_has_tool_errors=False, filesystem_step_has_tool_errors=False,
                python_step_has_tool_errors=False, session_id=session_id, notebook_id=self.notebook_id
            )
            return
        
        agent_type = step_agent.get_agent_type()
        agent_input_data: Union[str, PythonAgentInput] = context.get("prompt", "")
        if isinstance(agent_input_data, str):
            ai_logger.info(f"Step {step.step_id} agent type: {agent_type}, initial agent_input_data (prompt): {agent_input_data[:500]}...")
        else: # It's PythonAgentInput
            ai_logger.info(f"Step {step.step_id} agent type: {agent_type}, initial agent_input_data type: PythonAgentInput (details logged separately if Python step)")

        if step.step_type == StepType.PYTHON:
            ai_logger.info(f"Preparing data references for Python step {step.step_id}")
            try:
                resolved_data_refs = await self._prepare_data_references(step, step_results, session_id)
                query_for_python_agent = step.description
                if not query_for_python_agent:
                    query_for_python_agent = "# No specific description provided for Python step."
                    ai_logger.warning(f"Python step {step.step_id} has no description. Using default query.")
                
                python_agent_input_obj = PythonAgentInput(
                    user_query=query_for_python_agent, # Renamed 'query' to 'user_query'
                    data_references=resolved_data_refs,
                    notebook_id=self.notebook_id, # Added notebook_id
                    session_id=session_id # Added session_id
                )
                agent_input_data = python_agent_input_obj
                ai_logger.info(f"PythonAgentInput prepared for step {step.step_id} with {len(resolved_data_refs)} data refs.")
                ai_logger.info(f"Step {step.step_id} Python agent input: {python_agent_input_obj.model_dump_json(indent=2)}")
            except (IOError, ConnectionError, McpError) as e:
                error_msg = f"Critical error preparing data references for Python step {step.step_id}: {e}"
                ai_logger.error(error_msg, exc_info=True)
                yield StepErrorEvent(
                    step_id=step.step_id, error=error_msg, agent_type=AgentType.PYTHON,
                    step_type=StepType.PYTHON, session_id=session_id, notebook_id=self.notebook_id,
                    cell_id=None, cell_params=None, result=None
                )
                step_result.add_error(error_msg)
                step_results[step.step_id] = step_result
                yield StepExecutionCompleteEvent(
                    step_id=step.step_id, final_result_data=None, step_outputs=step_result.outputs,
                    step_error=step_result.get_combined_error(), github_step_has_tool_errors=False,
                    filesystem_step_has_tool_errors=False, python_step_has_tool_errors=True,
                    session_id=session_id, notebook_id=self.notebook_id
                )
                return
            except Exception as e:
                error_msg = f"Unexpected error preparing data references for Python step {step.step_id}: {e}\n{traceback.format_exc()}"
                ai_logger.error(error_msg, exc_info=True)
                yield StepErrorEvent(
                    step_id=step.step_id, error=error_msg, agent_type=AgentType.PYTHON,
                    step_type=StepType.PYTHON, session_id=session_id, notebook_id=self.notebook_id,
                    cell_id=None, cell_params=None, result=None
                )
                step_result.add_error(error_msg)
                step_results[step.step_id] = step_result
                yield StepExecutionCompleteEvent(
                    step_id=step.step_id, final_result_data=None, step_outputs=step_result.outputs,
                    step_error=step_result.get_combined_error(), github_step_has_tool_errors=False,
                    filesystem_step_has_tool_errors=False, python_step_has_tool_errors=True,
                    session_id=session_id, notebook_id=self.notebook_id
                )
                return

        generator = step_agent.execute(step, agent_input_data, session_id, context, db=db)
        async for event in generator:
            if isinstance(event, StatusUpdateEvent):
                ai_logger.info(f"Step {step.step_id} yielding StatusUpdateEvent: {event.status} - {event.message}")
                if not event.step_id:
                    yield StatusUpdateEvent(
                        status=event.status, agent_type=event.agent_type, message=event.message,
                        attempt=event.attempt, max_attempts=event.max_attempts, reason=event.reason,
                        step_id=step.step_id, original_plan_step_id=event.original_plan_step_id,
                        session_id=session_id, notebook_id=self.notebook_id
                    )
                else:
                    yield event
            elif isinstance(event, dict) and event.get("type") == "final_step_result" and step.step_type not in [StepType.GITHUB, StepType.FILESYSTEM, StepType.PYTHON]:
                final_result_data = event.get("result")
                ai_logger.info(f"Step {step.step_id} received final_step_result (non-tool based): {type(final_result_data)}")
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
                ai_logger.info(f"Step {step.step_id} received ToolSuccessEvent: {event.tool_name}")
                created_event_map = {
                    StepType.GITHUB: GitHubToolCellCreatedEvent,
                    StepType.FILESYSTEM: FileSystemToolCellCreatedEvent,
                    StepType.PYTHON: PythonToolCellCreatedEvent,
                }
                created_event_class = created_event_map.get(step.step_type)
                if created_event_class:
                    _, tool_event = await self._process_tool_success_event(
                        event=event, step=step, step_type=step.step_type, agent_type=agent_type,
                        plan_step_id_to_cell_ids=plan_step_id_to_cell_ids, cell_tools=cell_tools,
                        session_id=session_id, created_event_class=created_event_class,
                        step_result=step_result
                    )
                    yield tool_event
            elif isinstance(event, ToolErrorEvent):
                error_event_map = {
                    StepType.GITHUB: GitHubToolErrorEvent,
                    StepType.FILESYSTEM: FileSystemToolErrorEvent,
                    StepType.PYTHON: PythonToolErrorEvent,
                }
                error_event_class = error_event_map.get(step.step_type)
                if error_event_class:
                    ai_logger.info(f"Step {step.step_id} yielding {error_event_class.__name__} for tool {event.tool_name} error: {event.error}")
                    yield error_event_class(
                        original_plan_step_id=step.step_id, tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name, tool_args=event.tool_args, error=event.error,
                        session_id=session_id, notebook_id=self.notebook_id
                    )
                if step.step_type == StepType.FILESYSTEM: filesystem_step_has_tool_errors = True
                if step.step_type == StepType.PYTHON: python_step_has_tool_errors = True
                step_result.add_error(f"Error in tool {event.tool_name}: {event.error}")
            elif isinstance(event, BaseEvent):
                ai_logger.info(f"Step {step.step_id} yielding generic BaseEvent: {type(event)}")
                yield event
        
        if step.step_type == StepType.MARKDOWN and final_result_data:
            dependency_cell_ids = [cid for dep_id in step.dependencies for cid in plan_step_id_to_cell_ids.get(dep_id, [])]
            created_cell_id, cell_params, cell_error = await self.cell_creator.create_markdown_cell(
                cell_tools=cell_tools, step=step, result=final_result_data,
                dependency_cell_ids=dependency_cell_ids, session_id=session_id
            )
            if cell_error:
                step_result.add_error(cell_error)
            if created_cell_id:
                plan_step_id_to_cell_ids.setdefault(step.step_id, []).append(created_cell_id)
                step_result.add_cell_id(created_cell_id)
            
            if step_result.has_error():
                yield StepErrorEvent(
                    step_id=step.step_id,
                    step_type=step.step_type.value,
                    cell_id=str(created_cell_id) if created_cell_id else None,
                    cell_params=cell_params.model_dump() if cell_params else None,
                    result=None, # Or a minimal representation if appropriate for errors
                    error=step_result.get_combined_error(),
                    agent_type=agent_type,
                    session_id=session_id,
                    notebook_id=self.notebook_id
                )
            else:
                ai_logger.info(f"Step {step.step_id} (Markdown) yielding StepCompletedEvent. Cell ID: {created_cell_id}")
                yield StepCompletedEvent(
                    step_id=step.step_id,
                    step_type=step.step_type.value,
                    cell_id=str(created_cell_id) if created_cell_id else None,
                    cell_params=cell_params.model_dump() if cell_params else None,
                    result=final_result_data.model_dump() if final_result_data else None,
                    # No 'error' field for StepCompletedEvent
                    agent_type=agent_type,
                    session_id=session_id,
                    notebook_id=self.notebook_id
                )
        
        step_results[step.step_id] = step_result
        ai_logger.info(f"Step {step.step_id} execution complete. Final result data type: {type(final_result_data)}, Has error: {step_result.has_error()}")
        ai_logger.info(f"Step {step.step_id} final outputs: {step_result.outputs}")
        ai_logger.info(f"Step {step.step_id} combined error: {step_result.get_combined_error()}")
        yield StepExecutionCompleteEvent(
            step_id=step.step_id, final_result_data=final_result_data,
            step_outputs=step_result.outputs, step_error=step_result.get_combined_error(),
            github_step_has_tool_errors=github_step_has_tool_errors,
            filesystem_step_has_tool_errors=filesystem_step_has_tool_errors,
            python_step_has_tool_errors=python_step_has_tool_errors,
            session_id=session_id, notebook_id=self.notebook_id
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
        created_event_class: Type, # This should be Type[BaseEvent] or similar
        step_result: StepResult
    ) -> Tuple[Any, BaseEvent]:
        """Process a tool success event and create a cell for it"""
        dependency_cell_ids = [cid for dep_id in step.dependencies for cid in plan_step_id_to_cell_ids.get(dep_id, [])]
        
        agent_to_cell_type_map = {
            AgentType.GITHUB: CellType.GITHUB, AgentType.FILESYSTEM: CellType.FILESYSTEM,
            AgentType.PYTHON: CellType.PYTHON, AgentType.SQL: CellType.SQL,
            AgentType.S3: CellType.S3, AgentType.METRIC: CellType.METRIC
        }
        current_event_agent_type = event.agent_type if event.agent_type is not None else AgentType.UNKNOWN
        cell_type = agent_to_cell_type_map.get(current_event_agent_type, CellType.AI_QUERY)
        if cell_type == CellType.AI_QUERY and current_event_agent_type != AgentType.UNKNOWN:
             ai_logger.warning(
                f"No specific CellType mapping for AgentType '{current_event_agent_type.value}' "
                f"that executed tool '{event.tool_name}'. Defaulting cell to AI_QUERY."
            )

        created_cell_id, cell_params, cell_error = await self.cell_creator.create_tool_cell(
            cell_tools=cell_tools, step=step, cell_type=cell_type,
            agent_type=agent_type, tool_event=event,
            dependency_cell_ids=dependency_cell_ids, session_id=session_id
        )
        
        tool_name_str = event.tool_name or "unknown_tool"
        serialized_result = self.cell_creator._serialize_tool_result(event.tool_result, tool_name_str)
        step_result.add_output(serialized_result)
        
        if cell_error:
            ai_logger.warning(f"Error creating cell for {step_type.value} tool {event.tool_name}: {cell_error}")
            step_result.add_error(cell_error)
            tool_event = created_event_class(
                original_plan_step_id=step.step_id, cell_id=None,
                tool_name=event.tool_name or "", tool_args=event.tool_args or {},
                result=event.tool_result, cell_params=cell_params.model_dump() if cell_params else {},
                session_id=session_id, notebook_id=self.notebook_id
            )
            return serialized_result, tool_event
        
        if created_cell_id:
            step_result.add_cell_id(created_cell_id)
            plan_step_id_to_cell_ids.setdefault(step.step_id, []).append(created_cell_id)

        tool_event = created_event_class(
            original_plan_step_id=step.step_id,
            cell_id=str(created_cell_id) if created_cell_id else None,
            tool_name=event.tool_name, tool_args=event.tool_args or {},
            result=event.tool_result, cell_params=cell_params.model_dump() if cell_params else None,
            session_id=session_id, notebook_id=self.notebook_id
        )
        return serialized_result, tool_event
    
    def _build_step_context(
        self,
        step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, StepResult] # Ensure this is StepResult
    ) -> Dict[str, Any]:
        """Build context from dependencies for a step"""
        context = {"prompt": step.description}
        
        ai_logger.info(f"Building context for step {step.step_id}. Initial prompt: {context['prompt'][:1000]}...")
        if step.parameters:
            params_str = "\n".join([f"- {k}: {v}" for k, v in step.parameters.items()])
            context["prompt"] += f"\n\nParameters for this step:\n{params_str}"
            ai_logger.info(f"Step {step.step_id} parameters added to prompt: {params_str}")
        
        dependency_context = []
        for dep_id in step.dependencies:
            if dep_id in executed_steps:
                dep_step_info = executed_steps[dep_id]

                if not isinstance(dep_step_info, dict):
                    ai_logger.warning(f"Dependency info for {dep_id} is not a dictionary: {type(dep_step_info)}. Skipping context for this dependency.")
                    dependency_context.append(f"- Dependency '{dep_id}': Information malformed or unavailable.")
                    continue

                raw_dep_step_model_data = dep_step_info.get('step')
                dep_overall_error = dep_step_info.get('error')
                dep_result_obj = step_results.get(dep_id)

                if not raw_dep_step_model_data: # This check handles if 'step' key is missing or its value is None
                    ai_logger.warning(f"Could not find step model data for dependency ID {dep_id} in executed_steps.")
                    dependency_context.append(f"- Dependency '{dep_id}': Step model data unavailable.")
                    continue
                
                # Ensure raw_dep_step_model_data is a dict before attempting to create InvestigationStepModel from it
                if not isinstance(raw_dep_step_model_data, dict):
                    ai_logger.error(f"Step model data for dependency {dep_id} is not a dictionary: {type(raw_dep_step_model_data)}. Skipping.")
                    dependency_context.append(f"- Dependency '{dep_id}': Invalid step model data format.")
                    continue

                try:
                    dep_step_model = InvestigationStepModel(**raw_dep_step_model_data)
                except Exception as e:
                    ai_logger.error(f"Failed to parse InvestigationStepModel for dependency {dep_id} from data {raw_dep_step_model_data}: {e}")
                    dependency_context.append(f"- Dependency '{dep_id}': Error parsing step model data.")
                    continue
                    
                context_limit = 500
                context_line = f"- Dependency '{dep_id}' ({dep_step_model.step_type.value}): {dep_step_model.description[:context_limit]}..."
                
                if isinstance(dep_result_obj, StepResult) and dep_result_obj.outputs:
                    context_line += " (Outputs Available)"
                    outputs_summary_lines = []
                    for i, out_data in enumerate(dep_result_obj.outputs):
                        summary_prefix = f"  - Output {i+1}:"
                        if isinstance(out_data, dict):
                            if out_data.get("status") == "success" and "stdout" in out_data and out_data.get("stdout") is not None:
                                outputs_summary_lines.append(f"{summary_prefix} (stdout): {str(out_data['stdout'])[:context_limit].strip()}...")
                            elif "path" in out_data and "content" in out_data:
                                outputs_summary_lines.append(f"{summary_prefix} (file: {out_data['path']}): {str(out_data['content'])[:context_limit].strip()}...")
                            elif "search_results" in out_data:
                                search_results_data = out_data.get("search_results")
                                count = len(search_results_data) if isinstance(search_results_data, list) else 0
                                first_item = str(search_results_data[0])[:context_limit] if search_results_data and count > 0 else ""
                                outputs_summary_lines.append(f"{summary_prefix} ({count} search results found).{' First: ' + first_item + '...' if count > 0 else ''}")
                            else:
                                outputs_summary_lines.append(f"{summary_prefix} {str(out_data)[:context_limit].strip()}...")
                        elif isinstance(out_data, list):
                             outputs_summary_lines.append(f"{summary_prefix} (list with {len(out_data)} items). First: {str(out_data[0])[:context_limit]}..." if out_data else f"{summary_prefix} (empty list).")
                        elif isinstance(out_data, QueryResult):
                             outputs_summary_lines.append(f"{summary_prefix} {str(out_data.data)[:context_limit].strip()}...")
                        else:
                            outputs_summary_lines.append(f"{summary_prefix} {str(out_data)[:context_limit].strip()}...")
                    
                    if outputs_summary_lines: context_line += "\n" + "\n".join(outputs_summary_lines)
                    else: context_line += " (Step had outputs, but they could not be summarized for context.)"
                elif dep_overall_error:
                    context_line += f" (Failed: {str(dep_overall_error)[:context_limit]}... No usable output data for context.)"
                elif isinstance(dep_result_obj, StepResult) and dep_result_obj.has_error():
                     combined_error = dep_result_obj.get_combined_error()
                     error_snippet = str(combined_error)[:context_limit] if combined_error else "Unknown error"
                     context_line += f" (Failed: {error_snippet}... No usable output data for context.)"
                else:
                    context_line += " (Completed with no specific output data for context.)"
                dependency_context.append(context_line)
        
        if dependency_context:
            context["prompt"] += f"\n\nContext from Dependencies:\n" + "\n".join(dependency_context)
            ai_logger.info(f"Step {step.step_id} full dependency context: {' '.join(dependency_context)}")
        else:
            context["prompt"] += "\n\nContext from Dependencies:\nNo preceding steps or no relevant output from them.\n"
            ai_logger.info(f"Step {step.step_id} no dependency context to add.")
        return context
    
    def _format_findings_for_report(
        self,
        step: InvestigationStepModel,
        executed_steps: Dict[str, Any],
        step_results: Dict[str, StepResult] # Ensure this is StepResult
    ) -> str:
        """Format findings from previous steps for the report"""
        findings_text_lines = []
        ai_logger.info(f"Formatting findings for report step {step.step_id}. Executed steps count: {len(executed_steps)}")
        for step_id, step_info in executed_steps.items():
            if step_id == step.step_id: continue
            
            step_obj_data_any = step_info.get("step") # Use a different var name to avoid confusion before type checking
            step_res_obj = step_results.get(step_id) # This is StepResult
            step_error_msg = step_info.get("error") # Overall error for the step from execution loop

            desc = "N/A (description unavailable)"
            type_val = "unknown (type unavailable)"

            dep_step_model: Optional[InvestigationStepModel] = None

            if isinstance(step_obj_data_any, dict):
                try:
                    dep_step_model = InvestigationStepModel(**step_obj_data_any)
                except Exception as e:
                    ai_logger.error(f"Failed to parse InvestigationStepModel for step_id '{step_id}' in _format_findings_for_report from dict: {e}")
            elif isinstance(step_obj_data_any, InvestigationStepModel):
                dep_step_model = step_obj_data_any
            elif step_obj_data_any is None:
                ai_logger.warning(f"Step data for step_id '{step_id}' is None in _format_findings_for_report.")
            else:
                ai_logger.warning(f"Unexpected type for step_obj_data for step_id '{step_id}': {type(step_obj_data_any)} in _format_findings_for_report.")

            if dep_step_model:
                desc = dep_step_model.description
                if isinstance(dep_step_model.step_type, StepType):
                    type_val = dep_step_model.step_type.value
                elif dep_step_model.step_type: # Should be StepType, but handle if it's a raw string
                    type_val = str(dep_step_model.step_type)
                else:
                    type_val = "unknown (step_type missing)"
            
            result_str = ""
            if step_error_msg: # Overall error from main loop takes precedence
                result_str = f"Error: {step_error_msg}"
            elif isinstance(step_res_obj, StepResult):
                if step_res_obj.has_error():
                    result_str = f"Error: {step_res_obj.get_combined_error()}"
                elif step_res_obj.outputs:
                    if step_res_obj.step_type == StepType.MARKDOWN and step_res_obj.outputs:
                        output = step_res_obj.outputs[0]
                        data_to_show = output.data if isinstance(output, QueryResult) else output
                        result_str = f"Data: {str(data_to_show)[:500]}..."
                    else:
                        formatted_outputs = [f"  - Output {i+1}: {str(output)[:200]}..." for i, output in enumerate(step_res_obj.outputs)]
                        result_str = "Data (multiple outputs):\n" + "\n".join(formatted_outputs) if formatted_outputs else "Step completed, but no specific outputs captured."
                else:
                    result_str = "Step completed, but no outputs available in StepResult."
            # Removed legacy QueryResult/list handling as step_results now consistently holds StepResult
            else:
                result_str = "Step completed without explicit data/error capture in StepResult."
            
            findings_text_lines.append(f"Step {step_id} ({type_val}): {desc}\nResult:\n{result_str}")
        
        return "\n---\n".join(findings_text_lines) if findings_text_lines else "No preceding investigation steps were successfully executed or produced results."
