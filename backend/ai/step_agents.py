"""
Agent implementations for different step types.
"""

import logging
from typing import Any, Dict, AsyncGenerator, Union
from abc import ABC, abstractmethod

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from backend.ai.models import InvestigationStepModel
from backend.ai.events import (
    BaseEvent, AgentType, StatusType, StatusUpdateEvent,
    ToolSuccessEvent, ToolErrorEvent, FinalStatusEvent, FatalErrorEvent, # Added FatalErrorEvent
    GitHubStepProcessingCompleteEvent, FileSystemStepProcessingCompleteEvent,
    PythonStepProcessingCompleteEvent # Added
)
from backend.core.query_result import MarkdownQueryResult, InvestigationReport, Finding
from backend.ai.github_query_agent import GitHubQueryAgent
from backend.ai.investigation_report_agent import InvestigationReportAgent
from backend.ai.filesystem_agent import FileSystemAgent
from backend.ai.python_agent import PythonAgent
from backend.ai.models import PythonAgentInput, FileDataRef # Added for PythonStepAgent
from backend.ai.prompts.investigation_prompts import MARKDOWN_GENERATOR_SYSTEM_PROMPT
from backend.config import get_settings

ai_logger = logging.getLogger("ai")

class StepAgent(ABC):
    """Base class for step-specific agent implementations"""
    def __init__(self, notebook_id: str):
        self.notebook_id = notebook_id
        self.settings = get_settings()
    
    @abstractmethod
    async def execute(self, 
                      step: InvestigationStepModel, 
                      agent_input_data: Union[str, PythonAgentInput], # Changed from prompt: str
                      session_id: str, 
                      context: Dict[str, Any]) -> AsyncGenerator[Union[BaseEvent, Dict[str, Any]], None]:
        """Execute the step and yield events/results."""
        pass
    
    @abstractmethod
    def get_agent_type(self) -> AgentType:
        """Return the agent type for this step agent."""
        pass


class MarkdownStepAgent(StepAgent):
    """Agent for executing markdown steps"""
    def __init__(self, notebook_id: str, model: OpenAIModel):
        super().__init__(notebook_id)
        self.model = model
        self.markdown_generator = Agent(
            self.model,
            output_type=MarkdownQueryResult,
            system_prompt=MARKDOWN_GENERATOR_SYSTEM_PROMPT
        )
    
    def get_agent_type(self) -> AgentType:
        return AgentType.MARKDOWN
    
    async def execute(self, 
                     step: InvestigationStepModel, 
                     agent_input_data: Union[str, PythonAgentInput], # Changed from prompt
                     session_id: str, 
                     context: Dict[str, Any]) -> AsyncGenerator[Union[BaseEvent, Dict[str, Any]], None]:
        """Generate markdown content for the step"""
        if not isinstance(agent_input_data, str):
            # This case should ideally not happen for MarkdownStepAgent
            # Or handle it by converting/logging if necessary
            raise TypeError(f"MarkdownStepAgent expects a string prompt, got {type(agent_input_data)}")
        prompt = agent_input_data

        yield StatusUpdateEvent(
            status=StatusType.AGENT_ITERATING, 
            agent_type=self.get_agent_type(), 
            message="Generating Markdown...", 
            attempt=None, 
            max_attempts=None, 
            reason=None, 
            step_id=step.step_id, 
            original_plan_step_id=None, 
            session_id=session_id, 
            notebook_id=self.notebook_id
        )
        
        result = await self.markdown_generator.run(prompt)
        if isinstance(result.output, MarkdownQueryResult):
            final_result = result.output
        else:
            ai_logger.warning(f"Markdown generator did not return MarkdownQueryResult. Got: {type(result.output)}. Falling back.")
            fallback_content = str(result.output) if result.output else ''
            final_result = MarkdownQueryResult(query=step.description, data=fallback_content)
            
        yield {"type": "final_step_result", "result": final_result}


class GitHubStepAgent(StepAgent):
    """Agent for executing GitHub steps"""
    def __init__(self, notebook_id: str):
        super().__init__(notebook_id)
        self.github_generator = None  # Lazily initialized
    
    def get_agent_type(self) -> AgentType:
        return AgentType.GITHUB
    
    async def execute(self, 
                     step: InvestigationStepModel, 
                     agent_input_data: Union[str, PythonAgentInput], # Changed from prompt
                     session_id: str, 
                     context: Dict[str, Any]) -> AsyncGenerator[Union[BaseEvent, Dict[str, Any]], None]:
        """Execute a GitHub query step"""
        if not isinstance(agent_input_data, str):
            raise TypeError(f"GitHubStepAgent expects a string prompt, got {type(agent_input_data)}")
        prompt = agent_input_data

        # Lazy initialization
        if self.github_generator is None:
            ai_logger.info("Lazily initializing GitHubQueryAgent.")
            self.github_generator = GitHubQueryAgent(notebook_id=self.notebook_id)
        
        ai_logger.info(f"Processing GitHub step {step.step_id} using run_query generator...")
        
        # Forward events from the GitHub agent
        async for event_data in self.github_generator.run_query(
            prompt,
            session_id=session_id,
            notebook_id=self.notebook_id
        ):
            # Forward specific event types
            if isinstance(event_data, (StatusUpdateEvent, ToolSuccessEvent, ToolErrorEvent)):
                yield event_data
            elif isinstance(event_data, FinalStatusEvent):
                ai_logger.info(f"GitHub query finished for step {step.step_id} with status: {event_data.status}")
            elif isinstance(event_data, BaseEvent):
                ai_logger.warning(f"Unexpected BaseEvent type {type(event_data)} from GitHub agent: {event_data!r}")
            else:
                ai_logger.warning(f"Unexpected non-BaseEvent type {type(event_data)} from GitHub agent: {event_data!r}")
        
        # Signal completion of this step
        yield GitHubStepProcessingCompleteEvent(
            original_plan_step_id=step.step_id,
            session_id=session_id,
            notebook_id=self.notebook_id
        )


class FileSystemStepAgent(StepAgent):
    """Agent for executing filesystem steps"""
    def __init__(self, notebook_id: str):
        super().__init__(notebook_id)
        self.filesystem_generator = None  # Lazily initialized
    
    def get_agent_type(self) -> AgentType:
        return AgentType.FILESYSTEM
    
    async def execute(self, 
                     step: InvestigationStepModel, 
                     agent_input_data: Union[str, PythonAgentInput], # Changed from prompt
                     session_id: str, 
                     context: Dict[str, Any]) -> AsyncGenerator[Union[BaseEvent, Dict[str, Any]], None]:
        """Execute a filesystem query step"""
        if not isinstance(agent_input_data, str):
            raise TypeError(f"FileSystemStepAgent expects a string prompt, got {type(agent_input_data)}")
        prompt = agent_input_data
        
        # Lazy initialization
        if self.filesystem_generator is None:
            ai_logger.info("Lazily initializing FileSystemAgent.")
            self.filesystem_generator = FileSystemAgent(notebook_id=self.notebook_id)
        
        ai_logger.info(f"Processing Filesystem step {step.step_id} using run_query generator...")
        
        # Forward events from the filesystem agent
        async for event_data in self.filesystem_generator.run_query(
            prompt,
            session_id=session_id,
            notebook_id=self.notebook_id
        ):
            # Forward relevant events
            if isinstance(event_data, (ToolSuccessEvent, ToolErrorEvent, StatusUpdateEvent)):
                yield event_data
            elif isinstance(event_data, FinalStatusEvent):
                ai_logger.info(f"FileSystem query finished for step {step.step_id} with status: {event_data.status}")
            elif isinstance(event_data, BaseEvent):
                ai_logger.warning(f"Unexpected BaseEvent type {type(event_data)} from Filesystem agent: {event_data!r}")
            else:
                ai_logger.warning(f"Unexpected non-BaseEvent type {type(event_data)} from Filesystem agent: {event_data!r}")
        
        # Signal completion of this step
        yield FileSystemStepProcessingCompleteEvent(
            original_plan_step_id=step.step_id,
            session_id=session_id,
            notebook_id=self.notebook_id
        )


class PythonStepAgent(StepAgent):
    """Agent for executing Python steps"""
    def __init__(self, notebook_id: str):
        super().__init__(notebook_id)
        self.python_generator = None  # Lazily initialized
    
    def get_agent_type(self) -> AgentType:
        return AgentType.PYTHON
    
    async def execute(self, 
                     step: InvestigationStepModel, 
                     agent_input_data: Union[str, PythonAgentInput], # Changed parameter name
                     session_id: str, # session_id is available here but PythonAgent.run_query might not need it if it's in agent_input_data
                     context: Dict[str, Any]) -> AsyncGenerator[Union[BaseEvent, Dict[str, Any]], None]:
        """Execute a Python query step"""
        if not isinstance(agent_input_data, PythonAgentInput):
            # This case should ideally not happen if step_processor correctly sets agent_input_data
            raise TypeError(f"PythonStepAgent expects PythonAgentInput, got {type(agent_input_data)}")
        
        # Lazy initialization
        if self.python_generator is None:
            ai_logger.info("Lazily initializing PythonAgent for Python steps.")
            self.python_generator = PythonAgent(notebook_id=self.notebook_id) # notebook_id is from self
        
        ai_logger.info(f"Processing Python step {step.step_id} using run_query generator...")
        
        # Forward events from the Python agent
        # PythonAgent.run_query should expect PythonAgentInput which contains notebook_id and session_id
        async for event_data in self.python_generator.run_query(agent_input=agent_input_data):
            # Forward relevant events
            if isinstance(event_data, (ToolSuccessEvent, ToolErrorEvent, StatusUpdateEvent, FatalErrorEvent)):
                yield event_data
            elif isinstance(event_data, FinalStatusEvent):
                ai_logger.info(f"Python query finished for step {step.step_id} with status: {event_data.status}")
            elif isinstance(event_data, BaseEvent):
                ai_logger.warning(f"Unexpected BaseEvent type {type(event_data)} from Python agent: {event_data!r}")
            else:
                ai_logger.warning(f"Unexpected non-BaseEvent type {type(event_data)} from Python agent: {event_data!r}")
        
        # Signal completion of this step
        yield PythonStepProcessingCompleteEvent( # Corrected event
            original_plan_step_id=step.step_id,
            session_id=session_id,
            notebook_id=self.notebook_id
        )


class ReportStepAgent(StepAgent):
    """Agent for generating investigation reports"""
    def __init__(self, notebook_id: str):
        super().__init__(notebook_id)
        self.report_generator = None  # Lazily initialized
    
    def get_agent_type(self) -> AgentType:
        return AgentType.INVESTIGATION_REPORT_GENERATOR
    
    async def execute(self, 
                     step: InvestigationStepModel, 
                     agent_input_data: Union[str, PythonAgentInput], # Changed from prompt
                     session_id: str, 
                     context: Dict[str, Any]) -> AsyncGenerator[Union[BaseEvent, Dict[str, Any]], None]:
        """Generate an investigation report"""
        if not isinstance(agent_input_data, str):
            raise TypeError(f"ReportStepAgent expects a string prompt, got {type(agent_input_data)}")
        # prompt = agent_input_data # Not directly used by report_generator.run_report_generation

        # Extract required context
        findings_text = context.get("findings_text", "")
        original_query = context.get("original_query", "") # This seems to be the main "prompt" for the report
        
        # Lazy initialization
        if self.report_generator is None:
            ai_logger.info("Lazily initializing InvestigationReportAgent.")
            self.report_generator = InvestigationReportAgent(notebook_id=self.notebook_id)
        
        ai_logger.info(f"Processing Report step {step.step_id}...")
        final_report_object = None
        report_error = None
        
        try:
            async for result_part in self.report_generator.run_report_generation(
                original_query=original_query,
                findings_summary=findings_text,
                session_id=session_id,
                notebook_id=self.notebook_id
            ):
                if isinstance(result_part, InvestigationReport):
                    final_report_object = result_part
                    report_error = final_report_object.error
                    yield StatusUpdateEvent(
                        status=StatusType.SUCCESS,
                        message="Report object generated",
                        agent_type=self.get_agent_type(),
                        attempt=None,
                        max_attempts=None,
                        reason=None,
                        step_id=step.step_id,
                        original_plan_step_id=None,
                        session_id=session_id,
                        notebook_id=self.notebook_id
                    )
                    break
                elif isinstance(result_part, StatusUpdateEvent):
                    # Forward status updates
                    if result_part.session_id is None:
                        result_part.session_id = session_id
                    if result_part.notebook_id is None:
                        result_part.notebook_id = self.notebook_id
                    yield result_part
                else:
                    ai_logger.warning(f"Unexpected item from report generator: {type(result_part)}")
            
            if final_report_object is None:
                report_error = "Report agent did not return a final report object."
                ai_logger.error(report_error)
                # Create a placeholder error report
                final_report_object = self._create_error_report(original_query, report_error)
                
        except Exception as e:
            report_error = f"Error during report generation: {e}"
            ai_logger.error(report_error, exc_info=True)
            # Create a placeholder error report
            final_report_object = self._create_error_report(original_query, report_error)
        
        yield {"type": "final_step_result", "result": final_report_object, "error": report_error}
    
    def _create_error_report(self, query: str, error_message: str) -> InvestigationReport:
        """Create a placeholder error report when generation fails"""
        return InvestigationReport(
            query=query,
            title="Report Generation Failed",
            issue_summary=Finding(summary="Generation Error", details=None, code_reference=None, supporting_quotes=None),
            root_cause=Finding(summary="Generation Error", details=None, code_reference=None, supporting_quotes=None),
            error=error_message,
            status=None,
            status_reason="Generation Error",
            estimated_severity=None,
            root_cause_confidence=None,
            proposed_fix=None,
            proposed_fix_confidence=None,
            affected_context=None,
            key_components=[],
            related_items=[],
            suggested_next_steps=[],
            tags=[]
        )
