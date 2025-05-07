"""
Pydantic Models for Agent Communication Events
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

class BaseEvent(BaseModel):
    """Base class for all events to potentially hold common fields if needed later."""
    session_id: Optional[str] = Field(None, description="Session ID associated with the event")
    notebook_id: Optional[str] = Field(None, description="Notebook ID associated with the event")

class EventType(str, Enum):
    STATUS_UPDATE = "status_update"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_SUCCESS = "tool_success"
    TOOL_ERROR = "tool_error"
    FINAL_STATUS = "final_status"
    FATAL_ERROR = "fatal_error"
    PLAN_CREATED = "plan_created"
    PLAN_CELL_CREATED = "plan_cell_created"
    STEP_STARTED = "step_started"
    STEP_EXECUTION_COMPLETE = "step_execution_complete" # Internal
    STEP_COMPLETED = "step_completed"
    STEP_ERROR = "step_error"
    GITHUB_TOOL_CELL_CREATED = "github_tool_cell_created"
    GITHUB_TOOL_ERROR = "github_tool_error"
    SUMMARY_STARTED = "summary_started"
    SUMMARY_UPDATE = "summary_update"
    SUMMARY_CELL_CREATED = "summary_cell_created"
    SUMMARY_CELL_ERROR = "summary_cell_error"
    INVESTIGATION_COMPLETE = "investigation_complete"
    CLARIFICATION_NEEDED = "clarification"
    REPORT_GENERATION_STARTED = "report_generation_started"
    REPORT_GENERATION_UPDATE = "report_generation_update"
    REPORT_CELL_CREATED = "report_cell_created"
    REPORT_CELL_ERROR = "report_cell_error"
    GITHUB_STEP_PROCESSING_COMPLETE = "github_step_processing_complete"
    FILESYSTEM_TOOL_CELL_CREATED = "filesystem_tool_cell_created"
    FILESYSTEM_TOOL_ERROR = "filesystem_tool_error"
    FILESYSTEM_STEP_PROCESSING_COMPLETE = "filesystem_step_processing_complete"
    # Python Agent Specific
    PYTHON_TOOL_CELL_CREATED = "python_tool_cell_created"
    PYTHON_TOOL_ERROR = "python_tool_error"
    PYTHON_STEP_PROCESSING_COMPLETE = "python_step_processing_complete" # Signal agent finished

class AgentType(str, Enum):
    GITHUB = "github_agent"
    SUMMARIZATION = "summarization_generator"
    INVESTIGATION_PLANNER = "investigation_planner"
    CHAT = "chat_agent"
    MARKDOWN = "markdown" # Used as agent type for markdown steps
    INVESTIGATION_REPORTER = "investigation_reporter"
    UNKNOWN = "unknown"
    FILESYSTEM = "filesystem"
    PLANNER = "planner"
    PYTHON = "python"
    PLAN_REVISER = "plan_reviser"
    INVESTIGATION_REPORT_GENERATOR = "investigation_report_generator"
    SQL = "sql"
    S3 = "s3"
    METRIC = "metric"
class StatusType(str, Enum):
    # General Statuses
    STARTING = "starting"
    AGENT_ITERATING = "agent_iterating"
    CONNECTION_READY = "connection_ready"
    AGENT_CREATED = "agent_created"
    STARTING_ATTEMPTS = "starting_attempts"
    ATTEMPT_START = "attempt_start"
    MCP_CONNECTION_ACTIVE = "mcp_connection_active"
    AGENT_RUN_COMPLETE = "agent_run_complete"
    RETRYING = "retrying"
    COMPLETE = "complete"
    SUCCESS = "success"
    ERROR = "error"
    STEP_PROCESSING_COMPLETE = "step_processing_complete"
    STEP_PROCESSING_ERROR = "step_processing_error"
    STEP_PROCESSING_STARTED = "step_processing_started"
    
    # Tool Statuses
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_SUCCESS = "tool_success"
    MCP_ERROR = "mcp_error" # Specific tool error
    MODEL_ERROR = "model_error" # Error from the LLM behavior
    TIMESTAMP_ERROR = "timestamp_error" # Specific error during run
    GENERAL_ERROR_RUN = "general_error_run" # Other errors during run
    FATAL_MCP_ERROR = "fatal_mcp_error"
    FILE_PREPARATION_ERROR = "file_preparation_error" # Error during local file staging
    
    # Final Statuses
    FINISHED_SUCCESS = "finished_success"
    FINISHED_ERROR = "finished_error"
    FINISHED_NO_RESULT = "finished_no_result"
    FATAL_ERROR = "fatal_error"
    
    # Investigation Plan Statuses
    PLAN_CREATED = "plan_created"
    PLAN_CELL_CREATED = "plan_cell_created"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed" # Also maps to SUCCESS
    STEP_ERROR = "step_error" # Also maps to ERROR
    GITHUB_TOOL_CELL_CREATED = "github_tool_cell_created" # Also maps to SUCCESS
    GITHUB_TOOL_ERROR = "github_tool_error" # Also maps to ERROR
    
    # Summarization Statuses
    SUMMARY_STARTED = "summary_started"
    SUMMARY_UPDATE = "summary_update"
    SUMMARY_CELL_CREATED = "summary_cell_created"
    SUMMARY_CELL_ERROR = "summary_cell_error"
    
    # Chat Statuses
    CLARIFICATION_NEEDED = "clarification_needed"
    

# --- Event Models (using Enums) ---

class StatusUpdateEvent(BaseEvent):
    type: EventType = Field(default=EventType.STATUS_UPDATE, description="Event type identifier")
    status: StatusType = Field(..., description="Specific status code")
    agent_type: Optional[AgentType] = Field(None, description="Identifier for the agent generating the event")
    message: Optional[str] = Field(None, description="Human-readable status message")
    attempt: Optional[int] = Field(None, description="Attempt number, if applicable")
    max_attempts: Optional[int] = Field(None, description="Maximum attempts, if applicable")
    reason: Optional[str] = Field(None, description="Reason for status (e.g., 'retrying')")
    step_id: Optional[str] = Field(None, description="Associated plan step ID, if applicable")
    original_plan_step_id: Optional[str] = Field(None, description="Original plan step ID, for events forwarded from sub-agents")


class ToolCallRequestedEvent(BaseEvent):
    type: EventType = Field(default=EventType.TOOL_CALL_REQUESTED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.TOOL_CALL_REQUESTED, description="Fixed status for this event type")
    agent_type: Optional[AgentType] = Field(None, description="Agent requesting the tool call")
    attempt: Optional[int] = Field(None, description="Attempt number, if applicable")
    tool_call_id: Optional[str] = Field(..., description="Unique ID for the tool call") # Made Optional if sometimes unavailable early
    tool_name: Optional[str] = Field(..., description="Name of the tool being called") # Made Optional if sometimes unavailable early
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments passed to the tool")
    original_plan_step_id: Optional[str] = Field(None, description="Original plan step ID")


class ToolSuccessEvent(BaseEvent):
    type: EventType = Field(default=EventType.TOOL_SUCCESS, description="Event type identifier")
    status: StatusType = Field(default=StatusType.TOOL_SUCCESS, description="Fixed status for this event type")
    agent_type: Optional[AgentType] = Field(None, description="Agent reporting tool success")
    attempt: Optional[int] = Field(None, description="Attempt number, if applicable")
    tool_call_id: Optional[str] = Field(..., description="Unique ID of the completed tool call")
    tool_name: Optional[str] = Field(..., description="Name of the successful tool")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments passed to the tool")
    tool_result: Any = Field(..., description="Result returned by the tool")
    original_plan_step_id: Optional[str] = Field(None, description="Original plan step ID")


class ToolErrorEvent(BaseEvent):
    type: EventType = Field(default=EventType.TOOL_ERROR, description="Event type identifier")
    status: StatusType = Field(..., description="Specific error status")
    agent_type: Optional[AgentType] = Field(None, description="Agent reporting the error")
    attempt: Optional[int] = Field(None, description="Attempt number, if applicable")
    tool_call_id: Optional[str] = Field(None, description="ID of the failed tool call, if available")
    tool_name: Optional[str] = Field(None, description="Name of the failed tool, if available")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments passed to the failed tool, if available")
    error: str = Field(..., description="Details of the error that occurred")
    message: Optional[str] = Field(None, description="Optional human-readable error message")
    original_plan_step_id: Optional[str] = Field(None, description="Original plan step ID")


class GitHubStepProcessingCompleteEvent(BaseEvent):
    type: EventType = Field(default=EventType.GITHUB_STEP_PROCESSING_COMPLETE, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUCCESS, description="Fixed status for this event type") # Indicate success
    agent_type: AgentType = Field(default=AgentType.GITHUB, description="Fixed agent type")
    original_plan_step_id: Optional[str] = Field(..., description="Original plan step ID being marked as complete")


class FinalStatusEvent(BaseEvent):
    type: EventType = Field(default=EventType.FINAL_STATUS, description="Event type identifier")
    status: StatusType = Field(..., description="Final status code")
    agent_type: Optional[AgentType] = Field(None, description="Agent reporting the final status")
    attempts: Optional[int] = Field(None, description="Number of attempts made")
    message: Optional[str] = Field(None, description="Optional final message")


class FatalErrorEvent(BaseEvent):
    type: EventType = Field(default=EventType.FATAL_ERROR, description="Event type identifier")
    status: StatusType = Field(default=StatusType.FATAL_ERROR, description="Fixed status for this event type")
    agent_type: Optional[AgentType] = Field(None, description="Agent reporting the fatal error")
    error: str = Field(..., description="Details of the fatal error")

# --- Events specific to Investigation Agent ---
# (These were already yielded as dicts, converting them too)

class PlanCreatedEvent(BaseEvent):
    type: EventType = Field(default=EventType.PLAN_CREATED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.PLAN_CREATED, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.INVESTIGATION_PLANNER, description="Fixed agent type")
    thinking: Optional[str] = Field(None, description="AI reasoning for the plan")
    # Maybe add plan details later if needed


class PlanCellCreatedEvent(BaseEvent):
    type: EventType = Field(default=EventType.PLAN_CELL_CREATED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.PLAN_CELL_CREATED, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.INVESTIGATION_PLANNER, description="Fixed agent type")
    cell_id: Optional[str] = Field(None, description="ID of the created plan cell")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used to create the cell")


class StepStartedEvent(BaseEvent):
    type: EventType = Field(default=EventType.STEP_STARTED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.STEP_STARTED, description="Fixed status")
    agent_type: Optional[AgentType] = Field(None, description="Agent type responsible for the step")
    step_id: str = Field(..., description="ID of the plan step being started")


class StepExecutionCompleteEvent(BaseEvent):
    # This is yielded internally within _handle_step_execution
    # Might not need to be yielded externally, but defining for consistency
    type: EventType = Field(default=EventType.STEP_EXECUTION_COMPLETE, description="Internal event type identifier")
    step_id: str = Field(..., description="ID of the completed plan step")
    final_result_data: Optional[Any] = Field(None, description="Final result data (non-GitHub)") # Type needs refinement (QueryResult?)
    step_error: Optional[str] = Field(None, description="Error during step execution")
    github_step_has_tool_errors: bool = Field(False, description="Did any GitHub tool fail within this step?")
    filesystem_step_has_tool_errors: bool = Field(False, description="Did any Filesystem tool fail within this step?")
    python_step_has_tool_errors: bool = Field(False, description="Did any Python tool fail within this step?")
    step_outputs: Optional[List[Any]] = Field(None, description="Collected outputs for the step (especially for GitHub multi-tool steps)")


class StepCompletedEvent(BaseEvent):
    # Yielded externally for non-GitHub steps
    type: EventType = Field(default=EventType.STEP_COMPLETED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUCCESS, description="Fixed status")
    agent_type: Optional[AgentType] = Field(None, description="Agent type responsible for the step")
    step_id: str = Field(..., description="ID of the completed plan step")
    step_type: Optional[str] = Field(None, description="Type of the completed step (e.g., 'markdown')")
    cell_id: Optional[str] = Field(None, description="ID of the cell created for this step")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used to create the cell")
    result: Optional[Any] = Field(None, description="Result data associated with the step") # Type needs refinement


class StepErrorEvent(BaseEvent):
    # Yielded externally for non-GitHub steps that failed
    type: EventType = Field(default=EventType.STEP_ERROR, description="Event type identifier")
    status: StatusType = Field(default=StatusType.ERROR, description="Fixed status")
    agent_type: Optional[AgentType] = Field(None, description="Agent type responsible for the step")
    step_id: str = Field(..., description="ID of the failed plan step")
    step_type: Optional[str] = Field(None, description="Type of the failed step")
    cell_id: Optional[str] = Field(None, description="ID of the cell created (if any)")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used to create the cell (if any)")
    result: Optional[Any] = Field(None, description="Result data (likely containing error info)") # Type needs refinement
    error: Optional[str] = Field(None, description="Error message for the step failure") # Redundant with result.error?


class GitHubToolCellCreatedEvent(BaseEvent):
    # Yielded when a specific GitHub tool call results in a cell
    type: EventType = Field(default=EventType.GITHUB_TOOL_CELL_CREATED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUCCESS, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.GITHUB, description="Fixed agent type")
    original_plan_step_id: str = Field(..., description="Original plan step ID this tool belongs to")
    cell_id: Optional[str] = Field(..., description="ID of the created GitHub tool cell")
    tool_name: Optional[str] = Field(..., description="Name of the specific tool")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Result content for the cell") # Type needs refinement
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used to create the cell")


class GitHubToolErrorEvent(BaseEvent):
    # Yielded when a specific GitHub tool call fails (even if an error cell is created)
    type: EventType = Field(default=EventType.GITHUB_TOOL_ERROR, description="Event type identifier")
    status: StatusType = Field(default=StatusType.ERROR, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.GITHUB, description="Fixed agent type")
    original_plan_step_id: str = Field(..., description="Original plan step ID this tool belongs to")
    tool_call_id: Optional[str] = Field(None, description="ID of the failed tool call, if available")
    tool_name: Optional[str] = Field(..., description="Name of the specific tool that failed")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments passed to the tool")
    error: str = Field(..., description="Error message from the failed tool")


class SummaryStartedEvent(BaseEvent):
    type: EventType = Field(default=EventType.SUMMARY_STARTED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUMMARY_STARTED, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.SUMMARIZATION, description="Fixed agent type")


class SummaryUpdateEvent(BaseEvent):
    # Can be used for intermediate status updates from summarizer if needed
    type: EventType = Field(default=EventType.SUMMARY_UPDATE, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUMMARY_UPDATE, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.SUMMARIZATION, description="Fixed agent type")
    update_info: Dict[str, Any] = Field(..., description="Dictionary containing the specific update details")


class SummaryCellCreatedEvent(BaseEvent):
    type: EventType = Field(default=EventType.SUMMARY_CELL_CREATED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUMMARY_CELL_CREATED, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.SUMMARIZATION, description="Fixed agent type")
    cell_id: Optional[str] = Field(..., description="ID of the created summary cell")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used to create the cell")
    error: Optional[str] = Field(None, description="Error during summarization generation, if any")


class SummaryCellErrorEvent(BaseEvent):
    # If creating the summary cell itself fails
    type: EventType = Field(default=EventType.SUMMARY_CELL_ERROR, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUMMARY_CELL_ERROR, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.SUMMARIZATION, description="Fixed agent type")
    error: str = Field(..., description="Error message from cell creation failure")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters attempted for cell creation")


class InvestigationCompleteEvent(BaseEvent):
    type: EventType = Field(default=EventType.INVESTIGATION_COMPLETE, description="Event type identifier")
    status: StatusType = Field(default=StatusType.COMPLETE, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.INVESTIGATION_PLANNER, description="Fixed agent type")


# --- Events specific to Chat Agent ---

class ClarificationNeededEvent(BaseEvent):
    type: EventType = Field(default=EventType.CLARIFICATION_NEEDED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.CLARIFICATION_NEEDED, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.CHAT, description="Fixed agent type")
    message: str = Field(..., description="The clarification question to ask the user")


# --- Final Report Generation Events (Mirroring Summary Events) --- 

class ReportGenerationStartedEvent(BaseEvent):
    type: EventType = Field(default=EventType.REPORT_GENERATION_STARTED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.STARTING, description="Fixed status for start") # Use STARTING or appropriate status
    agent_type: AgentType = Field(default=AgentType.INVESTIGATION_REPORTER, description="Fixed agent type")

class ReportGenerationUpdateEvent(BaseEvent):
    type: EventType = Field(default=EventType.REPORT_GENERATION_UPDATE, description="Event type identifier")
    agent_type: AgentType = Field(default=AgentType.INVESTIGATION_REPORTER, description="Fixed agent type")
    update_info: Dict[str, Any] = Field(..., description="Dictionary containing status details (e.g., status, message, attempt)")

class ReportCellCreatedEvent(BaseEvent):
    type: EventType = Field(default=EventType.REPORT_CELL_CREATED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUCCESS, description="Fixed status for success") # Use SUCCESS or appropriate status
    agent_type: AgentType = Field(default=AgentType.INVESTIGATION_REPORTER, description="Fixed agent type")
    cell_id: Optional[str] = Field(..., description="ID of the created report cell")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used to create the cell")
    error: Optional[str] = Field(None, description="Any error during report generation (not cell creation)")

class ReportCellErrorEvent(BaseEvent):
    type: EventType = Field(default=EventType.REPORT_CELL_ERROR, description="Event type identifier")
    status: StatusType = Field(default=StatusType.ERROR, description="Fixed status for error") # Use ERROR or appropriate status
    agent_type: AgentType = Field(default=AgentType.INVESTIGATION_REPORTER, description="Fixed agent type")
    error: str = Field(..., description="Error message from cell creation failure")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters attempted for cell creation")

# --- New Filesystem Events ---

class FileSystemToolCellCreatedEvent(BaseEvent):
    type: EventType = Field(default=EventType.FILESYSTEM_TOOL_CELL_CREATED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUCCESS, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.FILESYSTEM, description="Fixed agent type")
    original_plan_step_id: str = Field(..., description="Original plan step ID this tool belongs to")
    cell_id: Optional[str] = Field(..., description="ID of the created filesystem tool cell")
    tool_name: Optional[str] = Field(..., description="Name of the specific tool")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Result content for the cell")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used to create the cell")

class FileSystemToolErrorEvent(BaseEvent):
    type: EventType = Field(default=EventType.FILESYSTEM_TOOL_ERROR, description="Event type identifier")
    status: StatusType = Field(default=StatusType.ERROR, description="Fixed status")
    agent_type: AgentType = Field(default=AgentType.FILESYSTEM, description="Fixed agent type")
    original_plan_step_id: str = Field(..., description="Original plan step ID this tool belongs to")
    tool_call_id: Optional[str] = Field(None, description="ID of the failed tool call, if available")
    tool_name: Optional[str] = Field(..., description="Name of the specific tool that failed")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments passed to the tool")
    error: str = Field(..., description="Error message from the failed tool")

class FileSystemStepProcessingCompleteEvent(BaseEvent):
    type: EventType = Field(default=EventType.FILESYSTEM_STEP_PROCESSING_COMPLETE, description="Event type identifier")
    status: StatusType = Field(default=StatusType.SUCCESS, description="Fixed status for this event type")
    agent_type: AgentType = Field(default=AgentType.FILESYSTEM, description="Fixed agent type")
    original_plan_step_id: Optional[str] = Field(..., description="Original plan step ID being marked as complete")


# --- Python Agent Specific Events ---

class PythonToolCellCreatedEvent(BaseEvent):
    """Event indicating a cell was created for a successful Python tool execution."""
    type: EventType = Field(default=EventType.PYTHON_TOOL_CELL_CREATED, description="Event type identifier")
    status: StatusType = Field(default=StatusType.TOOL_SUCCESS, description="Fixed status for success")
    agent_type: AgentType = Field(default=AgentType.PYTHON, description="Fixed agent type")
    original_plan_step_id: str = Field(..., description="The ID of the original investigation plan step")
    cell_id: Optional[str] = Field(..., description="The ID of the created notebook cell") # Made Optional
    tool_name: str = Field(..., description="The name of the specific Python tool executed (e.g., 'run_python_code')")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="The arguments passed to the tool")
    result: Optional[Any] = Field(..., description="The result object from the tool execution to store in the cell")
    cell_params: Optional[Dict[str, Any]] = Field(None, description="Parameters used by the backend to create the cell (for potential frontend use)")

class PythonToolErrorEvent(BaseEvent):
    """Event indicating an error occurred during Python tool execution or cell creation."""
    type: EventType = Field(default=EventType.PYTHON_TOOL_ERROR, description="Event type identifier")
    status: StatusType = Field(default=StatusType.ERROR, description="Fixed status for error")
    agent_type: AgentType = Field(default=AgentType.PYTHON, description="Fixed agent type")
    original_plan_step_id: Optional[str] = Field(None, description="The ID of the original plan step, if known")
    tool_call_id: Optional[str] = Field(None, description="The unique ID for the specific tool call that failed")
    tool_name: Optional[str] = Field(None, description="The name of the tool that failed")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="The arguments passed to the failed tool")
    error: str = Field(..., description="The error message")

class PythonStepProcessingCompleteEvent(BaseEvent):
    """Event indicating the Python agent has finished processing all tool calls for a plan step."""
    type: EventType = Field(default=EventType.PYTHON_STEP_PROCESSING_COMPLETE, description="Event type identifier")
    status: StatusType = Field(default=StatusType.STEP_PROCESSING_COMPLETE, description="Fixed status for step completion")
    agent_type: AgentType = Field(default=AgentType.PYTHON, description="Fixed agent type")
    original_plan_step_id: str = Field(..., description="Original plan step ID being marked as complete")
