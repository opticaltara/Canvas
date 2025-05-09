"""
Data models and common types for the AI agent system.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import UnexpectedModelBehavior # Added import
from pydantic_ai.providers.openai import OpenAIProvider # Though not directly used in SafeOpenAIModel, good for context
# Corrected imports for OpenAI response types
from openai.types.chat import ChatCompletion as ChatCompletions # OpenAI's ChatCompletion is the response type
from pydantic_ai.messages import ModelResponse # Import ModelResponse directly
from pydantic import model_validator  # For custom validation logic

class StepType(str, Enum):
    """Enumeration of possible step types in an investigation."""
    MARKDOWN = "markdown"
    GITHUB = "github"
    FILESYSTEM = "filesystem"
    PYTHON = "python"
    INVESTIGATION_REPORT = "investigation_report"


class StepCategory(str, Enum):
    """Categorization of steps for UI organization."""
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
        description="The specific tool name to be executed in this step, if applicable",
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @model_validator(mode='after')
    def _validate_single_filesystem_step(self):
        """Ensure that the plan does not contain more than one FILESYSTEM step.

        Rationale:
            Empirically, merging multiple filesystem actions into one step
            improves reliability and avoids redundant I/O. Plans that contain
            more than one FILESYSTEM step are therefore considered invalid
            and should be re-generated or revised by the planner.

        Notes:
            • Markdown steps are exempt because they are purely narrative.
            • We only apply the constraint to FILESYSTEM for now, but this
              logic can be extended to other StepTypes if desired.
        """
        steps = self.steps if self.steps is not None else []
        filesystem_steps = [s for s in steps if s.step_type == StepType.FILESYSTEM]
        if len(filesystem_steps) > 1:
            step_ids = ", ".join(s.step_id for s in filesystem_steps)
            raise ValueError(
                f"Investigation plan contains multiple FILESYSTEM steps "
                f"({len(filesystem_steps)} found: {step_ids}). Combine related "
                "filesystem actions into a single step."
            )
        return self


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
    message_history: List[Any] = Field(
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


class SafeOpenAIModel(OpenAIModel):
    """
    Custom OpenAIModel that handles cases where the 'created' timestamp
    might be missing or None in the API response (e.g., from OpenRouter).
    """
    def _process_response(self, response: ChatCompletions) -> ModelResponse: # Use ModelResponse as the return type
        """
        Processes the API response, providing a default for 'created' if missing.
        """
        # Ensure response.created is an int. If None, use current time.
        # If it's already a valid int/float, fromtimestamp will handle it.
        created_timestamp = response.created
        if created_timestamp is None:
            default_ts = int(datetime.now(timezone.utc).timestamp())
            # print(f"Warning: API response 'created' timestamp was None. Using default: {default_ts}") # Optional: for debugging
            created_timestamp = default_ts
        
        # Ensure response, choices, and message exist to prevent AttributeError
        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            error_message = "Invalid or empty API response structure (missing choices or message)."
            # Log the problematic response if possible
            raw_resp_str = str(response.model_dump(mode='json') if response and hasattr(response, 'model_dump') else response)
            # print(f"Error: {error_message} Response: {raw_resp_str[:500]}") # Optional: for debugging
            # Raise an exception that the agent's error handling can catch
            raise UnexpectedModelBehavior(f"{error_message} Raw response: {raw_resp_str[:200]}")

        # Re-assign to ensure the superclass sees the potentially corrected timestamp
        response.created = created_timestamp

        # Now, it should be safe to call the superclass method
        return super()._process_response(response)


class FileDataRef(BaseModel):
    type: str = Field(..., description="Type of data reference, e.g., 'content_string' or 'fsmcp_path'.") # content_string, fsmcp_path
    value: str = Field(..., description="The actual CSV content string or the path string on the Filesystem MCP.")
    original_filename: Optional[str] = Field(None, description="The original filename, if known. Useful for naming the local permanent file.")
    source_cell_id: Optional[str] = Field(None, description="ID of the cell that originally produced this data reference.")
    # mime_type: Optional[str] = Field(None, description="MIME type of the file, e.g., 'text/csv'.") # Future enhancement


class PythonAgentInput(BaseModel):
    user_query: str = Field(..., description="The user's direct instruction or code for the Python cell.")
    notebook_id: str # Changed from UUID to str to match PythonAgent's notebook_id type
    session_id: str # Added session_id as it's useful for namespacing stored files
    dependency_cell_ids: Optional[Dict[str, List[str]]] = Field(default_factory=dict)