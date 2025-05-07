"""
StepResult class for tracking comprehensive results from investigation steps.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.ai.models import StepType

class StepResult:
    """Container for comprehensive step results that combines results from all cells"""
    def __init__(self, step_id: str, step_type: Optional[StepType]):
        self.step_id: str = step_id
        self.step_type: Optional[StepType] = step_type
        self.outputs: List[Any] = []  # List of all outputs from this step
        self.errors: List[str] = []   # List of all errors encountered
        self.cell_ids: List[UUID] = [] # List of all cell IDs created for this step
        self.metadata: Dict[str, Any] = {} # Additional metadata about the step

    def add_output(self, output: Any) -> None:
        """Add an output from a cell to this step's results"""
        self.outputs.append(output)
    
    def add_error(self, error: str) -> None:
        """Add an error encountered during step execution"""
        if error and error not in self.errors:
            self.errors.append(error)
    
    def add_cell_id(self, cell_id: UUID) -> None:
        """Add a cell ID associated with this step"""
        if cell_id and cell_id not in self.cell_ids:
            self.cell_ids.append(cell_id)
    
    def has_error(self) -> bool:
        """Check if this step encountered any errors"""
        return len(self.errors) > 0
    
    def get_primary_output(self) -> Any:
        """Get the main output for simple step types like Markdown"""
        return self.outputs[0] if self.outputs else None
    
    def get_combined_error(self) -> Optional[str]:
        """Get a combined error message for all errors"""
        if not self.errors:
            return None
        if len(self.errors) == 1:
            return self.errors[0]
        return "; ".join(f"Error {i+1}: {err}" for i, err in enumerate(self.errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for storage and context building"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value if self.step_type else "unknown",
            "outputs": self.outputs,
            "errors": self.errors,
            "cell_count": len(self.cell_ids),
            "has_error": self.has_error(),
            "metadata": self.metadata
        }