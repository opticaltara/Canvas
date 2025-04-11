from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from backend.core.cell import CellType


class InvestigationStep(BaseModel):
    """A step in the investigation plan"""
    step_type: str = Field(description="Type of step (e.g., 'sql', 'python', 'markdown')")
    description: str = Field(description="Description of what this step will do")
    dependencies: List[str] = Field(
        description="List of step IDs this step depends on",
        default_factory=list
    )
    parameters: Dict[str, Any] = Field(
        description="Parameters for the step",
        default_factory=dict
    )


class InvestigationPlan(BaseModel):
    """The complete investigation plan"""
    steps: List[InvestigationStep] = Field(
        description="Steps to execute in the investigation"
    )
    thinking: Optional[str] = Field(
        description="Reasoning behind the investigation plan",
        default=None
    )


class PlanExecutionState(BaseModel):
    """Tracks the execution state of an investigation plan"""
    plan: InvestigationPlan
    step_to_cell_map: Dict[int, UUID] = Field(default_factory=dict)
    completed_steps: List[int] = Field(default_factory=list)
    current_step: Optional[int] = None
    
    def get_next_executable_step(self) -> Optional[InvestigationStep]:
        """
        Get the next step that can be executed based on dependencies
        
        Returns:
            The next executable step, or None if no steps are ready
        """
        for i, step in enumerate(self.plan.steps):
            # Skip completed steps
            if i in self.completed_steps:
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = all(dep in self.completed_steps for dep in step.dependencies)
            if deps_satisfied:
                return step
        
        return None
    
    def mark_step_completed(self, step_id: int) -> None:
        """Mark a step as completed"""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
        
        if self.current_step == step_id:
            self.current_step = None
    
    def set_current_step(self, step_id: int) -> None:
        """Set the current step being executed"""
        self.current_step = step_id
    
    def is_complete(self) -> bool:
        """Check if all steps in the plan have been completed"""
        return set(self.completed_steps) == set(range(len(self.plan.steps)))
    
    def map_step_to_cell(self, step_id: int, cell_id: UUID) -> None:
        """Map a step ID to a cell ID"""
        self.step_to_cell_map[step_id] = cell_id
    
    def get_cell_for_step(self, step_id: int) -> Optional[UUID]:
        """Get the cell ID for a step"""
        return self.step_to_cell_map.get(step_id)


class PlanAdapter:
    """
    Adapts plans from different formats to the internal InvestigationPlan format
    """
    @staticmethod
    def from_dict(data: Dict) -> InvestigationPlan:
        """Convert a dictionary (e.g., from AI output) to an InvestigationPlan"""
        steps = []
        
        for step_data in data.get("steps", []):
            # Convert cell_type from string to enum
            cell_type_str = step_data.get("cell_type", "markdown").upper()
            try:
                cell_type = CellType[cell_type_str]
            except KeyError:
                # Default to markdown if type is unknown
                cell_type = CellType.MARKDOWN
            
            step = InvestigationStep(
                step_type=cell_type_str.lower(),
                description=step_data.get("description", ""),
                dependencies=step_data.get("depends_on", []),
                parameters={"content": step_data.get("content", "")}
            )
            steps.append(step)
        
        return InvestigationPlan(
            steps=steps,
            thinking=data.get("thinking")
        )
    
    @staticmethod
    def to_dict(plan: InvestigationPlan) -> Dict:
        """Convert an InvestigationPlan to a dictionary"""
        return {
            "steps": [
                {
                    "step_type": step.step_type,
                    "description": step.description,
                    "dependencies": step.dependencies,
                    "parameters": step.parameters
                }
                for step in plan.steps
            ],
            "thinking": plan.thinking
        }