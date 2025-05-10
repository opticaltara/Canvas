import asyncio
from uuid import uuid4
import pytest

from backend.ai.agent import AIAgent
from backend.ai.models import InvestigationPlanModel
from backend.ai.events import PlanConfirmedEvent, PlanCancelledEvent, StepErrorEvent, InvestigationCompleteEvent
from backend.services.notebook_manager import NotebookManager
from backend.ai.chat_tools import NotebookCellTools

# ---- Dummy stubs ----------------------------------------------------------------

class DummyNotebookManager(NotebookManager):
    def __init__(self):
        pass  # bypass real init

class DummyCellTools(NotebookCellTools):
    def __init__(self):
        pass

# ---- Helper to build a minimal plan ---------------------------------------------

def _make_empty_plan(plan_id: str):
    return InvestigationPlanModel(
        plan_id=plan_id,
        status="draft",
        thinking="t",
        hypothesis=None,
        steps=[],
    )

# ---- Tests ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confirm_plan_success():
    agent = AIAgent(
        notebook_id=str(uuid4()),
        available_data_sources=[],
        notebook_manager=DummyNotebookManager(),
    )

    plan_id = str(uuid4())
    agent._draft_plans[plan_id] = _make_empty_plan(plan_id)

    events = [
        e async for e in agent.confirm_plan(
            plan_id=plan_id,
            edited_plan_json=None,
            session_id="sess",
            cell_tools=DummyCellTools(),
            notebook_id="nb",
            db_session=None,
        )
    ]

    # First event must be PlanConfirmedEvent
    assert isinstance(events[0], PlanConfirmedEvent)
    # Last event must be InvestigationCompleteEvent because plan has no steps
    assert isinstance(events[-1], InvestigationCompleteEvent)
    # Draft removed
    assert plan_id not in agent._draft_plans


@pytest.mark.asyncio
async def test_cancel_plan_success():
    agent = AIAgent(
        notebook_id=str(uuid4()),
        available_data_sources=[],
        notebook_manager=DummyNotebookManager(),
    )
    plan_id = str(uuid4())
    agent._draft_plans[plan_id] = _make_empty_plan(plan_id)

    events = [e async for e in agent.cancel_plan(plan_id, "sess", "nb")]
    assert isinstance(events[0], PlanCancelledEvent)
    assert plan_id not in agent._draft_plans


@pytest.mark.asyncio
async def test_cancel_plan_missing():
    agent = AIAgent(
        notebook_id=str(uuid4()),
        available_data_sources=[],
        notebook_manager=DummyNotebookManager(),
    )
    events = [e async for e in agent.cancel_plan("nonexistent", "sess", "nb")]
    assert isinstance(events[0], StepErrorEvent) 