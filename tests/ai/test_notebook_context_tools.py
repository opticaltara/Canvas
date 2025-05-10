import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

# The autouse fixture for mocking settings has been moved to tests/conftest.py

# --- Mock settings before other backend imports ---
class MockSettings(BaseModel):
    # Add fields that are actually used by the modules under test or their imports
    # For now, an empty model might suffice if no specific settings are accessed during import
    pass

@patch('backend.config.get_settings', return_value=MockSettings())
def setup_mock_settings(mock_get_settings):
    # This function is just to activate the patch during import time via a decorator trick
    # or by calling it. For pytest, a fixture or conftest.py might be cleaner.
    pass

# Call it to ensure patch is active before the imports below that trigger get_settings()
# However, pytest collection might happen before this explicit call.
# A more robust way is to use a fixture or ensure the patch is module-level for pytest.

# For now, let's assume the test runner or conftest.py might handle this, 
# or we place the patch more strategically. The error occurs during collection, 
# so a simple top-level patch might be tricky.

# --- End Mock settings ---


# Imports from the module to be tested
# Patching get_settings before this import chain is crucial.
# One way is to put the patch in a conftest.py or an autouse fixture.

# Let's try an autouse fixture to apply the patch early.
@pytest.fixture(autouse=True)
def mock_settings_globally(monkeypatch):
    mock_settings_instance = MockSettings()
    monkeypatch.setattr("backend.config.get_settings", lambda: mock_settings_instance)


from backend.ai.notebook_context_tools import (
    create_notebook_context_tools,
    ListCellsParams,
    CellSortByOption,
)
from backend.services.notebook_manager import NotebookManager
# We'll need to mock or define simple versions of these for testing
# from backend.core.cell import CellType, CellStatus # Assuming these enums are used

# Simple Enum-like classes for mocking, if actual enums are complex to import/use directly
class MockCellType(str):
    PYTHON = "python"
    MARKDOWN = "markdown"
    GITHUB = "github"

    @property
    def value(self):
        return str(self)

class MockCellStatus(str):
    SUCCESS = "success"
    ERROR = "error"
    RUNNING = "running"
    IDLE = "idle"

    @property
    def value(self):
        return str(self)


# Mock Pydantic models for Notebook and Cell to avoid deep dependencies
class MockCell(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: MockCellType = MockCellType("python")
    status: MockCellStatus = MockCellStatus("idle")
    content: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    position: Optional[int] = None # Allow position to be optional

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        # Simplified model_dump for testing
        return {
            "id": str(self.id),
            "type": self.type.value,
            "status": self.status.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "position": self.position,
        }

class MockNotebook(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    cells: Dict[UUID, MockCell] = Field(default_factory=dict)


@pytest.fixture
def mock_notebook_manager():
    manager = AsyncMock(spec=NotebookManager)
    manager.get_notebook = AsyncMock()
    manager.get_cell = AsyncMock()
    return manager

@pytest.fixture
def sample_notebook_id() -> str:
    return str(uuid4())

# --- Tests for create_notebook_context_tools ---

def test_create_notebook_context_tools_with_manager(mock_notebook_manager, sample_notebook_id):
    tools = create_notebook_context_tools(sample_notebook_id, mock_notebook_manager)
    assert len(tools) == 2
    assert callable(tools[0])
    assert callable(tools[1])
    # Check for specific names if they are set, e.g.
    assert tools[0].__name__ == "list_cells"
    assert tools[1].__name__ == "get_cell"

def test_create_notebook_context_tools_without_manager(sample_notebook_id):
    tools = create_notebook_context_tools(sample_notebook_id, None)
    assert len(tools) == 0

# --- Test Suite for list_cells and get_cell ---
# We'll need to patch 'get_db_session' from backend.ai.notebook_context_tools

@pytest.mark.asyncio
@patch('backend.ai.notebook_context_tools.get_db_session')
async def test_list_cells_empty_notebook(mock_get_db_session, mock_notebook_manager, sample_notebook_id):
    mock_db_session = AsyncMock()
    mock_get_db_session.return_value.__aenter__.return_value = mock_db_session # Ensure __aenter__ returns the mock

    empty_notebook = MockNotebook(id=UUID(sample_notebook_id), cells={})
    mock_notebook_manager.get_notebook.return_value = empty_notebook

    tools = create_notebook_context_tools(sample_notebook_id, mock_notebook_manager)
    list_cells_func = tools[0]

    params = ListCellsParams()
    result = await list_cells_func(params)

    assert result == []
    mock_notebook_manager.get_notebook.assert_called_once_with(mock_db_session, UUID(sample_notebook_id))

# More tests will be added here for list_cells and get_cell functionality 