import pytest
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional

from backend.ai.chat_agent import ChatAgentService
from backend.ai.chat_tools import NotebookCellTools, ListCellsParams # Import necessary tools/params
from backend.services.notebook_manager import NotebookManager # Import for mocking

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Helper to create mock cell data
def create_mock_cell(
    cell_id: str,
    cell_type: str,
    content: str,
    status: str = "completed",
    output_content: Optional[Any] = None,
    output_error: Optional[str] = None,
    updated_at_offset_secs: int = 0, # 0 = most recent
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    updated_at = now - timedelta(seconds=updated_at_offset_secs)
    return {
        "id": cell_id,
        "cell_type": cell_type,
        "content": content,
        "status": status,
        "output": {
            "content": output_content,
            "error": output_error,
            "execution_time": 1.0,
            "timestamp": updated_at.isoformat() # Use updated_at for consistency
        } if output_content is not None or output_error is not None else None,
        "metadata": metadata or {},
        "created_at": (updated_at - timedelta(seconds=10)).isoformat(), # Assume created before update
        "updated_at": updated_at.isoformat(),
        "dependencies": [], # Keep simple for these tests
        "tool_call_id": "dummy_tool_call_id",
        "tool_name": None,
        "tool_arguments": None,
        "connection_id": None
    }

@pytest.fixture
def mock_notebook_manager_for_chat():
    return MagicMock(spec=NotebookManager)

@pytest.fixture
def chat_agent_service(mock_notebook_manager_for_chat):
    service = ChatAgentService(notebook_manager=mock_notebook_manager_for_chat, redis_client=None)
    # Create a mock for NotebookCellTools and assign the async mock for list_cells to it
    mock_tools_instance = MagicMock(spec=NotebookCellTools)
    mock_tools_instance.list_cells = AsyncMock() # Assign the AsyncMock here
    service.cell_tools = mock_tools_instance # Assign the correctly mocked tools instance
    return service

# --- Tests for _prepare_notebook_context ---

async def test_prepare_context_no_cells(chat_agent_service: ChatAgentService):
    """Test context preparation when list_cells returns no cells."""
    assert chat_agent_service.cell_tools is not None
    chat_agent_service.cell_tools.list_cells.return_value = {
        "success": True,
        "cells": [] # Success, but cells list is empty
    }
    context = await chat_agent_service._prepare_notebook_context("nb1", "some query")
    # Check that an empty string is returned
    assert context == ""
    chat_agent_service.cell_tools.list_cells.assert_awaited_once()

async def test_prepare_context_list_cells_fails(chat_agent_service: ChatAgentService):
    """Test context preparation when list_cells fails."""
    assert chat_agent_service.cell_tools is not None
    chat_agent_service.cell_tools.list_cells.return_value = {
        "success": False,
        "error": "DB error",
        "cells": []
    }
    context = await chat_agent_service._prepare_notebook_context("nb1", "some query")
    # Check that an empty string is returned when list_cells fails
    assert context == ""
    chat_agent_service.cell_tools.list_cells.assert_awaited_once()

async def test_prepare_context_recency_and_limit(chat_agent_service: ChatAgentService):
    """Test selection based on recency and MAX_CONTEXT_CELLS limit."""
    assert chat_agent_service.cell_tools is not None
    mock_cells = [
        create_mock_cell("c1", "python", "print(1)", output_content="1", updated_at_offset_secs=0),
        create_mock_cell("c2", "markdown", "Older", output_content=None, updated_at_offset_secs=10),
        create_mock_cell("c3", "python", "print(3)", output_content="3", updated_at_offset_secs=5),
        create_mock_cell("c4", "python", "print(4)", output_content="4", updated_at_offset_secs=2),
        create_mock_cell("c5", "markdown", "Very Old", output_content=None, updated_at_offset_secs=20),
        create_mock_cell("c6", "python", "print(6)", output_content="6", updated_at_offset_secs=1),
    ]
    chat_agent_service.cell_tools.list_cells.return_value = {"success": True, "cells": mock_cells}
    
    context = await chat_agent_service._prepare_notebook_context("nb1", "unrelated query")
    
    assert f"Top 5 Relevant Cells" in context
    assert "ID: c1" in context
    assert "ID: c6" in context
    assert "ID: c4" in context
    assert "ID: c3" in context
    assert "ID: c2" in context
    assert "ID: c5" not in context # The oldest
    chat_agent_service.cell_tools.list_cells.assert_awaited_once()

async def test_prepare_context_keyword_match(chat_agent_service: ChatAgentService):
    """Test keyword matching boosts relevance."""
    assert chat_agent_service.cell_tools is not None
    mock_cells = [
        create_mock_cell("c1", "python", "df = load_data('file.csv')", updated_at_offset_secs=10), # Matches df
        create_mock_cell("c2", "markdown", "Summary of results", updated_at_offset_secs=0), # Recent, but no match
        create_mock_cell("c3", "python", "print(df.head())", output_content="col1,col2...", updated_at_offset_secs=5), # Matches df
    ]
    chat_agent_service.cell_tools.list_cells.return_value = {"success": True, "cells": mock_cells}

    context = await chat_agent_service._prepare_notebook_context("nb1", "show the `df` head again")
    
    assert f"Top 3 Relevant Cells" in context
    assert "ID: c3" in context
    assert "ID: c1" in context
    assert "ID: c2" in context # Still included as it's recent

    # Refined assertions for keyword matching test
    # Check that the block for cell c3 contains the relevant content keyword
    c3_context_found = False
    for line in context.splitlines():
        if "ID: c3" in line and "Content:" in line and "print(df.head())" in line:
             c3_context_found = True
             break
    assert c3_context_found, "Context for cell c3 with 'print(df.head())' not found or formatted incorrectly."
    
    # Check that the block for cell c3 contains the relevant output keyword
    c3_output_found = False
    for line in context.splitlines():
        if "ID: c3" in line and "Output:" in line and "col1,col2..." in line:
            c3_output_found = True
            break
    assert c3_output_found, "Context for cell c3 output 'col1,col2...' not found or formatted incorrectly."

    chat_agent_service.cell_tools.list_cells.assert_awaited_once()

async def test_prepare_context_cell_type_preference(chat_agent_service: ChatAgentService):
    """Test cell type preference based on query."""
    assert chat_agent_service.cell_tools is not None
    mock_cells = [
        create_mock_cell("c1", "markdown", "# Setup", updated_at_offset_secs=5), 
        create_mock_cell("c2", "python", "import pandas", output_content=None, updated_at_offset_secs=0), # Recent python
        create_mock_cell("c3", "markdown", "Conclusion", updated_at_offset_secs=10), 
    ]
    chat_agent_service.cell_tools.list_cells.return_value = {"success": True, "cells": mock_cells}

    context = await chat_agent_service._prepare_notebook_context("nb1", "run python script")
    
    assert f"Top 3 Relevant Cells" in context
    assert "ID: c2" in context
    assert context.find("[Cell 1 - ID: c2") != -1 # Check it appears first due to high score
    chat_agent_service.cell_tools.list_cells.assert_awaited_once()

async def test_prepare_context_python_structured_output(chat_agent_service: ChatAgentService):
    """Test smart summarization of structured Python outputs."""
    assert chat_agent_service.cell_tools is not None
    df_output = "--- DataFrame ---\n| col | val |\n|-----|-----|\n| A   | 1   |\n--- End DataFrame ---"
    json_output = "<JSON_OUTPUT>{\"key\": \"value\", \"count\": 5}</JSON_OUTPUT>"
    plot_output = "Some text before <PLOT_BASE64>ABCDEFG==</PLOT_BASE64> and after."
    plain_output = "Just regular print output."
    mock_cells = [
        create_mock_cell("c_df", "python", "show df", output_content=df_output, updated_at_offset_secs=0),
        create_mock_cell("c_json", "python", "show json", output_content=json_output, updated_at_offset_secs=1),
        create_mock_cell("c_plot", "python", "show plot", output_content=plot_output, updated_at_offset_secs=2),
        create_mock_cell("c_plain", "python", "show plain", output_content=plain_output, updated_at_offset_secs=3),
        create_mock_cell("c_error", "python", "show error", output_error="Traceback...", updated_at_offset_secs=4),
    ]
    chat_agent_service.cell_tools.list_cells.return_value = {"success": True, "cells": mock_cells}

    context = await chat_agent_service._prepare_notebook_context("nb1", "check outputs")

    assert "Python Output: DataFrame displayed" in context
    assert "| col | val |" in context # Check part of the df markdown is included
    assert "Python Output: JSON data provided" in context
    assert "{\"key\": \"value\"" in context # Check part of JSON is included
    assert "Python Output: Plot generated" in context
    assert "BASE64 content not shown" in context
    assert "Python Output: Just regular print output." in context
    assert "Error: Traceback..." in context
    chat_agent_service.cell_tools.list_cells.assert_awaited_once()

async def test_prepare_context_truncation(chat_agent_service: ChatAgentService):
    """Test truncation of long content and outputs."""
    assert chat_agent_service.cell_tools is not None
    long_string = "A" * 1000
    mock_cells = [
        create_mock_cell("c_long", "python", long_string, output_content=long_string, updated_at_offset_secs=0),
    ]
    chat_agent_service.cell_tools.list_cells.return_value = {"success": True, "cells": mock_cells}
    # Assuming CONTEXT_SUMMARY_TRUNCATE_LIMIT = 500 in chat_agent
    expected_truncated_len = 500 
    
    context = await chat_agent_service._prepare_notebook_context("nb1", "long content test")

    # Find the lines for content and output
    content_line_prefix = "Content: AAAA"
    # Use the correct prefix based on the summarizer logic
    output_line_prefix = "Python Output: AAAA"
    
    # Refined check for truncation:
    # 1. Check if a line contains the prefix
    content_prefix_found = any(content_line_prefix in line for line in context.splitlines())
    output_prefix_found = any(output_line_prefix in line for line in context.splitlines())
    assert content_prefix_found, f"Content line starting with '{content_line_prefix}' not found in context:\n{context}"
    assert output_prefix_found, f"Output line starting with '{output_line_prefix}' not found in context:\n{context}"
    
    # 2. Check if the truncation marker exists anywhere in the full context string
    # Use the exact marker text from utils._truncate_output
    # Use raw string r"..." to avoid escape sequence issues
    truncation_marker = r"\n\n...[output truncated – original length 1000 characters]"
    # We need to check against the raw context string, not line-by-line
    assert truncation_marker in context, f"Truncation marker {truncation_marker!r} not found in raw context:\n{context!r}"
    
    chat_agent_service.cell_tools.list_cells.assert_awaited_once() 