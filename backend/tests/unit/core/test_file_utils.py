import pytest
import os
import shutil
from unittest.mock import patch, mock_open

from backend.core.file_utils import stage_file_content, IMPORTED_DATA_BASE_PATH

# Test data
TEST_NOTEBOOK_ID = "test_notebook"
TEST_SESSION_ID = "test_session"
TEST_CONTENT = "col1,col2\nval1,val2"
TEST_ORIGINAL_FILENAME = "my_data.csv"
TEST_SOURCE_CELL_ID = "cell_abc123"

@pytest.fixture(autouse=True)
def cleanup_test_data_dir():
    """Cleans up the test data directory before and after each test."""
    test_notebook_dir = os.path.join(IMPORTED_DATA_BASE_PATH, TEST_NOTEBOOK_ID)
    if os.path.exists(test_notebook_dir):
        shutil.rmtree(test_notebook_dir)
    yield
    if os.path.exists(test_notebook_dir):
        shutil.rmtree(test_notebook_dir)

@pytest.mark.asyncio
async def test_stage_file_content_success_with_original_filename():
    """Test successful staging of file content with an original filename."""
    staged_path = await stage_file_content(
        content=TEST_CONTENT,
        original_filename=TEST_ORIGINAL_FILENAME,
        notebook_id=TEST_NOTEBOOK_ID,
        session_id=TEST_SESSION_ID,
        source_cell_id=TEST_SOURCE_CELL_ID
    )

    assert os.path.exists(staged_path)
    assert TEST_NOTEBOOK_ID in staged_path
    assert TEST_SESSION_ID in staged_path
    assert "my_data" in os.path.basename(staged_path) # Check for sanitized original name part
    assert staged_path.endswith(".csv")

    with open(staged_path, 'r', encoding='utf-8') as f:
        assert f.read() == TEST_CONTENT

@pytest.mark.asyncio
async def test_stage_file_content_success_no_original_filename():
    """Test successful staging when no original filename is provided."""
    staged_path = await stage_file_content(
        content="some other data",
        original_filename=None,
        notebook_id=TEST_NOTEBOOK_ID,
        session_id=TEST_SESSION_ID
    )

    assert os.path.exists(staged_path)
    assert "dataset" in os.path.basename(staged_path) # Default name part
    assert staged_path.endswith(".csv")

    with open(staged_path, 'r', encoding='utf-8') as f:
        assert f.read() == "some other data"

@pytest.mark.asyncio
async def test_stage_file_content_filename_sanitization():
    """Test filename sanitization logic."""
    original_filename = "file with spaces & special!@#$.txt"
    staged_path = await stage_file_content(
        content="data",
        original_filename=original_filename,
        notebook_id=TEST_NOTEBOOK_ID,
        session_id=TEST_SESSION_ID
    )
    
    base_name = os.path.basename(staged_path)
    assert "file_with_spaces___special____" in base_name 
    assert base_name.endswith(".csv") # Extension is changed to .csv

@pytest.mark.asyncio
async def test_stage_file_content_no_extension_in_original():
    """Test handling of original filename without an extension."""
    original_filename = "filenoext"
    staged_path = await stage_file_content(
        content="data",
        original_filename=original_filename,
        notebook_id=TEST_NOTEBOOK_ID,
        session_id=TEST_SESSION_ID
    )
    assert "filenoext_" in os.path.basename(staged_path)
    assert os.path.basename(staged_path).endswith(".csv") # .csv is added

@pytest.mark.asyncio
@patch("backend.core.file_utils.os.makedirs")
async def test_stage_file_content_makedirs_oserror(mock_makedirs):
    """Test OSError during directory creation."""
    mock_makedirs.side_effect = OSError("Test makedirs error")
    
    with pytest.raises(OSError) as excinfo:
        await stage_file_content(
            content=TEST_CONTENT,
            original_filename=TEST_ORIGINAL_FILENAME,
            notebook_id=TEST_NOTEBOOK_ID,
            session_id=TEST_SESSION_ID
        )
    assert "Test makedirs error" in str(excinfo.value)

@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open)
@patch("backend.core.file_utils.os.makedirs") # Mock makedirs to succeed
async def test_stage_file_content_write_ioerror(mock_makedirs_success, mock_file_open):
    """Test IOError during file writing."""
    mock_file_open.side_effect = IOError("Test write error")

    with pytest.raises(IOError) as excinfo:
        await stage_file_content(
            content=TEST_CONTENT,
            original_filename=TEST_ORIGINAL_FILENAME,
            notebook_id=TEST_NOTEBOOK_ID,
            session_id=TEST_SESSION_ID
        )
    assert "Test write error" in str(excinfo.value)

@pytest.mark.asyncio
async def test_stage_file_content_unique_filenames():
    """Test that subsequent calls with same original filename produce unique staged files."""
    path1 = await stage_file_content(
        content="content1",
        original_filename="duplicate.csv",
        notebook_id=TEST_NOTEBOOK_ID,
        session_id=TEST_SESSION_ID
    )
    path2 = await stage_file_content(
        content="content2",
        original_filename="duplicate.csv",
        notebook_id=TEST_NOTEBOOK_ID,
        session_id=TEST_SESSION_ID
    )
    assert os.path.exists(path1)
    assert os.path.exists(path2)
    assert path1 != path2
    with open(path1, 'r') as f1, open(path2, 'r') as f2:
        assert f1.read() == "content1"
        assert f2.read() == "content2"
