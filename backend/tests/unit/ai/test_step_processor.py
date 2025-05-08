import pytest
import os
import shutil
import uuid
from unittest.mock import patch, AsyncMock, MagicMock

from backend.ai.step_processor import StepProcessor
from backend.ai.models import InvestigationStepModel, StepType, FileDataRef
from backend.ai.step_result import StepResult
from backend.core.file_utils import IMPORTED_DATA_BASE_PATH
from backend.services.connection_manager import ConnectionManager
from backend.db.models import Connection  # For mock connection config

# Test constants
TEST_NOTEBOOK_ID = "proc_notebook_test"
TEST_SESSION_ID = "proc_session_test"

@pytest.fixture
def step_processor_instance(monkeypatch):
    """Fixture to create a StepProcessor instance with mocked dependencies."""
    mock_model = MagicMock() # Mock OpenAIModel
    mock_connection_manager = AsyncMock(spec=ConnectionManager)
    
    # Mock get_default_connection to return a mock Connection object
    mock_db_connection = Connection(
        id=uuid.uuid4(),
        name="default_fs",
        type="filesystem",
        config={"base_path": "/projects/test_fs_root"}, # Example config
        user_id="test_user",
        created_at=MagicMock(),
        updated_at=MagicMock()
    )
    mock_connection_manager.get_default_connection = AsyncMock(return_value=mock_db_connection)

    processor = StepProcessor(
        notebook_id=TEST_NOTEBOOK_ID,
        model=mock_model,
        connection_manager=mock_connection_manager
    )
    return processor

@pytest.fixture(autouse=True)
def cleanup_imported_data_dir():
    """Cleans up the test data directory before and after each test."""
    test_notebook_dir = os.path.join(IMPORTED_DATA_BASE_PATH, TEST_NOTEBOOK_ID)
    if os.path.exists(test_notebook_dir):
        shutil.rmtree(test_notebook_dir)
    yield
    if os.path.exists(test_notebook_dir):
        shutil.rmtree(test_notebook_dir)


class TestStepProcessorPrepareDataReferences:

    @pytest.mark.asyncio
    async def test_prepare_data_references_no_dependencies(self, step_processor_instance: StepProcessor):
        """Test with a step that has no dependencies."""
        step = InvestigationStepModel(
            step_id="step1",
            description="No deps step",
            step_type=StepType.PYTHON,
            dependencies=[]
        )
        step_results = {}
        
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        assert resolved_refs == []

    @pytest.mark.asyncio
    async def test_prepare_data_references_dependency_not_filesystem(self, step_processor_instance: StepProcessor):
        """Test when a dependency is not a FileSystem step."""
        step = InvestigationStepModel(
            step_id="step2",
            description="Python step",
            step_type=StepType.PYTHON,
            dependencies=["dep1"]
        )
        step_results = {
            "dep1": StepResult(step_id="dep1", step_type=StepType.MARKDOWN) # Not FileSystem
        }
        
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        assert resolved_refs == []

    @pytest.mark.asyncio
    async def test_prepare_data_references_fs_dep_no_outputs(self, step_processor_instance: StepProcessor):
        """Test when a FileSystem dependency has no outputs."""
        step = InvestigationStepModel(
            step_id="step3",
            description="Python step",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_1"]
        )
        fs_result = StepResult(step_id="dep_fs_1", step_type=StepType.FILESYSTEM)
        # fs_result.outputs is already [] by default
        step_results = {"dep_fs_1": fs_result}
        
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        assert resolved_refs == []

    @pytest.mark.asyncio
    async def test_prepare_data_references_fs_dep_with_content_string_output(self, step_processor_instance: StepProcessor):
        """Test FS dependency providing direct content string."""
        step = InvestigationStepModel(
            step_id="step_py",
            description="Python step with content input",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_content"]
        )
        
        fs_result = StepResult(step_id="dep_fs_content", step_type=StepType.FILESYSTEM)
        fs_result.add_output("colA,colB\n1,2") # Direct string content
        fs_result.add_cell_id(uuid.uuid4()) # Add a dummy cell ID
        
        step_results = {"dep_fs_content": fs_result}
        
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        
        assert len(resolved_refs) == 1
        ref = resolved_refs[0]
        assert isinstance(ref, FileDataRef)
        assert ref.type == "local_staged_path"
        assert os.path.exists(ref.value)
        assert TEST_NOTEBOOK_ID in ref.value
        assert TEST_SESSION_ID in ref.value
        assert "staged_from_content_dep_fs_content_output_0" in os.path.basename(ref.value)
        with open(ref.value, 'r') as f:
            assert f.read() == "colA,colB\n1,2"
        assert ref.source_cell_id == str(fs_result.cell_ids[0])

    @pytest.mark.asyncio
    @patch("backend.ai.step_processor.get_handler")
    async def test_prepare_data_references_fs_dep_with_fsmcp_path(self, mock_get_handler, step_processor_instance: StepProcessor):
        """Test FS dependency providing an fsmcp_path that needs resolution."""
        mock_fs_handler_instance = AsyncMock()
        mock_fs_handler_instance.execute_tool_call = AsyncMock(return_value="file_content_from_mcp")
        mock_get_handler.return_value = mock_fs_handler_instance

        step = InvestigationStepModel(
            step_id="step_py_mcp",
            description="Python step with MCP path input",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_mcp_path"]
        )
        
        fs_mcp_result = StepResult(step_id="dep_fs_mcp_path", step_type=StepType.FILESYSTEM)
        # Output is a dict indicating a path to be read by MCP
        fs_mcp_result.add_output({"path": "/projects/data/mcp_file.csv", "original_filename": "original_mcp.csv"})
        fs_mcp_result.add_cell_id(uuid.uuid4())
        
        step_results = {"dep_fs_mcp_path": fs_mcp_result}
        
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        
        assert len(resolved_refs) == 1
        ref = resolved_refs[0]
        assert ref.type == "local_staged_path"
        assert os.path.exists(ref.value)
        assert "original_mcp" in os.path.basename(ref.value) # Uses original_filename from dict
        with open(ref.value, 'r') as f:
            assert f.read() == "file_content_from_mcp"
        
        mock_get_handler.assert_called_once_with("filesystem")
        step_processor_instance.connection_manager.get_default_connection.assert_called_once_with("filesystem")
        mock_fs_handler_instance.execute_tool_call.assert_called_once_with(
            tool_name="read_file",
            tool_args={"path": "/projects/data/mcp_file.csv"},
            db_config={"base_path": "/projects/test_fs_root"}, # From mocked Connection
            correlation_id=TEST_SESSION_ID
        )

    @pytest.mark.asyncio
    async def test_prepare_data_references_fs_dep_output_dict_with_content(self, step_processor_instance: StepProcessor):
        """Test FS dependency output as dict with direct content."""
        step = InvestigationStepModel(
            step_id="step_py_dict_content",
            description="Python step with dict content input",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_dict_content"]
        )
        
        fs_result = StepResult(step_id="dep_fs_dict_content", step_type=StepType.FILESYSTEM)
        fs_result.add_output({
            "content": "dict_colA,dict_colB\nvalA,valB",
            "original_filename": "from_dict.txt", # Note: will be converted to .csv by stage_file_content
            "path": "/projects/some_path_ignored.txt" # Path should be ignored if content is present
        })
        fs_result.add_cell_id(uuid.uuid4())
        
        step_results = {"dep_fs_dict_content": fs_result}
        
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        
        assert len(resolved_refs) == 1
        ref = resolved_refs[0]
        assert ref.type == "local_staged_path"
        assert os.path.exists(ref.value)
        assert "from_dict" in os.path.basename(ref.value)
        assert ref.value.endswith(".csv") # Staging forces .csv
        with open(ref.value, 'r') as f:
            assert f.read() == "dict_colA,dict_colB\nvalA,valB"

    @pytest.mark.asyncio
    @patch("backend.ai.step_processor.get_handler")
    async def test_prepare_data_references_fs_dep_mcp_read_fails(self, mock_get_handler, step_processor_instance: StepProcessor):
        """Test when reading from Filesystem MCP fails."""
        mock_fs_handler_instance = AsyncMock()
        mock_fs_handler_instance.execute_tool_call = AsyncMock(side_effect=IOError("MCP read failed"))
        mock_get_handler.return_value = mock_fs_handler_instance

        step = InvestigationStepModel(
            step_id="step_py_mcp_fail",
            description="Test MCP read failure",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_mcp_fail_path"]
        )
        fs_mcp_result = StepResult(step_id="dep_fs_mcp_fail_path", step_type=StepType.FILESYSTEM)
        fs_mcp_result.add_output({"path": "/projects/fail/mcp_file.csv"})
        step_results = {"dep_fs_mcp_fail_path": fs_mcp_result}

        with pytest.raises(IOError) as excinfo:
            await step_processor_instance._prepare_data_references(
                step, step_results, TEST_SESSION_ID
            )
        assert "Failed to resolve fsmcp_path /projects/fail/mcp_file.csv" in str(excinfo.value)
        assert "MCP read failed" in str(excinfo.value) # Check for original error message

    @pytest.mark.asyncio
    @patch("backend.core.file_utils.stage_file_content", new_callable=AsyncMock) # Mock stage_file_content directly
    async def test_prepare_data_references_staging_fails(self, mock_stage_file_content, step_processor_instance: StepProcessor):
        """Test when stage_file_content itself raises an error."""
        mock_stage_file_content.side_effect = IOError("Staging disk full")

        step = InvestigationStepModel(
            step_id="step_py_stage_fail",
            description="Test staging failure",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_stage_fail"]
        )
        fs_result = StepResult(step_id="dep_fs_stage_fail", step_type=StepType.FILESYSTEM)
        fs_result.add_output("content_to_stage")
        step_results = {"dep_fs_stage_fail": fs_result}
        
        # This test expects _prepare_data_references to log the error and continue,
        # returning an empty list of resolved_refs if staging fails for an item.
        # The exception from stage_file_content should be caught within _prepare_data_references.
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        assert resolved_refs == [] # Expect no refs if staging failed
        mock_stage_file_content.assert_called_once() # Ensure it was called

    @pytest.mark.asyncio
    async def test_prepare_data_references_fs_dep_with_list_of_paths_output(self, step_processor_instance: StepProcessor):
        """Test FS dependency providing a list of fsmcp_paths (takes first)."""
        # This test assumes that if a list of paths is given, only the first one is processed.
        # This matches the current implementation detail: `fsmcp_path = output_item[0]`
        
        mock_fs_handler_instance = AsyncMock()
        mock_fs_handler_instance.execute_tool_call = AsyncMock(return_value="content_from_first_path")
        
        with patch("backend.ai.step_processor.get_handler", return_value=mock_fs_handler_instance) as mock_get_handler_patch:
            step = InvestigationStepModel(
                step_id="step_py_list_paths",
                description="Python step with list of paths input",
                step_type=StepType.PYTHON,
                dependencies=["dep_fs_list_paths"]
            )
            
            fs_list_result = StepResult(step_id="dep_fs_list_paths", step_type=StepType.FILESYSTEM)
            fs_list_result.add_output(["/projects/data/file1.csv", "/projects/data/file2.txt"]) # List of paths
            fs_list_result.add_cell_id(uuid.uuid4())
            
            step_results = {"dep_fs_list_paths": fs_list_result}
            
            resolved_refs = await step_processor_instance._prepare_data_references(
                step, step_results, TEST_SESSION_ID
            )
            
            assert len(resolved_refs) == 1
            ref = resolved_refs[0]
            assert ref.type == "local_staged_path"
            assert os.path.exists(ref.value)
            assert "file1" in os.path.basename(ref.value) # From first path
            with open(ref.value, 'r') as f:
                assert f.read() == "content_from_first_path"
            
            mock_get_handler_patch.assert_called_once_with("filesystem")
            mock_fs_handler_instance.execute_tool_call.assert_called_once_with(
                tool_name="read_file",
                tool_args={"path": "/projects/data/file1.csv"}, # Only first path used
                db_config={"base_path": "/projects/test_fs_root"},
                correlation_id=TEST_SESSION_ID
            )

    @pytest.mark.asyncio
    async def test_prepare_data_references_fs_dep_with_string_fsmcp_path(self, step_processor_instance: StepProcessor):
        """Test FS dependency providing a direct string fsmcp_path."""
        mock_fs_handler_instance = AsyncMock()
        mock_fs_handler_instance.execute_tool_call = AsyncMock(return_value="content_from_direct_path_str")
        
        with patch("backend.ai.step_processor.get_handler", return_value=mock_fs_handler_instance):
            step = InvestigationStepModel(
                step_id="step_py_str_path",
                description="Python step with string path input",
                step_type=StepType.PYTHON,
                dependencies=["dep_fs_str_path"]
            )
            
            fs_str_path_result = StepResult(step_id="dep_fs_str_path", step_type=StepType.FILESYSTEM)
            fs_str_path_result.add_output("/projects/data/direct_file.csv") # Direct string path
            fs_str_path_result.add_cell_id(uuid.uuid4())
            
            step_results = {"dep_fs_str_path": fs_str_path_result}
            
            resolved_refs = await step_processor_instance._prepare_data_references(
                step, step_results, TEST_SESSION_ID
            )
            
            assert len(resolved_refs) == 1
            ref = resolved_refs[0]
            assert ref.type == "local_staged_path"
            assert os.path.exists(ref.value)
            assert "direct_file" in os.path.basename(ref.value)
            with open(ref.value, 'r') as f:
                assert f.read() == "content_from_direct_path_str"
            
            mock_fs_handler_instance.execute_tool_call.assert_called_once_with(
                tool_name="read_file",
                tool_args={"path": "/projects/data/direct_file.csv"},
                db_config={"base_path": "/projects/test_fs_root"},
                correlation_id=TEST_SESSION_ID
            )

    @pytest.mark.asyncio
    async def test_prepare_data_references_fs_dep_unsupported_output_item(self, step_processor_instance: StepProcessor):
        """Test FS dependency with an unsupported output item type (e.g., int)."""
        step = InvestigationStepModel(
            step_id="step_py_unsupported",
            description="Python step with unsupported dep output",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_unsupported"]
        )
        
        fs_unsupported_result = StepResult(step_id="dep_fs_unsupported", step_type=StepType.FILESYSTEM)
        fs_unsupported_result.add_output(12345) # Integer, not a string, list, or dict
        fs_unsupported_result.add_cell_id(uuid.uuid4())
        
        step_results = {"dep_fs_unsupported": fs_unsupported_result}
        
        # Expect this to log a warning and skip the item, resulting in no resolved_refs
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        assert resolved_refs == []

    @pytest.mark.asyncio
    async def test_prepare_data_references_fs_dep_with_error_in_result(self, step_processor_instance: StepProcessor):
        """Test FS dependency that itself has an error."""
        step = InvestigationStepModel(
            step_id="step_py_dep_error",
            description="Python step with dep error",
            step_type=StepType.PYTHON,
            dependencies=["dep_fs_with_error"]
        )
        
        fs_error_result = StepResult(step_id="dep_fs_with_error", step_type=StepType.FILESYSTEM)
        fs_error_result.add_output("some output before error") # This output should be ignored due to error
        fs_error_result.add_error("Filesystem tool failed execution")
        fs_error_result.add_cell_id(uuid.uuid4())
        
        step_results = {"dep_fs_with_error": fs_error_result}
        
        resolved_refs = await step_processor_instance._prepare_data_references(
            step, step_results, TEST_SESSION_ID
        )
        assert resolved_refs == [] # No refs should be prepared if dependency had an error
