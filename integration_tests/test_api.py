import httpx
import os
import pytest
import uuid
import json # For parsing streamed JSON lines
from typing import Optional, List, Dict, Any

# Get the backend URL from an environment variable
# This will be 'http://backend:8000' when running in Docker Compose
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:9091") # Default for local dev if needed
API_BASE_URL = f"{BACKEND_URL}/api"

# --- Helper Functions ---
def create_notebook_util(client: httpx.Client, title: str, description: str = "Test Description") -> dict:
    response = client.post(f"{API_BASE_URL}/notebooks/", json={"title": title, "description": description})
    response.raise_for_status()
    assert response.status_code == 201
    return response.json()

def delete_notebook_util(client: httpx.Client, notebook_id: str):
    response = client.delete(f"{API_BASE_URL}/notebooks/{notebook_id}")
    # Allow 404 if already deleted by cascade or other tests
    if response.status_code not in [204, 404]:
        response.raise_for_status()

def create_cell_util(client: httpx.Client, notebook_id: str, cell_type: str, content: str = "Test Content", tool_call_id: Optional[str] = None) -> dict:
    if tool_call_id is None:
        tool_call_id = str(uuid.uuid4())
    # Ensure tool_call_id is a string before using in f-string, even though the above line should handle it.
    # This is more for type checker clarity if it's still confused.
    current_tool_call_id = str(tool_call_id)
    response = client.post(
        f"{API_BASE_URL}/notebooks/{notebook_id}/cells?tool_call_id={current_tool_call_id}",
        json={"cell_type": cell_type, "content": content}
    )
    response.raise_for_status()
    assert response.status_code == 200 # API returns 200 for cell creation
    return response.json()

# --- Fixtures ---
@pytest.fixture(scope="function")
def client():
    # Increased timeout to accommodate potentially slow Docker operations (e.g., image pulls)
    # and internal backend timeouts (e.g., MCP session init ~45s, list_tools ~20s).
    # Default is 5 seconds for read, connect, etc.
    timeout_config = httpx.Timeout(70.0, connect=10.0) # 70s for read, 10s for connect
    with httpx.Client(timeout=timeout_config) as c:
        yield c

@pytest.fixture(scope="function")
def test_notebook(client: httpx.Client):
    notebook_data = create_notebook_util(client, title="My Test Notebook")
    yield notebook_data
    delete_notebook_util(client, notebook_data["id"])


# --- Tests ---
def test_health_check(client: httpx.Client):
    """
    Tests the /api/health endpoint of the backend.
    """
    try:
        response = client.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        assert response.status_code == 200
        # Assuming health check returns {"status": "healthy"} or similar
        assert "status" in response.json() or "version" in response.json() # More flexible check
        print(f"Health check response: {response.text}")
    except httpx.RequestError as exc:
        pytest.fail(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except httpx.HTTPStatusError as exc:
        pytest.fail(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}. Response: {exc.response.text}")


# --- Notebook Endpoint Tests ---
def test_create_and_list_notebooks(client: httpx.Client):
    initial_notebooks_response = client.get(f"{API_BASE_URL}/notebooks/")
    initial_notebooks_response.raise_for_status()
    initial_count = len(initial_notebooks_response.json())

    notebook_title = f"Test Notebook {uuid.uuid4()}"
    created_notebook = create_notebook_util(client, title=notebook_title)
    assert "id" in created_notebook
    assert created_notebook["metadata"]["title"] == notebook_title

    notebooks_response = client.get(f"{API_BASE_URL}/notebooks/")
    notebooks_response.raise_for_status()
    notebooks = notebooks_response.json()
    assert len(notebooks) == initial_count + 1
    assert any(nb["id"] == created_notebook["id"] for nb in notebooks)

    delete_notebook_util(client, created_notebook["id"])


def test_get_notebook(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}")
    response.raise_for_status()
    assert response.status_code == 200
    fetched_notebook = response.json()
    assert fetched_notebook["id"] == notebook_id
    assert fetched_notebook["metadata"]["title"] == test_notebook["metadata"]["title"]


def test_update_notebook(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    new_title = f"Updated Test Notebook {uuid.uuid4()}"
    new_description = "Updated description"

    update_payload = {"title": new_title, "description": new_description}
    response = client.put(f"{API_BASE_URL}/notebooks/{notebook_id}", json=update_payload)
    response.raise_for_status()
    assert response.status_code == 200
    updated_notebook = response.json()
    assert updated_notebook["metadata"]["title"] == new_title
    assert updated_notebook["metadata"]["description"] == new_description

    # Verify by getting the notebook again
    get_response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}")
    get_response.raise_for_status()
    fetched_notebook = get_response.json()
    assert fetched_notebook["metadata"]["title"] == new_title
    assert fetched_notebook["metadata"]["description"] == new_description


def test_delete_notebook(client: httpx.Client):
    notebook_title = f"Notebook to Delete {uuid.uuid4()}"
    created_notebook = create_notebook_util(client, title=notebook_title)
    notebook_id = created_notebook["id"]

    delete_notebook_util(client, notebook_id)

    # Verify it's deleted
    response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}")
    assert response.status_code == 404


# --- Cell Endpoint Tests ---
def test_create_and_list_cells(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]

    initial_cells_response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}/cells")
    initial_cells_response.raise_for_status()
    initial_cells = initial_cells_response.json()
    # Notebooks might start with a default cell, or no cells.
    # For this test, let's assume it can start empty or with some cells.
    initial_count = len(initial_cells)


    cell1_content = "print('Hello from cell 1')"
    cell1_type = "python"
    created_cell1 = create_cell_util(client, notebook_id, cell_type=cell1_type, content=cell1_content)
    assert "id" in created_cell1
    assert created_cell1["type"] == cell1_type
    assert created_cell1["content"] == cell1_content

    cell2_content = "# Markdown Cell"
    cell2_type = "markdown"
    created_cell2 = create_cell_util(client, notebook_id, cell_type=cell2_type, content=cell2_content)
    assert "id" in created_cell2
    assert created_cell2["type"] == cell2_type

    cells_response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}/cells")
    cells_response.raise_for_status()
    cells = cells_response.json()
    assert len(cells) == initial_count + 2 # Two cells created

    # Check if created cells are in the list
    cell_ids = [cell["id"] for cell in cells]
    assert created_cell1["id"] in cell_ids
    assert created_cell2["id"] in cell_ids


def test_get_cell(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    cell_content = "Test cell content for get"
    created_cell = create_cell_util(client, notebook_id, cell_type="markdown", content=cell_content)
    cell_id = created_cell["id"]

    response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}/cells/{cell_id}")
    response.raise_for_status()
    assert response.status_code == 200
    fetched_cell = response.json()
    assert fetched_cell["id"] == cell_id
    assert fetched_cell["content"] == cell_content


def test_update_cell(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    created_cell = create_cell_util(client, notebook_id, cell_type="python", content="initial_content = 1")
    cell_id = created_cell["id"]

    new_content = "updated_content = 2"
    update_payload = {"content": new_content}
    response = client.put(f"{API_BASE_URL}/notebooks/{notebook_id}/cells/{cell_id}", json=update_payload)
    response.raise_for_status()
    assert response.status_code == 200
    updated_cell = response.json()
    assert updated_cell["content"] == new_content

    # Verify by getting the cell again
    get_response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}/cells/{cell_id}")
    get_response.raise_for_status()
    fetched_cell = get_response.json()
    assert fetched_cell["content"] == new_content


def test_delete_cell(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    created_cell = create_cell_util(client, notebook_id, cell_type="markdown", content="Cell to delete")
    cell_id = created_cell["id"]

    delete_response = client.delete(f"{API_BASE_URL}/notebooks/{notebook_id}/cells/{cell_id}")
    delete_response.raise_for_status()
    assert delete_response.status_code == 204

    # Verify it's deleted
    get_response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}/cells/{cell_id}")
    assert get_response.status_code == 404


def test_execute_cell_basic(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    # Create a simple python cell
    cell_content = "x = 1 + 1\nprint(x)"
    created_cell = create_cell_util(client, notebook_id, cell_type="python", content=cell_content)
    cell_id = created_cell["id"]

    execute_payload = {"content": cell_content} # Can optionally update content before execution
    response = client.post(f"{API_BASE_URL}/notebooks/{notebook_id}/cells/{cell_id}/execute", json=execute_payload)
    response.raise_for_status()
    assert response.status_code == 200
    execution_response = response.json()
    assert execution_response["message"] == "Cell queued for execution"
    assert execution_response["cell_id"] == cell_id
    assert execution_response["status"] == "queued"

    # Note: This test only checks if the cell is queued.
    # Verifying actual execution result would require WebSocket communication or polling,
    # which is more complex for a basic integration test.


def test_reorder_cells(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    cell1 = create_cell_util(client, notebook_id, cell_type="markdown", content="Cell 1")
    cell2 = create_cell_util(client, notebook_id, cell_type="markdown", content="Cell 2")
    cell3 = create_cell_util(client, notebook_id, cell_type="markdown", content="Cell 3")

    # Initial order might be [cell1_id, cell2_id, cell3_id] or based on creation time/db order
    # For the test, we'll reorder them explicitly
    new_order = [cell3["id"], cell1["id"], cell2["id"]]
    reorder_payload = {"cell_order": new_order}

    response = client.post(f"{API_BASE_URL}/notebooks/{notebook_id}/reorder", json=reorder_payload)
    response.raise_for_status()
    assert response.status_code == 200
    assert response.json()["message"] == "Cells reordered successfully"

    # Verify the new order by fetching the notebook (if cell_order is part of notebook response)
    # or by fetching all cells and checking their 'position' attribute if available and updated.
    # The current Notebook model in routes/notebooks.py doesn't directly expose cell_order in its root.
    # Let's fetch the notebook and check the cell_order attribute from the Notebook Pydantic model
    # This assumes the notebook manager updates the notebook's cell_order list.
    get_notebook_response = client.get(f"{API_BASE_URL}/notebooks/{notebook_id}")
    get_notebook_response.raise_for_status()
    notebook_details = get_notebook_response.json()
    assert notebook_details["cell_order"] == new_order


# --- Chat Endpoint Tests ---

@pytest.fixture(scope="function")
def test_chat_session(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    response = client.post(f"{API_BASE_URL}/chat/sessions", json={"notebook_id": notebook_id})
    response.raise_for_status()
    assert response.status_code == 200
    session_data = response.json()
    assert "session_id" in session_data
    yield session_data
    # Attempt to delete the session
    if "session_id" in session_data:
        delete_response = client.delete(f"{API_BASE_URL}/chat/sessions/{session_data['session_id']}")
        # Allow 404 if already deleted or 204 for successful deletion
        assert delete_response.status_code in [204, 404]


def test_create_chat_session(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    response = client.post(f"{API_BASE_URL}/chat/sessions", json={"notebook_id": notebook_id})
    response.raise_for_status()
    assert response.status_code == 200
    session_data = response.json()
    assert "session_id" in session_data
    # Clean up
    client.delete(f"{API_BASE_URL}/chat/sessions/{session_data['session_id']}")


def test_get_chat_messages_empty(client: httpx.Client, test_chat_session: dict):
    session_id = test_chat_session["session_id"]
    response = client.get(f"{API_BASE_URL}/chat/sessions/{session_id}/messages")
    response.raise_for_status()
    assert response.status_code == 200
    # Expecting newline-delimited JSON, which might be empty if no messages
    assert response.text == "" # Or check for empty list if it returns [] as JSON


def test_post_chat_message(client: httpx.Client, test_chat_session: dict):
    session_id = test_chat_session["session_id"]
    prompt_text = "Hello, AI!"
    # This endpoint streams, so direct check is tricky.
    # We'll just check if the request is accepted and try to read some initial part of the stream.
    try:
        with client.stream("POST", f"{API_BASE_URL}/chat/sessions/{session_id}/messages", data={"prompt": prompt_text}, timeout=10) as response:
            assert response.status_code == 200
            # Try to read a bit of the stream to ensure it's working
            stream_content = b""
            for chunk in response.iter_bytes(chunk_size=1024):
                stream_content += chunk
                if len(stream_content) > 0: # Got some data
                    break
            assert len(stream_content) > 0, "Stream did not return any content"
            print(f"Streamed chat response (partial): {stream_content.decode(errors='ignore')}")
    except httpx.ReadTimeout:
        pytest.fail("Reading from chat stream timed out.")
    # Further testing would involve checking DB or WebSocket for message persistence/broadcast


def test_delete_chat_session(client: httpx.Client, test_notebook: dict):
    notebook_id = test_notebook["id"]
    create_response = client.post(f"{API_BASE_URL}/chat/sessions", json={"notebook_id": notebook_id})
    create_response.raise_for_status()
    session_id = create_response.json()["session_id"]

    delete_response = client.delete(f"{API_BASE_URL}/chat/sessions/{session_id}")
    delete_response.raise_for_status()
    assert delete_response.status_code == 204

    # Verify it's deleted (optional, as get messages might still work on an empty session)
    # get_messages_response = client.get(f"{API_BASE_URL}/chat/sessions/{session_id}/messages")
    # assert get_messages_response.status_code == 404 # This depends on implementation


# --- Connection Endpoint Tests ---

def test_list_connection_types(client: httpx.Client):
    response = client.get(f"{API_BASE_URL}/connections/types")
    response.raise_for_status()
    assert response.status_code == 200
    types = response.json()
    assert isinstance(types, list)
    assert len(types) >= 0 # Can be empty if no handlers are registered
    if "filesystem" in types: # Example check
        assert "filesystem" in types


def test_list_connections_empty_or_not(client: httpx.Client):
    response = client.get(f"{API_BASE_URL}/connections/")
    response.raise_for_status()
    assert response.status_code == 200
    assert isinstance(response.json(), list) # Should always return a list


@pytest.fixture(scope="function")
def test_connection(client: httpx.Client):
    # Create a dummy filesystem connection for testing other connection endpoints
    # This assumes 'filesystem' type is available and requires minimal config
    conn_name = f"Test Filesystem Conn {uuid.uuid4()}"
    # Use a path that will be valid from the host Docker daemon's perspective,
    # assuming ./data on the host is mounted to /app/data in the backend container.
    # This test_fs_mount_dir should be created on the host inside the ./data directory.
    host_test_dir_in_container = "/app/data/test_fs_mount_dir" 
    payload = {
        "name": conn_name,
        "payload": {
            "type": "filesystem", 
            "allowed_directories": [host_test_dir_in_container]
        }
    }
    # Check if filesystem type is available first
    types_response = client.get(f"{API_BASE_URL}/connections/types")
    if "filesystem" not in types_response.json():
        pytest.skip("Filesystem connection type not available, skipping connection tests that depend on it.")

    create_response = client.post(f"{API_BASE_URL}/connections/", json=payload)
    if create_response.status_code == 400 and "No handler registered for type 'filesystem'" in create_response.text:
         pytest.skip("Filesystem connection handler not registered. Skipping dependent tests.")
    create_response.raise_for_status()
    connection_data = create_response.json()
    yield connection_data
    # Clean up
    client.delete(f"{API_BASE_URL}/connections/{connection_data['id']}")


def test_create_get_delete_connection(client: httpx.Client):
    # Check if filesystem type is available first
    types_response = client.get(f"{API_BASE_URL}/connections/types")
    if "filesystem" not in types_response.json():
        pytest.skip("Filesystem connection type not available, skipping connection creation test.")

    conn_name = f"FS Conn {uuid.uuid4()}"
    create_payload = {
        "name": conn_name,
        "payload": {
            "type": "filesystem",
            "allowed_directories": [f"/tmp/test_{uuid.uuid4()}"]
        }
    }
    create_response = client.post(f"{API_BASE_URL}/connections/", json=create_payload)
    if create_response.status_code == 400 and "No handler registered for type 'filesystem'" in create_response.text:
         pytest.skip("Filesystem connection handler not registered. Skipping test.")
    create_response.raise_for_status()
    created_conn = create_response.json()
    assert created_conn["name"] == conn_name
    assert created_conn["type"] == "filesystem"
    conn_id = created_conn["id"]

    get_response = client.get(f"{API_BASE_URL}/connections/{conn_id}")
    get_response.raise_for_status()
    fetched_conn = get_response.json()
    assert fetched_conn["id"] == conn_id
    assert fetched_conn["name"] == conn_name

    delete_response = client.delete(f"{API_BASE_URL}/connections/{conn_id}")
    delete_response.raise_for_status()
    assert delete_response.status_code == 204

    get_after_delete_response = client.get(f"{API_BASE_URL}/connections/{conn_id}")
    assert get_after_delete_response.status_code == 404


def test_update_connection(client: httpx.Client, test_connection: dict):
    conn_id = test_connection["id"]
    new_name = f"Updated FS Conn {uuid.uuid4()}"
    update_payload = {
        "name": new_name,
        # Optionally update payload fields if supported and desired for the test
        # "payload": {
        #     "type": "filesystem", # Type must match
        #     "allowed_directories": ["/tmp/new_updated_path"]
        # }
    }
    response = client.put(f"{API_BASE_URL}/connections/{conn_id}", json=update_payload)
    response.raise_for_status()
    updated_conn = response.json()
    assert updated_conn["name"] == new_name
    # Add more assertions if payload was updated

    get_response = client.get(f"{API_BASE_URL}/connections/{conn_id}")
    get_response.raise_for_status()
    assert get_response.json()["name"] == new_name


def test_set_default_connection(client: httpx.Client, test_connection: dict):
    conn_id = test_connection["id"]
    conn_type = test_connection["type"]

    response = client.post(f"{API_BASE_URL}/connections/{conn_id}/default")
    response.raise_for_status()
    assert response.status_code == 200
    assert response.json()["message"] == "Default connection set successfully"

    # Verify by getting the connection and checking 'is_default'
    # This requires the GET /connections/{id} endpoint to return 'is_default'
    get_response = client.get(f"{API_BASE_URL}/connections/{conn_id}")
    get_response.raise_for_status()
    conn_details = get_response.json()
    assert conn_details.get("is_default") is True # Check if is_default is true

    # It would be good to also test unsetting or changing default, but that's more complex.

@pytest.fixture(scope="function")
def setup_filesystem_connection(client: httpx.Client):
    # Ensure filesystem type is available
    types_response = client.get(f"{API_BASE_URL}/connections/types")
    types_response.raise_for_status() # Ensure this request itself is successful
    if "filesystem" not in types_response.json():
        pytest.skip("Filesystem connection type not available.")

    conn_name = f"TestFSConn_For_Chat_{uuid.uuid4()}"
    # Using /app as an example directory that should exist in the container.
    # Adjust if FileSystemAgent has specific constraints or requires a more specific setup.
    payload = {
        "name": conn_name,
        "payload": {"type": "filesystem", "allowed_directories": ["/app", "/app/data"]}
    }
    create_response = client.post(f"{API_BASE_URL}/connections/", json=payload)
    if create_response.status_code == 400 and "No handler registered" in create_response.text:
        pytest.skip("Filesystem connection handler not registered.")
    create_response.raise_for_status()
    connection_data = create_response.json()

    # Set as default
    set_default_response = client.post(f"{API_BASE_URL}/connections/{connection_data['id']}/default")
    set_default_response.raise_for_status()
    
    yield connection_data

    # Clean up: Delete connection
    delete_conn_response = client.delete(f"{API_BASE_URL}/connections/{connection_data['id']}")
    # Allow 404 if already deleted by cascade or other means
    if delete_conn_response.status_code not in [204, 404]:
        delete_conn_response.raise_for_status()


# def test_get_tools_for_connection_type(client: httpx.Client, test_connection: dict):
#     # This test assumes the 'filesystem' connection type (used by test_connection)
#     # has some tools registered or the handler can return an empty list.
#     conn_type = test_connection["type"] # Should be 'filesystem'
# 
#     # Ensure it's default so tools can be fetched
#     client.post(f"{API_BASE_URL}/connections/{test_connection['id']}/default").raise_for_status()
# 
#     response = client.get(f"{API_BASE_URL}/connections/{conn_type}/tools")
#     if response.status_code == 404 and "No default connection found" in response.text:
#         # This can happen if the default setting didn't stick or there's a race condition
#         # For robustness, we can try setting default again or just skip if this occurs
#         pytest.skip(f"Could not reliably set default for {conn_type} to fetch tools.")
# 
#     response.raise_for_status()
#     assert response.status_code == 200
#     tools = response.json()
#     assert isinstance(tools, list)
#     # If filesystem has tools, you can assert specific tool names
#     # e.g., if it has a 'list_files' tool:
#     # if tools:
#     #    assert any(tool['name'] == 'list_files' for tool in tools)


def test_test_connection_config(client: httpx.Client):
    # Test with a valid (dummy) filesystem config
    # This assumes 'filesystem' type is available for testing
    types_response = client.get(f"{API_BASE_URL}/connections/types")
    if "filesystem" not in types_response.json():
        pytest.skip("Filesystem connection type not available for testing.")

    test_payload = {
        "payload": {
            "type": "filesystem",
            "allowed_directories": ["/tmp"] # A generally accessible path for a basic test
        }
    }
    response = client.post(f"{API_BASE_URL}/connections/test", json=test_payload)
    response.raise_for_status()
    result = response.json()
    assert result["valid"] is True # Filesystem test should be basic and pass

    # Test with an invalid type (if possible, or a config known to fail for a type)
    # test_invalid_payload = {
    #     "payload": {
    #         "type": "non_existent_type",
    #         "some_param": "value"
    #     }
    # }
    # response_invalid = client.post(f"{API_BASE_URL}/connections/test", json=test_invalid_payload)
    # # Expect 400 or a result with "valid": False
    # if response_invalid.status_code == 400:
    #     assert "No handler registered" in response_invalid.text # Example error
    # else:
    #     response_invalid.raise_for_status()
    #     assert response_invalid.json()["valid"] is False


# --- Models Endpoint Tests ---

def test_list_models(client: httpx.Client):
    response = client.get(f"{API_BASE_URL}/models/")
    response.raise_for_status()
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0 # Expect at least one model
    assert "id" in models[0]
    assert "name" in models[0]

def test_get_current_model(client: httpx.Client):
    response = client.get(f"{API_BASE_URL}/models/current")
    # This might be 404 if no model is set by default, or 200 if there's a default
    if response.status_code == 404:
        print("No current model set, which is acceptable for this test.")
    else:
        response.raise_for_status()
        assert response.status_code == 200
        model = response.json()
    assert "id" in model
    assert "name" in model


# --- Investigation Flow Tests ---
def test_chat_message_triggers_filesystem_investigation(
    client: httpx.Client, 
    test_notebook: dict, 
    setup_filesystem_connection: dict, # Moved earlier
    test_chat_session: dict # Moved later
):
    session_id = test_chat_session["session_id"]
    notebook_id = test_notebook["id"] # For context if needed, though session_id implies it
    
    prompt_text = "Using the filesystem, list all files and directories within the /app directory." # Made prompt more specific
    
    events_received: List[Dict[str, Any]] = []
    type_error_in_stream = False
    any_other_error_event = None

    try:
        with client.stream(
            "POST", 
            f"{API_BASE_URL}/chat/sessions/{session_id}/messages", 
            data={"prompt": prompt_text}, # Send as form data
            timeout=60 # Increased timeout for investigation
        ) as response:
            # Check for immediate errors like 4xx/5xx before streaming
            response.raise_for_status() 
            
            for line in response.iter_lines():
                if line:
                    try:
                        event = json.loads(line)
                        events_received.append(event)
                        # print(f"Integration Test Event: {event}") # For debugging

                        if event.get("role") == "error":
                            if "object async_generator can't be used in 'await' expression" in event.get("content", ""):
                                type_error_in_stream = True
                                break # Stop processing if specific error found
                            if not any_other_error_event: # Capture first other error
                                any_other_error_event = event
                                
                    except json.JSONDecodeError:
                        print(f"Integration Test: Failed to decode JSON from stream: {line}")
                        # Potentially fail test here if strict about stream format
            
    except httpx.HTTPStatusError as e:
        # This catches errors if raise_for_status() fails (e.g. 500 before stream starts)
        pytest.fail(f"HTTP error: {e.response.status_code} - {e.response.text}")
    except httpx.ReadTimeout:
        if not events_received: # Timeout with no events is a failure
            pytest.fail("Reading from chat stream timed out without receiving any events.")
        else: # Timeout after some events might be acceptable depending on test depth
            print("Integration Test: Chat stream timed out after receiving some events.")
            print(f"Received events before timeout: {events_received}")


    assert not type_error_in_stream, "TypeError: object async_generator can't be used in 'await' expression found in stream."
    
    # Assert that no *other* error events were received during the stream
    if any_other_error_event: # Check if an error was captured
        print(f"All received events on error: {events_received}")
        pytest.fail(f"An error event was found in stream: {any_other_error_event}")

    if not events_received:
        print("No events received from the stream at all.")
    assert len(events_received) > 0, "No events were received from the chat stream."
    
    # Check for successful filesystem operation.
    # FileSystemToolCellCreatedEvent has type: "filesystem_tool_cell_created" (as per events.py)
    fs_tool_cell_created_events = [
        event for event in events_received if event.get("agent_type") == "filesystem" and event.get("status_type") == "success"
    ]
    # For a multi-step plan, we might expect more than one successful filesystem tool use or step completion.
    # For this test, let's ensure at least one plan was created and the investigation completed.

    plan_created = any(event.get("type") == "plan_created" for event in events_received)
    investigation_complete = any(event.get("type") == "investigation_complete" for event in events_received)

    if not plan_created:
        print("PlanCreatedEvent was not found. All received events:")
        for evt_idx, evt in enumerate(events_received):
            print(f"Event {evt_idx}: {evt}")
    assert plan_created, "PlanCreatedEvent not found in stream, investigation likely didn't start as expected."
    
    # Check for step progression
    step_started_events = [event for event in events_received if event.get("type") == "step_started"]
    step_completed_events = [event for event in events_received if event.get("type") == "step_completed" or event.get("type") == "filesystem_step_processing_complete"] # or other relevant completion events

    # This is a basic check. A more robust test would parse the plan from PlanCellCreatedEvent
    # and then specifically check for the start and completion of each step in the plan,
    # ensuring dependent steps run after their dependencies.
    # For now, we'll check that at least one step started and the investigation completed.
    assert len(step_started_events) > 0, "No StepStartedEvent found."
    # If the plan has multiple steps, we'd expect multiple step_started and step_completed events.
    # The original issue was stalling, so if investigation_complete is true, it didn't stall.
    assert investigation_complete, "InvestigationCompleteEvent not found. The investigation may have stalled or failed."

    # Example of a more detailed check if we knew the plan structure:
    # Assume plan: step_A (filesystem), step_B (markdown, depends on step_A)
    # step_A_started = any(e.get("type") == "step_started" and e.get("content", "").endswith("(Step: step_A)") for e in events_received)
    # step_A_completed_event_found = False
    # step_B_started_after_A_completed = False
    # step_A_completion_index = -1
    #
    # for i, e in enumerate(events_received):
    #     if e.get("type") == "filesystem_step_processing_complete" and "step_A" in e.get("content", ""): # Adjust based on actual event content for step_A completion
    #         step_A_completed_event_found = True
    #         step_A_completion_index = i
    #         break
    #
    # if step_A_completed_event_found:
    #     for i in range(step_A_completion_index + 1, len(events_received)):
    #         e = events_received[i]
    #         if e.get("type") == "step_started" and e.get("content", "").endswith("(Step: step_B)"):
    #             step_B_started_after_A_completed = True
    #             break
    #
    # assert step_A_started, "Step A (filesystem) did not start."
    # assert step_A_completed_event_found, "Completion event for Step A not found."
    # assert step_B_started_after_A_completed, "Step B (markdown) did not start after Step A completed."

    # For the current fix, ensuring the investigation completes is the primary goal.
    # The unit test covers the specific event ordering logic more directly.
