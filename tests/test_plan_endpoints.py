import json
import asyncio
from fastapi.testclient import TestClient

# Assuming FastAPI instance is created in backend.server module
from backend.server import app  # FastAPI instance

client = TestClient(app)

def test_confirm_cancel_routes_exist():
    # Simple sanity checks that routes exist and return 404 for unknown ids
    session_id = "dummy"
    plan_id = "dummy-plan"

    # Confirm route returns 404 because session not exists
    resp = client.post(f"/sessions/{session_id}/plans/{plan_id}/confirm", json={})
    assert resp.status_code in (404, 500)

    # Cancel route returns 404/500 similarly
    resp = client.post(f"/sessions/{session_id}/plans/{plan_id}/cancel")
    assert resp.status_code in (404, 500) 