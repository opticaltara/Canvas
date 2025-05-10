import pytest
from unittest.mock import MagicMock

# Mock Pydantic models for Settings to avoid deep dependencies during test collection
class MockSettings(MagicMock):
    # Add any specific attributes your Settings model might have that are accessed
    # during import or early setup. For now, MagicMock should be flexible.
    # If specific fields are needed, define them like in a Pydantic model:
    # example_setting: str = "default_value"
    pass

@pytest.fixture(autouse=True)
def mock_app_settings(monkeypatch):
    """Auto-used fixture to mock app-wide settings during test runs."""
    print("Applying mock_app_settings fixture") # For debugging if it runs
    settings_instance = MockSettings()
    # Add any default attributes to the instance if needed
    # settings_instance.some_expected_db_url = "mock_db_url"
    
    monkeypatch.setattr("backend.config.get_settings", lambda: settings_instance)
    # Also patch where it might be directly imported in database.py if that's an issue
    # This depends on how get_settings is imported and used in database.py
    # If database.py does `from backend.config import get_settings`, the above is fine.
    # If it does `import backend.config; backend.config.get_settings()`, also fine. 

@pytest.fixture(autouse=True)
def mock_db_level_get_settings(monkeypatch):
    """Auto-used fixture to mock get_settings specifically for backend.db.database module imports."""
    print("Applying mock_db_level_get_settings fixture to backend.db.database.get_settings") # Debug output
    
    settings_instance = MockSettings()
    # Example: if database.py immediately uses settings.db_type or settings.db_file:
    # settings_instance.db_type = "sqlite_mock"
    # settings_instance.db_file = ":memory:"
    # The original get_settings() also prints some keys, MagicMock handles these accesses.

    def mock_get_settings_func():
        # This function will replace get_settings in backend.db.database
        return settings_instance

    # Patch get_settings in the module where it's looked up by backend.db.database.py at its import time.
    monkeypatch.setattr("backend.db.database.get_settings", mock_get_settings_func) 