"""
Utilities for handling files within the backend, particularly for staging agent inputs.
"""
import os
import re
import uuid
import logging
from typing import Optional

# TODO: Move IMPORTED_DATA_BASE_PATH to a central config/constants location
# Using the same path as defined in PythonAgent for consistency
IMPORTED_DATA_BASE_PATH = "data/imported_datasets" 

file_utils_logger = logging.getLogger(__name__)

async def stage_file_content(
    content: str, 
    original_filename: Optional[str], 
    notebook_id: str, 
    session_id: str,
    source_cell_id: Optional[str] = None # Added for logging/tracking
) -> str:
    """
    Saves the given string content to a unique, persistent local file path
    within the backend's designated storage area (inside the container).

    Args:
        content: The string content of the file.
        original_filename: The original filename, used for naming the staged file.
        notebook_id: The ID of the notebook context.
        session_id: The ID of the current session context.
        source_cell_id: Optional ID of the cell that produced this content.

    Returns:
        The absolute path to the newly created local file.

    Raises:
        OSError: If directory creation fails.
        IOError: If file writing fails.
    """
    # Sanitize original_filename if provided, or use a generic name
    base_filename = "dataset"
    if original_filename:
        # Basic sanitization: replace non-alphanumeric with underscore
        sanitized_name = re.sub(r'[^\w\.-]', '_', original_filename)
        # Ensure it ends with .csv if it's a CSV, or add if no extension
        # TODO: Make extension handling more robust if non-CSV files are expected
        name_part, ext_part = os.path.splitext(sanitized_name)
        if not ext_part: # if no extension
            sanitized_name += ".csv" # Assume CSV for now
        elif ext_part.lower() != ".csv": # if extension is not csv
            file_utils_logger.warning(f"Original filename '{original_filename}' does not end with .csv. Appending .csv to sanitized name '{name_part}'.")
            sanitized_name = name_part + ".csv"
        base_filename = sanitized_name

    # Create a unique filename to avoid collisions
    unique_id = str(uuid.uuid4().hex[:8])
    filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.csv"
    
    # Construct path: IMPORTED_DATA_BASE_PATH / notebook_id / session_id / filename
    dir_path = os.path.join(IMPORTED_DATA_BASE_PATH, notebook_id, session_id)
    
    try:
        os.makedirs(dir_path, exist_ok=True)
        file_utils_logger.debug(f"Ensured directory exists: {dir_path}")
    except OSError as e:
        file_utils_logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
        raise # Re-raise to be caught by caller

    local_permanent_path = os.path.join(dir_path, filename)
    absolute_path = os.path.abspath(local_permanent_path) # Ensure path is absolute

    try:
        with open(absolute_path, 'w', encoding='utf-8') as f:
            f.write(content) 
        file_utils_logger.info(f"Successfully wrote data to permanent local file: {absolute_path} (source: {original_filename or 'N/A'}, cell: {source_cell_id or 'N/A'})")
        return absolute_path
    except IOError as e:
        file_utils_logger.error(f"Failed to write to local file {absolute_path}: {e}", exc_info=True)
        raise # Re-raise to be caught by caller
