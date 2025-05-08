import logging
import json
import re
from typing import List, Dict, Any, Set, Type, Optional
from datetime import datetime, timezone

# It's crucial that this CellStatus import points to the correct location
# in your project. Adjust if necessary.
from backend.core.cell import CellStatus

context_utils_logger = logging.getLogger("ai.context_utils")

# --- Added from backend/ai/utils.py ---
TOOL_RESULT_MAX_CHARS = 10000  # Default from filesystem_agent, can be overridden by caller

def _truncate_output(content: Any, limit: int = TOOL_RESULT_MAX_CHARS) -> Any:
    """Truncate the given string if it exceeds *limit* characters, appending an indicator.

    Non-string content is returned unchanged so the caller can forward it as-is.
    """
    if not isinstance(content, str):
        return content  # Non-string results are returned unchanged
    if len(content) <= limit:
        return content
    truncated = content[:limit]
    return truncated + f"\n\n...[output truncated – original length {len(content)} characters]" 
# --- End Added ---

# Consider moving STOP_WORDS_CONTEXT_PREP to a shared constants file if used elsewhere.
STOP_WORDS_CONTEXT_PREP = set([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
    "could", "may", "might", "must", "about", "above", "after", "again", "against",
    "all", "am", "and", "any", "as", "at", "because", "before", "below",
    "between", "both", "but", "by", "com", "for", "from", "further", "here",
    "how", "i", "if", "in", "into", "it", "its", "itself", "just", "k", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "o", "of", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "r", "s", "same", "she", "so", "some", "such", "t", "than", "that",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "we", "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "you", "your", "yours", "yourself", "yourselves", "please", "ok", "good"
])

DEFAULT_CONTEXT_TRUNCATE_LIMIT = 300
MAX_TOOL_ARGS_TO_SHOW_IN_CONTEXT = 2
INDIVIDUAL_TOOL_ARG_TRUNCATE_LIMIT = 30


def _truncate_for_context(content: Any, limit: int = DEFAULT_CONTEXT_TRUNCATE_LIMIT) -> str:
    """Safely converts content to string and truncates it."""
    content_str = str(content)
    if len(content_str) <= limit:
        return content_str
    truncated = content_str[:limit]
    return truncated + f"... (truncated, original length {len(content_str)})"


def _parse_datetime_optional(ts_str: Optional[str]) -> datetime:
    """Parses an optional ISO datetime string to a timezone-aware datetime object."""
    if not ts_str:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        if ts_str.endswith('Z'): # Handle 'Z' for UTC
            ts_str = ts_str[:-1] + '+00:00'
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None: # If naive, assume UTC
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        context_utils_logger.warning(f"Could not parse timestamp: '{ts_str}'. Using min datetime.")
        return datetime.min.replace(tzinfo=timezone.utc)


def prepare_notebook_context_for_planner(
    all_cells_data: List[Dict[str, Any]],
    current_query: str,
    cell_status_enum: Type[CellStatus],
    max_context_cells: int = 5,
    context_summary_truncate_limit: int = DEFAULT_CONTEXT_TRUNCATE_LIMIT,
    allowed_cell_types: Optional[List[str]] = None
) -> str:
    """
    Formats relevant cell data from a notebook into a context string for the AI planner.
    It prioritizes recent, relevant cells with meaningful output or active status,
    excluding cells that ended in an error.

    Args:
        all_cells_data: A list of dictionaries, where each dict represents a cell's data.
                        Expected keys per cell: 'id', 'cell_type', 'status', 'content',
                        'output' (itself a dict with 'content' or 'error'),
                        'tool_name', 'tool_arguments', 'updated_at'.
        current_query: The user's current query, used for relevance scoring.
        cell_status_enum: The actual CellStatus enum type (e.g., backend.core.cell.CellStatus).
        max_context_cells: The maximum number of cells to include in the formatted context.
        context_summary_truncate_limit: Character limit for truncating content/output snippets.
        allowed_cell_types: A list of cell types to include in the context.

    Returns:
        A formatted string representing the notebook context, ready for an AI prompt.
        Returns an empty string if no suitable cells are found.
    """
    if not all_cells_data:
        context_utils_logger.info("No cells data provided; returning empty context.")
        return ""

    # 1. Relevance Scoring
    query_keywords: Set[str] = set(re.findall(r"`([^`]+)`", current_query.lower()))
    cleaned_query_words = [
        word for word in re.findall(r'\b\w+\b', current_query.lower())
        if word not in STOP_WORDS_CONTEXT_PREP and len(word) > 1
    ]
    query_keywords.update(cleaned_query_words)
    context_utils_logger.info(f"Query keywords for context scoring: {query_keywords}")

    scored_cells = []
    for cell_data in all_cells_data:
        score = 0.0 # Use float for scores
        cell_content_lower = str(cell_data.get('content', '')).lower()
        cell_type_lower = str(cell_data.get('cell_type', '')).lower()

        for kw in query_keywords:
            if kw in cell_content_lower:
                score += 10.0
            if kw in cell_type_lower:
                score += 5.0
        
        updated_at_dt = _parse_datetime_optional(cell_data.get('updated_at'))
        # Add a small recency bonus based on timestamp (newer = higher score)
        # This is a simple way; more complex decay functions could be used.
        # Ensure this bonus doesn't overshadow content relevance too much.
        score += updated_at_dt.timestamp() / 1_000_000_000.0 # Scale down significantly

        scored_cells.append({
            "data": cell_data,
            "score": score,
            "updated_at": updated_at_dt
        })

    scored_cells.sort(key=lambda x: (x["score"], x["updated_at"]), reverse=True)

    # 2. Filtering and Formatting
    filtered_cell_context_parts: List[str] = []
    # Evaluate a few more candidates than we might strictly need, to ensure good choices.
    MAX_CANDIDATES_TO_EVALUATE = max_context_cells + 5 

    for i, scored_cell_item in enumerate(scored_cells):
        if i >= MAX_CANDIDATES_TO_EVALUATE and len(filtered_cell_context_parts) >= max_context_cells:
            break # Optimization

        cell_data = scored_cell_item["data"]
        cell_id = cell_data.get('id', 'unknown_id')
        cell_type = cell_data.get('cell_type', 'unknown_type')
        status_str = cell_data.get('status', cell_status_enum.IDLE.value)

        if status_str == cell_status_enum.ERROR.value:
            context_utils_logger.info(f"Cell {cell_id} (status: ERROR) skipped from context.")
            continue

        # Filter by allowed cell types if provided
        if allowed_cell_types and cell_type not in allowed_cell_types:
            context_utils_logger.info(f"Cell {cell_id} (type: {cell_type}) skipped, not in allowed_cell_types: {allowed_cell_types}.")
            continue

        content_snippet = _truncate_for_context(cell_data.get('content', ''), limit=context_summary_truncate_limit)
        
        output_summary = "No specific output."
        has_meaningful_output = False
        cell_output_data = cell_data.get('output')

        if cell_output_data and cell_output_data.get('content') is not None:
            raw_output_content = cell_output_data.get('content')
            temp_summary = ""
            if cell_type == 'python' and isinstance(raw_output_content, str):
                # Basic checks for common Python rich output patterns
                df_match = re.search(r"--- DataFrame ---(.*?)--- End DataFrame ---", raw_output_content, re.DOTALL)
                plot_match = re.search(r"<PLOT_BASE64>(.*?)</PLOT_BASE64>", raw_output_content)
                json_match = re.search(r"<JSON_OUTPUT>(.*?)</JSON_OUTPUT>", raw_output_content, re.DOTALL)
                if df_match:
                    temp_summary = f"Python Output: DataFrame displayed (preview): {_truncate_for_context(df_match.group(1).strip(), limit=context_summary_truncate_limit)}"
                elif plot_match:
                    temp_summary = "Python Output: Plot generated." # Avoid showing base64
                elif json_match:
                    temp_summary = f"Python Output: JSON data: {_truncate_for_context(json_match.group(1).strip(), limit=context_summary_truncate_limit)}"
                else:
                    temp_summary = f"Python Output: {_truncate_for_context(str(raw_output_content), limit=context_summary_truncate_limit)}"
            elif isinstance(raw_output_content, list):
                temp_summary = f"Output (list): {_truncate_for_context(json.dumps(raw_output_content), limit=context_summary_truncate_limit)}"
            elif isinstance(raw_output_content, dict): # For general structured dict output
                temp_summary = f"Output (structured): {_truncate_for_context(json.dumps(raw_output_content), limit=context_summary_truncate_limit)}"
            else: # For plain string output or other types
                temp_summary = f"Output: {_truncate_for_context(str(raw_output_content), limit=context_summary_truncate_limit)}"
            
            # Define "meaningful" more carefully
            if temp_summary and not any(temp_summary.strip() == marker for marker in ["Output:", "Python Output:", "Output (list):", "Output (structured):"]):
                output_summary = temp_summary
                has_meaningful_output = True
        
        is_active_status = status_str in [cell_status_enum.RUNNING.value, cell_status_enum.QUEUED.value]

        if not has_meaningful_output and not is_active_status:
            context_utils_logger.info(f"Cell {cell_id} skipped: No meaningful output & not active.")
            continue
        
        # Prepare tool info
        tool_name = cell_data.get('tool_name')
        tool_args_dict = cell_data.get('tool_arguments')
        tool_info_parts = []
        if tool_name:
            tool_info_parts.append(f"Tool: `{tool_name}`")
        if tool_args_dict and isinstance(tool_args_dict, dict):
            summarized_args = []
            for arg_idx, (k, v) in enumerate(tool_args_dict.items()):
                if arg_idx >= MAX_TOOL_ARGS_TO_SHOW_IN_CONTEXT:
                    summarized_args.append("...")
                    break
                summarized_args.append(f"`{k}`: `{_truncate_for_context(str(v), limit=INDIVIDUAL_TOOL_ARG_TRUNCATE_LIMIT)}`")
            if summarized_args:
                tool_info_parts.append(f"Args: {{ {', '.join(summarized_args)} }}")
        
        tool_info_line = ""
        if tool_info_parts: # Add escaped newline for prompt
            tool_info_line = f"  ({', '.join(tool_info_parts)})\n" 

        status_note = ""
        if status_str == cell_status_enum.STALE.value and has_meaningful_output:
            status_note = " (Note: Output is from a previous execution before cell became stale)"

        # Escape newlines for the final prompt string
        content_for_prompt = content_snippet.replace("\n", "\\n")
        output_for_prompt = output_summary.replace("\n", "\\n")

        filtered_cell_context_parts.append(
            f"[Cell ID: {cell_id[:8]} - Type: {cell_type} - Status: {status_str}]\n"
            f"{tool_info_line}"
            f"  Content: {content_for_prompt}\n"
            f"  Output{status_note}: {output_for_prompt}"
        )

        if len(filtered_cell_context_parts) >= max_context_cells:
            break

    if not filtered_cell_context_parts:
        context_utils_logger.info("No cells met criteria for inclusion in context after filtering.")
        return ""

    context_header = f"--- Existing Notebook Context (Top {len(filtered_cell_context_parts)} Relevant Cells with Output or Active Status, most recent first based on relevance score) ---"
    # Join with escaped newlines for the prompt
    final_context_str = "\n".join([context_header] + filtered_cell_context_parts + ["--- End of Existing Notebook Context ---"])
    
    # Log a snippet of the generated context for easier debugging
    context_utils_logger.info(f"Generated notebook context (first 500 chars):\n{final_context_str[:500]}")
    return final_context_str 