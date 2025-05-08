from typing import Any

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