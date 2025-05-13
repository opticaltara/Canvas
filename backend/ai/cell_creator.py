"""
Cell creation utilities for the AI agent system.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from backend.ai.models import StepType, InvestigationStepModel
from backend.ai.events import AgentType, ToolSuccessEvent
from backend.core.cell import CellType, CellResult, CellStatus
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams
from backend.core.query_result import InvestigationReport
from backend.services.connection_manager import ConnectionManager

ai_logger = logging.getLogger("ai")

class CellCreator:
    """Handles cell creation for different step types"""
    def __init__(self, notebook_id: str, connection_manager: ConnectionManager):
        self.notebook_id = notebook_id
        self.connection_manager = connection_manager

    async def create_code_index_query_cell(
        self,
        cell_tools: NotebookCellTools,
        step: InvestigationStepModel,
        tool_event: ToolSuccessEvent, # Contains the qdrant-find results
        dependency_cell_ids: List[UUID],
        session_id: str
    ) -> Tuple[Optional[UUID], Optional[CreateCellParams], Optional[str]]:
        """Create a cell for code index query results and return its ID, params and any error"""
        if not tool_event.tool_call_id or tool_event.tool_name != "qdrant-find":
            return None, None, f"Tool event is not for qdrant-find or missing tool_call_id for step {step.step_id}"

        # The result from qdrant-find is expected to be a list of search hits (dictionaries)
        search_results = tool_event.tool_result if isinstance(tool_event.tool_result, list) else []

        # For content, we can create a summary or just store the raw results.
        # Let's create a simple markdown summary for the cell content.
        # The full results will be in cell.result.content.
        cell_content_lines = [f"## Code Search Results for: `{step.description}`\n"]
        if search_results:
            cell_content_lines.append(f"Found {len(search_results)} results:\n")
            for i, hit in enumerate(search_results[:5]): # Display first 5 hits in markdown
                file_path = hit.get("metadata", {}).get("file_path", "N/A")
                score = hit.get("score", "N/A")
                # Robust score formatting – handle non-numeric gracefully
                if isinstance(score, (int, float)):
                    score_str = f"{score:.4f}"
                else:
                    score_str = str(score)

                # snippet = hit.get("document", {}).get("page_content", "Snippet unavailable")[:200]  # Optionally include snippet
                # For now, let's assume the payload is directly the document content or a summary
                payload_content = str(hit.get("payload", "Content unavailable"))[:200]


                cell_content_lines.append(f"**{i+1}. File:** `{file_path}` (Score: {score_str})")
                cell_content_lines.append(f"   ```\n   {payload_content}...\n   ```")
            if len(search_results) > 5:
                cell_content_lines.append(f"\n... and {len(search_results) - 5} more results (see cell data for full list).")
        else:
            cell_content_lines.append("No results found.")
        
        cell_content = "\n".join(cell_content_lines)

        cell_metadata = {
            "session_id": session_id,
            "original_plan_step_id": step.step_id,
            "external_tool_call_id": tool_event.tool_call_id,
            "search_query": step.parameters.get("search_query", step.description),
            "collection_name": step.parameters.get("collection_name"),
        }

        # The full search_results list will be stored in cell.result.content
        serializable_result_content = self._serialize_tool_result(search_results, "qdrant-find_results")

        cell_params = CreateCellParams(
            notebook_id=self.notebook_id,
            cell_type=CellType.CODE_INDEX_QUERY,
            content=cell_content, # Markdown summary
            metadata=cell_metadata,
            dependencies=dependency_cell_ids,
            tool_call_id=uuid4(), # New tool_call_id for this cell creation action
            tool_name="qdrant-find_results_display", # Internal name for this display cell
            tool_arguments=step.parameters, # Store original query parameters
            result=CellResult(
                content=serializable_result_content, # Store full results here
                error=None, 
                execution_time=0.0 # Not an execution in itself
            ),
            status=CellStatus.SUCCESS # Cell is created successfully with data
        )

        try:
            cell_creation_result = await cell_tools.create_cell(params=cell_params)
            cell_id_str = cell_creation_result.get("cell_id")
            if cell_id_str:
                return UUID(cell_id_str), cell_params, None
            else:
                return None, cell_params, "Cell creation for code index query returned no cell_id"
        except Exception as e:
            ai_logger.error(f"Failed to create code_index_query cell for step {step.step_id}: {e}", exc_info=True)
            return None, cell_params, str(e)

    async def create_markdown_cell(
        self,
        cell_tools: NotebookCellTools, 
        step: InvestigationStepModel, 
        result: Any,  # Should be MarkdownQueryResult
        dependency_cell_ids: List[UUID],
        session_id: str
    ) -> Tuple[Optional[UUID], Optional[CreateCellParams], Optional[str]]:
        """Create a markdown cell and return its ID, params and any error"""
        error = getattr(result, "error", None)
        cell_content = getattr(result, "data", "") or ""
        
        cell_metadata = {
            "session_id": session_id,
            "step_id": step.step_id
        }
        
        cell_params = CreateCellParams(
            notebook_id=self.notebook_id,
            cell_type=CellType.MARKDOWN,
            content=cell_content,
            metadata=cell_metadata,
            dependencies=dependency_cell_ids,
            tool_call_id=uuid4()
        )
        
        try:
            cell_result = await cell_tools.create_cell(params=cell_params)
            cell_id_str = cell_result.get("cell_id")
            if cell_id_str:
                return UUID(cell_id_str), cell_params, None
            else:
                return None, cell_params, "Cell creation returned no cell_id"
        except Exception as e:
            ai_logger.error(f"Failed to create markdown cell for step {step.step_id}: {e}", exc_info=True)
            return None, cell_params, str(e)
    
    async def create_report_cell(
        self,
        cell_tools: NotebookCellTools, 
        step: InvestigationStepModel, 
        report: InvestigationReport,
        dependency_cell_ids: List[UUID],
        session_id: str
    ) -> Tuple[Optional[UUID], Optional[CreateCellParams], Optional[str]]:
        """Create an investigation report cell and return its ID, params and any error"""
        error = report.error
        report_cell_content = f"# Investigation Report: {report.title}\n\n_Structured report data generated._"
        report_data_dict = report.model_dump(mode='json')
        
        report_cell_metadata = {
            "session_id": session_id,
            "step_id": step.step_id,
            "is_investigation_report": True
        }
        
        report_cell_params = CreateCellParams(
            notebook_id=self.notebook_id,
            cell_type=CellType.INVESTIGATION_REPORT.value,
            content=report_cell_content,
            metadata=report_cell_metadata,
            dependencies=dependency_cell_ids,
            tool_call_id=uuid4(),
            result=CellResult(
                content=report_data_dict,
                error=error,
                execution_time=0.0
            ),
            status=CellStatus.ERROR if error else CellStatus.SUCCESS
        )
        
        try:
            report_cell_result = await cell_tools.create_cell(params=report_cell_params)
            report_cell_id_str = report_cell_result.get("cell_id")
            if report_cell_id_str:
                return UUID(report_cell_id_str), report_cell_params, None
            else:
                return None, report_cell_params, "Cell creation returned no cell_id"
        except Exception as e:
            ai_logger.error(f"Failed to create report cell for step {step.step_id}: {e}", exc_info=True)
            return None, report_cell_params, str(e)
    
    async def create_media_timeline_cell(
        self,
        cell_tools: NotebookCellTools,
        step: InvestigationStepModel,
        payload,  # MediaTimelinePayload object
        dependency_cell_ids: List[UUID],
        session_id: str
    ) -> Tuple[Optional[UUID], Optional[CreateCellParams], Optional[str]]:
        """Create a media timeline cell and return ID, params and error"""
        try:
            from backend.ai.media_agent import MediaTimelinePayload  # Local import to avoid circular reference
        except Exception:
            MediaTimelinePayload = None  # type: ignore

        error = None
        if hasattr(payload, 'error'):
            error = getattr(payload, 'error')

        # Build Markdown content
        content_lines = ["# Media Timeline", ""]

        # ------------------------
        # Hypothesis section
        # ------------------------
        from pydantic import BaseModel  # Local import to avoid top-level dependency
        if MediaTimelinePayload is not None and isinstance(payload, MediaTimelinePayload):
            hypothesis = payload.hypothesis if hasattr(payload, 'hypothesis') else "Media analysis results"
            if isinstance(hypothesis, str):
                content_lines.extend(["## Hypothesis", "", hypothesis, ""])
            elif isinstance(hypothesis, BaseModel):
                content_lines.append("## Hypothesis")
                files = getattr(hypothesis, 'files_likely_containing_bug', [])
                reasoning = getattr(hypothesis, 'reasoning', [])
                snippets = getattr(hypothesis, 'specific_code_snippets_to_check', [])

                if files:
                    content_lines.append("**Files likely containing the bug:**")
                    for f in files:
                        content_lines.append(f"- `{f}`")
                    content_lines.append("")

                if reasoning:
                    content_lines.append("**Reasoning:**")
                    for r in reasoning:
                        content_lines.append(f"- {r}")
                    content_lines.append("")

                if snippets:
                    content_lines.append("**Specific code snippets to check:**")
                    for s in snippets:
                        content_lines.append(f"- `{s}`")
                    content_lines.append("")
            else:
                # Fallback – just string-ify
                content_lines.extend(["## Hypothesis", "", str(hypothesis), ""])
        else:
            # Fallback – just string-ify
            content_lines.extend(["## Hypothesis", "", str(payload), ""])

        # ------------------------
        # Timeline events section
        # ------------------------
        if hasattr(payload, 'timeline_events') and payload.timeline_events:
            content_lines.append("## Timeline Events")
            for idx, evt in enumerate(payload.timeline_events, 1):
                desc = getattr(evt, 'description', '')
                image_id = getattr(evt, 'image_identifier', '')
                content_lines.append(f"{idx}. **{desc}**  ")
                content_lines.append(f"   Image: `{image_id}`")
                code_refs = getattr(evt, 'code_references', [])
                if code_refs:
                    refs_str = ', '.join(code_refs)
                    content_lines.append(f"   Code references: {refs_str}")
                content_lines.append("")
        else:
            content_lines.append("(No timeline events provided)")

        cell_content = "\n".join(content_lines)

        # Ensure payload is JSON-serialisable for notebook storage
        # Convert Pydantic models to dict; otherwise stringify as fallback
        from pydantic import BaseModel  # Local import
        if isinstance(payload, BaseModel):
            serialisable_payload = payload.model_dump(mode='json')
        elif isinstance(payload, (dict, list, str, int, float, bool)) or payload is None:
            serialisable_payload = payload
        else:
            serialisable_payload = str(payload)

        cell_metadata = {
            "session_id": session_id,
            "step_id": step.step_id,
            "is_media_timeline": True
        }

        from backend.core.cell import CellType, CellResult, CellStatus
        cell_params = CreateCellParams(
            notebook_id=self.notebook_id,
            cell_type=CellType.MEDIA_TIMELINE,
            content=cell_content,
            metadata=cell_metadata,
            dependencies=dependency_cell_ids,
            tool_call_id=uuid4(),
            result=CellResult(content=serialisable_payload, error=error, execution_time=0.0),
            status=CellStatus.ERROR if error else CellStatus.SUCCESS
        )

        try:
            cell_result = await cell_tools.create_cell(params=cell_params)
            cell_id_str = cell_result.get("cell_id")
            if cell_id_str:
                return UUID(cell_id_str), cell_params, None
            else:
                return None, cell_params, "Cell creation returned no cell_id"
        except Exception as e:
            ai_logger.error(f"Failed to create media timeline cell for step {step.step_id}: {e}", exc_info=True)
            return None, cell_params, str(e)
    
    async def create_tool_cell(
        self,
        cell_tools: NotebookCellTools,
        step: InvestigationStepModel,
        cell_type: CellType,
        agent_type: AgentType,
        tool_event: ToolSuccessEvent,
        dependency_cell_ids: List[UUID],
        session_id: str
    ) -> Tuple[Optional[UUID], Optional[CreateCellParams], Optional[str]]:
        """Create a cell for a tool call and return its ID, params and any error"""
        if not tool_event.tool_call_id or not tool_event.tool_name:
            return None, None, f"Tool event missing tool_call_id or tool_name for step {step.step_id}"
        
        # Get proper cell content based on the tool type
        if cell_type == CellType.PYTHON and tool_event.tool_args and 'python_code' in tool_event.tool_args:
            cell_content = tool_event.tool_args['python_code']
        else:
            cell_content = f"{cell_type.value.capitalize()}: {tool_event.tool_name}"
        
        # Get default connection ID for this type
        connection_type = cell_type.value.lower()
        default_connection_id = None
        try:
            default_connection = await self.connection_manager.get_default_connection(connection_type)
            if default_connection:
                default_connection_id = default_connection.id
            else:
                ai_logger.warning(f"No default {connection_type} connection found for step {step.step_id}")
        except Exception as conn_err:
            ai_logger.error(f"Error fetching default {connection_type} connection: {conn_err}", exc_info=True)
        
        # Prepare cell metadata
        cell_tool_call_id = uuid4()
        cell_metadata = {
            "session_id": session_id,
            "original_plan_step_id": step.step_id,
            "external_tool_call_id": tool_event.tool_call_id
        }
        
        # Serialize result content
        tool_result_content = tool_event.tool_result
        serializable_result_content = self._serialize_tool_result(tool_result_content, tool_event.tool_name)
        
        # Create cell parameters
        create_cell_kwargs = {
            "notebook_id": self.notebook_id,
            "cell_type": cell_type,
            "content": cell_content,
            "metadata": cell_metadata,
            "dependencies": dependency_cell_ids,
            "connection_id": default_connection_id,
            "tool_call_id": cell_tool_call_id,
            "tool_name": tool_event.tool_name,
            "tool_arguments": tool_event.tool_args or {},
            "result": CellResult(
                content=serializable_result_content,
                error=None,
                execution_time=0.0
            )
        }
        
        cell_params = CreateCellParams(**create_cell_kwargs)
        
        try:
            cell_creation_result = await cell_tools.create_cell(params=cell_params)
            cell_id = cell_creation_result.get("cell_id")
            if not cell_id:
                return None, cell_params, f"create_cell tool did not return a cell_id for {cell_type} tool"
            
            return UUID(cell_id), cell_params, None
        except Exception as e:
            error_msg = f"Failed creating cell for {cell_type} tool {tool_event.tool_name}: {e}"
            ai_logger.warning(error_msg, exc_info=True)
            return None, cell_params, error_msg
    
    def _serialize_tool_result(self, result: Any, tool_name: str) -> Optional[Union[Dict, List, str, int, float, bool]]:
        """Serialize tool result to a JSON-compatible format"""
        from pydantic import BaseModel
        
        if isinstance(result, (dict, list, str, int, float, bool, type(None))):
            return result
        elif isinstance(result, BaseModel):
            try:
                return result.model_dump(mode='json')
            except Exception as e:
                ai_logger.warning(f"Failed to dump Pydantic model result for {tool_name}: {e}")
                return str(result)
        else:
            try:
                return str(result)
            except Exception as e:
                ai_logger.warning(f"Failed to convert tool result for {tool_name} to string: {e}")
                return f"Error converting result to string: {e}"
