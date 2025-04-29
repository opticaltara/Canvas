# Implementation Plan: Adding Notebook Context to AI Agent

**Goal:** Enhance the `AIAgent`'s contextual understanding by providing it with a summary of the last N notebook cells (content, execution status, results) before it plans or executes actions.

**Reasoning:** This will allow the agent to:
*   Understand follow-up questions referencing previous results.
*   Avoid redundant work if information already exists.
*   Build upon previous findings for more complex analysis.
*   Provide a more cohesive and stateful user experience within the notebook.

---

## Prerequisites:

1.  **Cell-ToolCallRecord Link:** Ensure the mechanism linking a specific cell (`Cell.id`) to the `ToolCallRecord`(s) generated *for* or *by* that cell is robust. We need to reliably retrieve these records based on the cell ID.
2.  **Notebook Structure:** Assume the `Notebook` object contains an ordered list or dictionary of cells and a dictionary of `ToolCallRecord`s.

---

## Detailed Implementation Plan:

### Phase 1: Data Gathering & Summarization

1.  **Configuration:**
    *   **Action:** Add a new configuration variable, `AGENT_NOTEBOOK_CONTEXT_CELL_COUNT`, to `backend/config.py` (e.g., default to 5).
    *   **Reasoning:** Makes the number of context cells easily adjustable without code changes.
    *   **Code File:** `backend/config.py`

2.  **Helper Function for Context Retrieval:**
    *   **Action:** Create a new helper function, potentially within `backend/services/notebook_manager.py` or a new `backend/core/context_utils.py`. Let's call it `get_notebook_context_summary(notebook_id: UUID, context_cell_count: int) -> str`.
    *   **Reasoning:** Encapsulates the logic for fetching and summarizing notebook context, keeping `AIAgent` cleaner.
    *   **Function Logic:**
        *   Accept `notebook_id` and `context_cell_count`.
        *   Use `NotebookManager` to retrieve the `Notebook` object. Handle `NotebookNotFound` errors.
        *   Access the notebook's cells. Determine the order and select the *last* `context_cell_count` cells.
        *   Iterate through the selected cells:
            *   Get cell ID, type (`CellType`), and content/query.
            *   Find associated `ToolCallRecord`s via `parent_cell_id`.
            *   **Summarize Result/Status:** Based on `ToolCallRecord.status` and `ToolCallRecord.result`/`error`. Generate concise, type-specific summaries (e.g., "Log query returned X lines", "Metric query showed spike", "Found N PRs", "Failed with error: ..."). Handle cells without records.
            *   **Format Cell Summary:** Create a concise string snippet for each cell (e.g., `Cell [ID]: [Type], Status: [Status]. Query: [Query Summary]. Result: [Result Summary].`).
        *   Combine snippets into a single formatted string with a header.
        *   Handle edge cases (empty notebook, < N cells).
    *   **Code File:** New function in `backend/services/notebook_manager.py` or `backend/core/context_utils.py`. Requires access to `Notebook`, `Cell`, `ToolCallRecord`, `QueryResult` types.

### Phase 2: Integration with AIAgent

3.  **Update Investigation Dependencies:**
    *   **Action:** Add `notebook_context: Optional[str]` to `InvestigationDependencies` model in `backend/ai/agent.py`.
    *   **Reasoning:** Provides a structured way to pass the context summary to the planner.
    *   **Code File:** `backend/ai/agent.py`

4.  **Inject Context in `AIAgent.investigate`:**
    *   **Action:** Modify `AIAgent.investigate` method:
        *   Read `AGENT_NOTEBOOK_CONTEXT_CELL_COUNT` from settings.
        *   Call `get_notebook_context_summary` before creating `InvestigationDependencies`.
        *   Populate the `notebook_context` field in `deps` with the summary string.
    *   **Reasoning:** Gathers context just before planning, ensuring it's up-to-date.
    *   **Code File:** `backend/ai/agent.py` (`investigate` method)

### Phase 3: Agent Prompt Adaptation

5.  **Update Investigation Planner Prompt:**
    *   **Action:** Modify the `system_prompt` for the `investigation_planner` agent in `AIAgent.__init__`.
        *   Mention the new `notebook_context` field in `InvestigationDependencies`.
        *   Add instructions on how to use it (e.g., "Review the `notebook_context`... Use this context to understand follow-up requests... Leverage information from `notebook_context` if the user's query refers to previous cells...").
    *   **Reasoning:** Guides the LLM to actively use the provided context.
    *   **Code File:** `backend/ai/agent.py` (`__init__` method)

### Phase 4: Testing and Refinement

6.  **Unit Testing:**
    *   **Action:** Write unit tests for `get_notebook_context_summary`. Test edge cases (empty notebook, < N cells), various cell types, success/error states, summarization logic.
    *   **Reasoning:** Ensures core context gathering logic is robust.

7.  **Integration Testing:**
    *   **Action:** Create end-to-end test scenarios simulating user flows (follow-up questions, user cells, redundancy, error handling) and verify the agent uses the context appropriately.
    *   **Reasoning:** Validates the complete feature and agent behavior.

8.  **Refinement:**
    *   **Action:** Based on testing, refine summarization logic and planner prompt instructions for clarity, effectiveness, and token efficiency.
    *   **Reasoning:** Fine-tuning is crucial for optimal performance.

---

## Future Considerations (Optional):

*   **Passing Context to Other Agents:** Evaluate if other agents (e.g., `markdown_generator`) could benefit.
*   **More Sophisticated Context Selection:** Explore relevance-based selection (e.g., embeddings) instead of just the last N.
*   **Visual Cues:** Add UI indicators showing the agent is considering context. 