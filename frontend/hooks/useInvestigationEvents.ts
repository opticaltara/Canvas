"use client"

import { useState, useEffect, useCallback } from "react"
import { useWebSocket } from "./useWebSocket"
import { useToast } from "@/hooks/use-toast"
import { useCanvasStore } from "@/store/canvasStore" // Import useCanvasStore
import { type CellStatus} from "@/store/types"; // Import CellStatus & CellType

// Define a simple QueryResult type matching backend structure
export interface QueryResult {
  data: any;
  query: string;
  error?: string | null;
  metadata?: Record<string, any>;
  // Add fields specific to subclasses if needed, e.g., tool_calls for GithubQueryResult
  tool_calls?: any[] | null;
}

// Define event types based on the documentation
export interface BaseEvent {
  type: string
  status: string
  timestamp: number
}

export interface PlanCreatedEvent extends BaseEvent {
  type: "plan_created"
  status: "plan_created"
  thinking: string
}

export interface PlanCellCreatedEvent extends BaseEvent {
  type: "plan_cell_created"
  status: "plan_cell_created"
}

export interface StepCompletedEvent extends BaseEvent {
  type: string // "step_${stepId}_completed"
  status: "step_completed"
  step_id: string
  step_type: "log" | "metric" | "markdown" | "github" // Removed summarization, sql, python, ai_query (update if others are used)
  cell_id: string
  result: QueryResult // Use the imported QueryResult type from store/types
  cell_params: Partial<CellCreationParams>; // Add cell_params used by backend
  agent_type: string;
  is_single_step_plan?: boolean;
  tool_call_record_id?: string;
}

export interface PlanRevisedEvent extends BaseEvent {
  type: "plan_revised"
  status: "plan_revised"
  explanation: string
}

export interface ErrorEvent extends BaseEvent {
  type: "error"
  status: "error"
  message: string
}

// --- New Summarization Events ---
export interface SummaryStartedEvent extends BaseEvent {
  type: "summary_started";
  status: "summary_started";
  agent_type: "summarization_generator";
}

export interface SummaryUpdateEvent extends BaseEvent {
  type: "summary_update";
  status: "summary_update";
  agent_type: "summarization_generator";
  update_info: {
    status?: string; // e.g., "agent_ready", "attempt_start"
    message?: string;
    error?: string;
    attempt?: number;
    max_attempts?: number;
  };
}

export interface SummaryCellCreatedEvent extends BaseEvent {
  type: "summary_cell_created";
  status: "summary_cell_created";
  agent_type: "summarization_generator";
  cell_params: CellCreationParams; // Assuming backend sends params matching CellCreationParams
  cell_id: string;
  error: string | null; // Error during summary generation
}

export interface SummaryCellErrorEvent extends BaseEvent {
  type: "summary_cell_error";
  status: "summary_cell_error";
  agent_type: "summarization_generator";
  error: string; // Error during cell creation itself
}

// --- ADDED: GitHub Tool Events --- 
export interface GithubToolCellCreatedEvent extends BaseEvent {
  type: "github_tool_cell_created";
  status: "success";
  original_plan_step_id: string;
  cell_id: string;
  tool_call_record_id: string;
  tool_name: string;
  tool_args: Record<string, any>;
  result: any; // The actual result from the tool call
  agent_type: "github_tool";
  cell_params: Partial<CellCreationParams>; // Params used by backend to create cell
}

export interface GithubToolErrorEvent extends BaseEvent {
  type: "github_tool_error";
  status: "error";
  original_plan_step_id?: string; // Optional as it might fail before step ID known
  tool_name?: string;
  tool_args?: Record<string, any>;
  error: string; // Error message from the tool execution/cell creation
  agent_type: "github_tool";
}

// --- Investigation Event Union ---
export type InvestigationEvent =
  | PlanCreatedEvent
  | PlanCellCreatedEvent
  | StepCompletedEvent
  | PlanRevisedEvent
  | ErrorEvent
  // Add new summary events
  | SummaryStartedEvent
  | SummaryUpdateEvent
  | SummaryCellCreatedEvent
  | SummaryCellErrorEvent
  // ADDED: GitHub Tool Events
  | GithubToolCellCreatedEvent
  | GithubToolErrorEvent

// Define cell creation parameters
export interface CellCreationParams {
  id: string
  step_id: string
  type: string
  content: string
  status: "idle" | "running" | "success" | "error"
  result?: any
  error?: string
  metadata?: Record<string, any>
}

interface UseInvestigationEventsProps {
  notebookId: string
  onCreateCell: (params: CellCreationParams) => void
  onUpdateCell: (cellId: string, updates: Partial<CellCreationParams>) => void
  onError: (message: string) => void
}

export function useInvestigationEvents({
  notebookId,
  onCreateCell,
  onUpdateCell,
  onError,
}: UseInvestigationEventsProps) {
  const { status, messages, sendMessage } = useWebSocket(notebookId)
  const [isInvestigationRunning, setIsInvestigationRunning] = useState(false)
  const [currentPlan, setCurrentPlan] = useState<string | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string | null>(null)
  const [steps, setSteps] = useState<Record<string, StepCompletedEvent>>({})
  const { toast } = useToast()

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!messages.length) return

    // Process the latest message
    const message = messages[messages.length - 1]

    try {
      handleEvent(message as InvestigationEvent)
    } catch (error) {
      console.error("Error handling investigation event:", error)
      const errorMessage = error instanceof Error ? error.message : String(error);
      onError(`Failed to process event: ${errorMessage}`)
    }
  }, [messages])

  // Handle different event types
  const handleEvent = useCallback(
    (event: InvestigationEvent) => {
      console.log("Received investigation event:", event)

      switch (event.type) {
        case "plan_created":
          handlePlanCreated(event as PlanCreatedEvent)
          break
        case "plan_cell_created":
          handlePlanCellCreated()
          break
        case "plan_revised":
          handlePlanRevised(event as PlanRevisedEvent)
          break
        case "summary_started":
          handleSummaryStarted(event as SummaryStartedEvent);
          break;
        case "summary_update":
          handleSummaryUpdate(event as SummaryUpdateEvent);
          break;
        case "summary_cell_created":
          handleSummaryCellCreated(event as SummaryCellCreatedEvent);
          break;
        case "summary_cell_error":
          handleSummaryCellError(event as SummaryCellErrorEvent);
          break;
        case "github_tool_cell_created":
          handleGithubToolCellCreated(event as GithubToolCellCreatedEvent);
          break;
        case "github_tool_error":
          handleGithubToolError(event as GithubToolErrorEvent);
          break;
        case "error":
          handleError(event as ErrorEvent)
          break
        default:
          // Check if it's a step completion event
          if (event.type.startsWith("step_") && event.type.endsWith("_completed")) {
            handleStepCompleted(event as StepCompletedEvent)
          } else if (event.type === "investigation_complete") {
            // Handle the final completion signal if necessary (might be redundant with summary_cell_created)
            console.log("Received investigation_complete signal");
            // Optionally ensure UI state reflects completion if not already done by summary handler
            // setIsInvestigationRunning(false);
            // setCurrentStatus("Investigation finished.");
          } else if (event.type.startsWith("step_") && event.type.endsWith("_event")) {
            // Generic handler for other potential step events (like status updates)
            handleStepUpdate(event); // Placeholder function 
          } else {
            console.warn("Unknown event type:", event.type)
          }
      }
    },
    [onCreateCell, onUpdateCell, onError],
  )

  // Handle plan_created event - just update status, don't create a cell
  const handlePlanCreated = useCallback(
    (event: PlanCreatedEvent) => {
      setIsInvestigationRunning(true)
      setCurrentPlan(event.thinking)
      setCurrentStatus("Creating investigation plan...")

      toast({
        title: "Investigation Started",
        description: "AI is creating an investigation plan...",
      })
    },
    [toast],
  )

  // Handle plan_cell_created event - just update status, don't create a cell
  const handlePlanCellCreated = useCallback(() => {
    setCurrentStatus("Plan created, executing steps...")
  }, [])

  // Handle step_*_completed events - THIS creates cells
  const handleStepCompleted = useCallback(
    (event: StepCompletedEvent) => {
      // Store the step information
      setSteps((prev) => ({
        ...prev,
        [event.step_id]: event,
      }))

      // Update status
      setCurrentStatus(`Completed step ${event.step_id}`)

      // Use content directly from cell_params if available, otherwise derive (for non-github)
      let cellContent = event.cell_params?.content || event.result?.query || "";
      if (event.step_type === "markdown" && event.result?.data) {
        cellContent = String(event.result.data);
      }

      // Create the cell - THIS IS THE ONLY PLACE WE CREATE STEP CELLS
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: event.step_id,
        type: event.step_type,
        content: cellContent,
        status: event.result.error ? "error" : "success",
        result: event.result.data,
        error: event.result.error || undefined,
        metadata: { ...(event.result.metadata || {}), ...(event.cell_params?.metadata || {})}, // Merge metadata
      }

      // Create the cell
      onCreateCell(cellParams)

      toast({
        title: "Step Completed",
        description: `Investigation step ${event.step_id} completed`,
      })
    },
    [onCreateCell, toast],
  )

  // Handle plan_revised event - just update status, don't create a cell
  const handlePlanRevised = useCallback(
    (event: PlanRevisedEvent) => {
      setCurrentPlan((prev) => (prev ? `${prev}\n\n### Plan Revision\n${event.explanation}` : event.explanation))
      setCurrentStatus("Investigation plan revised")

      toast({
        title: "Plan Revised",
        description: "The investigation plan has been updated based on findings.",
      })
    },
    [currentPlan, toast],
  )

  // --- New Handlers for Summarization Flow ---

  const handleSummaryStarted = useCallback(
    (event: SummaryStartedEvent) => {
      setCurrentStatus("Generating final answer...");
      toast({
        title: "Summarizing Findings",
        description: "AI is generating the final answer...",
      });
    },
    [toast]
  );

  const handleSummaryUpdate = useCallback(
    (event: SummaryUpdateEvent) => {
      const { update_info } = event;
      let message = "Summarization in progress...";
      if (update_info.message) {
        message = update_info.message;
      } else if (update_info.status) {
        message = `Summarization status: ${update_info.status}`;
      }
      setCurrentStatus(message);

      // Optionally show toast for errors during update
      if (update_info.error) {
        console.error("Summarization update error:", update_info.error);
        toast({
          variant: "destructive",
          title: "Summarization Issue",
          description: `An issue occurred during summary generation: ${update_info.error}`,
        });
      }
    },
    [toast]
  );

  const handleSummaryCellCreated = useCallback(
    (event: SummaryCellCreatedEvent) => {
      // Extract cell parameters from the event
      const backendCellParams = event.cell_params;

      // Create the final summary cell
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: backendCellParams.step_id || "final_summary", // Use step_id from backend or default
        type: backendCellParams.type || "markdown",
        content: backendCellParams.content || "",
        status: event.error ? "error" : "success", // Status reflects summary generation error
        result: null, // No direct 'result' data for the summary cell itself
        error: event.error || undefined,
        metadata: backendCellParams.metadata,
      };

      onCreateCell(cellParams);

      // Final status updates
      setIsInvestigationRunning(false); // Investigation is now complete
      setCurrentStatus("Investigation complete.");

      if (event.error) {
        toast({
          title: "Investigation Complete (with Summary Issue)",
          description: `Investigation finished, but there was an issue generating the summary: ${event.error}`,
        });
      } else {
        toast({
          title: "Investigation Complete",
          description: "The investigation has finished and the final answer is ready.",
        });
      }
    },
    [onCreateCell, toast]
  );

  const handleSummaryCellError = useCallback(
    (event: SummaryCellErrorEvent) => {
      const errorMsg = `Failed to create final summary cell: ${event.error}`;
      console.error(errorMsg);
      onError(errorMsg); // Propagate error
      setCurrentStatus("Error creating final summary cell.");
      setIsInvestigationRunning(false); // Mark as complete even if cell creation failed

      toast({
        variant: "destructive",
        title: "Summary Cell Error",
        description: errorMsg,
      });
    },
    [onError, toast]
  );

  // --- ADDED: GitHub Tool Event Handlers --- 
  const handleGithubToolCellCreated = useCallback(
    (event: GithubToolCellCreatedEvent) => {
      setCurrentStatus(`Completed tool: ${event.tool_name}`);

      const backendCellParams = event.cell_params || {};
      const backendMetadata = backendCellParams.metadata || {};
      
      // Construct CellCreationParams for the frontend store
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: event.original_plan_step_id, // Link back to original plan step
        type: "github", // Explicitly set type
        content: backendCellParams.content || `GitHub: ${event.tool_name}`, // Use backend content or generate default
        status: "success", // Tool execution was successful
        result: event.result, // The actual result data from the tool
        error: undefined,
        // Combine metadata from backend params and event details
        metadata: {
          ...backendMetadata,
          tool_name: event.tool_name,
          tool_args: event.tool_args,
          tool_call_record_id: event.tool_call_record_id,
          pydantic_ai_tool_call_id: backendMetadata.pydantic_ai_tool_call_id, // Get original ID if present
          source_agent: event.agent_type,
        },
      };

      // ---> CHANGE: Use onUpdateCell instead of onCreateCell
      // Assuming the cell with event.cell_id already exists from an earlier step or initial load.
      // We only need to update its status and result.
      console.log(`[handleGithubToolCellCreated] Updating cell ${event.cell_id} with status: success and result.`);
      onUpdateCell(event.cell_id, {
          status: "success",
          result: event.result,
          // We might also want to update metadata if it changed, but status/result are key
          metadata: cellParams.metadata, // Pass updated metadata too
          content: cellParams.content // Update content just in case backend changed it
      });
      // Original call: onCreateCell(cellParams);

      toast({
        title: "GitHub Tool Completed",
        description: `Tool '${event.tool_name}' executed successfully.`,
      });
    },
    [onUpdateCell, toast]
  );

  const handleGithubToolError = useCallback(
    (event: GithubToolErrorEvent) => {
      const errorMsg = `GitHub tool '${event.tool_name || 'unknown'} failed: ${event.error}`;
      console.error(errorMsg);
      onError(errorMsg); // Propagate error
      setCurrentStatus(`Error executing tool: ${event.tool_name || 'unknown'}`);
      // No cell is created for tool errors

      toast({
        variant: "destructive",
        title: "GitHub Tool Error",
        description: errorMsg,
      });
    },
    [onError, toast]
  );
  // --- END ADDED ---

  // Placeholder for handling intermediate step updates (like GitHub status)
  const handleStepUpdate = useCallback((event: any) => {
    // Example: Update cell status visually if needed
    const stepId = event.type.split('_')[1]; // Extract step ID
    const updateInfo = event.update_info;
    console.log(`Received update for step ${stepId}:`, updateInfo);
    // Find the corresponding cell and update its status/display?
    // This might require finding the cellId associated with the stepId
    // and calling onUpdateCell. Needs more logic based on how cell IDs map to step IDs.
    // For now, just log it.
    setCurrentStatus(`Updating step ${stepId}: ${updateInfo?.message || updateInfo?.status || '...'}`);
  }, [onUpdateCell]); // Add dependencies if it interacts with state/props

  // Handle error event
  const handleError = useCallback(
    (event: ErrorEvent) => {
      setCurrentStatus(`Error: ${event.message}`)
      onError(event.message)

      toast({
        variant: "destructive",
        title: "Investigation Error",
        description: event.message,
      })
    },
    [onError, toast],
  )

  const handleCellCreation = useCallback(
    (message: any) => {
      console.log("🔍 handleCellCreation called with message:", message)

      try {
        // Parse the content if it's a string
        let parsedContent
        if (typeof message.content === "string") {
          try {
            parsedContent = JSON.parse(message.content)
            console.log("🔍 Successfully parsed message content:", parsedContent)
          } catch (e) {
            console.error("🔍 Error parsing message content:", e)
            parsedContent = message.content
          }
        } else {
          parsedContent = message.content
        }

        // Check for cell creation patterns
        if (
          (parsedContent.status_type === "plan_cell_created" || parsedContent.type === "cell_response") &&
          parsedContent.cell_params
        ) {
          console.log("🔍 Found cell creation pattern in message")
          console.log("🔍 Cell params:", parsedContent.cell_params)

          const { notebook_id, cell_type, content, position, metadata } = parsedContent.cell_params

          // Generate a temporary ID for the cell
          const tempId = `temp-${Date.now()}`
          console.log("🔍 Generated temporary ID for cell:", tempId)

          // Create cell data object
          const cellData = {
            id: tempId,
            notebook_id,
            type: cell_type,
            content,
            position,
            metadata,
            status: "idle" as CellStatus,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          }

          console.log("🔍 Created cell data object:", {
            id: cellData.id,
            notebook_id: cellData.notebook_id,
            type: cellData.type,
            content_length: cellData.content?.length || 0,
            metadata: cellData.metadata ? "present" : "absent",
          })

          // Log cell parameters for debugging
          console.log("🔍 Cell creation parameters:", {
            notebook_id,
            cell_type,
            content_length: content?.length || 0,
            metadata: metadata ? "present" : "absent",
          })

          // Add the cell directly to the canvas store
          console.log("🔍 Adding cell to canvas store:", tempId, `(${cell_type})`)
          const canvasStore = useCanvasStore.getState()
          console.log("🔍 Current active notebook ID in store:", canvasStore.activeNotebookId)
          console.log("🔍 Current cells count in store:", canvasStore.cells.length)

          canvasStore.handleCellUpdate(cellData)

          console.log(`🔍 Added new ${cell_type} cell from agent ${parsedContent.agent_type} directly to canvas store`)

          // Also call the onCreateCell callback if provided
          if (onCreateCell) {
            console.log("🔍 Calling onCreateCell callback")
            onCreateCell({
              id: tempId,
              step_id: '',
              type: cell_type,
              content,
              status: "idle",
              metadata,
            })
          }
        } else {
          console.log("🔍 Message does not match cell creation pattern:", parsedContent)
        }
      } catch (error) {
        console.error("🔍 Error in handleCellCreation:", error)
      }
    },
    [onCreateCell],
  )

  return {
    wsStatus: status,
    isInvestigationRunning,
    currentPlan,
    currentStatus,
    steps,
  }
}
